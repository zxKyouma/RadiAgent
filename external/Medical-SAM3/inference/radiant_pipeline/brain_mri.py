from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam3_inference import SAM3Model

from .orchestrator import CandidateSelector, FindingExtractor, ProposalGenerator, SegmentationRefiner
from .scoring import bbox_iou, compute_box_rerank_score, compute_mask_metrics, compute_text_rerank_score, lesion_size_prior
from .types import FindingTarget, ProposalCandidate, SegmentationSelection, StudyContext


DEFAULT_BRAIN_MRI_PROMPTS = [
    'meningioma',
    'brain tumor',
    'intracranial mass',
    'brain mass',
    'right frontal mass',
    'extra-axial mass',
    'enhancing mass',
    'tumor',
]


@dataclass(slots=True)
class BrainMriTextConfig:
    prompts: Sequence[str]
    top_k: int = 5
    cluster_iou: float = 0.3
    target_mask_frac: float = 0.003
    size_prior_strength: float = 0.3
    text_trust_score_threshold: float = 0.75
    text_trust_min_slices: int = 4
    text_trust_min_prompts: int = 4
    text_trust_boost: float = 0.35
    verbose: bool = False


@dataclass(slots=True)
class BrainMriVisualConfig:
    thresholds: Sequence[float] = (0.985, 0.99, 0.995)
    min_component_px: int = 12
    max_component_frac: float = 0.03
    max_candidates_per_slice: int = 8
    proposal_prompt: str = 'visual enhancement candidate'


class BrainMriFindingExtractor(FindingExtractor):
    def extract(self, context: StudyContext) -> FindingTarget:
        normalized = context.report_text.lower()
        laterality = 'unknown'
        if 'left' in normalized:
            laterality = 'left'
        elif 'right' in normalized:
            laterality = 'right'
        finding = context.metadata.get('finding_text') or 'intracranial mass'
        sub_anatomy = context.metadata.get('sub_anatomy')
        return FindingTarget(
            finding=str(finding),
            anatomy='brain',
            laterality=laterality,
            sub_anatomy=str(sub_anatomy) if sub_anatomy else None,
            modality_hint='brain_mri',
            support_status=str(context.metadata.get('support_status', 'unknown')),
            metadata={
                'case_id': context.case_id,
                'sequence': context.sequence,
            },
        )


class BrainMriTextProposalGenerator(ProposalGenerator):
    def __init__(self, model: SAM3Model, config: BrainMriTextConfig):
        self.model = model
        self.config = config

    def generate(self, context: StudyContext, target: FindingTarget) -> List[ProposalCandidate]:
        slice_indices = list(context.metadata['slice_indices'])
        mid_slice = int(context.metadata['mid_slice'])
        gt_volume = context.metadata.get('ground_truth_mask')
        slice_cache = _ensure_slice_cache(context, self.model)
        image_area = int(context.image_volume.shape[0] * context.image_volume.shape[1])
        candidates: List[ProposalCandidate] = []

        for slice_idx in slice_indices:
            gt_slice = None
            gt_summary = None
            if gt_volume is not None:
                gt_slice = gt_volume[:, :, slice_idx] > 0
                gt_summary = summarize_gt(gt_slice)
            if self.config.verbose:
                if gt_summary is not None:
                    print(f"\nSLICE {slice_idx} GT_PIXELS {gt_summary['gt_pixels']} GT_BBOX {gt_summary['gt_bbox_xyxy']}")
                else:
                    print(f"\nSLICE {slice_idx}")

            for prompt in self.config.prompts:
                raw_candidates = self.model.predict_text_candidates(
                    inference_state=slice_cache['slice_states'][slice_idx],
                    text_prompt=prompt,
                    top_k=self.config.top_k,
                )
                if not raw_candidates:
                    if self.config.verbose:
                        print(f"  PROMPT {prompt!r}: no candidates")
                    continue
                if self.config.verbose:
                    print(f"  PROMPT {prompt!r}: {len(raw_candidates)} candidates")

                for raw in raw_candidates:
                    candidate = ProposalCandidate(
                        slice_idx=slice_idx,
                        prompt=prompt,
                        rank=int(raw['rank']),
                        score=float(raw['score']),
                        bbox_xyxy=tuple(float(x) for x in raw['bbox_xyxy']),
                        area_px=int(raw['area_px']),
                        mask=np.asarray(raw['mask']),
                    )
                    candidate.metadata['generator'] = 'text'
                    candidate.metadata['slice_offset'] = slice_idx - mid_slice
                    candidate.area_frac = candidate.area_px / image_area
                    candidate.size_prior = lesion_size_prior(
                        candidate.area_frac,
                        self.config.target_mask_frac,
                        self.config.size_prior_strength,
                    )
                    if gt_slice is not None:
                        candidate.text_metrics = compute_mask_metrics(candidate.mask, gt_slice)
                    if self.config.verbose:
                        dice = candidate.text_metrics.dice if candidate.text_metrics else float('nan')
                        print(
                            '   ',
                            f'rank={candidate.rank}',
                            f'score={candidate.score:.4f}',
                            f'area={candidate.area_px}',
                            f'bbox={candidate.bbox_xyxy}',
                            f'dice={dice:.4f}' if candidate.text_metrics else 'dice=n/a',
                        )
                    candidates.append(candidate)

        if not candidates:
            return []

        cluster_candidates(candidates, self.config.cluster_iou)
        for candidate in candidates:
            candidate.rerank_score = compute_text_rerank_score(candidate)
        return candidates


class BrainMriVisualProposalGenerator(ProposalGenerator):
    def __init__(self, config: BrainMriVisualConfig):
        self.config = config

    def generate(self, context: StudyContext, target: FindingTarget) -> List[ProposalCandidate]:
        slice_indices = list(context.metadata['slice_indices'])
        mid_slice = int(context.metadata['mid_slice'])
        gt_volume = context.metadata.get('ground_truth_mask')
        image_area = int(context.image_volume.shape[0] * context.image_volume.shape[1])
        candidates: List[ProposalCandidate] = []

        for slice_idx in slice_indices:
            slice_img = np.asarray(context.image_volume[:, :, slice_idx], dtype=np.float32)
            normalized = _normalize_slice(slice_img)
            gt_slice = gt_volume[:, :, slice_idx] > 0 if gt_volume is not None else None
            slice_candidates: List[ProposalCandidate] = []

            for threshold in self.config.thresholds:
                binary = normalized >= float(threshold)
                for component_mask in _connected_components(binary):
                    area_px = int(component_mask.sum())
                    if area_px < self.config.min_component_px:
                        continue
                    area_frac = area_px / image_area
                    if area_frac > self.config.max_component_frac:
                        continue
                    bbox = _mask_to_bbox(component_mask)
                    if bbox is None:
                        continue
                    bbox_area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 1.0)
                    fill_ratio = min(area_px / bbox_area, 1.0)
                    score = float(threshold) * (0.6 + 0.4 * fill_ratio)
                    candidate = ProposalCandidate(
                        slice_idx=slice_idx,
                        prompt=self.config.proposal_prompt,
                        rank=0,
                        score=score,
                        bbox_xyxy=tuple(float(x) for x in bbox),
                        area_px=area_px,
                        mask=component_mask,
                    )
                    candidate.metadata['generator'] = 'visual'
                    candidate.metadata['visual_threshold'] = float(threshold)
                    candidate.metadata['slice_offset'] = slice_idx - mid_slice
                    candidate.area_frac = area_frac
                    candidate.size_prior = 1.0
                    candidate.bbox_fill_ratio = fill_ratio
                    if gt_slice is not None:
                        candidate.text_metrics = compute_mask_metrics(component_mask, gt_slice)
                    slice_candidates.append(candidate)

            if not slice_candidates:
                continue

            deduped: List[ProposalCandidate] = []
            for candidate in sorted(slice_candidates, key=lambda c: (float(c.score), float(c.area_px)), reverse=True):
                if any(bbox_iou(candidate.bbox_xyxy, existing.bbox_xyxy) >= 0.7 for existing in deduped):
                    continue
                deduped.append(candidate)
                if len(deduped) >= self.config.max_candidates_per_slice:
                    break
            candidates.extend(deduped)

        if not candidates:
            return []

        cluster_candidates(candidates, 0.3)
        for candidate in candidates:
            candidate.rerank_score = compute_text_rerank_score(candidate)
        return candidates


class SamBoxRefiner(SegmentationRefiner):
    def __init__(self, model: SAM3Model, config: BrainMriTextConfig):
        self.model = model
        self.config = config

    def refine(
        self,
        context: StudyContext,
        target: FindingTarget,
        candidates: Sequence[ProposalCandidate],
    ) -> List[ProposalCandidate]:
        slice_cache = _ensure_slice_cache(context, self.model)
        image_area = int(context.image_volume.shape[0] * context.image_volume.shape[1])
        gt_volume = context.metadata.get('ground_truth_mask')

        for candidate in candidates:
            shape = slice_cache['slice_shapes'][candidate.slice_idx]
            candidate.rounded_bbox_xyxy = round_box(candidate.bbox_xyxy, shape)
            bbox_width = max(float(candidate.bbox_xyxy[2]) - float(candidate.bbox_xyxy[0]), 1e-6)
            bbox_height = max(float(candidate.bbox_xyxy[3]) - float(candidate.bbox_xyxy[1]), 1e-6)
            bbox_area = float(bbox_width * bbox_height)
            candidate.metadata['bbox_area_px'] = bbox_area
            candidate.bbox_fill_ratio = min(1.0, float(candidate.area_px) / bbox_area)

            refined = self.model.predict_box_candidates(
                slice_cache['slice_states'][candidate.slice_idx],
                candidate.rounded_bbox_xyxy,
                shape,
                top_k=1,
            )
            if not refined:
                candidate.box_rerank_score = 0.15 * float(candidate.rerank_score or 0.0) * float(candidate.bbox_fill_ratio or 0.0)
                continue

            refined_best = refined[0]
            candidate.refined_score = float(refined_best['score'])
            candidate.refined_bbox_xyxy = tuple(float(x) for x in refined_best['bbox_xyxy'])
            candidate.refined_mask = np.asarray(refined_best['mask'])
            candidate.refined_area_px = int(refined_best['area_px'])
            candidate.refined_area_frac = candidate.refined_area_px / image_area
            candidate.refined_size_prior = lesion_size_prior(
                candidate.refined_area_frac,
                self.config.target_mask_frac,
                self.config.size_prior_strength,
            )
            refined_bbox_width = max(candidate.refined_bbox_xyxy[2] - candidate.refined_bbox_xyxy[0], 1e-6)
            refined_bbox_height = max(candidate.refined_bbox_xyxy[3] - candidate.refined_bbox_xyxy[1], 1e-6)
            refined_bbox_area = float(refined_bbox_width * refined_bbox_height)
            candidate.metadata['refined_bbox_area_px'] = refined_bbox_area
            candidate.refined_bbox_fill_ratio = min(1.0, float(candidate.refined_area_px) / refined_bbox_area)
            candidate.compactness_prior = math.sqrt(
                float(candidate.bbox_fill_ratio or 0.0) * float(candidate.refined_bbox_fill_ratio or 0.0)
            )
            candidate.refined_bbox_stability_iou = bbox_iou(candidate.bbox_xyxy, candidate.refined_bbox_xyxy)
            if gt_volume is not None:
                gt_slice = gt_volume[:, :, candidate.slice_idx] > 0
                candidate.refined_metrics = compute_mask_metrics(refined_best['mask'], gt_slice)
            candidate.box_rerank_score = compute_box_rerank_score(
                candidate,
                text_trust_score_threshold=self.config.text_trust_score_threshold,
                text_trust_min_slices=self.config.text_trust_min_slices,
                text_trust_min_prompts=self.config.text_trust_min_prompts,
                text_trust_boost=self.config.text_trust_boost,
            )
        return list(candidates)


class HeuristicCandidateSelector(CandidateSelector):
    def __init__(self) -> None:
        self.raw_best: Optional[ProposalCandidate] = None
        self.text_reranked_best: Optional[ProposalCandidate] = None
        self.box_reranked_best: Optional[ProposalCandidate] = None
        self.top_text_reranked: List[ProposalCandidate] = []
        self.top_box_reranked: List[ProposalCandidate] = []

    def select(
        self,
        context: StudyContext,
        target: FindingTarget,
        candidates: Sequence[ProposalCandidate],
    ) -> Optional[SegmentationSelection]:
        if not candidates:
            return None
        candidate_list = list(candidates)
        self.raw_best = max(candidate_list, key=lambda c: float(c.score))
        self.text_reranked_best = max(candidate_list, key=lambda c: float(c.rerank_score or 0.0))
        self.top_text_reranked = sorted(candidate_list, key=lambda c: float(c.rerank_score or 0.0), reverse=True)[:10]
        self.top_box_reranked = sorted(candidate_list, key=lambda c: float(c.box_rerank_score or 0.0), reverse=True)[:10]

        best_text = max(
            (candidate for candidate in candidate_list if candidate.metadata.get('generator') == 'text'),
            key=lambda c: float(c.box_rerank_score or 0.0),
            default=None,
        )
        best_visual = max(
            (candidate for candidate in candidate_list if candidate.metadata.get('generator') == 'visual'),
            key=lambda c: float(c.box_rerank_score or 0.0),
            default=None,
        )

        selected = max(candidate_list, key=lambda c: float(c.box_rerank_score or 0.0))
        strategy = 'text_then_box_rerank'
        if best_text is not None and best_visual is not None:
            text_supported = best_text.cluster_slice_count >= 3 and best_text.cluster_prompt_count >= 3
            text_close = float(best_text.box_rerank_score or 0.0) >= float(best_visual.box_rerank_score or 0.0) - 0.03
            if text_supported and text_close:
                selected = best_text
                strategy = 'text_preferred_with_visual_fallback'
            elif selected is best_visual:
                strategy = 'visual_fallback_then_box_rerank'
        self.box_reranked_best = selected
        return SegmentationSelection(strategy=strategy, candidate=selected)


def normalize_slice_to_rgb(slice_img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_img, [0.5, 99.5])
    scaled = np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (scaled * 255).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


def round_box(box_xyxy: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = box_xyxy
    x0 = int(max(0, min(width - 1, math.floor(x0))))
    y0 = int(max(0, min(height - 1, math.floor(y0))))
    x1 = int(max(x0 + 1, min(width, math.ceil(x1))))
    y1 = int(max(y0 + 1, min(height, math.ceil(y1))))
    return (x0, y0, x1, y1)


def summarize_gt(gt_slice: np.ndarray) -> Dict[str, object]:
    mask = gt_slice.astype(bool)
    ys, xs = np.where(mask)
    bbox = None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        'gt_pixels': int(mask.sum()),
        'gt_bbox_xyxy': bbox,
    }


def cluster_candidates(candidates: List[ProposalCandidate], iou_threshold: float) -> None:
    clusters: List[Dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            rep_box = cluster['representative_box']
            iou = bbox_iou(candidate.bbox_xyxy, rep_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is None:
            clusters.append({'representative_box': candidate.bbox_xyxy, 'members': [candidate]})
        else:
            clusters[best_idx]['members'].append(candidate)

    for cluster_id, cluster in enumerate(clusters, start=1):
        members = cluster['members']
        prompts = {member.prompt for member in members}
        slices = {member.slice_idx for member in members}
        for member in members:
            member.cluster_id = cluster_id
            member.cluster_prompt_count = len(prompts)
            member.cluster_slice_count = len(slices)


def candidate_to_summary(candidate: ProposalCandidate) -> Dict[str, object]:
    summary: Dict[str, object] = {
        'slice_idx': candidate.slice_idx,
        'prompt': candidate.prompt,
        'rank': candidate.rank,
        'score': candidate.score,
        'bbox_xyxy': list(candidate.bbox_xyxy),
        'area_px': candidate.area_px,
        'rounded_bbox_xyxy': list(candidate.rounded_bbox_xyxy) if candidate.rounded_bbox_xyxy is not None else None,
        'area_frac': candidate.area_frac,
        'size_prior': candidate.size_prior,
        'cluster_id': candidate.cluster_id,
        'cluster_prompt_count': candidate.cluster_prompt_count,
        'cluster_slice_count': candidate.cluster_slice_count,
        'rerank_score': candidate.rerank_score,
        'refined_score': candidate.refined_score,
        'refined_bbox_xyxy': list(candidate.refined_bbox_xyxy) if candidate.refined_bbox_xyxy is not None else None,
        'refined_mask_pixels': int(candidate.refined_mask.sum()) if candidate.refined_mask is not None else None,
        'refined_area_px': candidate.refined_area_px,
        'refined_area_frac': candidate.refined_area_frac,
        'refined_size_prior': candidate.refined_size_prior,
        'refined_bbox_stability_iou': candidate.refined_bbox_stability_iou,
        'bbox_fill_ratio': candidate.bbox_fill_ratio,
        'refined_bbox_fill_ratio': candidate.refined_bbox_fill_ratio,
        'compactness_prior': candidate.compactness_prior,
        'text_trust_bonus': candidate.text_trust_bonus,
        'box_rerank_score': candidate.box_rerank_score,
        'metadata': dict(candidate.metadata),
    }
    if candidate.text_metrics is not None:
        summary['text_metrics'] = {
            'pred_pixels': candidate.text_metrics.pred_pixels,
            'gt_pixels': candidate.text_metrics.gt_pixels,
            'intersection': candidate.text_metrics.intersection,
            'dice': candidate.text_metrics.dice,
            'iou': candidate.text_metrics.iou,
        }
    else:
        summary['text_metrics'] = None
    if candidate.refined_metrics is not None:
        summary['refined_metrics'] = {
            'pred_pixels': candidate.refined_metrics.pred_pixels,
            'gt_pixels': candidate.refined_metrics.gt_pixels,
            'intersection': candidate.refined_metrics.intersection,
            'dice': candidate.refined_metrics.dice,
            'iou': candidate.refined_metrics.iou,
        }
    else:
        summary['refined_metrics'] = None
    return summary


def _ensure_slice_cache(context: StudyContext, model: SAM3Model) -> Dict[str, object]:
    cache = context.metadata.setdefault('slice_cache', {})
    if cache.get('ready'):
        return cache

    slice_states: Dict[int, dict] = {}
    slice_shapes: Dict[int, Tuple[int, int]] = {}
    for slice_idx in context.metadata['slice_indices']:
        rgb = normalize_slice_to_rgb(context.image_volume[:, :, slice_idx])
        slice_states[slice_idx] = model.encode_image(rgb)
        slice_shapes[slice_idx] = rgb.shape[:2]
    cache['slice_states'] = slice_states
    cache['slice_shapes'] = slice_shapes
    cache['ready'] = True
    return cache


def _normalize_slice(slice_img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_img, [1.0, 99.8])
    return np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _connected_components(binary: np.ndarray) -> List[np.ndarray]:
    binary = np.asarray(binary, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[np.ndarray] = []

    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            component = np.zeros_like(binary, dtype=bool)
            ys, xs = zip(*pixels)
            component[np.asarray(ys), np.asarray(xs)] = True
            components.append(component)
    return components
