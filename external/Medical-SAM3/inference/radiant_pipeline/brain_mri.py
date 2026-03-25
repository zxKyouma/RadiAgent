from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from sam3_inference import SAM3Model
else:
    SAM3Model = Any

from .orchestrator import BoxProposer, CandidateSelector, FindingExtractor, Localizer, SegmentationRefiner
from .scoring import bbox_iou, compute_box_rerank_score, compute_mask_metrics, compute_text_rerank_score, lesion_size_prior
from .types import BoxProposal, LocalizerHypothesis, ProposalCandidate, SegmentationSelection, StructuredTarget, StudyContext


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


@dataclass(slots=True)
class BrainMriHeuristicLocalizerConfig:
    slice_radius: int = 2
    coarse_stride: int = 4
    shortlist_size: int = 3
    thresholds: Sequence[float] = (0.985, 0.99, 0.995)
    min_component_px: int = 12
    max_component_frac: float = 0.03
    min_center_separation: int = 3
    laterality_bonus: float = 0.1


class BrainMriFindingExtractor(FindingExtractor):
    def extract(self, context: StudyContext) -> StructuredTarget:
        normalized = context.report_text.lower()
        laterality = 'unknown'
        if 'left' in normalized:
            laterality = 'left'
        elif 'right' in normalized:
            laterality = 'right'
        finding = context.metadata.get('finding_text') or 'intracranial mass'
        sub_anatomy = context.metadata.get('sub_anatomy')
        return StructuredTarget(
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


class MetadataWindowLocalizer(Localizer):
    """Temporary localizer that reuses metadata-defined slice windows."""

    def localize(self, context: StudyContext, target: StructuredTarget) -> List[LocalizerHypothesis]:
        slice_indices = [int(idx) for idx in context.metadata.get('slice_indices', [])]
        if not slice_indices:
            return []
        center_slice = int(context.metadata.get('mid_slice', slice_indices[len(slice_indices) // 2]))
        return [
            LocalizerHypothesis(
                source='metadata_window',
                score=1.0,
                center_slice=center_slice,
                slice_indices=slice_indices,
                metadata={
                    'case_id': context.case_id,
                    'sequence': context.sequence,
                },
            )
        ]


class BrainMriHeuristicLocalizer(Localizer):
    """Cheap image-only localizer that shortlists promising slice windows."""

    def __init__(self, config: BrainMriHeuristicLocalizerConfig):
        self.config = config

    def localize(self, context: StudyContext, target: StructuredTarget) -> List[LocalizerHypothesis]:
        depth = int(context.image_volume.shape[2])
        centers = list(range(0, depth, max(1, int(self.config.coarse_stride))))
        if not centers or centers[-1] != depth - 1:
            centers.append(depth - 1)

        scored_centers = [
            (center, self._score_window(context.image_volume, center, target))
            for center in centers
        ]
        scored_centers.sort(key=lambda item: item[1], reverse=True)

        hypotheses: List[LocalizerHypothesis] = []
        for center, score in scored_centers:
            if score <= 0.0:
                continue
            if any(abs(center - existing.center_slice) < self.config.min_center_separation for existing in hypotheses):
                continue
            slice_indices = [
                s
                for s in range(center - self.config.slice_radius, center + self.config.slice_radius + 1)
                if 0 <= s < depth
            ]
            hypotheses.append(
                LocalizerHypothesis(
                    source='heuristic_localizer',
                    score=float(score),
                    center_slice=int(center),
                    slice_indices=slice_indices,
                    metadata={
                        'case_id': context.case_id,
                        'sequence': context.sequence,
                    },
                )
            )
            if len(hypotheses) >= self.config.shortlist_size:
                break
        return hypotheses

    def _score_window(self, image_volume: np.ndarray, center: int, target: StructuredTarget) -> float:
        depth = int(image_volume.shape[2])
        best_score = 0.0
        for slice_idx in range(center - self.config.slice_radius, center + self.config.slice_radius + 1):
            if slice_idx < 0 or slice_idx >= depth:
                continue
            slice_img = np.asarray(image_volume[:, :, slice_idx], dtype=np.float32)
            slice_score = self._score_slice(slice_img, target)
            if slice_score > best_score:
                best_score = slice_score
        return best_score

    def _score_slice(self, slice_img: np.ndarray, target: StructuredTarget) -> float:
        normalized = _normalize_slice(slice_img)
        image_area = float(normalized.shape[0] * normalized.shape[1])
        best_score = 0.0
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
                if target.laterality in {'left', 'right'}:
                    score += self._laterality_bonus(bbox, normalized.shape[1], target.laterality)
                best_score = max(best_score, score)
        return best_score

    def _laterality_bonus(self, bbox: Tuple[int, int, int, int], width: int, laterality: str) -> float:
        center_x = 0.5 * (bbox[0] + bbox[2])
        if laterality == 'left' and center_x < width * 0.5:
            return self.config.laterality_bonus
        if laterality == 'right' and center_x >= width * 0.5:
            return self.config.laterality_bonus
        return 0.0


class BrainMriTextProposalGenerator(BoxProposer):
    def __init__(self, model: SAM3Model, config: BrainMriTextConfig):
        self.model = model
        self.config = config

    def propose(
        self,
        context: StudyContext,
        target: StructuredTarget,
        hypothesis: LocalizerHypothesis,
    ) -> List[BoxProposal]:
        slice_indices = list(hypothesis.slice_indices)
        mid_slice = int(context.metadata.get('mid_slice', hypothesis.center_slice))
        gt_volume = context.metadata.get('ground_truth_mask')
        slice_cache = _ensure_slice_cache(context, self.model)
        image_area = int(context.image_volume.shape[0] * context.image_volume.shape[1])
        proposals: List[BoxProposal] = []

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
                    proposal = BoxProposal(
                        slice_idx=slice_idx,
                        prompt=prompt,
                        rank=int(raw['rank']),
                        score=float(raw['score']),
                        bbox_xyxy=tuple(float(x) for x in raw['bbox_xyxy']),
                        area_px=int(raw['area_px']),
                        source='text',
                        hypothesis=hypothesis,
                        mask=np.asarray(raw['mask']),
                        metadata={
                            'generator': 'text',
                            'slice_offset': slice_idx - mid_slice,
                            'area_frac': int(raw['area_px']) / image_area,
                        },
                    )
                    if gt_slice is not None:
                        proposal.metadata['text_metrics'] = compute_mask_metrics(proposal.mask, gt_slice)
                    if self.config.verbose:
                        metrics = proposal.metadata.get('text_metrics')
                        dice = metrics.dice if metrics is not None else float('nan')
                        print(
                            '   ',
                            f'rank={proposal.rank}',
                            f'score={proposal.score:.4f}',
                            f'area={proposal.area_px}',
                            f'bbox={proposal.bbox_xyxy}',
                            f'dice={dice:.4f}' if metrics is not None else 'dice=n/a',
                        )
                    proposals.append(proposal)

        return proposals


class BrainMriVisualProposalGenerator(BoxProposer):
    def __init__(self, config: BrainMriVisualConfig):
        self.config = config

    def propose(
        self,
        context: StudyContext,
        target: StructuredTarget,
        hypothesis: LocalizerHypothesis,
    ) -> List[BoxProposal]:
        slice_indices = list(hypothesis.slice_indices)
        mid_slice = int(context.metadata.get('mid_slice', hypothesis.center_slice))
        gt_volume = context.metadata.get('ground_truth_mask')
        image_area = int(context.image_volume.shape[0] * context.image_volume.shape[1])
        proposals: List[BoxProposal] = []

        for slice_idx in slice_indices:
            slice_img = np.asarray(context.image_volume[:, :, slice_idx], dtype=np.float32)
            normalized = _normalize_slice(slice_img)
            gt_slice = gt_volume[:, :, slice_idx] > 0 if gt_volume is not None else None
            slice_proposals: List[BoxProposal] = []

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
                    proposal = BoxProposal(
                        slice_idx=slice_idx,
                        prompt=self.config.proposal_prompt,
                        rank=0,
                        score=score,
                        bbox_xyxy=tuple(float(x) for x in bbox),
                        area_px=area_px,
                        source='visual',
                        hypothesis=hypothesis,
                        mask=component_mask,
                        metadata={
                            'generator': 'visual',
                            'visual_threshold': float(threshold),
                            'slice_offset': slice_idx - mid_slice,
                            'area_frac': area_frac,
                            'bbox_fill_ratio': fill_ratio,
                        },
                    )
                    if gt_slice is not None:
                        proposal.metadata['text_metrics'] = compute_mask_metrics(component_mask, gt_slice)
                    slice_proposals.append(proposal)

            if not slice_proposals:
                continue

            deduped: List[BoxProposal] = []
            for proposal in sorted(slice_proposals, key=lambda c: (float(c.score), float(c.area_px)), reverse=True):
                if any(bbox_iou(proposal.bbox_xyxy, existing.bbox_xyxy) >= 0.7 for existing in deduped):
                    continue
                deduped.append(proposal)
                if len(deduped) >= self.config.max_candidates_per_slice:
                    break
            proposals.extend(deduped)

        return proposals


class SamBoxRefiner(SegmentationRefiner):
    def __init__(self, model: SAM3Model, config: BrainMriTextConfig):
        self.model = model
        self.config = config

    def refine(
        self,
        context: StudyContext,
        target: StructuredTarget,
        proposals: Sequence[BoxProposal],
    ) -> List[ProposalCandidate]:
        slice_cache = _ensure_slice_cache(context, self.model)
        image_area = int(context.image_volume.shape[0] * context.image_volume.shape[1])
        gt_volume = context.metadata.get('ground_truth_mask')

        candidates: List[ProposalCandidate] = []
        for proposal in proposals:
            candidate = ProposalCandidate(
                slice_idx=proposal.slice_idx,
                prompt=proposal.prompt,
                rank=proposal.rank,
                score=proposal.score,
                bbox_xyxy=proposal.bbox_xyxy,
                area_px=proposal.area_px,
                mask=None if proposal.mask is None else np.asarray(proposal.mask),
                metadata=dict(proposal.metadata),
            )
            candidate.metadata.setdefault('generator', proposal.source)
            candidate.metadata['localizer_source'] = proposal.hypothesis.source
            candidate.metadata['localizer_score'] = proposal.hypothesis.score
            candidate.metadata['localizer_center_slice'] = proposal.hypothesis.center_slice
            candidate.area_frac = float(candidate.metadata.get('area_frac', candidate.area_px / image_area))
            candidate.size_prior = lesion_size_prior(
                candidate.area_frac,
                self.config.target_mask_frac,
                self.config.size_prior_strength,
            ) if proposal.source == 'text' else 1.0
            candidate.bbox_fill_ratio = float(candidate.metadata.get('bbox_fill_ratio', 0.0)) or None
            metrics = candidate.metadata.get('text_metrics')
            if metrics is not None:
                candidate.text_metrics = metrics
            candidates.append(candidate)

        if not candidates:
            return []

        cluster_candidates(candidates, self.config.cluster_iou)
        for candidate in candidates:
            candidate.rerank_score = compute_text_rerank_score(candidate)

        for candidate in candidates:
            shape = slice_cache['slice_shapes'][candidate.slice_idx]
            candidate.rounded_bbox_xyxy = round_box(candidate.bbox_xyxy, shape)
            bbox_width = max(float(candidate.bbox_xyxy[2]) - float(candidate.bbox_xyxy[0]), 1e-6)
            bbox_height = max(float(candidate.bbox_xyxy[3]) - float(candidate.bbox_xyxy[1]), 1e-6)
            bbox_area = float(bbox_width * bbox_height)
            candidate.metadata['bbox_area_px'] = bbox_area
            if candidate.bbox_fill_ratio is None:
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
        return candidates


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
        target: StructuredTarget,
        candidates: Sequence[ProposalCandidate],
    ) -> Optional[SegmentationSelection]:
        if not candidates:
            return None
        candidate_list = list(candidates)
        self.raw_best = max(candidate_list, key=lambda c: float(c.score))
        self.text_reranked_best = max(candidate_list, key=lambda c: float(c.rerank_score or 0.0))
        self.top_text_reranked = sorted(candidate_list, key=lambda c: float(c.rerank_score or 0.0), reverse=True)[:10]
        self.top_box_reranked = sorted(candidate_list, key=lambda c: float(c.box_rerank_score or 0.0), reverse=True)[:10]

        for candidate in candidate_list:
            candidate.metadata['selection_score'] = _selection_evidence_score(candidate)

        best_text = max(
            (candidate for candidate in candidate_list if candidate.metadata.get('generator') == 'text'),
            key=lambda c: float(c.metadata.get('selection_score', 0.0)),
            default=None,
        )
        best_visual = max(
            (candidate for candidate in candidate_list if candidate.metadata.get('generator') == 'visual'),
            key=lambda c: float(c.metadata.get('selection_score', 0.0)),
            default=None,
        )

        selected = max(candidate_list, key=lambda c: float(c.metadata.get('selection_score', 0.0)))
        strategy = 'evidence_ranked_selection'
        if best_text is not None and best_visual is not None:
            text_score = float(best_text.metadata.get('selection_score', 0.0))
            visual_score = float(best_visual.metadata.get('selection_score', 0.0))
            text_dice_proxy = 0.0 if best_text.refined_metrics is None else float(best_text.refined_metrics.dice)
            visual_dice_proxy = 0.0 if best_visual.refined_metrics is None else float(best_visual.refined_metrics.dice)
            text_supported = best_text.cluster_slice_count >= 3 and best_text.cluster_prompt_count >= 3
            visual_clearly_better = visual_score >= text_score - 0.03 and visual_dice_proxy >= max(0.5, text_dice_proxy + 0.25)
            text_clearly_bad = text_dice_proxy <= 0.05
            if visual_clearly_better and text_clearly_bad:
                selected = best_visual
                strategy = 'visual_promoted_over_bad_text'
            elif text_supported and text_score >= visual_score - 0.03:
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
        'selection_score': candidate.metadata.get('selection_score'),
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
    slice_indices = context.metadata.get('slice_indices')
    if slice_indices is None:
        slice_indices = list(range(context.image_volume.shape[2]))
    requested = tuple(int(idx) for idx in slice_indices)
    if cache.get('ready') and cache.get('slice_indices') == requested:
        return cache

    slice_states: Dict[int, dict] = {}
    slice_shapes: Dict[int, Tuple[int, int]] = {}
    for slice_idx in requested:
        rgb = normalize_slice_to_rgb(context.image_volume[:, :, slice_idx])
        slice_states[slice_idx] = model.encode_image(rgb)
        slice_shapes[slice_idx] = rgb.shape[:2]
    cache['slice_states'] = slice_states
    cache['slice_shapes'] = slice_shapes
    cache['slice_indices'] = requested
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


def _selection_evidence_score(candidate: ProposalCandidate) -> float:
    refined_score = float(candidate.refined_score or 0.0)
    compactness = float(candidate.compactness_prior or 0.0)
    stability = float(candidate.refined_bbox_stability_iou or 0.0)
    fill_ratio = float(candidate.refined_bbox_fill_ratio or candidate.bbox_fill_ratio or 0.0)
    localizer_score = float(candidate.metadata.get('localizer_score', 0.0))
    support_score = min(candidate.cluster_slice_count / 4.0, 1.0) + min(candidate.cluster_prompt_count / 4.0, 1.0)
    support_score *= 0.05
    text_bonus = float(candidate.text_trust_bonus or 0.0)

    score = (
        0.48 * refined_score
        + 0.16 * stability
        + 0.12 * compactness
        + 0.07 * fill_ratio
        + 0.07 * localizer_score
        + support_score
        + 0.05 * min(text_bonus, 1.0)
    )

    generator = str(candidate.metadata.get('generator', 'unknown'))
    if generator == 'text':
        if candidate.cluster_slice_count < 2 or candidate.cluster_prompt_count < 2:
            score -= 0.05
        if refined_score < 0.7 and stability < 0.55:
            score -= 0.08
    elif generator == 'visual':
        if refined_score >= 0.75 and compactness >= 0.35:
            score += 0.05

    return float(score)
