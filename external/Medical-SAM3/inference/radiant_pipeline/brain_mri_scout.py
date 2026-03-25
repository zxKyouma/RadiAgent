from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from sam3_inference import SAM3Model

from .brain_mri_runtime import BrainMriRouteConfig, build_pipeline_result
from .orchestrator import SegmentationPipeline
from .types import PipelineResult, StudyContext


def normalize_slice_to_rgb(slice_img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_img, [0.5, 99.5])
    scaled = np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (scaled * 255).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


@dataclass(slots=True)
class BrainMriScoutConfig:
    prompts: Sequence[str]
    slice_radius: int = 2
    coarse_stride: int = 8
    shortlist_size: int = 3
    text_weight: float = 0.8
    visual_weight: float = 0.2
    visual_threshold: float = 0.99
    min_component_px: int = 12
    max_component_frac: float = 0.03


@dataclass(slots=True)
class SliceScoutResult:
    best_center_idx: int
    slice_indices: List[int]
    best_result: PipelineResult
    coarse_scores: Dict[int, float]
    shortlisted_centers: List[int]


def scout_slice_window(
    context: StudyContext,
    pipeline: SegmentationPipeline,
    route_config: BrainMriRouteConfig,
    model: SAM3Model,
    config: BrainMriScoutConfig,
) -> SliceScoutResult:
    depth = context.image_volume.shape[2]
    centers = list(range(0, depth, max(1, int(config.coarse_stride))))
    coarse_scores = {
        center: _coarse_window_score(context.image_volume, center, model, config)
        for center in centers
    }
    shortlisted = sorted(centers, key=lambda center: coarse_scores[center], reverse=True)[: max(1, config.shortlist_size)]

    best_center = shortlisted[0]
    best_result: Optional[PipelineResult] = None
    for center in shortlisted:
        slice_indices = [
            s for s in range(center - config.slice_radius, center + config.slice_radius + 1)
            if 0 <= s < depth
        ]
        scout_context = StudyContext(
            case_id=context.case_id,
            modality=context.modality,
            sequence=context.sequence,
            image_volume=context.image_volume,
            report_text=context.report_text,
            metadata={
                **context.metadata,
                'slice_indices': slice_indices,
                'mid_slice': center,
                'slice_cache': {},
            },
        )
        artifacts = pipeline.run_detailed(scout_context)
        result = build_pipeline_result(scout_context, artifacts, route_config=route_config)
        if best_result is None or _result_rank(result) > _result_rank(best_result):
            best_center = center
            best_result = result

    assert best_result is not None
    final_slice_indices = [
        s for s in range(best_center - config.slice_radius, best_center + config.slice_radius + 1)
        if 0 <= s < depth
    ]
    return SliceScoutResult(
        best_center_idx=best_center,
        slice_indices=final_slice_indices,
        best_result=best_result,
        coarse_scores=coarse_scores,
        shortlisted_centers=shortlisted,
    )


def _coarse_window_score(
    image_volume: np.ndarray,
    center: int,
    model: SAM3Model,
    config: BrainMriScoutConfig,
) -> float:
    depth = image_volume.shape[2]
    slice_indices = [
        s for s in range(center - config.slice_radius, center + config.slice_radius + 1)
        if 0 <= s < depth
    ]
    if not slice_indices:
        return 0.0
    best_text = 0.0
    best_visual = 0.0
    for slice_idx in slice_indices:
        slice_img = np.asarray(image_volume[:, :, slice_idx], dtype=np.float32)
        rgb = normalize_slice_to_rgb(slice_img)
        inference_state = model.encode_image(rgb)
        for prompt in config.prompts:
            candidates = model.predict_text_candidates(inference_state, text_prompt=prompt, top_k=1)
            if candidates:
                best_text = max(best_text, float(candidates[0]['score']))
        best_visual = max(best_visual, _visual_slice_score(slice_img, config))
    return config.text_weight * best_text + config.visual_weight * best_visual


def _visual_slice_score(slice_img: np.ndarray, config: BrainMriScoutConfig) -> float:
    normalized = _normalize_slice(slice_img)
    image_area = float(normalized.shape[0] * normalized.shape[1])
    binary = normalized >= float(config.visual_threshold)
    best_score = 0.0
    for component_mask in _connected_components(binary):
        area_px = int(component_mask.sum())
        if area_px < config.min_component_px:
            continue
        area_frac = area_px / image_area
        if area_frac > config.max_component_frac:
            continue
        bbox = _mask_to_bbox(component_mask)
        if bbox is None:
            continue
        bbox_area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 1.0)
        fill_ratio = min(area_px / bbox_area, 1.0)
        score = float(config.visual_threshold) * (0.6 + 0.4 * fill_ratio)
        best_score = max(best_score, score)
    return best_score


def _result_rank(result: PipelineResult) -> tuple[int, float]:
    route_priority = {
        'text_primary': 2,
        'visual_fallback': 1,
        'abstain': 0,
    }
    score = 0.0
    if result.selection is not None:
        score = float(result.selection.candidate.box_rerank_score or 0.0)
    return (route_priority.get(result.route, -1), score)


def _normalize_slice(slice_img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_img, [1.0, 99.8])
    return np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _mask_to_bbox(mask: np.ndarray):
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
            pixels = []
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
