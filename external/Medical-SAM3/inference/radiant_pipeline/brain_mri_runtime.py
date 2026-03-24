from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .types import PipelineResult, PipelineRunArtifacts, ProposalCandidate, StudyContext, VolumeAssembly, VoxelBBoxXYZXYZ


@dataclass(slots=True)
class BrainMriRouteConfig:
    min_box_rerank_score: float = 0.5
    min_volume_voxels: int = 20


def build_pipeline_result(
    context: StudyContext,
    artifacts: PipelineRunArtifacts,
    route_config: Optional[BrainMriRouteConfig] = None,
) -> PipelineResult:
    route_config = route_config or BrainMriRouteConfig()
    warnings: List[str] = []
    selection = artifacts.selection
    if selection is None:
        return PipelineResult(
            route='abstain',
            target=artifacts.target,
            selection=None,
            candidates=artifacts.candidates,
            warnings=['No viable candidate survived proposal and refinement.'],
        )

    selected = selection.candidate
    final_score = float(selected.box_rerank_score or 0.0)
    if final_score < route_config.min_box_rerank_score:
        warnings.append(
            f'Final rerank score {final_score:.4f} is below the abstain threshold {route_config.min_box_rerank_score:.4f}.'
        )
        return PipelineResult(
            route='abstain',
            target=artifacts.target,
            selection=selection,
            candidates=artifacts.candidates,
            warnings=warnings,
        )

    route = 'visual_fallback' if selected.metadata.get('generator') == 'visual' else 'text_primary'
    volume_assembly = assemble_candidate_volume(context, selected, artifacts.candidates, source_route=route)
    if volume_assembly is None or volume_assembly.voxel_count < route_config.min_volume_voxels:
        warnings.append('Assembled provisional 3D mask is empty or too small.')
        return PipelineResult(
            route='abstain',
            target=artifacts.target,
            selection=selection,
            candidates=artifacts.candidates,
            warnings=warnings,
        )

    return PipelineResult(
        route=route,
        target=artifacts.target,
        selection=selection,
        volume_assembly=volume_assembly,
        candidates=artifacts.candidates,
        warnings=warnings,
    )


def assemble_candidate_volume(
    context: StudyContext,
    selected: ProposalCandidate,
    candidates: List[ProposalCandidate],
    source_route: str,
) -> Optional[VolumeAssembly]:
    support_candidates = _support_candidates(selected, candidates)
    if not support_candidates:
        return None

    mask_volume = np.zeros_like(context.image_volume, dtype=np.uint8)
    per_slice_voxels: Dict[int, int] = {}
    slice_indices: List[int] = []

    best_by_slice: Dict[int, ProposalCandidate] = {}
    for candidate in support_candidates:
        current = best_by_slice.get(candidate.slice_idx)
        if current is None or float(candidate.box_rerank_score or 0.0) > float(current.box_rerank_score or 0.0):
            best_by_slice[candidate.slice_idx] = candidate

    for slice_idx in sorted(best_by_slice):
        candidate = best_by_slice[slice_idx]
        refined_mask = candidate.refined_mask
        if refined_mask is None:
            continue
        binary = np.asarray(refined_mask).astype(bool)
        voxel_count = int(binary.sum())
        if voxel_count == 0:
            continue
        mask_volume[:, :, slice_idx] = binary.astype(np.uint8)
        per_slice_voxels[slice_idx] = voxel_count
        slice_indices.append(slice_idx)

    if not slice_indices:
        return None

    bbox = _volume_bbox(mask_volume)
    return VolumeAssembly(
        source_route=source_route,
        selected_slice_idx=selected.slice_idx,
        selected_cluster_id=selected.cluster_id,
        selected_prompt=selected.prompt,
        selected_generator=str(selected.metadata.get('generator', 'unknown')),
        slice_indices=slice_indices,
        per_slice_voxels=per_slice_voxels,
        voxel_count=int(mask_volume.sum()),
        bbox_xyzxyz=bbox,
        support_candidate_count=len(support_candidates),
        mask_volume=mask_volume,
    )


def summarize_pipeline_result(result: PipelineResult) -> Dict[str, object]:
    selection = result.selection.candidate if result.selection is not None else None
    summary: Dict[str, object] = {
        'route': result.route,
        'warnings': list(result.warnings),
        'target': {
            'finding': result.target.finding,
            'anatomy': result.target.anatomy,
            'laterality': result.target.laterality,
            'sub_anatomy': result.target.sub_anatomy,
            'support_status': result.target.support_status,
            'modality_hint': result.target.modality_hint,
            'metadata': dict(result.target.metadata),
        },
        'selection': None,
        'volume_assembly': None,
    }
    if selection is not None:
        summary['selection'] = {
            'strategy': result.selection.strategy,
            'slice_idx': selection.slice_idx,
            'prompt': selection.prompt,
            'generator': selection.metadata.get('generator'),
            'cluster_id': selection.cluster_id,
            'box_rerank_score': selection.box_rerank_score,
            'refined_score': selection.refined_score,
            'refined_bbox_xyxy': list(selection.refined_bbox_xyxy) if selection.refined_bbox_xyxy is not None else None,
            'refined_metrics': None if selection.refined_metrics is None else {
                'dice': selection.refined_metrics.dice,
                'iou': selection.refined_metrics.iou,
                'pred_pixels': selection.refined_metrics.pred_pixels,
                'gt_pixels': selection.refined_metrics.gt_pixels,
                'intersection': selection.refined_metrics.intersection,
            },
        }
    if result.volume_assembly is not None:
        summary['volume_assembly'] = {
            'source_route': result.volume_assembly.source_route,
            'selected_slice_idx': result.volume_assembly.selected_slice_idx,
            'selected_cluster_id': result.volume_assembly.selected_cluster_id,
            'selected_prompt': result.volume_assembly.selected_prompt,
            'selected_generator': result.volume_assembly.selected_generator,
            'slice_indices': list(result.volume_assembly.slice_indices),
            'per_slice_voxels': {str(k): v for k, v in result.volume_assembly.per_slice_voxels.items()},
            'voxel_count': result.volume_assembly.voxel_count,
            'bbox_xyzxyz': list(result.volume_assembly.bbox_xyzxyz) if result.volume_assembly.bbox_xyzxyz is not None else None,
            'support_candidate_count': result.volume_assembly.support_candidate_count,
        }
    return summary


def _support_candidates(selected: ProposalCandidate, candidates: List[ProposalCandidate]) -> List[ProposalCandidate]:
    selected_generator = selected.metadata.get('generator')
    selected_cluster = selected.cluster_id
    matching = [
        candidate for candidate in candidates
        if candidate.metadata.get('generator') == selected_generator
        and candidate.cluster_id == selected_cluster
        and candidate.refined_mask is not None
    ]
    if matching:
        return matching
    return [selected] if selected.refined_mask is not None else []


def _volume_bbox(mask_volume: np.ndarray) -> Optional[VoxelBBoxXYZXYZ]:
    xs, ys, zs = np.where(mask_volume > 0)
    if len(xs) == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(zs.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
        int(zs.max()) + 1,
    )
