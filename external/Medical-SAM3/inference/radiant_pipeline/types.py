from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BBoxXYXY = Tuple[float, float, float, float]
PixelBBoxXYXY = Tuple[int, int, int, int]
VoxelBBoxXYZXYZ = Tuple[int, int, int, int, int, int]


@dataclass(slots=True)
class FindingTarget:
    """Normalized finding extracted from a report or other clinical metadata."""

    finding: str
    anatomy: str
    laterality: str = 'unknown'
    sub_anatomy: Optional[str] = None
    size_mm: Sequence[float] = field(default_factory=list)
    certainty: str = 'unknown'
    modality_hint: Optional[str] = None
    support_status: str = 'unknown'
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class BoxPrompt:
    """Pixel-space box prompt for SAM refinement."""

    bbox_xyxy: PixelBBoxXYXY
    source: str


@dataclass(slots=True)
class CandidateMetrics:
    """Overlap metrics for debugging and evaluation."""

    pred_pixels: int
    gt_pixels: int
    intersection: int
    dice: float
    iou: float


@dataclass(slots=True)
class ProposalCandidate:
    """Generic proposal record used before and after refinement."""

    slice_idx: int
    prompt: str
    rank: int
    score: float
    bbox_xyxy: BBoxXYXY
    area_px: int
    mask: Optional[np.ndarray] = None
    rounded_bbox_xyxy: Optional[PixelBBoxXYXY] = None
    area_frac: Optional[float] = None
    size_prior: Optional[float] = None
    cluster_id: Optional[int] = None
    cluster_prompt_count: int = 1
    cluster_slice_count: int = 1
    rerank_score: Optional[float] = None
    refined_score: Optional[float] = None
    refined_bbox_xyxy: Optional[BBoxXYXY] = None
    refined_mask: Optional[np.ndarray] = None
    refined_area_px: Optional[int] = None
    refined_area_frac: Optional[float] = None
    refined_size_prior: Optional[float] = None
    refined_bbox_stability_iou: Optional[float] = None
    bbox_fill_ratio: Optional[float] = None
    refined_bbox_fill_ratio: Optional[float] = None
    compactness_prior: Optional[float] = None
    text_trust_bonus: float = 0.0
    box_rerank_score: Optional[float] = None
    text_metrics: Optional[CandidateMetrics] = None
    refined_metrics: Optional[CandidateMetrics] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationSelection:
    """Final selected candidate for a study or slice window."""

    strategy: str
    candidate: ProposalCandidate
    warnings: List[str] = field(default_factory=list)


@dataclass(slots=True)
class VolumeAssembly:
    """Provisional 3D mask assembled from refined slice-wise candidates."""

    source_route: str
    selected_slice_idx: int
    selected_cluster_id: Optional[int]
    selected_prompt: str
    selected_generator: str
    slice_indices: List[int]
    per_slice_voxels: Dict[int, int]
    voxel_count: int
    bbox_xyzxyz: Optional[VoxelBBoxXYZXYZ]
    support_candidate_count: int
    mask_volume: np.ndarray


@dataclass(slots=True)
class PipelineRunArtifacts:
    """Detailed artifacts emitted by the generic orchestration shell."""

    target: FindingTarget
    candidates: List[ProposalCandidate]
    selection: Optional[SegmentationSelection]


@dataclass(slots=True)
class PipelineResult:
    """User-facing pipeline result with route classification and 3D assembly."""

    route: str
    target: FindingTarget
    selection: Optional[SegmentationSelection]
    volume_assembly: Optional[VolumeAssembly] = None
    candidates: List[ProposalCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(slots=True)
class StudyContext:
    """Minimal study bundle consumed by the segmentation pipeline."""

    case_id: str
    modality: str
    sequence: str
    image_volume: np.ndarray
    report_text: str
    metadata: Dict[str, object] = field(default_factory=dict)
