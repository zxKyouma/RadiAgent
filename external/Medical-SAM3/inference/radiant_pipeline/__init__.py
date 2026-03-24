"""Reusable pipeline primitives for report-guided segmentation orchestration."""

from .brain_mri import (
    DEFAULT_BRAIN_MRI_PROMPTS,
    BrainMriFindingExtractor,
    BrainMriTextConfig,
    BrainMriTextProposalGenerator,
    BrainMriVisualConfig,
    BrainMriVisualProposalGenerator,
    HeuristicCandidateSelector,
    SamBoxRefiner,
    candidate_to_summary,
)
from .brain_mri_runtime import (
    BrainMriRouteConfig,
    build_pipeline_result,
    summarize_pipeline_result,
)
from .orchestrator import (
    CandidateSelector,
    FindingExtractor,
    ProposalGenerator,
    SegmentationPipeline,
    SegmentationRefiner,
)
from .scoring import (
    bbox_iou,
    compute_box_rerank_score,
    compute_mask_metrics,
    compute_text_rerank_score,
    lesion_size_prior,
)
from .types import (
    BoxPrompt,
    CandidateMetrics,
    FindingTarget,
    PipelineResult,
    PipelineRunArtifacts,
    ProposalCandidate,
    SegmentationSelection,
    StudyContext,
    VolumeAssembly,
)

__all__ = [
    'BoxPrompt',
    'CandidateMetrics',
    'CandidateSelector',
    'FindingExtractor',
    'FindingTarget',
    'PipelineResult',
    'PipelineRunArtifacts',
    'ProposalCandidate',
    'ProposalGenerator',
    'SegmentationPipeline',
    'SegmentationRefiner',
    'SegmentationSelection',
    'StudyContext',
    'VolumeAssembly',
    'bbox_iou',
    'DEFAULT_BRAIN_MRI_PROMPTS',
    'BrainMriFindingExtractor',
    'BrainMriRouteConfig',
    'BrainMriTextConfig',
    'BrainMriTextProposalGenerator',
    'BrainMriVisualConfig',
    'BrainMriVisualProposalGenerator',
    'HeuristicCandidateSelector',
    'SamBoxRefiner',
    'build_pipeline_result',
    'candidate_to_summary',
    'compute_box_rerank_score',
    'summarize_pipeline_result',
    'compute_mask_metrics',
    'compute_text_rerank_score',
    'lesion_size_prior',
]
