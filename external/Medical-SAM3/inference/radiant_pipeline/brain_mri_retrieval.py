from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence

import numpy as np

from .orchestrator import Localizer
from .types import LocalizerHypothesis, StructuredTarget, StudyContext


@dataclass(slots=True)
class RetrievalSlab:
    """A short overlapping volumetric slab used as a retrieval unit."""

    start_slice: int
    end_slice: int
    center_slice: int
    slice_indices: List[int]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class BrainMriRetrievalLocalizerConfig:
    """Configuration for slab-based retrieval localization."""

    slab_depth: int = 5
    slab_stride: int = 2
    shortlist_size: int = 3
    min_center_separation: int = 3
    expand_radius: int = 2
    merge_gap: int = 2
    laterality_mode: str = 'none'


class BrainMriRetrievalBackend(Protocol):
    """Backend that scores report text against candidate MRI slabs."""

    backend_name: str

    def score_slabs(
        self,
        context: StudyContext,
        target: StructuredTarget,
        slabs: Sequence[RetrievalSlab],
    ) -> List[float]:
        ...


class BrainMriRetrievalLocalizer(Localizer):
    """Window retriever that ranks overlapping slabs and returns top-k windows."""

    def __init__(self, backend: BrainMriRetrievalBackend, config: BrainMriRetrievalLocalizerConfig):
        self.backend = backend
        self.config = config

    def localize(self, context: StudyContext, target: StructuredTarget) -> List[LocalizerHypothesis]:
        slabs = generate_retrieval_slabs(
            depth=int(context.image_volume.shape[2]),
            slab_depth=self.config.slab_depth,
            slab_stride=self.config.slab_stride,
        )
        if not slabs:
            return []

        scores = self.backend.score_slabs(context, target, slabs)
        if len(scores) != len(slabs):
            raise ValueError(f'Backend returned {len(scores)} scores for {len(slabs)} slabs')

        ranked = sorted(zip(slabs, scores), key=lambda item: float(item[1]), reverse=True)
        selected: List[tuple[RetrievalSlab, float]] = []
        for slab, score in ranked:
            if any(abs(slab.center_slice - existing.center_slice) < self.config.min_center_separation for existing, _ in selected):
                continue
            selected.append((slab, float(score)))
            if len(selected) >= self.config.shortlist_size:
                break

        hypotheses: List[LocalizerHypothesis] = []
        depth = int(context.image_volume.shape[2])
        for idx, (slab, score) in enumerate(selected, start=1):
            expanded = self._expand_slice_indices(depth, slab.slice_indices)
            hypotheses.append(
                LocalizerHypothesis(
                    source='retrieval_localizer',
                    score=float(score),
                    center_slice=slab.center_slice,
                    slice_indices=expanded,
                    metadata={
                        'backend_name': getattr(self.backend, 'backend_name', type(self.backend).__name__),
                        'start_slice': slab.start_slice,
                        'end_slice': slab.end_slice,
                        'why_selected': f'top_{idx}_slab_score',
                        **dict(slab.metadata),
                    },
                )
            )
        return self._merge_hypotheses(hypotheses)

    def _expand_slice_indices(self, depth: int, slice_indices: Sequence[int]) -> List[int]:
        start = max(0, int(slice_indices[0]) - self.config.expand_radius)
        end = min(depth - 1, int(slice_indices[-1]) + self.config.expand_radius)
        return list(range(start, end + 1))

    def _merge_hypotheses(self, hypotheses: List[LocalizerHypothesis]) -> List[LocalizerHypothesis]:
        if not hypotheses:
            return []
        ordered = sorted(hypotheses, key=lambda h: h.slice_indices[0])
        merged: List[LocalizerHypothesis] = []
        for hypothesis in ordered:
            if not merged:
                merged.append(hypothesis)
                continue
            previous = merged[-1]
            if hypothesis.slice_indices[0] <= previous.slice_indices[-1] + self.config.merge_gap:
                combined_slices = sorted(set(previous.slice_indices) | set(hypothesis.slice_indices))
                better = hypothesis if hypothesis.score > previous.score else previous
                merged[-1] = LocalizerHypothesis(
                    source=better.source,
                    score=max(previous.score, hypothesis.score),
                    center_slice=better.center_slice,
                    slice_indices=combined_slices,
                    metadata={**previous.metadata, **hypothesis.metadata, 'merged_window': True},
                )
            else:
                merged.append(hypothesis)
        return merged


def generate_retrieval_slabs(depth: int, slab_depth: int, slab_stride: int) -> List[RetrievalSlab]:
    """Generate overlapping slab windows across the full study."""

    if depth <= 0:
        return []
    slab_depth = max(1, int(slab_depth))
    slab_stride = max(1, int(slab_stride))
    slab_depth = min(slab_depth, depth)

    starts = list(range(0, max(depth - slab_depth + 1, 1), slab_stride))
    final_start = max(depth - slab_depth, 0)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    slabs: List[RetrievalSlab] = []
    for start in starts:
        end = min(start + slab_depth - 1, depth - 1)
        slice_indices = list(range(start, end + 1))
        center = slice_indices[len(slice_indices) // 2]
        slabs.append(
            RetrievalSlab(
                start_slice=start,
                end_slice=end,
                center_slice=center,
                slice_indices=slice_indices,
            )
        )
    return slabs


def build_slab_rgb_preview(image_volume: np.ndarray, slab: RetrievalSlab) -> np.ndarray:
    """Build a simple 2.5D RGB preview from the slab center neighborhood."""

    center_pos = len(slab.slice_indices) // 2
    center_idx = slab.slice_indices[center_pos]
    prev_idx = slab.slice_indices[max(center_pos - 1, 0)]
    next_idx = slab.slice_indices[min(center_pos + 1, len(slab.slice_indices) - 1)]

    channels = []
    for slice_idx in (prev_idx, center_idx, next_idx):
        slice_img = np.asarray(image_volume[:, :, slice_idx], dtype=np.float32)
        lo, hi = np.percentile(slice_img, [0.5, 99.5])
        normalized = np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        channels.append((normalized * 255).astype(np.uint8))
    return np.stack(channels, axis=-1)
