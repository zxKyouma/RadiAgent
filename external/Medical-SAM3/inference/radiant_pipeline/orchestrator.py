from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from .types import FindingTarget, PipelineRunArtifacts, ProposalCandidate, SegmentationSelection, StudyContext


class FindingExtractor(Protocol):
    """Extracts normalized targets from report text or study metadata."""

    def extract(self, context: StudyContext) -> FindingTarget:
        ...


class ProposalGenerator(Protocol):
    """Produces candidate boxes or masks for a target on a study."""

    def generate(self, context: StudyContext, target: FindingTarget) -> List[ProposalCandidate]:
        ...


class SegmentationRefiner(Protocol):
    """Refines a candidate proposal into a sharper segmentation result."""

    def refine(
        self,
        context: StudyContext,
        target: FindingTarget,
        candidates: Sequence[ProposalCandidate],
    ) -> List[ProposalCandidate]:
        ...


class CandidateSelector(Protocol):
    """Chooses a final candidate from a refined candidate set."""

    def select(
        self,
        context: StudyContext,
        target: FindingTarget,
        candidates: Sequence[ProposalCandidate],
    ) -> Optional[SegmentationSelection]:
        ...


@dataclass(slots=True)
class SegmentationPipeline:
    """Generic orchestration shell for report-guided segmentation."""

    finding_extractor: FindingExtractor
    proposal_generators: Sequence[ProposalGenerator]
    refiner: SegmentationRefiner
    selector: CandidateSelector

    def run_detailed(self, context: StudyContext) -> PipelineRunArtifacts:
        target = self.finding_extractor.extract(context)
        all_candidates: List[ProposalCandidate] = []
        for generator in self.proposal_generators:
            all_candidates.extend(generator.generate(context, target))
        if not all_candidates:
            return PipelineRunArtifacts(target=target, candidates=[], selection=None)
        refined = self.refiner.refine(context, target, all_candidates)
        if not refined:
            return PipelineRunArtifacts(target=target, candidates=[], selection=None)
        selection = self.selector.select(context, target, refined)
        return PipelineRunArtifacts(target=target, candidates=list(refined), selection=selection)

    def run(self, context: StudyContext) -> Optional[SegmentationSelection]:
        return self.run_detailed(context).selection
