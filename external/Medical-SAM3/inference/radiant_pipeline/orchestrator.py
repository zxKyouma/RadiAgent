from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from .types import (
    BoxProposal,
    LocalizerHypothesis,
    PipelineRunArtifacts,
    ProposalCandidate,
    SegmentationSelection,
    StructuredTarget,
    StudyContext,
)


class FindingExtractor(Protocol):
    """Extracts normalized targets from report text or study metadata."""

    def extract(self, context: StudyContext) -> StructuredTarget:
        ...


class Localizer(Protocol):
    """Shortlists top-k study windows before any heavy refinement."""

    def localize(self, context: StudyContext, target: StructuredTarget) -> List[LocalizerHypothesis]:
        ...


class BoxProposer(Protocol):
    """Produces coarse per-slice box proposals inside a localization window."""

    def propose(
        self,
        context: StudyContext,
        target: StructuredTarget,
        hypothesis: LocalizerHypothesis,
    ) -> List[BoxProposal]:
        ...


class SegmentationRefiner(Protocol):
    """Refines coarse proposals into sharper segmentation results."""

    def refine(
        self,
        context: StudyContext,
        target: StructuredTarget,
        proposals: Sequence[BoxProposal],
    ) -> List[ProposalCandidate]:
        ...


class CandidateSelector(Protocol):
    """Chooses a final candidate from a refined candidate set."""

    def select(
        self,
        context: StudyContext,
        target: StructuredTarget,
        candidates: Sequence[ProposalCandidate],
    ) -> Optional[SegmentationSelection]:
        ...


@dataclass(slots=True)
class SegmentationPipeline:
    """Generic orchestration shell for report-guided segmentation."""

    finding_extractor: FindingExtractor
    localizer: Localizer
    box_proposers: Sequence[BoxProposer]
    refiner: SegmentationRefiner
    selector: CandidateSelector

    def run_detailed(self, context: StudyContext) -> PipelineRunArtifacts:
        target = self.finding_extractor.extract(context)
        if target.should_abstain:
            return PipelineRunArtifacts(target=target)

        localizer_hypotheses = self.localizer.localize(context, target)
        if not localizer_hypotheses:
            return PipelineRunArtifacts(target=target)

        all_box_proposals: List[BoxProposal] = []
        for hypothesis in localizer_hypotheses:
            for proposer in self.box_proposers:
                all_box_proposals.extend(proposer.propose(context, target, hypothesis))
        if not all_box_proposals:
            return PipelineRunArtifacts(target=target, localizer_hypotheses=list(localizer_hypotheses))

        refined = self.refiner.refine(context, target, all_box_proposals)
        if not refined:
            return PipelineRunArtifacts(
                target=target,
                localizer_hypotheses=list(localizer_hypotheses),
                box_proposals=list(all_box_proposals),
            )
        selection = self.selector.select(context, target, refined)
        return PipelineRunArtifacts(
            target=target,
            localizer_hypotheses=list(localizer_hypotheses),
            box_proposals=list(all_box_proposals),
            candidates=list(refined),
            selection=selection,
        )

    def run(self, context: StudyContext) -> Optional[SegmentationSelection]:
        return self.run_detailed(context).selection
