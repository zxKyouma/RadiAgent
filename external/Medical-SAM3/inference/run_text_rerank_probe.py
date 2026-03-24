#!/usr/bin/env python3
"""
Probe Medical-SAM3 text grounding on a BraTS case by:
- sweeping prompts across a local slice window
- keeping top-k text candidates per prompt/slice
- reranking candidates with simple lesion-oriented priors
- refining the selected candidate with box prompting
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sam3_inference import SAM3Model
from radiant_pipeline import SegmentationPipeline, StudyContext
from radiant_pipeline.brain_mri import (
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

RESULTS_DIR = Path(__file__).parent / 'results'
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', default='BraTS-MEN-00307-000')
    parser.add_argument('--sequence', default='t1c')
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--confidence-threshold', type=float, default=0.1)
    parser.add_argument('--cluster-iou', type=float, default=0.3)
    parser.add_argument('--target-mask-frac', type=float, default=0.003)
    parser.add_argument('--size-prior-strength', type=float, default=0.3)
    parser.add_argument('--text-trust-score-threshold', type=float, default=0.75)
    parser.add_argument('--text-trust-min-slices', type=int, default=4)
    parser.add_argument('--text-trust-min-prompts', type=int, default=4)
    parser.add_argument('--text-trust-boost', type=float, default=0.35)
    parser.add_argument(
        '--prompts',
        default=','.join(DEFAULT_BRAIN_MRI_PROMPTS),
        help='Comma-separated prompt list.',
    )
    parser.add_argument(
        '--visual-thresholds',
        default='0.985,0.99,0.995',
        help='Comma-separated thresholds for fallback visual proposals.',
    )
    parser.add_argument('--visual-min-component-px', type=int, default=12)
    parser.add_argument('--visual-max-component-frac', type=float, default=0.03)
    parser.add_argument('--visual-max-candidates-per-slice', type=int, default=8)
    parser.add_argument('--disable-visual-fallback', action='store_true')
    parser.add_argument(
        '--checkpoint',
        default=str(REPO_ROOT / 'checkpoint.pt'),
        help='Path to checkpoint file.',
    )
    parser.add_argument(
        '--output-json',
        default=None,
        help='Optional path to save a JSON summary.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
    if not prompts:
        raise SystemExit('No prompts provided')
    visual_thresholds = [float(value.strip()) for value in args.visual_thresholds.split(',') if value.strip()]

    img_path = REPO_ROOT / 'data/assets/brats-men-v1/studies' / args.case_id / f'{args.sequence}.nii.gz'
    mask_path = REPO_ROOT / 'data/assets/brats-men-v1/masks' / args.case_id / 'dominant-lesion.nii.gz'
    if not img_path.exists():
        raise SystemExit(f'Image not found: {img_path}')
    if not mask_path.exists():
        raise SystemExit(f'Mask not found: {mask_path}')

    img = nib.load(str(img_path)).get_fdata()
    gt = nib.load(str(mask_path)).get_fdata()
    lesion_slices = np.where(gt.sum(axis=(0, 1)) > 0)[0]
    if len(lesion_slices) == 0:
        raise SystemExit('Ground-truth mask is empty')

    mid = int(lesion_slices[len(lesion_slices) // 2])
    slices = [
        s
        for s in range(mid - args.slice_radius, mid + args.slice_radius + 1)
        if 0 <= s < img.shape[2]
    ]

    print(f'Case: {args.case_id}')
    print(f'Sequence: {args.sequence}')
    print(f'Middle lesion slice: {mid}')
    print(f'Slice window: {slices}')
    print(f'Prompts: {prompts}')

    model = SAM3Model(
        confidence_threshold=args.confidence_threshold,
        device='cuda',
        checkpoint_path=args.checkpoint,
    )
    config = BrainMriTextConfig(
        prompts=prompts,
        top_k=args.top_k,
        cluster_iou=args.cluster_iou,
        target_mask_frac=args.target_mask_frac,
        size_prior_strength=args.size_prior_strength,
        text_trust_score_threshold=args.text_trust_score_threshold,
        text_trust_min_slices=args.text_trust_min_slices,
        text_trust_min_prompts=args.text_trust_min_prompts,
        text_trust_boost=args.text_trust_boost,
        verbose=True,
    )
    selector = HeuristicCandidateSelector()
    proposal_generators = [BrainMriTextProposalGenerator(model=model, config=config)]
    if not args.disable_visual_fallback:
        proposal_generators.append(
            BrainMriVisualProposalGenerator(
                config=BrainMriVisualConfig(
                    thresholds=visual_thresholds,
                    min_component_px=args.visual_min_component_px,
                    max_component_frac=args.visual_max_component_frac,
                    max_candidates_per_slice=args.visual_max_candidates_per_slice,
                )
            )
        )
    pipeline = SegmentationPipeline(
        finding_extractor=BrainMriFindingExtractor(),
        proposal_generators=proposal_generators,
        refiner=SamBoxRefiner(model=model, config=config),
        selector=selector,
    )

    context = StudyContext(
        case_id=args.case_id,
        modality='brain_mri',
        sequence=args.sequence,
        image_volume=img,
        report_text='single dominant intracranial mass-like lesion',
        metadata={
            'ground_truth_mask': gt,
            'slice_indices': slices,
            'mid_slice': mid,
            'finding_text': 'single dominant intracranial mass-like lesion',
            'support_status': 'supported',
        },
    )
    selection = pipeline.run(context)
    if selection is None or selector.raw_best is None or selector.text_reranked_best is None or selector.box_reranked_best is None:
        raise SystemExit('No text candidates produced in the slice window')

    raw_best = selector.raw_best
    text_reranked_best = selector.text_reranked_best
    box_reranked_best = selector.box_reranked_best

    print('\nRAW TOP-1')
    print(
        f" slice={raw_best.slice_idx} prompt={raw_best.prompt!r} score={raw_best.score:.4f} "
        f"bbox={raw_best.bbox_xyxy} area={raw_best.area_px} "
        f"dice={raw_best.text_metrics.dice:.4f}"
    )
    print('\nTEXT-RERANKED BEST')
    print(
        f" slice={text_reranked_best.slice_idx} prompt={text_reranked_best.prompt!r} raw_score={text_reranked_best.score:.4f} "
        f"rerank_score={float(text_reranked_best.rerank_score or 0.0):.4f} bbox={text_reranked_best.bbox_xyxy} "
        f"rounded_bbox={text_reranked_best.rounded_bbox_xyxy} area={text_reranked_best.area_px} "
        f"dice={text_reranked_best.text_metrics.dice:.4f}"
    )
    print('\nBOX-RERANKED BEST')
    refined_dice = f"{box_reranked_best.refined_metrics.dice:.4f}" if box_reranked_best.refined_metrics else 'n/a'
    refined_iou = f"{box_reranked_best.refined_metrics.iou:.4f}" if box_reranked_best.refined_metrics else 'n/a'
    print(
        f" slice={box_reranked_best.slice_idx} prompt={box_reranked_best.prompt!r} raw_score={box_reranked_best.score:.4f} "
        f"text_rerank={float(box_reranked_best.rerank_score or 0.0):.4f} box_rerank={float(box_reranked_best.box_rerank_score or 0.0):.4f} "
        f"bbox={box_reranked_best.bbox_xyxy} rounded_bbox={box_reranked_best.rounded_bbox_xyxy} "
        f"refined_bbox={box_reranked_best.refined_bbox_xyxy} area={box_reranked_best.area_px} "
        f"refined_area={box_reranked_best.refined_area_px} "
        f"text_dice={box_reranked_best.text_metrics.dice:.4f} refined_dice={refined_dice} refined_iou={refined_iou}"
    )

    summary = {
        'case_id': args.case_id,
        'sequence': args.sequence,
        'middle_slice': mid,
        'slice_window': slices,
        'prompts': prompts,
        'visual_thresholds': visual_thresholds,
        'visual_fallback_enabled': not args.disable_visual_fallback,
        'raw_best': candidate_to_summary(raw_best),
        'text_reranked_best': candidate_to_summary(text_reranked_best),
        'box_reranked_best': candidate_to_summary(box_reranked_best),
        'top_text_reranked': [candidate_to_summary(candidate) for candidate in selector.top_text_reranked],
        'top_box_reranked': [candidate_to_summary(candidate) for candidate in selector.top_box_reranked],
    }

    output_json = Path(args.output_json) if args.output_json else (
        RESULTS_DIR / f'text_rerank_probe_{args.case_id}_{args.sequence}.json'
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    print(f'\nSaved summary to: {output_json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
