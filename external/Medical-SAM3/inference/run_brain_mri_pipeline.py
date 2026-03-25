#!/usr/bin/env python3
"""Run the brain MRI segmentation pipeline with explicit routing and provisional 3D assembly."""

import argparse
import json
import sys
from pathlib import Path

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
)
from radiant_pipeline.brain_mri_runtime import BrainMriRouteConfig, build_pipeline_result, summarize_pipeline_result
from radiant_pipeline.brain_mri_scout import BrainMriScoutConfig, scout_slice_window

RESULTS_DIR = Path(__file__).parent / 'results'
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', default='BraTS-MEN-00307-000')
    parser.add_argument('--sequence', default='t1c')
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--window-policy', choices=['scout', 'gt', 'full'], default='scout')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--confidence-threshold', type=float, default=0.1)
    parser.add_argument('--cluster-iou', type=float, default=0.3)
    parser.add_argument('--target-mask-frac', type=float, default=0.003)
    parser.add_argument('--size-prior-strength', type=float, default=0.3)
    parser.add_argument('--text-trust-score-threshold', type=float, default=0.75)
    parser.add_argument('--text-trust-min-slices', type=int, default=4)
    parser.add_argument('--text-trust-min-prompts', type=int, default=4)
    parser.add_argument('--text-trust-boost', type=float, default=0.35)
    parser.add_argument('--min-final-box-score', type=float, default=0.5)
    parser.add_argument('--min-volume-voxels', type=int, default=20)
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
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--checkpoint',
        default=str(REPO_ROOT / 'checkpoint.pt'),
        help='Path to checkpoint file.',
    )
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--output-mask', default=None)
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

    image_nii = nib.load(str(img_path))
    img = image_nii.get_fdata()
    gt = nib.load(str(mask_path)).get_fdata() if mask_path.exists() else None

    model = SAM3Model(
        confidence_threshold=args.confidence_threshold,
        device='cuda',
        checkpoint_path=args.checkpoint,
    )

    text_config = BrainMriTextConfig(
        prompts=prompts,
        top_k=args.top_k,
        cluster_iou=args.cluster_iou,
        target_mask_frac=args.target_mask_frac,
        size_prior_strength=args.size_prior_strength,
        text_trust_score_threshold=args.text_trust_score_threshold,
        text_trust_min_slices=args.text_trust_min_slices,
        text_trust_min_prompts=args.text_trust_min_prompts,
        text_trust_boost=args.text_trust_boost,
        verbose=args.verbose,
    )
    proposal_generators = [BrainMriTextProposalGenerator(model=model, config=text_config)]
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
        refiner=SamBoxRefiner(model=model, config=text_config),
        selector=HeuristicCandidateSelector(),
    )
    route_config = BrainMriRouteConfig(
        min_box_rerank_score=args.min_final_box_score,
        min_volume_voxels=args.min_volume_voxels,
    )
    base_context = StudyContext(
        case_id=args.case_id,
        modality='brain_mri',
        sequence=args.sequence,
        image_volume=img,
        report_text='single dominant intracranial mass-like lesion',
        metadata={
            'ground_truth_mask': gt,
            'finding_text': 'single dominant intracranial mass-like lesion',
            'support_status': 'supported',
        },
    )

    scout_result = None
    if args.window_policy == 'gt' and gt is not None and gt.sum() > 0:
        lesion_slices = np.where(gt.sum(axis=(0, 1)) > 0)[0]
        mid = int(lesion_slices[len(lesion_slices) // 2])
        slice_indices = [
            s for s in range(mid - args.slice_radius, mid + args.slice_radius + 1)
            if 0 <= s < img.shape[2]
        ]
    elif args.window_policy == 'full':
        mid = int(img.shape[2] // 2)
        slice_indices = list(range(img.shape[2]))
    else:
        scout_result = scout_slice_window(
            base_context,
            pipeline,
            route_config,
            model,
            BrainMriScoutConfig(
                prompts=('intracranial mass', 'enhancing mass'),
                slice_radius=args.slice_radius,
                coarse_stride=8,
                shortlist_size=3,
                visual_threshold=0.99,
                min_component_px=args.visual_min_component_px,
                max_component_frac=args.visual_max_component_frac,
            ),
        )
        mid = int(scout_result.best_center_idx)
        slice_indices = list(scout_result.slice_indices)

    print(f'Case: {args.case_id}')
    print(f'Sequence: {args.sequence}')
    print(f'Window policy: {args.window_policy}')
    if scout_result is not None:
        print(f'Scout best center: {scout_result.best_center_idx}')
        print(f'Scout route: {scout_result.best_result.route}')
        print(f'Scout shortlist: {scout_result.shortlisted_centers}')
    print(f'Slice window: {slice_indices}')
    print(f'Prompts: {prompts}')
    context = StudyContext(
        case_id=args.case_id,
        modality='brain_mri',
        sequence=args.sequence,
        image_volume=img,
        report_text='single dominant intracranial mass-like lesion',
        metadata={
            'ground_truth_mask': gt,
            'slice_indices': slice_indices,
            'mid_slice': mid,
            'finding_text': 'single dominant intracranial mass-like lesion',
            'support_status': 'supported',
        },
    )

    artifacts = pipeline.run_detailed(context)
    result = build_pipeline_result(context, artifacts, route_config=route_config)
    summary = summarize_pipeline_result(result)
    summary.update({
        'case_id': args.case_id,
        'sequence': args.sequence,
        'window_policy': args.window_policy,
        'scout_best_slice': None if scout_result is None else scout_result.best_center_idx,
        'slice_window': slice_indices,
        'prompts': prompts,
        'visual_thresholds': visual_thresholds,
    })

    print(f"Route: {result.route}")
    if result.selection is not None:
        candidate = result.selection.candidate
        refined_dice = 'n/a' if candidate.refined_metrics is None else f"{candidate.refined_metrics.dice:.4f}"
        refined_iou = 'n/a' if candidate.refined_metrics is None else f"{candidate.refined_metrics.iou:.4f}"
        print(
            f"Selected slice={candidate.slice_idx} prompt={candidate.prompt!r} generator={candidate.metadata.get('generator')} "
            f"score={float(candidate.box_rerank_score or 0.0):.4f} refined_dice={refined_dice} refined_iou={refined_iou}"
        )
    if result.volume_assembly is not None:
        print(
            f"Volume voxels={result.volume_assembly.voxel_count} slices={result.volume_assembly.slice_indices} "
            f"bbox_xyzxyz={result.volume_assembly.bbox_xyzxyz}"
        )
    if result.warnings:
        print('Warnings:')
        for warning in result.warnings:
            print(f' - {warning}')

    output_json = Path(args.output_json) if args.output_json else RESULTS_DIR / f'brain_mri_pipeline_{args.case_id}_{args.sequence}.json'
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_json}')

    if result.volume_assembly is not None:
        output_mask = Path(args.output_mask) if args.output_mask else RESULTS_DIR / f'brain_mri_pipeline_{args.case_id}_{args.sequence}_mask.nii.gz'
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        mask_nii = nib.Nifti1Image(result.volume_assembly.mask_volume.astype(np.uint8), image_nii.affine, image_nii.header)
        nib.save(mask_nii, str(output_mask))
        print(f'Saved provisional 3D mask to: {output_mask}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
