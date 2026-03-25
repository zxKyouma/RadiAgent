#!/usr/bin/env python3
"""Evaluate the full brain MRI pipeline on the local BraTS manifest."""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sam3_inference import SAM3Model
from radiant_pipeline import SegmentationPipeline, StudyContext
from radiant_pipeline.brain_mri import (
    DEFAULT_BRAIN_MRI_PROMPTS,
    BrainMriFindingExtractor,
    BrainMriHeuristicLocalizer,
    BrainMriHeuristicLocalizerConfig,
    BrainMriTextConfig,
    BrainMriTextProposalGenerator,
    BrainMriVisualConfig,
    BrainMriVisualProposalGenerator,
    HeuristicCandidateSelector,
    SamBoxRefiner,
)
from radiant_pipeline.biomedclip_backend import BiomedClipBackendConfig, BiomedClipRetrievalBackend
from radiant_pipeline.brain_mri_retrieval import BrainMriRetrievalLocalizer, BrainMriRetrievalLocalizerConfig
from radiant_pipeline.brain_mri_runtime import BrainMriRouteConfig, build_pipeline_result
from radiant_pipeline.medsiglip_backend import MedSiglipBackendConfig, MedSiglipRetrievalBackend

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / 'results'
DEFAULT_CASES = [
    'BraTS-MEN-00307-000',
    'BraTS-MEN-00488-000',
    'BraTS-MEN-01205-000',
    'BraTS-MEN-01280-000',
    'BraTS-MEN-01429-000',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', default='t1c')
    parser.add_argument('--cases', default=','.join(DEFAULT_CASES))
    parser.add_argument('--localizer', choices=['heuristic', 'biomedclip', 'medsiglip'], default='medsiglip')
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--coarse-stride', type=int, default=4)
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--thresholds', default='0.985,0.99,0.995')
    parser.add_argument('--min-component-px', type=int, default=12)
    parser.add_argument('--max-component-frac', type=float, default=0.03)
    parser.add_argument('--min-center-separation', type=int, default=3)
    parser.add_argument('--laterality-bonus', type=float, default=0.1)
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
    parser.add_argument('--success-dice-threshold', type=float, default=0.5)
    parser.add_argument('--disable-visual-fallback', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'checkpoint.pt'))
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def load_case(case_id: str, sequence: str) -> tuple[np.ndarray, np.ndarray, dict]:
    img_path = REPO_ROOT / 'data/assets/brats-men-v1/studies' / case_id / f'{sequence}.nii.gz'
    mask_path = REPO_ROOT / 'data/assets/brats-men-v1/masks' / case_id / 'dominant-lesion.nii.gz'
    manifest_path = REPO_ROOT / 'data/manifests/brats-men-v1/cases' / f'{case_id}.case.json'
    img = nib.load(str(img_path)).get_fdata()
    gt = nib.load(str(mask_path)).get_fdata()
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return img, gt, manifest


def build_localizer(args: argparse.Namespace, thresholds: list[float]):
    if args.localizer == 'heuristic':
        return BrainMriHeuristicLocalizer(
            BrainMriHeuristicLocalizerConfig(
                slice_radius=args.slice_radius,
                coarse_stride=args.coarse_stride,
                shortlist_size=args.shortlist_size,
                thresholds=thresholds,
                min_component_px=args.min_component_px,
                max_component_frac=args.max_component_frac,
                min_center_separation=args.min_center_separation,
                laterality_bonus=args.laterality_bonus,
            )
        )
    if args.localizer == 'biomedclip':
        backend = BiomedClipRetrievalBackend(BiomedClipBackendConfig(device='cuda'))
    else:
        backend = MedSiglipRetrievalBackend(MedSiglipBackendConfig(device='cuda'))
    return BrainMriRetrievalLocalizer(
        backend=backend,
        config=BrainMriRetrievalLocalizerConfig(
            slab_depth=max(3, 2 * args.slice_radius + 1),
            slab_stride=max(1, args.coarse_stride // 2),
            shortlist_size=args.shortlist_size,
            min_center_separation=args.min_center_separation,
        ),
    )


def main() -> int:
    args = parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
    cases = [case.strip() for case in args.cases.split(',') if case.strip()]

    model = SAM3Model(
        confidence_threshold=args.confidence_threshold,
        device='cuda',
        checkpoint_path=args.checkpoint,
    )
    text_config = BrainMriTextConfig(
        prompts=list(DEFAULT_BRAIN_MRI_PROMPTS),
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
    box_proposers = [BrainMriTextProposalGenerator(model=model, config=text_config)]
    if not args.disable_visual_fallback:
        box_proposers.append(
            BrainMriVisualProposalGenerator(
                config=BrainMriVisualConfig(
                    thresholds=thresholds,
                    min_component_px=args.min_component_px,
                    max_component_frac=args.max_component_frac,
                )
            )
        )

    pipeline = SegmentationPipeline(
        finding_extractor=BrainMriFindingExtractor(),
        localizer=build_localizer(args, thresholds),
        box_proposers=box_proposers,
        refiner=SamBoxRefiner(model=model, config=text_config),
        selector=HeuristicCandidateSelector(),
    )
    route_config = BrainMriRouteConfig(
        min_box_rerank_score=args.min_final_box_score,
        min_volume_voxels=args.min_volume_voxels,
    )

    results = []
    success_count = 0
    route_counts: dict[str, int] = {}
    total_start = time.perf_counter()

    for case_id in cases:
        img, gt, manifest = load_case(case_id, args.sequence)
        lesion_slices = np.where(gt.sum(axis=(0, 1)) > 0)[0]
        report_text = str(manifest.get('dominantFindingText', 'single dominant intracranial mass-like lesion'))
        context = StudyContext(
            case_id=case_id,
            modality='brain_mri',
            sequence=args.sequence,
            image_volume=img,
            report_text=report_text,
            metadata={
                'ground_truth_mask': gt,
                'finding_text': manifest.get('dominantFindingText', report_text),
                'support_status': 'supported',
            },
        )

        start = time.perf_counter()
        artifacts = pipeline.run_detailed(context)
        result = build_pipeline_result(context, artifacts, route_config=route_config)
        elapsed = time.perf_counter() - start
        route_counts[result.route] = route_counts.get(result.route, 0) + 1

        selection = result.selection.candidate if result.selection is not None else None
        refined_dice = None
        refined_iou = None
        localizer_hit = False
        localizer_centers = [int(h.center_slice) for h in artifacts.localizer_hypotheses]
        localizer_windows = [list(h.slice_indices) for h in artifacts.localizer_hypotheses]
        lesion_slice_set = set(int(x) for x in lesion_slices.tolist())
        for hypothesis in artifacts.localizer_hypotheses:
            if set(hypothesis.slice_indices) & lesion_slice_set:
                localizer_hit = True
                break
        if selection is not None and selection.refined_metrics is not None:
            refined_dice = float(selection.refined_metrics.dice)
            refined_iou = float(selection.refined_metrics.iou)
            if refined_dice >= args.success_dice_threshold:
                success_count += 1

        results.append({
            'case_id': case_id,
            'route': result.route,
            'elapsed_sec': elapsed,
            'localizer_hit_at_k': localizer_hit,
            'localizer_centers': localizer_centers,
            'localizer_windows': localizer_windows,
            'lesion_slice_range': [int(lesion_slices[0]), int(lesion_slices[-1])],
            'selection': None if selection is None else {
                'slice_idx': int(selection.slice_idx),
                'prompt': selection.prompt,
                'generator': selection.metadata.get('generator'),
                'box_rerank_score': float(selection.box_rerank_score or 0.0),
                'refined_dice': refined_dice,
                'refined_iou': refined_iou,
            },
            'warnings': list(result.warnings),
        })

    total_elapsed = time.perf_counter() - total_start
    summary = {
        'localizer': args.localizer,
        'case_count': len(results),
        'success_count': success_count,
        'success_rate': success_count / max(len(results), 1),
        'route_counts': route_counts,
        'avg_elapsed_sec': sum(item['elapsed_sec'] for item in results) / max(len(results), 1),
        'total_elapsed_sec': total_elapsed,
        'success_dice_threshold': args.success_dice_threshold,
        'results': results,
    }

    print(json.dumps(summary, indent=2))
    output_path = Path(args.output_json) if args.output_json else RESULTS_DIR / f'brain_mri_end_to_end_eval_{args.localizer}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
