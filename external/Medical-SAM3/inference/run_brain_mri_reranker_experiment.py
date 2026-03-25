#!/usr/bin/env python3
"""Train and evaluate a lightweight case-wise reranker on brain MRI candidates."""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

NUMERIC_FEATURES = [
    'score',
    'rerank_score',
    'box_rerank_score',
    'refined_score',
    'area_frac',
    'refined_area_frac',
    'size_prior',
    'refined_size_prior',
    'bbox_fill_ratio',
    'refined_bbox_fill_ratio',
    'compactness_prior',
    'refined_bbox_stability_iou',
    'localizer_score',
    'cluster_slice_count',
    'cluster_prompt_count',
    'text_trust_bonus',
]
CAT_FEATURES = [
    'generator',
    'prompt',
    'localizer_source',
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
    parser.add_argument('--success-dice-threshold', type=float, default=0.5)
    parser.add_argument('--disable-visual-fallback', action='store_true')
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'checkpoint.pt'))
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def load_case(case_id: str, sequence: str):
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


def build_pipeline(args: argparse.Namespace) -> SegmentationPipeline:
    thresholds = [float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
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
        verbose=False,
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
    return SegmentationPipeline(
        finding_extractor=BrainMriFindingExtractor(),
        localizer=build_localizer(args, thresholds),
        box_proposers=box_proposers,
        refiner=SamBoxRefiner(model=model, config=text_config),
        selector=HeuristicCandidateSelector(),
    )


def candidate_row(case_id, candidate, selected_candidate, threshold):
    refined = candidate.refined_metrics
    dice = None if refined is None else float(refined.dice)
    iou = None if refined is None else float(refined.iou)
    return {
        'case_id': case_id,
        'slice_idx': int(candidate.slice_idx),
        'prompt': candidate.prompt,
        'generator': str(candidate.metadata.get('generator', 'unknown')),
        'localizer_source': str(candidate.metadata.get('localizer_source', 'unknown')),
        'score': float(candidate.score),
        'rerank_score': float(candidate.rerank_score or 0.0),
        'box_rerank_score': float(candidate.box_rerank_score or 0.0),
        'refined_score': float(candidate.refined_score or 0.0),
        'area_frac': float(candidate.area_frac or 0.0),
        'refined_area_frac': float(candidate.refined_area_frac or 0.0),
        'size_prior': float(candidate.size_prior or 0.0),
        'refined_size_prior': float(candidate.refined_size_prior or 0.0),
        'bbox_fill_ratio': float(candidate.bbox_fill_ratio or 0.0),
        'refined_bbox_fill_ratio': float(candidate.refined_bbox_fill_ratio or 0.0),
        'compactness_prior': float(candidate.compactness_prior or 0.0),
        'refined_bbox_stability_iou': float(candidate.refined_bbox_stability_iou or 0.0),
        'localizer_score': float(candidate.metadata.get('localizer_score', 0.0)),
        'cluster_slice_count': int(candidate.cluster_slice_count),
        'cluster_prompt_count': int(candidate.cluster_prompt_count),
        'text_trust_bonus': float(candidate.text_trust_bonus or 0.0),
        'refined_dice': dice,
        'refined_iou': iou,
        'is_selected_baseline': selected_candidate is candidate,
        'is_positive': False if dice is None else dice >= threshold,
    }


def make_model() -> Pipeline:
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    pre = ColumnTransformer([
        ('num', numeric, NUMERIC_FEATURES),
        ('cat', categorical, CAT_FEATURES),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    return Pipeline([
        ('pre', pre),
        ('clf', clf),
    ])


def main() -> int:
    args = parse_args()
    cases = [case.strip() for case in args.cases.split(',') if case.strip()]
    pipeline = build_pipeline(args)

    rows = []
    case_meta = {}
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
        elapsed = time.perf_counter() - start
        selected = artifacts.selection.candidate if artifacts.selection is not None else None
        case_meta[case_id] = {
            'elapsed_sec': elapsed,
            'lesion_slice_range': [int(lesion_slices[0]), int(lesion_slices[-1])],
            'localizer_centers': [int(h.center_slice) for h in artifacts.localizer_hypotheses],
            'baseline_selected_dice': None if selected is None or selected.refined_metrics is None else float(selected.refined_metrics.dice),
        }
        for candidate in artifacts.candidates:
            rows.append(candidate_row(case_id, candidate, selected, args.success_dice_threshold))

    df = pd.DataFrame(rows)
    per_case = []
    baseline_success = 0
    reranker_success = 0

    for holdout in cases:
        train_df = df[df['case_id'] != holdout].copy()
        test_df = df[df['case_id'] == holdout].copy()
        if train_df.empty or test_df.empty:
            continue
        model = make_model()
        model.fit(train_df[NUMERIC_FEATURES + CAT_FEATURES], train_df['is_positive'].astype(int))
        test_df = test_df.copy()
        test_df['reranker_score'] = model.predict_proba(test_df[NUMERIC_FEATURES + CAT_FEATURES])[:, 1]
        chosen = test_df.sort_values(['reranker_score', 'refined_dice'], ascending=False).iloc[0]
        oracle = test_df.sort_values(['refined_dice', 'refined_score'], ascending=False).iloc[0]
        baseline = test_df[test_df['is_selected_baseline']].iloc[0] if test_df['is_selected_baseline'].any() else None
        baseline_dice = None if baseline is None else (None if pd.isna(baseline['refined_dice']) else float(baseline['refined_dice']))
        chosen_dice = None if pd.isna(chosen['refined_dice']) else float(chosen['refined_dice'])
        oracle_dice = None if pd.isna(oracle['refined_dice']) else float(oracle['refined_dice'])
        if baseline_dice is not None and baseline_dice >= args.success_dice_threshold:
            baseline_success += 1
        if chosen_dice is not None and chosen_dice >= args.success_dice_threshold:
            reranker_success += 1
        per_case.append({
            'case_id': holdout,
            'baseline_dice': baseline_dice,
            'reranker_dice': chosen_dice,
            'oracle_dice': oracle_dice,
            'baseline_prompt': None if baseline is None else baseline['prompt'],
            'baseline_generator': None if baseline is None else baseline['generator'],
            'reranker_prompt': chosen['prompt'],
            'reranker_generator': chosen['generator'],
            'oracle_prompt': oracle['prompt'],
            'oracle_generator': oracle['generator'],
            'localizer_centers': case_meta[holdout]['localizer_centers'],
            'lesion_slice_range': case_meta[holdout]['lesion_slice_range'],
        })

    summary = {
        'localizer': args.localizer,
        'success_dice_threshold': args.success_dice_threshold,
        'case_count': len(per_case),
        'baseline_success_count': baseline_success,
        'reranker_success_count': reranker_success,
        'baseline_success_rate': baseline_success / max(len(per_case), 1),
        'reranker_success_rate': reranker_success / max(len(per_case), 1),
        'total_elapsed_sec': time.perf_counter() - total_start,
        'cases': per_case,
    }

    output_path = Path(args.output_json) if args.output_json else RESULTS_DIR / f'brain_mri_reranker_experiment_{args.localizer}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
