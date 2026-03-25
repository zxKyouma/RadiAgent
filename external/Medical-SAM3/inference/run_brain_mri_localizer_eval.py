#!/usr/bin/env python3
"""Evaluate the heuristic brain MRI localizer on the BraTS 5-case manifest."""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from radiant_pipeline.brain_mri import (
    BrainMriFindingExtractor,
    BrainMriHeuristicLocalizer,
    BrainMriHeuristicLocalizerConfig,
)
from radiant_pipeline.biomedclip_backend import BiomedClipBackendConfig, BiomedClipRetrievalBackend
from radiant_pipeline.medsiglip_backend import MedSiglipBackendConfig, MedSiglipRetrievalBackend
from radiant_pipeline.brain_mri_retrieval import BrainMriRetrievalLocalizer, BrainMriRetrievalLocalizerConfig
from radiant_pipeline.types import StudyContext

REPO_ROOT = Path(__file__).resolve().parents[3]
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
    parser.add_argument('--localizer', choices=['heuristic', 'biomedclip', 'medsiglip'], default='heuristic')
    parser.add_argument('--cases', default=','.join(DEFAULT_CASES))
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--coarse-stride', type=int, default=4)
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--thresholds', default='0.985,0.99,0.995')
    parser.add_argument('--min-component-px', type=int, default=12)
    parser.add_argument('--max-component-frac', type=float, default=0.03)
    parser.add_argument('--min-center-separation', type=int, default=3)
    parser.add_argument('--laterality-bonus', type=float, default=0.1)
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


def main() -> int:
    args = parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
    cases = [case.strip() for case in args.cases.split(',') if case.strip()]

    if args.localizer == 'heuristic':
        localizer = BrainMriHeuristicLocalizer(
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
    elif args.localizer == 'biomedclip':
        localizer = BrainMriRetrievalLocalizer(
            backend=BiomedClipRetrievalBackend(BiomedClipBackendConfig(device='cuda')),
            config=BrainMriRetrievalLocalizerConfig(
                slab_depth=max(3, 2 * args.slice_radius + 1),
                slab_stride=max(1, args.coarse_stride // 2),
                shortlist_size=args.shortlist_size,
                min_center_separation=args.min_center_separation,
            ),
        )
    else:
        localizer = BrainMriRetrievalLocalizer(
            backend=MedSiglipRetrievalBackend(MedSiglipBackendConfig(device='cuda')),
            config=BrainMriRetrievalLocalizerConfig(
                slab_depth=max(3, 2 * args.slice_radius + 1),
                slab_stride=max(1, args.coarse_stride // 2),
                shortlist_size=args.shortlist_size,
                min_center_separation=args.min_center_separation,
            ),
        )
    extractor = BrainMriFindingExtractor()

    results = []
    total_start = time.perf_counter()
    hit_count = 0

    for case_id in cases:
        img, gt, manifest = load_case(case_id, args.sequence)
        lesion_slices = np.where(gt.sum(axis=(0, 1)) > 0)[0]
        lesion_slice_set = set(int(x) for x in lesion_slices.tolist())
        report_text = str(manifest.get('dominantFindingText', 'single dominant intracranial mass-like lesion'))
        context = StudyContext(
            case_id=case_id,
            modality='brain_mri',
            sequence=args.sequence,
            image_volume=img,
            report_text=report_text,
            metadata={
                'finding_text': manifest.get('dominantFindingText', 'single dominant intracranial mass-like lesion'),
                'support_status': 'supported',
            },
        )

        start = time.perf_counter()
        target = extractor.extract(context)
        hypotheses = localizer.localize(context, target)
        elapsed = time.perf_counter() - start

        hit = False
        best_overlap = 0
        best_center = None
        hypothesis_summaries = []
        for hypothesis in hypotheses:
            overlap = len(set(hypothesis.slice_indices) & lesion_slice_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_center = hypothesis.center_slice
            hit = hit or overlap > 0
            hypothesis_summaries.append({
                'center_slice': hypothesis.center_slice,
                'score': hypothesis.score,
                'slice_indices': list(hypothesis.slice_indices),
                'overlap_slices': overlap,
            })

        if hit:
            hit_count += 1
        results.append({
            'case_id': case_id,
            'hit_at_k': hit,
            'best_overlap_slices': best_overlap,
            'best_center_slice': best_center,
            'lesion_slice_range': [int(lesion_slices[0]), int(lesion_slices[-1])],
            'lesion_slice_count': int(len(lesion_slices)),
            'elapsed_sec': elapsed,
            'hypotheses': hypothesis_summaries,
        })

    total_elapsed = time.perf_counter() - total_start
    summary = {
        'case_count': len(results),
        'hit_at_k': hit_count,
        'hit_rate': hit_count / max(len(results), 1),
        'avg_elapsed_sec': sum(item['elapsed_sec'] for item in results) / max(len(results), 1),
        'total_elapsed_sec': total_elapsed,
        'localizer': args.localizer,
        'config': {
            'slice_radius': args.slice_radius,
            'coarse_stride': args.coarse_stride,
            'shortlist_size': args.shortlist_size,
            'thresholds': thresholds,
            'min_component_px': args.min_component_px,
            'max_component_frac': args.max_component_frac,
            'min_center_separation': args.min_center_separation,
            'laterality_bonus': args.laterality_bonus,
        },
        'results': results,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
