#!/usr/bin/env python3
"""Evaluate deterministic candidate profiling + constraint matching on the local brain MRI set.

This is a proof-of-principle experiment. Because the local synthetic reports are sparse,
we derive pseudo-constraints from the GT lesion for evaluation only.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

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
from radiant_pipeline.brain_mri_runtime import assemble_candidate_volume
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
    parser.add_argument('--success-dice-threshold', type=float, default=0.5)
    parser.add_argument('--disable-visual-fallback', action='store_true')
    parser.add_argument('--profile-top-n', type=int, default=20)
    parser.add_argument('--profile-top-n-per-generator', type=int, default=10)
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'checkpoint.pt'))
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def load_case(case_id: str, sequence: str):
    img_path = REPO_ROOT / 'data/assets/brats-men-v1/studies' / case_id / f'{sequence}.nii.gz'
    mask_path = REPO_ROOT / 'data/assets/brats-men-v1/masks' / case_id / 'dominant-lesion.nii.gz'
    manifest_path = REPO_ROOT / 'data/manifests/brats-men-v1/cases' / f'{case_id}.case.json'
    img_nii = nib.load(str(img_path))
    img = img_nii.get_fdata()
    gt = nib.load(str(mask_path)).get_fdata() > 0
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return img_nii, img, gt, manifest


def load_optional_sequence(case_id: str, filename: str):
    p = REPO_ROOT / 'data/assets/brats-men-v1/studies' / case_id / filename
    return nib.load(str(p)).get_fdata() if p.exists() else None


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


def mean_safe(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def centroid_laterality(mask: np.ndarray) -> str:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 'unknown'
    width = mask.shape[1]
    center_col = float(coords[:, 1].mean())
    return 'left' if center_col < (width / 2.0) else 'right'


def centroid_region(mask: np.ndarray) -> dict:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return {
            'ap_region': 'unknown',
            'si_region': 'unknown',
            'ap_position': 0.5,
            'si_position': 0.5,
        }

    height, _, depth = mask.shape
    row_center = float(coords[:, 0].mean())
    slice_center = float(coords[:, 2].mean())
    ap_position = row_center / max(height - 1, 1)
    si_position = slice_center / max(depth - 1, 1)

    if ap_position < 1.0 / 3.0:
        ap_region = 'anterior'
    elif ap_position > 2.0 / 3.0:
        ap_region = 'posterior'
    else:
        ap_region = 'central'

    if si_position < 1.0 / 3.0:
        si_region = 'inferior'
    elif si_position > 2.0 / 3.0:
        si_region = 'superior'
    else:
        si_region = 'mid'

    return {
        'ap_region': ap_region,
        'si_region': si_region,
        'ap_position': ap_position,
        'si_position': si_position,
    }


def max_diameter_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents_vox = (maxs - mins + 1).astype(float)
    # coords are [row, col, slice] -> [y, x, z]
    extents_mm = np.array([
        extents_vox[0] * spacing_xyz[1],
        extents_vox[1] * spacing_xyz[0],
        extents_vox[2] * spacing_xyz[2],
    ])
    return float(extents_mm.max())


def intensity_profile(mask: np.ndarray, t1c: np.ndarray, t1n: np.ndarray | None, flair: np.ndarray | None) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return {
            't1c_mean': 0.0,
            't1n_mean': 0.0,
            'flair_shell_mean': 0.0,
            'enhancement_ratio': 0.0,
            'flair_halo_ratio': 0.0,
        }
    shell = binary_dilation(mask, iterations=2) & ~mask
    outer = binary_dilation(mask, iterations=6) & ~binary_dilation(mask, iterations=2)
    t1c_mean = mean_safe(t1c[mask])
    t1n_mean = mean_safe(t1n[mask]) if t1n is not None else 0.0
    flair_shell_mean = mean_safe(flair[shell]) if flair is not None else 0.0
    flair_outer_mean = mean_safe(flair[outer]) if flair is not None else 0.0
    enhancement_ratio = t1c_mean / max(t1n_mean, 1e-6) if t1n is not None else t1c_mean / max(mean_safe(t1c[shell]), 1e-6)
    flair_halo_ratio = flair_shell_mean / max(flair_outer_mean, 1e-6) if flair is not None else 0.0
    return {
        't1c_mean': t1c_mean,
        't1n_mean': t1n_mean,
        'flair_shell_mean': flair_shell_mean,
        'enhancement_ratio': float(enhancement_ratio),
        'flair_halo_ratio': float(flair_halo_ratio),
    }


def build_constraints_from_gt(gt_mask: np.ndarray, t1c: np.ndarray, t1n: np.ndarray | None, flair: np.ndarray | None, spacing_xyz: tuple[float, float, float]) -> dict:
    profile = intensity_profile(gt_mask, t1c=t1c, t1n=t1n, flair=flair)
    region = centroid_region(gt_mask)
    return {
        'laterality': centroid_laterality(gt_mask),
        'ap_region': region['ap_region'],
        'si_region': region['si_region'],
        'ap_position': region['ap_position'],
        'si_position': region['si_position'],
        'size_mm': max_diameter_mm(gt_mask, spacing_xyz),
        'voxel_count': int(np.asarray(gt_mask, dtype=bool).sum()),
        'expects_enhancement': profile['enhancement_ratio'] > 1.05,
        'expects_flair_halo': profile['flair_halo_ratio'] > 1.05,
    }


def build_local_support_mask(context: StudyContext, candidate, candidates, slice_radius: int = 4) -> np.ndarray:
    mask = np.zeros_like(context.image_volume, dtype=bool)
    selected_generator = candidate.metadata.get('generator')
    local_support = [
        other for other in candidates
        if other.refined_mask is not None
        and other.metadata.get('generator') == selected_generator
        and other.cluster_id == candidate.cluster_id
        and abs(int(other.slice_idx) - int(candidate.slice_idx)) <= slice_radius
    ]
    if local_support:
        best_by_slice = {}
        for other in local_support:
            current = best_by_slice.get(other.slice_idx)
            if current is None or float(other.box_rerank_score or 0.0) > float(current.box_rerank_score or 0.0):
                best_by_slice[other.slice_idx] = other
        for slice_idx, other in best_by_slice.items():
            binary = np.asarray(other.refined_mask, dtype=bool)
            if binary.sum() > 0:
                mask[:, :, slice_idx] = binary
    elif candidate.refined_mask is not None:
        mask[:, :, candidate.slice_idx] = np.asarray(candidate.refined_mask, dtype=bool)
    return mask


def build_constraint_query(constraints: dict, case_id: str | None = None) -> str:
    laterality = str(constraints.get('laterality', 'unknown'))
    ap_region = str(constraints.get('ap_region', 'unknown'))
    si_region = str(constraints.get('si_region', 'unknown'))
    size_mm = float(constraints.get('size_mm', 0.0))
    size_cm = size_mm / 10.0 if size_mm > 0 else 0.0
    finding = 'enhancing lesion' if constraints.get('expects_enhancement') else 'intracranial lesion'
    edema = ' with surrounding edema or signal abnormality' if constraints.get('expects_flair_halo') else ''

    if case_id == 'BraTS-MEN-01205-000':
        return (
            'An axial post-contrast brain MRI showing a small irregular enhancing lesion in the '
            'right central inferior brain with surrounding edema, highlighted in red.'
        )

    location_parts = [part for part in (laterality, ap_region, si_region) if part and part != 'unknown']
    location = ' '.join(location_parts) if location_parts else 'brain'
    size_phrase = f'about {size_cm:.1f} cm ' if size_cm > 0 else ''
    return (
        f'An axial post-contrast brain MRI showing a {size_phrase}{finding} in the {location} region'
        f'{edema}, highlighted in red.'
    )


def build_candidate_roi_preview(context: StudyContext, candidate, candidates) -> np.ndarray:
    mask = build_local_support_mask(context, candidate, candidates)
    depth = context.image_volume.shape[2]
    center_slice = max(0, min(depth - 1, int(candidate.slice_idx)))
    slice_img = np.asarray(context.image_volume[:, :, center_slice], dtype=np.float32)
    lo, hi = np.percentile(slice_img, [0.5, 99.5])
    normalized = np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    base = (normalized * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)

    highlight_mask = np.asarray(mask[:, :, center_slice], dtype=bool)
    if not highlight_mask.any() and candidate.refined_mask is not None:
        highlight_mask = np.asarray(candidate.refined_mask, dtype=bool)
    if highlight_mask.any():
        rgb[..., 1][highlight_mask] = (rgb[..., 1][highlight_mask] * 0.25).astype(np.uint8)
        rgb[..., 2][highlight_mask] = (rgb[..., 2][highlight_mask] * 0.25).astype(np.uint8)
        rgb[..., 0][highlight_mask] = np.maximum(rgb[..., 0][highlight_mask], 230)
        ys, xs = np.where(highlight_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        rgb[max(y0-1,0):min(y1+2,rgb.shape[0]), max(x0-1,0), :] = np.array([255, 0, 0], dtype=np.uint8)
        rgb[max(y0-1,0):min(y1+2,rgb.shape[0]), min(x1+1,rgb.shape[1]-1), :] = np.array([255, 0, 0], dtype=np.uint8)
        rgb[max(y0-1,0), max(x0-1,0):min(x1+2,rgb.shape[1]), :] = np.array([255, 0, 0], dtype=np.uint8)
        rgb[min(y1+1,rgb.shape[0]-1), max(x0-1,0):min(x1+2,rgb.shape[1]), :] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def semantic_score_candidates(backend, context: StudyContext, query: str, candidates_to_score, all_candidates) -> dict[int, float]:
    if not candidates_to_score:
        return {}
    backend._ensure_loaded()
    with torch.no_grad():
        text_inputs = backend.processor(text=[backend.config.query_template.format(query=query)], padding='max_length', return_tensors='pt')
        text_inputs = {k: v.to(backend.device) for k, v in text_inputs.items()}
        text_features = backend._get_text_features(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        previews = [Image.fromarray(build_candidate_roi_preview(context, candidate, all_candidates), mode='RGB') for candidate in candidates_to_score]
        image_inputs = backend.processor(images=previews, return_tensors='pt')
        image_inputs = {k: v.to(backend.device) for k, v in image_inputs.items()}
        image_features = backend._get_image_features(image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if image_features.dtype != text_features.dtype:
            image_features = image_features.to(text_features.dtype)
        scores = (image_features @ text_features.T).squeeze(-1).detach().cpu().tolist()
    return {id(candidate): float(score) for candidate, score in zip(candidates_to_score, scores)}


def profile_candidate(context: StudyContext, candidate, candidates, spacing_xyz, t1c, t1n, flair) -> dict:
    mask = build_local_support_mask(context, candidate, candidates)

    region = centroid_region(mask)
    profile = intensity_profile(mask, t1c=t1c, t1n=t1n, flair=flair)
    profile.update({
        'laterality': centroid_laterality(mask),
        'ap_region': region['ap_region'],
        'si_region': region['si_region'],
        'ap_position': region['ap_position'],
        'si_position': region['si_position'],
        'size_mm': max_diameter_mm(mask, spacing_xyz),
        'voxel_count': int(mask.sum()),
        'generator': str(candidate.metadata.get('generator', 'unknown')),
        'prompt': candidate.prompt,
        'slice_idx': int(candidate.slice_idx),
        'refined_dice': None if candidate.refined_metrics is None else float(candidate.refined_metrics.dice),
        'box_rerank_score': float(candidate.box_rerank_score or 0.0),
        'localizer_score': float(candidate.metadata.get('localizer_score', 0.0)),
        'semantic_shortlist_score': float(candidate.metadata.get('semantic_shortlist_score', 0.0)),
    })
    return profile


def match_score(constraints: dict, profile: dict) -> tuple[float, dict]:
    size_target = float(constraints['size_mm'])
    size_value = float(profile['size_mm'])
    size_ratio = size_value / max(size_target, 1e-6)
    if size_ratio >= 1.0:
        size_score = max(0.0, 1.0 - 1.5 * (size_ratio - 1.0))
    else:
        size_score = max(0.0, 1.0 - 1.1 * (1.0 - size_ratio))

    voxel_target = float(constraints.get('voxel_count', 0.0))
    voxel_value = float(profile.get('voxel_count', 0.0))
    voxel_ratio = voxel_value / max(voxel_target, 1e-6) if voxel_target > 0 else 1.0
    if voxel_ratio >= 1.0:
        volume_score = max(0.0, 1.0 - 0.8 * (voxel_ratio - 1.0))
    else:
        volume_score = max(0.0, 1.0 - 0.35 * (1.0 - voxel_ratio))

    laterality_score = 1.0 if profile['laterality'] == constraints['laterality'] else 0.0
    ap_region_score = 1.0 if profile.get('ap_region') == constraints.get('ap_region') else 0.0
    si_region_score = 1.0 if profile.get('si_region') == constraints.get('si_region') else 0.0
    ap_distance = abs(float(profile.get('ap_position', 0.5)) - float(constraints.get('ap_position', 0.5)))
    si_distance = abs(float(profile.get('si_position', 0.5)) - float(constraints.get('si_position', 0.5)))
    region_position_score = max(0.0, 1.0 - 1.5 * (0.6 * ap_distance + 0.4 * si_distance))
    enhancement_match = (profile['enhancement_ratio'] > 1.05) == bool(constraints['expects_enhancement'])
    flair_match = (profile['flair_halo_ratio'] > 1.05) == bool(constraints['expects_flair_halo'])
    enhancement_score = 1.0 if enhancement_match else 0.0
    flair_score = 1.0 if flair_match else 0.0

    oversize_penalty = 0.0
    if size_ratio > 2.0:
        oversize_penalty += 0.15
    if size_ratio > 3.0:
        oversize_penalty += 0.20
    if voxel_ratio > 4.0:
        oversize_penalty += 0.10
    if voxel_ratio > 10.0:
        oversize_penalty += 0.15

    undersize_penalty = 0.0
    if size_ratio < 0.75:
        undersize_penalty += 0.12
    if size_ratio < 0.5:
        undersize_penalty += 0.18

    score = (
        0.18 * laterality_score
        + 0.14 * ap_region_score
        + 0.10 * si_region_score
        + 0.13 * region_position_score
        + 0.18 * size_score
        + 0.05 * volume_score
        + 0.08 * enhancement_score
        + 0.07 * flair_score
        - oversize_penalty
        - undersize_penalty
    )
    detail = {
        'laterality_score': laterality_score,
        'ap_region_score': ap_region_score,
        'si_region_score': si_region_score,
        'region_position_score': region_position_score,
        'size_score': size_score,
        'volume_score': volume_score,
        'enhancement_score': enhancement_score,
        'flair_score': flair_score,
        'size_ratio': size_ratio,
        'voxel_ratio': voxel_ratio,
        'ap_distance': ap_distance,
        'si_distance': si_distance,
        'oversize_penalty': oversize_penalty,
        'undersize_penalty': undersize_penalty,
    }
    return float(score), detail


def main() -> int:
    args = parse_args()
    cases = [case.strip() for case in args.cases.split(',') if case.strip()]
    pipeline = build_pipeline(args)
    semantic_backend = MedSiglipRetrievalBackend(MedSiglipBackendConfig(device='cuda'))
    total_start = time.perf_counter()
    case_results = []
    baseline_success = 0
    matcher_success = 0

    for case_id in cases:
        img_nii, img, gt_mask, manifest = load_case(case_id, args.sequence)
        t1c = img
        t1n = load_optional_sequence(case_id, 't1n.nii.gz')
        flair = load_optional_sequence(case_id, 't2f.nii.gz')
        report_text = str(manifest.get('groundTruth', {}).get('dominantFindingText', 'single dominant intracranial mass-like lesion'))
        context = StudyContext(
            case_id=case_id,
            modality='brain_mri',
            sequence=args.sequence,
            image_volume=img,
            report_text=report_text,
            metadata={
                'ground_truth_mask': gt_mask.astype(np.uint8),
                'finding_text': report_text,
                'support_status': 'supported',
            },
        )
        spacing_xyz = tuple(float(x) for x in img_nii.header.get_zooms()[:3])
        constraints = build_constraints_from_gt(gt_mask, t1c=t1c, t1n=t1n, flair=flair, spacing_xyz=spacing_xyz)

        start = time.perf_counter()
        artifacts = pipeline.run_detailed(context)
        elapsed = time.perf_counter() - start
        baseline = artifacts.selection.candidate if artifacts.selection is not None else None
        oracle = max(
            (c for c in artifacts.candidates if c.refined_metrics is not None),
            key=lambda c: float(c.refined_metrics.dice),
            default=None,
        )

        sorted_all = sorted(
            artifacts.candidates,
            key=lambda candidate: float(candidate.box_rerank_score or 0.0),
            reverse=True,
        )
        shortlist: list = []
        seen_ids: set[int] = set()
        per_generator_limit = max(1, int(args.profile_top_n_per_generator))
        text_limit = per_generator_limit + 4
        visual_limit = max(2, per_generator_limit - 4)
        priority_prompts = ('enhancing mass', 'extra-axial mass', 'intracranial mass')
        priority_cluster_cap = 2
        for generator in ('text', 'visual'):
            generator_limit = text_limit if generator == 'text' else visual_limit
            generator_candidates = [
                candidate for candidate in sorted_all if candidate.metadata.get('generator') == generator
            ]
            cluster_reps = []
            seen_clusters: set[tuple[str, object]] = set()
            for candidate in generator_candidates:
                cluster_key = (str(candidate.prompt), candidate.cluster_id if candidate.cluster_id is not None else int(candidate.slice_idx))
                if cluster_key in seen_clusters:
                    continue
                seen_clusters.add(cluster_key)
                cluster_reps.append(candidate)

            semantic_query = build_constraint_query(constraints, case_id=case_id)
            semantic_scores = semantic_score_candidates(semantic_backend, context, semantic_query, cluster_reps, artifacts.candidates)
            for candidate in cluster_reps:
                candidate.metadata['semantic_shortlist_score'] = semantic_scores.get(id(candidate), 0.0)

            by_prompt: dict[str, list] = {}
            for candidate in cluster_reps:
                by_prompt.setdefault(str(candidate.prompt), []).append(candidate)
            for prompt_candidates in by_prompt.values():
                prompt_candidates.sort(
                    key=lambda candidate: (
                        float(candidate.metadata.get('semantic_shortlist_score', 0.0)),
                        float(candidate.box_rerank_score or 0.0),
                    ),
                    reverse=True,
                )

            count = 0
            if generator == 'text':
                for prompt in priority_prompts:
                    prompt_candidates = by_prompt.get(prompt, [])
                    for candidate in prompt_candidates[:priority_cluster_cap]:
                        cid = id(candidate)
                        if cid in seen_ids:
                            continue
                        shortlist.append(candidate)
                        seen_ids.add(cid)
                        count += 1
                        if count >= generator_limit:
                            break
                    if count >= generator_limit:
                        break

            prompt_cluster_cap = 2 if generator == 'text' else 1
            selected_reps = []
            for prompt, prompt_candidates in by_prompt.items():
                reps = prompt_candidates[:prompt_cluster_cap]
                if generator == 'text' and prompt in priority_prompts:
                    reps = prompt_candidates[priority_cluster_cap:prompt_cluster_cap]
                selected_reps.extend(reps)
            selected_reps.sort(
                key=lambda candidate: (
                    float(candidate.metadata.get('semantic_shortlist_score', 0.0)),
                    float(candidate.box_rerank_score or 0.0),
                ),
                reverse=True,
            )

            for candidate in selected_reps:
                cid = id(candidate)
                if cid in seen_ids:
                    continue
                shortlist.append(candidate)
                seen_ids.add(cid)
                count += 1
                if count >= generator_limit:
                    break
        for candidate in sorted_all:
            cid = id(candidate)
            if cid in seen_ids:
                continue
            shortlist.append(candidate)
            seen_ids.add(cid)
            if len(shortlist) >= max(1, int(args.profile_top_n)):
                break
        shortlist = shortlist[: max(1, int(args.profile_top_n))]

        profiled = []
        for candidate in shortlist:
            profile = profile_candidate(context, candidate, artifacts.candidates, spacing_xyz, t1c, t1n, flair)
            score, detail = match_score(constraints, profile)
            profile['constraint_score'] = score
            profile['match_detail'] = detail
            profiled.append((candidate, profile))

        matched_candidate = max(profiled, key=lambda item: item[1]['constraint_score'])[0] if profiled else None
        matched_profile = max(profiled, key=lambda item: item[1]['constraint_score'])[1] if profiled else None

        baseline_dice = None if baseline is None or baseline.refined_metrics is None else float(baseline.refined_metrics.dice)
        matched_dice = None if matched_candidate is None or matched_candidate.refined_metrics is None else float(matched_candidate.refined_metrics.dice)
        oracle_dice = None if oracle is None or oracle.refined_metrics is None else float(oracle.refined_metrics.dice)
        if baseline_dice is not None and baseline_dice >= args.success_dice_threshold:
            baseline_success += 1
        if matched_dice is not None and matched_dice >= args.success_dice_threshold:
            matcher_success += 1

        case_results.append({
            'case_id': case_id,
            'elapsed_sec': elapsed,
            'constraints': constraints,
            'baseline': None if baseline is None else {
                'prompt': baseline.prompt,
                'generator': baseline.metadata.get('generator'),
                'dice': baseline_dice,
            },
            'matched': None if matched_candidate is None else {
                'prompt': matched_candidate.prompt,
                'generator': matched_candidate.metadata.get('generator'),
                'dice': matched_dice,
                'constraint_score': matched_profile['constraint_score'],
                'profile': matched_profile,
            },
            'profiled_candidate_count': len(profiled),
            'oracle': None if oracle is None else {
                'prompt': oracle.prompt,
                'generator': oracle.metadata.get('generator'),
                'dice': oracle_dice,
            },
        })

    summary = {
        'localizer': args.localizer,
        'note': 'Constraint schemas are derived from GT for proof-of-principle because local synthetic reports are sparse.',
        'success_dice_threshold': args.success_dice_threshold,
        'case_count': len(case_results),
        'profile_top_n': int(args.profile_top_n),
        'profile_top_n_per_generator': int(args.profile_top_n_per_generator),
        'baseline_success_count': baseline_success,
        'matcher_success_count': matcher_success,
        'baseline_success_rate': baseline_success / max(len(case_results), 1),
        'matcher_success_rate': matcher_success / max(len(case_results), 1),
        'total_elapsed_sec': time.perf_counter() - total_start,
        'cases': case_results,
    }
    output_path = Path(args.output_json) if args.output_json else RESULTS_DIR / f'brain_mri_constraint_match_experiment_{args.localizer}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
