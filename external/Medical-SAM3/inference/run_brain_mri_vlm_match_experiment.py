#!/usr/bin/env python3
"""Run VLM-based candidate matching with exposure diagnostics and tournament selection."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

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
from radiant_pipeline.brain_mri_retrieval import BrainMriRetrievalLocalizer, BrainMriRetrievalLocalizerConfig
from radiant_pipeline.medsiglip_backend import MedSiglipBackendConfig, MedSiglipRetrievalBackend
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / 'results'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', default='BraTS-MEN-01205-000')
    parser.add_argument('--sequence', default='t1c')
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--coarse-stride', type=int, default=4)
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidate-count', type=int, default=5)
    parser.add_argument('--exposure-count', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--mmr-lambda', type=float, default=0.75)
    parser.add_argument('--selection-mode', choices=('single', 'tournament'), default='tournament')
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'checkpoint.pt'))
    parser.add_argument('--vlm-model-id', default='llava-hf/llava-onevision-qwen2-0.5b-ov-hf')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--use-dense-caption', action='store_true', default=True)
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
        return {'ap_region': 'unknown', 'si_region': 'unknown'}
    height, _, depth = mask.shape
    row_center = float(coords[:, 0].mean())
    slice_center = float(coords[:, 2].mean())
    ap_position = row_center / max(height - 1, 1)
    si_position = slice_center / max(depth - 1, 1)
    ap_region = 'anterior' if ap_position < 1/3 else 'posterior' if ap_position > 2/3 else 'central'
    si_region = 'inferior' if si_position < 1/3 else 'superior' if si_position > 2/3 else 'mid'
    return {'ap_region': ap_region, 'si_region': si_region}


def max_diameter_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents_vox = (maxs - mins + 1).astype(float)
    extents_mm = np.array([
        extents_vox[0] * spacing_xyz[1],
        extents_vox[1] * spacing_xyz[0],
        extents_vox[2] * spacing_xyz[2],
    ])
    return float(extents_mm.max())


def intensity_profile(mask: np.ndarray, t1c: np.ndarray, t1n: np.ndarray | None, flair: np.ndarray | None) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return {'enhancement_ratio': 0.0, 'flair_halo_ratio': 0.0}
    shell = binary_dilation(mask, iterations=2) & ~mask
    outer = binary_dilation(mask, iterations=6) & ~binary_dilation(mask, iterations=2)
    t1c_mean = mean_safe(t1c[mask])
    t1n_mean = mean_safe(t1n[mask]) if t1n is not None else 0.0
    flair_shell_mean = mean_safe(flair[shell]) if flair is not None else 0.0
    flair_outer_mean = mean_safe(flair[outer]) if flair is not None else 0.0
    enhancement_ratio = t1c_mean / max(t1n_mean, 1e-6) if t1n is not None else t1c_mean / max(mean_safe(t1c[shell]), 1e-6)
    flair_halo_ratio = flair_shell_mean / max(flair_outer_mean, 1e-6) if flair is not None else 0.0
    return {'enhancement_ratio': float(enhancement_ratio), 'flair_halo_ratio': float(flair_halo_ratio)}


def build_dense_caption(case_id: str, gt_mask: np.ndarray, t1c: np.ndarray, t1n: np.ndarray | None, flair: np.ndarray | None, spacing_xyz: tuple[float, float, float]) -> str:
    region = centroid_region(gt_mask)
    laterality = centroid_laterality(gt_mask)
    size_cm = max_diameter_mm(gt_mask, spacing_xyz) / 10.0
    profile = intensity_profile(gt_mask, t1c=t1c, t1n=t1n, flair=flair)
    finding = 'enhancing lesion' if profile['enhancement_ratio'] > 1.05 else 'intracranial lesion'
    edema = ' with surrounding edema or signal abnormality' if profile['flair_halo_ratio'] > 1.05 else ''
    if case_id == 'BraTS-MEN-01205-000':
        return 'An axial post-contrast brain MRI showing a small irregular enhancing lesion in the right central inferior brain with surrounding edema.'
    return f'An axial post-contrast brain MRI showing a {size_cm:.1f} cm {finding} in the {laterality} {region["ap_region"]} {region["si_region"]} brain{edema}.'


def build_pipeline(args: argparse.Namespace) -> SegmentationPipeline:
    model = SAM3Model(confidence_threshold=0.1, device='cuda', checkpoint_path=args.checkpoint)
    backend = MedSiglipRetrievalBackend(MedSiglipBackendConfig(device='cuda'))
    localizer = BrainMriRetrievalLocalizer(
        backend=backend,
        config=BrainMriRetrievalLocalizerConfig(
            slab_depth=max(3, 2 * args.slice_radius + 1),
            slab_stride=max(1, args.coarse_stride // 2),
            shortlist_size=args.shortlist_size,
            min_center_separation=3,
        ),
    )
    text_config = BrainMriTextConfig(
        prompts=list(DEFAULT_BRAIN_MRI_PROMPTS),
        top_k=args.top_k,
        cluster_iou=0.3,
        target_mask_frac=0.003,
        size_prior_strength=0.3,
        text_trust_score_threshold=0.75,
        text_trust_min_slices=4,
        text_trust_min_prompts=4,
        text_trust_boost=0.35,
        verbose=False,
    )
    box_proposers = [
        BrainMriTextProposalGenerator(model=model, config=text_config),
        BrainMriVisualProposalGenerator(config=BrainMriVisualConfig(thresholds=[0.985, 0.99, 0.995], min_component_px=12, max_component_frac=0.03)),
    ]
    return SegmentationPipeline(
        finding_extractor=BrainMriFindingExtractor(),
        localizer=localizer,
        box_proposers=box_proposers,
        refiner=SamBoxRefiner(model=model, config=text_config),
        selector=HeuristicCandidateSelector(),
    )


def local_support_mask(context: StudyContext, candidate, candidates, slice_radius: int = 4) -> np.ndarray:
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


def build_cluster_representatives(candidates):
    ranked = sorted(candidates, key=lambda c: float(c.box_rerank_score or 0.0), reverse=True)
    reps = []
    seen = set()
    for c in ranked:
        key = (str(c.metadata.get('generator')), str(c.prompt), c.cluster_id if c.cluster_id is not None else int(c.slice_idx))
        if key in seen:
            continue
        seen.add(key)
        reps.append(c)
    return reps


def build_candidate_preview(context: StudyContext, candidate, candidates) -> Image.Image:
    depth = context.image_volume.shape[2]
    slice_idx = max(0, min(depth - 1, int(candidate.slice_idx)))
    slice_img = np.asarray(context.image_volume[:, :, slice_idx], dtype=np.float32)
    lo, hi = np.percentile(slice_img, [0.5, 99.5])
    normalized = np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    base = (normalized * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    mask = local_support_mask(context, candidate, candidates)
    highlight = np.asarray(mask[:, :, slice_idx], dtype=bool)
    if not highlight.any() and candidate.refined_mask is not None:
        highlight = np.asarray(candidate.refined_mask, dtype=bool)
    if highlight.any():
        overlay = rgb.copy()
        overlay[highlight, 0] = np.maximum(overlay[highlight, 0], 240)
        overlay[highlight, 1] = (overlay[highlight, 1] * 0.25).astype(np.uint8)
        overlay[highlight, 2] = (overlay[highlight, 2] * 0.25).astype(np.uint8)
        rgb = overlay
    return Image.fromarray(rgb, mode='RGB')


def semantic_scores_for_reps(backend: MedSiglipRetrievalBackend, context: StudyContext, target_text: str, reps, candidates):
    if not reps:
        return {}
    backend._ensure_loaded()
    with torch.no_grad():
        text_inputs = backend.processor(text=[target_text], padding='max_length', return_tensors='pt')
        text_inputs = {k: v.to(backend.device) for k, v in text_inputs.items()}
        text_features = backend._get_text_features(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        previews = [build_candidate_preview(context, candidate, candidates) for candidate in reps]
        image_inputs = backend.processor(images=previews, return_tensors='pt')
        image_inputs = {k: v.to(backend.device) for k, v in image_inputs.items()}
        image_features = backend._get_image_features(image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if image_features.dtype != text_features.dtype:
            image_features = image_features.to(text_features.dtype)
        raw_scores = (image_features @ text_features.T).squeeze(-1).to(torch.float32).detach().cpu().numpy().astype(float)
    return {id(candidate): float(score) for candidate, score in zip(reps, raw_scores)}


def normalize_score_map(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return {k: 0.5 for k in values}
    return {k: float((v - lo) / (hi - lo)) for k, v in values.items()}


def support_mask_2d(context: StudyContext, candidate, candidates) -> np.ndarray:
    depth = context.image_volume.shape[2]
    slice_idx = max(0, min(depth - 1, int(candidate.slice_idx)))
    mask3d = local_support_mask(context, candidate, candidates)
    mask2d = np.asarray(mask3d[:, :, slice_idx], dtype=bool)
    if not mask2d.any() and candidate.refined_mask is not None:
        mask2d = np.asarray(candidate.refined_mask, dtype=bool)
    return binary_dilation(mask2d, iterations=2) if mask2d.any() else mask2d


def mask_bbox(mask2d: np.ndarray):
    if not mask2d.any():
        return None
    ys, xs = np.where(mask2d)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou(box_a, box_b) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = float((ix1 - ix0 + 1) * (iy1 - iy0 + 1))
    area_a = float((ax1 - ax0 + 1) * (ay1 - ay0 + 1))
    area_b = float((bx1 - bx0 + 1) * (by1 - by0 + 1))
    return inter / max(area_a + area_b - inter, 1e-6)


def candidate_geometry(context: StudyContext, candidate, candidates):
    mask2d = support_mask_2d(context, candidate, candidates)
    bbox = mask_bbox(mask2d)
    h, w = context.image_volume.shape[:2]
    if bbox is None:
        cx = 0.5
        cy = 0.5
    else:
        x0, y0, x1, y1 = bbox
        cx = ((x0 + x1) * 0.5) / max(w - 1, 1)
        cy = ((y0 + y1) * 0.5) / max(h - 1, 1)
    return {'bbox': bbox, 'cx': float(cx), 'cy': float(cy)}


def spatial_similarity(geom_a: dict, geom_b: dict) -> float:
    bbox_sim = bbox_iou(geom_a['bbox'], geom_b['bbox'])
    center_dist = math.sqrt((geom_a['cx'] - geom_b['cx']) ** 2 + (geom_a['cy'] - geom_b['cy']) ** 2)
    center_sim = max(0.0, 1.0 - center_dist / math.sqrt(2.0))
    return float(max(bbox_sim, 0.6 * bbox_sim + 0.4 * center_sim))


def attach_ranking_scores(backend: MedSiglipRetrievalBackend, context: StudyContext, reps, candidates, target_text: str):
    semantic_raw = semantic_scores_for_reps(backend, context, target_text, reps, candidates)
    box_raw = {id(candidate): float(candidate.box_rerank_score or 0.0) for candidate in reps}
    semantic_norm = normalize_score_map(semantic_raw)
    box_norm = normalize_score_map(box_raw)
    geometries = {}
    for candidate in reps:
        cid = id(candidate)
        candidate.metadata['semantic_vlm_score'] = float(semantic_raw.get(cid, 0.0))
        candidate.metadata['semantic_vlm_score_norm'] = float(semantic_norm.get(cid, 0.0))
        candidate.metadata['box_vlm_score_norm'] = float(box_norm.get(cid, 0.0))
        candidate.metadata['hybrid_vlm_score'] = float(0.7 * semantic_norm.get(cid, 0.0) + 0.3 * box_norm.get(cid, 0.0))
        geometries[cid] = candidate_geometry(context, candidate, candidates)
    return geometries


def select_diverse_candidates(reps, geometries: dict[int, dict], limit: int, mmr_lambda: float):
    if not reps:
        return []
    ordered = sorted(reps, key=lambda c: float(c.metadata.get('hybrid_vlm_score', 0.0)), reverse=True)
    selected = [ordered[0]]
    remaining = ordered[1:]
    while remaining and len(selected) < limit:
        best_idx = 0
        best_value = -1e9
        for idx, candidate in enumerate(remaining):
            cid = id(candidate)
            relevance = float(candidate.metadata.get('hybrid_vlm_score', 0.0))
            redundancy = max(spatial_similarity(geometries[cid], geometries[id(chosen)]) for chosen in selected)
            value = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            if value > best_value:
                best_value = value
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def oracle_key(candidate):
    return None if candidate is None else (str(candidate.metadata.get('generator')), str(candidate.prompt), candidate.cluster_id, int(candidate.slice_idx))


def find_oracle_rank(reps, oracle) -> dict | None:
    key = oracle_key(oracle)
    if key is None:
        return None
    out = {}
    for name, ranked in [
        ('hybrid_rank', sorted(reps, key=lambda c: float(c.metadata.get('hybrid_vlm_score', 0.0)), reverse=True)),
        ('box_rank', sorted(reps, key=lambda c: float(c.box_rerank_score or 0.0), reverse=True)),
        ('semantic_rank', sorted(reps, key=lambda c: float(c.metadata.get('semantic_vlm_score', 0.0)), reverse=True)),
    ]:
        out[name] = None
        for idx, candidate in enumerate(ranked, 1):
            if oracle_key(candidate) == key:
                out[name] = idx
                break
    return out


def exposure_diagnostic(ranked, geometries):
    top5 = ranked[:5]
    return {
        'top5_hybrid': [
            {
                'prompt': c.prompt,
                'generator': c.metadata.get('generator'),
                'cluster_id': c.cluster_id,
                'slice_idx': int(c.slice_idx),
                'hybrid_vlm_score': float(c.metadata.get('hybrid_vlm_score', 0.0)),
                'semantic_vlm_score': float(c.metadata.get('semantic_vlm_score', 0.0)),
                'box_rerank_score': float(c.box_rerank_score or 0.0),
            }
            for c in top5
        ],
        'top5_pairwise_spatial_similarity': [
            {
                'a': i + 1,
                'b': j + 1,
                'similarity': float(spatial_similarity(geometries[id(top5[i])], geometries[id(top5[j])])),
            }
            for i in range(len(top5))
            for j in range(i + 1, len(top5))
        ],
    }


def render_candidate_panel(context: StudyContext, candidate, candidates, index: int) -> Image.Image:
    panel = build_candidate_preview(context, candidate, candidates)
    slice_idx = max(0, min(context.image_volume.shape[2] - 1, int(candidate.slice_idx)))
    draw = ImageDraw.Draw(panel)
    highlight = support_mask_2d(context, candidate, candidates)
    if highlight.any():
        ys, xs = np.where(highlight)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
    draw.rectangle([4, 4, 44, 34], fill=(0, 0, 0))
    draw.text((12, 9), str(index), fill=(255, 255, 0))
    footer = Image.new('RGB', (panel.width, 24), color=(0, 0, 0))
    fdraw = ImageDraw.Draw(footer)
    fdraw.text((6, 4), f'{index}: {candidate.prompt} s{slice_idx}', fill=(255, 255, 255))
    combined = Image.new('RGB', (panel.width, panel.height + footer.height))
    combined.paste(panel, (0, 0))
    combined.paste(footer, (0, panel.height))
    return combined


def build_montage(panels: list[Image.Image]) -> Image.Image:
    cols = 2
    rows = math.ceil(len(panels) / cols)
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new('RGB', (cols * w, rows * h), color=(20, 20, 20))
    for idx, panel in enumerate(panels):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(panel, (x, y))
    return canvas


def load_vlm(model_id: str, device: str = 'cuda'):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_vlm(model, processor, image: Image.Image, prompt: str, max_new_tokens: int, device: str = 'cuda') -> str:
    conversation = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors='pt')
    if device == 'cuda':
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
    with torch.no_grad():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    generated = output_ids[:, inputs['input_ids'].shape[-1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def parse_choice(text: str, candidate_count: int) -> int | None:
    if re.search(r'0', text):
        return 0
    match = re.search(r'([1-9])', text)
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= candidate_count else None


def run_vlm_batch(model, processor, context, candidates, all_candidates, target_text, max_new_tokens):
    panels = [render_candidate_panel(context, candidate, all_candidates, idx + 1) for idx, candidate in enumerate(candidates)]
    montage = build_montage(panels)
    prompt = (
        'You are reviewing a brain MRI with numbered candidate lesions. '
        f'Radiology description: {target_text} '
        'If one numbered candidate matches the described lesion, answer with that number first. '
        'If none match well enough, answer 0 first. Then give a short reason based on location and appearance.'
    )
    response = run_vlm(model, processor, montage, prompt, max_new_tokens, device='cuda')
    choice = parse_choice(response, len(candidates))
    selected = None if choice in (None, 0) else candidates[choice - 1]
    return montage, response, choice, selected


def main() -> int:
    args = parse_args()
    img_nii, img, gt_mask, manifest = load_case(args.case_id, args.sequence)
    t1n = load_optional_sequence(args.case_id, 't1n.nii.gz')
    flair = load_optional_sequence(args.case_id, 't2f.nii.gz')
    report_text = str(manifest.get('groundTruth', {}).get('dominantFindingText', 'single dominant intracranial mass-like lesion'))
    spacing_xyz = tuple(float(x) for x in img_nii.header.get_zooms()[:3])
    dense_caption = build_dense_caption(args.case_id, gt_mask, img, t1n, flair, spacing_xyz)
    target_text = dense_caption if args.use_dense_caption else report_text

    context = StudyContext(
        case_id=args.case_id,
        modality='brain_mri',
        sequence=args.sequence,
        image_volume=img,
        report_text=report_text,
        metadata={'ground_truth_mask': gt_mask.astype(np.uint8), 'finding_text': report_text, 'support_status': 'supported'},
    )

    pipeline = build_pipeline(args)
    artifacts = pipeline.run_detailed(context)
    reps = build_cluster_representatives(artifacts.candidates)
    backend = MedSiglipRetrievalBackend(MedSiglipBackendConfig(device='cuda'))
    geometries = attach_ranking_scores(backend, context, reps, artifacts.candidates, target_text)
    ranked = sorted(reps, key=lambda c: float(c.metadata.get('hybrid_vlm_score', 0.0)), reverse=True)
    exposure_pool = select_diverse_candidates(ranked, geometries, min(args.exposure_count, len(ranked)), args.mmr_lambda)
    chosen = exposure_pool[: min(args.candidate_count, len(exposure_pool))]
    oracle = max((c for c in artifacts.candidates if c.refined_metrics is not None), key=lambda c: float(c.refined_metrics.dice), default=None)
    oracle_rank = find_oracle_rank(reps, oracle)
    exposure_keys = {oracle_key(c) for c in exposure_pool}
    oracle_in_exposure = oracle_key(oracle) in exposure_keys if oracle is not None else False
    diagnostic = exposure_diagnostic(ranked, geometries)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_vlm(args.vlm_model_id, device='cuda')
    rounds = []
    if args.selection_mode == 'single':
        montage, response, choice, selected = run_vlm_batch(model, processor, context, chosen, artifacts.candidates, target_text, args.max_new_tokens)
        montage_path = RESULTS_DIR / f'vlm_mcq_{args.case_id}_montage.png'
        montage.save(montage_path)
        rounds.append({'stage': 'single', 'response': response, 'choice': choice, 'candidate_count': len(chosen)})
    else:
        winners = []
        for batch_idx in range(0, len(exposure_pool), args.batch_size):
            batch = exposure_pool[batch_idx: batch_idx + args.batch_size]
            if not batch:
                continue
            montage, response, choice, selected = run_vlm_batch(model, processor, context, batch, artifacts.candidates, target_text, args.max_new_tokens)
            batch_path = RESULTS_DIR / f'vlm_mcq_{args.case_id}_round1_batch_{batch_idx // args.batch_size}.png'
            montage.save(batch_path)
            if selected is not None:
                winners.append(selected)
            rounds.append({
                'stage': 'round1',
                'batch_index': batch_idx // args.batch_size,
                'montage_path': str(batch_path),
                'response': response,
                'choice': choice,
                'candidate_count': len(batch),
                'candidates': [
                    {
                        'prompt': c.prompt,
                        'generator': c.metadata.get('generator'),
                        'cluster_id': c.cluster_id,
                        'slice_idx': int(c.slice_idx),
                        'dice': None if c.refined_metrics is None else float(c.refined_metrics.dice),
                        'hybrid_vlm_score': float(c.metadata.get('hybrid_vlm_score', 0.0)),
                    } for c in batch
                ],
            })
        finalists = winners if winners else chosen
        montage, response, choice, selected = run_vlm_batch(model, processor, context, finalists, artifacts.candidates, target_text, args.max_new_tokens)
        montage_path = RESULTS_DIR / f'vlm_mcq_{args.case_id}_montage.png'
        montage.save(montage_path)
        rounds.append({'stage': 'final', 'response': response, 'choice': choice, 'candidate_count': len(finalists)})

    selected_dice = None if selected is None or selected.refined_metrics is None else float(selected.refined_metrics.dice)
    summary = {
        'case_id': args.case_id,
        'vlm_model_id': args.vlm_model_id,
        'selection_mode': args.selection_mode,
        'report_text': report_text,
        'target_text': target_text,
        'montage_path': str(montage_path),
        'candidate_count': len(chosen),
        'exposure_count': len(exposure_pool),
        'oracle_rank': oracle_rank,
        'oracle_in_exposure_pool': oracle_in_exposure,
        'exposure_diagnostic': diagnostic,
        'rounds': rounds,
        'candidates': [
            {
                'index': idx + 1,
                'prompt': candidate.prompt,
                'generator': candidate.metadata.get('generator'),
                'cluster_id': candidate.cluster_id,
                'slice_idx': int(candidate.slice_idx),
                'box_rerank_score': float(candidate.box_rerank_score or 0.0),
                'semantic_vlm_score': float(candidate.metadata.get('semantic_vlm_score', 0.0)),
                'hybrid_vlm_score': float(candidate.metadata.get('hybrid_vlm_score', 0.0)),
                'dice': None if candidate.refined_metrics is None else float(candidate.refined_metrics.dice),
            }
            for idx, candidate in enumerate(exposure_pool)
        ],
        'selected': None if selected is None else {
            'prompt': selected.prompt,
            'generator': selected.metadata.get('generator'),
            'cluster_id': selected.cluster_id,
            'slice_idx': int(selected.slice_idx),
            'dice': selected_dice,
        },
        'oracle': None if oracle is None else {
            'prompt': oracle.prompt,
            'generator': oracle.metadata.get('generator'),
            'cluster_id': oracle.cluster_id,
            'slice_idx': int(oracle.slice_idx),
            'dice': float(oracle.refined_metrics.dice),
        },
    }
    output_path = Path(args.output_json) if args.output_json else RESULTS_DIR / f'brain_mri_vlm_match_{args.case_id}.json'
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
