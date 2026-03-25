#!/usr/bin/env python3
"""Run a VLM -> point prompt -> SAM3 experiment on a multi-slice localized montage."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))

from sam3_inference import SAM3Model
from run_brain_mri_vlm_match_experiment import (
    build_dense_caption,
    load_case,
    load_optional_sequence,
    load_vlm,
    run_vlm,
)
from google import genai
from google.genai import types
from radiant_pipeline import StudyContext
from radiant_pipeline.brain_mri import BrainMriFindingExtractor
from radiant_pipeline.brain_mri_retrieval import BrainMriRetrievalLocalizer, BrainMriRetrievalLocalizerConfig
from radiant_pipeline.medsiglip_backend import MedSiglipBackendConfig, MedSiglipRetrievalBackend

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / 'results'
SLICE_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', default='BraTS-MEN-01205-000')
    parser.add_argument('--sequence', default='t1c')
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--coarse-stride', type=int, default=4)
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--num-slices', type=int, default=5)
    parser.add_argument('--grid-step', type=int, default=32)
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'checkpoint.pt'))
    parser.add_argument('--vlm-provider', choices=('local','gemini'), default='local')
    parser.add_argument('--gemini-two-stage', action='store_true', default=False)
    parser.add_argument('--vlm-model-id', default='llava-hf/llava-onevision-qwen2-0.5b-ov-hf')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--use-dense-caption', action='store_true', default=True)
    parser.add_argument('--use-oracle-window', action='store_true', default=False)
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--max-localizer-hypotheses', type=int, default=1)
    parser.add_argument('--point-refine-radius', type=int, default=4)
    parser.add_argument('--point-refine-step', type=int, default=4)
    parser.add_argument('--global-scout-num-slices', type=int, default=5)
    parser.add_argument('--max-gemini-calls-per-case', type=int, default=4)
    parser.add_argument('--disable-global-scout', action='store_true', default=False)
    return parser.parse_args()


def render_slice_with_guides(slice_img: np.ndarray, label: str, slice_idx: int, step: int = 32) -> Image.Image:
    lo, hi = np.percentile(slice_img, [0.5, 99.5])
    normalized = np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    base = (normalized * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    image = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(image)
    h, w = slice_img.shape
    ruler_color = (60, 180, 255)
    tick_color = (255, 255, 0)
    # Light border ruler with ticks gives the VLM a reference frame without painting over pathology.
    draw.rectangle([0, 0, w - 1, h - 1], outline=ruler_color, width=1)
    for x in range(0, w, step):
        draw.line([(x, 0), (x, min(8, h - 1))], fill=ruler_color, width=1)
        draw.line([(x, max(0, h - 9)), (x, h - 1)], fill=ruler_color, width=1)
        if x + 2 < w - 24:
            draw.text((x + 2, 2), str(x), fill=tick_color)
    for y in range(0, h, step):
        draw.line([(0, y), (min(8, w - 1), y)], fill=ruler_color, width=1)
        draw.line([(max(0, w - 9), y), (w - 1, y)], fill=ruler_color, width=1)
        if y + 2 < h - 12:
            draw.text((2, y + 2), str(y), fill=tick_color)
    draw.rectangle([4, 4, 80, 32], fill=(0, 0, 0))
    draw.text((10, 9), f'{label} s{slice_idx}', fill=(255, 255, 0))
    return image


def build_montage(panels: list[Image.Image]) -> Image.Image:
    cols = 2 if len(panels) <= 4 else 3
    rows = math.ceil(len(panels) / cols)
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new('RGB', (cols * w, rows * h), color=(20, 20, 20))
    for idx, panel in enumerate(panels):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(panel, (x, y))
    return canvas


def sample_slice_indices(slice_indices: list[int], num_slices: int) -> list[int]:
    if not slice_indices:
        return []
    ordered = sorted(set(int(x) for x in slice_indices))
    if len(ordered) <= num_slices:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num_slices)
    chosen = []
    seen = set()
    for pos in positions:
        idx = ordered[int(round(float(pos)))]
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)
    for idx in ordered:
        if len(chosen) >= num_slices:
            break
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)
    return sorted(chosen)


def oracle_window(gt_mask: np.ndarray, num_slices: int) -> list[int]:
    nonzero = np.argwhere(gt_mask > 0)
    if len(nonzero) == 0:
        return [int(gt_mask.shape[2] // 2)]
    slices = sorted(set(int(x) for x in nonzero[:, 2].tolist()))
    return sample_slice_indices(slices, num_slices)


def sample_global_scout_indices(num_total_slices: int, num_slices: int) -> list[int]:
    if num_total_slices <= 0:
        return []
    if num_total_slices <= num_slices:
        return list(range(num_total_slices))
    positions = np.linspace(0.1 * (num_total_slices - 1), 0.9 * (num_total_slices - 1), num_slices)
    chosen = []
    seen = set()
    for pos in positions:
        idx = int(round(float(pos)))
        idx = max(0, min(num_total_slices - 1, idx))
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)
    return chosen


def parse_slice_and_point(text: str, labels_to_slices: dict[str, int], width: int, height: int):
    upper = text.upper()
    if 'NONE' in upper or 'NULL' in upper or '[-1, -1]' in upper or '[-1,-1]' in upper:
        return None, None
    label_match = re.search(r'"SLICE"\s*:\s*"?([A-Z])"?', upper)
    if not label_match:
        label_match = re.search(r'\b([A-Z])\b', upper)
    label = label_match.group(1) if label_match else None
    if label not in labels_to_slices:
        return None, None
    point_match = re.search(r'\[\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*\]', text)
    if not point_match:
        point_match = re.search(r'"x"\s*:\s*(-?\d{1,4}).*?"y"\s*:\s*(-?\d{1,4})', text, flags=re.S)
    if not point_match:
        nums = re.findall(r'-?\d{1,4}', text)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
        else:
            return label, None
    else:
        x, y = int(point_match.group(1)), int(point_match.group(2))
    if x < 0 or y < 0:
        return None, None
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return label, (x, y)




def run_gemini(
    image: Image.Image,
    prompt: str,
    model_id: str,
    *,
    call_counter: list[int] | None = None,
    max_calls: int | None = None,
) -> str:
    if call_counter is not None:
        if max_calls is not None and call_counter[0] >= max_calls:
            return 'None'
        call_counter[0] += 1
    client = genai.Client()
    response = client.models.generate_content(
        model=model_id,
        contents=[prompt, image],
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text or ''


def parse_slice_choice(text: str, labels_to_slices: dict[str, int]):
    upper = text.upper()
    if 'NONE' in upper or 'NULL' in upper:
        return None
    label_match = re.search(r'"SLICE"\s*:\s*"?([A-Z])"?', upper)
    if not label_match:
        label_match = re.search(r'\b([A-Z])\b', upper)
    label = label_match.group(1) if label_match else None
    return label if label in labels_to_slices else None


def parse_point_only(text: str, width: int, height: int):
    if 'NONE' in text.upper() or 'NULL' in text.upper() or '[-1, -1]' in text or '[-1,-1]' in text:
        return None
    point_match = re.search(r'\[\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*\]', text)
    if not point_match:
        point_match = re.search(r'"x"\s*:\s*(-?\d{1,4}).*?"y"\s*:\s*(-?\d{1,4})', text, flags=re.S)
    if not point_match:
        nums = re.findall(r'-?\d{1,4}', text)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
        else:
            return None
    else:
        x, y = int(point_match.group(1)), int(point_match.group(2))
    if x < 0 or y < 0:
        return None
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))

def parse_point_only_normalized(text: str, width: int, height: int):
    if 'NONE' in text.upper() or 'NULL' in text.upper() or '[-1, -1]' in text or '[-1,-1]' in text:
        return None
    point_match = re.search(r'\[\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*\]', text)
    if not point_match:
        point_match = re.search(r'"x"\s*:\s*(-?\d{1,4}).*?"y"\s*:\s*(-?\d{1,4})', text, flags=re.S)
    if not point_match:
        nums = re.findall(r'-?\d{1,4}', text)
        if len(nums) >= 2:
            x_norm, y_norm = int(nums[0]), int(nums[1])
        else:
            return None
    else:
        x_norm, y_norm = int(point_match.group(1)), int(point_match.group(2))
    if x_norm < 0 or y_norm < 0:
        return None
    x_norm = max(0, min(1000, x_norm))
    y_norm = max(0, min(1000, y_norm))
    x = int(round((x_norm / 1000.0) * (width - 1)))
    y = int(round((y_norm / 1000.0) * (height - 1)))
    return x, y


def dice_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 0.0 if denom == 0 else (2.0 * inter / denom)


def build_window_payload(
    case_id: str,
    img: np.ndarray,
    gt_mask: np.ndarray,
    slice_list: list[int],
    grid_step: int,
    suffix: str,
):
    labels_to_slices = {}
    panels = []
    panel_meta = []
    for idx, slice_idx in enumerate(slice_list):
        label = SLICE_LABELS[idx]
        labels_to_slices[label] = int(slice_idx)
        slice_img = np.asarray(img[:, :, slice_idx], dtype=np.float32)
        panels.append(render_slice_with_guides(slice_img, label=label, slice_idx=int(slice_idx), step=grid_step))
        panel_meta.append({'label': label, 'slice_idx': int(slice_idx), 'gt_pixels': int(np.asarray(gt_mask[:, :, slice_idx], dtype=bool).sum())})
    montage = build_montage(panels)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    montage_path = RESULTS_DIR / f'vlm_point_montage_{case_id}_{suffix}.png'
    montage.save(montage_path)
    return labels_to_slices, panel_meta, montage, montage_path


def build_localizer_summary(hypothesis):
    if hypothesis is None:
        return None
    return {
        'source': hypothesis.source,
        'score': float(hypothesis.score),
        'center_slice': int(hypothesis.center_slice),
        'slice_indices': [int(x) for x in hypothesis.slice_indices],
    }


def generate_refine_offsets(radius: int, step: int) -> list[tuple[int, int]]:
    step = max(1, int(step))
    radius = max(0, int(radius))
    axis = sorted(set([0] + list(range(-radius, radius + 1, step)) + [-radius, radius]))
    offsets = []
    for dx in axis:
        for dy in axis:
            offsets.append((int(dx), int(dy)))
    offsets = sorted(set(offsets), key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0]), abs(d[1])))
    return offsets


def refine_point_with_sam(
    sam: SAM3Model,
    state: dict,
    base_point: tuple[int, int],
    img_size: tuple[int, int],
    radius: int,
    step: int,
):
    img_h, img_w = img_size
    tested = []
    best = None
    seen = set()
    for dx, dy in generate_refine_offsets(radius, step):
        x = max(0, min(img_w - 1, base_point[0] + dx))
        y = max(0, min(img_h - 1, base_point[1] + dy))
        point_xy = (int(x), int(y))
        if point_xy in seen:
            continue
        seen.add(point_xy)
        candidates = sam.predict_point_candidates(state, point_xy=point_xy, img_size=img_size, top_k=1)
        if not candidates:
            tested.append({'point_xy': [point_xy[0], point_xy[1]], 'status': 'no_candidate'})
            continue
        candidate = candidates[0]
        distance = abs(point_xy[0] - base_point[0]) + abs(point_xy[1] - base_point[1])
        refine_score = float(candidate['score']) - 0.05 * math.log1p(max(int(candidate['area_px']), 1)) - 0.02 * float(distance)
        tested.append({
            'point_xy': [point_xy[0], point_xy[1]],
            'sam_score': float(candidate['score']),
            'area_px': int(candidate['area_px']),
            'distance_px': int(distance),
            'refine_score': float(refine_score),
        })
        if best is None or refine_score > best['refine_score']:
            best = {
                'point_xy': point_xy,
                'mask': np.asarray(candidate['mask'], dtype=bool),
                'sam_score': float(candidate['score']),
                'area_px': int(candidate['area_px']),
                'distance_px': int(distance),
                'refine_score': float(refine_score),
            }
    return best, tested


def run_two_stage_gemini_over_windows(
    *,
    case_id: str,
    img: np.ndarray,
    gt_mask: np.ndarray,
    target_text: str,
    hypotheses,
    num_slices: int,
    grid_step: int,
    model_id: str,
    max_hypotheses: int,
    call_counter: list[int] | None = None,
    max_calls: int | None = None,
):
    stage1_prompt = (
        'You are reviewing a montage of brain MRI slices labeled by letter. '
        f'Radiology description: {target_text} '
        'Choose the single slice that best shows the lesion. '
        'If the lesion is not visibly present in any shown slice, you MUST output None instead of guessing on healthy anatomy. '
        'Output only JSON in the form {"slice": "A"}. If no slice shows the lesion, output None.'
    )
    attempts = []
    chosen = None
    hypotheses = list(hypotheses[:max(1, max_hypotheses)])
    for hyp_rank, hypothesis in enumerate(hypotheses, start=1):
        slice_list = sample_slice_indices(hypothesis.slice_indices, num_slices)
        labels_to_slices, panel_meta, montage, montage_path = build_window_payload(
            case_id=case_id,
            img=img,
            gt_mask=gt_mask,
            slice_list=slice_list,
            grid_step=grid_step,
            suffix=f'h{hyp_rank}',
        )
        stage1_response = run_gemini(
            montage,
            stage1_prompt,
            model_id,
            call_counter=call_counter,
            max_calls=max_calls,
        )
        slice_label = parse_slice_choice(stage1_response, labels_to_slices)
        attempt = {
            'hypothesis_rank': hyp_rank,
            'hypothesis': build_localizer_summary(hypothesis),
            'slice_candidates': panel_meta,
            'montage_path': str(montage_path),
            'stage1_response': stage1_response,
            'selected_slice_label': slice_label,
            'scout_type': 'localizer_window',
        }
        attempts.append(attempt)
        if slice_label is not None:
            chosen = {
                'slice_label': slice_label,
                'selected_slice_idx': labels_to_slices[slice_label],
                'panel_meta': panel_meta,
                'montage_path': str(montage_path),
                'hypothesis': hypothesis,
                'stage1_response': stage1_response,
                'attempts': attempts,
                'scout_type': 'localizer_window',
            }
            break
    if chosen is None:
        chosen = {
            'slice_label': None,
            'selected_slice_idx': None,
            'panel_meta': attempts[-1]['slice_candidates'] if attempts else [],
            'montage_path': attempts[-1]['montage_path'] if attempts else None,
            'hypothesis': hypotheses[0] if hypotheses else None,
            'stage1_response': attempts[-1]['stage1_response'] if attempts else 'None',
            'attempts': attempts,
            'scout_type': 'localizer_window',
        }
    return chosen


def main() -> int:
    args = parse_args()
    img_nii, img, gt_mask, manifest = load_case(args.case_id, args.sequence)
    t1n = load_optional_sequence(args.case_id, 't1n.nii.gz')
    flair = load_optional_sequence(args.case_id, 't2f.nii.gz')
    report_text = str(manifest.get('groundTruth', {}).get('dominantFindingText', 'single dominant intracranial mass-like lesion'))
    spacing_xyz = tuple(float(x) for x in img_nii.header.get_zooms()[:3])
    target_text = build_dense_caption(args.case_id, gt_mask, img, t1n, flair, spacing_xyz) if args.use_dense_caption else report_text

    context = StudyContext(
        case_id=args.case_id,
        modality='brain_mri',
        sequence=args.sequence,
        image_volume=img,
        report_text=report_text,
        metadata={'ground_truth_mask': gt_mask.astype('uint8'), 'finding_text': report_text, 'support_status': 'supported'},
    )

    finder = BrainMriFindingExtractor()
    target = finder.extract(context)
    backend = MedSiglipRetrievalBackend(MedSiglipBackendConfig(device='cuda'))
    localizer = BrainMriRetrievalLocalizer(
        backend=backend,
        config=BrainMriRetrievalLocalizerConfig(
            slab_depth=max(3, 2 * args.slice_radius + 1),
            slab_stride=max(1, args.coarse_stride // 2),
            shortlist_size=max(args.shortlist_size, args.max_localizer_hypotheses),
            min_center_separation=3,
        ),
    )
    hypotheses = localizer.localize(context, target)
    top_hypothesis = hypotheses[0] if hypotheses else None
    gemini_call_count = [0]

    window_attempts = []
    scout_fallback_used = False
    raw_point_xy = None
    raw_point_inside_gt = None
    point_refine_summary = None
    response = None
    slice_label = None
    selected_slice_idx = None
    point_xy = None
    panel_meta = []
    montage_path = None
    slice_source = 'localizer_top_window'

    if args.use_oracle_window:
        slice_list = oracle_window(gt_mask, args.num_slices)
        slice_source = 'oracle_window'
        labels_to_slices, panel_meta, montage, montage_path_obj = build_window_payload(
            case_id=args.case_id,
            img=img,
            gt_mask=gt_mask,
            slice_list=slice_list,
            grid_step=args.grid_step,
            suffix='oracle',
        )
        montage_path = str(montage_path_obj)
        prompt = (
            'You are reviewing a montage of brain MRI slices labeled by letter, each with x and y guide lines. '
            f'Radiology description: {target_text} '
            'Choose the one slice that best shows the lesion and return a point inside the visibly brightest enhancing core of the lesion on that slice, not a rough outer center. '
            'Prefer the enhancing core over surrounding edema, ventricle-like spaces, or broad mass effect. '
            'Output only JSON in the form {"slice": "A", "lesion_center": [x, y]}. '
            'If the lesion is not visible on any shown slice, output None.'
        )
        if args.vlm_provider == 'gemini':
            response = run_gemini(montage, prompt, args.vlm_model_id, call_counter=gemini_call_count, max_calls=args.max_gemini_calls_per_case)
        else:
            model, processor = load_vlm(args.vlm_model_id, device='cuda')
            response = run_vlm(model, processor, montage, prompt, args.max_new_tokens, device='cuda')
        slice_label, point_xy = parse_slice_and_point(response, labels_to_slices, img.shape[1], img.shape[0])
        selected_slice_idx = None if slice_label is None else labels_to_slices[slice_label]
    elif args.vlm_provider == 'gemini' and args.gemini_two_stage:
        selection = run_two_stage_gemini_over_windows(
            case_id=args.case_id,
            img=img,
            gt_mask=gt_mask,
            target_text=target_text,
            hypotheses=hypotheses,
            num_slices=args.num_slices,
            grid_step=args.grid_step,
            model_id=args.vlm_model_id,
            max_hypotheses=args.max_localizer_hypotheses,
            call_counter=gemini_call_count,
            max_calls=args.max_gemini_calls_per_case,
        )
        window_attempts = selection['attempts']
        panel_meta = selection['panel_meta']
        montage_path = selection['montage_path']
        slice_label = selection['slice_label']
        selected_slice_idx = selection['selected_slice_idx']
        slice_source = 'localizer_hypothesis_retry'
        response_payload = {'attempts': [
            {
                'hypothesis_rank': attempt['hypothesis_rank'],
                'selected_slice_label': attempt['selected_slice_label'],
                'stage1_response': attempt['stage1_response'],
                'montage_path': attempt['montage_path'],
                'scout_type': attempt.get('scout_type', 'localizer_window'),
            }
            for attempt in window_attempts
        ]}
        if slice_label is None and not args.disable_global_scout:
            scout_fallback_used = True
            global_slice_list = sample_global_scout_indices(img.shape[2], args.global_scout_num_slices)
            global_labels_to_slices, global_panel_meta, global_montage, global_montage_path_obj = build_window_payload(
                case_id=args.case_id,
                img=img,
                gt_mask=gt_mask,
                slice_list=global_slice_list,
                grid_step=args.grid_step,
                suffix='global_scout',
            )
            global_prompt = (
                'You are reviewing a coarse full-study scout montage of brain MRI slices labeled by letter. '
                f'Radiology description: {target_text} '
                'Choose the single slice that best shows the lesion anywhere in the study. '
                'If the lesion is not visibly present in any shown slice, you MUST output None instead of guessing on healthy anatomy. '
                'Output only JSON in the form {"slice": "A"}. If no slice shows the lesion, output None.'
            )
            global_stage1_response = run_gemini(
                global_montage,
                global_prompt,
                args.vlm_model_id,
                call_counter=gemini_call_count,
                max_calls=args.max_gemini_calls_per_case,
            )
            global_slice_label = parse_slice_choice(global_stage1_response, global_labels_to_slices)
            global_attempt = {
                'hypothesis_rank': None,
                'hypothesis': None,
                'slice_candidates': global_panel_meta,
                'montage_path': str(global_montage_path_obj),
                'stage1_response': global_stage1_response,
                'selected_slice_label': global_slice_label,
                'scout_type': 'global_scout',
            }
            window_attempts.append(global_attempt)
            response_payload['attempts'].append({
                'hypothesis_rank': None,
                'selected_slice_label': global_slice_label,
                'stage1_response': global_stage1_response,
                'montage_path': str(global_montage_path_obj),
                'scout_type': 'global_scout',
            })
            if global_slice_label is not None:
                slice_label = global_slice_label
                selected_slice_idx = global_labels_to_slices[global_slice_label]
                panel_meta = global_panel_meta
                montage_path = str(global_montage_path_obj)
                slice_source = 'global_scout_fallback'
        if slice_label is not None:
            single_slice = np.asarray(img[:, :, selected_slice_idx], dtype=np.float32)
            single_panel = render_slice_with_guides(single_slice, label=slice_label, slice_idx=int(selected_slice_idx), step=args.grid_step)
            stage2_prompt = (
                'You previously selected this brain MRI slice as the best view of the lesion. '
                f'Radiology description: {target_text} '
                'Return a point inside the visibly brightest enhancing core of the lesion on this single full-resolution slice. '
                'Do not return a rough outer center. Avoid surrounding edema, ventricles, or broad mass effect. '
                'If no lesion is visible on this slice, you MUST output None instead of guessing. '
                'Output only JSON in the form {"lesion_center": [x, y]}. If no lesion is visible, output None.'
            )
            stage2_response = run_gemini(
                single_panel,
                stage2_prompt,
                args.vlm_model_id,
                call_counter=gemini_call_count,
                max_calls=args.max_gemini_calls_per_case,
            )
            raw_point_xy = parse_point_only(stage2_response, img.shape[1], img.shape[0])
            point_xy = raw_point_xy
            response_payload['stage2'] = stage2_response
        response = json.dumps(response_payload)
    else:
        slice_list = sample_slice_indices(top_hypothesis.slice_indices if top_hypothesis is not None else [img.shape[2] // 2], args.num_slices)
        labels_to_slices, panel_meta, montage, montage_path_obj = build_window_payload(
            case_id=args.case_id,
            img=img,
            gt_mask=gt_mask,
            slice_list=slice_list,
            grid_step=args.grid_step,
            suffix='top1',
        )
        montage_path = str(montage_path_obj)
        prompt = (
            'You are reviewing a montage of brain MRI slices labeled by letter, each with x and y guide lines. '
            f'Radiology description: {target_text} '
            'Choose the one slice that best shows the lesion and return a point inside the visibly brightest enhancing core of the lesion on that slice, not a rough outer center. '
            'Prefer the enhancing core over surrounding edema, ventricle-like spaces, or broad mass effect. '
            'Output only JSON in the form {"slice": "A", "lesion_center": [x, y]}. '
            'If the lesion is not visible on any shown slice, output None.'
        )
        if args.vlm_provider == 'gemini':
            response = run_gemini(montage, prompt, args.vlm_model_id, call_counter=gemini_call_count, max_calls=args.max_gemini_calls_per_case)
        else:
            model, processor = load_vlm(args.vlm_model_id, device='cuda')
            response = run_vlm(model, processor, montage, prompt, args.max_new_tokens, device='cuda')
        slice_label, point_xy = parse_slice_and_point(response, labels_to_slices, img.shape[1], img.shape[0])
        selected_slice_idx = None if slice_label is None else labels_to_slices[slice_label]
        raw_point_xy = point_xy

    pred_mask = None
    pred_dice = None
    point_inside_gt = None
    gt_pixels = None
    pred_pixels = None
    if selected_slice_idx is not None and point_xy is not None:
        slice_img = np.asarray(img[:, :, selected_slice_idx], dtype=np.float32)
        gt_slice = np.asarray(gt_mask[:, :, selected_slice_idx], dtype=bool)
        if raw_point_xy is not None:
            raw_point_inside_gt = bool(gt_slice[raw_point_xy[1], raw_point_xy[0]])
        gt_pixels = int(gt_slice.sum())
        sam = SAM3Model(confidence_threshold=0.1, device='cuda', checkpoint_path=args.checkpoint)
        rgb = np.stack([slice_img, slice_img, slice_img], axis=-1)
        lo, hi = np.percentile(rgb, [0.5, 99.5])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
        state = sam.encode_image(rgb)
        best_point, tested_points = refine_point_with_sam(
            sam=sam,
            state=state,
            base_point=point_xy,
            img_size=slice_img.shape,
            radius=args.point_refine_radius,
            step=args.point_refine_step,
        )
        if best_point is not None:
            point_xy = best_point['point_xy']
            pred_mask = np.asarray(best_point['mask'], dtype=bool)
            pred_pixels = int(pred_mask.sum())
            point_refine_summary = {
                'base_point_xy': None if raw_point_xy is None else [int(raw_point_xy[0]), int(raw_point_xy[1])],
                'selected_point_xy': [int(point_xy[0]), int(point_xy[1])],
                'selected_sam_score': float(best_point['sam_score']),
                'selected_area_px': int(best_point['area_px']),
                'selected_distance_px': int(best_point['distance_px']),
                'selected_refine_score': float(best_point['refine_score']),
                'num_tested_points': len(tested_points),
                'tested_points': tested_points,
            }
            point_inside_gt = bool(gt_slice[point_xy[1], point_xy[0]])
            pred_dice = dice_2d(pred_mask, gt_slice)

    summary = {
        'case_id': args.case_id,
        'report_text': report_text,
        'target_text': target_text,
        'slice_source': slice_source,
        'slice_candidates': panel_meta,
        'window_attempts': window_attempts,
        'localizer_top': build_localizer_summary(top_hypothesis),
        'localizer_hypotheses': [build_localizer_summary(h) for h in hypotheses[:max(1, args.max_localizer_hypotheses)]],
        'montage_path': montage_path,
        'vlm_model_id': args.vlm_model_id,
        'vlm_response': response,
        'selected_slice_label': slice_label,
        'selected_slice_idx': selected_slice_idx,
        'raw_point_xy': None if raw_point_xy is None else [int(raw_point_xy[0]), int(raw_point_xy[1])],
        'raw_point_inside_gt': raw_point_inside_gt,
        'point_xy': None if point_xy is None else [int(point_xy[0]), int(point_xy[1])],
        'point_refine_summary': point_refine_summary,
        'point_inside_gt': point_inside_gt,
        'gt_pixels': gt_pixels,
        'pred_pixels': pred_pixels,
        'point_prompt_dice': pred_dice,
        'gemini_call_count': gemini_call_count[0] if args.vlm_provider == 'gemini' else 0,
        'max_gemini_calls_per_case': args.max_gemini_calls_per_case if args.vlm_provider == 'gemini' else None,
        'scout_fallback_used': scout_fallback_used,
    }
    output_path = Path(args.output_json) if args.output_json else RESULTS_DIR / f'brain_mri_vlm_point_{args.case_id}.json'
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
