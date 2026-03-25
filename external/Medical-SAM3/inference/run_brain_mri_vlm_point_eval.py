#!/usr/bin/env python3
"""Evaluate the montage VLM point-prompt branch across local brain MRI cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_brain_mri_vlm_point_experiment import (
    RESULTS_DIR,
    build_dense_caption,
    build_montage,
    dice_2d,
    load_case,
    load_optional_sequence,
    load_vlm,
    oracle_window,
    parse_slice_and_point,
    render_slice_with_guides,
    run_vlm,
    sample_slice_indices,
)
from sam3_inference import SAM3Model
from radiant_pipeline import StudyContext
from radiant_pipeline.brain_mri import BrainMriFindingExtractor
from radiant_pipeline.brain_mri_retrieval import BrainMriRetrievalLocalizer, BrainMriRetrievalLocalizerConfig
from radiant_pipeline.medsiglip_backend import MedSiglipBackendConfig, MedSiglipRetrievalBackend

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='BraTS-MEN-00307-000,BraTS-MEN-00488-000,BraTS-MEN-01205-000,BraTS-MEN-01280-000,BraTS-MEN-01429-000')
    parser.add_argument('--sequence', default='t1c')
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--coarse-stride', type=int, default=4)
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--num-slices', type=int, default=5)
    parser.add_argument('--grid-step', type=int, default=40)
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'checkpoint.pt'))
    parser.add_argument('--vlm-model-id', default='llava-hf/llava-onevision-qwen2-0.5b-ov-hf')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--use-dense-caption', action='store_true', default=True)
    parser.add_argument('--success-dice-threshold', type=float, default=0.5)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = [c.strip() for c in args.cases.split(',') if c.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_vlm(args.vlm_model_id, device='cuda')
    sam = SAM3Model(confidence_threshold=0.1, device='cuda', checkpoint_path=args.checkpoint)

    case_results = []
    success_count = 0
    for case_id in cases:
        img_nii, img, gt_mask, manifest = load_case(case_id, args.sequence)
        t1n = load_optional_sequence(case_id, 't1n.nii.gz')
        flair = load_optional_sequence(case_id, 't2f.nii.gz')
        report_text = str(manifest.get('groundTruth', {}).get('dominantFindingText', 'single dominant intracranial mass-like lesion'))
        spacing_xyz = tuple(float(x) for x in img_nii.header.get_zooms()[:3])
        target_text = build_dense_caption(case_id, gt_mask, img, t1n, flair, spacing_xyz) if args.use_dense_caption else report_text

        context = StudyContext(
            case_id=case_id,
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
                shortlist_size=args.shortlist_size,
                min_center_separation=3,
            ),
        )
        hypotheses = localizer.localize(context, target)
        top_hypothesis = hypotheses[0] if hypotheses else None
        slice_list = sample_slice_indices(top_hypothesis.slice_indices if top_hypothesis is not None else [img.shape[2] // 2], args.num_slices)

        labels_to_slices = {}
        panels = []
        panel_meta = []
        for idx, slice_idx in enumerate(slice_list):
            label = chr(ord('A') + idx)
            labels_to_slices[label] = int(slice_idx)
            slice_img = np.asarray(img[:, :, slice_idx], dtype=np.float32)
            panels.append(render_slice_with_guides(slice_img, label=label, slice_idx=int(slice_idx), step=args.grid_step))
            panel_meta.append({'label': label, 'slice_idx': int(slice_idx), 'gt_pixels': int(np.asarray(gt_mask[:, :, slice_idx], dtype=bool).sum())})

        montage = build_montage(panels)
        montage_path = RESULTS_DIR / f'vlm_point_eval_{case_id}.png'
        montage.save(montage_path)
        prompt = (
            'You are reviewing a montage of brain MRI slices labeled by letter, each with x and y guide lines. '
            f'Radiology description: {target_text} '
            'Choose the one slice that best shows the lesion and return the lesion center on that slice. '
            'Output only JSON in the form {"slice": "A", "lesion_center": [x, y]}. '
            'If the lesion is not visible on any shown slice, output None.'
        )
        response = run_vlm(model, processor, montage, prompt, args.max_new_tokens, device='cuda')
        slice_label, point_xy = parse_slice_and_point(response, labels_to_slices, img.shape[1], img.shape[0])
        selected_slice_idx = None if slice_label is None else labels_to_slices[slice_label]
        pred_dice = None
        point_inside_gt = None
        if selected_slice_idx is not None and point_xy is not None:
            slice_img = np.asarray(img[:, :, selected_slice_idx], dtype=np.float32)
            gt_slice = np.asarray(gt_mask[:, :, selected_slice_idx], dtype=bool)
            point_inside_gt = bool(gt_slice[point_xy[1], point_xy[0]])
            rgb = np.stack([slice_img, slice_img, slice_img], axis=-1)
            lo, hi = np.percentile(rgb, [0.5, 99.5])
            rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            rgb = (rgb * 255).astype(np.uint8)
            state = sam.encode_image(rgb)
            pred_mask = sam.predict_point(state, point_xy=point_xy, img_size=slice_img.shape)
            if pred_mask is not None:
                pred_dice = dice_2d(np.asarray(pred_mask[0], dtype=bool), gt_slice)
        if pred_dice is not None and pred_dice >= args.success_dice_threshold:
            success_count += 1
        case_results.append({
            'case_id': case_id,
            'localizer_top': None if top_hypothesis is None else {
                'score': float(top_hypothesis.score),
                'center_slice': int(top_hypothesis.center_slice),
                'slice_indices': [int(x) for x in top_hypothesis.slice_indices],
            },
            'slice_candidates': panel_meta,
            'montage_path': str(montage_path),
            'vlm_response': response,
            'selected_slice_label': slice_label,
            'selected_slice_idx': selected_slice_idx,
            'point_xy': None if point_xy is None else [int(point_xy[0]), int(point_xy[1])],
            'point_inside_gt': point_inside_gt,
            'point_prompt_dice': pred_dice,
        })

    summary = {
        'cases': case_results,
        'case_count': len(case_results),
        'success_dice_threshold': args.success_dice_threshold,
        'success_count': success_count,
        'success_rate': (success_count / len(case_results)) if case_results else 0.0,
        'vlm_model_id': args.vlm_model_id,
    }
    output_path = Path(args.output_json) if args.output_json else RESULTS_DIR / 'brain_mri_vlm_point_eval.json'
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
