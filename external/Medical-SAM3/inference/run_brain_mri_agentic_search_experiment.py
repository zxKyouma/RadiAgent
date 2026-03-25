#!/usr/bin/env python3
"""Run an agentic batch-search loop over the full candidate representative pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_brain_mri_vlm_match_experiment import (
    RESULTS_DIR,
    attach_ranking_scores,
    build_cluster_representatives,
    build_dense_caption,
    build_pipeline,
    find_oracle_rank,
    load_case,
    load_optional_sequence,
    load_vlm,
    oracle_key,
    parse_choice,
    render_candidate_panel,
    run_vlm,
    select_diverse_candidates,
)
from radiant_pipeline import StudyContext
from radiant_pipeline.medsiglip_backend import MedSiglipBackendConfig, MedSiglipRetrievalBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', default='BraTS-MEN-01205-000')
    parser.add_argument('--sequence', default='t1c')
    parser.add_argument('--slice-radius', type=int, default=2)
    parser.add_argument('--coarse-stride', type=int, default=4)
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--max-batches', type=int, default=0, help='0 means search all batches')
    parser.add_argument('--mmr-lambda', type=float, default=0.75)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--vlm-model-id', default='llava-hf/llava-onevision-qwen2-0.5b-ov-hf')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--use-dense-caption', action='store_true', default=True)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def build_montage(panels):
    from PIL import Image
    import math
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


def run_vlm_batch(model, processor, context, batch, all_candidates, target_text, max_new_tokens):
    panels = [render_candidate_panel(context, candidate, all_candidates, idx + 1) for idx, candidate in enumerate(batch)]
    montage = build_montage(panels)
    prompt = (
        'You are an expert radiology agent. '
        f'The report target is: {target_text} '
        'Inspect this batch of numbered candidate lesions. '
        'If one candidate matches the report, answer with the number first. '
        'If none match, answer 0 first. Then give a short reason.'
    )
    response = run_vlm(model, processor, montage, prompt, max_new_tokens, device='cuda')
    choice = parse_choice(response, len(batch))
    selected = None if choice in (None, 0) else batch[choice - 1]
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
        metadata={'ground_truth_mask': gt_mask.astype('uint8'), 'finding_text': report_text, 'support_status': 'supported'},
    )

    if args.checkpoint is None:
        args.checkpoint = str(Path(__file__).resolve().parents[3] / 'checkpoint.pt')
    pipeline = build_pipeline(args)
    artifacts = pipeline.run_detailed(context)
    reps = build_cluster_representatives(artifacts.candidates)
    backend = MedSiglipRetrievalBackend(MedSiglipBackendConfig(device='cuda'))
    geometries = attach_ranking_scores(backend, context, reps, artifacts.candidates, target_text)
    ranked = sorted(reps, key=lambda c: float(c.metadata.get('hybrid_vlm_score', 0.0)), reverse=True)
    diverse_order = select_diverse_candidates(ranked, geometries, len(ranked), args.mmr_lambda)

    oracle = max((c for c in artifacts.candidates if c.refined_metrics is not None), key=lambda c: float(c.refined_metrics.dice), default=None)
    oracle_rank = find_oracle_rank(reps, oracle)
    diverse_rank = None
    ok = oracle_key(oracle)
    if ok is not None:
        for idx, candidate in enumerate(diverse_order, 1):
            if oracle_key(candidate) == ok:
                diverse_rank = idx
                break

    batches = [diverse_order[i:i + args.batch_size] for i in range(0, len(diverse_order), args.batch_size)]
    if args.max_batches > 0:
        batches = batches[:args.max_batches]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_vlm(args.vlm_model_id, device='cuda')
    rounds = []
    selected = None
    for batch_idx, batch in enumerate(batches):
        montage, response, choice, picked = run_vlm_batch(model, processor, context, batch, artifacts.candidates, target_text, args.max_new_tokens)
        batch_path = RESULTS_DIR / f'agentic_vlm_{args.case_id}_batch_{batch_idx}.png'
        montage.save(batch_path)
        rounds.append({
            'batch_index': batch_idx,
            'montage_path': str(batch_path),
            'response': response,
            'choice': choice,
            'candidate_count': len(batch),
            'contains_oracle': any(oracle_key(c) == ok for c in batch) if ok is not None else False,
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
        if picked is not None:
            selected = picked
            break

    summary = {
        'case_id': args.case_id,
        'vlm_model_id': args.vlm_model_id,
        'report_text': report_text,
        'target_text': target_text,
        'total_representatives': len(reps),
        'oracle_rank': oracle_rank,
        'oracle_diverse_rank': diverse_rank,
        'batch_size': args.batch_size,
        'searched_batches': len(rounds),
        'max_batches': args.max_batches,
        'selected': None if selected is None else {
            'prompt': selected.prompt,
            'generator': selected.metadata.get('generator'),
            'cluster_id': selected.cluster_id,
            'slice_idx': int(selected.slice_idx),
            'dice': None if selected.refined_metrics is None else float(selected.refined_metrics.dice),
        },
        'oracle': None if oracle is None else {
            'prompt': oracle.prompt,
            'generator': oracle.metadata.get('generator'),
            'cluster_id': oracle.cluster_id,
            'slice_idx': int(oracle.slice_idx),
            'dice': float(oracle.refined_metrics.dice),
        },
        'rounds': rounds,
    }
    output_path = Path(args.output_json) if args.output_json else RESULTS_DIR / f'brain_mri_agentic_search_{args.case_id}.json'
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved summary to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
