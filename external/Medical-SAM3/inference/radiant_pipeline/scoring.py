from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .types import CandidateMetrics, ProposalCandidate


def bbox_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(ix1 - ix0, 0.0)
    ih = max(iy1 - iy0, 0.0)
    inter = iw * ih
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def lesion_size_prior(area_frac: float, target_mask_frac: float, strength: float) -> float:
    area_frac = max(area_frac, 1e-6)
    target_mask_frac = max(target_mask_frac, 1e-6)
    strength = max(strength, 0.0)
    return math.exp(-strength * abs(math.log(area_frac / target_mask_frac)))


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> CandidateMetrics:
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) else 1.0
    iou = inter / union if union else 1.0
    return CandidateMetrics(
        pred_pixels=pred_sum,
        gt_pixels=gt_sum,
        intersection=inter,
        dice=float(dice),
        iou=float(iou),
    )


def compute_text_rerank_score(candidate: ProposalCandidate) -> float:
    return (
        float(candidate.score) * float(candidate.size_prior or 0.0)
        + 0.02 * max(int(candidate.cluster_prompt_count) - 1, 0)
        + 0.03 * max(int(candidate.cluster_slice_count) - 1, 0)
    )


def compute_text_trust_bonus(
    candidate: ProposalCandidate,
    score_threshold: float,
    min_slices: int,
    min_prompts: int,
    boost: float,
) -> float:
    if candidate.refined_score is None:
        return 0.0
    if candidate.score < score_threshold:
        return 0.0
    if candidate.cluster_slice_count < min_slices:
        return 0.0
    if candidate.cluster_prompt_count < min_prompts:
        return 0.0

    score_margin = (candidate.score - score_threshold) / max(1.0 - score_threshold, 1e-6)
    support_scale = min(
        candidate.cluster_slice_count / max(min_slices, 1),
        candidate.cluster_prompt_count / max(min_prompts, 1),
        1.5,
    )
    return boost * max(score_margin, 0.0) * support_scale * float(candidate.refined_score)


def compute_box_rerank_score(
    candidate: ProposalCandidate,
    text_trust_score_threshold: float = 0.75,
    text_trust_min_slices: int = 4,
    text_trust_min_prompts: int = 4,
    text_trust_boost: float = 0.35,
) -> float:
    support_bonus = (
        0.005 * max(int(candidate.cluster_prompt_count) - 1, 0)
        + 0.01 * max(int(candidate.cluster_slice_count) - 1, 0)
    )
    text_trust_bonus = compute_text_trust_bonus(
        candidate,
        score_threshold=text_trust_score_threshold,
        min_slices=text_trust_min_slices,
        min_prompts=text_trust_min_prompts,
        boost=text_trust_boost,
    )
    candidate.text_trust_bonus = text_trust_bonus

    if candidate.refined_score is None:
        return 0.15 * float(candidate.rerank_score or 0.0) * float(candidate.bbox_fill_ratio or 0.0)

    return (
        0.15 * float(candidate.rerank_score or 0.0)
        + 0.55
        * float(candidate.refined_score)
        * float(candidate.refined_size_prior or 0.0)
        * float(candidate.compactness_prior or 0.0)
        + 0.15
        * float(candidate.refined_bbox_stability_iou or 0.0)
        * float(candidate.compactness_prior or 0.0)
        + support_bonus
        + text_trust_bonus
    )
