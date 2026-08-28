# Copyright (c) Megvii, Inc. and its affiliates.
# Confidence threshold analysis utilities inspired by D-FINE-seg's Validator.

import os
from loguru import logger

import numpy as np


def find_best_confidence_threshold(
    coco_eval,
    class_names,
    thresholds=None,
    save_dir=None,
    coco_gt=None,
    coco_dt=None,
):
    """Sweep confidence thresholds to find the best one by F1 score.

    Uses COCO evaluation results to compute precision/recall at each threshold
    by examining which detections survive after re-filtering by score.

    Args:
        coco_eval: pycocotools COCOeval or CocoEvalOpt object
        class_names: list of class name strings
        thresholds: list of float thresholds to sweep (default: 0.1 to 0.95 step 0.05)
        save_dir: optional directory to save threshold analysis plots
        coco_gt: optional COCO ground truth object (for fallback when evalImgs is None)
        coco_dt: optional COCO detections object (for fallback when evalImgs is None)

    Returns:
        dict with keys:
            - best_threshold: float, threshold with highest F1
            - best_f1: float, F1 at best threshold
            - thresholds: list of swept thresholds
            - precisions: list of precision values
            - recalls: list of recall values
            - f1s: list of F1 values
    """
    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(2, 20)]  # 0.10 to 0.95

    # Get evaluation data from COCO
    eval_imgs = getattr(coco_eval, "evalImgs", None)

    # If FastCocoEvalOp was used, self.evalImgs is None in Python to save memory.
    # Fallback: re-evaluate with standard Python COCOeval on current image & category IDs.
    if eval_imgs is None or len(eval_imgs) == 0:
        coco_gt = coco_gt or getattr(coco_eval, "cocoGt", None)
        coco_dt = coco_dt or getattr(coco_eval, "cocoDt", None)
        if coco_gt is not None and coco_dt is not None:
            try:
                from pycocotools.cocoeval import COCOeval as StandardCOCOeval

                std_eval = StandardCOCOeval(coco_gt, coco_dt, "bbox")
                std_eval.params.imgIds = list(coco_eval.params.imgIds)
                std_eval.params.catIds = list(coco_eval.params.catIds)
                std_eval.params.iouThrs = list(coco_eval.params.iouThrs)
                std_eval.params.maxDets = list(coco_eval.params.maxDets)
                std_eval.evaluate()
                eval_imgs = std_eval.evalImgs
            except Exception as e:
                logger.warning(f"Fallback COCO evaluation for threshold analysis failed: {e}")

    if eval_imgs is None or len(eval_imgs) == 0:
        logger.warning("No evaluation data available for confidence threshold analysis")
        return None

    # Filter eval_imgs to only area_idx == 0 ('all' area range) to prevent double counting
    params = getattr(coco_eval, "params", None)
    if params is not None:
        num_cats = len(getattr(params, "catIds", []))
        num_area = len(getattr(params, "areaRng", []))
        num_imgs = len(getattr(params, "imgIds", []))
        if num_cats > 0 and num_area > 0 and num_imgs > 0 and len(eval_imgs) == num_cats * num_area * num_imgs:
            filtered_eval_imgs = []
            for cat_i in range(num_cats):
                for img_i in range(num_imgs):
                    idx = cat_i * num_area * num_imgs + 0 * num_imgs + img_i
                    filtered_eval_imgs.append(eval_imgs[idx])
            eval_imgs = filtered_eval_imgs

    # Collect all detection scores and match info
    all_gt_count = 0
    per_threshold_tp = {t: 0 for t in thresholds}
    per_threshold_fp = {t: 0 for t in thresholds}

    # Parse evalImgs to extract dt scores and matches
    for eval_img in eval_imgs:
        if eval_img is None:
            continue

        dt_scores = eval_img.get("dtScores", [])
        dt_matches = eval_img.get("dtMatches", None)  # [T x D] array
        gt_ignore = eval_img.get("gtIgnore", [])

        # Count non-ignored GTs
        num_gt = sum(1 for ig in gt_ignore if not ig) if gt_ignore is not None else 0
        all_gt_count += num_gt

        if dt_matches is None or len(dt_scores) == 0:
            continue

        # dt_matches shape: [num_iou_thresholds x num_detections]
        # Use first IoU threshold (0.50) for matching
        if dt_matches.ndim == 1:
            matches_at_iou50 = dt_matches
        else:
            matches_at_iou50 = dt_matches[0]  # IoU = 0.50

        for det_idx, score in enumerate(dt_scores):
            if det_idx >= len(matches_at_iou50):
                break
            is_tp = matches_at_iou50[det_idx] > 0

            for thresh in thresholds:
                if score >= thresh:
                    if is_tp:
                        per_threshold_tp[thresh] += 1
                    else:
                        per_threshold_fp[thresh] += 1

    if all_gt_count == 0:
        logger.warning("No ground truth annotations found for threshold analysis")
        return None

    precisions = []
    recalls = []
    f1s = []

    for thresh in thresholds:
        tp = per_threshold_tp[thresh]
        fp = per_threshold_fp[thresh]
        fn = all_gt_count - tp

        precision = tp / max(tp + fp, 1e-6)
        recall = tp / max(tp + fn, 1e-6)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    best_idx = int(np.argmax(f1s))
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("CONFIDENCE THRESHOLD ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Best threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    logger.info(
        f"At best threshold: "
        f"Precision={precisions[best_idx]:.4f}, "
        f"Recall={recalls[best_idx]:.4f}"
    )
    logger.info("=" * 60)

    result = {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
    }

    # Save plots if directory specified
    if save_dir is not None:
        try:
            _save_threshold_plots(thresholds, precisions, recalls, f1s, best_idx, save_dir)
        except Exception as e:
            logger.warning(f"Failed to save threshold analysis plots: {e}")

    return result


def _save_threshold_plots(thresholds, precisions, recalls, f1s, best_idx, save_dir):
    """Save precision/recall/F1 vs threshold plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping threshold analysis plots")
        return

    os.makedirs(save_dir, exist_ok=True)

    # F1 vs Threshold plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(thresholds, f1s, "b-", linewidth=2, label="F1")
    ax.axvline(
        x=thresholds[best_idx],
        color="r",
        linestyle="--",
        label=f"Best: {thresholds[best_idx]:.2f} (F1={f1s[best_idx]:.4f})",
    )
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score vs Confidence Threshold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "f1_vs_threshold.png"), dpi=150)
    plt.close(fig)

    # Precision/Recall vs Threshold plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(thresholds, precisions, "g-", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls, "b-", linewidth=2, label="Recall")
    ax.plot(thresholds, f1s, "r--", linewidth=2, label="F1")
    ax.axvline(
        x=thresholds[best_idx],
        color="gray",
        linestyle=":",
        label=f"Best F1: {thresholds[best_idx]:.2f}",
    )
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Confidence Threshold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "pr_vs_threshold.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Threshold analysis plots saved to {save_dir}")
