"""Sweep confidence thresholds and plot a precision-recall curve."""

import os
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from deepforest import utilities
from deepforest.evaluate import evaluate_boxes
from deepforest.main import deepforest

DEFAULT_THRESHOLDS: list[float] = np.linspace(0.0, 0.9, num=10).round(3).tolist()


def normalize_thresholds(thresholds: Sequence[float] | None) -> list[float]:
    """Return a sorted, deduplicated list of thresholds.

    Args:
        thresholds: User-supplied thresholds, or None to use defaults
            (0.0 to 0.9 in steps of 0.1).

    Returns:
        Sorted list of unique thresholds.
    """
    values = DEFAULT_THRESHOLDS if not thresholds else thresholds
    return sorted({round(float(t), 4) for t in values})


def run_validation(cfg: DictConfig, thresholds: Sequence[float]) -> pd.DataFrame:
    """Run predictions once at score_thresh=0 and evaluate at each threshold.

    Args:
        cfg: Hydra configuration object with validation settings.
        thresholds: Score thresholds to evaluate.

    Returns:
        DataFrame with columns score_thresh, box_precision, box_recall.
    """
    cfg.score_thresh = 0.0
    model = deepforest(config=cfg)

    predictions = model.predict_file(
        csv_file=cfg.validation.csv_file, root_dir=cfg.validation.root_dir
    )

    ground_df = utilities.read_file(
        cfg.validation.csv_file, root_dir=cfg.validation.root_dir
    )

    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        filtered = predictions[predictions.score >= threshold]
        results = evaluate_boxes(
            predictions=filtered,
            ground_df=ground_df,
            iou_threshold=cfg.validation.iou_threshold,
        )
        records.append(
            {
                "score_thresh": threshold,
                "box_precision": results.get("box_precision", np.nan),
                "box_recall": results.get("box_recall", np.nan),
            }
        )

    return pd.DataFrame(records)


def plot_pr_curve(
    df: pd.DataFrame, output_path: str, label_thresholds: bool = True
) -> None:
    """Plot a precision-recall curve and save to disk.

    Args:
        df: DataFrame with box_precision, box_recall, score_thresh columns.
        output_path: Path to save the PNG plot.
        label_thresholds: Whether to annotate each point with its threshold value.
    """
    plot_df = df.dropna(subset=["box_precision", "box_recall"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(plot_df["box_recall"], plot_df["box_precision"], marker="o")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall by score_thresh")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)

    if label_thresholds:
        for _, row in plot_df.iterrows():
            ax.annotate(
                f"{row['score_thresh']:.2f}",
                (row["box_recall"], row["box_precision"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sweep_scores(
    cfg: DictConfig,
    output_dir: str,
    thresholds: Sequence[float] | None = None,
    label_thresholds: bool = True,
) -> tuple[str, str]:
    """Sweep confidence thresholds and write a CSV and precision-recall plot.

    Args:
        cfg: Hydra configuration object. Must have validation.csv_file,
            validation.root_dir, and validation.iou_threshold set.
        output_dir: Directory where results will be written.
        thresholds: Score thresholds to evaluate. Defaults to 0.0-0.9 in
            steps of 0.1.
        label_thresholds: Whether to annotate the plot with threshold values.

    Returns:
        Tuple of (csv_path, plot_path).

    Raises:
        ValueError: If validation.csv_file or validation.root_dir are not set.
    """
    if cfg.validation.csv_file is None:
        raise ValueError("validation.csv_file must be set in the config")
    if cfg.validation.root_dir is None:
        raise ValueError("validation.root_dir must be set in the config")

    thresholds = normalize_thresholds(thresholds)
    os.makedirs(output_dir, exist_ok=True)

    results_df = run_validation(cfg, thresholds)

    csv_path = os.path.join(output_dir, "precision_recall_thresholds.csv")
    results_df.to_csv(csv_path, index=False)

    plot_path = os.path.join(output_dir, "precision_recall_curve.png")
    plot_pr_curve(results_df, plot_path, label_thresholds)

    return csv_path, plot_path
