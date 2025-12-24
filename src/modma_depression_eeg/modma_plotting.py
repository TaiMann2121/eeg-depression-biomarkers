"""
Plotting utilities for MODMA depression EEG project.

These functions are designed to be called from notebooks/scripts and optionally
save figures to disk (e.g., assets/figures/*.png).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _maybe_savefig(savepath: Optional[str | Path], dpi: int = 200) -> None:
    if savepath is None:
        return
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=dpi, bbox_inches="tight")


def plot_ablation_results(
    ablation_df: pd.DataFrame,
    *,
    title: str = "Feature ablation (ROC AUC)",
    savepath: Optional[str | Path] = None,
) -> None:
    """
    Bar chart of AUC mean +/- std for each feature group.
    Expects columns: group, auc_mean, auc_std
    """
    df = ablation_df.copy()
    # keep stable order (best on top) for readability
    df = df.sort_values("auc_mean", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(df["group"], df["auc_mean"], xerr=df["auc_std"])
    plt.xlabel("ROC AUC (mean ± std)")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.grid(True, axis="x", alpha=0.25)
    _maybe_savefig(savepath)
    plt.show()


def plot_top_coefficients(
    stability_df: pd.DataFrame,
    *,
    k: int = 10,
    title: str = "Top stable coefficients (logistic regression)",
    savepath: Optional[str | Path] = None,
) -> None:
    """
    Horizontal bar plot of top-k coefficients using signed mean_coef.
    Expects columns: mean_coef, std_coef, abs_mean
    """
    top = stability_df.head(k).copy()
    # plot from smallest abs_mean to largest so the biggest appears at bottom
    top = top.sort_values("abs_mean", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(top.index, top["mean_coef"], xerr=top["std_coef"])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Coefficient (mean ± std across folds)")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.25)
    _maybe_savefig(savepath)
    plt.show()
