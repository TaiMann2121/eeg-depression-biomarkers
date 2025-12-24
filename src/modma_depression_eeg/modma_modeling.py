"""
Modeling utilities for MODMA depression EEG project.

This module is intentionally lightweight:
- It assumes you already have a per-subject feature table (one row per subject)
  with a binary label column.
- Features are expected to include bandpower columns like:
    rbp_delta_EEG1, rbp_theta_EEG25, rbp_alpha_EEG46, rbp_beta_EEG62, rbp_gamma_EEG89, ...
  and optional asymmetry columns like:
    alpha_asym_log_right_minus_left, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


BANDS = ("delta", "theta", "alpha", "beta", "gamma")


@dataclass(frozen=True)
class CVResult:
    """Container for cross-validation scores."""
    scores: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        # sample std to match typical reporting
        return float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0


def make_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_logreg_pipeline(
    *,
    class_weight: str | dict | None = "balanced",
    penalty: str = "l2",
    solver: str = "liblinear",
    max_iter: int = 5000,
    random_state: int = 42,
) -> Pipeline:
    """Standard baseline: z-score + logistic regression."""
    clf = LogisticRegression(
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def split_xy(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    drop_cols: Sequence[str] = ("subject_id",),
) -> Tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise KeyError(f"label_col='{label_col}' not in df.columns")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [label_col])
    y = df[label_col].astype(int)
    return X, y


def get_feature_groups(columns: Iterable[str]) -> Dict[str, List[str]]:
    """
    Build feature groups based on naming conventions.
    - "rbp_<band>_" => bandpower features per band
    - "rbp_" => all bandpower features
    - "asym" in name OR name contains 'asym_' => asymmetry features
    """
    cols = list(columns)

    rbp_all = [c for c in cols if c.startswith("rbp_")]
    asym = [c for c in cols if ("asym" in c.lower())]

    per_band: Dict[str, List[str]] = {}
    for b in BANDS:
        per_band[b] = [c for c in rbp_all if f"rbp_{b}_" in c]

    groups: Dict[str, List[str]] = {
        "all": cols,
        "rbp_all": rbp_all,
        "asym": asym,
    }
    for b in BANDS:
        groups[f"rbp_{b}"] = per_band[b]

    return groups


def eval_auc(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: Optional[StratifiedKFold] = None,
    pipe: Optional[Pipeline] = None,
) -> CVResult:
    """Evaluate ROC AUC under cross-validation."""
    if cv is None:
        cv = make_cv()
    if pipe is None:
        pipe = make_logreg_pipeline()

    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return CVResult(scores=scores)


def run_feature_ablation(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    drop_cols: Sequence[str] = ("subject_id",),
    n_splits: int = 5,
    random_state: int = 42,
    pipe: Optional[Pipeline] = None,
) -> pd.DataFrame:
    """
    Evaluate AUC for:
      - all features
      - all bandpower (rbp_*)
      - asymmetry only (if present)
      - per-band (delta/theta/alpha/beta/gamma)
    Returns a DataFrame with mean/std and feature counts.
    """
    X, y = split_xy(df, label_col=label_col, drop_cols=drop_cols)
    groups = get_feature_groups(X.columns)

    cv = make_cv(n_splits=n_splits, random_state=random_state)
    if pipe is None:
        pipe = make_logreg_pipeline()

    rows = []
    # Order matters (nicer for plots)
    order = [
        ("All features", "all"),
        ("All bandpower (rbp_*)", "rbp_all"),
        ("Asymmetry only", "asym"),
        ("delta only", "rbp_delta"),
        ("theta only", "rbp_theta"),
        ("alpha only", "rbp_alpha"),
        ("beta only", "rbp_beta"),
        ("gamma only", "rbp_gamma"),
    ]

    for display, key in order:
        cols = groups.get(key, [])
        if not cols:
            # skip groups that don't exist in the dataset
            continue
        res = eval_auc(X[cols], y, cv=cv, pipe=pipe)
        rows.append(
            {
                "group": display,
                "key": key,
                "n_features": len(cols),
                "auc_mean": res.mean,
                "auc_std": res.std,
            }
        )

    out = pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    return out


def coef_stability(
    X: pd.DataFrame,
    y: pd.Series,
    cols: Sequence[str],
    *,
    cv: Optional[StratifiedKFold] = None,
    pipe: Optional[Pipeline] = None,
) -> pd.DataFrame:
    """
    Fit logistic regression on each fold and compute coefficient stability:
      mean_coef, std_coef, abs_mean, sign
    """
    if cv is None:
        cv = make_cv()
    if pipe is None:
        pipe = make_logreg_pipeline()

    coef_mat = []
    for train_idx, _test_idx in cv.split(X[cols], y):
        pipe.fit(X.iloc[train_idx][cols], y.iloc[train_idx])
        coefs = pipe.named_steps["clf"].coef_[0]
        coef_mat.append(coefs)

    coef_mat = np.vstack(coef_mat)
    coef_mean = coef_mat.mean(axis=0)
    coef_std = coef_mat.std(axis=0, ddof=1) if coef_mat.shape[0] > 1 else np.zeros_like(coef_mean)

    stab = (
        pd.DataFrame(
            {
                "mean_coef": coef_mean,
                "std_coef": coef_std,
            },
            index=list(cols),
        )
        .assign(abs_mean=lambda d: d["mean_coef"].abs())
        .assign(sign=lambda d: np.sign(d["mean_coef"]).astype(float))
        .sort_values("abs_mean", ascending=False)
    )
    return stab


def top_k_features(stability_df: pd.DataFrame, k: int = 10) -> List[str]:
    return stability_df.head(k).index.tolist()


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    cols: Sequence[str],
    *,
    cv: Optional[StratifiedKFold] = None,
    pipe: Optional[Pipeline] = None,
) -> CVResult:
    return eval_auc(X[list(cols)], y, cv=cv, pipe=pipe)
