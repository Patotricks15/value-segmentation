"""Predicted CLV and churn proxy from BG/NBD + Gamma-Gamma, and the median two-by-two segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

SEGMENT_LABELS = {
    (True, True): "Stars",
    (True, False): "Loyal at risk",
    (False, True): "Growth",
    (False, False): "Low value, lapsing",
}


@dataclass
class FittedModels:
    bgnbd: BetaGeoFitter
    gg: GammaGammaFitter | None
    fallback_order_value: float
    summary: pd.DataFrame


def fit_models(obs: pd.DataFrame, obs_end: pd.Timestamp) -> FittedModels:
    """Fit BG/NBD on purchase days and Gamma-Gamma on repeat buyers, using the observation window only."""
    summ = summary_data_from_transaction_data(
        obs,
        "customer_id",
        "timestamp",
        monetary_value_col="revenue",
        observation_period_end=obs_end,
        freq="D",
    )
    bgnbd = BetaGeoFitter(penalizer_coef=0.0)
    bgnbd.fit(summ["frequency"], summ["recency"], summ["T"])
    repeat = summ[summ["frequency"] > 0]
    gg = None
    if len(repeat) > 0:
        gg = GammaGammaFitter(penalizer_coef=0.0)
        gg.fit(repeat["frequency"], repeat["monetary_value"])
    return FittedModels(
        bgnbd=bgnbd, gg=gg, fallback_order_value=float(obs["revenue"].mean()), summary=summ
    )


def predict_clv(models: FittedModels, customers: pd.Index, horizon_days: int) -> np.ndarray:
    """Expected revenue over the next `horizon_days`: expected purchases times expected order value."""
    summ = models.summary.reindex(customers)
    freq = summ["frequency"].to_numpy(float)
    rec = summ["recency"].to_numpy(float)
    T = summ["T"].to_numpy(float)
    purchases = np.asarray(
        models.bgnbd.conditional_expected_number_of_purchases_up_to_time(horizon_days, freq, rec, T),
        dtype=float,
    )
    spend = np.full(len(customers), models.fallback_order_value)
    repeat = freq > 0
    if models.gg is not None and repeat.any():
        spend[repeat] = np.asarray(
            models.gg.conditional_expected_average_profit(
                freq[repeat], summ["monetary_value"].to_numpy(float)[repeat]
            ),
            dtype=float,
        )
    out = purchases * spend
    if not np.all(np.isfinite(out)):
        raise ValueError("BG/NBD or Gamma-Gamma produced non-finite predictions")
    return out


def predict_alive(models: FittedModels, customers: pd.Index) -> np.ndarray:
    """BG/NBD probability that each customer is still active at the end of the observation window."""
    summ = models.summary.reindex(customers)
    p = models.bgnbd.conditional_probability_alive(
        summ["frequency"].to_numpy(float),
        summ["recency"].to_numpy(float),
        summ["T"].to_numpy(float),
    )
    return np.asarray(p, dtype=float)


def assign_segments(scores: pd.DataFrame) -> pd.DataFrame:
    """Cut predicted CLV and churn probability at their medians; label the four cells.

    `scores` has columns `pred_clv` and `p_alive`. Churn proxy is `1 - p_alive`.
    A customer is "high value" when `pred_clv` is strictly above the median, and "high churn risk" when
    `1 - p_alive` is strictly above its median. Ties at the median fall into the lower group.
    """
    out = scores.copy()
    out["churn_proxy"] = 1.0 - out["p_alive"]
    clv_median = float(out["pred_clv"].median())
    churn_median = float(out["churn_proxy"].median())
    out["high_value"] = out["pred_clv"] > clv_median
    out["high_churn"] = out["churn_proxy"] > churn_median
    # Quadrant label uses the "retain" orientation: high value and low churn = Stars.
    out["segment"] = [
        SEGMENT_LABELS[(bool(v), not bool(c))] for v, c in zip(out["high_value"], out["high_churn"])
    ]
    out.attrs["clv_median"] = clv_median
    out.attrs["churn_median"] = churn_median
    return out


def segment_profile(segmented: pd.DataFrame, holdout_revenue: pd.Series) -> pd.DataFrame:
    """Per-segment size, predicted and realised holdout outcomes."""
    df = segmented.join(holdout_revenue.rename("holdout_revenue"), how="left")
    df["holdout_revenue"] = df["holdout_revenue"].fillna(0.0)
    df["bought_in_holdout"] = df["holdout_revenue"] > 0
    total_hold = df["holdout_revenue"].sum()
    rows = []
    for name in SEGMENT_LABELS.values():
        g = df[df["segment"] == name]
        rows.append(
            {
                "segment": name,
                "customers": len(g),
                "share_of_customers": len(g) / len(df),
                "mean_pred_clv": g["pred_clv"].mean(),
                "mean_p_alive": g["p_alive"].mean(),
                "mean_holdout_revenue": g["holdout_revenue"].mean(),
                "share_bought_in_holdout": g["bought_in_holdout"].mean(),
                "share_of_holdout_revenue": g["holdout_revenue"].sum() / total_hold if total_hold > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)
