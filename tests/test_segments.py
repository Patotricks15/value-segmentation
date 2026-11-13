import numpy as np
import pandas as pd
import pytest

from value_segments.data import clean_orders, make_windows, split_windows
from value_segments.segments import (
    SEGMENT_LABELS,
    assign_segments,
    fit_models,
    predict_alive,
    predict_clv,
    segment_profile,
)


def _scores() -> pd.DataFrame:
    # Four customers per quadrant, so both medians fall between groups.
    return pd.DataFrame(
        {
            "pred_clv": [100, 90, 80, 70, 5, 4, 3, 2],
            "p_alive": [0.95, 0.9, 0.2, 0.1, 0.95, 0.9, 0.2, 0.1],
        },
        index=[f"c{i}" for i in range(8)],
    )


def test_assign_segments_quadrants():
    seg = assign_segments(_scores())
    assert seg.loc["c0", "segment"] == "Stars"  # high value, active
    assert seg.loc["c2", "segment"] == "Loyal at risk"  # high value, churn-prone
    assert seg.loc["c4", "segment"] == "Growth"  # low value, active
    assert seg.loc["c6", "segment"] == "Low value, lapsing"


def test_assign_segments_medians_recorded():
    seg = assign_segments(_scores())
    assert seg.attrs["clv_median"] == pytest.approx(_scores()["pred_clv"].median())
    assert seg.attrs["churn_median"] == pytest.approx((1 - _scores()["p_alive"]).median())


def test_assign_segments_uses_strict_median():
    scores = pd.DataFrame({"pred_clv": [1.0, 2.0, 3.0], "p_alive": [0.5, 0.5, 0.5]})
    seg = assign_segments(scores)
    # median is 2.0; ties at the median are not "high value"
    assert list(seg["high_value"]) == [False, False, True]


def test_segment_profile_shares_sum_to_one():
    seg = assign_segments(_scores())
    hold = pd.Series([50.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], index=seg.index)
    prof = segment_profile(seg, hold)
    assert prof["share_of_customers"].sum() == pytest.approx(1.0)
    assert prof["share_of_holdout_revenue"].sum() == pytest.approx(1.0)
    assert set(prof["segment"]) == set(SEGMENT_LABELS.values())


def test_segment_profile_handles_zero_holdout_revenue():
    seg = assign_segments(_scores())
    prof = segment_profile(seg, pd.Series(dtype=float))
    assert (prof["share_of_holdout_revenue"] == 0.0).all()
    assert (prof["share_bought_in_holdout"] == 0.0).all()


def _toy_transactions(n_customers: int = 3000) -> pd.DataFrame:
    """Simulated BG/NBD purchase histories over 365 days, so the model has real dropout signal to fit."""
    rng = np.random.default_rng(0)
    r, alpha, a, b = 1.5, 30.0, 1.0, 3.0
    rows = []
    for cid in range(n_customers):
        lam = rng.gamma(shape=r, scale=1.0 / alpha)
        p_drop = rng.beta(a, b)
        spend = rng.gamma(shape=4.0, scale=25.0)
        t, alive = 0.0, True
        while alive:
            t += rng.exponential(1.0 / lam)
            if t >= 365:
                break
            rows.append(
                {
                    "customer_id": f"c{cid}",
                    "order_id": f"o{cid}-{len(rows)}",
                    "timestamp": pd.Timestamp("2017-01-01") + pd.Timedelta(days=float(t)),
                    "revenue": float(spend * rng.uniform(0.7, 1.3)),
                    "freight": 5.0,
                    "n_items": 1,
                }
            )
            alive = rng.random() > p_drop
    df = pd.DataFrame(rows)
    df["day"] = df["timestamp"].dt.normalize()
    return df


def test_models_use_observation_window_only():
    txns = _toy_transactions()
    w = make_windows("2017-01-01", t_obs_months=6, t_hold_months=3)
    obs, _ = split_windows(txns, w)
    models = fit_models(obs, w.obs_end)
    assert set(models.summary.index) == set(obs["customer_id"].unique())
    assert models.summary["T"].max() <= (w.obs_end - w.obs_start).days


def test_predictions_are_finite_non_negative_and_probabilities_bounded():
    txns = _toy_transactions()
    w = make_windows("2017-01-01", t_obs_months=6, t_hold_months=3)
    obs, _ = split_windows(txns, w)
    models = fit_models(obs, w.obs_end)
    customers = models.summary.index
    clv = predict_clv(models, customers, horizon_days=90)
    p = predict_alive(models, customers)
    assert np.all(np.isfinite(clv)) and np.all(clv >= 0)
    assert np.all((p >= 0) & (p <= 1))


def test_clean_orders_drops_non_positive_revenue():
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b"],
            "customer_id": ["x", "y"],
            "order_status": ["delivered", "delivered"],
            "order_purchase_timestamp": ["2017-02-01", "2017-02-02"],
        }
    )
    customers = pd.DataFrame({"customer_id": ["x", "y"], "customer_unique_id": ["ux", "uy"]})
    items = pd.DataFrame(
        {"order_id": ["a", "b"], "price": [10.0, 0.0], "freight_value": [1.0, 1.0]}
    )
    out = clean_orders(orders, customers, items)
    assert list(out["customer_id"]) == ["ux"]
