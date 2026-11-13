"""Loading and observation/holdout cuts for Olist Brazilian E-Commerce."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DELIVERED = "delivered"


@dataclass(frozen=True)
class Windows:
    """Observation window [obs_start, obs_end) and holdout window [obs_end, hold_end)."""

    obs_start: pd.Timestamp
    obs_end: pd.Timestamp
    hold_end: pd.Timestamp


def load_olist(data_dir: str | Path) -> pd.DataFrame:
    """One row per delivered order: unique customer, order time, price total, freight total, item count."""
    root = Path(data_dir)
    orders = pd.read_csv(
        root / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    )
    customers = pd.read_csv(root / "olist_customers_dataset.csv", usecols=["customer_id", "customer_unique_id"])
    items = pd.read_csv(
        root / "olist_order_items_dataset.csv", usecols=["order_id", "price", "freight_value"]
    )
    return clean_orders(orders, customers, items)


def clean_orders(orders: pd.DataFrame, customers: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Keep delivered orders only; revenue is item price (freight is reported separately)."""
    delivered = orders[orders["order_status"] == DELIVERED]
    per_order = items.groupby("order_id").agg(
        revenue=("price", "sum"),
        freight=("freight_value", "sum"),
        n_items=("price", "size"),
    )
    merged = delivered.merge(customers, on="customer_id", how="inner").merge(
        per_order, left_on="order_id", right_index=True, how="inner"
    )
    out = pd.DataFrame(
        {
            "customer_id": merged["customer_unique_id"].astype(str),
            "order_id": merged["order_id"],
            "timestamp": pd.to_datetime(merged["order_purchase_timestamp"]),
            "revenue": merged["revenue"].astype(float),
            "freight": merged["freight"].astype(float),
            "n_items": merged["n_items"].astype(int),
        }
    ).reset_index(drop=True)
    out = out[out["revenue"] > 0]
    out["day"] = out["timestamp"].dt.normalize()
    return out


def make_windows(obs_start: str, t_obs_months: int = 12, t_hold_months: int = 6) -> Windows:
    start = pd.Timestamp(obs_start).normalize().replace(day=1)
    obs_end = start + pd.DateOffset(months=t_obs_months)
    hold_end = obs_end + pd.DateOffset(months=t_hold_months)
    return Windows(obs_start=start, obs_end=obs_end, hold_end=hold_end)


def split_windows(df: pd.DataFrame, windows: Windows) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = df[(df["day"] >= windows.obs_start) & (df["day"] < windows.obs_end)]
    hold = df[(df["day"] >= windows.obs_end) & (df["day"] < windows.hold_end)]
    return obs, hold
