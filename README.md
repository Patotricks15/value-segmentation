# LT05 — Two-by-two customer segmentation (Olist)

**Question.** Which customers should get which level of investment?

This repo cuts predicted customer value and a churn proxy at their medians, on real Olist data, and checks each
segment against what those customers actually bought in the next six months. The main result is negative: on
Olist, the churn axis collapses, and the segments barely separate realised behaviour. The one segment that does
stand out is small (664 customers).

## Method (stated before results)

| Item | Choice |
|---|---|
| Data | Olist Brazilian E-Commerce, delivered orders, revenue = item price (freight excluded) |
| Customer id | `customer_unique_id` |
| Observation window | 2017-01-01 to 2018-01-01 (fits BG/NBD and Gamma-Gamma; only 2017 customers are segmented) |
| Holdout window | 2018-01-01 to 2018-07-01 (never seen by the models; used only for checks) |
| Axis 1: predicted CLV | BG/NBD expected purchases × Gamma-Gamma expected order value over the 6-month holdout horizon |
| Axis 2: churn proxy | `1 - P(alive)` from BG/NBD at the end of 2017 |
| Cut | Median on each axis; strictly above the median = "high" |
| Deviation | The plan allowed a churn probability from CH02 "if available". CH02 is not in this portfolio, so the churn proxy is BG/NBD P(alive) |

## Results (real Olist data, executed in `notebooks/lt05_value_segmentation.ipynb`)

Customers scored: 42,136 (all customers with a 2017 purchase). Medians: predicted 6-month revenue 1.51 BRL; churn proxy **0.000**.

| Segment | Customers | Share | Mean pred. CLV | Mean P(alive) | Share buying in H1 2018 | Mean H1 2018 revenue | Share of H1 2018 revenue |
|---|---|---|---|---|---|---|---|
| Stars (high value, low churn) | 20,380 | 48.4% | 2.21 | 1.00 | 1.26% | 1.61 | 45.9% |
| Loyal at risk (high value, high churn) | 664 | 1.6% | 10.00 | 0.11 | 6.33% | 10.83 | 10.1% |
| Growth (low value, low churn) | 21,046 | 49.9% | 1.11 | 1.00 | 1.14% | 1.49 | 44.0% |
| Low value, lapsing (low value, high churn) | 46 | 0.1% | 1.12 | 0.04 | 0.00% | 0.00 | 0.0% |

Checks on the holdout:

- Overall, 1.3% of 2017 customers bought again in H1 2018.
- Spearman correlation of predicted CLV with realised H1 2018 revenue: **0.022**. This is almost no signal.
- AUC of P(alive) for buying in the holdout, all customers: **0.469** (below chance, because of ties).
- Repeat buyers in 2017 are only **1.7%** of customers (710). Among them, the AUC of P(alive) is **0.675**, so there is some signal here.
- P(alive) equals 1.0 for **98.3%** of customers. These are essentially one-time buyers, for whom BG/NBD cannot tell churn apart.

## What this means

1. **The churn axis is degenerate on Olist.** The median is 0, so "high churn" means "any repeat-buyer behaviour
   that lowers P(alive)". That group is 710 customers. The grid is really a CLV median split among one-time buyers,
   plus a small repeat-buyer group.
2. **Stars and Growth are not different in what they do next.** Their holdout purchase rates are 1.26% and 1.14%,
   and their revenue shares are about 46% and 44%. Splitting them on predicted CLV does not identify who will buy.
3. **The one segment that stands out is "Loyal at risk".** It is 664 customers with a 6.3% holdout purchase rate,
   about 5× the base rate, and 10% of holdout revenue from 1.6% of customers. This is the repeat-buyer group, and its
   value is real. Its name is misleading, though: its low P(alive) is only a weak churn signal here.
4. **Predicted CLV is not a useful ranking on this data.** Spearman 0.02 means the BG/NBD + Gamma-Gamma ranking does
   not track 2018 revenue. Holdout revenue splits almost evenly between the high-CLV half (46%) and the low-CLV half
   (44%), so the ranking is not capturing who spends.

## Action table

The table below states what the segments would suggest if the grid were reliable. **The evidence here does not
support differential investment across the two "high-value" segments (Stars vs Growth).** Only the repeat-buyer
group shows a different outcome, so the actions below are conditional and should be read that way.

| Segment | Suggested action | Strength of evidence |
|---|---|---|
| Loyal at risk (664) | Retention outreach; these customers buy at about 5× the base rate | Moderate: real difference in holdout, but small group and one half-year window |
| Stars (20,380) | Keep standard service; no evidence they need more than Growth | Weak: no measurable gap with Growth in purchase rate |
| Growth (21,046) | Standard acquisition-to-second-purchase programme; the main gap is that 98% never return | Weak: the grid cannot identify which of them will return |
| Low value, lapsing (46) | No investment | Insufficient sample |

## Limitations

- The churn proxy is BG/NBD P(alive), not a churn model. It is almost constant on Olist, so the median cut is
  not informative. A churn model built for one-time buyers (for example, a second-purchase classifier using
  first-order features) would be the right next step, but it is out of scope for LT05.
- One six-month holdout window, one data source. No significance test was run on the segment differences.
- Revenue is item price. Freight is excluded, as in LT03.
- Olist's licence is CC BY-NC-SA 4.0 (non-commercial). See [data/README.md](data/README.md).

## Reproduce

```bash
export CLV_DATA_PATH=~/.local/share/clv-data/olist   # folder with the nine Olist CSVs
make install
make reproduce        # tests + executed notebook
```

## Files

- `src/value_segments/data.py`: loader and window cuts (same as LT03).
- `src/value_segments/segments.py`: BG/NBD + Gamma-Gamma fit, CLV and P(alive) prediction, median segmentation, profiles.
- `tests/test_segments.py`: 8 tests. Logic tests use simulated BG/NBD histories; the simulation has real dropout
  signal, so the model fits.
- `notebooks/lt05_value_segmentation.py` (jupytext) and `.ipynb` (executed).
- `reports/lt05_segment_profile.csv`, `reports/lt05_summary.json`: aggregates only. No customer-level rows.
