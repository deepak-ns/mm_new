"""
metrics.py
----------
Computes business KPIs from simulation DataFrames produced by simulator.py.

Metrics:
  total_revenue     : sum of tx_amount across all users/days
  total_cost        : sum of offer_cost
  net_revenue       : total_revenue - total_cost
  retention_rate    : fraction of users active (tx_amount > 0) in last 7 days
  avg_reward        : mean per-step reward
  avg_daily_revenue : total_revenue / n_days
  offer_completion_rate: fraction of offer steps where offer_completed=True
  state_progression : how states shift from day 0 → day N (dict)
"""

import numpy as np
import pandas as pd
from state_engine import STATES, N_STATES


def compute_metrics(sim_df: pd.DataFrame, n_days: int = 90) -> dict:
    """
    Parameters
    ----------
    sim_df  : DataFrame from simulator.simulate_cohort()
    n_days  : simulation horizon (for daily averages)

    Returns
    -------
    dict of scalar KPI values
    """
    n_users = sim_df["user_id"].nunique()

    total_revenue = sim_df["tx_amount"].sum()
    total_cost    = sim_df["offer_cost"].sum()
    net_revenue   = total_revenue - total_cost
    avg_reward    = sim_df["reward"].mean()

    # Retention: % of users who transacted in the final 7 days
    final_days  = sim_df[sim_df["day"] >= (n_days - 7)]
    retained    = final_days.groupby("user_id")["tx_amount"].sum()
    retention_rate = (retained > 0).mean()

    # Average daily revenue
    avg_daily_revenue = total_revenue / n_days

    # Offer completion rate (among steps where an offer was sent)
    offer_steps = sim_df[sim_df["action"] > 0]
    offer_completion_rate = (
        offer_steps["offer_completed"].mean() if len(offer_steps) > 0 else 0.0
    )

    # Revenue per user
    revenue_per_user = total_revenue / n_users if n_users > 0 else 0.0

    return {
        "total_revenue":        round(total_revenue, 2),
        "total_cost":           round(total_cost, 2),
        "net_revenue":          round(net_revenue, 2),
        "avg_reward":           round(avg_reward, 4),
        "retention_rate":       round(retention_rate, 4),
        "avg_daily_revenue":    round(avg_daily_revenue, 2),
        "offer_completion_rate": round(offer_completion_rate, 4),
        "revenue_per_user":     round(revenue_per_user, 2),
        "n_users":              n_users,
        "n_days":               n_days,
    }


def compare_policies(
    baseline_df: pd.DataFrame,
    mdp_df:      pd.DataFrame,
    n_days:      int = 90,
) -> pd.DataFrame:
    """
    Returns a side-by-side metrics comparison DataFrame.
    """
    bm = compute_metrics(baseline_df, n_days)
    mm = compute_metrics(mdp_df,      n_days)

    rows = []
    for key in bm:
        if key in ("n_users", "n_days"):
            continue
        rows.append({
            "Metric":       key.replace("_", " ").title(),
            "Baseline":     bm[key],
            "MDP Optimal":  mm[key],
            "Improvement":  round(mm[key] - bm[key], 4),
            "Improvement %": round(
                (mm[key] - bm[key]) / abs(bm[key]) * 100 if bm[key] != 0 else 0, 2
            ),
        })
    return pd.DataFrame(rows)


def daily_revenue_series(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-day aggregated revenue & cost for time-series charts.
    """
    return (
        sim_df.groupby("day")
        .agg(
            revenue=("tx_amount",   "sum"),
            cost   =("offer_cost",  "sum"),
            reward =("reward",      "mean"),
        )
        .reset_index()
    )


def state_distribution_over_time(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the fraction of users in each state per day.
    """
    total_users = sim_df["user_id"].nunique()
    dist = (
        sim_df.groupby(["day", "state_name"])
        .size()
        .reset_index(name="count")
    )
    dist["fraction"] = dist["count"] / total_users
    return dist


def action_distribution(sim_df: pd.DataFrame) -> pd.Series:
    """Fraction of steps per action type."""
    return sim_df["action_name"].value_counts(normalize=True).round(4)
