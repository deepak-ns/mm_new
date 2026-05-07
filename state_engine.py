"""
state_engine.py
---------------
Maps user-level feature vectors → discrete MDP states.

States (6):
  0  High Value       – high spender, active, completes offers
  1  Offer Responsive – moderate spender but strong offer engagement
  2  Low Value        – low spending, infrequent but present
  3  At Risk          – previously active, now receding
  4  Inactive         – very low activity across all signals
  5  New / Unknown    – not enough history to classify confidently

Each state has a clear business interpretation that drives the policy.
"""

import numpy as np
import pandas as pd

# ── State constants ────────────────────────────────────────────────────────────

STATES = {
    0: "High Value",
    1: "Offer Responsive",
    2: "Low Value",
    3: "At Risk",
    4: "Inactive",
    5: "New / Unknown",
}

N_STATES = len(STATES)

STATE_COLORS = {
    "High Value":       "#2ecc71",
    "Offer Responsive": "#3498db",
    "Low Value":        "#f39c12",
    "At Risk":          "#e74c3c",
    "Inactive":         "#95a5a6",
    "New / Unknown":    "#9b59b6",
}


# ── Rule-based classifier ──────────────────────────────────────────────────────

def classify_state(row: pd.Series) -> int:
    """
    Classify a single user (one row of user_features) into a state index.

    Decision rules use percentile thresholds derived from intuitive business
    logic rather than hard-coded dollar amounts so the engine scales to any
    dataset.
    """
    spend         = row.get("total_spend",       0)
    n_tx          = row.get("n_transactions",    0)
    completion    = row.get("completion_rate",   0)
    recency       = row.get("recency",           999)
    n_received    = row.get("n_offers_received", 0)
    view_rate     = row.get("view_rate",         0)

    # ── New / Unknown: almost no data ─────────────────────────────────────────
    if n_tx == 0 and n_received <= 1:
        return 5  # New / Unknown

    # ── Inactive: barely any activity at all ──────────────────────────────────
    if n_tx <= 1 and recency > 20:
        return 4  # Inactive

    # ── At Risk: used to transact but hasn't lately ───────────────────────────
    if n_tx >= 2 and recency > 12:
        return 3  # At Risk

    # ── High Value: top spenders who also engage with offers ──────────────────
    if spend > 30 and n_tx >= 5 and completion >= 0.25:
        return 0  # High Value

    # ── Offer Responsive: strong funnel engagement regardless of spend ─────────
    if (view_rate >= 0.4 or completion >= 0.20) and n_received >= 3:
        return 1  # Offer Responsive

    # ── Low Value: moderate/low spenders with weak offer engagement ───────────
    return 2  # Low Value


def assign_states(user_features: pd.DataFrame) -> pd.DataFrame:
    """
    Applies classify_state row-wise and adds a `state` (int) and
    `state_name` (str) column to user_features.
    Returns the enriched DataFrame.
    """
    df = user_features.copy()
    df["state"]      = df.apply(classify_state, axis=1)
    df["state_name"] = df["state"].map(STATES)
    return df


# ── Utility ────────────────────────────────────────────────────────────────────

def state_distribution(df_with_states: pd.DataFrame) -> pd.Series:
    """Returns value_counts of state_name for plotting."""
    return df_with_states["state_name"].value_counts()


def state_index(name: str) -> int:
    """Convert state name → index (raises KeyError if unknown)."""
    inv = {v: k for k, v in STATES.items()}
    return inv[name]
