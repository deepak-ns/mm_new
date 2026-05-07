"""
state_engine.py
---------------
Maps user-level feature vectors to discrete MDP states.

States (5):
  0  High Value       - high spender, active, completes offers
  1  Offer Responsive - moderate spender but strong offer engagement
  2  Low Value        - low spending, infrequent but present
  3  At Risk          - previously active, now receding
  4  Inactive         - very low activity across all signals

Each state has a clear business interpretation that drives the policy.
"""

import pandas as pd

STATES = {
    0: "High Value",
    1: "Offer Responsive",
    2: "Low Value",
    3: "At Risk",
    4: "Inactive",
}

N_STATES = len(STATES)

STATE_COLORS = {
    "High Value":       "#2ecc71",
    "Offer Responsive": "#3498db",
    "Low Value":        "#f39c12",
    "At Risk":          "#e74c3c",
    "Inactive":         "#95a5a6",
}


def classify_state(row: pd.Series) -> int:
    """
    Classify a single user (one row of user_features) into a state index.

    Decision rules use intuitive business logic rather than hard-coded dollar
    amounts so the engine scales to any dataset.
    """
    spend      = row.get("total_spend", 0)
    n_tx       = row.get("n_transactions", 0)
    completion = row.get("completion_rate", 0)
    recency    = row.get("recency", 999)
    n_received = row.get("n_offers_received", 0)
    view_rate  = row.get("view_rate", 0)

    # Users with almost no history are now treated as inactive.
    if (n_tx == 0 and n_received <= 1) or (n_tx <= 1 and recency > 20):
        return 4

    if n_tx >= 2 and recency > 12:
        return 3

    if spend > 30 and n_tx >= 5 and completion >= 0.25:
        return 0

    if (view_rate >= 0.4 or completion >= 0.20) and n_received >= 3:
        return 1

    return 2


def assign_states(user_features: pd.DataFrame) -> pd.DataFrame:
    """
    Applies classify_state row-wise and adds a `state` (int) and
    `state_name` (str) column to user_features.
    """
    df = user_features.copy()
    df["state"] = df.apply(classify_state, axis=1)
    df["state_name"] = df["state"].map(STATES)
    return df


def state_distribution(df_with_states: pd.DataFrame) -> pd.Series:
    """Returns value_counts of state_name for plotting."""
    return df_with_states["state_name"].value_counts()


def state_index(name: str) -> int:
    """Convert state name to index (raises KeyError if unknown)."""
    inv = {v: k for k, v in STATES.items()}
    return inv[name]
