"""
transition_builder.py
---------------------
Extracts (state, action, next_state) tuples from the event log and builds
the empirical transition probability matrix P[s][a][s'].

Action mapping (4 actions):
  0  no_offer
  1  bogo
  2  discount
  3  informational
"""

import numpy as np
import pandas as pd
from state_engine import N_STATES, classify_state

ACTIONS = {
    0: "no_offer",
    1: "bogo",
    2: "discount",
    3: "informational",
}

N_ACTIONS = len(ACTIONS)

OFFER_TYPE_TO_ACTION = {
    "bogo": 1,
    "discount": 2,
    "informational": 3,
}


def extract_transitions(
    merged: pd.DataFrame,
    user_features: pd.DataFrame,
) -> list[tuple[int, int, int]]:
    """
    Returns a list of (state, action, next_state) tuples.

    Deprecated offer types are ignored by the preprocessing layer, and this
    mapping falls back to no_offer for any unsupported offer type.
    """
    user_idx = user_features.set_index("person")
    transitions = []

    for person, group in merged.groupby("person"):
        if person not in user_idx.index:
            continue

        user_row = user_idx.loc[person]
        cur_state = int(user_row["state"]) if "state" in user_row.index else None
        if cur_state is None:
            continue

        group = group.sort_values("time")
        offer_events = group[group["event"] == "offer received"]

        if offer_events.empty:
            transitions.append((cur_state, 0, cur_state))
            continue

        for _, offer_row in offer_events.iterrows():
            otype = offer_row.get("offer_type", "")
            action = OFFER_TYPE_TO_ACTION.get(otype, 0)

            after = group[group["time"] > offer_row["time"]]
            if after.empty:
                next_state = cur_state
            else:
                tx_after = after[after["event"] == "transaction"]["amount"]
                cmp_after = after[after["event"] == "offer completed"]
                n_rx_after = len(after[after["event"] == "offer received"])

                n_tx_after = len(tx_after)
                spend_after = tx_after.sum()
                comp_after = len(cmp_after)
                n_rx_after = max(n_rx_after, 1)

                mini = {
                    "total_spend": spend_after + user_row.get("total_spend", 0) * 0.5,
                    "n_transactions": n_tx_after + user_row.get("n_transactions", 0) * 0.5,
                    "completion_rate": comp_after / n_rx_after,
                    "recency": max(0, user_row.get("recency", 0) - 3),
                    "n_offers_received": n_rx_after,
                    "view_rate": user_row.get("view_rate", 0),
                    "avg_transaction": user_row.get("avg_transaction", 0),
                }
                next_state = classify_state(pd.Series(mini))

            transitions.append((cur_state, action, next_state))

    return transitions


def _domain_prior() -> np.ndarray:
    """
    Domain-informed prior distribution over next states for each (state, action).

    States: 0=High Value, 1=Offer Responsive, 2=Low Value,
            3=At Risk, 4=Inactive
    Actions: 0=no_offer, 1=bogo, 2=discount, 3=informational
    """
    prior = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    prior[0, 0] = [0.84, 0.05, 0.05, 0.04, 0.02]
    prior[0, 1] = [0.86, 0.08, 0.03, 0.02, 0.01]
    prior[0, 2] = [0.83, 0.06, 0.06, 0.03, 0.02]
    prior[0, 3] = [0.78, 0.09, 0.07, 0.04, 0.02]

    prior[1, 0] = [0.06, 0.62, 0.20, 0.09, 0.03]
    prior[1, 1] = [0.22, 0.66, 0.08, 0.03, 0.01]
    prior[1, 2] = [0.16, 0.62, 0.15, 0.05, 0.02]
    prior[1, 3] = [0.09, 0.60, 0.22, 0.07, 0.02]

    prior[2, 0] = [0.02, 0.10, 0.62, 0.21, 0.05]
    prior[2, 1] = [0.05, 0.22, 0.57, 0.12, 0.04]
    prior[2, 2] = [0.05, 0.20, 0.60, 0.12, 0.03]
    prior[2, 3] = [0.02, 0.14, 0.64, 0.16, 0.04]

    prior[3, 0] = [0.02, 0.05, 0.16, 0.48, 0.29]
    prior[3, 1] = [0.05, 0.21, 0.31, 0.31, 0.12]
    prior[3, 2] = [0.04, 0.19, 0.33, 0.31, 0.13]
    prior[3, 3] = [0.02, 0.11, 0.22, 0.42, 0.23]

    prior[4, 0] = [0.01, 0.02, 0.05, 0.10, 0.82]
    prior[4, 1] = [0.02, 0.11, 0.24, 0.21, 0.42]
    prior[4, 2] = [0.02, 0.09, 0.22, 0.19, 0.48]
    prior[4, 3] = [0.01, 0.06, 0.12, 0.13, 0.68]

    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            row = prior[s, a]
            assert abs(row.sum() - 1.0) < 1e-6, (
                f"Prior row s={s},a={a} sums to {row.sum():.4f}, not 1.0"
            )

    return prior


def build_transition_matrix(
    transitions: list[tuple[int, int, int]],
    laplace_alpha: float = 5.0,
) -> np.ndarray:
    """
    Constructs P[s, a, s'] of shape (N_STATES, N_ACTIONS, N_STATES).
    """
    counts = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=float)
    for (s, a, sp) in transitions:
        counts[s, a, sp] += 1

    prior_counts = _domain_prior() * laplace_alpha
    posterior = counts + prior_counts
    row_sums = posterior.sum(axis=2, keepdims=True)
    return posterior / row_sums


def build_all(
    merged: pd.DataFrame,
    user_features: pd.DataFrame,
) -> tuple[np.ndarray, list]:
    """
    Full pipeline: extract transitions, then build matrix.
    Returns (P, transitions_list).
    """
    transitions = extract_transitions(merged, user_features)
    P = build_transition_matrix(transitions)
    return P, transitions
