"""
transition_builder.py
---------------------
Extracts (state, action, next_state) tuples from the event log and builds
the empirical transition probability matrix P[s][a][s'].

Action mapping (5 actions):
  0  no_offer
  1  bogo
  2  discount
  3  reward
  4  informational

Since the raw data contains offer_type strings, we map them to action indices.
The "next state" is estimated by:
  - Looking at the user's state *after* receiving + completing (or ignoring) an offer
  - Using a sliding 24-hour observation window

For (state, action) pairs with zero observations we apply Laplace smoothing
so every cell has a non-zero probability (required for convergence of Value Iteration).
"""

import numpy as np
import pandas as pd
from state_engine import N_STATES, classify_state, STATES

# ── Action constants ───────────────────────────────────────────────────────────

ACTIONS = {
    0: "no_offer",
    1: "bogo",
    2: "discount",
    3: "reward",
    4: "informational",
}

N_ACTIONS = len(ACTIONS)

OFFER_TYPE_TO_ACTION = {
    "bogo":          1,
    "discount":      2,
    "reward":        3,
    "informational": 4,
}


# ── Transition extraction ──────────────────────────────────────────────────────

def extract_transitions(
    merged: pd.DataFrame,
    user_features: pd.DataFrame,
) -> list[tuple[int, int, int]]:
    """
    Returns a list of (state, action, next_state) tuples.

    Logic:
      For each user, sort events by time.
      Whenever an offer is received → action = offer_type mapped action.
      The 'current state' is the user's overall state (static approximation).
      The 'next state' is estimated by re-classifying the user on only the
      events AFTER the offer was sent (forward-looking slice).
    """
    # Index user features by person for quick lookup
    user_idx = user_features.set_index("person")

    transitions = []

    for person, group in merged.groupby("person"):
        if person not in user_idx.index:
            continue

        user_row    = user_idx.loc[person]
        cur_state   = int(user_row["state"]) if "state" in user_row.index else None
        if cur_state is None:
            continue

        group = group.sort_values("time")
        offer_events = group[group["event"] == "offer received"]

        if offer_events.empty:
            # No offer → record a "no_offer" self-transition
            transitions.append((cur_state, 0, cur_state))
            continue

        for _, offer_row in offer_events.iterrows():
            # Determine action from offer type
            otype  = offer_row.get("offer_type", "")
            action = OFFER_TYPE_TO_ACTION.get(otype, 0)

            # Estimate next state from events after this offer
            after = group[group["time"] > offer_row["time"]]
            if after.empty:
                next_state = cur_state
            else:
                # Build a mini feature snapshot from post-offer activity
                tx_after = after[after["event"] == "transaction"]["amount"]
                cmp_after = after[after["event"] == "offer completed"]
                n_rx_after = len(after[after["event"] == "offer received"])

                n_tx_after    = len(tx_after)
                spend_after   = tx_after.sum()
                comp_after    = len(cmp_after)
                n_rx_after    = max(n_rx_after, 1)

                # Build a lightweight feature dict for re-classification
                mini = {
                    "total_spend":       spend_after + user_row.get("total_spend", 0) * 0.5,
                    "n_transactions":    n_tx_after  + user_row.get("n_transactions", 0) * 0.5,
                    "completion_rate":   comp_after / n_rx_after,
                    "recency":           max(0, user_row.get("recency", 0) - 3),
                    "n_offers_received": n_rx_after,
                    "view_rate":         user_row.get("view_rate", 0),
                    "avg_transaction":   user_row.get("avg_transaction", 0),
                }
                next_state = classify_state(pd.Series(mini))

            transitions.append((cur_state, action, next_state))

    return transitions


# ── Matrix builder ─────────────────────────────────────────────────────────────

def _domain_prior() -> np.ndarray:
    """
    Domain-informed prior distribution over next states for each (state, action).
    Shape: (N_STATES, N_ACTIONS, N_STATES)

    Encodes business intuition:
      - Good offers to Inactive/At Risk users should have some chance of
        moving them to Low Value or Offer Responsive (re-engagement).
      - Sending no_offer keeps users roughly in their current state.
      - High Value users are sticky regardless of action.

    States: 0=High Value, 1=Offer Responsive, 2=Low Value,
            3=At Risk, 4=Inactive, 5=New/Unknown
    Actions: 0=no_offer, 1=bogo, 2=discount, 3=reward, 4=informational
    """
    # Default: stay in current state (identity-ish prior)
    # Shape (N_STATES, N_ACTIONS, N_STATES)
    prior = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    # ── High Value (0) ─────────────────────────────────────────────────────
    # Very sticky; good offers keep them there, bad sends push to At Risk
    prior[0, 0] = [0.80, 0.05, 0.05, 0.05, 0.03, 0.02]  # no_offer
    prior[0, 1] = [0.82, 0.08, 0.04, 0.03, 0.02, 0.01]  # bogo
    prior[0, 2] = [0.80, 0.06, 0.06, 0.04, 0.02, 0.02]  # discount
    prior[0, 3] = [0.85, 0.07, 0.03, 0.03, 0.01, 0.01]  # reward  ← best
    prior[0, 4] = [0.75, 0.08, 0.07, 0.05, 0.03, 0.02]  # informational

    # ── Offer Responsive (1) ───────────────────────────────────────────────
    # Good offers can upgrade to High Value; ignoring them risks downgrade
    prior[1, 0] = [0.05, 0.60, 0.20, 0.10, 0.03, 0.02]  # no_offer
    prior[1, 1] = [0.20, 0.65, 0.10, 0.03, 0.01, 0.01]  # bogo     ← best
    prior[1, 2] = [0.15, 0.60, 0.15, 0.05, 0.03, 0.02]  # discount
    prior[1, 3] = [0.18, 0.62, 0.12, 0.04, 0.02, 0.02]  # reward
    prior[1, 4] = [0.08, 0.58, 0.22, 0.07, 0.03, 0.02]  # informational

    # ── Low Value (2) ──────────────────────────────────────────────────────
    # Offers can lift to Offer Responsive; without them drift to At Risk
    prior[2, 0] = [0.02, 0.10, 0.60, 0.20, 0.05, 0.03]  # no_offer
    prior[2, 1] = [0.05, 0.20, 0.55, 0.12, 0.05, 0.03]  # bogo
    prior[2, 2] = [0.05, 0.18, 0.58, 0.12, 0.04, 0.03]  # discount  ← best
    prior[2, 3] = [0.04, 0.15, 0.58, 0.14, 0.05, 0.04]  # reward
    prior[2, 4] = [0.02, 0.12, 0.62, 0.16, 0.05, 0.03]  # informational

    # ── At Risk (3) ────────────────────────────────────────────────────────
    # Re-engagement offers can push to Low Value or Offer Responsive
    # Without intervention they slide to Inactive
    prior[3, 0] = [0.02, 0.05, 0.15, 0.45, 0.28, 0.05]  # no_offer  ← bad
    prior[3, 1] = [0.05, 0.20, 0.30, 0.30, 0.12, 0.03]  # bogo      ← re-engage
    prior[3, 2] = [0.04, 0.18, 0.32, 0.30, 0.13, 0.03]  # discount  ← re-engage
    prior[3, 3] = [0.06, 0.15, 0.28, 0.32, 0.15, 0.04]  # reward
    prior[3, 4] = [0.02, 0.10, 0.20, 0.40, 0.22, 0.06]  # informational

    # ── Inactive (4) ───────────────────────────────────────────────────────
    # Hard to move; best case is drift to New/Unknown or Low Value
    # Aggressive offers (bogo/reward) have small but real chance of re-engage
    prior[4, 0] = [0.01, 0.02, 0.05, 0.10, 0.75, 0.07]  # no_offer  ← stays inactive
    prior[4, 1] = [0.02, 0.10, 0.22, 0.20, 0.38, 0.08]  # bogo      ← best re-engage
    prior[4, 2] = [0.02, 0.08, 0.20, 0.18, 0.44, 0.08]  # discount
    prior[4, 3] = [0.03, 0.08, 0.18, 0.18, 0.45, 0.08]  # reward
    prior[4, 4] = [0.01, 0.05, 0.10, 0.12, 0.62, 0.10]  # informational

    # ── New / Unknown (5) ──────────────────────────────────────────────────
    # First interactions define trajectory; informational works well here
    prior[5, 0] = [0.03, 0.05, 0.15, 0.05, 0.10, 0.62]  # no_offer  ← stays unknown
    prior[5, 1] = [0.05, 0.20, 0.25, 0.05, 0.05, 0.40]  # bogo
    prior[5, 2] = [0.04, 0.18, 0.28, 0.05, 0.05, 0.40]  # discount
    prior[5, 3] = [0.06, 0.15, 0.22, 0.05, 0.05, 0.47]  # reward
    prior[5, 4] = [0.04, 0.22, 0.28, 0.04, 0.04, 0.38]  # informational ← best

    # Sanity-check each row sums to 1
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            row = prior[s, a]
            assert abs(row.sum() - 1.0) < 1e-6, \
                f"Prior row s={s},a={a} sums to {row.sum():.4f}, not 1.0"

    return prior


def build_transition_matrix(
    transitions: list[tuple[int, int, int]],
    laplace_alpha: float = 5.0,
) -> np.ndarray:
    """
    Constructs P[s, a, s'] of shape (N_STATES, N_ACTIONS, N_STATES).

    Uses domain-informed Bayesian priors (instead of flat Laplace smoothing)
    so that sparse (state, action) pairs — like Inactive + bogo — still
    produce sensible re-engagement probabilities rather than a flat uniform.

    The prior acts as pseudo-counts: laplace_alpha controls how strongly
    the prior dominates when real data is scarce.
    """
    # Raw empirical counts from extracted transitions
    counts = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=float)
    for (s, a, sp) in transitions:
        counts[s, a, sp] += 1

    # Domain prior as pseudo-counts
    prior     = _domain_prior()
    # Scale prior: each (s,a) row gets laplace_alpha total pseudo-count weight
    prior_counts = prior * laplace_alpha   # shape (N_STATES, N_ACTIONS, N_STATES)

    # Posterior = empirical counts + prior pseudo-counts
    posterior = counts + prior_counts

    # Normalise each (s, a) row to sum to 1
    row_sums = posterior.sum(axis=2, keepdims=True)
    P = posterior / row_sums

    return P  # shape: (N_STATES, N_ACTIONS, N_STATES)


# ── Public entry point ─────────────────────────────────────────────────────────

def build_all(
    merged: pd.DataFrame,
    user_features: pd.DataFrame,
) -> tuple[np.ndarray, list]:
    """
    Full pipeline: extract transitions → build matrix.
    Returns (P, transitions_list).
    """
    transitions = extract_transitions(merged, user_features)
    P           = build_transition_matrix(transitions)
    return P, transitions