"""
simulator.py
------------
Monte Carlo simulator that runs a cohort of users through 90 simulated days
under a given policy, collecting per-step reward, state, and action data.

Each "day" one action is taken per user. The next state is sampled from P[s,a,·].
Transaction amounts are drawn from a state-conditional distribution.

Returns a structured list of episode records used by metrics.py.
"""

import numpy as np
import pandas as pd
from state_engine import N_STATES, STATES
from transition_builder import N_ACTIONS, ACTIONS
from reward_engine import instant_reward, ACTION_COST, STATE_ACTION_COMPLETION_PROB

# ── Per-state transaction amount distribution (mean, std) ─────────────────────

STATE_TX_PARAMS = {
    0: (12.0, 3.0),   # High Value
    1: ( 6.0, 2.5),   # Offer Responsive
    2: ( 3.5, 2.0),   # Low Value
    3: ( 5.0, 2.5),   # At Risk
    4: ( 1.0, 1.0),   # Inactive
    5: ( 2.0, 1.5),   # New / Unknown
}

# Probability that a user transacts on any given day (per state)
STATE_TX_PROB = {0: 0.70, 1: 0.55, 2: 0.40, 3: 0.45, 4: 0.10, 5: 0.30}

# Offer reward credited on completion
ACTION_OFFER_REWARD = {0: 0.0, 1: 5.0, 2: 2.5, 3: 8.0, 4: 0.0}


def simulate_cohort(
    initial_states: np.ndarray,
    policy:         np.ndarray,
    P:              np.ndarray,
    n_days:         int   = 90,
    seed:           int   = 0,
    policy_name:    str   = "policy",
) -> pd.DataFrame:
    """
    Simulate a cohort of users for n_days under a fixed policy.

    Parameters
    ----------
    initial_states : array of shape (n_users,) — starting state per user
    policy         : array of shape (N_STATES,) — action per state
    P              : transition matrix (N_STATES, N_ACTIONS, N_STATES)
    n_days         : simulation horizon
    seed           : random seed for reproducibility
    policy_name    : label attached to every row (used for comparison plots)

    Returns
    -------
    DataFrame with columns:
      user_id, day, state, action, next_state,
      tx_amount, offer_completed, reward, policy
    """
    rng     = np.random.default_rng(seed)
    n_users = len(initial_states)
    states  = initial_states.copy()

    records = []

    for day in range(n_days):
        for u in range(n_users):
            s      = int(states[u])
            a      = int(policy[s])

            # ── Sample transaction ─────────────────────────────────────────
            mu, sigma = STATE_TX_PARAMS[s]
            tx_prob   = STATE_TX_PROB[s]
            transacts = rng.random() < tx_prob
            amount    = max(0.0, rng.normal(mu, sigma)) if transacts else 0.0

            # ── Sample offer completion ────────────────────────────────────
            comp_prob       = STATE_ACTION_COMPLETION_PROB[s, a]
            offer_completed = (a > 0) and (rng.random() < comp_prob)
            offer_reward    = ACTION_OFFER_REWARD[a] if offer_completed else 0.0

            # ── Compute reward ─────────────────────────────────────────────
            r = instant_reward(s, a, amount, offer_completed, offer_reward)

            # ── Sample next state ──────────────────────────────────────────
            sp = int(rng.choice(N_STATES, p=P[s, a]))

            records.append({
                "user_id":        u,
                "day":            day,
                "state":          s,
                "state_name":     STATES[s],
                "action":         a,
                "action_name":    ACTIONS[a],
                "next_state":     sp,
                "tx_amount":      round(amount, 2),
                "offer_completed": offer_completed,
                "offer_reward":   round(offer_reward, 2),
                "offer_cost":     ACTION_COST[a],
                "reward":         round(r, 4),
                "policy":         policy_name,
            })

            states[u] = sp  # transition

    return pd.DataFrame(records)


def run_both_policies(
    initial_states:   np.ndarray,
    baseline_policy:  np.ndarray,
    optimal_policy:   np.ndarray,
    P:                np.ndarray,
    n_days:           int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: runs baseline then MDP policy on the same cohort.
    Returns (baseline_df, mdp_df).
    """
    baseline_df = simulate_cohort(
        initial_states, baseline_policy, P,
        n_days=n_days, seed=1, policy_name="Baseline"
    )
    mdp_df = simulate_cohort(
        initial_states, optimal_policy, P,
        n_days=n_days, seed=2, policy_name="MDP Optimal"
    )
    return baseline_df, mdp_df
