"""
reward_engine.py
----------------
Defines and computes the immediate reward function R(s, a).

Reward components:
  +  Expected transaction revenue for state s
  +  Bonus if the offer is typically completed (completion_rate × offer_reward)
  -  Cost of sending the offer (difficulty / 2 as a proxy for marketing spend)
  -  Penalty for spamming inactive / new users

The reward matrix R[s, a] (shape N_STATES × N_ACTIONS) is pre-computed once
and passed into the MDP solver.
"""

import numpy as np
from state_engine import STATES, N_STATES
from transition_builder import ACTIONS, N_ACTIONS

# ── Per-state base revenue (expected transaction value) ────────────────────────
# These are business estimates; in production derive from user_features means.

STATE_BASE_REVENUE = {
    0: 12.0,   # High Value       – high expected transaction
    1:  6.0,   # Offer Responsive – moderate spend, triggered by offers
    2:  3.5,   # Low Value        – below-average spend
    3:  5.0,   # At Risk          – decent history but declining
    4:  1.0,   # Inactive         – very low chance of transacting
    5:  2.0,   # New / Unknown    – uncertain, moderate potential
}

# ── Offer cost (proxy for difficulty-based marketing spend) ───────────────────
# Maps action index → cost in dollars
ACTION_COST = {
    0: 0.0,   # no_offer       – zero cost
    1: 2.0,   # bogo           – give-away cost
    2: 1.5,   # discount       – margin loss
    3: 2.5,   # reward         – reward credit cost
    4: 0.5,   # informational  – low cost (email / push)
}

# ── Offer completion bonus ─────────────────────────────────────────────────────
# Expected bonus if the user completes the offer
ACTION_COMPLETION_BONUS = {
    0: 0.0,   # no_offer
    1: 4.0,   # bogo
    2: 2.5,   # discount
    3: 5.0,   # reward
    4: 0.5,   # informational (brand lift)
}

# ── Completion probability by (state, action) ─────────────────────────────────
# How likely a user in each state is to complete a given offer type
STATE_ACTION_COMPLETION_PROB = np.array([
    # no  bogo disc  rwd  info
    [0.0, 0.6, 0.5,  0.7, 0.3],  # High Value
    [0.0, 0.7, 0.6,  0.5, 0.4],  # Offer Responsive
    [0.0, 0.3, 0.4,  0.2, 0.2],  # Low Value
    [0.0, 0.4, 0.5,  0.3, 0.2],  # At Risk
    [0.0, 0.1, 0.2,  0.1, 0.05], # Inactive
    [0.0, 0.4, 0.3,  0.3, 0.25], # New / Unknown
])  # shape: (N_STATES, N_ACTIONS)

# ── Spam penalty: sending offers to unresponsive users ─────────────────────────
# Added negative signal to discourage wasteful sending
SPAM_PENALTY = np.array([
    # no  bogo disc  rwd  info
    [0.0, 0.0, 0.0,  0.0, 0.0],  # High Value       – no penalty
    [0.0, 0.0, 0.0,  0.0, 0.0],  # Offer Responsive – no penalty
    [0.0, 0.5, 0.5,  0.5, 0.1],  # Low Value        – light penalty
    [0.0, 0.3, 0.2,  0.4, 0.1],  # At Risk          – mild
    [0.0, 1.5, 1.2,  1.5, 0.3],  # Inactive         – heavy penalty
    [0.0, 0.2, 0.1,  0.2, 0.0],  # New / Unknown    – small
])


# ── Build reward matrix ────────────────────────────────────────────────────────

def build_reward_matrix() -> np.ndarray:
    """
    Computes R[s, a] = base_revenue[s]
                     + completion_prob[s,a] * completion_bonus[a]
                     - cost[a]
                     - spam_penalty[s,a]

    Returns array of shape (N_STATES, N_ACTIONS).
    """
    R = np.zeros((N_STATES, N_ACTIONS))

    base_revenue      = np.array([STATE_BASE_REVENUE[s] for s in range(N_STATES)])
    cost_vec          = np.array([ACTION_COST[a]            for a in range(N_ACTIONS)])
    completion_bonus  = np.array([ACTION_COMPLETION_BONUS[a] for a in range(N_ACTIONS)])

    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            R[s, a] = (
                base_revenue[s]
                + STATE_ACTION_COMPLETION_PROB[s, a] * completion_bonus[a]
                - cost_vec[a]
                - SPAM_PENALTY[s, a]
            )

    return R


# ── Instantaneous reward for a single (state, action, amount) observation ─────

def instant_reward(
    state:        int,
    action:       int,
    amount:       float = 0.0,
    offer_completed: bool = False,
    offer_reward: float = 0.0,
) -> float:
    """
    Used by the simulator for step-by-step reward calculation.

    Parameters
    ----------
    state           : current user state index
    action          : action taken
    amount          : transaction amount in this step (0 if no purchase)
    offer_completed : True if the user completed the offer this step
    offer_reward    : reward value credited on completion
    """
    r = amount
    if offer_completed:
        r += offer_reward
    r -= ACTION_COST[action]
    r -= SPAM_PENALTY[state, action]
    return r
