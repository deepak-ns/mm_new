"""
reward_engine.py
----------------
Defines and computes the immediate reward function R(s, a).

Reward components:
  + Expected transaction revenue for state s
  + Bonus if the offer is typically completed
  - Cost of sending the offer
  - Penalty for spamming inactive users
"""

import numpy as np
from state_engine import N_STATES
from transition_builder import N_ACTIONS

STATE_BASE_REVENUE = {
    0: 12.0,  # High Value
    1: 6.0,   # Offer Responsive
    2: 3.5,   # Low Value
    3: 5.0,   # At Risk
    4: 1.0,   # Inactive
}

ACTION_COST = {
    0: 0.0,  # no_offer
    1: 2.0,  # bogo
    2: 1.5,  # discount
    3: 0.5,  # informational
}

ACTION_COMPLETION_BONUS = {
    0: 0.0,  # no_offer
    1: 4.0,  # bogo
    2: 2.5,  # discount
    3: 0.5,  # informational brand lift
}

STATE_ACTION_COMPLETION_PROB = np.array([
    # no  bogo disc info
    [0.0, 0.6, 0.5, 0.3],   # High Value
    [0.0, 0.7, 0.6, 0.4],   # Offer Responsive
    [0.0, 0.3, 0.4, 0.2],   # Low Value
    [0.0, 0.4, 0.5, 0.2],   # At Risk
    [0.0, 0.1, 0.2, 0.05],  # Inactive
])

SPAM_PENALTY = np.array([
    # no  bogo disc info
    [0.0, 0.0, 0.0, 0.0],  # High Value
    [0.0, 0.0, 0.0, 0.0],  # Offer Responsive
    [0.0, 0.5, 0.5, 0.1],  # Low Value
    [0.0, 0.3, 0.2, 0.1],  # At Risk
    [0.0, 1.5, 1.2, 0.3],  # Inactive
])


def build_reward_matrix() -> np.ndarray:
    """
    Computes R[s, a] = base_revenue[s]
                     + completion_prob[s,a] * completion_bonus[a]
                     - cost[a]
                     - spam_penalty[s,a]
    """
    R = np.zeros((N_STATES, N_ACTIONS))

    base_revenue = np.array([STATE_BASE_REVENUE[s] for s in range(N_STATES)])
    cost_vec = np.array([ACTION_COST[a] for a in range(N_ACTIONS)])
    completion_bonus = np.array([ACTION_COMPLETION_BONUS[a] for a in range(N_ACTIONS)])

    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            R[s, a] = (
                base_revenue[s]
                + STATE_ACTION_COMPLETION_PROB[s, a] * completion_bonus[a]
                - cost_vec[a]
                - SPAM_PENALTY[s, a]
            )

    return R


def instant_reward(
    state: int,
    action: int,
    amount: float = 0.0,
    offer_completed: bool = False,
    offer_reward: float = 0.0,
) -> float:
    """
    Used by the simulator for step-by-step reward calculation.
    """
    r = amount
    if offer_completed:
        r += offer_reward
    r -= ACTION_COST[action]
    r -= SPAM_PENALTY[state, action]
    return r
