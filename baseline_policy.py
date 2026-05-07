"""
baseline_policy.py
------------------
Implements a deterministic rule-based policy used as a performance baseline
against the MDP optimal policy.

Rule logic:
  High Value       -> bogo
  Offer Responsive -> bogo
  Low Value        -> discount
  At Risk          -> discount
  Inactive         -> informational
"""

import numpy as np
from state_engine import STATES, N_STATES
from transition_builder import ACTIONS

BASELINE_POLICY: dict[int, int] = {
    0: 1,  # High Value       -> bogo
    1: 1,  # Offer Responsive -> bogo
    2: 2,  # Low Value        -> discount
    3: 2,  # At Risk          -> discount
    4: 3,  # Inactive         -> informational
}


def baseline_action(state: int) -> int:
    """
    Returns the deterministic action for a given state under the baseline policy.
    Falls back to no_offer (0) for any unrecognised state.
    """
    return BASELINE_POLICY.get(state, 0)


def baseline_policy_vector() -> np.ndarray:
    """
    Returns a 1-D array pi of shape (N_STATES,) where pi[s] = action index.
    """
    return np.array([BASELINE_POLICY[s] for s in range(N_STATES)], dtype=int)


def describe_baseline() -> str:
    """
    Returns a human-readable summary of the baseline policy.
    """
    lines = ["Baseline (rule-based) policy:"]
    lines.append(f"  {'State':<20} {'Action'}")
    lines.append("  " + "-" * 35)
    for s, a in BASELINE_POLICY.items():
        lines.append(f"  {STATES[s]:<20} {ACTIONS[a]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_baseline())
