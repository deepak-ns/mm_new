"""
baseline_policy.py
------------------
Implements a deterministic rule-based policy used as a performance baseline
against the MDP optimal policy.

Rule logic:
  High Value       → reward        (keep VIP customers loyal)
  Offer Responsive → bogo          (double-down on engaged users)
  Low Value        → discount      (price incentive to spend more)
  At Risk          → discount      (win-back with a price signal)
  Inactive         → informational (soft re-engagement, low cost)
  New / Unknown    → informational (educate before committing spend)
"""

import numpy as np
from state_engine import STATES, N_STATES
from transition_builder import ACTIONS, N_ACTIONS

# ── Policy map: state index → action index ────────────────────────────────────

BASELINE_POLICY: dict[int, int] = {
    0: 3,  # High Value       → reward
    1: 1,  # Offer Responsive → bogo
    2: 2,  # Low Value        → discount
    3: 2,  # At Risk          → discount
    4: 4,  # Inactive         → informational
    5: 4,  # New / Unknown    → informational
}


def baseline_action(state: int) -> int:
    """
    Returns the deterministic action for a given state under the baseline policy.
    Falls back to no_offer (0) for any unrecognised state.
    """
    return BASELINE_POLICY.get(state, 0)


def baseline_policy_vector() -> np.ndarray:
    """
    Returns a 1-D array π of shape (N_STATES,) where π[s] = action index.
    Convenient for passing into the simulator and metrics.
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
