"""
mdp_solver.py
-------------
Implements the MDP Value Iteration algorithm to compute the optimal policy π*.

Bellman optimality equation:
    V*(s) = max_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V*(s') ]

Runs until the maximum change in value function drops below `epsilon`
(convergence criterion) or hits `max_iter` iterations.

Returns:
  - V*    : optimal value function, shape (N_STATES,)
  - π*    : optimal policy, shape (N_STATES,) — integer action indices
  - history: list of (iteration, max_delta) for convergence plotting
"""

import numpy as np
from state_engine import N_STATES, STATES
from transition_builder import N_ACTIONS, ACTIONS
from reward_engine import build_reward_matrix


def value_iteration(
    P:        np.ndarray,
    R:        np.ndarray,
    gamma:    float = 0.95,
    epsilon:  float = 1e-6,
    max_iter: int   = 10_000,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Value Iteration solver.

    Parameters
    ----------
    P        : Transition matrix, shape (N_STATES, N_ACTIONS, N_STATES)
    R        : Reward matrix,     shape (N_STATES, N_ACTIONS)
    gamma    : Discount factor  [0, 1)
    epsilon  : Convergence threshold (max Bellman error)
    max_iter : Safety cap on iterations

    Returns
    -------
    V       : Converged value function,  shape (N_STATES,)
    policy  : Optimal greedy policy,     shape (N_STATES,)
    history : List of (iter, max_delta) for diagnostics
    """
    assert P.shape == (N_STATES, N_ACTIONS, N_STATES), \
        f"P must be ({N_STATES},{N_ACTIONS},{N_STATES}), got {P.shape}"
    assert R.shape == (N_STATES, N_ACTIONS), \
        f"R must be ({N_STATES},{N_ACTIONS}), got {R.shape}"
    assert 0 <= gamma < 1, "gamma must be in [0, 1)"

    V       = np.zeros(N_STATES)          # initialise value function to zero
    history = []

    for iteration in range(1, max_iter + 1):
        V_old = V.copy()

        # Q(s, a) = R(s, a) + γ · Σ_{s'} P(s'|s,a) · V(s')
        # Shape: (N_STATES, N_ACTIONS)
        Q = R + gamma * np.einsum("san,n->sa", P, V)

        # Bellman update: V(s) = max_a Q(s, a)
        V = Q.max(axis=1)

        max_delta = np.abs(V - V_old).max()
        history.append((iteration, float(max_delta)))

        if max_delta < epsilon:
            break

    # Greedy policy: π*(s) = argmax_a Q(s, a)
    Q_final = R + gamma * np.einsum("san,n->sa", P, V)
    policy  = Q_final.argmax(axis=1)

    return V, policy, history


def policy_summary(policy: np.ndarray) -> str:
    """Human-readable policy table."""
    lines = ["MDP Optimal Policy:"]
    lines.append(f"  {'State':<20} {'Action':<15} {'State Index'}")
    lines.append("  " + "-" * 45)
    for s, a in enumerate(policy):
        lines.append(f"  {STATES[s]:<20} {ACTIONS[a]:<15} s={s}")
    return "\n".join(lines)


# ── Module self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with a uniform random transition matrix
    P_test = np.ones((N_STATES, N_ACTIONS, N_STATES)) / N_STATES
    R_test = build_reward_matrix()
    V, pi, hist = value_iteration(P_test, R_test)
    print(f"Converged in {len(hist)} iterations.")
    print(policy_summary(pi))
    print("\nValue function V*:")
    for s in range(N_STATES):
        print(f"  {STATES[s]:<20} V={V[s]:.4f}")
