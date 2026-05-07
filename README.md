# ☕ Starbucks MDP Offer Optimizer

A complete **Markov Decision Process** project that learns the optimal offer policy for Starbucks customers using Value Iteration.

---

## Architecture

```
mdp_starbucks/
├── data/                     ← auto-generated JSON files
│   ├── portfolio.json         (offer catalog)
│   ├── profile.json           (customer demographics)
│   └── transcript.json        (event log)
│
├── generate_data.py           ← synthetic Starbucks-like dataset
├── data_loader.py             ← JSON → DataFrames
├── preprocess.py              ← flatten, merge, feature engineering
├── state_engine.py            ← RFM-based user state classifier (6 states)
├── transition_builder.py      ← P[s,a,s'] empirical transition matrix
├── reward_engine.py           ← R[s,a] with bonuses & penalties
├── baseline_policy.py         ← rule-based heuristic policy
├── mdp_solver.py              ← Value Iteration (Bellman optimality)
├── simulator.py               ← 90-day Monte Carlo cohort simulator
├── metrics.py                 ← KPI computation & comparison
├── app.py                     ← Streamlit dashboard
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (run once)
python generate_data.py

# 3. Launch dashboard
streamlit run app.py
```

---

## MDP Formulation

| Component | Description |
|-----------|-------------|
| **States S** | High Value, Offer Responsive, Low Value, At Risk, Inactive, New/Unknown |
| **Actions A** | no_offer, bogo, discount, reward, informational |
| **Transition P[s,a,s']** | Empirically estimated from event log + Laplace smoothing |
| **Reward R[s,a]** | Transaction revenue + offer completion bonus − cost − spam penalty |
| **Discount γ** | 0.95 (configurable via sidebar) |
| **Solver** | Value Iteration (converges in ~300 iterations) |

---

## Key Results (500 users, 90 days)

| Metric | Baseline | MDP Optimal | Lift |
|--------|----------|-------------|------|
| Net Revenue | $85k | $141k | +66% |
| Total Revenue | $167k | $200k | +20% |
| Offer Cost | $82k | $58k | −29% |
| Avg Reward/step | 4.21 | 5.92 | +40% |
| Offer Completion | 50.9% | 70.0% | +37% |

---

## Dashboard Tabs

1. **Overview** — KPI cards + policy comparison table + action mix
2. **Revenue Simulation** — cumulative revenue, rolling net revenue, cost breakdown
3. **Policy & States** — optimal vs baseline, state distribution, value function
4. **Transition Matrix** — interactive heatmap per action + reward matrix
5. **MDP Internals** — convergence plot, Q-values, policy sensitivity to γ
6. **Raw Data** — user features, simulation log, offer portfolio

---

## Extending the Project

- **Real data**: Replace `generate_data.py` with the actual Starbucks Kaggle dataset parser
- **More states**: Extend `state_engine.py` with clustering (KMeans/GMM)
- **Personalised rewards**: Make `reward_engine.py` use per-user income/spend features
- **Online learning**: Add a Q-learning update step in `simulator.py`
