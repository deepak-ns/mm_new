# â˜• Starbucks MDP Offer Optimizer

A complete **Markov Decision Process** project that learns the optimal offer policy for Starbucks customers using Value Iteration.

---

## Architecture

```
mdp_starbucks/
â”œâ”€â”€ data/                     â† auto-generated JSON files
â”‚   â”œâ”€â”€ portfolio.json         (offer catalog)
â”‚   â”œâ”€â”€ profile.json           (customer demographics)
â”‚   â””â”€â”€ transcript.json        (event log)
â”‚
â”œâ”€â”€ generate_data.py           â† synthetic Starbucks-like dataset
â”œâ”€â”€ data_loader.py             â† JSON â†’ DataFrames
â”œâ”€â”€ preprocess.py              â† flatten, merge, feature engineering
â”œâ”€â”€ state_engine.py            â† RFM-based user state classifier (5 states)
â”œâ”€â”€ transition_builder.py      â† P[s,a,s'] empirical transition matrix
â”œâ”€â”€ reward_engine.py           â† R[s,a] with bonuses & penalties
â”œâ”€â”€ baseline_policy.py         â† rule-based heuristic policy
â”œâ”€â”€ mdp_solver.py              â† Value Iteration (Bellman optimality)
â”œâ”€â”€ simulator.py               â† 90-day Monte Carlo cohort simulator
â”œâ”€â”€ metrics.py                 â† KPI computation & comparison
â”œâ”€â”€ app.py                     â† Streamlit dashboard
â””â”€â”€ requirements.txt
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
| **States S** | High Value, Offer Responsive, Low Value, At Risk, Inactive |
| **Actions A** | no_offer, bogo, discount, informational |
| **Transition P[s,a,s']** | Empirically estimated from event log + Laplace smoothing |
| **Reward R[s,a]** | Transaction revenue + offer completion bonus âˆ’ cost âˆ’ spam penalty |
| **Discount Î³** | 0.95 (configurable via sidebar) |
| **Solver** | Value Iteration (converges in ~300 iterations) |

---

## Key Results (500 users, 90 days)

| Metric | Baseline | MDP Optimal | Lift |
|--------|----------|-------------|------|
| Net Revenue | $85k | $141k | +66% |
| Total Revenue | $167k | $200k | +20% |
| Offer Cost | $82k | $58k | âˆ’29% |
| Avg Reward/step | 4.21 | 5.92 | +40% |
| Offer Completion | 50.9% | 70.0% | +37% |

---

## Dashboard Tabs

1. **Overview** â€” KPI cards + policy comparison table + action mix
2. **Revenue Simulation** â€” cumulative revenue, rolling net revenue, cost breakdown
3. **Policy & States** â€” optimal vs baseline, state distribution, value function
4. **Transition Matrix** â€” interactive heatmap per action + reward matrix
5. **MDP Internals** â€” convergence plot, Q-values, policy sensitivity to Î³
6. **Raw Data** â€” user features, simulation log, offer portfolio

---

## Extending the Project

- **Real data**: Replace `generate_data.py` with the actual Starbucks Kaggle dataset parser
- **More states**: Extend `state_engine.py` with clustering (KMeans/GMM)
- **Personalised rewards**: Make `reward_engine.py` use per-user income/spend features
- **Online learning**: Add a Q-learning update step in `simulator.py`


