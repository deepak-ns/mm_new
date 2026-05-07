"""
app.py
------
Streamlit dashboard for the MDP-Based Offer Optimization project.

Run with:
    streamlit run app.py

Pipeline (cached):
  1. Generate / load data
  2. Preprocess & featurise
  3. Assign states
  4. Build transition matrix
  5. Solve MDP (Value Iteration)
  6. Simulate baseline + MDP policy (90 days)
  7. Display metrics, charts, policy tables
"""

import os, sys, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Make sure data/ exists before anything imports data_loader ─────────────────
pathlib.Path("data").mkdir(exist_ok=True)

# ── Imports from project modules ───────────────────────────────────────────────
from generate_data    import make_portfolio, make_profile, make_transcript
from data_loader      import load_all
from preprocess       import merge_all, build_user_features
from state_engine     import assign_states, STATES, N_STATES, STATE_COLORS
from transition_builder import build_all, ACTIONS, N_ACTIONS
from reward_engine    import build_reward_matrix
from baseline_policy  import baseline_policy_vector, BASELINE_POLICY
from mdp_solver       import value_iteration, policy_summary
from simulator        import run_both_policies
from metrics          import (
    compare_policies, daily_revenue_series,
    state_distribution_over_time, action_distribution, compute_metrics
)

import json

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Starbucks MDP Offer Optimizer",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS — dark espresso theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #0f0d0b;
      color: #f5efe6;
  }
  h1, h2, h3 { font-family: 'Playfair Display', serif; }

  .stMetric {
      background: #1c1713;
      border: 1px solid #3d2e1e;
      border-radius: 12px;
      padding: 1rem 1.2rem;
  }
  .stMetric label { color: #a08060 !important; font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .stMetric [data-testid="metric-container"] > div:nth-child(2) { font-size: 1.8rem; font-weight: 600; color: #f5efe6; }

  [data-testid="stSidebar"] {
      background: #140f0a;
      border-right: 1px solid #3d2e1e;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
      color: #c8a97a;
  }

  .section-header {
      font-family: 'Playfair Display', serif;
      font-size: 1.4rem;
      color: #c8a97a;
      border-bottom: 1px solid #3d2e1e;
      padding-bottom: 0.4rem;
      margin-bottom: 1rem;
  }

  div[data-testid="stTabs"] button {
      color: #a08060;
      font-family: 'DM Sans', sans-serif;
  }
  div[data-testid="stTabs"] button[aria-selected="true"] {
      color: #f5efe6;
      border-bottom-color: #c8a97a;
  }

  .stDataFrame { background: #1c1713; }

  .badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 500;
  }

  /* Plotly chart background override */
  .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1c1713",
    font_color="#f5efe6",
    font_family="DM Sans",
    xaxis=dict(gridcolor="#2d2318", linecolor="#3d2e1e"),
    yaxis=dict(gridcolor="#2d2318", linecolor="#3d2e1e"),
)

PALETTE = ["#c8a97a", "#e8734a", "#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

# ══════════════════════════════════════════════════════════════════════════════
# Data generation helper
# ══════════════════════════════════════════════════════════════════════════════

def ensure_data(n_users: int):
    files = ["data/portfolio.json", "data/profile.json", "data/transcript.json"]
    if not all(pathlib.Path(f).exists() for f in files):
        portfolio  = make_portfolio()
        profiles   = make_profile(n=n_users)
        transcript = make_transcript(profiles, portfolio)
        with open("data/portfolio.json",  "w") as f: json.dump(portfolio,  f)
        with open("data/profile.json",    "w") as f: json.dump(profiles,   f)
        with open("data/transcript.json", "w") as f: json.dump(transcript, f)

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_pipeline(n_users: int, n_days: int, gamma: float):
    ensure_data(n_users)
    portfolio, profile, transcript = load_all()

    # Preprocess
    merged        = merge_all(portfolio, profile, transcript)
    user_features = build_user_features(merged)
    user_features = assign_states(user_features)

    # Transition matrix
    P, transitions = build_all(merged, user_features)

    # Reward matrix
    R = build_reward_matrix()

    # Solve MDP
    V, optimal_policy, vi_history = value_iteration(P, R, gamma=gamma)

    # Baseline policy vector
    baseline_pol = baseline_policy_vector()

    # Simulate
    initial_states = user_features["state"].values[:n_users]
    baseline_df, mdp_df = run_both_policies(
        initial_states, baseline_pol, optimal_policy, P, n_days=n_days
    )

    return {
        "portfolio":       portfolio,
        "user_features":   user_features,
        "P":               P,
        "R":               R,
        "V":               V,
        "optimal_policy":  optimal_policy,
        "baseline_policy": baseline_pol,
        "vi_history":      vi_history,
        "baseline_df":     baseline_df,
        "mdp_df":          mdp_df,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ☕ MDP Optimizer")
    st.markdown("---")
    n_users  = st.slider("Cohort size (users)",    200, 2000, 1000, step=100)
    n_days   = st.slider("Simulation horizon (days)", 30, 180, 90,  step=10)
    gamma    = st.slider("Discount factor γ",     0.80, 0.99, 0.95, step=0.01)
    st.markdown("---")
    run_btn  = st.button("🚀 Run Pipeline", width="stretch")
    st.markdown("---")
    st.markdown("""
**Actions**
- `no_offer` · `bogo` · `discount`
- `reward` · `informational`

**States**
- High Value · Offer Responsive
- Low Value · At Risk
- Inactive · New/Unknown
    """)

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='font-family:Playfair Display,serif; font-size:2.4rem; color:#c8a97a; margin-bottom:0'>
  Starbucks MDP Offer Optimizer
</h1>
<p style='color:#a08060; margin-top:4px; margin-bottom:1.5rem; font-size:0.95rem'>
  Markov Decision Process · Value Iteration · 90-Day Revenue Simulation
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Run pipeline
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Running pipeline…"):
    data = run_pipeline(n_users, n_days, gamma)

uf     = data["user_features"]
bdf    = data["baseline_df"]
mdf    = data["mdp_df"]
V      = data["V"]
pi     = data["optimal_policy"]
bpi    = data["baseline_policy"]
P      = data["P"]
R      = data["R"]
vi_hist = data["vi_history"]

bm = compute_metrics(bdf, n_days)
mm = compute_metrics(mdf, n_days)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📊 Overview",
    "📈 Revenue Simulation",
    "🗺️ Policy & States",
    "🔁 Transition Matrix",
    "⚙️ MDP Internals",
    "🗃️ Raw Data",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — Overview KPIs
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    kpi_items = [
        ("💰 Net Revenue (MDP)",   f"${mm['net_revenue']:,.0f}",   f"Baseline: ${bm['net_revenue']:,.0f}"),
        ("📈 Total Revenue (MDP)", f"${mm['total_revenue']:,.0f}", f"+{mm['total_revenue']-bm['total_revenue']:,.0f} vs baseline"),
        ("🎯 Retention Rate",      f"{mm['retention_rate']*100:.1f}%", f"Baseline: {bm['retention_rate']*100:.1f}%"),
        ("✅ Offer Completion",    f"{mm['offer_completion_rate']*100:.1f}%", f"Baseline: {bm['offer_completion_rate']*100:.1f}%"),
    ]
    for col, (label, val, delta) in zip(cols, kpi_items):
        col.metric(label, val, delta)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison table
    st.markdown("<div class='section-header'>Policy Comparison Table</div>", unsafe_allow_html=True)
    cmp_df = compare_policies(bdf, mdf, n_days)
    
    def style_improvement(val):
        if isinstance(val, float):
            color = "#2ecc71" if val > 0 else "#e74c3c" if val < 0 else "#f5efe6"
            return f"color: {color}"
        return ""
    
    styled = cmp_df.style.map(style_improvement, subset=["Improvement", "Improvement %"])
    st.dataframe(styled, width="stretch", hide_index=True)

    # Action distribution side-by-side
    st.markdown("<div class='section-header'>Action Distribution by Policy</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    for col, df, title in [(c1, bdf, "Baseline"), (c2, mdf, "MDP Optimal")]:
        ad = action_distribution(df).reset_index()
        ad.columns = ["Action", "Fraction"]
        fig = px.bar(
            ad, x="Action", y="Fraction",
            color="Action", color_discrete_sequence=PALETTE,
            title=f"{title} — Action Mix",
        )
        fig.update_layout(**PLOT_THEME, title_font_color="#c8a97a", showlegend=False)
        col.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Revenue Simulation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>Daily Revenue: Baseline vs MDP</div>", unsafe_allow_html=True)

    brev = daily_revenue_series(bdf)
    mrev = daily_revenue_series(mdf)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=brev["day"], y=brev["revenue"].cumsum(),
        name="Baseline", line=dict(color="#e8734a", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=mrev["day"], y=mrev["revenue"].cumsum(),
        name="MDP Optimal", line=dict(color="#c8a97a", width=2.5),
        fill="tonexty", fillcolor="rgba(200,169,122,0.07)",
    ))
    fig.update_layout(**PLOT_THEME, title="Cumulative Revenue", height=360,
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)

    # Daily net revenue
    with c1:
        fig2 = go.Figure()
        b_net = brev["revenue"] - brev["cost"]
        m_net = mrev["revenue"] - mrev["cost"]
        fig2.add_trace(go.Scatter(x=brev["day"], y=b_net.rolling(7).mean(),
                                  name="Baseline", line=dict(color="#e8734a")))
        fig2.add_trace(go.Scatter(x=mrev["day"], y=m_net.rolling(7).mean(),
                                  name="MDP Optimal", line=dict(color="#c8a97a")))
        fig2.update_layout(**PLOT_THEME, title="7-Day Rolling Net Revenue", height=300,
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, width="stretch")

    # Average reward per day
    with c2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=brev["day"], y=brev["reward"].rolling(7).mean(),
                                  name="Baseline", line=dict(color="#e8734a")))
        fig3.add_trace(go.Scatter(x=mrev["day"], y=mrev["reward"].rolling(7).mean(),
                                  name="MDP Optimal", line=dict(color="#c8a97a")))
        fig3.update_layout(**PLOT_THEME, title="7-Day Rolling Avg Reward per Step", height=300,
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3, width="stretch")

    # Cost breakdown
    st.markdown("<div class='section-header'>Cost vs Revenue Breakdown</div>", unsafe_allow_html=True)
    bar_data = pd.DataFrame({
        "Metric": ["Revenue", "Cost", "Net Revenue"],
        "Baseline":    [bm["total_revenue"], bm["total_cost"], bm["net_revenue"]],
        "MDP Optimal": [mm["total_revenue"], mm["total_cost"], mm["net_revenue"]],
    })
    fig4 = px.bar(bar_data.melt(id_vars="Metric", var_name="Policy", value_name="USD"),
                  x="Metric", y="USD", color="Policy", barmode="group",
                  color_discrete_map={"Baseline": "#e8734a", "MDP Optimal": "#c8a97a"})
    fig4.update_layout(**PLOT_THEME, height=320, legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig4, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Policy & States
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>Optimal Policy vs Baseline</div>", unsafe_allow_html=True)

    policy_rows = []
    for s in range(N_STATES):
        policy_rows.append({
            "State":            STATES[s],
            "Baseline Action":  ACTIONS[bpi[s]],
            "MDP Action":       ACTIONS[pi[s]],
            "V*(s)":            round(float(V[s]), 3),
            "Changed?":         "✅ Yes" if pi[s] != bpi[s] else "—",
        })
    policy_df = pd.DataFrame(policy_rows)
    st.dataframe(policy_df, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)

    # State distribution (initial)
    with c1:
        st.markdown("<div class='section-header'>Initial State Distribution</div>", unsafe_allow_html=True)
        state_dist = uf["state_name"].value_counts().reset_index()
        state_dist.columns = ["State", "Count"]
        state_dist["Color"] = state_dist["State"].map(STATE_COLORS)
        fig5 = px.pie(state_dist, names="State", values="Count",
                      color="State", color_discrete_map=STATE_COLORS,
                      hole=0.45)
        fig5.update_layout(**PLOT_THEME, height=340,
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig5.update_traces(textfont_color="#f5efe6")
        st.plotly_chart(fig5, width="stretch")

    # State drift over simulation
    with c2:
        st.markdown("<div class='section-header'>State Distribution Over Time (MDP)</div>", unsafe_allow_html=True)
        sdot = state_distribution_over_time(mdf)
        fig6 = px.area(sdot, x="day", y="fraction", color="state_name",
                       color_discrete_map=STATE_COLORS)
        fig6.update_layout(**PLOT_THEME, height=340,
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           yaxis_title="Fraction of Users")
        st.plotly_chart(fig6, width="stretch")

    # Value function bar
    st.markdown("<div class='section-header'>Value Function V*(s) — Long-Run Expected Return</div>", unsafe_allow_html=True)
    vf_df = pd.DataFrame({"State": [STATES[s] for s in range(N_STATES)], "V*": V})
    fig7 = px.bar(vf_df, x="State", y="V*",
                  color="State", color_discrete_map=STATE_COLORS)
    fig7.update_layout(**PLOT_THEME, height=300, showlegend=False)
    st.plotly_chart(fig7, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Transition Matrix
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-header'>Transition Probability Heatmaps P[s, a, s']</div>", unsafe_allow_html=True)

    action_select = st.selectbox(
        "Select action to inspect",
        options=list(ACTIONS.values()),
        index=0,
    )
    a_idx = list(ACTIONS.values()).index(action_select)

    P_slice = P[:, a_idx, :]   # shape (N_STATES, N_STATES)
    state_labels = [STATES[s] for s in range(N_STATES)]

    fig8 = px.imshow(
        P_slice,
        x=state_labels, y=state_labels,
        color_continuous_scale="YlOrBr",
        labels=dict(x="Next State s'", y="Current State s", color="Prob"),
        text_auto=".2f",
        aspect="auto",
        title=f"P[s, a='{action_select}', s']",
    )
    fig8.update_layout(**PLOT_THEME, height=420,
                       coloraxis_colorbar=dict(title="Prob", tickfont_color="#f5efe6"),
                       xaxis_tickangle=-30)
    st.plotly_chart(fig8, width="stretch")

    st.markdown("<div class='section-header'>Reward Matrix R[s, a]</div>", unsafe_allow_html=True)
    fig9 = px.imshow(
        R,
        x=list(ACTIONS.values()),
        y=state_labels,
        color_continuous_scale="RdYlGn",
        text_auto=".2f",
        labels=dict(x="Action", y="State", color="Reward"),
        aspect="auto",
    )
    fig9.update_layout(**PLOT_THEME, height=360,
                       coloraxis_colorbar=dict(title="Reward", tickfont_color="#f5efe6"))
    st.plotly_chart(fig9, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MDP Internals
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-header'>Value Iteration Convergence</div>", unsafe_allow_html=True)

    vi_df = pd.DataFrame(vi_hist, columns=["Iteration", "Max Delta"])
    fig10 = px.line(vi_df, x="Iteration", y="Max Delta", log_y=True,
                    title="Bellman Residual (log scale)")
    fig10.update_layout(**PLOT_THEME, height=320)
    fig10.update_traces(line_color="#c8a97a")
    st.plotly_chart(fig10, width="stretch")

    st.info(f"Converged in **{len(vi_hist)}** iterations  |  γ = {gamma}  |  Final residual = {vi_hist[-1][1]:.2e}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Q-Values (Final)</div>", unsafe_allow_html=True)
        gamma_v = gamma
        Q_final = R + gamma_v * np.einsum("san,n->sa", P, V)
        q_df = pd.DataFrame(Q_final, index=state_labels, columns=list(ACTIONS.values()))
        fig11 = px.imshow(q_df, color_continuous_scale="YlOrBr",
                          text_auto=".2f", aspect="auto",
                          labels=dict(x="Action", y="State", color="Q"))
        fig11.update_layout(**PLOT_THEME, height=340)
        st.plotly_chart(fig11, width="stretch")

    with c2:
        st.markdown("<div class='section-header'>State Value Function</div>", unsafe_allow_html=True)
        v_df = pd.DataFrame({"State": state_labels, "V*(s)": V, "Optimal Action": [ACTIONS[a] for a in pi]})
        fig12 = px.bar(v_df, x="V*(s)", y="State", orientation="h",
                       color="Optimal Action", color_discrete_sequence=PALETTE,
                       text="Optimal Action")
        fig12_theme = {
            **PLOT_THEME,
            "yaxis": {**PLOT_THEME["yaxis"], "categoryorder": "total ascending"},
        }
        fig12.update_layout(**fig12_theme, height=340, showlegend=False)
        st.plotly_chart(fig12, width="stretch")

    # Sensitivity: vary gamma
    st.markdown("<div class='section-header'>Policy Sensitivity to γ</div>", unsafe_allow_html=True)
    gamma_range = np.arange(0.80, 1.00, 0.02)
    sensitivity_rows = []
    for g in gamma_range:
        Vg, pig, _ = value_iteration(P, R, gamma=g, max_iter=500)
        sensitivity_rows.append({
            "gamma": round(g, 2),
            **{STATES[s]: ACTIONS[pig[s]] for s in range(N_STATES)},
        })
    sens_df = pd.DataFrame(sensitivity_rows)
    st.dataframe(sens_df.set_index("gamma"), width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Raw Data
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-header'>User Feature Table</div>", unsafe_allow_html=True)
    display_cols = [
        "person", "state_name", "total_spend", "n_transactions",
        "avg_transaction", "completion_rate", "view_rate",
        "recency", "income",
    ]
    st.dataframe(uf[display_cols].head(500), width="stretch", hide_index=True)

    st.markdown("<div class='section-header'>Sample Simulation Log (MDP, first 200 rows)</div>", unsafe_allow_html=True)
    st.dataframe(mdf.head(200), width="stretch", hide_index=True)

    st.markdown("<div class='section-header'>Offer Portfolio</div>", unsafe_allow_html=True)
    port_df = pd.DataFrame(data["portfolio"])
    st.dataframe(port_df, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<hr style='border-color:#3d2e1e; margin-top:2rem'>
<p style='text-align:center; color:#5a4030; font-size:0.8rem'>
  MDP Offer Optimizer · Starbucks Dataset · Value Iteration · Built with Streamlit
</p>
""", unsafe_allow_html=True)
