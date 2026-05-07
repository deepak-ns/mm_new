"""
preprocess.py
-------------
Flattens the transcript value column, merges all three DataFrames,
and engineers per-user RFM-style features used by the state engine.
"""

import pandas as pd
import numpy as np
from data_loader import load_all


# ── Flatten ────────────────────────────────────────────────────────────────────

def flatten_transcript(transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the dict stored in the `value` column into proper columns:
      - amount      : transaction value (NaN for non-transaction events)
      - offer_id    : offer identifier   (NaN for pure transactions)
      - offer_reward: reward credited on completion
    """
    value_df = pd.json_normalize(transcript["value"])

    # Rename to avoid clash with portfolio 'reward'
    if "reward" in value_df.columns:
        value_df = value_df.rename(columns={"reward": "offer_reward"})

    flat = pd.concat(
        [transcript[["event", "person", "time"]].reset_index(drop=True),
         value_df.reset_index(drop=True)],
        axis=1,
    )
    flat["amount"]       = flat.get("amount",       pd.Series(dtype=float))
    flat["offer_id"]     = flat.get("offer_id",     pd.Series(dtype=str))
    flat["offer_reward"] = flat.get("offer_reward", pd.Series(dtype=float))
    return flat


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_all(
    portfolio: pd.DataFrame,
    profile:   pd.DataFrame,
    transcript: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a single enriched DataFrame with one row per transcript event.
    Joins offer metadata (difficulty, offer_type) and user demographics.
    """
    valid_offer_types = {"bogo", "discount", "informational"}
    portfolio = portfolio[portfolio["offer_type"].isin(valid_offer_types)].copy()
    valid_offer_ids = set(portfolio["id"])

    flat = flatten_transcript(transcript)
    flat = flat[
        flat["offer_id"].isna()
        | flat["offer_id"].isin(valid_offer_ids)
    ].copy()

    # Join offer metadata where available
    merged = flat.merge(
        portfolio[["id", "offer_type", "difficulty", "reward", "duration"]].rename(
            columns={"id": "offer_id", "reward": "portfolio_reward"}
        ),
        on="offer_id",
        how="left",
    )

    # Join user profile
    merged = merged.merge(
        profile.rename(columns={"id": "person"}),
        on="person",
        how="left",
    )

    # Derived time column (hours → day index)
    merged["day"] = (merged["time"] // 24).astype(int)

    return merged


# ── Per-user feature engineering ───────────────────────────────────────────────

def build_user_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the event log into one row per user with RFM-style features:

    Column               Description
    ─────────────────── ─────────────────────────────────────────────────────
    total_spend          Sum of all transaction amounts
    n_transactions       Number of transaction events
    avg_transaction      Mean transaction value
    n_offers_received    Offers pushed to the user
    n_offers_viewed      Offers the user opened
    n_offers_completed   Offers successfully redeemed
    view_rate            n_viewed / n_received  (offer engagement rate)
    completion_rate      n_completed / n_received
    total_reward         Sum of offer_reward credits earned
    last_active_day      Latest day with any activity
    recency              Days since last activity (relative to max day)
    tenure_days          Days since membership start (normalised)
    income               User income (from profile)
    """
    tx  = merged[merged["event"] == "transaction"]
    rcv = merged[merged["event"] == "offer received"]
    vwd = merged[merged["event"] == "offer viewed"]
    cmp = merged[merged["event"] == "offer completed"]

    max_day = merged["day"].max()

    feats = (
        merged.groupby("person")
        .agg(
            income          =("income",       "first"),
            tenure_days     =("became_member_on", lambda s: (
                pd.Timestamp.today() - s.iloc[0]
            ).days if pd.notna(s.iloc[0]) else 0),
        )
        .reset_index()
    )

    # Transaction features
    tx_feats = (
        tx.groupby("person")["amount"]
        .agg(total_spend="sum", n_transactions="count", avg_transaction="mean")
        .reset_index()
    )

    # Offer funnel features
    rcv_feats = rcv.groupby("person").size().rename("n_offers_received").reset_index()
    vwd_feats = vwd.groupby("person").size().rename("n_offers_viewed").reset_index()
    cmp_feats = cmp.groupby("person").size().rename("n_offers_completed").reset_index()
    rwd_feats = (
        cmp.groupby("person")["offer_reward"].sum().rename("total_reward").reset_index()
    )

    # Recency
    recency_feats = (
        merged.groupby("person")["day"]
        .max()
        .rename("last_active_day")
        .reset_index()
    )
    recency_feats["recency"] = max_day - recency_feats["last_active_day"]

    # Merge all feature sets
    for sub in [tx_feats, rcv_feats, vwd_feats, cmp_feats, rwd_feats, recency_feats]:
        feats = feats.merge(sub, on="person", how="left")

    # Fill missing numeric values with 0
    num_cols = [
        "total_spend", "n_transactions", "avg_transaction",
        "n_offers_received", "n_offers_viewed", "n_offers_completed",
        "total_reward", "recency",
    ]
    feats[num_cols] = feats[num_cols].fillna(0)

    # Derived rates (guard against division by zero)
    feats["view_rate"] = np.where(
        feats["n_offers_received"] > 0,
        feats["n_offers_viewed"] / feats["n_offers_received"],
        0.0,
    )
    feats["completion_rate"] = np.where(
        feats["n_offers_received"] > 0,
        feats["n_offers_completed"] / feats["n_offers_received"],
        0.0,
    )

    return feats


# ── Public entry point ─────────────────────────────────────────────────────────

def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.
    Returns (merged_events_df, user_features_df).
    """
    portfolio, profile, transcript = load_all()
    merged = merge_all(portfolio, profile, transcript)
    user_features = build_user_features(merged)
    return merged, user_features
