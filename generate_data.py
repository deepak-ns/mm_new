"""
generate_data.py
----------------
Generates synthetic Starbucks-like JSON datasets:
  - portfolio.json : offer catalog
  - profile.json   : customer demographics
  - transcript.json: event log (offer received/viewed/completed, transactions)

Run once before any other module:
    python generate_data.py
"""

import json
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── Portfolio ──────────────────────────────────────────────────────────────────
def make_portfolio():
    offers = [
        {"id": "offer_bogo_1",   "offer_type": "bogo",          "difficulty": 5,  "reward": 5,  "duration": 7,  "channels": ["email","mobile"]},
        {"id": "offer_bogo_2",   "offer_type": "bogo",          "difficulty": 10, "reward": 10, "duration": 5,  "channels": ["web","mobile"]},
        {"id": "offer_disc_1",   "offer_type": "discount",      "difficulty": 7,  "reward": 3,  "duration": 7,  "channels": ["email","web"]},
        {"id": "offer_disc_2",   "offer_type": "discount",      "difficulty": 10, "reward": 2,  "duration": 10, "channels": ["email","mobile","web"]},
        {"id": "offer_info_1",   "offer_type": "informational", "difficulty": 0,  "reward": 0,  "duration": 4,  "channels": ["email"]},
        {"id": "offer_info_2",   "offer_type": "informational", "difficulty": 0,  "reward": 0,  "duration": 3,  "channels": ["mobile","web"]},
        {"id": "offer_reward_1", "offer_type": "reward",        "difficulty": 15, "reward": 8,  "duration": 10, "channels": ["email","mobile","web"]},
        {"id": "offer_reward_2", "offer_type": "reward",        "difficulty": 20, "reward": 10, "duration": 7,  "channels": ["email"]},
    ]
    return offers


# ── Profile ────────────────────────────────────────────────────────────────────
def make_profile(n=2000):
    genders    = np.random.choice(["M", "F", "O"], size=n, p=[0.47, 0.47, 0.06])
    ages       = np.random.normal(loc=54, scale=17, size=n).clip(18, 95).astype(int)
    # Mix of incomes: lower variance to create inactive/low-value users too
    incomes    = np.random.lognormal(mean=10.8, sigma=0.65, size=n).clip(15_000, 150_000).astype(int)
    became_mem = [
        (datetime(2013, 1, 1) + timedelta(days=int(d))).strftime("%Y%m%d")
        for d in np.random.randint(0, 365*6, size=n)
    ]
    profiles = []
    for i in range(n):
        profiles.append({
            "id":                  f"user_{i:04d}",
            "gender":              genders[i],
            "age":                 int(ages[i]),
            "income":              int(incomes[i]),
            "became_member_on":    became_mem[i],
        })
    return profiles


# ── Transcript ─────────────────────────────────────────────────────────────────
def make_transcript(profiles, portfolio, n_events_per_user=(8, 30)):
    """
    For each user simulate a sequence of events over 30 days (720 hrs).
    Event types:
      - transaction  : {"amount": float}
      - offer received: {"offer_id": str}
      - offer viewed  : {"offer_id": str}
      - offer completed: {"offer_id": str, "reward": float}
    """
    offer_ids  = [o["id"] for o in portfolio]
    offer_map  = {o["id"]: o for o in portfolio}
    events     = []
    event_id   = 0

    for user in profiles:
        uid      = user["id"]
        income   = user["income"]
        # high income → more transactions / higher spend
        spend_mu = 4 + (income / 150_000) * 8
        # 15% of users are nearly inactive (0-3 events)
        if random.random() < 0.15:
            n_events = random.randint(0, 3)
            hours = sorted(np.random.choice(720, size=n_events, replace=False).tolist()) if n_events > 0 else []
        # 10% are brand new (1-2 events only)
        elif random.random() < 0.10:
            n_events = random.randint(1, 2)
            hours = sorted(np.random.choice(720, size=n_events, replace=False).tolist())
        # 12% are at-risk: had activity but mostly old (high recency)
        elif random.random() < 0.12:
            n_events = random.randint(3, 8)
            # Force all their events into the first half of the time window
            # so recency ends up high (they haven't been active lately)
            hours = sorted(np.random.choice(360, size=n_events, replace=False).tolist())
        else:
            n_events = random.randint(*n_events_per_user)
            hours = sorted(np.random.choice(720, size=n_events, replace=False).tolist())

        pending_offers = {}   # offer_id → hour_received

        for h in hours:
            roll = random.random()

            if pending_offers and roll < 0.30:
                # view a pending offer
                oid = random.choice(list(pending_offers.keys()))
                events.append({
                    "event":  "offer viewed",
                    "person": uid,
                    "time":   h,
                    "value":  {"offer_id": oid}
                })

            elif pending_offers and roll < 0.55:
                # complete a pending offer
                oid    = random.choice(list(pending_offers.keys()))
                o_info = offer_map[oid]
                events.append({
                    "event":  "offer completed",
                    "person": uid,
                    "time":   h,
                    "value":  {"offer_id": oid, "reward": o_info["reward"]}
                })
                del pending_offers[oid]

            elif roll < 0.70:
                # receive a new offer
                oid = random.choice(offer_ids)
                pending_offers[oid] = h
                events.append({
                    "event":  "offer received",
                    "person": uid,
                    "time":   h,
                    "value":  {"offer_id": oid}
                })

            else:
                # plain transaction
                amount = max(0.5, np.random.normal(spend_mu, 2.0))
                events.append({
                    "event":  "transaction",
                    "person": uid,
                    "time":   h,
                    "value":  {"amount": round(amount, 2)}
                })

            event_id += 1

    return events


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating synthetic Starbucks dataset…")

    portfolio  = make_portfolio()
    profiles   = make_profile(n=2000)
    transcript = make_transcript(profiles, portfolio)

    with open("data/portfolio.json",  "w") as f: json.dump(portfolio,  f, indent=2)
    with open("data/profile.json",    "w") as f: json.dump(profiles,   f, indent=2)
    with open("data/transcript.json", "w") as f: json.dump(transcript, f, indent=2)

    import os, pathlib
    pathlib.Path("data").mkdir(exist_ok=True)
    with open("data/portfolio.json",  "w") as f: json.dump(portfolio,  f, indent=2)
    with open("data/profile.json",    "w") as f: json.dump(profiles,   f, indent=2)
    with open("data/transcript.json", "w") as f: json.dump(transcript, f, indent=2)

    print(f"  portfolio  : {len(portfolio)} offers")
    print(f"  profiles   : {len(profiles)} customers")
    print(f"  transcript : {len(transcript)} events")
    print("Done — files written to data/")