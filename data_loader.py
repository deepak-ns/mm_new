"""
data_loader.py
--------------
Responsible only for reading the three raw JSON files from disk and returning
them as pandas DataFrames.  No transformation logic lives here.
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _read_json(filename: str) -> list:
    """Load a JSON file from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python generate_data.py` first."
        )
    with open(path, "r") as f:
        return json.load(f)


def load_portfolio() -> pd.DataFrame:
    """
    Returns offer catalog with columns:
      id, offer_type, difficulty, reward, duration, channels
    """
    records = _read_json("portfolio.json")
    df = pd.DataFrame(records)
    # Explode channels list into a pipe-separated string for easy filtering
    df["channels"] = df["channels"].apply(
        lambda c: "|".join(c) if isinstance(c, list) else c
    )
    return df


def load_profile() -> pd.DataFrame:
    """
    Returns customer demographics with columns:
      id, gender, age, income, became_member_on
    """
    records = _read_json("profile.json")
    df = pd.DataFrame(records)
    df["became_member_on"] = pd.to_datetime(
        df["became_member_on"], format="%Y%m%d", errors="coerce"
    )
    return df


def load_transcript() -> pd.DataFrame:
    """
    Returns raw event log with columns:
      event, person, time, value   (value is still a dict here)
    """
    records = _read_json("transcript.json")
    df = pd.DataFrame(records)
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader — returns (portfolio, profile, transcript)."""
    return load_portfolio(), load_profile(), load_transcript()
