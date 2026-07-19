from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from api.config import HISTORY_PATH
except ImportError:  # pragma: no cover - supports direct script execution
    from config import HISTORY_PATH

FEATURE_COLUMNS = [
    "lag1",
    "lag2",
    "lag7",
    "Return_1d",
    "Return_7d",
    "RSI_14",
    "RSI_zone",
    "volume_ratio",
    "volume_ratio_7d",
    "uptrend",
    "BB_position",
    "MACD_signal",
    "Closing_P",
    "Highest_P",
    "Lowest_P",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the same engineered features used during training."""
    df = df.copy()

    df["lag1"] = df["Closing_P"].shift(1)
    df["lag2"] = df["Closing_P"].shift(2)
    df["lag7"] = df["Closing_P"].shift(7)

    df["Return_1d"] = df["Closing_P"].pct_change(1) * 100
    df["Return_7d"] = df["Closing_P"].pct_change(7) * 100

    df["EMA_9"] = df["Closing_P"].ewm(span=9).mean()
    df["EMA_20"] = df["Closing_P"].ewm(span=20).mean()

    delta = df["Closing_P"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Closing_P"].ewm(span=12).mean()
    ema26 = df["Closing_P"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    df["volume_ratio"] = df["Volume"] / df["Volume"].shift(1)
    df["volume_ratio_7d"] = df["Volume"] / df["Volume"].rolling(7).mean()

    df["uptrend"] = (df["EMA_9"] > df["EMA_20"]).astype(int)

    df["RSI_zone"] = pd.cut(df["RSI_14"], bins=[0, 30, 70, 100], labels=[0, 1, 2]).astype(float)

    df["MACD_signal"] = (df["MACD"] > df["Signal"]).astype(int)

    bb_mid = df["Closing_P"].rolling(20).mean()
    std = df["Closing_P"].rolling(20).std()
    bb_upper = bb_mid + 2 * std
    bb_lower = bb_mid - 2 * std
    df["BB_position"] = (df["Closing_P"] - bb_lower) / (bb_upper - bb_lower)

    return df


def load_history(path: str | Path = HISTORY_PATH) -> list[dict[str, Any]]:
    """Load the compact runtime history state used for inference."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_features(history_rows: list[dict[str, Any]], payload: dict[str, Any]) -> pd.DataFrame:
    """Create the feature frame for one prediction request."""
    history_df = pd.DataFrame(history_rows)
    new_row = pd.DataFrame(
        [
            {
                "Date": datetime.today().strftime("%Y-%m-%d"),
                "Closing_P": payload["Closing_P"],
                "Highest_P": payload["Highest_P"],
                "Lowest_P": payload["Lowest_P"],
                "Volume": payload["Volume"],
            }
        ]
    )

    df = pd.concat([history_df, new_row], ignore_index=True)
    df = build_features(df)
    latest_row = df.tail(1)
    return latest_row[FEATURE_COLUMNS].fillna(0)
