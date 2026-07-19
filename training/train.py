from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "BTC_7years.csv"
OUTPUT_PATH = BASE_DIR / "output" / "model.pkl"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and clean the BTC dataset for training."""
    df = pd.read_csv(path)
    rename_map = {
        "Price": "Date",
        "Close": "Closing_P",
        "High": "Highest_P",
        "Low": "Lowest_P",
        "Open": "Opening_P",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features used by the model."""
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

    df["BB_Middle"] = df["Closing_P"].rolling(20).mean()
    std = df["Closing_P"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (2 * std)
    df["BB_Lower"] = df["BB_Middle"] - (2 * std)

    df["volume_ratio"] = df["Volume"] / df["Volume"].shift(1)
    df["volume_ratio_7d"] = df["Volume"] / df["Volume"].rolling(7).mean()
    df["uptrend"] = (df["EMA_9"] > df["EMA_20"]).astype(int)

    df["BB_position"] = (df["Closing_P"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["MACD_signal"] = (df["MACD"] > df["Signal"]).astype(int)
    df["RSI_zone"] = pd.cut(df["RSI_14"], bins=[0, 30, 70, 100], labels=[0, 1, 2]).astype(float)
    df["NextDay_Closing_P"] = df["Closing_P"].shift(-1)
    return df


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Create the final feature matrix and target vector."""
    features = [
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
    prepared = df.dropna().reset_index(drop=True)
    X = prepared[features]
    y = prepared["NextDay_Closing_P"].to_numpy()
    return X, y


def threshold_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.02) -> float:
    """Compute percentage of predictions within the given error threshold."""
    err = np.abs((y_true - y_pred) / y_true)
    return np.mean(err <= threshold) * 100


def train_model() -> tuple[object, float]:
    """Train the stacking regressor and save it to the output directory."""
    raw_df = load_dataset()
    engineered_df = build_features(raw_df)
    X, y = build_training_frame(engineered_df)

    tscv = TimeSeriesSplit(n_splits=2)
    scores: list[float] = []
    models: list[object] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        model = StackingRegressor(
            estimators=[("lr", LinearRegression())],
            final_estimator=Ridge(alpha=4),
            cv=5,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        scores.append(threshold_accuracy(yte, pred))
        models.append(model)
        print(f"Fold {fold} → Accuracy: {round(scores[-1], 2)}%")

    latest_row = X.tail(1)
    all_preds = [model.predict(latest_row)[0] for model in models]
    avg_prediction = np.mean(all_preds)
    print("\nAVG Accuracy:", round(np.mean(scores), 2), "%")
    print("\nAveraged Prediction (CV Models):", round(avg_prediction, 2))

    final_model = StackingRegressor(
        estimators=[("lr", LinearRegression())],
        final_estimator=Ridge(alpha=4),
        cv=5,
    )
    final_model.fit(X, y)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH}")
    return final_model, float(np.mean(scores))


if __name__ == "__main__":
    train_model()
