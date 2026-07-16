import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
MD_df_btc = pd.read_csv("BTC_7years.csv")


rename_map = {
    'Price': 'Date',
    'Close': 'Closing_P',
    'High': 'Highest_P',
    'Low': 'Lowest_P',
    'Open': 'Opening_P',
    'Volume': 'Volume'
}

MD_df_btc.rename(columns=rename_map, inplace=True)


# ─────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────
MD_df_btc['lag1'] = MD_df_btc['Closing_P'].shift(1)
MD_df_btc['lag2'] = MD_df_btc['Closing_P'].shift(2)
MD_df_btc['lag7'] = MD_df_btc['Closing_P'].shift(7)

MD_df_btc['Return_1d'] = MD_df_btc['Closing_P'].pct_change(1) * 100
MD_df_btc['Return_7d'] = MD_df_btc['Closing_P'].pct_change(7) * 100

MD_df_btc['EMA_9']  = MD_df_btc['Closing_P'].ewm(span=9).mean()
MD_df_btc['EMA_20'] = MD_df_btc['Closing_P'].ewm(span=20).mean()

# RSI
delta = MD_df_btc['Closing_P'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
MD_df_btc['RSI_14'] = 100 - (100 / (1 + rs))

# MACD
ema12 = MD_df_btc['Closing_P'].ewm(span=12).mean()
ema26 = MD_df_btc['Closing_P'].ewm(span=26).mean()

MD_df_btc['MACD'] = ema12 - ema26
MD_df_btc['Signal'] = MD_df_btc['MACD'].ewm(span=9).mean()

# Bollinger Bands
MD_df_btc['BB_Middle'] = MD_df_btc['Closing_P'].rolling(20).mean()
std = MD_df_btc['Closing_P'].rolling(20).std()

MD_df_btc['BB_Upper'] = MD_df_btc['BB_Middle'] + (2 * std)
MD_df_btc['BB_Lower'] = MD_df_btc['BB_Middle'] - (2 * std)

# Volume
MD_df_btc['volume_ratio'] = MD_df_btc['Volume'] / MD_df_btc['Volume'].shift(1)
MD_df_btc['volume_ratio_7d'] = MD_df_btc['Volume'] / MD_df_btc['Volume'].rolling(7).mean()

# Extra
MD_df_btc['uptrend'] = (MD_df_btc['EMA_9'] > MD_df_btc['EMA_20']).astype(int)

MD_df_btc['BB_position'] = (
    (MD_df_btc['Closing_P'] - MD_df_btc['BB_Lower']) /
    (MD_df_btc['BB_Upper'] - MD_df_btc['BB_Lower'])
)

MD_df_btc['MACD_signal'] = ((MD_df_btc['MACD'] > MD_df_btc['Signal']).astype(int))

MD_df_btc['RSI_zone'] = pd.cut(
    MD_df_btc['RSI_14'],
    bins=[0, 30, 70, 100],
    labels=[0, 1, 2]
).astype(float)

# ─────────────────────────────────────────
# TARGET
# ─────────────────────────────────────────
MD_df_btc["NextDay_Closing_P"] = MD_df_btc["Closing_P"].shift(-1)

MD_df_btc.to_csv("MD_BTC_7_years.csv")

MD_df_btc = MD_df_btc.dropna().reset_index(drop=True)


# ─────────────────────────────────────────
# FEATURES + TARGET
# ─────────────────────────────────────────
features = [
    'lag1','lag2','lag7',
    'Return_1d','Return_7d',
    'RSI_14','RSI_zone',
    'volume_ratio','volume_ratio_7d',
    'uptrend','BB_position','MACD_signal',
    'Closing_P','Highest_P','Lowest_P',
]

X = MD_df_btc[features]
y = MD_df_btc["NextDay_Closing_P"].values

# ─────────────────────────────────────────
# ACCURACY FUNCTION
# ─────────────────────────────────────────
def threshold_accuracy(y_true, y_pred, threshold=0.02):
    err = np.abs((y_true - y_pred) / y_true)
    return np.mean(err <= threshold) * 100

# ─────────────────────────────────────────
# TIME SERIES SPLIT
# ─────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=2)
#5=62.64

print("=" * 60)
print("TIME SERIES STACKING REGRESSOR")
print("=" * 60)

models = []
scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):

    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    model = StackingRegressor(
        estimators=[
            ('xgb', XGBRegressor(
                n_estimators=100,
                learning_rate=2,
                #2=67.22
                max_depth=3,
                random_state=42
            )),
            ('lr', LinearRegression()),
        ],
        final_estimator=Ridge(alpha=4),
        cv=5
    )

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = threshold_accuracy(yte, pred)
    scores.append(acc)

    models.append(model)

    print(f"Fold {fold} → Accuracy: {round(acc,2)}%")

print("\nAVG Accuracy:", round(np.mean(scores), 2), "%")

latest_row = X.tail(1)

all_preds = [m.predict(latest_row)[0] for m in models]
avg_prediction = np.mean(all_preds)

print("\nAveraged Prediction (CV Models):", round(avg_prediction, 2))


# final_model = StackingRegressor(
#     estimators=[
#         ('xgb', XGBRegressor(
#             n_estimators=100,
#             learning_rate=2,
#             max_depth=3,
#             random_state=42
#         )),
#         ('lr', LinearRegression()),
#     ],
#     final_estimator=Ridge(alpha=4),
#     cv=5
# )
#
# final_model.fit(X, y)

# # Save model for FastAPI
# joblib.dump(final_model, "p/model.pkl")

# Final prediction
# final_pred = final_model.predict(latest_row)[0]
#
# print("\nFinal Production Prediction:", round(final_pred, 2))