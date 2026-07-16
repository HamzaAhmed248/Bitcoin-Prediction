from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

app = FastAPI()


model = joblib.load("model.pkl")



MD_df_btc = pd.read_csv("MD_BTC_7_years.csv")


MD_df_btc = MD_df_btc.sort_values("Date").reset_index(drop=True)


class UserInput(BaseModel):
    Closing_P: float
    Highest_P: float
    Lowest_P: float
    Volume: float



def build_features(df):

    df = df.copy()

    # LAGS
    df["lag1"] = df["Closing_P"].shift(1)
    df["lag2"] = df["Closing_P"].shift(2)
    df["lag7"] = df["Closing_P"].shift(7)

    # RETURNS
    df["Return_1d"] = df["Closing_P"].pct_change() * 100
    df["Return_7d"] = df["Closing_P"].pct_change(7) * 100

    # EMA
    df["EMA_9"] = df["Closing_P"].ewm(span=9).mean()
    df["EMA_20"] = df["Closing_P"].ewm(span=20).mean()

    # RSI
    delta = df["Closing_P"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Closing_P"].ewm(span=12).mean()
    ema26 = df["Closing_P"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    # Volume features
    df["volume_ratio"] = df["Volume"] / df["Volume"].shift(1)
    df["volume_ratio_7d"] = df["Volume"] / df["Volume"].rolling(7).mean()

    # trend
    df["uptrend"] = (df["EMA_9"] > df["EMA_20"]).astype(int)


    df["RSI_zone"] = pd.cut(
        df["RSI_14"],
        bins=[0, 30, 70, 100],
        labels=[0, 1, 2]
    ).astype(float)


    df["MACD_signal"] = (df["MACD"] > df["Signal"]).astype(int)

    # Bollinger position
    bb_mid = df["Closing_P"].rolling(20).mean()
    std = df["Closing_P"].rolling(20).std()

    bb_upper = bb_mid + 2 * std
    bb_lower = bb_mid - 2 * std

    df["BB_position"] = (df["Closing_P"] - bb_lower) / (bb_upper - bb_lower)

    return df



@app.get("/")
def serve_about_root():
    return FileResponse("about.html")

@app.get("/about")
def serve_about():
    return FileResponse("about.html")

@app.get("/prediction")
def serve_ui():
    return FileResponse("ui.html")

@app.get("/style.css")
def serve_css():
    return FileResponse("style.css")

@app.get("/nav.html")
def serve_nav():
    return FileResponse("nav.html")



@app.post("/predict")
def predict(data: UserInput):

    global MD_df_btc


    new_row = pd.DataFrame([{
        "Date": datetime.today().strftime('%Y-%m-%d'),
        "Closing_P": data.Closing_P,
        "Highest_P": data.Highest_P,
        "Lowest_P": data.Lowest_P,
        "Volume": data.Volume
    }])

    df = pd.concat([MD_df_btc, new_row], ignore_index=True)


    df = build_features(df)


    latest_row = df.tail(1)


    features = [
        'lag1','lag2','lag7',
        'Return_1d','Return_7d',
        'RSI_14','RSI_zone',
        'volume_ratio','volume_ratio_7d',
        'uptrend','BB_position','MACD_signal',
        'Closing_P','Highest_P','Lowest_P'
    ]

    latest_row = latest_row[features].fillna(0)


    prediction = model.predict(latest_row)[0]

    return {
        "prediction": round(float(prediction), 2)
    }