import pickle
import pandas as pd

def load_price_data():
    with open("cache_equity.pkl", "rb") as f:
        return pickle.load(f)

def load_macro_data():
    with open("cache_macro.pkl", "rb") as f:
        return pickle.load(f)

def build_macro_df(macro, macro_tickers, feat_lag):
    df = pd.concat(
        {alias: macro[sym]["Close"] for sym, alias in macro_tickers.items()},
        axis=1
    ).dropna()
    df["SP500_Ret"] = df["SP500_Close"].pct_change()
    df["VIX_Δ"]    = df["VIX_Close"].pct_change()
    df["TNX_Δ"]    = df["TNX_Close"].pct_change()
    return df.shift(feat_lag).ffill(limit=1)

def load_sentiment(sent_file):
    df = pd.read_csv(sent_file, parse_dates=["date"])
    df["stock"] = df["stock"].str.upper().str.strip()
    df["date"]  = df["date"].dt.normalize()
    return df.groupby(["stock","date"])["Sentiment"].mean()
