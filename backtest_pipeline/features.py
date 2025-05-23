import numpy as np
import pandas as pd

def rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = -delta.clip(upper=0).rolling(n).mean()
    rs    = gain / loss
    return 100 - 100 / (1 + rs)

def add_feats(df, tkr, daily_sent, config):
    df["Return"]     = df["Close"].pct_change()
    df["MA5"]        = df["Close"].rolling(5).mean()
    df["MA10"]       = df["Close"].rolling(10).mean()
    df["MA20"]       = df["Close"].rolling(20).mean()
    df["MA50"]       = df["Close"].rolling(50).mean()
    df["RSI"]        = rsi(df["Close"])
    df["Volatility"] = df["Return"].rolling(5).std()
    df["Volume_Chg"] = df["Volume"].pct_change()
    df["Lag1"]       = df["Return"].shift(1)
    df["Lag2"]       = df["Return"].shift(2)
    df["Lag3"]       = df["Return"].shift(3)
    df["MACD"]       = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()

    idx = df.index.normalize()
    if tkr in daily_sent.index.get_level_values(0):
        s = daily_sent.xs(tkr, level=0).reindex(idx, fill_value=0)
    else:
        s = pd.Series(0.0, index=idx)
    df["News_Sent"] = s.values

    fut_max = df["Close"]        .shift(-1)        .rolling(config.HORIZON, min_periods=config.HORIZON)        .max()
    df["Target"] = ((fut_max / df["Close"]) >= (1 + config.RISE_THR)).astype(int)

    df[config.LOCAL_FEATS] = df[config.LOCAL_FEATS].shift(config.FEAT_LAG)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna()
