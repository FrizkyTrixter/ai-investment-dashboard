import os

BASE_DIR = "backtest_pipeline"
os.makedirs(BASE_DIR, exist_ok=True)

files = {
    "__init__.py": "",
    "config.py": """\
# config.py

START = "2010-01-01"
NEWS_END = "2020-12-31"

TRAIN_WIN_DAYS = 100
REBALANCE_DAYS = 30

RISE_THR = 0.10
HORIZON = 100
N_TREES = 100
CONF_THRESH = 0.50
FEAT_LAG = 1
LEVERAGE = 1.5

ENTRY_FEE_BP = 10
EXIT_FEE_BP = 10
SLIPPAGE_BP = 5

DCA_AMOUNT = 0
MONTHLY_CONTRIBUTION = 0
INITIAL_CASH = 1000.0

SENT_CACHE_FILE = "sent_cache.pkl"

MACRO_TICKERS = {
    "^GSPC": "SP500_Close",
    "^VIX":  "VIX_Close",
    "^TNX":  "TNX_Close",
}

LOCAL_FEATS = [
    "Return","MA5","MA10","MA20","MA50","RSI",
    "Volatility","Volume_Chg","Lag1","Lag2","Lag3",
    "MACD","News_Sent",
]

MACRO_FEATS = [
    "SP500_Ret","VIX_Close","VIX_Δ","TNX_Close","TNX_Δ",
]

FEATURES = LOCAL_FEATS + MACRO_FEATS
""",
    "data_loader.py": """\
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
""",
    "features.py": """\
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

    fut_max = df["Close"]\
        .shift(-1)\
        .rolling(config.HORIZON, min_periods=config.HORIZON)\
        .max()
    df["Target"] = ((fut_max / df["Close"]) >= (1 + config.RISE_THR)).astype(int)

    df[config.LOCAL_FEATS] = df[config.LOCAL_FEATS].shift(config.FEAT_LAG)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna()
""",
    "backtest.py": """\
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import add_feats

def run_backtest(prices, macro_df, daily_sent, config):
    frames = []
    for tkr, df_price in prices.items():
        tmp = add_feats(df_price.copy(), tkr, daily_sent, config)
        tmp = tmp.join(macro_df, how="left").ffill(limit=1)
        tmp["Ticker"] = tkr
        frames.append(tmp)

    master = pd.concat(frames)
    dates  = sorted(master.index.unique())

    rf = RandomForestClassifier(
        n_estimators=config.N_TREES,
        max_depth=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cash  = total = config.INITIAL_CASH
    book  = []
    metrics = []
    rebalance = 0
    last_month = None

    for i in range(config.TRAIN_WIN_DAYS, len(dates)-1):
        day = dates[i]
        if rebalance % config.REBALANCE_DAYS != 0:
            rebalance += 1
            continue
        rebalance += 1

        start = dates[i - config.TRAIN_WIN_DAYS]
        train = master[(master.index >= start) & (master.index < day)]
        test  = master[master.index == day]
        if train.empty or test.empty:
            continue

        X_tr, y_tr = train[config.FEATURES], train["Target"]
        X_ts, y_ts = test[config.FEATURES],  test["Target"]

        mask_tr = np.isfinite(X_tr).all(axis=1)
        mask_ts = np.isfinite(X_ts).all(axis=1)
        X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
        X_ts, y_ts = X_ts[mask_ts], y_ts[mask_ts]
        if X_tr.empty or X_ts.empty:
            continue

        rf.fit(X_tr, y_tr)
        probs = rf.predict_proba(X_ts)[:,1]
        sig   = pd.DataFrame({
            "Ticker": test["Ticker"].values,
            "Prob":   probs
        }).set_index("Ticker")

        picks = sig[sig["Prob"] >= config.CONF_THRESH].index.tolist()
        if not picks:
            continue
        weights = sig.loc[picks,"Prob"] / sig.loc[picks,"Prob"].sum()

        # SELL
        for pos in book.copy():
            if pos["tkr"] not in picks:
                price = float(prices[pos["tkr"]].loc[day,"Close"])
                eff   = price * (1 - config.SLIPPAGE_BP/1e4)
                cash += pos["shares"] * eff * (1 - config.EXIT_FEE_BP/1e4)
                book.remove(pos)

        # BUY / rebalance
        alloc = cash * config.LEVERAGE
        for tkr in picks:
            w     = weights[tkr]
            price = float(prices[tkr].loc[day,"Close"])
            eff   = price * (1 + config.SLIPPAGE_BP/1e4)
            shares = alloc * w / eff
            cost   = shares * eff * (1 + config.ENTRY_FEE_BP/1e4)
            if cost <= cash:
                cash -= cost
                book.append({"tkr": tkr, "shares": shares})

        nav = cash + sum(
            float(prices[p["tkr"]].loc[day,"Close"]) * p["shares"]
            for p in book
        )
        pnl = nav - total
        metrics.append({"Date": day, "NAV": nav, "P&L": pnl})

        cash  += config.DCA_AMOUNT
        total += config.DCA_AMOUNT
        m = day.month
        if last_month is None:
            last_month = m
        if m != last_month:
            cash  += config.MONTHLY_CONTRIBUTION
            total += config.MONTHLY_CONTRIBUTION
            last_month = m

    return pd.DataFrame(metrics).set_index("Date")
""",
    "main.py": """\
from data_loader import load_price_data, load_macro_data, build_macro_df, load_sentiment
from backtest import run_backtest
import config

def main():
    prices    = load_price_data()
    macro     = load_macro_data()
    macro_df  = build_macro_df(macro, config.MACRO_TICKERS, config.FEAT_LAG)
    daily_sent= load_sentiment(config.SENT_CACHE_FILE)
    results   = run_backtest(prices, macro_df, daily_sent, config)

    print("=== Final Summary ===")
    print(results.tail())
    print(f"Final NAV: ${results['NAV'].iloc[-1]:.2f} | P&L: ${results['P&L'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
"""
}

for name, content in files.items():
    path = os.path.join(BASE_DIR, name)
    with open(path, "w") as f:
        f.write(content)

print(f"Created modular pipeline in ./{BASE_DIR}/")
