from backtest_pipeline.features import add_feats
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def run_backtest_stream(prices, macro_df, daily_sent, config):
    import datetime

    frames = []
    for tkr, df_price in prices.items():
        tmp = add_feats(df_price.copy(), tkr, daily_sent, config)
        tmp = tmp.join(macro_df, how="left").ffill(limit=1)
        tmp["Ticker"] = tkr
        frames.append(tmp)
    master = pd.concat(frames)
    dates = sorted(master.index.unique())

    rf = RandomForestClassifier(
        n_estimators=config.N_TREES,
        max_depth=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cash = total_contributed = config.INITIAL_CASH
    book = []
    rebalance_counter = 0

    for i in range(config.TRAIN_WIN_DAYS, len(dates) - 1):
        day = dates[i]
        if rebalance_counter % config.REBALANCE_DAYS != 0:
            rebalance_counter += 1
            continue
        rebalance_counter += 1

        train_start = dates[i - config.TRAIN_WIN_DAYS]
        train_df = master[(master.index >= train_start) & (master.index < day)]
        test_df = master[master.index == day]
        if train_df.empty or test_df.empty:
            continue

        X_tr = train_df[config.FEATURES]
        y_tr = train_df["Target"]
        X_ts = test_df[config.FEATURES]

        mask_tr = np.isfinite(X_tr).all(axis=1)
        mask_ts = np.isfinite(X_ts).all(axis=1)
        X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
        X_ts = X_ts[mask_ts]
        test_df = test_df[mask_ts]

        if X_tr.empty or X_ts.empty:
            continue

        rf.fit(X_tr, y_tr)
        probs = rf.predict_proba(X_ts)[:, 1]
        sig = pd.DataFrame({"Ticker": test_df["Ticker"].values, "Prob": probs}).set_index("Ticker")

        picks = sig[sig["Prob"] >= config.CONF_THRESH].index.tolist()
        if not picks:
            continue
        weights = sig.loc[picks, "Prob"] / sig.loc[picks, "Prob"].sum()

        for pos in book.copy():
            if pos["tkr"] not in picks:
                price_series = prices[pos["tkr"]].loc[day, "Close"]
                price = float(price_series.iloc[0] if isinstance(price_series, pd.Series) else price_series)
                effective = price * (1 - config.SLIPPAGE_BP / 1e4)
                proceeds = pos["shares"] * effective * (1 - config.EXIT_FEE_BP / 1e4)
                cash += proceeds
                book.remove(pos)

        alloc_cash = cash * config.LEVERAGE
        for tkr in picks:
            w = weights.loc[tkr]
            alloc = alloc_cash * w
            price_series = prices[tkr].loc[day, "Close"]
            price = float(price_series.iloc[0] if isinstance(price_series, pd.Series) else price_series)
            effective = price * (1 + config.SLIPPAGE_BP / 1e4)
            shares = alloc / effective
            cost = shares * effective * (1 + config.ENTRY_FEE_BP / 1e4)
            if cost > cash:
                continue
            cash -= cost
            book.append({"tkr": tkr, "shares": shares})

        nav = cash + sum(
            (float(prices[p["tkr"]].loc[day, "Close"].iloc[0] if isinstance(prices[p["tkr"]].loc[day, "Close"], pd.Series)
             else prices[p["tkr"]].loc[day, "Close"])) * p["shares"]
            for p in book
        )
        pnl = nav - total_contributed
        yield f"{day.date()}: NAV = ${nav:,.2f} | P&L = ${pnl:,.2f}"

        cash += config.DCA_AMOUNT
        total_contributed += config.DCA_AMOUNT
        month = day.month
        if 'last_month' not in locals():
            last_month = month
        if month != last_month:
            cash += config.MONTHLY_CONTRIBUTION
            total_contributed += config.MONTHLY_CONTRIBUTION
            last_month = month

    yield f"=== Backtest Complete ==="
