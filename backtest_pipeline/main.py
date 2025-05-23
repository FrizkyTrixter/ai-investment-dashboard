from data_loader import load_price_data, load_macro_data, build_macro_df, load_sentiment
from backtest import run_backtest
import config

def main():
    prices    = load_price_data()
    macro     = load_macro_data()
    macro_df  = build_macro_df(macro, config.MACRO_TICKERS, config.FEAT_LAG)
    daily_sent= load_sentiment(config.SENT_CACHE_FILE)
    results   = run_backtest(prices, macro_df, daily_sent, config)


if __name__ == "__main__":
    main()
