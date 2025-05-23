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

SENT_CACHE_FILE = "scored_headlines_full.csv"

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
