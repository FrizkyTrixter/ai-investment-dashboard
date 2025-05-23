# AI-Powered Investment Dashboard

This project is an AI-enhanced investment dashboard that analyzes market sentiment from news headlines and macroeconomic data to generate trading signals. It includes a backend processing pipeline, a backtest simulator, and a frontend web interface that visualizes predicted returns and investment performance over time.

## 🚀 Overview

The system works as follows:

1. **Headline Scoring**: News headlines are scored for sentiment and relevance using machine learning.
2. **Backtesting**: The pipeline uses historical data to simulate trades and compute net asset value (NAV) and profit/loss (P&L).
3. **Frontend Generation**: A JavaScript-based dashboard displays results day-by-day with dynamic updates.
4. **Caching**: Previously computed results are cached in `.pkl` files for faster loading.

## 🗂 Project Structure

```
frontend/
├── api.py                        # FastAPI server (not currently used)
├── generate_frontend.py         # Entry point for generating and launching the frontend
├── scaffold.py                  # Scaffolds the web interface from a template
├── cache_equity.pkl             # Pickled equity cache for fast loading
├── cache_macro.pkl              # Pickled macroeconomic cache
├── scored_headlines_full.csv    # Scored news dataset
├── frontend_web/
│   ├── app.js                   # Dynamic script updating UI with prediction results
│   ├── index.html               # Static HTML layout
│   └── styles.css               # CSS for basic styling
└── backtest_pipeline/
    ├── main.py                  # Backtesting logic and strategy runner
    ├── config.py                # Configuration values and constants
    ├── features.py              # Feature engineering utilities
    ├── data_loader.py          # Historical data loading (news + financials)
    └── __init__.py              # Package init
```

## 📊 Output Sample

When launched, the dashboard simulates the backtest by rendering output such as:

```
2020-08-09: NAV = $998.51 | P&L = $-1.49
2020-09-21: NAV = $1006.59 | P&L = $6.59
2020-11-02: NAV = $1090.85 | P&L = $90.85
```

This simulates a real trading environment without requiring actual investment.

## 🧑‍💻 How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/FrizkyTrixter/ai-investment-dashboard.git
cd ai-investment-dashboard/frontend
```

### Step 2: Set up the environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the dashboard

```bash
python generate_frontend.py
```

This will:
- Load cached data
- Run the backtest strategy
- Scaffold and open the HTML dashboard
- Display investment performance line-by-line

## ✅ Requirements

- Python 3.10+
- pip

All required packages are listed in `requirements.txt`.

## 🌱 Future Work

- Deploy frontend to a web host
- Connect live data APIs for real-time updates
- Add support for multiple strategies and performance comparison
- Visual graphs (Plotly or D3.js)

## 🧠 Author

Created by [Mateo Day](https://github.com/FrizkyTrixter)  
AI, ML, and Financial Technology Enthusiast
