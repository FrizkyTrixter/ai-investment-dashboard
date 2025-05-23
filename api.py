from flask import Flask, request, Response, send_from_directory
from backtest_pipeline import config, data_loader, backtest

app = Flask(__name__, static_folder="frontend_web", static_url_path="")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/stream', methods=['POST'])
def stream_backtest():
    # ✅ FIX: Read request JSON BEFORE the generator
    params = request.get_json()

    def generate():
        try:
            config.N_TREES = int(params.get("n_trees", 100))
            config.HORIZON = int(params.get("horizon", 100))
            config.RISE_THR = float(params.get("rise_thr", 0.10))

            prices = data_loader.load_price_data()
            macro = data_loader.load_macro_data()
            macro_df = data_loader.build_macro_df(macro, config.MACRO_TICKERS, config.FEAT_LAG)
            sentiment = data_loader.load_sentiment(config.SENT_CACHE_FILE)

            # Yield each line from the streaming backtest
            for line in backtest.run_backtest_stream(prices, macro_df, sentiment, config):
                yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("🚀 Serving on http://localhost:5000")
    app.run(debug=True, threaded=True)
