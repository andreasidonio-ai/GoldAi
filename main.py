import os
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from models.strategy_classifier import StrategyClassifier
from models.trend_model import TrendPredictor
from news_module.news_fetcher import NewsFetcher
from news_module.sentiment_analyzer import NewsSentimentHandler
from lot_optimizer.optimizer import LotSizeOptimizer
from mt5_connector.trade_executor import TradeExecutor
from mt5_connector.account_handler import AccountHandler
from mt5_connector.order_manager import OrderManager
from strategies.scalping import ScalpingStrategy
from strategies.day_trading import DayTradingStrategy
from strategies.swing import SwingTradingStrategy

logging.basicConfig(level=logging.INFO)

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 200

# Load local .env (for local dev only — in production set env vars securely)
load_dotenv()

MT5_ACCOUNT = os.getenv("ACCOUNT")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")  # optional depending on broker

if not MT5_ACCOUNT or not MT5_PASSWORD:
    logging.error("Missing MT5 credentials. Set ACCOUNT and MT5_PASSWORD in environment or .env.")
    raise SystemExit(1)

# Initialize MT5 with login/password/server
if not mt5.initialize(login=int(MT5_ACCOUNT), password=MT5_PASSWORD, server=MT5_SERVER):
    logging.error(f"MT5 initialization failed: {mt5.last_error()}")
    mt5.shutdown()
    raise SystemExit(1)

logging.info(f"MT5 initialized for account {MT5_ACCOUNT}")

account = AccountHandler()
acc_info = account.get_account_info()
if not acc_info:
    logging.error("Account info retrieval failed")
    raise SystemExit(1)

# Load models
classifier = StrategyClassifier()
sentiment_handler = NewsSentimentHandler()
lot_optimizer = LotSizeOptimizer()
trade_executor = TradeExecutor(SYMBOL)
order_manager = OrderManager(SYMBOL)
trend_model = TrendPredictor()

# Load training for classifier (example data, should be persisted in real case)
classifier.train(pd.DataFrame({
    "volatility": [0.5, 1.2, 0.3],
    "volume": [100000, 300000, 50000],
    "momentum": [1.1, -0.8, 0.4],
    "sentiment_score": [0.9, -0.6, 0.2],
    "strategy": ["swing", "scalping", "day_trading"]
}))

# Get latest price data
data = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Compute inputs
volatility = df['close'].rolling(window=10).std().iloc[-1]
volume = df['real_volume'].iloc[-1]
momentum = df['close'].pct_change().mean()

# News + sentiment
fetcher = NewsFetcher()
headlines = fetcher.fetch_news()
sentiment = sentiment_handler.process_news(headlines)
sent_score = sentiment['sentiment_score']

# Predict strategy
strategy_type = classifier.predict([volatility, volume, momentum, sent_score])

# Run selected strategy
if strategy_type == 'scalping':
    strat = ScalpingStrategy(SYMBOL)
elif strategy_type == 'day_trading':
    strat = DayTradingStrategy(SYMBOL)
else:
    strat = SwingTradingStrategy(SYMBOL)

signals = strat.generate_signals(df)
latest_signal = signals['signal'].iloc[-1]

# Execute trade
if latest_signal in ['buy', 'sell']:
    price = mt5.symbol_info_tick(SYMBOL).ask if latest_signal == 'buy' else mt5.symbol_info_tick(SYMBOL).bid
    stop_loss = price - 10 if latest_signal == 'buy' else price + 10
    take_profit = price + 20 if latest_signal == 'buy' else price - 20
    lot = lot_optimizer.optimize(
        balance=acc_info['balance'],
        entry_price=price,
        stop_price=stop_loss
    )
    trade_executor.send_order(action=latest_signal, lot=lot, price=price, sl=stop_loss, tp=take_profit)

mt5.shutdown()
