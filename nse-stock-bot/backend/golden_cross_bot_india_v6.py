import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import schedule
import time
from datetime import datetime, time as dt_time
import logging
import re
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from flask import Flask, jsonify, request, render_template_string
import threading
import os
import json

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')

class GoldenCrossBotIndia:
    def __init__(self, symbols=['HDFCBANK.NS', 'TCS.NS', 'RELIANCE.NS'], initial_capital=100000,
                 sender_email='your_email@gmail.com', email_password='your_app_password'):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.capital = {sym: initial_capital for sym in symbols}
        self.positions = {sym: 0 for sym in symbols}
        self.entry_prices = {sym: 0 for sym in symbols}
        self.stop_loss = 0.02
        self.risk_per_trade = 0.01
        self.models_long = {}
        self.models_short = {}
        self.dfs_long = {}
        self.dfs_short = {}
        self.positive_keywords = ['strong', 'record', 'upgrade', 'growth', 'positive', 'rally']
        self.negative_keywords = ['decline', 'downgrade', 'weak', 'loss', 'concern', 'drop']
        self.sender_email = sender_email or os.getenv('SENDER_EMAIL')
        self.email_password = email_password or os.getenv('EMAIL_PASSWORD')
        self.recipient_email = sender_email

    def send_email(self, subject, body):
        msg = MimeMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email
        msg['Subject'] = subject
        msg.attach(MimeText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(self.sender_email, self.email_password)
            server.send_message(msg)

    def fetch_data(self, symbol, period='2y', interval='1d'):
        try:
            df = yf.download(symbol, period=period, interval=interval)
            if df.empty:
                logger.error(f"No data fetched for {symbol}")
                return pd.DataFrame()
            df['short_ma'] = df['Close'].rolling(window=50).mean()
            df['long_ma'] = df['Close'].rolling(window=200).mean()
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_news(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news if hasattr(ticker, 'news') else []
            return [item['title'] for item in news[:3]]
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def analyze_sentiment(self, headlines):
        if not headlines:
            return 0.0
        score = sum(1 for h in headlines for kw in self.positive_keywords if kw in h.lower()) - \
                sum(1 for h in headlines for kw in self.negative_keywords if kw in h.lower())
        return max(min(score / len(headlines), 1.0), -1.0)

    def calculate_indicators(self, df, short_period, long_period):
        df['rsi'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.bbands(df['Close']).T
        df['stoch_k'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHk_14_3_3']
        df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['volume_ma'] = ta.sma(df['Volume'], length=20)
        return df.fillna(method='ffill').fillna(method='bfill')

    def train_ml_model(self, df, features, models):
        if df.empty or len(df) < 200:
            return
        df['target'] = np.where(df['short_ma'].shift(-1) > df['long_ma'].shift(-1), 1, 0)
        X = df[features].dropna()
        y = df.loc[X.index, 'target']
        if len(X) == 0 or len(y) == 0:
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[df.index[0]] = model

    def generate_signals(self, symbol, df, short_period, long_period, models):
        if df.empty or len(df) < long_period:
            return 0
        short_ma = df['short_ma'].iloc[-1]
        long_ma = df['long_ma'].iloc[-1]
        if short_ma > long_ma and df['short_ma'].iloc[-2] <= df['long_ma'].iloc[-2]:
            return 1  # Buy
        elif short_ma < long_ma and df['short_ma'].iloc[-2] >= df['long_ma'].iloc[-2]:
            return -1  # Sell
        return 0  # Hold

    def generate_signal(self, symbol):
        if symbol not in self.dfs_long:
            self.dfs_long[symbol] = self.fetch_data(symbol, '2y', '1d')
            self.dfs_long[symbol] = self.calculate_indicators(self.dfs_long[symbol], 50, 200)
            self.train_ml_model(self.dfs_long[symbol], ['short_ma', 'long_ma', 'rsi', 'Volume'], self.models_long)
        if symbol not in self.dfs_short:
            self.dfs_short[symbol] = self.fetch_data(symbol, '5d', '1h')
            self.dfs_short[symbol] = self.calculate_indicators(self.dfs_short[symbol], 5, 20)
            self.train_ml_model(self.dfs_short[symbol], ['short_ma', 'long_ma', 'rsi', 'Volume'], self.models_short)
        long_signal = self.generate_signals(symbol, self.dfs_long[symbol], 50, 200, self.models_long)
        short_signal = self.generate_signals(symbol, self.dfs_short[symbol], 5, 20, self.models_short)
        headlines = self.fetch_news(symbol)
        sentiment = self.analyze_sentiment(headlines)
        combined_signal = 1 if (long_signal >= 0 and short_signal == 1) or sentiment > 0.5 else -1 if (long_signal <= 0 and short_signal == -1) or sentiment < -0.5 else 0
        return combined_signal, sentiment, headlines, long_signal, short_signal

    def get_latest_data(self):
        data = {}
        for symbol in self.symbols:
            df_long = self.dfs_long.get(symbol, self.fetch_data(symbol, '2y', '1d'))
            df_short = self.dfs_short.get(symbol, self.fetch_data(symbol, '5d', '1h'))
            df_long = self.calculate_indicators(df_long, 50, 200)
            df_short = self.calculate_indicators(df_short, 5, 20)
            signal, sentiment, headlines, long_sig, short_sig = self.generate_signal(symbol)
            latest = df_long.iloc[-1]
            recent_df = yf.download(symbol, period='1d', interval='1h')
            recent_prices = recent_df['Close'].tail(5).tolist() if len(recent_df) >= 5 else [latest['Close']] * 5
            analysis = {
                'rsi': round(latest['rsi'], 2),
                'macd': round(latest['macd'] - latest['macd_signal'], 2),
                'bb_position': 'Upper' if latest['Close'] > latest['bb_upper'] else 'Lower' if latest['Close'] < latest['bb_lower'] else 'Middle',
                'volume_confirm': 'High' if latest['Volume'] > latest['volume_ma'] * 1.2 else 'Low',
                'stoch': round(latest['stoch_k'], 2),
                'atr': round(latest['atr'], 2),
                'insights': [
                    f"RSI {latest['rsi']}: {'Oversold (Buy?)' if latest['rsi'] < 30 else 'Overbought (Sell?)' if latest['rsi'] > 70 else 'Neutral'}",
                    f"MACD: {latest['macd'] - latest['macd_signal']:.2f} {'Bullish' if latest['macd'] > latest['macd_signal'] else 'Bearish'}",
                    f"Volume: {latest['volume_confirm']} (Confirms trend)"
                ]
            }
            data[symbol] = {
                'price': latest['Close'],
                'combined_signal': 'Buy' if signal == 1 else 'Sell' if signal == -1 else 'Hold',
                'long_signal': 'Buy' if long_sig == 1 else 'Sell' if long_sig == -1 else 'Hold',
                'short_signal': 'Buy' if short_sig == 1 else 'Sell' if short_sig == -1 else 'Hold',
                'sentiment': round(sentiment, 2),
                'position': 'Long' if self.positions[symbol] == 1 else 'None',
                'capital': round(self.capital[symbol], 2),
                'news': [{'text': h, 'type': 'positive' if sentiment > 0.5 else 'negative' if sentiment < -0.5 else 'neutral'} for h in headlines],
                'recent_prices': recent_prices,
                'price_labels': [f'-{5-i}h' for i in range(5)],
                **analysis
            }
        return data

    def simulate_trade(self, symbol, side):
        if symbol not in self.symbols: return {'error': 'Unknown symbol'}
        price = self.dfs_long[symbol]['Close'].iloc[-1]
        qty = int((self.risk_per_trade * self.capital[symbol]) / (price * self.stop_loss))
        if side == 'buy' and self.positions[symbol] == 0:
            self.positions[symbol] = 1
            self.entry_prices[symbol] = price
            self.capital[symbol] -= qty * price
            logger.info(f"PAPER BUY {symbol}: {qty} shares @ ₹{price}")
        elif side == 'sell' and self.positions[symbol] == 1:
            self.positions[symbol] = 0
            self.capital[symbol] += qty * (price - self.entry_prices[symbol])
            logger.info(f"PAPER SELL {symbol}: {qty} shares @ ₹{price}")
        return {'status': 'success', 'capital': self.capital[symbol], 'position': self.positions[symbol]}

    def backtest(self, symbol, years=2):
        df = self.fetch_data(symbol, f'{years}y', '1d')
        if df.empty: return
        df = self.calculate_indicators(df, 50, 200)
        total_profit = 0
        position = 0
        for i in range(1, len(df)):
            signal = self.generate_signals(symbol, df.iloc[:i+1], 50, 200, {})
            if signal == 1 and position == 0:
                position = 1
                entry_price = df['Close'].iloc[i]
            elif signal == -1 and position == 1:
                position = 0
                exit_price = df['Close'].iloc[i]
                total_profit += exit_price - entry_price
        logger.info(f"Backtest {symbol} ({years}y): Profit ₹{total_profit:.2f}")

    def check_market_hours(self):
        ist = datetime.utcnow() + pd.Timedelta(hours=5, minutes=30)
        market_open = dt_time(9, 15) <= ist.time() <= dt_time(15, 30) and ist.weekday() < 5
        return market_open

    def monitor_live(self):
        while True:
            if self.check_market_hours():
                data = self.get_latest_data()
                for symbol in self.symbols:
                    signal = data[symbol]['combined_signal']
                    if signal in ['Buy', 'Sell']:
                        self.send_email(f"Signal for {symbol}", f"Signal: {signal} @ ₹{data[symbol]['price']}")
            time.sleep(300)  # Check every 5 minutes

    def run(self):
        schedule.every(5).minutes.do(self.get_latest_data)
        schedule.every().day.at("09:00").do(lambda: [self.backtest(sym) for sym in self.symbols])
        while True:
            schedule.run_pending()
            time.sleep(1)

bot = GoldenCrossBotIndia(sender_email=None, email_password=None)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/status')
def get_status():
    return jsonify({'market_open': bot.check_market_hours(), 'stocks': [{'symbol': sym, **bot.get_latest_data()[sym]} for sym in bot.symbols], 'last_check': datetime.now().isoformat()})

@app.route('/trade', methods=['POST'])
def trade():
    data = request.get_json()
    symbol, side = data.get('symbol'), data.get('side')
    result = bot.simulate_trade(symbol, side)
    return jsonify(result)

if __name__ == "__main__":
    threading.Thread(target=bot.run, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)