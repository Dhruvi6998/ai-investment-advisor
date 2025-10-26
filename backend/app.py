from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import sqlite3
from contextlib import contextmanager
import re
import time

app = Flask(__name__)

# Secret key for sessions
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Session settings
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Initialize CORS
CORS(app, supports_credentials=True)

def is_allowed_origin(origin):
    if not origin:
        return False
    if origin.startswith('http://localhost:'):
        return True
    vercel_patterns = [
        r'^https://.*\.vercel\.app$',
        r'^https://.*-.*\.vercel\.app$',
    ]
    return any(re.match(p, origin) for p in vercel_patterns)

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin and is_allowed_origin(origin):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response

DATABASE = 'investment_advisor.db'

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_database():
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                quantity REAL NOT NULL,
                purchase_price REAL NOT NULL,
                purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, ticker)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                current_price REAL,
                predicted_price REAL,
                recommendation TEXT,
                confidence TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        print("✅ Database initialized successfully!")

class Database:
    @staticmethod
    def get_user_by_username(username):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            return cursor.fetchone()
    
    @staticmethod
    def create_user(username, password, email=None):
        with get_db() as conn:
            cursor = conn.cursor()
            password_hash = generate_password_hash(password)
            cursor.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, password_hash, email))
            return cursor.lastrowid
    
    @staticmethod
    def update_last_login(user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
    
    @staticmethod
    def get_portfolio(user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM portfolio WHERE user_id = ? ORDER BY purchase_date DESC', (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def add_stock(user_id, ticker, quantity, purchase_price, notes=None):
        with get_db() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO portfolio (user_id, ticker, quantity, purchase_price, notes) VALUES (?, ?, ?, ?, ?)',
                    (user_id, ticker, quantity, purchase_price, notes))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                cursor.execute('''UPDATE portfolio 
                    SET quantity = quantity + ?, 
                        purchase_price = (purchase_price * quantity + ? * ?) / (quantity + ?)
                    WHERE user_id = ? AND ticker = ?''',
                    (quantity, purchase_price, quantity, quantity, user_id, ticker))
                return None
    
    @staticmethod
    def remove_stock(user_id, ticker):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM portfolio WHERE user_id = ? AND ticker = ?', (user_id, ticker))
            return cursor.rowcount > 0
    
    @staticmethod
    def update_stock(user_id, ticker, quantity=None, purchase_price=None):
        with get_db() as conn:
            cursor = conn.cursor()
            if quantity is not None and purchase_price is not None:
                cursor.execute('UPDATE portfolio SET quantity = ?, purchase_price = ? WHERE user_id = ? AND ticker = ?',
                    (quantity, purchase_price, user_id, ticker))
            elif quantity is not None:
                cursor.execute('UPDATE portfolio SET quantity = ? WHERE user_id = ? AND ticker = ?',
                    (quantity, user_id, ticker))
            return cursor.rowcount > 0
    
    @staticmethod
    def save_analysis(user_id, ticker, current_price, predicted_price, recommendation, confidence):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO analysis_history 
                (user_id, ticker, current_price, predicted_price, recommendation, confidence)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (user_id, ticker, current_price, predicted_price, recommendation, confidence))
    
    @staticmethod
    def get_analysis_history(user_id, ticker=None, limit=10):
        with get_db() as conn:
            cursor = conn.cursor()
            if ticker:
                cursor.execute('SELECT * FROM analysis_history WHERE user_id = ? AND ticker = ? ORDER BY analysis_date DESC LIMIT ?',
                    (user_id, ticker, limit))
            else:
                cursor.execute('SELECT * FROM analysis_history WHERE user_id = ? ORDER BY analysis_date DESC LIMIT ?',
                    (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]

class AdvancedAIAdvisor:
    @staticmethod
    def fetch_stock_data(ticker, period="2y"):
        """Fetch stock data with aggressive retry and multiple methods"""
        try:
            print(f"📊 Fetching {ticker}...")
            
            # Method 1: Direct download (most reliable)
            try:
                time.sleep(0.3)  # Rate limiting
                df = yf.download(ticker, period=period, progress=False, timeout=30)
                
                if df is not None and not df.empty and len(df) > 0:
                    print(f"   ✓ Method 1 success: {len(df)} points")
                    current_price = float(df['Close'].iloc[-1])
                    
                    # Get info separately
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info or {}
                    except:
                        info = {}
                    
                    return {
                        'success': True,
                        'data': df,
                        'info': info,
                        'current_price': current_price,
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE'),
                        'beta': info.get('beta', 1.0)
                    }
            except Exception as e1:
                print(f"   ✗ Method 1 failed: {str(e1)[:50]}")
            
            # Method 2: Ticker object with history
            try:
                time.sleep(0.3)
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)
                
                if df is not None and not df.empty and len(df) > 0:
                    print(f"   ✓ Method 2 success: {len(df)} points")
                    current_price = float(df['Close'].iloc[-1])
                    
                    try:
                        info = stock.info or {}
                    except:
                        info = {}
                    
                    return {
                        'success': True,
                        'data': df,
                        'info': info,
                        'current_price': current_price,
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE'),
                        'beta': info.get('beta', 1.0)
                    }
            except Exception as e2:
                print(f"   ✗ Method 2 failed: {str(e2)[:50]}")
            
            # Method 3: Try shorter periods
            for p in ["1y", "6mo", "3mo", "1mo", "5d"]:
                try:
                    time.sleep(0.3)
                    df = yf.download(ticker, period=p, progress=False, timeout=20)
                    
                    if df is not None and not df.empty and len(df) > 0:
                        print(f"   ✓ Method 3 ({p}) success: {len(df)} points")
                        current_price = float(df['Close'].iloc[-1])
                        
                        return {
                            'success': True,
                            'data': df,
                            'info': {},
                            'current_price': current_price,
                            'sector': 'Unknown',
                            'industry': 'Unknown',
                            'market_cap': 0,
                            'pe_ratio': None,
                            'beta': 1.0
                        }
                except:
                    continue
            
            print(f"   ❌ All methods failed for {ticker}")
            return {'success': False, 'error': f'Unable to fetch data for {ticker}'}
            
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        delta = prices_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50
    
    @staticmethod
    def calculate_macd(prices):
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        exp1 = prices_series.ewm(span=12, adjust=False).mean()
        exp2 = prices_series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return {
            'macd': macd.iloc[-1] if len(macd) > 0 else 0,
            'signal': signal.iloc[-1] if len(signal) > 0 else 0,
            'histogram': histogram.iloc[-1] if len(histogram) > 0 else 0
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20):
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        sma = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        current_price = prices_series.iloc[-1]
        
        if len(upper_band) > 0 and len(lower_band) > 0:
            band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
            position = ((current_price - lower_band.iloc[-1]) / band_width * 100) if band_width > 0 else 50
        else:
            position = 50
            
        return {
            'upper': upper_band.iloc[-1] if len(upper_band) > 0 else current_price,
            'lower': lower_band.iloc[-1] if len(lower_band) > 0 else current_price,
            'position': position
        }
    
    @staticmethod
    def predict_future_price(hist, days_ahead=30):
        if len(hist) < 60:
            return None
        
        prices = hist['Close'].values
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(prices), len(prices) + days_ahead).reshape(-1, 1)
        linear_pred = model.predict(future_X)[-1]
        
        prices_series = pd.Series(prices)
        ema_short = prices_series.ewm(span=20).mean()
        ema_long = prices_series.ewm(span=50).mean()
        ema_trend = ema_short.iloc[-1] - ema_long.iloc[-1]
        ema_pred = prices[-1] + (ema_trend * (days_ahead / 30))
        
        returns = prices_series.pct_change().dropna()
        avg_return = returns.mean()
        volatility = returns.std()
        historical_pred = prices[-1] * (1 + avg_return * days_ahead)
        
        predicted_price = (linear_pred * 0.4 + ema_pred * 0.3 + historical_pred * 0.3)
        confidence = max(0, min(100, 100 - (volatility * 100 * 2)))
        
        return {
            'predicted_price': round(predicted_price, 2),
            'current_price': round(prices[-1], 2),
            'change_pct': round(((predicted_price - prices[-1]) / prices[-1]) * 100, 2),
            'confidence': round(confidence, 1),
            'days_ahead': days_ahead,
            'volatility': round(volatility, 4)
        }
    
    @staticmethod
    def calculate_risk_score(stock_data, metrics):
        risk_factors = []
        
        volatility = metrics.get('volatility', 0)
        if volatility > 0.5:
            risk_factors.append(('Very High Volatility', 30))
        elif volatility > 0.35:
            risk_factors.append(('High Volatility', 20))
        elif volatility > 0.25:
            risk_factors.append(('Moderate Volatility', 10))
        else:
            risk_factors.append(('Low Volatility', 0))
        
        if metrics['current_price'] < metrics['sma_50']:
            if metrics['current_price'] < metrics['sma_20']:
                risk_factors.append(('Strong Downtrend', 25))
            else:
                risk_factors.append(('Downtrend', 15))
        else:
            risk_factors.append(('Uptrend', 0))
        
        rsi = metrics.get('rsi', 50)
        if rsi > 70:
            risk_factors.append(('Overbought (RSI)', 15))
        elif rsi < 30:
            risk_factors.append(('Oversold (RSI)', 10))
        else:
            risk_factors.append(('Neutral RSI', 0))
        
        if metrics['profit_loss_pct'] < -20:
            risk_factors.append(('Heavy Losses', 20))
        elif metrics['profit_loss_pct'] < -10:
            risk_factors.append(('Significant Losses', 12))
        elif metrics['profit_loss_pct'] < 0:
            risk_factors.append(('Minor Losses', 5))
        
        if metrics.get('weight', 0) > 30:
            risk_factors.append(('Over-concentrated', 10))
        elif metrics.get('weight', 0) > 20:
            risk_factors.append(('High Concentration', 5))
        
        total_risk = sum(score for _, score in risk_factors)
        
        return {
            'score': min(100, total_risk),
            'level': 'Low' if total_risk < 30 else 'Moderate' if total_risk < 60 else 'High',
            'factors': [factor for factor, _ in risk_factors]
        }
    
    @staticmethod
    def calculate_comprehensive_metrics(ticker, quantity, purchase_price):
        stock_data = AdvancedAIAdvisor.fetch_stock_data(ticker)
        if not stock_data['success']:
            return None
        
        hist = stock_data['data']
        current_price = stock_data['current_price']
        total_value = current_price * quantity
        total_cost = purchase_price * quantity
        profit_loss = total_value - total_cost
        profit_loss_pct = ((current_price - purchase_price) / purchase_price) * 100
        
        prices = hist['Close'].values if isinstance(hist['Close'], pd.Series) else hist['Close']
        prices_series = pd.Series(prices)
        
        sma_20 = prices_series.rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else current_price
        sma_50 = prices_series.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else current_price
        sma_200 = prices_series.rolling(window=200).mean().iloc[-1] if len(prices) >= 200 else current_price
        
        rsi = AdvancedAIAdvisor.calculate_rsi(prices)
        macd = AdvancedAIAdvisor.calculate_macd(prices)
        bollinger = AdvancedAIAdvisor.calculate_bollinger_bands(prices)
        
        returns = prices_series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        returns_7d = ((prices[-1] - prices[-7]) / prices[-7] * 100) if len(prices) >= 7 else 0
        returns_30d = ((prices[-1] - prices[-30]) / prices[-30] * 100) if len(prices) >= 30 else 0
        returns_90d = ((prices[-1] - prices[-90]) / prices[-90] * 100) if len(prices) >= 90 else 0
        returns_1y = ((prices[-1] - prices[-252]) / prices[-252] * 100) if len(prices) >= 252 else 0
        
        volumes = hist['Volume'].values if isinstance(hist['Volume'], pd.Series) else hist['Volume']
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else avg_volume
        volume_trend = 'Increasing' if recent_volume > avg_volume * 1.2 else 'Decreasing' if recent_volume < avg_volume * 0.8 else 'Normal'
        
        prediction = AdvancedAIAdvisor.predict_future_price(hist, days_ahead=30)
        
        metrics = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'quantity': quantity,
            'total_value': round(total_value, 2),
            'profit_loss': round(profit_loss, 2),
            'profit_loss_pct': round(profit_loss_pct, 2),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'sma_200': round(sma_200, 2),
            'rsi': round(rsi, 2),
            'macd': round(macd['macd'], 4),
            'macd_signal': round(macd['signal'], 4),
            'macd_histogram': round(macd['histogram'], 4),
            'bollinger_upper': round(bollinger['upper'], 2),
            'bollinger_lower': round(bollinger['lower'], 2),
            'bollinger_position': round(bollinger['position'], 1),
            'volatility': round(volatility, 4),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'beta': round(stock_data.get('beta', 1.0), 2),
            'returns_7d': round(returns_7d, 2),
            'returns_30d': round(returns_30d, 2),
            'returns_90d': round(returns_90d, 2),
            'returns_1y': round(returns_1y, 2),
            'volume_trend': volume_trend,
            'avg_volume': int(avg_volume),
            'sector': stock_data['sector'],
            'industry': stock_data['industry'],
            'market_cap': stock_data.get('market_cap', 0),
            'pe_ratio': stock_data.get('pe_ratio'),
            'prediction': prediction
        }
        
        metrics['risk_analysis'] = AdvancedAIAdvisor.calculate_risk_score(stock_data, metrics)
        
        return metrics
    
    @staticmethod
    def generate_ai_recommendation(metrics):
        score = 0
        reasons = []
        signals = {'bullish': 0, 'bearish': 0}
        
        if metrics['current_price'] > metrics['sma_20'] > metrics['sma_50']:
            score += 3
            signals['bullish'] += 1
            reasons.append(f"✓ Strong uptrend: Price above SMA20 and SMA50")
        elif metrics['current_price'] < metrics['sma_20'] < metrics['sma_50']:
            score -= 3
            signals['bearish'] += 1
            reasons.append(f"✗ Downtrend: Price below both moving averages")
        
        if metrics['rsi'] < 30:
            score += 2
            signals['bullish'] += 1
            reasons.append(f"✓ Oversold (RSI: {metrics['rsi']:.1f}) - Potential bounce")
        elif metrics['rsi'] > 70:
            score -= 2
            signals['bearish'] += 1
            reasons.append(f"✗ Overbought (RSI: {metrics['rsi']:.1f}) - Potential correction")
        
        if metrics['macd'] > metrics['macd_signal'] and metrics['macd_histogram'] > 0:
            score += 2
            signals['bullish'] += 1
            reasons.append(f"✓ Bullish MACD crossover")
        elif metrics['macd'] < metrics['macd_signal'] and metrics['macd_histogram'] < 0:
            score -= 2
            signals['bearish'] += 1
            reasons.append(f"✗ Bearish MACD")
        
        if metrics['bollinger_position'] < 20:
            score += 1
            signals['bullish'] += 1
            reasons.append(f"✓ Near lower Bollinger Band")
        elif metrics['bollinger_position'] > 80:
            score -= 1
            signals['bearish'] += 1
            reasons.append(f"✗ Near upper Bollinger Band")
        
        if metrics['profit_loss_pct'] > 30:
            score += 2
            reasons.append(f"✓ Excellent gains: +{metrics['profit_loss_pct']:.1f}%")
        elif metrics['profit_loss_pct'] > 15:
            score += 1
            reasons.append(f"✓ Strong performance: +{metrics['profit_loss_pct']:.1f}%")
        elif metrics['profit_loss_pct'] < -20:
            score -= 3
            reasons.append(f"✗ Heavy losses: {metrics['profit_loss_pct']:.1f}%")
        elif metrics['profit_loss_pct'] < -10:
            score -= 2
            reasons.append(f"✗ Significant losses: {metrics['profit_loss_pct']:.1f}%")
        
        if metrics['returns_30d'] > 10:
            score += 2
            signals['bullish'] += 1
            reasons.append(f"✓ Strong momentum: +{metrics['returns_30d']:.1f}% (30d)")
        elif metrics['returns_30d'] < -10:
            score -= 2
            signals['bearish'] += 1
            reasons.append(f"✗ Weak momentum: {metrics['returns_30d']:.1f}% (30d)")
        
        if metrics['prediction']:
            pred = metrics['prediction']
            if pred['change_pct'] > 10 and pred['confidence'] > 60:
                score += 3
                signals['bullish'] += 1
                reasons.append(f"✓ AI predicts +{pred['change_pct']:.1f}% in {pred['days_ahead']} days")
            elif pred['change_pct'] < -10 and pred['confidence'] > 60:
                score -= 3
                signals['bearish'] += 1
                reasons.append(f"✗ AI predicts {pred['change_pct']:.1f}% decline")
        
        risk = metrics['risk_analysis']
        if risk['level'] == 'High':
            score -= 2
            reasons.append(f"⚠ High risk profile")
        elif risk['level'] == 'Low':
            score += 1
            reasons.append(f"✓ Low risk profile")
        
        if metrics['volume_trend'] == 'Increasing':
            score += 1
            signals['bullish'] += 1
            reasons.append(f"✓ Volume increasing")
        
        if score >= 6:
            action = "STRONG_BUY"
            confidence = "Very High"
            summary = "Multiple bullish indicators align. Strong buy opportunity."
        elif score >= 3:
            action = "BUY"
            confidence = "High"
            summary = "Positive technical signals. Good buying opportunity."
        elif score >= 1:
            action = "HOLD_OR_BUY"
            confidence = "Medium"
            summary = "Some positive signals. Consider adding to position."
        elif score >= -1:
            action = "HOLD"
            confidence = "Medium"
            summary = "Mixed signals. Hold current position."
        elif score >= -3:
            action = "CONSIDER_SELLING"
            confidence = "Medium"
            summary = "Some concerning signals. Monitor closely."
        elif score >= -5:
            action = "SELL"
            confidence = "High"
            summary = "Multiple bearish indicators. Consider reducing position."
        else:
            action = "STRONG_SELL"
            confidence = "Very High"
            summary = "Strong bearish signals. Exit position recommended."
        
        return {
            'action': action,
            'confidence': confidence,
            'summary': summary,
            'score': score,
            'reasons': reasons,
            'signals': signals
        }
    
    @staticmethod
    def analyze_portfolio_comprehensive(holdings):
        if not holdings:
            return {
                'analysis': [],
                'portfolio_health': 'No holdings',
                'diversification_score': 0,
                'suggested_actions': [],
                'sector_analysis': {}
            }
        
        analysis = []
        total_value = 0
        sector_allocation = defaultdict(float)
        
        for holding in holdings:
            metrics = AdvancedAIAdvisor.calculate_comprehensive_metrics(
                holding['ticker'], holding['quantity'], holding['purchase_price'])
            if metrics:
                metrics['weight'] = 0
                recommendation = AdvancedAIAdvisor.generate_ai_recommendation(metrics)
                metrics['recommendation'] = recommendation
                analysis.append(metrics)
                total_value += metrics['total_value']
                sector_allocation[metrics['sector']] += metrics['total_value']
        
        for stock in analysis:
            stock['weight'] = (stock['total_value'] / total_value * 100) if total_value > 0 else 0
        
        sector_weights = {sector: (value/total_value*100) if total_value > 0 else 0 
                         for sector, value in sector_allocation.items()}
        
        num_sectors = len(sector_weights)
        sector_balance = 100 - (max(sector_weights.values()) if sector_weights else 100)
        num_stocks = len(analysis)
        diversification_score = min(100, (num_sectors * 15) + (num_stocks * 5) + sector_balance)
        
        total_pl = sum(s['profit_loss'] for s in analysis)
        avg_pl_pct = np.mean([s['profit_loss_pct'] for s in analysis])
        avg_risk = np.mean([s['risk_analysis']['score'] for s in analysis])
        
        if avg_pl_pct > 15 and diversification_score > 70 and avg_risk < 40:
            health = "Excellent"
        elif avg_pl_pct > 10 and diversification_score > 60:
            health = "Very Good"
        elif avg_pl_pct > 5 and diversification_score > 50:
            health = "Good"
        elif avg_pl_pct > 0 and diversification_score > 40:
            health = "Fair"
        else:
            health = "Needs Attention"
        
        suggested_actions = AdvancedAIAdvisor.generate_portfolio_actions(
            analysis, sector_weights, total_value, diversification_score)
        
        return {
            'analysis': analysis,
            'portfolio_health': health,
            'total_value': round(total_value, 2),
            'total_profit_loss': round(total_pl, 2),
            'avg_return': round(avg_pl_pct, 2),
            'avg_risk_score': round(avg_risk, 1),
            'diversification_score': round(diversification_score, 1),
            'num_stocks': num_stocks,
            'num_sectors': num_sectors,
            'sector_analysis': sector_weights,
            'suggested_actions': suggested_actions
        }
    
    @staticmethod
    def generate_portfolio_actions(analysis, sector_weights, total_value, div_score):
        actions = []
        
        max_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else (None, 0)
        if max_sector[1] > 40:
            actions.append({
                'priority': 'High',
                'type': 'Diversification',
                'action': f"Reduce {max_sector[0]} exposure from {max_sector[1]:.1f}% to below 35%",
                'reason': 'Sector over-concentration increases portfolio risk'
            })
        
        for stock in analysis:
            rec = stock['recommendation']
            if rec['action'] in ['STRONG_SELL', 'SELL']:
                actions.append({
                    'priority': 'High',
                    'type': 'Exit Position',
                    'ticker': stock['ticker'],
                    'action': f"Sell {stock['ticker']} ({rec['action']})",
                    'reason': rec['summary']
                })
            elif rec['action'] in ['STRONG_BUY', 'BUY']:
                actions.append({
                    'priority': 'Medium',
                    'type': 'Add Position',
                    'ticker': stock['ticker'],
                    'action': f"Buy more {stock['ticker']} ({rec['action']})",
                    'reason': rec['summary']
                })
        
        if div_score < 50:
            actions.append({
                'priority': 'High',
                'type': 'Diversification',
                'action': f"Add {8 - len(analysis)} more stocks across different sectors",
                'reason': f"Current diversification score is {div_score:.0f}/100"
            })
        
        return sorted(actions, key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2}[x['priority']])

# Routes
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    
    if Database.get_user_by_username(username):
        return jsonify({'success': False, 'message': 'User already exists'}), 400
    
    user_id = Database.create_user(username, password, email)
    return jsonify({'success': True, 'message': 'Registration successful', 'user_id': user_id})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    
    user = Database.get_user_by_username(username)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    if not check_password_hash(user['password'], password):
        return jsonify({'success': False, 'message': 'Invalid password'}), 401
    
    Database.update_last_login(user['id'])
    session['user_id'] = user['id']
    session['username'] = username
    
    return jsonify({'success': True, 'message': 'Login successful', 'username': username, 'user_id': user['id']})

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/check-auth', methods=['GET'])
def check_auth():
    user_id = session.get('user_id')
    username = session.get('username')
    return jsonify({'authenticated': user_id is not None, 'username': username, 'user_id': user_id})

@app.route('/portfolio', methods=['GET'])
def get_portfolio():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    portfolio = Database.get_portfolio(user_id)
    return jsonify({'success': True, 'portfolio': portfolio})

@app.route('/portfolio/add', methods=['POST'])
def add_stock():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    data = request.json
    ticker = data.get('ticker', '').upper()
    quantity = float(data.get('quantity'))
    purchase_price = float(data.get('purchase_price'))
    notes = data.get('notes', '')
    
    stock_data = AdvancedAIAdvisor.fetch_stock_data(ticker, period="5d")
    if not stock_data['success']:
        return jsonify({'success': False, 'message': 'Invalid ticker symbol'}), 400
    
    stock_id = Database.add_stock(user_id, ticker, quantity, purchase_price, notes)
    
    return jsonify({'success': True, 'message': 'Stock added to portfolio', 'stock_id': stock_id})

@app.route('/portfolio/remove', methods=['POST'])
def remove_stock():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    data = request.json
    ticker = data.get('ticker', '').upper()
    success = Database.remove_stock(user_id, ticker)
    
    if success:
        return jsonify({'success': True, 'message': 'Stock removed from portfolio'})
    else:
        return jsonify({'success': False, 'message': 'Stock not found in portfolio'}), 404

@app.route('/portfolio/update', methods=['POST'])
def update_stock():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    data = request.json
    ticker = data.get('ticker', '').upper()
    quantity = data.get('quantity')
    purchase_price = data.get('purchase_price')
    success = Database.update_stock(user_id, ticker, quantity, purchase_price)
    
    if success:
        return jsonify({'success': True, 'message': 'Stock updated'})
    else:
        return jsonify({'success': False, 'message': 'Stock not found'}), 404

@app.route('/analyze', methods=['GET'])
def analyze_portfolio():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    portfolio = Database.get_portfolio(user_id)
    analysis = AdvancedAIAdvisor.analyze_portfolio_comprehensive(portfolio)
    
    for stock in analysis.get('analysis', []):
        if stock.get('prediction'):
            Database.save_analysis(user_id, stock['ticker'], stock['current_price'],
                stock['prediction']['predicted_price'], stock['recommendation']['action'],
                stock['recommendation']['confidence'])
    
    return jsonify({'success': True, 'analysis': analysis})

@app.route('/analysis-history', methods=['GET'])
def get_analysis_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    ticker = request.args.get('ticker')
    limit = int(request.args.get('limit', 10))
    history = Database.get_analysis_history(user_id, ticker, limit)
    
    return jsonify({'success': True, 'history': history})

@app.route('/rebalance', methods=['GET'])
def get_rebalancing():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    portfolio = Database.get_portfolio(user_id)
    
    if not portfolio or len(portfolio) < 2:
        return jsonify({'success': True, 'rebalancing': {
            'rebalancing_needed': False,
            'message': 'Need at least 2 stocks to analyze rebalancing'
        }})
    
    analysis = []
    total_value = 0
    
    for holding in portfolio:
        metrics = AdvancedAIAdvisor.calculate_comprehensive_metrics(
            holding['ticker'], holding['quantity'], holding['purchase_price'])
        if metrics:
            analysis.append(metrics)
            total_value += metrics['total_value']
    
    for stock in analysis:
        stock['current_weight'] = (stock['total_value'] / total_value * 100) if total_value > 0 else 0
    
    target_weight = 100 / len(analysis)
    rebalancing_actions = []
    
    for stock in analysis:
        diff = stock['current_weight'] - target_weight
        if abs(diff) > 10:
            target_value = (target_weight / 100) * total_value
            current_value = stock['total_value']
            dollar_diff = target_value - current_value
            shares_diff = dollar_diff / stock['current_price']
            
            if diff > 0:
                action = f"Reduce {stock['ticker']} by {abs(diff):.1f}% (sell ~{abs(int(shares_diff))} shares ≈ ${abs(dollar_diff):.0f})"
            else:
                action = f"Increase {stock['ticker']} by {abs(diff):.1f}% (buy ~{abs(int(shares_diff))} shares ≈ ${abs(dollar_diff):.0f})"
            
            rebalancing_actions.append({
                'ticker': stock['ticker'],
                'action': action,
                'current_weight': round(stock['current_weight'], 2),
                'target_weight': round(target_weight, 2),
                'difference': round(diff, 2),
                'shares_to_trade': int(shares_diff),
                'value_to_trade': round(dollar_diff, 2)
            })
    
    return jsonify({'success': True, 'rebalancing': {
        'rebalancing_needed': len(rebalancing_actions) > 0,
        'actions': rebalancing_actions,
        'current_allocation': [{'ticker': s['ticker'], 'weight': round(s['current_weight'], 2),
            'value': round(s['total_value'], 2)} for s in analysis],
        'target_weight': round(target_weight, 2),
        'total_portfolio_value': round(total_value, 2),
        'num_stocks': len(analysis)
    }})

@app.route('/stock-search', methods=['GET'])
def search_stock():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker required'}), 400
    
    stock_data = AdvancedAIAdvisor.fetch_stock_data(ticker, period="5d")
    if not stock_data['success']:
        return jsonify({'success': False, 'message': 'Stock not found'}), 404
    
    info = stock_data['info']
    return jsonify({'success': True, 'stock': {
        'ticker': ticker,
        'name': info.get('longName', ticker),
        'current_price': stock_data['current_price'],
        'currency': info.get('currency', 'USD'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A')
    }})

@app.route('/db-info', methods=['GET'])
def get_db_info():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT username, email, created_at, last_login FROM users WHERE id = ?', (user_id,))
        user_info = dict(cursor.fetchone())
        cursor.execute('SELECT COUNT(*) as stock_count FROM portfolio WHERE user_id = ?', (user_id,))
        portfolio_stats = dict(cursor.fetchone())
        cursor.execute('SELECT COUNT(*) as analysis_count FROM analysis_history WHERE user_id = ?', (user_id,))
        analysis_stats = dict(cursor.fetchone())
    
    return jsonify({'success': True, 'user_info': user_info, 'portfolio_stats': portfolio_stats,
        'analysis_stats': analysis_stats, 'database_file': DATABASE})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'AI Investment Advisor API - yfinance'}), 200

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'AI Investment Advisor API',
        'version': '2.0 - yfinance',
        'data_provider': 'Yahoo Finance (yfinance)',
        'endpoints': {
            'auth': ['/register', '/login', '/logout', '/check-auth'],
            'portfolio': ['/portfolio', '/portfolio/add', '/portfolio/remove', '/portfolio/update'],
            'analysis': ['/analyze', '/analysis-history', '/rebalance', '/stock-search'],
            'info': ['/db-info', '/health']
        }
    }), 200

if __name__ == '__main__':
    print("🔧 Initializing database...")
    init_database()
    
    print("\n🚀 AI Investment Advisor Starting...")
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"📊 Server running on port {port}")
    print(f"📡 Data Provider: Yahoo Finance (yfinance)")
    print("\n🌍 GLOBAL MARKET SUPPORT:")
    print("   ✓ US Stocks: AAPL, MSFT, GOOGL, TSLA")
    print("   ✓ India NSE: TCS.NS, RELIANCE.NS, INFY.NS")
    print("   ✓ India BSE: TCS.BO, RELIANCE.BO, INFY.BO")
    print("   ✓ UK, Europe, Asia markets")
    print("\n✅ Ready to analyze portfolios!")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)