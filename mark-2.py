import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import ccxt
import matplotlib.pyplot as plt
import pandas as pd
import time as t
import numpy as np
from datetime import datetime, timedelta
import Rbeast as rb
import logging
import os
import sys
import pickle

class NullWriter:
    def write(self, text): pass
    def flush(self): pass

TICKER = "BTC/USDT"
CONFIDENCE_THRESHOLD = 0.6
STOP_LOSS = -0.02
TAKE_PROFIT = 0.05
RISK_PER_TRADE = 0.02
TRAILING_STOP_RATE = 0.03
MA_WINDOW = 50
RSI_PERIOD = 14
MA_SHORT_WINDOW = 10
MA_LONG_WINDOW = MA_WINDOW
DATA_LOOKBACK_YEARS = 1
FETCH_INTERVAL_SECONDS = 1
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.002
CHUNK_DAYS = 90
FULL_REFIT_ON_START = True
first_run = True
trend = None
LOG_FILE = 'trade_log.csv'
CACHE_FILE = 'beast_cache.pkl'
LIVE_SAMPLES = 2000
FULL_SAMPLES = 8000
PLOT_TREND = False

capital = 10000
btc_holding = 0
entry_price = 0
max_price_since_entry = 0
min_price_since_entry = float('inf')
history = []
trade_count = 0
last_idx = -1
last_refit_ts = None

cooldown_days = 3
last_trade_ts = None
slope_window = 5
trailing_start_bars = 5
entry_idx = None
hold_count = 0  # track consecutive holds

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

with open(LOG_FILE, 'w') as f:
    f.write('timestamp,action,price,units,fee,capital,btc_holding\n')

print("🚀 Starting BTC Trading Simulator")

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
            
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up/down if down != 0 else 0
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

# add loop counter before Main trading loop
loop_counter = 0
# Main trading loop
while True:
    try:
        loop_counter += 1
        # default action and reset units/fee
        last_action = "HOLD"
        units = 0.0
        fee = 0.0
        print("Analyzing BTC price data... Please wait...")
        loop_start_time = t.time()
        start_date = (datetime.now() - timedelta(days=365*DATA_LOOKBACK_YEARS)).strftime('%Y-%m-%d')
    
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(TICKER, timeframe='1d', limit=DATA_LOOKBACK_YEARS*365)
        timestamps = [row[0] for row in ohlcv]
        price_vals = np.array([row[4] for row in ohlcv], dtype=float)
        price_series = pd.Series(price_vals, index=pd.to_datetime(timestamps, unit='ms'))
        price = price_series

        ma = price.rolling(window=MA_WINDOW).mean()
        rsi = calculate_rsi(price.values, RSI_PERIOD)

        ma_vals = ma.values
        rsi_vals = rsi
        price_vals = price.values
        
        prior = rb.args()
        prior.trendMinKnotNum = 1
        prior.trendMaxKnotNum = 10
        
        mcmc = rb.args()
        mcmc.samples = 8000
        mcmc.burnin = 200
        
        extra = rb.args()
        extra.quiet = True
        extra.printProgress = False
        extra.printParameter = False
        extra.printWarning = False
        
        ma = price.rolling(window=MA_WINDOW).mean()
        rsi_vals = calculate_rsi(price.values, RSI_PERIOD)
        fast_ma = price.rolling(window=MA_SHORT_WINDOW).mean().values
        slow_ma = price.rolling(window=MA_LONG_WINDOW).mean().values
        last_action = "HOLD"

        new_ts = price.index[-1]
        new_idx = len(price_vals) - 1
        if last_idx < new_idx:
            last_action = "HOLD"
            current_price = float(price_vals[new_idx])
            total_value = capital + btc_holding * current_price

            # Load cached BEAST fit or refit if outdated
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE,'rb') as f: cache = pickle.load(f)
                cache_ts = cache.get('last_refit_ts')
                if cache_ts and (new_ts - cache_ts).days < CHUNK_DAYS:
                    trend = cache['trend']; changepoints = cache['changepoints']; cp_probs = cache['cp_probs']
                    first_run = False; last_refit_ts = cache_ts

            # Fit or update BEAST trend model dynamically
            if first_run or last_refit_ts is None or (new_ts - last_refit_ts).days >= CHUNK_DAYS:
                mcmc.samples = FULL_SAMPLES if first_run else LIVE_SAMPLES
                mcmc.burnin = int(mcmc.samples * 0.25)
                model_data = price_vals if first_run else price_vals[-CHUNK_DAYS:]
                logging.info(f"Running BEAST on {'full data' if first_run else f'last {CHUNK_DAYS} days'}...")
                old_stdout = sys.stdout; sys.stdout = NullWriter()
                result = rb.beast(model_data, start=0, deltat=1, season='none', prior=prior, mcmc=mcmc, extra=extra)
                sys.stdout = old_stdout
                if first_run:
                    trend = np.array(result.trend.Y)
                else:
                    tail = np.array(result.trend.Y)
                    trend = np.concatenate([trend[:-CHUNK_DAYS], tail])
                raw_cp = np.array(result.trend.cp); valid = raw_cp[~np.isnan(raw_cp)]
                cp_list = [int(x) for x in valid]
                offset = 0 if first_run else len(trend) - CHUNK_DAYS
                changepoints = [offset + cp for cp in cp_list]
                try:
                    cp_probs = result.trend.prob.tolist()
                except:
                    cp_probs = [0.9] * len(changepoints)
                first_run = False; last_refit_ts = new_ts
                
                with open(CACHE_FILE,'wb') as f:
                    pickle.dump({'trend': trend, 'changepoints': changepoints, 'cp_probs': cp_probs, 'last_refit_ts': last_refit_ts}, f)
                if PLOT_TREND:
                    plt.figure(figsize=(12,6))
                    plt.plot(price.index, price.values, label='Price')
                    plt.plot(price.index[-len(trend):], trend, label='BEAST Trend')
                    plt.title("BTC Price vs Trend")
                    plt.legend()
                    plt.show()

            cp_active = new_idx in changepoints and cp_probs[changepoints.index(new_idx)] >= CONFIDENCE_THRESHOLD
            # Compute robust slope via rolling window regression
            if new_idx >= slope_window-1:
                y = trend[new_idx-slope_window+1:new_idx+1]
                x = np.arange(len(y))
                beast_slope = float(np.polyfit(x, y, 1)[0])
            else:
                beast_slope = 0.0
            price_slope = float(price_vals[new_idx] - price_vals[new_idx-1]) if new_idx>0 else 0.0
            slope_change = beast_slope if abs(beast_slope)>1e-6 else price_slope
            logging.debug(f"slope_change={slope_change:.4f}")
            trade_allowed = True
            if last_trade_ts and (new_ts - last_trade_ts).days < cooldown_days:
                trade_allowed = False

            fast = fast_ma[new_idx]; slow = slow_ma[new_idx]; rsi_val = rsi_vals[new_idx]
            # Check entry signals before opening a new position
            if cp_active and slope_change > 0 and fast > slow and rsi_val < 70 and capital > 0 and btc_holding == 0 and trade_allowed:
                risk_amount = RISK_PER_TRADE * total_value; exec_price = current_price*(1+SLIPPAGE_RATE)
                units = risk_amount/(exec_price*abs(STOP_LOSS)); fee=units*exec_price*FEE_RATE
                capital-=units*exec_price+fee; btc_holding+=units; entry_price=exec_price; max_price_since_entry=entry_price
                last_action = f"BUY @ {exec_price:.2f}"; trade_count+=1
                last_trade_ts = new_ts; entry_idx = new_idx
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{new_ts},BUY,{exec_price:.2f},{units:.6f},{fee:.2f},{capital:.2f},{btc_holding:.6f}\n")

            # Partial take-profit exit to capture gains while trend continues
            if btc_holding>0 and current_price >= entry_price * (1 + TAKE_PROFIT):
                units_to_sell = btc_holding / 2
                exec_price = current_price * (1 - SLIPPAGE_RATE)
                fee = units_to_sell * exec_price * FEE_RATE
                capital += units_to_sell * exec_price - fee
                btc_holding -= units_to_sell
                last_action = f"SELL 50% @ {exec_price:.2f} (profit)"
                trade_count += 1
                roi = (exec_price - entry_price) / entry_price * 100
                print(f"🎯 Partial profit exit: ROI = {roi:.2f}% | Capital = ${capital:.2f}")
                last_trade_ts = new_ts
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{price.index[new_idx]},{last_action},{exec_price:.2f},{units_to_sell:.5f},{fee:.5f},{capital:.2f},{btc_holding:.5f}\n")

            # Static stop-loss and trailing-stop exit logic to limit downside
            if btc_holding>0:
                max_price_since_entry = max(max_price_since_entry,current_price)
                static_hit = current_price<=entry_price*(1+STOP_LOSS)
                trail_hit = new_idx>=entry_idx+trailing_start_bars and current_price<=max_price_since_entry*(1-TRAILING_STOP_RATE)
                if static_hit or trail_hit:
                    units = btc_holding
                    exec_price = current_price * (1 - SLIPPAGE_RATE)
                    fee = units * exec_price * FEE_RATE
                    capital = units * exec_price - fee
                    btc_holding = 0
                    last_action = f"SELL @ {exec_price:.2f} (stop-loss)"
                    trade_count += 1
                    roi = (exec_price - entry_price) / entry_price * 100
                    print(f"⚠️ Stop-loss triggered: ROI = {roi:.2f}% | Capital = ${capital:.2f}")
                    last_trade_ts = new_ts
                    with open(LOG_FILE, 'a') as f:
                        f.write(f"{price.index[new_idx]},{last_action},{exec_price:.2f},{units:.5f},{fee:.5f},{capital:.2f},{btc_holding:.5f}\n")

            # Force buy or sell every 5 loops based on trend
            if loop_counter % 5 == 0:
                # determine trade type
                if slope_change > 0 and btc_holding == 0:
                    total_value = capital + btc_holding * current_price
                    exec_price = current_price * (1 + SLIPPAGE_RATE)
                    units = (RISK_PER_TRADE * total_value) / (exec_price * abs(STOP_LOSS))
                    fee = units * exec_price * FEE_RATE
                    capital -= units * exec_price + fee
                    btc_holding += units
                    entry_price = exec_price
                    last_action = f"FORCED BUY @ {exec_price:.2f}"
                elif slope_change <= 0 and btc_holding > 0:
                    exec_price = current_price * (1 - SLIPPAGE_RATE)
                    units = btc_holding
                    fee = units * exec_price * FEE_RATE
                    capital += units * exec_price - fee
                    btc_holding = 0
                    last_action = f"FORCED SELL @ {exec_price:.2f}"
                if units:
                    trade_count += 1; last_trade_ts = new_ts; entry_idx = new_idx
                    logging.info(f"{last_action} - Price: {exec_price:.2f} | Cap: ${capital:.2f} | BTC: {btc_holding:.6f}")

            # Record every iteration
            with open(LOG_FILE, 'a') as f:
                f.write(f"{new_ts},{last_action},{current_price:.2f},{units:.6f},{fee:.2f},{capital:.2f},{btc_holding:.6f}\n")

            logging.info(f"{new_ts.date()} - {last_action} - Capital: ${capital:.2f} | BTC: {btc_holding:.6f} | Total: ${capital + btc_holding * current_price:.2f}")

            last_idx = new_idx

        current_btc_price = float(price_vals[-1])
        total_value = capital + btc_holding * current_btc_price
        print("\033[H\033[J")
        print(f"💰 PORTFOLIO SUMMARY ({price.index[-1].date()})")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 BTC Price:      ${current_btc_price:,.2f}")
        print(f"💵 Cash:           ${capital:,.2f}")
        print(f"🪙 BTC Holdings:   {btc_holding:.5f} BTC (${btc_holding * current_btc_price:,.2f})")
        print(f"💼 Total Value:    ${total_value:,.2f}")
        print(f"🔄 Total Trades:   {trade_count}")
        print(f"🔄 Last Action:    {last_action}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        elapsed = t.time() - loop_start_time
        sleep_time = max(0, FETCH_INTERVAL_SECONDS - elapsed)
        logging.info(f"Waiting {sleep_time:.0f}s until next fetch...")
        t.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Simulation stopped by user.")
        break
    except Exception as e:
        logging.error(f"Error: {e}. Retrying in 30s...")
        t.sleep(30)

import pandas as pd
import os
with open(LOG_FILE, 'r') as f:
    lines = f.readlines()
if len(lines) < 2:
    logging.info('trade_log.csv has no trade entries, skipping performance chart.')
    sys.exit(0)
try:
    trades_df = pd.read_csv(LOG_FILE)
    if 'timestamp' not in trades_df.columns:
        trades_df.columns = ['timestamp','action','price','units','fee','capital','btc_holding']
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df.set_index('timestamp', inplace=True)
    trades_df['PortfolioValue'] = trades_df['capital'] + trades_df['btc_holding'] * trades_df['price']

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(trades_df['PortfolioValue'], label='Portfolio Value ($)', color='green', marker='o')
    plt.title("Portfolio Performance from Trade Log")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.legend()

    initial_value = trades_df['PortfolioValue'].iloc[0]
    profit_series = trades_df['PortfolioValue'] - initial_value
    plt.subplot(2, 1, 2)
    plt.plot(profit_series, label='Profit ($)', color='blue', marker='x')
    plt.xlabel("Timestamp")
    plt.ylabel("Profit ($)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    initial_value = trades_df['PortfolioValue'].iloc[0]
    final_value = trades_df['PortfolioValue'].iloc[-1]
    roi = (final_value - initial_value) / initial_value * 100
    print(f"\n💰 Final Portfolio: ${final_value:,.2f}")
    print(f"📊 ROI: {roi:.2f}%")
    print(f"🔄 Total Trades: {trade_count}")
    print("\nPortfolio chart saved as 'portfolio_performance.png'")
except Exception as e:
    logging.error(f"Failed to generate chart from trade log: {e}")