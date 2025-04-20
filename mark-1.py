import warnings
# Fuck me
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
import sys

# Suppress stdout during Rbeast execution
class NullWriter:
    def write(self, text): pass
    def flush(self): pass

TICKER = "BTC-USD"
CONFIDENCE_THRESHOLD = 0.75
STOP_LOSS = -0.05
TAKE_PROFIT = 0.10
MA_WINDOW = 50
RSI_PERIOD = 14
DATA_LOOKBACK_YEARS = 3
FETCH_INTERVAL_SECONDS = 60  
FEE_RATE = 0.001       
SLIPPAGE_RATE = 0.002  
RISK_PER_TRADE = 0.02  

capital = 100
btc_holding = 0
entry_price = 0
history = []
trade_count = 0

print("🚀 Starting BTC Trading Simulator")
print("Type 'exit' anytime to quit.\n")

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

while True:
    try:
        # fetch interval
        loop_start_time = t.time()
        start_date = (datetime.now() - timedelta(days=365*DATA_LOOKBACK_YEARS)).strftime('%Y-%m-%d')
    
        exchange = ccxt.binance({ 'enableRateLimit': True })
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=DATA_LOOKBACK_YEARS*365)
        timestamps = [row[0] for row in ohlcv]
        price_vals = np.array([row[4] for row in ohlcv], dtype=float)
        price_series = pd.Series(price_vals, index=pd.to_datetime(timestamps, unit='ms'))
        price = price_series
        ma = price.rolling(window=MA_WINDOW).mean()
        rsi = calculate_rsi(price.values, RSI_PERIOD)

        # truth checks
        ma_vals = ma.values
        rsi_vals = rsi
        
        # output suppressed
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
        
        print(f"Analyzing BTC price data... Please wait...")
        
        # Suppress all output n
        old_stdout = sys.stdout
        sys.stdout = NullWriter()
        result = rb.beast(price.values, start=0, deltat=1, season='none', 
                         prior=prior, mcmc=mcmc, extra=extra)
        sys.stdout = old_stdout
        
        # convert the BEAST output attributes
        changepoints = result.trend.cp
        if isinstance(changepoints, np.ndarray):
            changepoints = list(changepoints)
            
        trend = result.trend.Y
        
        # Use actual changepoint probabilities if provided, else default high confidence
        try:
            cp_probs = result.trend.prob.tolist()
        except Exception:
            cp_probs = [0.9] * len(changepoints)
        
        last_action = "HOLD"
        total_value = capital + btc_holding * float(price.iloc[-1])
        for i in range(len(price_vals)):
            current_price = float(price_vals[i])
            
            # Check if this index is in the changepoints list
            if i in changepoints:
                cp_index = changepoints.index(i)
                prob = cp_probs[cp_index]
                
                if prob > CONFIDENCE_THRESHOLD:
                    # Calculate slope for trend direction
                    slope_change = 0
                    if i > 0 and i < len(trend):
                        slope_change = float(trend[i] - trend[i-1])
                    
                    # use numpy array for moving average
                    ma_condition = False
                    ma_value = ma_vals[i]
                    if not np.isnan(ma_value):
                        ma_condition = current_price > ma_value if slope_change > 0 else current_price < ma_value
                    
                    # use numpy array for RSI
                    rsi_condition = False
                    rsi_value = rsi_vals[i]
                    if not np.isnan(rsi_value):
                        rsi_condition = rsi_value < 70 if slope_change > 0 else rsi_value > 75
                    
                    # Buy logic
                    if slope_change > 0 and capital > 0 and ma_condition and rsi_condition:
                        risk_amount = RISK_PER_TRADE * total_value
                        units = risk_amount / current_price
                        exec_price = current_price * (1 + SLIPPAGE_RATE)
                        fee = units * exec_price * FEE_RATE
                        capital -= units * exec_price + fee
                        btc_holding += units
                        entry_price = exec_price
                        last_action = f"BUY {units:.5f} @ {exec_price:.2f}"
                        trade_count += 1
                    
                    # Sell logic (Exit for postions with slipage and fees)
                    elif slope_change < 0 and btc_holding > 0 and (ma_condition or rsi_condition):
                        units = btc_holding
                        exec_price = current_price * (1 - SLIPPAGE_RATE)
                        fee = units * exec_price * FEE_RATE
                        capital += units * exec_price - fee
                        btc_holding = 0
                        last_action = f"SELL {units:.5f} @ {exec_price:.2f}"
                        trade_count += 1

            # Stop loss and take profit logic
            if btc_holding > 0:
                # Calculate dynamic risk parameters safely
                recent_price_slice = price.iloc[max(0, i-20):i+1]
                
                if len(recent_price_slice) > 1:
                    recent_volatility = float(recent_price_slice.pct_change().std() * 16)
                    dynamic_stop = max(STOP_LOSS, -recent_volatility)
                    dynamic_tp = max(TAKE_PROFIT, recent_volatility*1.5)
                else:
                    dynamic_stop = STOP_LOSS
                    dynamic_tp = TAKE_PROFIT
                
                change = float((current_price - entry_price) / entry_price)
                if change <= dynamic_stop or change >= dynamic_tp:
                    # simulate slippage and fees on exit
                    units = btc_holding
                    exec_price = current_price * (1 - SLIPPAGE_RATE)
                    fee = units * exec_price * FEE_RATE
                    capital += units * exec_price - fee
                    btc_holding = 0
                    action_type = 'STOP LOSS' if change <= dynamic_stop else 'TAKE PROFIT'
                    last_action = f"{action_type} @ {exec_price:.2f} ({change:.1%})"
                    trade_count += 1

        # compute current BTC price and total virtual value
        current_btc_price = float(price.iloc[-1])
        total_value = capital + btc_holding * current_btc_price
        history.append((price.index[-1], total_value, last_action))
        
        print("\033[H\033[J")
        
        print(f"💰 PORTFOLIO SUMMARY ({price.index[-1].date()})")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 BTC Price:      ${current_btc_price:,.2f}")
        print(f"💵 Cash:           ${capital:,.2f}")
        print(f"🪙 BTC Holdings:   {btc_holding:.5f} BTC (${btc_holding * current_btc_price:,.2f})")
        print(f"💼 Total Value:    ${total_value:,.2f}")
        print(f"🔄 Total Trades:   {trade_count}")
        print(f"🔄 Last Action:    {last_action}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # automatically wait to respect fetch interval
        elapsed = t.time() - loop_start_time
        sleep_time = max(0, FETCH_INTERVAL_SECONDS - elapsed)
        print(f"Waiting {sleep_time:.0f} seconds until next fetch...")
        t.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped.")
        break
    except Exception as e:
        print(f"Error: {str(e)}. Retrying in 30 seconds...")
        t.sleep(30)

if len(history) > 0:
    df = pd.DataFrame(history, columns=['Date', 'PortfolioValue', 'Action'])
    df.set_index('Date', inplace=True)

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['PortfolioValue'], label='Portfolio Value ($)', color='green')
    plt.title("Portfolio Performance")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(price[-len(df):], label='BTC Price ($)', color='orange')
    plt.xlabel("Date")
    plt.ylabel("BTC Price ($)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    
    initial_value = capital
    final_value = df['PortfolioValue'].iloc[-1]
    roi = (final_value - initial_value) / initial_value * 100
    
    print(f"\n💰 Final Portfolio: ${final_value:,.2f}")
    print(f"📊 ROI: {roi:.2f}%")
    print(f"🔄 Total Trades: {trade_count}")
    print("\nPortfolio chart saved as 'portfolio_performance.png'")
