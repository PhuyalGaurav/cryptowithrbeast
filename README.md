# Crypto Trading Simulator with RBEAST

Just learning data science cause exams delayed ig.

---

## 🚀 Overview
This script fetches daily BTC/USDT prices from Binance, analyzes trends with RBEAST, applies common trading indicators (moving average, RSI), and executes simulated buy/sell orders with realistic slippage and fees. All capital remains virtual—no live orders or real money.

## 🔍 Features
- Real‑time OHLCV data via CCXT from Binance
- Trend detection using BEAST (RBEAST Python binding)
- Technical indicators: 50‑day moving average, 14‑period RSI
- Dynamic stop‑loss & take‑profit based on recent volatility
- Position sizing: risk a fixed % (default 2%) of total portfolio
- Simulated slippage (0.2%) and trading fees (0.1%) per trade
- Automated loop with configurable fetch interval (default 60 s)
- Virtual order book—keeps capital in-memory and logs each trade
- History logging and performance chart saved to `portfolio_performance.png`



## ⚙️ Configuration
Open `mark-1.py` and adjust any of these constants near the top:

- `DATA_LOOKBACK_YEARS` – how many years of daily history to load
- `MA_WINDOW` – moving average window size (days)
- `RSI_PERIOD` – RSI lookback period
- `RISK_PER_TRADE` – fraction of portfolio to risk per trade
- `FEE_RATE`, `SLIPPAGE_RATE` – simulated cost settings
- `FETCH_INTERVAL_SECONDS` – wait time between loops

## ▶️ Usage
Run the simulator from your terminal:

```bash
python mark-1.py
```

The program will print a portfolio summary each cycle, including cash balance, BTC holdings, total value, and last action. It automatically sleeps to respect your fetch interval. Press `Ctrl+C` to stop.

When the loop ends, a performance chart is saved as `portfolio_performance.png`, and you’ll see final ROI and trade stats.

## 📚 How It Works
1. **Data Fetch** – grab historical daily candles from Binance.  
2. **Indicator Calc** – compute MA and RSI on closing prices.  
3. **Trend Modeling** – fit BEAST model to detect trend changepoints + probabilities.  
4. **Signal Logic** – on high‑confidence changepoints, check price vs. MA & RSI for entry/exit.  
5. **Order Simulation** – simulate a market order with slippage and fees, update virtual portfolio.  
6. **Risk Controls** – enforce stop‑loss/take‑profit based on recent volatility.  

## ⚠️ Limitations
- **Simulation Only**: no real orders are placed.  
- **Data Granularity**: uses daily candles; intraday strategies aren’t supported yet.  
- **No Liquidity Impact**: large order sizes on real exchanges might move the market.  
- **Backtesting**: forward‑testing only; no historical backtest engine included.  

## 📈 Next Steps
- Add intraday (minute‑level) data & faster loops  
- Integrate a proper backtesting framework (e.g., Backtrader)  
- Track partial fills, order status, and real exchange API calls  
- Expand to multiple assets or portfolio strategies  

## Credits
- [RBEAST Python Binding](https://github.com/zhaokg/Rbeast) – for trend detection and changepoint analysis