#!/usr/bin/env python3
"""
;weknf;wef - Momentum, cross_sectional_momentum Trading Strategy

Strategy Type: momentum, cross_sectional_momentum
Description: ewf
Created: 2025-08-08T11:17:02.858Z

WARNING: This is a template implementation. Thoroughly backtest before live trading.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class weknfwefStrategy:
    """
    ;weknf;wef Implementation
    
    Strategy Type: momentum, cross_sectional_momentum
    Risk Level: Monitor drawdowns and position sizes carefully
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.positions = {}
        self.performance_metrics = {}
        logger.info(f"Initialized ;weknf;wef strategy")
        
    def get_default_config(self):
        """Default configuration parameters"""
        return {
            'max_position_size': 0.05,  # 5% max position size
            'stop_loss_pct': 0.05,      # 5% stop loss
            'lookback_period': 20,       # 20-day lookback
            'rebalance_freq': 'daily',   # Rebalancing frequency
            'transaction_costs': 0.001,  # 0.1% transaction costs
        }
    
    def load_data(self, symbols, start_date, end_date):
        """Load market data for analysis"""
        try:
            import yfinance as yf
            data = yf.download(symbols, start=start_date, end=end_date)
            logger.info(f"Loaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

# =============================================================================
# USER'S STRATEGY IMPLEMENTATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class MomentumStrategy:
    def __init__(self, data, short_window=20, long_window=50, risk_free_rate=0.01):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.risk_free_rate = risk_free_rate
        self.signals = None
        self.positions = None

    def generate_signals(self):
        self.data['short_mavg'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['long_mavg'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()
        self.data['signals'] = 0
        self.data['signals'][self.short_window:] = np.where(self.data['short_mavg'][self.short_window:] > self.data['long_mavg'][self.short_window:], 1, 0)
        self.signals = self.data['signals']
        logging.info("Signals generated.")

    def backtest(self):
        self.data['positions'] = self.signals.diff()
        self.data['strategy_returns'] = self.data['close'].pct_change() * self.signals.shift(1)
        self.data['cumulative_strategy_returns'] = (1 + self.data['strategy_returns']).cumprod()
        self.data['cumulative_market_returns'] = (1 + self.data['close'].pct_change()).cumprod()
        logging.info("Backtesting completed.")

    def calculate_performance_metrics(self):
        total_return = self.data['cumulative_strategy_returns'].iloc[-1] - 1
        daily_returns = self.data['strategy_returns'].dropna()
        sharpe_ratio = (daily_returns.mean() - self.risk_free_rate / 252) / daily_returns.std() * np.sqrt(252)
        max_drawdown = (self.data['cumulative_strategy_returns'] / self.data['cumulative_strategy_returns'].cummax() - 1).min()
        logging.info("Performance metrics calculated.")
        return total_return, sharpe_ratio, max_drawdown

def generate_sample_data(num_days=252):
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='B')
    prices = np.random.normal(loc=100, scale=1, size=num_days).cumsum()
    data = pd.DataFrame(data={'close': prices}, index=dates)
    return data

if __name__ == "__main__":
    sample_data = generate_sample_data()
    strategy = MomentumStrategy(data=sample_data)
    strategy.generate_signals()
    strategy.backtest()
    total_return, sharpe_ratio, max_drawdown = strategy.calculate_performance_metrics()
    print("Total Return: %.2f" % total_return)
    print("Sharpe Ratio: %.2f" % sharpe_ratio)
    print("Max Drawdown: %.2f" % max_drawdown)
    plt.figure(figsize=(12, 6))
    plt.plot(strategy.data['cumulative_strategy_returns'], label='Strategy Returns')
    plt.plot(strategy.data['cumulative_market_returns'], label='Market Returns')
    plt.title('Strategy vs Market Returns')
    plt.legend()
    plt.show()

# =============================================================================
# STRATEGY EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    strategy = weknfwefStrategy()
    print(f"Strategy '{strategyName}' initialized successfully!")
    
    # Example data loading
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"Loading data for symbols: {symbols}")
    data = strategy.load_data(symbols, start_date, end_date)
    
    if data is not None:
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("Strategy ready for backtesting!")
    else:
        print("Failed to load data. Check your internet connection.")
