"""
WTI Trading Strategy Analysis
==============================

This script evaluates forecasting models from a TRADING perspective.

Core Philosophy:
- MSE is NOT the goal - we care about making money
- A model with higher MSE but better direction prediction is MORE useful
- Transaction costs and slippage are critical - they can kill a strategy

Three Key Questions Addressed:
1. Trading Signal Construction - How to convert predictions to trades
2. Model Selection - Which model to use for actual trading  
3. Practical Issues - Noise, overfitting, transaction costs

Author: WTI Trading Strategy Framework
"""

import numpy as np
import pandas as pd
import os
import re
from glob import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Trading strategy configuration with realistic cost assumptions"""
    
    # Transaction Costs (be conservative - real costs are often higher)
    commission_rate: float = 0.001      # 0.1% broker commission per trade
    slippage_rate: float = 0.0005       # 0.05% market impact / slippage
    
    # Signal Generation Thresholds
    # Only trade when predicted return exceeds threshold (reduces noise trades)
    long_threshold: float = 0.001       # Go long if pred_return > 0.1%
    short_threshold: float = -0.001     # Go short if pred_return < -0.1%
    
    # Risk Management
    max_position: float = 1.0           # Maximum position size (100% = fully invested)
    
    # Evaluation Parameters
    risk_free_rate: float = 0.02        # Annual risk-free rate for Sharpe calculation
    trading_days_per_year: int = 252    # Standard trading days


# =============================================================================
# SIGNAL GENERATION MODULE
# =============================================================================

class SignalGenerator:
    """
    Convert model predictions to trading signals.
    
    This is critical - the same predictions can generate very different
    signals depending on the method used.
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def threshold_signal(self, predictions: np.ndarray) -> np.ndarray:
        """
        Threshold-based signal: trade only on strong predictions.
        
        Logic:
        - If pred > threshold: LONG (+1)
        - If pred < -threshold: SHORT (-1)  
        - Otherwise: FLAT (0)
        
        This reduces trading frequency and filters out noise.
        """
        signals = np.zeros_like(predictions)
        signals[predictions > self.config.long_threshold] = 1
        signals[predictions < self.config.short_threshold] = -1
        return signals
    
    def direction_signal(self, predictions: np.ndarray) -> np.ndarray:
        """
        Simple direction signal: always be in the market.
        
        Logic:
        - If pred > 0: LONG (+1)
        - If pred <= 0: SHORT (-1)
        
        More aggressive - trades every day.
        """
        signals = np.sign(predictions)
        signals[signals == 0] = 1  # Treat zero as slightly positive
        return signals
    
    def percentile_signal(self, predictions: np.ndarray,
                          long_pct: float = 70,
                          short_pct: float = 30) -> np.ndarray:
        """
        Percentile-based signal: trade relative to prediction distribution.
        
        More robust when prediction scale varies.
        """
        signals = np.zeros_like(predictions)
        long_thresh = np.percentile(predictions, long_pct)
        short_thresh = np.percentile(predictions, short_pct)
        signals[predictions > long_thresh] = 1
        signals[predictions < short_thresh] = -1
        return signals


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """
    Realistic backtesting with transaction costs.
    
    IMPORTANT: This is still simplified. Real-world factors not included:
    - Order book dynamics and partial fills
    - Market regime changes
    - Funding costs for leveraged positions
    - Margin requirements
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def run_backtest(self, 
                     signals: np.ndarray, 
                     actual_returns: np.ndarray) -> Dict:
        """
        Execute backtest with transaction costs.
        
        Parameters:
        -----------
        signals : Trading signals [-1, 0, 1]
        actual_returns : Realized returns
        
        Returns:
        --------
        Dictionary with performance metrics
        """
        n = len(signals)
        
        # Initialize tracking arrays
        positions = np.zeros(n)
        strategy_returns = np.zeros(n)
        transaction_costs = np.zeros(n)
        
        prev_position = 0
        
        for t in range(n):
            current_position = signals[t] * self.config.max_position
            
            # Calculate transaction cost on position change
            position_change = abs(current_position - prev_position)
            if position_change > 0:
                cost = position_change * (self.config.commission_rate + 
                                          self.config.slippage_rate)
                transaction_costs[t] = cost
            
            # Strategy return = position * actual_return - costs
            positions[t] = current_position
            strategy_returns[t] = current_position * actual_returns[t] - transaction_costs[t]
            
            prev_position = current_position
        
        return self._calculate_metrics(strategy_returns, actual_returns, 
                                       positions, transaction_costs)
    
    def _calculate_metrics(self, 
                           strategy_returns: np.ndarray,
                           actual_returns: np.ndarray,
                           positions: np.ndarray,
                           transaction_costs: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Filter out initial NaN/zero period if any
        valid_mask = ~np.isnan(strategy_returns)
        strategy_returns = strategy_returns[valid_mask]
        actual_returns = actual_returns[valid_mask]
        positions = positions[valid_mask]
        transaction_costs = transaction_costs[valid_mask]
        
        if len(strategy_returns) == 0:
            return self._empty_metrics()
        
        # Basic Returns
        total_return = np.prod(1 + strategy_returns) - 1
        n_days = len(strategy_returns)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        
        # Volatility
        daily_vol = np.std(strategy_returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe Ratio (annualized)
        excess_return = annual_return - self.config.risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 1e-8 else 0
        
        # Sortino Ratio (downside deviation only)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino = excess_return / downside_vol
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_dd = np.max(drawdowns)
        
        # Calmar Ratio
        calmar = annual_return / max_dd if max_dd > 1e-8 else 0
        
        # Win Rate
        trades = strategy_returns[strategy_returns != 0]
        if len(trades) > 0:
            win_rate = np.mean(trades > 0)
        else:
            win_rate = 0.5
        
        # Profit Factor
        gains = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        total_gains = np.sum(gains) if len(gains) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
        profit_factor = total_gains / total_losses
        
        # Trade Statistics
        position_changes = np.abs(np.diff(positions))
        num_trades = np.sum(position_changes > 0)
        total_costs = np.sum(transaction_costs)
        
        # Buy & Hold Comparison
        buy_hold = np.prod(1 + actual_returns) - 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'total_costs': total_costs,
            'buy_hold_return': buy_hold,
            'excess_vs_buyhold': total_return - buy_hold,
            'cumulative': cumulative
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no valid data"""
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 1,
            'calmar_ratio': 0, 'win_rate': 0.5, 'profit_factor': 0,
            'num_trades': 0, 'total_costs': 0, 'buy_hold_return': 0,
            'excess_vs_buyhold': 0, 'cumulative': np.array([1])
        }


# =============================================================================
# MODEL EVALUATION FOR TRADING
# =============================================================================

class TradingEvaluator:
    """
    Evaluate models on TRADING metrics, not just MSE.
    
    Key Insight: MSE optimization ≠ Profit maximization
    
    What we actually care about:
    1. Direction Accuracy - Can we predict up/down?
    2. IC (Information Coefficient) - How correlated are predictions with outcomes?
    3. Risk-Adjusted Returns - Can we make money after costs?
    4. Stability - Is the edge consistent over time?
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.signal_gen = SignalGenerator(config)
        self.backtest = BacktestEngine(config)
    
    def evaluate(self, 
                 predictions: np.ndarray, 
                 actuals: np.ndarray,
                 model_id: str = "Unknown") -> Dict:
        """
        Comprehensive trading-focused evaluation.
        """
        # Flatten arrays
        pred = predictions.flatten()
        actual = actuals.flatten()
        
        # Ensure same length
        min_len = min(len(pred), len(actual))
        pred = pred[:min_len]
        actual = actual[:min_len]
        
        # Remove NaN
        valid = ~(np.isnan(pred) | np.isnan(actual))
        pred = pred[valid]
        actual = actual[valid]
        
        if len(pred) < 10:
            return {'model_id': model_id, 'error': 'Insufficient data'}
        
        metrics = {'model_id': model_id}
        
        # =================================================================
        # 1. TRADITIONAL METRICS (for reference only)
        # =================================================================
        metrics['mse'] = np.mean((pred - actual) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(pred - actual))
        
        # =================================================================
        # 2. DIRECTION PREDICTION (THIS IS WHAT MATTERS)
        # =================================================================
        pred_dir = np.sign(pred)
        actual_dir = np.sign(actual)
        
        # Exclude zero returns
        nonzero = actual != 0
        if np.sum(nonzero) > 0:
            metrics['direction_accuracy'] = np.mean(pred_dir[nonzero] == actual_dir[nonzero])
        else:
            metrics['direction_accuracy'] = 0.5
        
        # =================================================================
        # 3. INFORMATION COEFFICIENT (IC)
        # =================================================================
        # Pearson IC
        if len(pred) > 1 and np.std(pred) > 1e-8 and np.std(actual) > 1e-8:
            metrics['ic'] = np.corrcoef(pred, actual)[0, 1]
            if np.isnan(metrics['ic']):
                metrics['ic'] = 0
        else:
            metrics['ic'] = 0
        
        # Rank IC (Spearman - more robust to outliers)
        if len(pred) > 1:
            rank_ic, _ = stats.spearmanr(pred, actual)
            metrics['rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0
        else:
            metrics['rank_ic'] = 0
        
        # =================================================================
        # 4. IC STABILITY (detect overfitting)
        # =================================================================
        half = len(pred) // 2
        if half > 10:
            ic_first = np.corrcoef(pred[:half], actual[:half])[0, 1]
            ic_second = np.corrcoef(pred[half:], actual[half:])[0, 1]
            
            ic_first = 0 if np.isnan(ic_first) else ic_first
            ic_second = 0 if np.isnan(ic_second) else ic_second
            
            metrics['ic_first_half'] = ic_first
            metrics['ic_second_half'] = ic_second
            metrics['ic_decay'] = ic_first - ic_second  # Positive = getting worse
        else:
            metrics['ic_first_half'] = metrics['ic']
            metrics['ic_second_half'] = metrics['ic']
            metrics['ic_decay'] = 0
        
        # =================================================================
        # 5. QUINTILE ANALYSIS
        # =================================================================
        # Check if high predictions lead to high returns (monotonicity)
        if len(pred) >= 20:
            try:
                quintiles = pd.qcut(pred, 5, labels=False, duplicates='drop')
                q_returns = pd.DataFrame({'q': quintiles, 'r': actual}).groupby('q')['r'].mean()
                
                if len(q_returns) >= 2:
                    metrics['quintile_spread'] = float(q_returns.iloc[-1] - q_returns.iloc[0])
                    metrics['quintile_monotonic'] = q_returns.is_monotonic_increasing
                else:
                    metrics['quintile_spread'] = 0
                    metrics['quintile_monotonic'] = False
            except:
                metrics['quintile_spread'] = 0
                metrics['quintile_monotonic'] = False
        else:
            metrics['quintile_spread'] = 0
            metrics['quintile_monotonic'] = False
        
        # =================================================================
        # 6. BACKTEST WITH COSTS
        # =================================================================
        # Threshold-based signals (conservative)
        signals_threshold = self.signal_gen.threshold_signal(pred)
        bt_threshold = self.backtest.run_backtest(signals_threshold, actual)
        
        metrics['sharpe_ratio'] = bt_threshold['sharpe_ratio']
        metrics['sortino_ratio'] = bt_threshold['sortino_ratio']
        metrics['max_drawdown'] = bt_threshold['max_drawdown']
        metrics['total_return'] = bt_threshold['total_return']
        metrics['annual_return'] = bt_threshold['annual_return']
        metrics['win_rate'] = bt_threshold['win_rate']
        metrics['profit_factor'] = bt_threshold['profit_factor']
        metrics['num_trades'] = bt_threshold['num_trades']
        metrics['total_costs'] = bt_threshold['total_costs']
        metrics['buy_hold_return'] = bt_threshold['buy_hold_return']
        metrics['excess_return'] = bt_threshold['excess_vs_buyhold']
        metrics['cumulative'] = bt_threshold['cumulative']
        
        # Also test direction-only signals (aggressive)
        signals_direction = self.signal_gen.direction_signal(pred)
        bt_direction = self.backtest.run_backtest(signals_direction, actual)
        metrics['sharpe_direction_signal'] = bt_direction['sharpe_ratio']
        
        return metrics


# =============================================================================
# RESULT LOADING
# =============================================================================

def load_results(results_dir: str = './results/') -> Dict:
    """
    Load all prediction results from the results directory.
    
    Expected structure:
    results/
    ├── WTI_DLinear_pl5.../
    │   ├── pred.npy
    │   └── true.npy
    └── ...
    """
    results = {}
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results
    
    for dirname in os.listdir(results_dir):
        dirpath = os.path.join(results_dir, dirname)
        
        if not os.path.isdir(dirpath):
            continue
        
        pred_path = os.path.join(dirpath, 'pred.npy')
        true_path = os.path.join(dirpath, 'true.npy')
        
        if os.path.exists(pred_path) and os.path.exists(true_path):
            results[dirname] = {
                'predictions': np.load(pred_path),
                'actuals': np.load(true_path)
            }
    
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(all_metrics: List[Dict], 
                    output_path: str = './WTI_trading_report.txt') -> str:
    """Generate comprehensive trading analysis report"""
    
    report = """
================================================================================
                    WTI TRADING STRATEGY ANALYSIS REPORT
================================================================================

EVALUATION PHILOSOPHY
---------------------
This report evaluates models from a TRADING perspective, NOT just prediction accuracy.

Key Principle: Low MSE ≠ Good Trading Strategy

What we actually evaluate:
1. Direction Accuracy - Can the model predict if price goes UP or DOWN?
2. Information Coefficient - Is there a correlation between predictions and outcomes?
3. Risk-Adjusted Returns - Can we make money AFTER transaction costs?
4. Stability - Is the predictive power consistent over time?

================================================================================
"""
    
    # Convert to DataFrame and sort by Sharpe Ratio
    df = pd.DataFrame(all_metrics)
    
    if 'sharpe_ratio' not in df.columns:
        report += "\nERROR: No valid results found.\n"
        with open(output_path, 'w') as f:
            f.write(report)
        return report
    
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    # =================================================================
    # Helper function: Simplify model name for display only
    # =================================================================
    def simplify_model_name(full_name: str) -> str:
        """Remove common prefixes to make model names more readable in reports/plots."""
        # Remove common prefixes
        prefixes_to_remove = [
            'long_term_forecast_WTI-log_',
            'long_term_forecast_WTI_',
            'long_term_forecast_',
        ]
        name = full_name
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name
    
    # Create display name column (for reports/plots only)
    df['display_name'] = df['model_id'].apply(simplify_model_name)
    
    # =================================================================
    # EXPORT CSV: Full model rankings with complete filenames
    # =================================================================
    csv_path = output_path.replace('.txt', '_rankings.csv')
    
    # Select columns for CSV export (use model_id, NOT display_name)
    csv_columns = [
        'model_id',           # Full model directory name
        'sharpe_ratio',
        'direction_accuracy',
        'ic',
        'rank_ic',
        'ic_decay',
        'max_drawdown',
        'total_return',
        'annual_return',
        'win_rate',
        'profit_factor',
        'num_trades',
        'total_costs',
        'buy_hold_return',
        'excess_return',
        'mse',
        'rmse',
        'mae',
        'sortino_ratio',
        'ic_first_half',
        'ic_second_half'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in csv_columns if col in df.columns]
    
    # Add rank column
    df_export = df[available_columns].copy()
    df_export.insert(0, 'rank', range(1, len(df_export) + 1))
    
    # Save CSV
    df_export.to_csv(csv_path, index=False)
    print(f"  Model rankings saved to: {csv_path}")
    
    # =================================================================
    # SECTION 1: MODEL RANKINGS (using simplified display names)
    # =================================================================
    report += "\n" + "="*80 + "\n"
    report += "SECTION 1: MODEL RANKINGS (by Sharpe Ratio)\n"
    report += "="*80 + "\n\n"
    
    report += f"{'Rank':<5} {'Model':<50} {'Sharpe':>8} {'Dir.Acc':>8} {'IC':>8} {'MaxDD':>8}\n"
    report += "-"*85 + "\n"
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        # Use simplified display name for report
        display_name = row['display_name'][:48] if len(row['display_name']) > 48 else row['display_name']
        report += f"{i:<5} {display_name:<50} {row['sharpe_ratio']:>8.3f} "
        report += f"{row['direction_accuracy']:>7.1%} {row['ic']:>8.4f} "
        report += f"{row['max_drawdown']:>7.1%}\n"
    
    # =================================================================
    # SECTION 2: DETAILED ANALYSIS
    # =================================================================
    report += "\n" + "="*80 + "\n"
    report += "SECTION 2: DETAILED MODEL ANALYSIS\n"
    report += "="*80 + "\n"
    
    for _, row in df.head(10).iterrows():  # Top 10 models
        
        # Determine tradability
        is_tradable = (row['sharpe_ratio'] > 0.5 and 
                       row['direction_accuracy'] > 0.52 and
                       row['ic'] > 0.02)
        
        status = "✓ POTENTIALLY TRADABLE" if is_tradable else "✗ NOT RECOMMENDED"
        
        # Use simplified display name for report header
        report += f"""
--------------------------------------------------------------------------------
Model: {row['display_name']}
Status: {status}
--------------------------------------------------------------------------------

  PREDICTION QUALITY
  ------------------
  Direction Accuracy:  {row['direction_accuracy']:.2%}  {'[Good]' if row['direction_accuracy'] > 0.53 else '[Weak]' if row['direction_accuracy'] > 0.51 else '[Poor]'}
  Information Coef:    {row['ic']:.4f}  {'[Strong]' if row['ic'] > 0.05 else '[Moderate]' if row['ic'] > 0.02 else '[Weak]'}
  Rank IC:             {row['rank_ic']:.4f}
  IC Decay:            {row['ic_decay']:.4f}  {'[Stable]' if abs(row['ic_decay']) < 0.03 else '[Unstable - possible overfit]'}

  TRADING PERFORMANCE (after costs)
  ---------------------------------
  Sharpe Ratio:        {row['sharpe_ratio']:.3f}  {'[Good]' if row['sharpe_ratio'] > 1.0 else '[Moderate]' if row['sharpe_ratio'] > 0.5 else '[Poor]'}
  Sortino Ratio:       {row['sortino_ratio']:.3f}
  Max Drawdown:        {row['max_drawdown']:.1%}  {'[Acceptable]' if row['max_drawdown'] < 0.2 else '[High Risk]'}
  
  Total Return:        {row['total_return']:.2%}
  Annual Return:       {row['annual_return']:.2%}
  Win Rate:            {row['win_rate']:.1%}
  Profit Factor:       {row['profit_factor']:.2f}

  TRADING STATISTICS
  ------------------
  Number of Trades:    {row['num_trades']:.0f}
  Transaction Costs:   {row['total_costs']:.4f}
  Buy & Hold Return:   {row['buy_hold_return']:.2%}
  Excess Return:       {row['excess_return']:.2%}  {'[Beat B&H]' if row['excess_return'] > 0 else '[Underperformed B&H]'}

  REFERENCE METRICS (not for trading decisions)
  ---------------------------------------------
  MSE:                 {row['mse']:.6f}
  RMSE:                {row['rmse']:.6f}
  MAE:                 {row['mae']:.6f}
"""
    
    # =================================================================
    # SECTION 3: INTERPRETATION GUIDE
    # =================================================================
    report += """
================================================================================
SECTION 3: INTERPRETATION GUIDE
================================================================================

DIRECTION ACCURACY
  > 55%:   Strong signal - rare but valuable
  52-55%:  Moderate signal - potentially tradable with good risk management
  50-52%:  Weak signal - barely better than random
  < 50%:   No signal - model is useless or inverted

INFORMATION COEFFICIENT (IC)
  > 0.10:  Excellent - very rare in practice
  0.05-0.10: Good - worth pursuing
  0.02-0.05: Moderate - marginal, needs optimization
  < 0.02:  Weak - likely noise

SHARPE RATIO (after costs)
  > 2.0:   Excellent strategy
  1.0-2.0: Good strategy
  0.5-1.0: Moderate - needs improvement
  0-0.5:   Marginal - high risk of loss
  < 0:     Losing strategy

IC DECAY (first half - second half)
  |decay| < 0.02: Stable model
  |decay| 0.02-0.05: Some instability
  |decay| > 0.05: Likely overfitting - use with caution

================================================================================
SECTION 4: RECOMMENDATIONS
================================================================================
"""
    
    # Find best model
    best = df.iloc[0]
    
    if best['sharpe_ratio'] > 0.5 and best['direction_accuracy'] > 0.52:
        report += f"""
RECOMMENDED MODEL: {best['display_name']}

This model shows promising characteristics:
- Direction Accuracy: {best['direction_accuracy']:.1%}
- Sharpe Ratio: {best['sharpe_ratio']:.2f}
- IC: {best['ic']:.4f}

NEXT STEPS:
1. Validate on completely out-of-sample data (e.g., recent months)
2. Test robustness across different market conditions
3. Implement proper position sizing based on prediction confidence
4. Set stop-loss and maximum drawdown limits
5. Paper trade before live trading

WARNINGS:
- Past performance does not guarantee future results
- The IC decay of {best['ic_decay']:.4f} suggests {'stability' if abs(best['ic_decay']) < 0.03 else 'potential instability'}
- Transaction costs assumed: {0.15}% per round trip
"""
    else:
        report += """
NO MODEL IS RECOMMENDED FOR LIVE TRADING

All tested models show insufficient predictive power for profitable trading.

POSSIBLE REASONS:
1. WTI returns may be largely unpredictable (Efficient Market Hypothesis)
2. The features used don't capture relevant information
3. The model architectures may not suit this data
4. The prediction horizon may be too long

SUGGESTED IMPROVEMENTS:
1. Add more informative features:
   - Sentiment data (news, social media)
   - Macro indicators (interest rates, USD index)
   - Inventory data (EIA reports)
   - Futures curve structure (contango/backwardation)
   
2. Try different targets:
   - Volatility prediction (often more predictable than returns)
   - Direction classification (up/down) instead of return magnitude
   - Regime prediction (trending/ranging)

3. Reduce prediction horizon:
   - Try pred_len=1 (next day) if current horizon is longer
   - Shorter horizons are generally more predictable

4. Feature engineering:
   - Add technical indicators (RSI, MACD, Bollinger Bands)
   - Include lagged returns at different intervals
   - Consider cross-asset features (S&P 500, USD, Gold)
"""
    
    report += "\n" + "="*80 + "\n"
    report += f"Report generated: {pd.Timestamp.now()}\n"
    report += "="*80 + "\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    return report


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(all_metrics: List[Dict], output_dir: str = './analysis_plots/'):
    """Generate visualization of trading results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(all_metrics)
    
    if 'sharpe_ratio' not in df.columns:
        print("No valid results to plot")
        return
    
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    # Helper function to simplify model names for display
    def simplify_model_name(full_name: str) -> str:
        """Remove common prefixes to make model names more readable."""
        prefixes_to_remove = [
            'long_term_forecast_WTI-log_',
            'long_term_forecast_WTI_',
            'long_term_forecast_',
        ]
        name = full_name
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name
    
    # Create display names for plots
    df['display_name'] = df['model_id'].apply(simplify_model_name)
    
    # --- Figure 1: Model Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Direction Accuracy
    ax = axes[0, 0]
    colors = ['green' if x > 0.52 else 'orange' if x > 0.50 else 'red' 
              for x in df['direction_accuracy'].head(10)]
    ax.barh(range(10), df['direction_accuracy'].head(10), color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels([m[:40] for m in df['display_name'].head(10)], fontsize=8)
    ax.axvline(0.5, color='red', linestyle='--', label='Random (50%)')
    ax.axvline(0.52, color='orange', linestyle='--', label='Threshold (52%)')
    ax.set_xlabel('Direction Accuracy')
    ax.set_title('Direction Accuracy by Model')
    ax.legend()
    
    # Sharpe Ratio
    ax = axes[0, 1]
    colors = ['green' if x > 1.0 else 'orange' if x > 0.5 else 'red' 
              for x in df['sharpe_ratio'].head(10)]
    ax.barh(range(10), df['sharpe_ratio'].head(10), color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels([m[:40] for m in df['display_name'].head(10)], fontsize=8)
    ax.axvline(0, color='red', linestyle='--')
    ax.axvline(0.5, color='orange', linestyle='--')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio by Model (after costs)')
    
    # IC vs Direction Accuracy
    ax = axes[1, 0]
    ax.scatter(df['ic'], df['direction_accuracy'], 
               c=df['sharpe_ratio'], cmap='RdYlGn', s=100)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Information Coefficient (IC)')
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('IC vs Direction Accuracy (color = Sharpe)')
    plt.colorbar(ax.collections[0], ax=ax, label='Sharpe Ratio')
    
    # IC Decay (Overfitting Check)
    ax = axes[1, 1]
    ax.scatter(df['ic_first_half'], df['ic_second_half'], 
               c=df['sharpe_ratio'], cmap='RdYlGn', s=100)
    ax.plot([df['ic'].min(), df['ic'].max()], 
            [df['ic'].min(), df['ic'].max()], 'k--', alpha=0.5)
    ax.set_xlabel('IC (First Half)')
    ax.set_ylabel('IC (Second Half)')
    ax.set_title('IC Stability Check (points below line = overfitting)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # --- Figure 2: Cumulative Returns (Top 5 Models) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for _, row in df.head(5).iterrows():
        if 'cumulative' in row and len(row['cumulative']) > 1:
            ax.plot(row['cumulative'], label=row['display_name'][:40])
    
    ax.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative Return (1 = initial)')
    ax.set_title('Cumulative Returns - Top 5 Models')
    ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='WTI Trading Strategy Analysis')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory containing model prediction results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis outputs (default: same as results_dir)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate per trade (default: 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.0005,
                        help='Slippage rate per trade (default: 0.05%%)')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Signal threshold for trading (default: 0.1%%)')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    print("="*80)
    print("WTI TRADING STRATEGY ANALYSIS")
    print("="*80)
    print(f"\nResults Directory: {args.results_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Configuration
    config = TradingConfig(
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        long_threshold=args.threshold,
        short_threshold=-args.threshold
    )
    
    print(f"\nTransaction Costs: {(config.commission_rate + config.slippage_rate)*100:.2f}% per trade")
    print(f"Signal Threshold: ±{config.long_threshold*100:.2f}%")
    
    # Load results
    print("\n" + "-"*40)
    print("Loading model predictions...")
    results = load_results(args.results_dir)
    
    if len(results) == 0:
        print(f"""
ERROR: No prediction results found in {args.results_dir}

Please run the experiment script first:
    bash WTI_trading_experiments.sh

Expected file structure:
    {args.results_dir}/
    ├── WTI_DLinear_pl5.../
    │   ├── pred.npy
    │   └── true.npy
    └── ...
        """)
        exit(1)
    
    print(f"Found {len(results)} model results")
    
    # Evaluate each model
    print("\n" + "-"*40)
    print("Evaluating models for trading...")
    
    evaluator = TradingEvaluator(config)
    all_metrics = []
    
    for model_id, data in results.items():
        print(f"  Evaluating: {model_id[:50]}...")
        metrics = evaluator.evaluate(
            data['predictions'],
            data['actuals'],
            model_id
        )
        if 'error' not in metrics:
            all_metrics.append(metrics)
    
    if len(all_metrics) == 0:
        print("ERROR: No valid model results to analyze")
        exit(1)
    
    # Generate report
    print("\n" + "-"*40)
    print("Generating trading analysis report...")
    report_path = os.path.join(output_dir, 'WTI_trading_report.txt')
    generate_report(all_metrics, report_path)
    
    # Generate plots
    print("\n" + "-"*40)
    print("Generating visualization...")
    plots_dir = os.path.join(output_dir, 'analysis_plots')
    plot_results(all_metrics, plots_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {report_path}")
    print(f"  - {report_path.replace('.txt', '_rankings.csv')}  <-- Full model rankings CSV")
    print(f"  - {plots_dir}/model_comparison.png")
    print(f"  - {plots_dir}/cumulative_returns.png")
    print("="*80)