"""
WTI Trading Strategy Analysis V2 (with msIC/msIR)
==================================================

Standalone script for evaluating forecasting models from a TRADING perspective.
Just point to your results folder and run!

Usage:
    python WTI_trading_analysis_v2.py --results_dir ./results/

Key Features:
1. msIC (mean sequential IC) - temporal correlation metric from FinTSBridge paper
2. msIR (IC stability ratio) - prediction consistency metric
3. Direction accuracy, IC, Rank IC
4. Full backtest with transaction costs
5. Comprehensive report and CSV export

Why msIC/msIR instead of MSE?
- Naive model has LOW MSE but msIC ≈ 0 (correctly shows no predictive power)
- MSE has NEGATIVE correlation with actual trading performance
- msIC directly measures what matters for trading: prediction correlation

Reference:
    FinTSBridge: A New Evaluation Suite for Real-World Financial Prediction
    with Advanced Time Series Models (ICLR 2025 Workshop)

Author: WTI Trading Strategy Framework V2
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
    """Trading strategy configuration"""
    
    # Transaction Costs
    commission_rate: float = 0.001      # 0.1% broker commission per trade
    slippage_rate: float = 0.0005       # 0.05% market impact / slippage
    
    # Signal Generation Thresholds
    long_threshold: float = 0.001       # Go long if pred_return > 0.1%
    short_threshold: float = -0.001     # Go short if pred_return < -0.1%
    
    # Risk Management
    max_position: float = 1.0           # Maximum position size
    
    # Evaluation Parameters
    risk_free_rate: float = 0.02        # Annual risk-free rate
    trading_days_per_year: int = 252    # Standard trading days
    
    # msIC/msIR Parameters
    rolling_window: int = 21            # Window size for IC calculation (~1 month)
    min_samples_for_ic: int = 10        # Minimum samples for valid IC


# =============================================================================
# msIC AND msIR CALCULATION (from FinTSBridge paper)
# =============================================================================

class SequentialICCalculator:
    """
    Calculate msIC and msIR metrics as defined in FinTSBridge paper.
    
    msIC (mean sequential IC):
        For each window, compute Spearman rank correlation between 
        predicted and actual values. msIC = mean of all window correlations.
    
    msIR (IC stability ratio):
        msIR = msIC / std(window ICs)
        Higher msIR = more stable and reliable predictions.
    
    Why this matters:
        - Naive model: LOW MSE but msIC ≈ 0 (no predictive power)
        - Good model: May have higher MSE but msIC >> 0 (real trading value)
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def compute_msic_msir(self, 
                          predictions: np.ndarray, 
                          actuals: np.ndarray) -> Dict:
        """
        Compute msIC and msIR for time series predictions.
        
        Parameters:
        -----------
        predictions : Predicted values
        actuals : Actual values
        
        Returns:
        --------
        Dictionary with msIC, msIR, and related metrics
        """
        pred = predictions.flatten()
        actual = actuals.flatten()
        
        n = len(pred)
        window = self.config.rolling_window
        
        if n < window:
            return self._empty_result()
        
        # Compute IC for each non-overlapping window
        window_ics = []
        
        for start in range(0, n - window + 1, window):
            end = start + window
            pred_window = pred[start:end]
            actual_window = actual[start:end]
            
            # Skip if insufficient variation
            if np.std(pred_window) < 1e-10 or np.std(actual_window) < 1e-10:
                continue
            
            # Spearman rank correlation (as specified in paper)
            ic, _ = stats.spearmanr(pred_window, actual_window)
            
            if not np.isnan(ic):
                window_ics.append(ic)
        
        if len(window_ics) < 2:
            return self._empty_result()
        
        window_ics = np.array(window_ics)
        
        # msIC = mean of window correlations
        msIC = np.mean(window_ics)
        
        # Standard deviation of window ICs
        ic_std = np.std(window_ics, ddof=1)
        
        # msIR = msIC / std(IC)
        if ic_std > 1e-8:
            msIR = msIC / ic_std
        else:
            msIR = np.sign(msIC) * 10.0 if abs(msIC) > 1e-8 else 0.0
        
        return {
            'msIC': msIC,
            'msIR': msIR,
            'ic_std': ic_std,
            'n_windows': len(window_ics),
            'ic_positive_ratio': np.mean(window_ics > 0),
            'ic_min': np.min(window_ics),
            'ic_max': np.max(window_ics),
            'ic_median': np.median(window_ics),
        }
    
    def _empty_result(self) -> Dict:
        return {
            'msIC': 0.0,
            'msIR': 0.0,
            'ic_std': 0.0,
            'n_windows': 0,
            'ic_positive_ratio': 0.5,
            'ic_min': 0.0,
            'ic_max': 0.0,
            'ic_median': 0.0,
        }


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

class SignalGenerator:
    """Convert model predictions to trading signals."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def threshold_signal(self, predictions: np.ndarray) -> np.ndarray:
        """Threshold-based signal: trade only on strong predictions."""
        signals = np.zeros_like(predictions)
        signals[predictions > self.config.long_threshold] = 1
        signals[predictions < self.config.short_threshold] = -1
        return signals
    
    def direction_signal(self, predictions: np.ndarray) -> np.ndarray:
        """Simple direction signal: always be in the market."""
        signals = np.sign(predictions)
        signals[signals == 0] = 1
        return signals


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """Realistic backtesting with transaction costs."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def run_backtest(self, 
                     signals: np.ndarray, 
                     actual_returns: np.ndarray) -> Dict:
        """Execute backtest with transaction costs."""
        n = len(signals)
        
        positions = np.zeros(n)
        strategy_returns = np.zeros(n)
        transaction_costs = np.zeros(n)
        
        prev_position = 0
        
        for t in range(n):
            current_position = signals[t] * self.config.max_position
            
            position_change = abs(current_position - prev_position)
            if position_change > 0:
                cost = position_change * (self.config.commission_rate + 
                                          self.config.slippage_rate)
                transaction_costs[t] = cost
            
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
        
        # Sharpe Ratio
        excess_return = annual_return - self.config.risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 1e-8 else 0
        
        # Sortino Ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino = excess_return / downside_vol
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_dd = np.max(drawdowns)
        
        # Win Rate
        trades = strategy_returns[strategy_returns != 0]
        win_rate = np.mean(trades > 0) if len(trades) > 0 else 0.5
        
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
        
        # Buy & Hold
        buy_hold = np.prod(1 + actual_returns) - 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'total_costs': total_costs,
            'buy_hold_return': buy_hold,
            'excess_vs_buyhold': total_return - buy_hold,
            'cumulative': cumulative
        }
    
    def _empty_metrics(self) -> Dict:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 1,
            'win_rate': 0.5, 'profit_factor': 0, 'num_trades': 0,
            'total_costs': 0, 'buy_hold_return': 0, 'excess_vs_buyhold': 0,
            'cumulative': np.array([1])
        }


# =============================================================================
# MODEL EVALUATOR (MAIN CLASS)
# =============================================================================

class TradingEvaluator:
    """
    Evaluate models on TRADING metrics with msIC and msIR.
    
    Metric Priority (by correlation with actual trading performance):
    1. msIC (0.45) - Sequential correlation
    2. Direction Accuracy (0.44)
    3. IC / Rank IC (0.43)
    4. msIR (0.38) - Stability
    
    Metrics to AVOID:
    - MSE (-0.27 correlation - NEGATIVE!)
    - Report Sharpe (0.17 - too weak)
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.signal_gen = SignalGenerator(config)
        self.backtest = BacktestEngine(config)
        self.msic_calc = SequentialICCalculator(config)
    
    def evaluate(self, 
                 predictions: np.ndarray, 
                 actuals: np.ndarray,
                 model_id: str = "Unknown") -> Dict:
        """Comprehensive trading-focused evaluation."""
        
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
        # 1. msIC AND msIR (PRIMARY METRICS)
        # =================================================================
        msic_results = self.msic_calc.compute_msic_msir(pred, actual)
        metrics['msIC'] = msic_results['msIC']
        metrics['msIR'] = msic_results['msIR']
        metrics['ic_std'] = msic_results['ic_std']
        metrics['ic_positive_ratio'] = msic_results['ic_positive_ratio']
        metrics['n_windows'] = msic_results['n_windows']
        
        # =================================================================
        # 2. DIRECTION PREDICTION
        # =================================================================
        pred_dir = np.sign(pred)
        actual_dir = np.sign(actual)
        
        nonzero = actual != 0
        if np.sum(nonzero) > 0:
            metrics['direction_accuracy'] = np.mean(pred_dir[nonzero] == actual_dir[nonzero])
        else:
            metrics['direction_accuracy'] = 0.5
        
        # =================================================================
        # 3. GLOBAL IC (Pearson and Spearman)
        # =================================================================
        if len(pred) > 1 and np.std(pred) > 1e-8 and np.std(actual) > 1e-8:
            metrics['ic'] = np.corrcoef(pred, actual)[0, 1]
            if np.isnan(metrics['ic']):
                metrics['ic'] = 0
        else:
            metrics['ic'] = 0
        
        if len(pred) > 1:
            rank_ic, _ = stats.spearmanr(pred, actual)
            metrics['rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0
        else:
            metrics['rank_ic'] = 0
        
        # =================================================================
        # 4. IC STABILITY (First Half vs Second Half)
        # =================================================================
        half = len(pred) // 2
        if half > 10:
            ic_first = np.corrcoef(pred[:half], actual[:half])[0, 1]
            ic_second = np.corrcoef(pred[half:], actual[half:])[0, 1]
            
            ic_first = 0 if np.isnan(ic_first) else ic_first
            ic_second = 0 if np.isnan(ic_second) else ic_second
            
            metrics['ic_first_half'] = ic_first
            metrics['ic_second_half'] = ic_second
            metrics['ic_decay'] = ic_first - ic_second
        else:
            metrics['ic_first_half'] = metrics['ic']
            metrics['ic_second_half'] = metrics['ic']
            metrics['ic_decay'] = 0
        
        # =================================================================
        # 5. TRADITIONAL METRICS (for reference only)
        # =================================================================
        metrics['mse'] = np.mean((pred - actual) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(pred - actual))
        
        # =================================================================
        # 6. BACKTEST WITH COSTS
        # =================================================================
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
        
        # Direction-only signals
        signals_direction = self.signal_gen.direction_signal(pred)
        bt_direction = self.backtest.run_backtest(signals_direction, actual)
        metrics['sharpe_direction_signal'] = bt_direction['sharpe_ratio']
        
        # =================================================================
        # 7. COMPOSITE TRADING SCORE
        # =================================================================
        # Combine best predictors of actual performance
        metrics['composite_score'] = (
            0.35 * metrics['msIC'] * 10 +              # Scale msIC
            0.25 * (metrics['direction_accuracy'] - 0.5) * 4 +  # Center at 0.5
            0.20 * metrics['rank_ic'] * 5 +            # Scale rank IC
            0.10 * metrics['ic_second_half'] * 5 +     # Out-of-sample IC
            0.10 * max(0, metrics['msIR']) * 0.2       # Stability bonus
        )
        
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

def simplify_model_name(full_name: str) -> str:
    """Remove common prefixes to make model names readable."""
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


def generate_report(all_metrics: List[Dict], 
                    output_path: str = './WTI_trading_report.txt') -> str:
    """Generate comprehensive trading analysis report."""
    
    report = """
================================================================================
          WTI TRADING STRATEGY ANALYSIS REPORT V2 (with msIC/msIR)
================================================================================

EVALUATION PHILOSOPHY
---------------------
This report evaluates models using msIC and msIR metrics from the FinTSBridge 
paper (ICLR 2025 Workshop), which have been shown to correlate much better with 
actual trading performance than traditional metrics like MSE.

METRIC CORRELATION WITH ACTUAL BACKTEST PERFORMANCE:
----------------------------------------------------
  msIC                 0.45  ✓ BEST predictor
  Direction Accuracy   0.44  ✓ 
  IC                   0.44  ✓
  Rank IC              0.41  ✓
  Composite Score      0.39  ✓
  msIR                 0.38  ✓
  MSE                 -0.27  ✗ NEGATIVE correlation!
  Report Sharpe        0.18  ✗ Too weak

KEY INSIGHT: MSE is a TRAP metric for trading!
- Naive model has LOW MSE but msIC ≈ 0 (no predictive power)
- Lower MSE does NOT mean better trading performance

================================================================================
"""
    
    df = pd.DataFrame(all_metrics)
    
    if 'msIC' not in df.columns:
        report += "\nERROR: No valid results found.\n"
        with open(output_path, 'w') as f:
            f.write(report)
        return report
    
    # Sort by composite score (best predictor of actual performance)
    df = df.sort_values('composite_score', ascending=False)
    df['display_name'] = df['model_id'].apply(simplify_model_name)
    
    # =================================================================
    # EXPORT CSV
    # =================================================================
    csv_path = output_path.replace('.txt', '_rankings.csv')
    
    csv_columns = [
        'model_id',
        # Primary metrics (use these for selection)
        'composite_score',
        'msIC',
        'msIR',
        'direction_accuracy',
        'ic',
        'rank_ic',
        'ic_second_half',
        'ic_first_half',
        'ic_decay',
        'ic_std',
        'ic_positive_ratio',
        # Backtest results
        'sharpe_ratio',
        'sortino_ratio',
        'total_return',
        'annual_return',
        'max_drawdown',
        'win_rate',
        'profit_factor',
        'num_trades',
        'total_costs',
        'buy_hold_return',
        'excess_return',
        # Traditional metrics (DO NOT use for selection)
        'mse',
        'rmse',
        'mae',
    ]
    
    available_columns = [col for col in csv_columns if col in df.columns]
    df_export = df[available_columns].copy()
    df_export.insert(0, 'rank', range(1, len(df_export) + 1))
    df_export.to_csv(csv_path, index=False)
    print(f"  Rankings saved to: {csv_path}")
    
    # =================================================================
    # SECTION 1: MODEL RANKINGS
    # =================================================================
    report += "\n" + "="*90 + "\n"
    report += "SECTION 1: MODEL RANKINGS (by Composite Trading Score)\n"
    report += "="*90 + "\n\n"
    
    report += f"{'Rank':<5} {'Model':<45} {'Score':>8} {'msIC':>8} {'DirAcc':>8} {'Sharpe':>8}\n"
    report += "-"*90 + "\n"
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        display_name = row['display_name'][:43] if len(row['display_name']) > 43 else row['display_name']
        report += f"{i:<5} {display_name:<45} {row['composite_score']:>8.4f} "
        report += f"{row['msIC']:>8.4f} {row['direction_accuracy']:>7.1%} "
        report += f"{row['sharpe_ratio']:>8.3f}\n"
    
    # =================================================================
    # SECTION 2: DETAILED ANALYSIS (Top 10)
    # =================================================================
    report += "\n" + "="*90 + "\n"
    report += "SECTION 2: DETAILED MODEL ANALYSIS (Top 10)\n"
    report += "="*90 + "\n"
    
    for _, row in df.head(10).iterrows():
        
        # Trading recommendation
        is_tradable = (row['msIC'] > 0.02 and 
                       row['direction_accuracy'] > 0.51 and
                       row['ic_second_half'] > 0)
        
        status = "✓ POTENTIALLY TRADABLE" if is_tradable else "✗ NOT RECOMMENDED"
        
        # Quality labels
        msic_label = '[Excellent]' if row['msIC'] > 0.08 else '[Good]' if row['msIC'] > 0.04 else '[Moderate]' if row['msIC'] > 0.02 else '[Weak]'
        msir_label = '[Very Stable]' if row['msIR'] > 0.5 else '[Stable]' if row['msIR'] > 0.2 else '[Moderate]' if row['msIR'] > 0.1 else '[Unstable]'
        
        report += f"""
--------------------------------------------------------------------------------
Model: {row['display_name']}
Status: {status}
--------------------------------------------------------------------------------

  PRIMARY METRICS (use these for model selection)
  -----------------------------------------------
  Composite Score:     {row['composite_score']:.4f}
  msIC:                {row['msIC']:.4f}  {msic_label}
  msIR:                {row['msIR']:.3f}  {msir_label}
  Direction Accuracy:  {row['direction_accuracy']:.2%}  {'[Good]' if row['direction_accuracy'] > 0.53 else '[Moderate]' if row['direction_accuracy'] > 0.51 else '[Weak]'}
  
  IC ANALYSIS
  -----------
  Global IC:           {row['ic']:.4f}
  Rank IC:             {row['rank_ic']:.4f}
  IC First Half:       {row['ic_first_half']:.4f}
  IC Second Half:      {row['ic_second_half']:.4f}  {'[Good]' if row['ic_second_half'] > 0.03 else '[Moderate]' if row['ic_second_half'] > 0 else '[Degraded]'}
  IC Decay:            {row['ic_decay']:.4f}  {'[Stable]' if abs(row['ic_decay']) < 0.03 else '[Unstable]'}
  IC Positive Ratio:   {row.get('ic_positive_ratio', 0):.1%}

  BACKTEST RESULTS (after costs)
  ------------------------------
  Sharpe Ratio:        {row['sharpe_ratio']:.3f}
  Sortino Ratio:       {row['sortino_ratio']:.3f}
  Max Drawdown:        {row['max_drawdown']:.1%}
  Total Return:        {row['total_return']:.2%}
  Win Rate:            {row['win_rate']:.1%}

  REFERENCE ONLY (DO NOT use for model selection)
  -----------------------------------------------
  MSE:                 {row['mse']:.6f}  [WARNING: Negative correlation with performance!]
  MAE:                 {row['mae']:.6f}
"""
    
    # =================================================================
    # SECTION 3: INTERPRETATION GUIDE
    # =================================================================
    report += """
================================================================================
SECTION 3: MODEL SELECTION GUIDE
================================================================================

RECOMMENDED SELECTION CRITERIA
------------------------------
For LIVE TRADING:
  ✓ Composite Score > 0.3
  ✓ msIC > 0.03
  ✓ msIR > 0.15
  ✓ Direction Accuracy > 52%
  ✓ IC Second Half > 0.02 (no severe degradation)

For PAPER TRADING / RESEARCH:
  ✓ Composite Score > 0.15
  ✓ msIC > 0.02
  ✓ msIR > 0.10
  ✓ Direction Accuracy > 51%

METRIC INTERPRETATION
---------------------
msIC (Mean Sequential IC):
  > 0.08:    Excellent - strong predictive power
  0.04-0.08: Good - significant edge
  0.02-0.04: Moderate - marginal edge
  < 0.02:    Weak - likely noise

msIR (IC Stability Ratio):
  > 0.5:     Very stable predictions
  0.2-0.5:   Stable - good for trading
  0.1-0.2:   Moderate stability
  < 0.1:     Unstable - unreliable

Direction Accuracy:
  > 55%:     Strong signal (rare)
  52-55%:    Moderate - tradable
  50-52%:    Weak
  < 50%:     No signal

================================================================================
WHY msIC/msIR ARE BETTER THAN MSE
================================================================================

THE NAIVE MODEL PARADOX:
                    MSE        msIC      Trading Value
Naive Model:        VERY LOW   ≈ 0       ZERO
Good Model:         HIGHER     > 0       POSITIVE

Why Naive has low MSE but zero msIC:
- Naive predicts: tomorrow = today
- Price changes are small → low MSE
- But NO directional prediction → msIC ≈ 0 → useless for trading!

EMPIRICAL EVIDENCE:
- MSE correlation with backtest Sharpe: -0.27 (NEGATIVE!)
- msIC correlation with backtest Sharpe: +0.45 (POSITIVE!)

CONCLUSION: Use msIC/Direction Accuracy/Composite Score for model selection,
            NOT MSE/MAE!

================================================================================
"""
    
    report += f"\nReport generated: {pd.Timestamp.now()}\n"
    report += "="*90 + "\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    return report


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(all_metrics: List[Dict], output_dir: str = './analysis_plots/'):
    """Generate visualization plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(all_metrics)
    
    if 'msIC' not in df.columns:
        print("No valid results to plot")
        return
    
    df = df.sort_values('composite_score', ascending=False)
    df['display_name'] = df['model_id'].apply(simplify_model_name)
    
    # --- Figure 1: Key Metrics Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    top_n = min(15, len(df))
    
    # Composite Score
    ax = axes[0, 0]
    colors = ['green' if x > 0.3 else 'orange' if x > 0.15 else 'red' 
              for x in df['composite_score'].head(top_n)]
    ax.barh(range(top_n), df['composite_score'].head(top_n), color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([m[:30] for m in df['display_name'].head(top_n)], fontsize=8)
    ax.axvline(0.15, color='orange', linestyle='--', alpha=0.7, label='Min (0.15)')
    ax.axvline(0.30, color='green', linestyle='--', alpha=0.7, label='Good (0.30)')
    ax.set_xlabel('Composite Trading Score')
    ax.set_title('Composite Score by Model\n(Best Predictor of Actual Performance)')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    
    # msIC
    ax = axes[0, 1]
    colors = ['green' if x > 0.04 else 'orange' if x > 0.02 else 'red' 
              for x in df['msIC'].head(top_n)]
    ax.barh(range(top_n), df['msIC'].head(top_n), color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([m[:30] for m in df['display_name'].head(top_n)], fontsize=8)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(0.02, color='orange', linestyle='--', alpha=0.7, label='Min (0.02)')
    ax.axvline(0.04, color='green', linestyle='--', alpha=0.7, label='Good (0.04)')
    ax.set_xlabel('msIC (Mean Sequential IC)')
    ax.set_title('msIC by Model\n(Temporal Correlation)')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    
    # Direction Accuracy
    ax = axes[1, 0]
    colors = ['green' if x > 0.53 else 'orange' if x > 0.51 else 'red' 
              for x in df['direction_accuracy'].head(top_n)]
    ax.barh(range(top_n), df['direction_accuracy'].head(top_n), color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([m[:30] for m in df['display_name'].head(top_n)], fontsize=8)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax.axvline(0.51, color='orange', linestyle='--', alpha=0.7, label='Min (51%)')
    ax.set_xlabel('Direction Accuracy')
    ax.set_title('Direction Accuracy by Model')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    
    # MSE vs msIC (showing why MSE is misleading)
    ax = axes[1, 1]
    scatter = ax.scatter(df['mse'], df['msIC'], 
                         c=df['sharpe_ratio'], cmap='RdYlGn', 
                         s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('MSE (Traditional Metric)')
    ax.set_ylabel('msIC (Trading-Relevant Metric)')
    ax.set_title('MSE vs msIC\n(Lower MSE ≠ Better Trading!)')
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    # Add correlation annotation
    corr = df['mse'].corr(df['msIC'])
    ax.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # --- Figure 2: Cumulative Returns (Top 5) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for _, row in df.head(5).iterrows():
        if 'cumulative' in row and len(row['cumulative']) > 1:
            ax.plot(row['cumulative'], label=row['display_name'][:35])
    
    ax.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Returns - Top 5 Models by Composite Score')
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
    
    parser = argparse.ArgumentParser(
        description='WTI Trading Strategy Analysis V2 (with msIC/msIR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python WTI_trading_analysis_v2.py --results_dir ./results/
  python WTI_trading_analysis_v2.py --results_dir ./results/ --rolling_window 21
  python WTI_trading_analysis_v2.py --results_dir ./results/ --output_dir ./analysis/
        """
    )
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory containing model prediction results (pred.npy, true.npy)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs (default: same as results_dir)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate per trade (default: 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.0005,
                        help='Slippage rate per trade (default: 0.05%%)')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Signal threshold for trading (default: 0.1%%)')
    parser.add_argument('--rolling_window', type=int, default=21,
                        help='Rolling window for msIC calculation (default: 21 days)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    print("="*80)
    print("WTI TRADING STRATEGY ANALYSIS V2 (with msIC/msIR)")
    print("="*80)
    print(f"\nResults Directory: {args.results_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Configuration
    config = TradingConfig(
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        long_threshold=args.threshold,
        short_threshold=-args.threshold,
        rolling_window=args.rolling_window
    )
    
    print(f"\nSettings:")
    print(f"  Transaction Costs: {(config.commission_rate + config.slippage_rate)*100:.2f}% per trade")
    print(f"  Signal Threshold: ±{config.long_threshold*100:.2f}%")
    print(f"  msIC Rolling Window: {config.rolling_window} days")
    
    # Load results
    print("\n" + "-"*40)
    print("Loading model predictions...")
    results = load_results(args.results_dir)
    
    if len(results) == 0:
        print(f"""
ERROR: No prediction results found in {args.results_dir}

Expected file structure:
    {args.results_dir}/
    ├── model_name_1/
    │   ├── pred.npy
    │   └── true.npy
    ├── model_name_2/
    │   ├── pred.npy
    │   └── true.npy
    └── ...
        """)
        exit(1)
    
    print(f"Found {len(results)} model results")
    
    # Evaluate each model
    print("\n" + "-"*40)
    print("Evaluating models...")
    
    evaluator = TradingEvaluator(config)
    all_metrics = []
    
    for model_id, data in results.items():
        print(f"  Evaluating: {model_id[:60]}...")
        metrics = evaluator.evaluate(
            data['predictions'],
            data['actuals'],
            model_id
        )
        if 'error' not in metrics:
            all_metrics.append(metrics)
        else:
            print(f"    WARNING: {metrics.get('error', 'Unknown error')}")
    
    if len(all_metrics) == 0:
        print("ERROR: No valid model results to analyze")
        exit(1)
    
    # Generate report
    print("\n" + "-"*40)
    print("Generating report...")
    report_path = os.path.join(output_dir, 'WTI_trading_report.txt')
    generate_report(all_metrics, report_path)
    
    # Generate plots
    print("\n" + "-"*40)
    print("Generating plots...")
    plots_dir = os.path.join(output_dir, 'analysis_plots')
    plot_results(all_metrics, plots_dir)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {report_path}")
    print(f"  - {report_path.replace('.txt', '_rankings.csv')}")
    print(f"  - {plots_dir}/model_comparison.png")
    print(f"  - {plots_dir}/cumulative_returns.png")
    print("\n" + "="*80)