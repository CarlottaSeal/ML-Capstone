# =============================================================================
# Hypothesis 1.1: Mean-Reversion vs Momentum Detection Analysis
# =============================================================================
# 
# 研究问题: 模型预测是符合 Mean-Reversion 还是 Momentum 模式?
#
# 方法论:
# --------
# 1. 预测与过去收益相关性 (Prediction-Past Return Correlation)
#    - 计算: corr(pred_t, past_return_t)
#    - Mean-Reversion: 负相关 (过去涨 → 预测跌)
#    - Momentum: 正相关 (过去涨 → 预测涨)
#
# 2. 条件预测分析 (Conditional Prediction Analysis)
#    - 当过去N天收益 > 0 时，平均预测值是正还是负?
#    - Mean-Reversion: 平均预测 < 0
#    - Momentum: 平均预测 > 0
#
# 3. 符号翻转率 (Sign Flip Rate)
#    - 预测符号与过去收益符号相反的比例
#    - Mean-Reversion: > 55%
#    - Momentum: < 45%
#
# 4. Hurst Exponent (预测序列)
#    - H < 0.5: 反持续性 (Mean-Reversion)
#    - H > 0.5: 持续性 (Momentum)
#    - H ≈ 0.5: 随机游走
#
# 5. 预测自相关 (Prediction Autocorrelation)
#    - 负自相关: Mean-Reversion
#    - 正自相关: Momentum
#
# 判定标准:
# ---------
# - 至少3个指标一致 → 归类为该模式
# - 指标冲突或接近阈值 → "Neither/Unclear"
#
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置参数
# =============================================================================

# 结果文件夹路径 - 请根据实际情况修改
RESULTS_DIR = "./results_WTI_trading_extend_20251128_183138"

# 过去收益计算窗口 (用于判断过去是涨还是跌)
PAST_WINDOW = 15  # 过去15天的累计收益

# 判定阈值
CORR_THRESHOLD = 0.05  # 相关性阈值
SIGN_FLIP_MR_THRESHOLD = 0.55  # Mean-Reversion 符号翻转率阈值
SIGN_FLIP_MOM_THRESHOLD = 0.45  # Momentum 符号翻转率阈值
HURST_THRESHOLD_LOW = 0.45  # Hurst < 0.45 → Mean-Reversion
HURST_THRESHOLD_HIGH = 0.55  # Hurst > 0.55 → Momentum

CORR_THRESHOLD = 0.03           # 更严格：0.05 → 0.03
SIGN_FLIP_MR_THRESHOLD = 0.53   # 更严格：0.55 → 0.53
SIGN_FLIP_MOM_THRESHOLD = 0.47  # 更严格：0.45 → 0.47
HURST_THRESHOLD_LOW = 0.47      # 更严格：0.45 → 0.47
HURST_THRESHOLD_HIGH = 0.53     # 更严格：0.55 → 0.53

# =============================================================================
# 辅助函数
# =============================================================================

def extract_model_info(folder_name):
    """从文件夹名称提取模型信息"""
    parts = folder_name.split('_')
    model_type = None
    for part in parts:
        if part in ['Transformer', 'Informer', 'TimesNet', 'TimeMixer', 
                    'PatchTST', 'iTransformer', 'DLinear', 'TiDE', 
                    'Crossformer', 'FEDformer', 'Autoformer', 'Naive']:
            model_type = part
            break
    return model_type or 'Unknown'


def load_predictions(pred_file):
    """
    加载预测结果文件
    预期格式: date, true_0, true_1, ..., pred_0, pred_1, ...
    """
    df = pd.read_csv(pred_file)
    return df


def calculate_hurst_exponent(series, max_lag=20):
    """
    计算 Hurst Exponent (R/S 方法)
    H < 0.5: 反持续性 (Mean-Reversion)
    H = 0.5: 随机游走
    H > 0.5: 持续性 (Momentum)
    """
    series = np.array(series)
    series = series[~np.isnan(series)]
    
    if len(series) < max_lag * 2:
        return np.nan
    
    lags = range(2, max_lag + 1)
    rs_values = []
    
    for lag in lags:
        rs_list = []
        for start in range(0, len(series) - lag, lag):
            subseries = series[start:start + lag]
            if len(subseries) < lag:
                continue
            
            mean = np.mean(subseries)
            cumdev = np.cumsum(subseries - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(subseries, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(np.nan)
    
    # 过滤有效值
    valid_idx = ~np.isnan(rs_values)
    if np.sum(valid_idx) < 3:
        return np.nan
    
    log_lags = np.log(np.array(lags)[valid_idx])
    log_rs = np.log(np.array(rs_values)[valid_idx])
    
    # 线性回归
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    
    return slope


def analyze_model_pattern(df, model_name):
    """
    分析单个模型的 Mean-Reversion vs Momentum 模式
    
    Returns:
        dict: 包含所有诊断指标和最终判定
    """
    results = {
        'model': model_name,
        'n_samples': len(df)
    }
    
    # 获取预测列和真实值列
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    true_cols = [c for c in df.columns if c.startswith('true_')]
    
    if not pred_cols or not true_cols:
        results['error'] = 'Missing pred/true columns'
        return results
    
    # 使用第一个预测horizon (pred_0 = t+1 预测)
    pred = df['pred_0'].values
    true = df['true_0'].values
    
    # 计算过去收益 (使用 true 的滞后值作为过去收益的代理)
    # 注意: 这里我们用 true_0 的滞后值作为 "过去收益"
    past_return = np.roll(true, PAST_WINDOW)
    past_return[:PAST_WINDOW] = np.nan
    
    # 也可以用累计过去收益
    past_cum_return = pd.Series(true).rolling(window=PAST_WINDOW).sum().shift(1).values
    
    # 过滤有效数据
    valid_mask = ~(np.isnan(past_cum_return) | np.isnan(pred))
    past_valid = past_cum_return[valid_mask]
    pred_valid = pred[valid_mask]
    true_valid = true[valid_mask]
    
    if len(pred_valid) < 30:
        results['error'] = 'Insufficient valid samples'
        return results
    
    # -------------------------------------------------------------------------
    # 诊断 1: 预测与过去收益相关性
    # -------------------------------------------------------------------------
    corr_pred_past, p_value = stats.pearsonr(pred_valid, past_valid)
    results['corr_pred_past'] = corr_pred_past
    results['corr_pred_past_pvalue'] = p_value
    
    if corr_pred_past < -CORR_THRESHOLD:
        results['corr_signal'] = 'Mean-Reversion'
    elif corr_pred_past > CORR_THRESHOLD:
        results['corr_signal'] = 'Momentum'
    else:
        results['corr_signal'] = 'Neither'
    
    # -------------------------------------------------------------------------
    # 诊断 2: 条件预测分析
    # -------------------------------------------------------------------------
    # 当过去收益 > 0 时的平均预测
    past_positive_mask = past_valid > 0
    past_negative_mask = past_valid < 0
    
    avg_pred_when_past_positive = np.mean(pred_valid[past_positive_mask]) if np.sum(past_positive_mask) > 0 else np.nan
    avg_pred_when_past_negative = np.mean(pred_valid[past_negative_mask]) if np.sum(past_negative_mask) > 0 else np.nan
    
    results['avg_pred_past_positive'] = avg_pred_when_past_positive
    results['avg_pred_past_negative'] = avg_pred_when_past_negative
    
    # 判断: 过去涨时预测跌 (MR) 还是预测涨 (Mom)
    if not np.isnan(avg_pred_when_past_positive):
        if avg_pred_when_past_positive < 0:
            results['conditional_signal'] = 'Mean-Reversion'
        elif avg_pred_when_past_positive > 0:
            results['conditional_signal'] = 'Momentum'
        else:
            results['conditional_signal'] = 'Neither'
    else:
        results['conditional_signal'] = 'Unknown'
    
    # -------------------------------------------------------------------------
    # 诊断 3: 符号翻转率
    # -------------------------------------------------------------------------
    # 预测符号与过去收益符号相反的比例
    pred_sign = np.sign(pred_valid)
    past_sign = np.sign(past_valid)
    
    # 排除 0 的情况
    non_zero_mask = (pred_sign != 0) & (past_sign != 0)
    if np.sum(non_zero_mask) > 0:
        sign_flip_rate = np.mean(pred_sign[non_zero_mask] != past_sign[non_zero_mask])
    else:
        sign_flip_rate = np.nan
    
    results['sign_flip_rate'] = sign_flip_rate
    
    if not np.isnan(sign_flip_rate):
        if sign_flip_rate > SIGN_FLIP_MR_THRESHOLD:
            results['sign_flip_signal'] = 'Mean-Reversion'
        elif sign_flip_rate < SIGN_FLIP_MOM_THRESHOLD:
            results['sign_flip_signal'] = 'Momentum'
        else:
            results['sign_flip_signal'] = 'Neither'
    else:
        results['sign_flip_signal'] = 'Unknown'
    
    # -------------------------------------------------------------------------
    # 诊断 4: Hurst Exponent (预测序列)
    # -------------------------------------------------------------------------
    hurst = calculate_hurst_exponent(pred_valid)
    results['hurst_exponent'] = hurst
    
    if not np.isnan(hurst):
        if hurst < HURST_THRESHOLD_LOW:
            results['hurst_signal'] = 'Mean-Reversion'
        elif hurst > HURST_THRESHOLD_HIGH:
            results['hurst_signal'] = 'Momentum'
        else:
            results['hurst_signal'] = 'Neither'
    else:
        results['hurst_signal'] = 'Unknown'
    
    # -------------------------------------------------------------------------
    # 诊断 5: 预测自相关 (Lag-1)
    # -------------------------------------------------------------------------
    pred_autocorr = pd.Series(pred_valid).autocorr(lag=1)
    results['pred_autocorr'] = pred_autocorr
    
    if not np.isnan(pred_autocorr):
        if pred_autocorr < -CORR_THRESHOLD:
            results['autocorr_signal'] = 'Mean-Reversion'
        elif pred_autocorr > CORR_THRESHOLD:
            results['autocorr_signal'] = 'Momentum'
        else:
            results['autocorr_signal'] = 'Neither'
    else:
        results['autocorr_signal'] = 'Unknown'
    
    # -------------------------------------------------------------------------
    # 综合判定
    # -------------------------------------------------------------------------
    signals = [
        results.get('corr_signal', 'Unknown'),
        results.get('conditional_signal', 'Unknown'),
        results.get('sign_flip_signal', 'Unknown'),
        results.get('hurst_signal', 'Unknown'),
        results.get('autocorr_signal', 'Unknown')
    ]
    
    # 统计各类信号的数量
    mr_count = signals.count('Mean-Reversion')
    mom_count = signals.count('Momentum')
    neither_count = signals.count('Neither')
    
    results['mr_votes'] = mr_count
    results['mom_votes'] = mom_count
    results['neither_votes'] = neither_count
    
    # 最终判定: 至少3票一致
    if mr_count >= 3:
        results['final_pattern'] = 'Mean-Reversion'
    elif mom_count >= 3:
        results['final_pattern'] = 'Momentum'
    elif neither_count >= 3:
        results['final_pattern'] = 'Neither (No Clear Pattern)'
    else:
        results['final_pattern'] = 'Mixed/Unclear'
    
    # -------------------------------------------------------------------------
    # 额外分析: 按预测horizon分析
    # -------------------------------------------------------------------------
    horizon_patterns = []
    for i, col in enumerate(pred_cols[:6]):  # 前6个horizon
        if col in df.columns:
            pred_h = df[col].values
            valid_h = ~(np.isnan(past_cum_return) | np.isnan(pred_h))
            if np.sum(valid_h) > 30:
                corr_h, _ = stats.pearsonr(pred_h[valid_h], past_cum_return[valid_h])
                horizon_patterns.append({
                    'horizon': i,
                    'corr': corr_h,
                    'pattern': 'MR' if corr_h < -CORR_THRESHOLD else ('Mom' if corr_h > CORR_THRESHOLD else 'N')
                })
    
    results['horizon_analysis'] = horizon_patterns
    
    return results


def find_prediction_files(results_dir):
    """
    在结果文件夹中查找所有预测文件
    """
    prediction_files = []
    
    # 遍历所有子文件夹
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            # 查找预测文件 (通常命名为 *pred*.csv 或 *prediction*.csv)
            for pattern in ['*pred*.csv', '*prediction*.csv', '*results*.csv']:
                files = glob.glob(os.path.join(folder_path, pattern))
                for f in files:
                    prediction_files.append({
                        'folder': folder,
                        'file': f,
                        'model_type': extract_model_info(folder)
                    })
            
            # 如果没找到，尝试查找任何CSV文件
            if not any(pf['folder'] == folder for pf in prediction_files):
                csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
                for f in csv_files:
                    prediction_files.append({
                        'folder': folder,
                        'file': f,
                        'model_type': extract_model_info(folder)
                    })
    
    return prediction_files


# =============================================================================
# 主分析函数
# =============================================================================

def run_analysis(results_dir=RESULTS_DIR):
    """
    运行完整分析
    """
    print("=" * 100)
    print("HYPOTHESIS 1.1: MEAN-REVERSION vs MOMENTUM PATTERN ANALYSIS")
    print("=" * 100)
    print(f"\nResults directory: {results_dir}")
    print(f"Past return window: {PAST_WINDOW} days")
    print(f"Correlation threshold: ±{CORR_THRESHOLD}")
    print()
    
    # 查找预测文件
    pred_files = find_prediction_files(results_dir)
    print(f"Found {len(pred_files)} prediction files")
    
    if len(pred_files) == 0:
        print("\n⚠️  No prediction files found!")
        print("Please check the results directory structure.")
        print("\nExpected structure:")
        print("  results_dir/")
        print("    └── model_folder_1/")
        print("        └── predictions.csv (with pred_0, pred_1, ... and true_0, true_1, ... columns)")
        return None
    
    # 分析每个模型
    all_results = []
    
    for pf in pred_files:
        print(f"\nAnalyzing: {pf['folder']}")
        try:
            df = load_predictions(pf['file'])
            result = analyze_model_pattern(df, pf['folder'])
            result['model_type'] = pf['model_type']
            result['file'] = pf['file']
            all_results.append(result)
            
            if 'error' not in result:
                print(f"  → Pattern: {result['final_pattern']}")
                print(f"     Votes: MR={result['mr_votes']}, Mom={result['mom_votes']}, Neither={result['neither_votes']}")
        except Exception as e:
            print(f"  ⚠️  Error: {str(e)}")
            all_results.append({
                'model': pf['folder'],
                'model_type': pf['model_type'],
                'error': str(e)
            })
    
    # 汇总结果
    results_df = pd.DataFrame(all_results)
    
    # 打印汇总表格
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    
    if 'final_pattern' in results_df.columns:
        summary_cols = ['model_type', 'final_pattern', 'mr_votes', 'mom_votes', 
                        'corr_pred_past', 'sign_flip_rate', 'hurst_exponent']
        available_cols = [c for c in summary_cols if c in results_df.columns]
        
        print(results_df[available_cols].to_string(index=False))
        
        # 按模型类型汇总
        print("\n" + "=" * 100)
        print("PATTERN DISTRIBUTION BY MODEL TYPE")
        print("=" * 100)
        
        pattern_summary = results_df.groupby(['model_type', 'final_pattern']).size().unstack(fill_value=0)
        print(pattern_summary)
        
        # 整体统计
        print("\n" + "=" * 100)
        print("OVERALL STATISTICS")
        print("=" * 100)
        
        pattern_counts = results_df['final_pattern'].value_counts()
        total = len(results_df)
        
        for pattern, count in pattern_counts.items():
            pct = count / total * 100
            print(f"  {pattern}: {count} ({pct:.1f}%)")
        
        # 判定结论
        print("\n" + "=" * 100)
        print("CONCLUSION")
        print("=" * 100)
        
        dominant_pattern = pattern_counts.index[0] if len(pattern_counts) > 0 else 'Unknown'
        dominant_count = pattern_counts.iloc[0] if len(pattern_counts) > 0 else 0
        dominant_pct = dominant_count / total * 100 if total > 0 else 0
        
        if dominant_pct > 60:
            print(f"\n✓ DOMINANT PATTERN: {dominant_pattern} ({dominant_pct:.1f}% of models)")
        elif dominant_pct > 40:
            print(f"\n◐ WEAK DOMINANT PATTERN: {dominant_pattern} ({dominant_pct:.1f}% of models)")
        else:
            print(f"\n✗ NO CLEAR DOMINANT PATTERN")
            print(f"  Models show mixed behavior - no consistent mean-reversion or momentum signal")
    
    # 保存结果
    output_file = 'hypothesis1_1_mr_vs_momentum_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    return results_df


# =============================================================================
# 运行分析
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # 可以通过命令行参数指定结果目录
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = RESULTS_DIR
    
    # 检查目录是否存在
    if not os.path.exists(results_dir):
        print(f"⚠️  Directory not found: {results_dir}")
        print("\nPlease provide the correct path:")
        print("  python hypothesis1_1_analysis.py /path/to/results_folder")
        print("\nOr modify RESULTS_DIR in the script.")
    else:
        results_df = run_analysis(results_dir)