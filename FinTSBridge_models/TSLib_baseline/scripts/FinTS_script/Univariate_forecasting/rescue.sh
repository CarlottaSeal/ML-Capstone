#!/usr/bin/env bash
# =============================================================================
# Rescue Script: Evaluate existing results from multi-period experiment
# Based on WTI_tuned.sh file handling logic
# =============================================================================

set -euo pipefail

source /home/rguan/project/venv/bin/activate

# =============================================================================
# FIND OR CREATE RESULTS DIRECTORY
# =============================================================================

# Find the most recent MultiPeriod experiment directory
RESULTS_DIR=$(ls -td ./results_MultiPeriod_* 2>/dev/null | head -1 || true)

if [ -z "${RESULTS_DIR}" ]; then
    # Create a new results directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RESULTS_DIR="./results_MultiPeriod_rescue_${TIMESTAMP}"
    mkdir -p ${RESULTS_DIR}
    echo "Created new results directory: ${RESULTS_DIR}"
else
    echo "Found existing results directory: ${RESULTS_DIR}"
fi

EVAL_DIR="${RESULTS_DIR}/evaluation"
mkdir -p ${EVAL_DIR}

# Export for Python scripts - CRITICAL: must export these
export EVAL_DIR="${EVAL_DIR}"
export EVAL_SUMMARY="${EVAL_DIR}/all_results.csv"

# Initialize CSV
echo "period,model,iteration,mse,mae,msIC,msIR,sign_change_ratio,pred_variability,direction_accuracy,pos_pred_ratio,is_regime_detector,result_dir" > ${EVAL_SUMMARY}

echo ""
echo "=============================================================="
echo "STEP 1: COPYING RESULT DIRECTORIES"
echo "=============================================================="

# =============================================================================
# COPY ALL RESULT DIRECTORIES (following WTI_tuned.sh pattern)
# =============================================================================

COPY_COUNT=0

if [ -d "./results" ]; then
    echo "Scanning ./results/ for experiment directories..."
    echo ""
    
    # Find and copy directories matching our experiment pattern
    # Using the exact pattern from WTI_tuned.sh
    while IFS= read -r -d '' dir; do
        dir_name=$(basename "$dir")
        
        # Check if it's from our experiment (RECENT10 or POST_COVID)
        if [[ "$dir_name" == *"RECENT10"* ]] || [[ "$dir_name" == *"POST_COVID"* ]]; then
            # Copy to our results directory if not already there
            if [ ! -d "${RESULTS_DIR}/${dir_name}" ]; then
                cp -r "$dir" "${RESULTS_DIR}/"
                echo "  Copied: ${dir_name}"
                COPY_COUNT=$((COPY_COUNT + 1))
            else
                echo "  Already exists: ${dir_name}"
            fi
        fi
    done < <(find ./results -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
fi

echo ""
echo "Total directories copied: ${COPY_COUNT}"

# Also copy any result txt files
echo ""
echo "Scanning for result .txt files..."
for f in ./result_long_term_forecast_*.txt ./result*.txt; do
    if [ -f "$f" ]; then
        fname=$(basename "$f")
        if [[ "$fname" == *"RECENT10"* ]] || [[ "$fname" == *"POST_COVID"* ]]; then
            if [ ! -f "${RESULTS_DIR}/${fname}" ]; then
                cp "$f" "${RESULTS_DIR}/"
                echo "  Copied file: ${fname}"
            fi
        fi
    fi
done

# =============================================================================
# STEP 2: EVALUATE ALL RESULTS
# =============================================================================

echo ""
echo "=============================================================="
echo "STEP 2: EVALUATING RESULTS"
echo "=============================================================="

EVAL_COUNT=0

# Now scan the RESULTS_DIR (which has all the copied directories)
echo "Scanning ${RESULTS_DIR}/ for result directories..."
echo ""

while IFS= read -r -d '' result_dir; do
    dir_name=$(basename "$result_dir")
    
    # Skip evaluation directory
    [[ "$dir_name" == "evaluation" ]] && continue
    
    # Skip non-directories
    [ ! -d "$result_dir" ] && continue
    
    # Check if it's from our experiment (RECENT10 or POST_COVID)
    PERIOD=""
    if [[ "$dir_name" == *"RECENT10"* ]]; then
        PERIOD="RECENT10"
    elif [[ "$dir_name" == *"POST_COVID"* ]]; then
        PERIOD="POST_COVID"
    else
        continue
    fi
    
    # Extract iteration number (last number after underscore)
    ITER=$(echo "$dir_name" | rev | cut -d'_' -f1 | rev)
    
    # Validate ITER is a number
    if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
        ITER="0"
    fi
    
    # Extract model name
    MODEL=""
    for m in Naive DLinear PatchTST iTransformer TimeMixer Transformer Autoformer FEDformer TiDE TimesNet Crossformer TSMixer Koopa Nonstationary_Transformer; do
        if [[ "$dir_name" == *"_${m}_"* ]]; then
            MODEL="$m"
            break
        fi
    done
    
    if [ -z "$MODEL" ]; then
        echo "  Skipping (unknown model): $dir_name"
        continue
    fi
    
    echo "  Evaluating: ${PERIOD} / ${MODEL} / iter ${ITER}"
    
    # Run evaluation
    python3 << PYTHON_EVAL
import numpy as np
import pandas as pd
from scipy import stats
import os
import sys

result_dir = "${result_dir}"
period = "${PERIOD}"
model_name = "${MODEL}"
iteration = "${ITER}"
eval_summary = os.environ.get('EVAL_SUMMARY', "${EVAL_SUMMARY}")

def load_predictions(result_dir):
    """Load predictions from result directory."""
    # Try npy files first
    pred_path = os.path.join(result_dir, 'pred.npy')
    true_path = os.path.join(result_dir, 'true.npy')
    if os.path.exists(pred_path) and os.path.exists(true_path):
        pred = np.load(pred_path)
        true = np.load(true_path)
        if pred.ndim == 3:
            pred = pred[:, :, 0]
            true = true[:, :, 0]
        elif pred.ndim == 1:
            pred = pred.reshape(-1, 1)
            true = true.reshape(-1, 1)
        return pred, true
    
    # Try data_table.csv
    csv_path = os.path.join(result_dir, 'data_table.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        pred_cols = sorted([c for c in df.columns if c.startswith('pred_')])
        true_cols = sorted([c for c in df.columns if c.startswith('true_')])
        if pred_cols and true_cols:
            return df[pred_cols].values, df[true_cols].values
    
    return None, None

def calculate_msIC(pred, true):
    n_samples = pred.shape[0]
    pred_len = pred.shape[1] if pred.ndim > 1 else 1
    
    if pred_len < 3:
        p = pred.flatten()
        t = true.flatten()
        if np.std(p) < 1e-10:
            return 0.0
        corr, _ = stats.spearmanr(p, t)
        return corr if not np.isnan(corr) else 0.0
    
    correlations = []
    for i in range(n_samples):
        y_pred = pred[i, :]
        y_true = true[i, :]
        if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
            continue
        rho, _ = stats.spearmanr(y_true, y_pred)
        if not np.isnan(rho):
            correlations.append(rho)
    
    return np.mean(correlations) if correlations else 0.0

def calculate_msIR(pred, true):
    n_samples = pred.shape[0]
    pred_len = pred.shape[1] if pred.ndim > 1 else 1
    
    if pred_len < 3:
        p = pred.flatten()
        t = true.flatten()
        window = min(60, len(p) // 4)
        if window < 20:
            return 0.0
        
        correlations = []
        for i in range(window, len(p)):
            wp = p[i-window:i]
            wt = t[i-window:i]
            if np.std(wp) > 1e-10:
                rho, _ = stats.spearmanr(wp, wt)
                if not np.isnan(rho):
                    correlations.append(rho)
        
        if len(correlations) < 2 or np.std(correlations) < 1e-10:
            return 0.0
        return np.mean(correlations) / np.std(correlations)
    
    correlations = []
    for i in range(n_samples):
        y_pred = pred[i, :]
        y_true = true[i, :]
        if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
            continue
        rho, _ = stats.spearmanr(y_true, y_pred)
        if not np.isnan(rho):
            correlations.append(rho)
    
    if len(correlations) < 2 or np.std(correlations) < 1e-10:
        return 0.0
    
    return np.mean(correlations) / np.std(correlations)

def calculate_sign_change_ratio(pred):
    p = pred.flatten()
    if len(p) < 2:
        return 0.0
    signs = np.sign(p)
    changes = np.sum(np.abs(np.diff(signs)) > 0)
    return changes / (len(p) - 1)

def calculate_pred_variability(pred):
    p = pred.flatten()
    mean_abs = np.mean(np.abs(p))
    if mean_abs < 1e-10:
        return 0.0
    return np.std(p) / mean_abs

def calculate_direction_accuracy(pred, true):
    p = pred.flatten()
    t = true.flatten()
    return np.mean(np.sign(p) == np.sign(t))

def calculate_pos_pred_ratio(pred):
    p = pred.flatten()
    return np.mean(p > 0)

# Main evaluation
pred, true = load_predictions(result_dir)

if pred is None:
    print(f"    ERROR: Cannot load predictions")
    with open(eval_summary, 'a') as f:
        f.write(f"{period},{model_name},{iteration},NA,NA,NA,NA,NA,NA,NA,NA,NA,{result_dir}\n")
    sys.exit(0)

# Calculate all metrics
mse = np.mean((pred - true) ** 2)
mae = np.mean(np.abs(pred - true))
msIC = calculate_msIC(pred, true)
msIR = calculate_msIR(pred, true)
sign_chg = calculate_sign_change_ratio(pred)
pred_var = calculate_pred_variability(pred)
dir_acc = calculate_direction_accuracy(pred, true)
pos_ratio = calculate_pos_pred_ratio(pred)

is_regime = "YES" if sign_chg < 0.1 else "NO"

with open(eval_summary, 'a') as f:
    f.write(f"{period},{model_name},{iteration},{mse:.6f},{mae:.6f},{msIC:.4f},{msIR:.4f},{sign_chg:.4f},{pred_var:.4f},{dir_acc:.4f},{pos_ratio:.4f},{is_regime},{result_dir}\n")

print(f"    MSE: {mse:.6f} | msIC: {msIC:.4f} | DirAcc: {dir_acc:.2%} | Regime: {is_regime}")

PYTHON_EVAL

    EVAL_COUNT=$((EVAL_COUNT + 1))
    
done < <(find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

echo ""
echo "Total results evaluated: ${EVAL_COUNT}"

if [ ${EVAL_COUNT} -eq 0 ]; then
    echo "ERROR: No results found to evaluate!"
    echo "Please check that ./results/ contains directories with RECENT10 or POST_COVID in their names"
    exit 1
fi

# =============================================================================
# STEP 3: FINAL ANALYSIS
# =============================================================================

echo ""
echo "=============================================================="
echo "STEP 3: FINAL ANALYSIS"
echo "=============================================================="

# Re-export for final analysis
export EVAL_DIR="${EVAL_DIR}"
export EVAL_SUMMARY="${EVAL_SUMMARY}"

python3 << 'FINAL_ANALYSIS'
import pandas as pd
import numpy as np
import os

eval_dir = os.environ.get('EVAL_DIR', './evaluation')
eval_summary = os.environ.get('EVAL_SUMMARY', os.path.join(eval_dir, 'all_results.csv'))

print(f"Loading results from: {eval_summary}")

# Load results
df = pd.read_csv(eval_summary)

# Filter out failed runs
df = df[df['mse'] != 'NA'].copy()
for col in ['mse', 'mae', 'msIC', 'msIR', 'sign_change_ratio', 'pred_variability', 'direction_accuracy', 'pos_pred_ratio']:
    df[col] = df[col].astype(float)

if len(df) == 0:
    print("ERROR: No valid results found!")
    exit(1)

print(f"Found {len(df)} valid results")
print("\n" + "=" * 100)
print("CROSS-PERIOD MODEL COMPARISON")
print("=" * 100)

# Aggregate by period and model
agg = df.groupby(['period', 'model']).agg({
    'mse': ['mean', 'std'],
    'msIC': ['mean', 'std'],
    'msIR': ['mean'],
    'direction_accuracy': ['mean'],
    'sign_change_ratio': ['mean'],
    'pos_pred_ratio': ['mean'],
    'is_regime_detector': lambda x: (x == 'YES').sum(),
    'iteration': 'count'
}).round(4)

agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg = agg.reset_index()
agg = agg.rename(columns={'iteration_count': 'n_runs', 'is_regime_detector_<lambda>': 'regime_count'})

# Print results by period
for period in sorted(df['period'].unique()):
    period_df = agg[agg['period'] == period].sort_values('msIC_mean', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"PERIOD: {period}")
    print(f"{'='*60}")
    print(f"{'Model':<28} {'msIC':<14} {'msIR':<8} {'DirAcc':<8} {'SignChg':<8} {'PosPred':<8} {'Regime?':<10}")
    print("-" * 95)
    
    for _, row in period_df.iterrows():
        regime_flag = f"⚠️ {int(row['regime_count'])}/{int(row['n_runs'])}" if row['regime_count'] > 0 else "OK"
        msic_std = row['msIC_std'] if not pd.isna(row['msIC_std']) else 0
        print(f"{row['model']:<28} "
              f"{row['msIC_mean']:.4f}±{msic_std:.3f} "
              f"{row['msIR_mean']:.3f}    "
              f"{row['direction_accuracy_mean']:.2%}   "
              f"{row['sign_change_ratio_mean']:.3f}    "
              f"{row['pos_pred_ratio_mean']:.2%}   "
              f"{regime_flag}")

# Cross-period comparison
print("\n" + "=" * 100)
print("MODEL STABILITY ACROSS PERIODS")
print("=" * 100)

pivot = agg.pivot(index='model', columns='period', values='msIC_mean')
if len(pivot.columns) >= 2:
    print(f"\n{'Model':<28}", end="")
    for col in sorted(pivot.columns):
        print(f"{col:<15}", end="")
    print(f"{'Diff':<10} {'Stable?':<10}")
    print("-" * 85)
    
    for model in pivot.index:
        print(f"{model:<28}", end="")
        values = []
        for col in sorted(pivot.columns):
            val = pivot.loc[model, col]
            values.append(val)
            if pd.isna(val):
                print(f"{'N/A':<15}", end="")
            else:
                print(f"{val:.4f}         ", end="")
        
        if len(values) >= 2 and not any(pd.isna(values)):
            diff = abs(values[0] - values[1])
            stable = "✓ Yes" if diff < 0.02 else "✗ No"
            print(f"{diff:.4f}     {stable}")
        else:
            print("N/A")

# Best model for each period
print("\n" + "=" * 100)
print("BEST MODEL FOR EACH PERIOD (excluding regime detectors)")
print("=" * 100)

for period in sorted(df['period'].unique()):
    period_df = agg[(agg['period'] == period) & (agg['regime_count'] == 0)]
    if len(period_df) > 0:
        best = period_df.sort_values('msIC_mean', ascending=False).iloc[0]
        print(f"\n{period}:")
        print(f"  Best Model: {best['model']}")
        print(f"  msIC: {best['msIC_mean']:.4f} ± {best['msIC_std']:.4f}")
        print(f"  msIR: {best['msIR_mean']:.4f}")
        print(f"  Direction Accuracy: {best['direction_accuracy_mean']:.2%}")
    else:
        period_df = agg[agg['period'] == period]
        if len(period_df) > 0:
            best = period_df.sort_values('msIC_mean', ascending=False).iloc[0]
            print(f"\n{period}: (⚠️ all models are regime detectors)")
            print(f"  Best Model: {best['model']}")
            print(f"  msIC: {best['msIC_mean']:.4f}")

# Recommendations
print("\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)

# Find model that works best across periods
cross_period = agg.groupby('model').agg({
    'msIC_mean': 'mean',
    'msIC_std': 'mean',
    'regime_count': 'sum'
}).reset_index()
cross_period['stability'] = cross_period['msIC_mean'] / (cross_period['msIC_std'] + 0.01)
non_regime = cross_period[cross_period['regime_count'] == 0]

if len(non_regime) > 0:
    most_stable = non_regime.sort_values('stability', ascending=False).iloc[0]
    highest_ic = non_regime.sort_values('msIC_mean', ascending=False).iloc[0]
    
    print(f"\n1. Most Stable Model (consistent across periods): {most_stable['model']}")
    print(f"   Average msIC: {most_stable['msIC_mean']:.4f}")
    
    print(f"\n2. Highest Average msIC: {highest_ic['model']}")
    print(f"   Average msIC: {highest_ic['msIC_mean']:.4f}")
    
    if most_stable['model'] != highest_ic['model']:
        print(f"\n⚠️  Note: Most stable ≠ highest IC. Consider your priority:")
        print(f"   - For live trading: prefer stable model ({most_stable['model']})")
        print(f"   - For max returns: prefer high IC model ({highest_ic['model']})")
else:
    print("\n⚠️  All models show regime detector behavior!")

# Save results
agg.to_csv(os.path.join(eval_dir, 'period_comparison.csv'), index=False)
print(f"\nResults saved to: {os.path.join(eval_dir, 'period_comparison.csv')}")

# Save best model info
best_info = []
for period in sorted(df['period'].unique()):
    period_df = agg[(agg['period'] == period) & (agg['regime_count'] == 0)]
    if len(period_df) > 0:
        best = period_df.sort_values('msIC_mean', ascending=False).iloc[0]
        best_info.append({
            'period': period,
            'best_model': best['model'],
            'msIC': best['msIC_mean'],
            'msIR': best['msIR_mean'],
            'direction_accuracy': best['direction_accuracy_mean']
        })

if best_info:
    pd.DataFrame(best_info).to_csv(os.path.join(eval_dir, 'best_models.csv'), index=False)
    print(f"Best models saved to: {os.path.join(eval_dir, 'best_models.csv')}")

FINAL_ANALYSIS

echo ""
echo "=============================================================="
echo "RESCUE COMPLETE"
echo "=============================================================="
echo "Results directory: ${RESULTS_DIR}"
echo "Evaluation directory: ${EVAL_DIR}"
echo ""
echo "Files created:"
ls -la ${EVAL_DIR}/
echo ""
echo "Directories in results:"
ls -d ${RESULTS_DIR}/*/ 2>/dev/null | wc -l
echo "total result directories"