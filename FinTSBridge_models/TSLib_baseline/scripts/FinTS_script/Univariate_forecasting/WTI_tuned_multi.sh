#!/usr/bin/env bash
# =============================================================================
# Multi-Period Comparison Experiment
# =============================================================================
#
# Purpose: Compare model performance across different market regimes
#
# Datasets:
#   - RECENT10: 2015-2025 (10 years, includes pre/post COVID transition)
#   - POST_COVID: 2020-2025 (5 years, pure post-COVID regime)
#
# Based on FinTSBridge paper evaluation metrics (msIC, msIR)
# File handling based on WTI_tuned.sh
#
# =============================================================================

set -euo pipefail

# Activate virtual environment
source /home/rguan/project/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# CONFIGURATION
# =============================================================================

ORIGINAL_DATA="WTI-log.csv"
ROOT_PATH="./dataset/FBD/"
TARGET="daily_return"
FREQ="d"
CHANNEL_NUM=6

# Experiment settings
ITERATIONS=3
TRAIN_EPOCHS=100
PATIENCE=15
BATCH_SIZE=32
PRED_LEN=5

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="MultiPeriod_${TIMESTAMP}"
RESULTS_DIR="./results_${EXPERIMENT_NAME}"
EVAL_DIR="${RESULTS_DIR}/evaluation"

mkdir -p ${RESULTS_DIR}
mkdir -p ${EVAL_DIR}

# Create a timestamp file to mark the start of this experiment (from WTI_tuned.sh)
START_TIMESTAMP="${RESULTS_DIR}/.start_timestamp"
touch ${START_TIMESTAMP}
START_TIME=$(date +%s)

LOG_FILE="${RESULTS_DIR}/experiment_log.txt"
EVAL_SUMMARY="${EVAL_DIR}/all_results.csv"

# Export for Python scripts
export EVAL_SUMMARY="${EVAL_SUMMARY}"
export EVAL_DIR="${EVAL_DIR}"

# Initialize CSV
echo "period,model,iteration,mse,mae,msIC,msIR,sign_change_ratio,pred_variability,direction_accuracy,pos_pred_ratio,is_regime_detector,result_dir" > ${EVAL_SUMMARY}

echo "==============================================================" | tee ${LOG_FILE}
echo "MULTI-PERIOD COMPARISON EXPERIMENT" | tee -a ${LOG_FILE}
echo "Started: $(date)" | tee -a ${LOG_FILE}
echo "Results will be saved to: ${RESULTS_DIR}" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

# =============================================================================
# STEP 1: PREPARE PERIOD-SPECIFIC DATASETS
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "[STEP 1] Preparing period-specific datasets..." | tee -a ${LOG_FILE}

python3 << PREPARE_DATA
import pandas as pd
import os

root_path = "${ROOT_PATH}"
original_file = "${ORIGINAL_DATA}"

# Read original data
df = pd.read_csv(os.path.join(root_path, original_file))
df['date'] = pd.to_datetime(df['date'])

print(f"Original data: {len(df)} rows")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Create period datasets
periods = {
    'WTI-log-2015.csv': '2015-01-01',  # ~10 years
    'WTI-log-2020.csv': '2020-01-01',  # ~5 years
}

for filename, start_date in periods.items():
    subset = df[df['date'] >= start_date].copy()
    output_path = os.path.join(root_path, filename)
    subset.to_csv(output_path, index=False)
    
    # Calculate expected train/val/test split
    n = len(subset)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    
    print(f"\n{filename}:")
    print(f"  Rows: {n}")
    print(f"  Date range: {subset['date'].min().date()} to {subset['date'].max().date()}")
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")
    print(f"  Train end date: ~{subset['date'].iloc[n_train].date()}")
    print(f"  Test start date: ~{subset['date'].iloc[n - n_test].date()}")

print("\nDatasets prepared successfully!")
PREPARE_DATA

echo "Datasets prepared." | tee -a ${LOG_FILE}

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

train_model() {
    local MODEL=$1
    local MODEL_ID=$2
    local DATA_FILE=$3
    local SEQ_LEN=$4
    local D_MODEL=$5
    local DES=$6
    
    echo "  Training ${MODEL}..."
    
    # Model-specific adjustments
    local LABEL_LEN=48
    local LR=0.0005
    local DROP=0.3
    local D_FF=$((D_MODEL * 2))
    local EXTRA_ARGS=""
    local EPOCHS=${TRAIN_EPOCHS}
    local BS=${BATCH_SIZE}
    
    # Adjust label_len based on seq_len
    if [ ${SEQ_LEN} -le 128 ]; then
        LABEL_LEN=24
    elif [ ${SEQ_LEN} -le 256 ]; then
        LABEL_LEN=32
    fi
    
    case ${MODEL} in
        "Naive")
            EPOCHS=1
            ;;
        "DLinear")
            LR=0.001
            ;;
        "TimeMixer")
            LABEL_LEN=0
            D_MODEL=16
            D_FF=32
            LR=0.01
            BS=64
            EXTRA_ARGS="--down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --channel_independence 0"
            ;;
        "TiDE")
            D_MODEL=256
            D_FF=256
            LR=0.01
            BS=256
            EXTRA_ARGS="--d_layers 2"
            ;;
        "TimesNet")
            D_FF=${D_MODEL}
            EXTRA_ARGS="--top_k 5 --num_kernels 6"
            ;;
        "Crossformer")
            D_FF=${D_MODEL}
            EXTRA_ARGS="--top_k 5 --seg_len 24"
            ;;
        "Autoformer"|"FEDformer")
            EXTRA_ARGS="--moving_avg 25 --factor 3"
            ;;
        "Koopa")
            LR=0.001
            ;;
        "Nonstationary_Transformer")
            D_MODEL=128
            D_FF=128
            EXTRA_ARGS="--p_hidden_dims 256 256 --p_hidden_layers 2"
            ;;
    esac
    
    # Run training
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_FILE} \
        --model_id ${MODEL_ID} \
        --model ${MODEL} \
        --data custom \
        --features MS \
        --seq_len ${SEQ_LEN} \
        --label_len ${LABEL_LEN} \
        --pred_len ${PRED_LEN} \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in ${CHANNEL_NUM} \
        --dec_in ${CHANNEL_NUM} \
        --c_out 1 \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --n_heads 4 \
        --dropout ${DROP} \
        --freq ${FREQ} \
        --target ${TARGET} \
        --train_epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --learning_rate ${LR} \
        --batch_size ${BS} \
        --itr ${ITERATIONS} \
        --num_workers 0 \
        --des "${DES}" \
        ${EXTRA_ARGS} || {
            echo "  WARNING: Training failed for ${MODEL}, continuing..."
            return 1
        }
    
    return 0
}

# =============================================================================
# PERIOD AND MODEL CONFIGURATION
# =============================================================================

# Format: PERIOD_NAME:DATA_FILE:SEQ_LEN:D_MODEL
declare -a PERIODS=(
    "RECENT10:WTI-log-2015.csv:256:64"
    "POST_COVID:WTI-log-2020.csv:128:32"
)

# All models to test
declare -a MODELS=("Naive" "DLinear" "PatchTST" "iTransformer" "TimeMixer" "Transformer" "Autoformer" "FEDformer" "TiDE" "TimesNet" "Crossformer" "TSMixer" "Koopa" "Nonstationary_Transformer")

# =============================================================================
# STEP 2: TRAINING
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "[STEP 2] Training models..." | tee -a ${LOG_FILE}

for PERIOD_CONFIG in "${PERIODS[@]}"; do
    # Parse config
    IFS=':' read -r PERIOD_NAME DATA_FILE SEQ_LEN D_MODEL <<< "${PERIOD_CONFIG}"
    
    echo "" | tee -a ${LOG_FILE}
    echo "==============================================================" | tee -a ${LOG_FILE}
    echo "PERIOD: ${PERIOD_NAME}" | tee -a ${LOG_FILE}
    echo "Data: ${DATA_FILE} | SeqLen: ${SEQ_LEN} | D_Model: ${D_MODEL}" | tee -a ${LOG_FILE}
    echo "==============================================================" | tee -a ${LOG_FILE}
    
    for MODEL in "${MODELS[@]}"; do
        MODEL_ID="${PERIOD_NAME}_${MODEL}_pl${PRED_LEN}"
        DES="${PERIOD_NAME}"
        
        echo "" | tee -a ${LOG_FILE}
        echo "[${PERIOD_NAME}] Model: ${MODEL}" | tee -a ${LOG_FILE}
        
        # Train
        train_model "${MODEL}" "${MODEL_ID}" "${DATA_FILE}" "${SEQ_LEN}" "${D_MODEL}" "${DES}" 2>&1 | tee -a ${LOG_FILE}
    done
done

# =============================================================================
# STEP 3: COLLECT RESULTS (from WTI_tuned.sh)
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "[STEP 3] Collecting results..." | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

COPY_COUNT=0

# Copy ALL result directories from ./results/ that were created after experiment start
if [ -d "./results" ]; then
    echo "Scanning ./results/ for new directories..." | tee -a ${LOG_FILE}
    
    # Find and copy directories modified after START_TIME
    while IFS= read -r -d '' dir; do
        # Get modification time of directory
        DIR_MTIME=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null)
        
        if [ -n "$DIR_MTIME" ] && [ "$DIR_MTIME" -ge "$START_TIME" ]; then
            cp -r "$dir" "${RESULTS_DIR}/"
            echo "  Copied directory: $(basename $dir)" | tee -a ${LOG_FILE}
            COPY_COUNT=$((COPY_COUNT + 1))
        fi
    done < <(find ./results -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
fi

# Copy result txt files created after experiment start
echo "Scanning for result .txt files..." | tee -a ${LOG_FILE}
for f in ./result_long_term_forecast_*.txt ./result*.txt; do
    if [ -f "$f" ]; then
        FILE_MTIME=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null)
        
        if [ -n "$FILE_MTIME" ] && [ "$FILE_MTIME" -ge "$START_TIME" ]; then
            cp "$f" "${RESULTS_DIR}/"
            echo "  Copied file: $(basename $f)" | tee -a ${LOG_FILE}
            COPY_COUNT=$((COPY_COUNT + 1))
        fi
    fi
done

echo "" | tee -a ${LOG_FILE}
echo "Total files/directories copied: ${COPY_COUNT}" | tee -a ${LOG_FILE}

# Remove the timestamp file
rm -f ${START_TIMESTAMP}

# =============================================================================
# STEP 4: EVALUATE RESULTS
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "[STEP 4] Evaluating results..." | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

# Re-export for evaluation
export EVAL_SUMMARY="${EVAL_SUMMARY}"
export EVAL_DIR="${EVAL_DIR}"

EVAL_COUNT=0

# Scan the RESULTS_DIR for result directories
while IFS= read -r -d '' result_dir; do
    dir_name=$(basename "$result_dir")
    
    # Skip evaluation directory and non-result directories
    [[ "$dir_name" == "evaluation" ]] && continue
    [[ "$dir_name" == .* ]] && continue
    [[ ! "$dir_name" == long_term_forecast_* ]] && continue
    
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

echo "" | tee -a ${LOG_FILE}
echo "Total results evaluated: ${EVAL_COUNT}" | tee -a ${LOG_FILE}

# =============================================================================
# STEP 5: FINAL ANALYSIS
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "[STEP 5] Final Analysis..." | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

# Re-export variables for Python
export EVAL_SUMMARY="${EVAL_SUMMARY}"
export EVAL_DIR="${EVAL_DIR}"

python3 << 'FINAL_ANALYSIS'
import pandas as pd
import numpy as np
import os

eval_summary = os.environ.get('EVAL_SUMMARY', './evaluation/all_results.csv')
eval_dir = os.environ.get('EVAL_DIR', './evaluation')

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
print(f"\nDetailed results saved to: {os.path.join(eval_dir, 'period_comparison.csv')}")

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

# =============================================================================
# SUMMARY
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "EXPERIMENT COMPLETE" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "End Time: $(date)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Output Files:" | tee -a ${LOG_FILE}
echo "  Results:           ${RESULTS_DIR}/" | tee -a ${LOG_FILE}
echo "  All Results CSV:   ${EVAL_SUMMARY}" | tee -a ${LOG_FILE}
echo "  Period Comparison: ${EVAL_DIR}/period_comparison.csv" | tee -a ${LOG_FILE}
echo "  Best Models:       ${EVAL_DIR}/best_models.csv" | tee -a ${LOG_FILE}
echo "  Log:               ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# List what was collected
echo "Files and directories collected:" | tee -a ${LOG_FILE}
ls -la ${RESULTS_DIR}/ | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}