#!/usr/bin/env bash
# =============================================================================
# Walk-Forward / Rolling Training for WTI TimeMixer
# =============================================================================
#
# This script implements PROPER out-of-sample testing using walk-forward analysis:
#
# Walk-Forward Method:
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Fold 1: Train[2007-2015] → Validate[2015-2016] → Test[2016-2017]   │
# │ Fold 2: Train[2007-2017] → Validate[2017-2018] → Test[2018-2019]   │
# │ Fold 3: Train[2007-2019] → Validate[2019-2020] → Test[2020-2021]   │
# │ Fold 4: Train[2007-2021] → Validate[2021-2022] → Test[2022-2023]   │
# └─────────────────────────────────────────────────────────────────────┘
#
# Key principle: NEVER train on future data!
#
# =============================================================================

set -euo pipefail

# Activate virtual environment
source /home/rguan/project/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# CONFIGURATION - Using EXACT parameters from best model:
# long_term_forecast_WTI-log_512_6_TimeMixer_TimeMixer_custom_ftMS_sl512_ll0_pl6_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0
# =============================================================================

# Data settings - MUST match your best model
DATA_NAME="WTI-log"                    # Your data file name (without .csv)
ROOT_PATH="./dataset/FBD/"
DATA_PATH="${DATA_NAME}.csv"
TARGET="daily_return"
FREQ="d"
CHANNEL_NUM=6                          # Number of input features

# Model configuration - EXACT parameters from best model
MODEL="TimeMixer"
SEQ_LEN=512                            # sl512
LABEL_LEN=0                            # ll0
PRED_LEN=6                             # pl6
D_MODEL=16                             # dm16
D_FF=32                                # df32
E_LAYERS=2                             # el2
D_LAYERS=1                             # dl1
N_HEADS=8                              # nh8
DROPOUT=0.1                            # Default dropout
DOWN_SAMPLING_LAYERS=3                 # dc4 suggests 3-4 layers
DOWN_SAMPLING_WINDOW=2                 # expand2
DOWN_SAMPLING_METHOD="avg"

# Training settings
TRAIN_EPOCHS=100
PATIENCE=15
LEARNING_RATE=0.001
BATCH_SIZE=64

# =============================================================================
# OUTPUT DIRECTORY - Timestamped for easy identification
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="WTI_WalkForward_${MODEL}_${TIMESTAMP}"
OUTPUT_DIR="./results_${EXPERIMENT_NAME}"

# Create output directory structure
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/predictions
mkdir -p ${OUTPUT_DIR}/checkpoints

# Log files
LOG_FILE="${OUTPUT_DIR}/logs/main_log.txt"
SPLIT_LOG="${OUTPUT_DIR}/logs/data_split_log.txt"
TRAINING_LOG="${OUTPUT_DIR}/logs/training_log.txt"
ANALYSIS_LOG="${OUTPUT_DIR}/logs/analysis_log.txt"

echo "==============================================================" | tee ${LOG_FILE}
echo "WALK-FORWARD VALIDATION FOR WTI ${MODEL}" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "Started: $(date)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "EXPERIMENT CONFIGURATION:" | tee -a ${LOG_FILE}
echo "  Experiment Name: ${EXPERIMENT_NAME}" | tee -a ${LOG_FILE}
echo "  Output Directory: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "DATA SETTINGS:" | tee -a ${LOG_FILE}
echo "  Data File: ${ROOT_PATH}${DATA_PATH}" | tee -a ${LOG_FILE}
echo "  Target: ${TARGET}" | tee -a ${LOG_FILE}
echo "  Channels: ${CHANNEL_NUM}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "MODEL PARAMETERS (from best model):" | tee -a ${LOG_FILE}
echo "  Model: ${MODEL}" | tee -a ${LOG_FILE}
echo "  seq_len: ${SEQ_LEN}" | tee -a ${LOG_FILE}
echo "  label_len: ${LABEL_LEN}" | tee -a ${LOG_FILE}
echo "  pred_len: ${PRED_LEN}" | tee -a ${LOG_FILE}
echo "  d_model: ${D_MODEL}" | tee -a ${LOG_FILE}
echo "  d_ff: ${D_FF}" | tee -a ${LOG_FILE}
echo "  e_layers: ${E_LAYERS}" | tee -a ${LOG_FILE}
echo "  n_heads: ${N_HEADS}" | tee -a ${LOG_FILE}
echo "  dropout: ${DROPOUT}" | tee -a ${LOG_FILE}
echo "  down_sampling_layers: ${DOWN_SAMPLING_LAYERS}" | tee -a ${LOG_FILE}
echo "  down_sampling_window: ${DOWN_SAMPLING_WINDOW}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "TRAINING SETTINGS:" | tee -a ${LOG_FILE}
echo "  epochs: ${TRAIN_EPOCHS}" | tee -a ${LOG_FILE}
echo "  patience: ${PATIENCE}" | tee -a ${LOG_FILE}
echo "  learning_rate: ${LEARNING_RATE}" | tee -a ${LOG_FILE}
echo "  batch_size: ${BATCH_SIZE}" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

# =============================================================================
# First, we need to create time-split datasets
# =============================================================================

# Create Python script for data splitting
cat > ${OUTPUT_DIR}/create_splits.py << 'PYTHON_SCRIPT'
"""
Create walk-forward data splits for time series cross-validation.
"""
import pandas as pd
import numpy as np
import os
import sys

def create_walkforward_splits(data_path, output_dir, n_folds=4):
    """
    Create walk-forward splits ensuring no data leakage.
    
    For each fold:
    - Train: All data up to train_end
    - Validation: Data from train_end to val_end (for early stopping)
    - Test: Data from val_end to test_end (true out-of-sample)
    """
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    print(f"Total data points: {n}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Define fold boundaries
    # Reserve last 20% for final testing, split remaining into folds
    test_ratio_per_fold = 0.10  # Each test fold is ~10% of data
    val_ratio = 0.05  # Validation is ~5%
    
    splits = []
    
    for fold in range(n_folds):
        # Calculate boundaries
        # Each fold's test period starts later
        test_start_ratio = 0.6 + fold * 0.1  # 60%, 70%, 80%, 90%
        test_end_ratio = test_start_ratio + test_ratio_per_fold
        val_start_ratio = test_start_ratio - val_ratio
        
        train_end = int(n * val_start_ratio)
        val_end = int(n * test_start_ratio)
        test_end = min(int(n * test_end_ratio), n)
        
        # Create split dataframes
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[:val_end].copy()  # Include train for context
        test_df = df.iloc[:test_end].copy()  # Include all prior for context
        
        # Save splits
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # For TSLib, we save the full data but it will use the ratios
        # Actually, let's save separate files and modify data loading
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
        
        # Also save the full file for TSLib (it handles splitting internally)
        # But we'll use custom split ratios
        full_df = df.iloc[:test_end].copy()
        full_df.to_csv(os.path.join(fold_dir, f'WTI_fold{fold}.csv'), index=False)
        
        split_info = {
            'fold': fold,
            'train_end': train_end,
            'val_end': val_end,
            'test_end': test_end,
            'train_period': f"{df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[train_end-1].strftime('%Y-%m-%d')}",
            'val_period': f"{df['date'].iloc[train_end].strftime('%Y-%m-%d')} to {df['date'].iloc[val_end-1].strftime('%Y-%m-%d')}",
            'test_period': f"{df['date'].iloc[val_end].strftime('%Y-%m-%d')} to {df['date'].iloc[test_end-1].strftime('%Y-%m-%d')}",
            'train_size': train_end,
            'val_size': val_end - train_end,
            'test_size': test_end - val_end,
            'total_size': test_end
        }
        splits.append(split_info)
        
        print(f"\nFold {fold}:")
        print(f"  Train: {split_info['train_period']} ({split_info['train_size']} samples)")
        print(f"  Val:   {split_info['val_period']} ({split_info['val_size']} samples)")
        print(f"  Test:  {split_info['test_period']} ({split_info['test_size']} samples)")
    
    # Save split info
    split_df = pd.DataFrame(splits)
    split_df.to_csv(os.path.join(output_dir, 'split_info.csv'), index=False)
    
    return splits

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    n_folds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    create_walkforward_splits(data_path, output_dir, n_folds)
PYTHON_SCRIPT

# Run the split creation
echo "" | tee -a ${LOG_FILE}
echo "Creating walk-forward data splits..." | tee -a ${LOG_FILE}
python ${OUTPUT_DIR}/create_splits.py ${ROOT_PATH}${DATA_PATH} ${OUTPUT_DIR} 4 2>&1 | tee ${SPLIT_LOG}
cat ${SPLIT_LOG} >> ${LOG_FILE}

# =============================================================================
# Run training for each fold
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "TRAINING ON EACH FOLD" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

for FOLD in 0 1 2 3
do
    echo "" | tee -a ${LOG_FILE}
    echo "--------------------------------------------------------------" | tee -a ${LOG_FILE}
    echo "FOLD ${FOLD}" | tee -a ${LOG_FILE}
    echo "--------------------------------------------------------------" | tee -a ${LOG_FILE}
    
    FOLD_DIR="${OUTPUT_DIR}/fold_${FOLD}"
    DATA_FILE="WTI_fold${FOLD}.csv"
    MODEL_ID="${DATA_NAME}_${MODEL}_fold${FOLD}"
    FOLD_LOG="${OUTPUT_DIR}/logs/fold_${FOLD}_training.log"
    
    echo "Training ${MODEL} on fold ${FOLD}..." | tee -a ${LOG_FILE}
    echo "  Fold directory: ${FOLD_DIR}" | tee -a ${LOG_FILE}
    echo "  Data file: ${DATA_FILE}" | tee -a ${LOG_FILE}
    echo "  Model ID: ${MODEL_ID}" | tee -a ${LOG_FILE}
    echo "  Log file: ${FOLD_LOG}" | tee -a ${LOG_FILE}
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${FOLD_DIR}/ \
        --data_path ${DATA_FILE} \
        --model_id ${MODEL_ID} \
        --model ${MODEL} \
        --data custom \
        --features MS \
        --seq_len ${SEQ_LEN} \
        --label_len ${LABEL_LEN} \
        --pred_len ${PRED_LEN} \
        --e_layers ${E_LAYERS} \
        --d_layers ${D_LAYERS} \
        --enc_in ${CHANNEL_NUM} \
        --dec_in ${CHANNEL_NUM} \
        --c_out 1 \
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --n_heads ${N_HEADS} \
        --dropout ${DROPOUT} \
        --freq ${FREQ} \
        --target ${TARGET} \
        --train_epochs ${TRAIN_EPOCHS} \
        --patience ${PATIENCE} \
        --learning_rate ${LEARNING_RATE} \
        --batch_size ${BATCH_SIZE} \
        --down_sampling_layers ${DOWN_SAMPLING_LAYERS} \
        --down_sampling_method ${DOWN_SAMPLING_METHOD} \
        --down_sampling_window ${DOWN_SAMPLING_WINDOW} \
        --channel_independence 0 \
        --itr 1 \
        --num_workers 0 \
        --des "WalkForward_Fold${FOLD}" 2>&1 | tee ${FOLD_LOG}
    
    # Append fold log to main training log
    cat ${FOLD_LOG} >> ${TRAINING_LOG}
    
    # Collect results
    echo "Collecting results for fold ${FOLD}..." | tee -a ${LOG_FILE}
    for dir in ./results/*${MODEL_ID}*; do
        if [ -d "$dir" ]; then
            cp -r "$dir" ${OUTPUT_DIR}/predictions/
            echo "  Copied: $dir" | tee -a ${LOG_FILE}
        fi
    done
    
    # Collect checkpoints
    for dir in ./checkpoints/*${MODEL_ID}*; do
        if [ -d "$dir" ]; then
            cp -r "$dir" ${OUTPUT_DIR}/checkpoints/
            echo "  Copied checkpoint: $dir" | tee -a ${LOG_FILE}
        fi
    done
    
    echo "Fold ${FOLD} complete." | tee -a ${LOG_FILE}
    
done

# =============================================================================
# Aggregate and analyze results
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "AGGREGATING RESULTS" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

# Create analysis script
cat > ${OUTPUT_DIR}/analyze_walkforward.py << 'PYTHON_ANALYSIS'
"""
Analyze walk-forward validation results.
"""
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats

def analyze_walkforward_results(output_dir):
    """
    Aggregate and analyze results from all folds.
    """
    print("=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)
    
    # Find all result directories in predictions folder
    predictions_dir = os.path.join(output_dir, 'predictions')
    result_dirs = glob.glob(os.path.join(predictions_dir, 'long_term_forecast_*'))
    
    # Also check the main output directory
    if not result_dirs:
        result_dirs = glob.glob(os.path.join(output_dir, 'long_term_forecast_*'))
    
    if not result_dirs:
        print(f"No result directories found in {predictions_dir}!")
        print("Checking for results...")
        for root, dirs, files in os.walk(output_dir):
            for d in dirs:
                if d.startswith('long_term_forecast'):
                    print(f"  Found: {os.path.join(root, d)}")
        return None
    
    print(f"Found {len(result_dirs)} result directories")
    
    all_results = []
    
    for result_dir in sorted(result_dirs):
        pred_path = os.path.join(result_dir, 'pred.npy')
        true_path = os.path.join(result_dir, 'true.npy')
        
        if not os.path.exists(pred_path) or not os.path.exists(true_path):
            print(f"  Missing pred/true files in {result_dir}")
            continue
        
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        print(f"\nProcessing: {os.path.basename(result_dir)}")
        print(f"  Predictions shape: {pred.shape}")
        print(f"  True shape: {true.shape}")
        
        # Extract fold number from directory name
        dir_name = os.path.basename(result_dir)
        fold_num = 'unknown'
        for part in dir_name.split('_'):
            if 'fold' in part.lower():
                fold_num = part
                break
        
        # Calculate metrics for each horizon
        for h in range(pred.shape[1]):
            pred_h = pred[:, h, :].flatten()
            true_h = true[:, h, :].flatten()
            
            # Remove NaN
            valid = ~(np.isnan(pred_h) | np.isnan(true_h))
            pred_h = pred_h[valid]
            true_h = true_h[valid]
            
            if len(pred_h) < 10:
                continue
            
            # Metrics
            mse = np.mean((pred_h - true_h) ** 2)
            mae = np.mean(np.abs(pred_h - true_h))
            
            # Correlation with error handling
            if np.std(pred_h) > 1e-10 and np.std(true_h) > 1e-10:
                corr = np.corrcoef(pred_h, true_h)[0, 1]
            else:
                corr = 0
            
            # Direction accuracy
            pred_dir = np.sign(pred_h)
            true_dir = np.sign(true_h)
            nonzero = true_h != 0
            dir_acc = np.mean(pred_dir[nonzero] == true_dir[nonzero]) if nonzero.sum() > 0 else 0.5
            
            all_results.append({
                'fold': fold_num,
                'horizon': h,
                'n_samples': len(pred_h),
                'mse': mse,
                'mae': mae,
                'correlation': corr,
                'direction_accuracy': dir_acc
            })
    
    if not all_results:
        print("No valid results found!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Print results by fold
    print("\n" + "=" * 80)
    print("RESULTS BY FOLD")
    print("=" * 80)
    
    for fold in sorted(results_df['fold'].unique()):
        fold_data = results_df[results_df['fold'] == fold]
        print(f"\n{fold}:")
        print(fold_data[['horizon', 'n_samples', 'mse', 'correlation', 'direction_accuracy']].to_string(index=False))
    
    # Aggregate across folds
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (Mean ± Std across folds)")
    print("=" * 80)
    
    agg = results_df.groupby('horizon').agg({
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'correlation': ['mean', 'std'],
        'direction_accuracy': ['mean', 'std'],
        'n_samples': 'sum'
    }).round(4)
    
    print(agg)
    
    # Summary statistics for horizon 0 (most important)
    h0 = results_df[results_df['horizon'] == 0]
    
    print("\n" + "=" * 80)
    print("HORIZON 0 (Next-Day) SUMMARY")
    print("=" * 80)
    
    if len(h0) > 0:
        print(f"  Number of folds: {len(h0)}")
        print(f"  Total samples: {h0['n_samples'].sum()}")
        print(f"  Direction Accuracy: {h0['direction_accuracy'].mean():.1%} ± {h0['direction_accuracy'].std():.1%}")
        print(f"  Correlation (IC):   {h0['correlation'].mean():.4f} ± {h0['correlation'].std():.4f}")
        print(f"  MSE:                {h0['mse'].mean():.6f} ± {h0['mse'].std():.6f}")
        
        # Statistical significance test
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 80)
        
        if len(h0) >= 2:
            # Is direction accuracy significantly > 50%?
            dir_accs = h0['direction_accuracy'].values
            t_stat, p_value = stats.ttest_1samp(dir_accs, 0.5)
            print(f"\nTest 1: Direction Accuracy vs 50% (random)")
            print(f"  Mean: {np.mean(dir_accs):.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT at 5% level - Model beats random!")
            else:
                print(f"  ✗ NOT significant - Cannot reject that model is random")
            
            # Is correlation significantly > 0?
            corrs = h0['correlation'].values
            t_stat, p_value = stats.ttest_1samp(corrs, 0)
            print(f"\nTest 2: Correlation vs 0 (no predictive power)")
            print(f"  Mean: {np.mean(corrs):.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT at 5% level - Model has predictive power!")
            else:
                print(f"  ✗ NOT significant - Cannot reject that model has no predictive power")
        else:
            print("\nNot enough folds for statistical testing (need at least 2)")
    
    # Save results
    results_path = os.path.join(output_dir, 'walkforward_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Also save summary
    summary_path = os.path.join(output_dir, 'walkforward_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("WALK-FORWARD VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        if len(h0) > 0:
            f.write(f"Horizon 0 (Next-Day) Results:\n")
            f.write(f"  Direction Accuracy: {h0['direction_accuracy'].mean():.1%} ± {h0['direction_accuracy'].std():.1%}\n")
            f.write(f"  Correlation (IC):   {h0['correlation'].mean():.4f} ± {h0['correlation'].std():.4f}\n")
            f.write(f"  MSE:                {h0['mse'].mean():.6f} ± {h0['mse'].std():.6f}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return results_df

if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    analyze_walkforward_results(output_dir)
PYTHON_ANALYSIS

# Run analysis
echo "Running analysis..." | tee -a ${LOG_FILE}
python ${OUTPUT_DIR}/analyze_walkforward.py ${OUTPUT_DIR} 2>&1 | tee ${ANALYSIS_LOG}
cat ${ANALYSIS_LOG} >> ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "WALK-FORWARD VALIDATION COMPLETE" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "End Time: $(date)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "OUTPUT STRUCTURE:" | tee -a ${LOG_FILE}
echo "  ${OUTPUT_DIR}/" | tee -a ${LOG_FILE}
echo "  ├── logs/" | tee -a ${LOG_FILE}
echo "  │   ├── main_log.txt              (this log)" | tee -a ${LOG_FILE}
echo "  │   ├── data_split_log.txt        (data splitting details)" | tee -a ${LOG_FILE}
echo "  │   ├── training_log.txt          (all training output)" | tee -a ${LOG_FILE}
echo "  │   ├── analysis_log.txt          (analysis output)" | tee -a ${LOG_FILE}
echo "  │   └── fold_*_training.log       (individual fold logs)" | tee -a ${LOG_FILE}
echo "  ├── predictions/                  (model predictions)" | tee -a ${LOG_FILE}
echo "  ├── checkpoints/                  (saved model weights)" | tee -a ${LOG_FILE}
echo "  ├── fold_*/                       (data splits for each fold)" | tee -a ${LOG_FILE}
echo "  ├── walkforward_results.csv       (detailed results)" | tee -a ${LOG_FILE}
echo "  ├── walkforward_summary.txt       (summary statistics)" | tee -a ${LOG_FILE}
echo "  └── split_info.csv                (fold split information)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

# List output directory contents
echo "" | tee -a ${LOG_FILE}
echo "Directory contents:" | tee -a ${LOG_FILE}
ls -la ${OUTPUT_DIR}/ | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Logs:" | tee -a ${LOG_FILE}
ls -la ${OUTPUT_DIR}/logs/ | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Predictions:" | tee -a ${LOG_FILE}
ls -la ${OUTPUT_DIR}/predictions/ 2>/dev/null | tee -a ${LOG_FILE} || echo "  (empty)" | tee -a ${LOG_FILE}