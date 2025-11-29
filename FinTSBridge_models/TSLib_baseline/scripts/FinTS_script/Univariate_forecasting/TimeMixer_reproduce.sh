#!/usr/bin/env bash
# =============================================================================
# Reproduce Best TimeMixer Model - Parameter Verification Test
# =============================================================================
#
# Purpose: Verify if the good performance (Sharpe=3.647, Dir.Acc=77.9%) 
#          is due to specific parameters or random chance
#
# Method: Run the EXACT same parameters multiple times (itr=3) 
#         to check consistency
#
# Best model parameters (from filename):
# long_term_forecast_WTI-log_512_6_TimeMixer_TimeMixer_custom_ftMS_sl512_ll0_pl6_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0
#
# =============================================================================

set -euo pipefail

# Activate virtual environment
source /home/rguan/project/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# OUTPUT DIRECTORY WITH TIMESTAMP
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="TimeMixer_Reproduce_${TIMESTAMP}"
OUTPUT_DIR="./results_${EXPERIMENT_NAME}"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs

LOG_FILE="${OUTPUT_DIR}/logs/experiment_log.txt"

# =============================================================================
# EXACT PARAMETERS FROM BEST MODEL
# =============================================================================

# Data settings
DATA_NAME="WTI-log"
ROOT_PATH="./dataset/FBD/"
DATA_PATH="${DATA_NAME}.csv"
TARGET="daily_return"
FREQ="d"
ENC_IN=6           # Number of input features

# Model: TimeMixer
MODEL="TimeMixer"

# EXACT parameters from best model filename:
# sl512_ll0_pl6_dm16_nh8_el2_dl1_df32_expand2_dc4
SEQ_LEN=512        # sl512
LABEL_LEN=0        # ll0
PRED_LEN=6         # pl6
D_MODEL=16         # dm16
N_HEADS=8          # nh8
E_LAYERS=2         # el2
D_LAYERS=1         # dl1
D_FF=32            # df32
DOWN_SAMPLING_LAYERS=3    # dc4 means 4 scales = 3 down_sampling_layers
DOWN_SAMPLING_WINDOW=2    # expand2

# Training settings
TRAIN_EPOCHS=100
PATIENCE=15
BATCH_SIZE=64
LEARNING_RATE=0.001
DROPOUT=0.1

# Number of iterations for statistical significance
ITERATIONS=3

# =============================================================================
# LOGGING
# =============================================================================

echo "==============================================================" | tee ${LOG_FILE}
echo "TIMEMIXER REPRODUCTION EXPERIMENT" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "Started: $(date)" | tee -a ${LOG_FILE}
echo "Output Directory: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "EXACT PARAMETERS FROM BEST MODEL:" | tee -a ${LOG_FILE}
echo "  Data: ${DATA_NAME}" | tee -a ${LOG_FILE}
echo "  seq_len: ${SEQ_LEN}" | tee -a ${LOG_FILE}
echo "  label_len: ${LABEL_LEN}" | tee -a ${LOG_FILE}
echo "  pred_len: ${PRED_LEN}" | tee -a ${LOG_FILE}
echo "  d_model: ${D_MODEL}" | tee -a ${LOG_FILE}
echo "  d_ff: ${D_FF}" | tee -a ${LOG_FILE}
echo "  n_heads: ${N_HEADS}" | tee -a ${LOG_FILE}
echo "  e_layers: ${E_LAYERS}" | tee -a ${LOG_FILE}
echo "  d_layers: ${D_LAYERS}" | tee -a ${LOG_FILE}
echo "  down_sampling_layers: ${DOWN_SAMPLING_LAYERS}" | tee -a ${LOG_FILE}
echo "  down_sampling_window: ${DOWN_SAMPLING_WINDOW}" | tee -a ${LOG_FILE}
echo "  dropout: ${DROPOUT}" | tee -a ${LOG_FILE}
echo "  iterations: ${ITERATIONS}" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

# =============================================================================
# Function to collect results
# =============================================================================
collect_results() {
    local model_id=$1
    echo "Collecting results for ${model_id}..." | tee -a ${LOG_FILE}
    
    for dir in ./results/*${model_id}*; do
        if [ -d "$dir" ]; then
            cp -r "$dir" ${OUTPUT_DIR}/
            echo "  Copied: $dir" | tee -a ${LOG_FILE}
        fi
    done
    
    for dir in ./checkpoints/*${model_id}*; do
        if [ -d "$dir" ]; then
            cp -r "$dir" ${OUTPUT_DIR}/
            echo "  Copied checkpoint: $dir" | tee -a ${LOG_FILE}
        fi
    done
}

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "Starting training with ${ITERATIONS} iterations..." | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

MODEL_ID="${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_${MODEL}_Reproduce"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${MODEL_ID} \
    --model ${MODEL} \
    --data custom \
    --features MS \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${PRED_LEN} \
    --e_layers ${E_LAYERS} \
    --d_layers ${D_LAYERS} \
    --enc_in ${ENC_IN} \
    --dec_in ${ENC_IN} \
    --c_out 1 \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --n_heads ${N_HEADS} \
    --dropout ${DROPOUT} \
    --freq ${FREQ} \
    --target ${TARGET} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --down_sampling_layers ${DOWN_SAMPLING_LAYERS} \
    --down_sampling_method avg \
    --down_sampling_window ${DOWN_SAMPLING_WINDOW} \
    --channel_independence 0 \
    --use_norm 1 \
    --decomp_method moving_avg \
    --moving_avg 25 \
    --itr ${ITERATIONS} \
    --num_workers 0 \
    --des "Reproduce" 2>&1 | tee ${OUTPUT_DIR}/logs/training_log.txt

# Collect results
collect_results ${MODEL_ID}

# =============================================================================
# Also run the "bad" parameters for comparison
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "Running comparison with 'bad' parameters (dm32, df64, pl5)" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

MODEL_ID_BAD="${DATA_NAME}_${SEQ_LEN}_5_${MODEL}_BadParams"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${MODEL_ID_BAD} \
    --model ${MODEL} \
    --data custom \
    --features MS \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len 5 \
    --e_layers ${E_LAYERS} \
    --d_layers ${D_LAYERS} \
    --enc_in ${ENC_IN} \
    --dec_in ${ENC_IN} \
    --c_out 1 \
    --d_model 32 \
    --d_ff 64 \
    --n_heads ${N_HEADS} \
    --dropout ${DROPOUT} \
    --freq ${FREQ} \
    --target ${TARGET} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --down_sampling_layers ${DOWN_SAMPLING_LAYERS} \
    --down_sampling_method avg \
    --down_sampling_window ${DOWN_SAMPLING_WINDOW} \
    --channel_independence 0 \
    --use_norm 1 \
    --decomp_method moving_avg \
    --moving_avg 25 \
    --itr ${ITERATIONS} \
    --num_workers 0 \
    --des "BadParams" 2>&1 | tee ${OUTPUT_DIR}/logs/training_bad_log.txt

# Collect results
collect_results ${MODEL_ID_BAD}

# =============================================================================
# CREATE ANALYSIS SCRIPT
# =============================================================================

cat > ${OUTPUT_DIR}/analyze_results.py << 'PYTHON_SCRIPT'
"""
Analyze reproduction results to check if good performance is reproducible.
"""
import numpy as np
import pandas as pd
import os
import glob
from scipy import stats

def analyze_model_results(results_dir):
    """Analyze all model results in the directory."""
    
    print("=" * 80)
    print("REPRODUCTION EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Find all result directories
    result_dirs = glob.glob(os.path.join(results_dir, 'long_term_forecast_*'))
    
    if not result_dirs:
        print(f"No results found in {results_dir}")
        return
    
    all_results = []
    
    for result_dir in sorted(result_dirs):
        pred_path = os.path.join(result_dir, 'pred.npy')
        true_path = os.path.join(result_dir, 'true.npy')
        
        if not os.path.exists(pred_path):
            continue
            
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        dir_name = os.path.basename(result_dir)
        
        # Extract iteration number
        parts = dir_name.split('_')
        itr = parts[-1] if parts[-1].isdigit() else '0'
        
        # Determine model type
        if 'BadParams' in dir_name:
            model_type = 'Bad (dm32, df64, pl5)'
        elif 'Reproduce' in dir_name:
            model_type = 'Good (dm16, df32, pl6)'
        else:
            model_type = 'Unknown'
        
        # Calculate metrics for horizon 0
        pred_h0 = pred[:, 0, :].flatten()
        true_h0 = true[:, 0, :].flatten()
        
        valid = ~(np.isnan(pred_h0) | np.isnan(true_h0))
        pred_h0 = pred_h0[valid]
        true_h0 = true_h0[valid]
        
        # Metrics
        mse = np.mean((pred_h0 - true_h0) ** 2)
        
        # Direction accuracy
        dir_acc = np.mean(np.sign(pred_h0) == np.sign(true_h0))
        
        # IC (correlation)
        ic = np.corrcoef(pred_h0, true_h0)[0, 1] if np.std(pred_h0) > 1e-10 else 0
        
        # Simple Sharpe (direction-based strategy)
        returns = true_h0
        positions = np.sign(pred_h0)
        strategy_returns = positions * returns
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        all_results.append({
            'model_type': model_type,
            'iteration': itr,
            'n_samples': len(pred_h0),
            'mse': mse,
            'direction_accuracy': dir_acc,
            'ic': ic,
            'sharpe': sharpe,
            'dir_name': dir_name
        })
    
    if not all_results:
        print("No valid results found!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Print results by model type
    print("\n" + "-" * 80)
    print("DETAILED RESULTS BY ITERATION")
    print("-" * 80)
    
    for model_type in df['model_type'].unique():
        print(f"\n{model_type}:")
        subset = df[df['model_type'] == model_type]
        print(subset[['iteration', 'n_samples', 'direction_accuracy', 'ic', 'sharpe']].to_string(index=False))
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    
    summary = df.groupby('model_type').agg({
        'direction_accuracy': ['mean', 'std', 'min', 'max'],
        'ic': ['mean', 'std', 'min', 'max'],
        'sharpe': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(summary)
    
    # Statistical comparison
    print("\n" + "-" * 80)
    print("STATISTICAL TESTS")
    print("-" * 80)
    
    good_results = df[df['model_type'].str.contains('Good')]['direction_accuracy'].values
    bad_results = df[df['model_type'].str.contains('Bad')]['direction_accuracy'].values
    
    if len(good_results) >= 2:
        # Test if "good" params direction accuracy > 50%
        t_stat, p_value = stats.ttest_1samp(good_results, 0.5)
        print(f"\n'Good' params vs 50% (random):")
        print(f"  Mean: {np.mean(good_results):.4f}")
        print(f"  t-stat: {t_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05 and np.mean(good_results) > 0.5:
            print("  ✓ Significantly better than random!")
        else:
            print("  ✗ NOT significantly better than random")
    
    if len(good_results) >= 2 and len(bad_results) >= 2:
        # Compare good vs bad
        t_stat, p_value = stats.ttest_ind(good_results, bad_results)
        print(f"\n'Good' vs 'Bad' params comparison:")
        print(f"  Good mean: {np.mean(good_results):.4f}")
        print(f"  Bad mean: {np.mean(bad_results):.4f}")
        print(f"  t-stat: {t_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  ✓ Significant difference between parameter sets!")
        else:
            print("  ✗ NO significant difference")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    good_mean = np.mean(good_results) if len(good_results) > 0 else 0
    good_std = np.std(good_results) if len(good_results) > 1 else 0
    
    print(f"""
'Good' parameters (dm16, df32, pl6) results:
  Direction Accuracy: {good_mean:.1%} ± {good_std:.1%}

If results are consistent (~77% across all iterations):
  → The good performance is REPRODUCIBLE
  → Parameters matter for this specific test set
  → But still need Walk-Forward to confirm real predictive power

If results vary widely (some ~77%, some ~50%):
  → The good performance is due to RANDOM INITIALIZATION
  → Not reproducible even with same parameters

If all results are ~50%:
  → Original result may have been a FLUKE
  → Or there's some other factor we're missing
""")
    
    # Save results
    df.to_csv(os.path.join(results_dir, 'reproduction_results.csv'), index=False)
    print(f"\nResults saved to: {os.path.join(results_dir, 'reproduction_results.csv')}")

if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    analyze_model_results(results_dir)
PYTHON_SCRIPT

# =============================================================================
# RUN ANALYSIS
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "RUNNING ANALYSIS" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

python ${OUTPUT_DIR}/analyze_results.py ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/logs/analysis_log.txt

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo "" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "EXPERIMENT COMPLETE" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}
echo "End Time: $(date)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Output Directory: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Files:" | tee -a ${LOG_FILE}
ls -la ${OUTPUT_DIR}/ | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "To re-run analysis:" | tee -a ${LOG_FILE}
echo "  python ${OUTPUT_DIR}/analyze_results.py ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}