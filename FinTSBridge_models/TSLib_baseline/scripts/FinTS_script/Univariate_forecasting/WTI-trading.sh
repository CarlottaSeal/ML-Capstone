#!/usr/bin/env bash
# =============================================================================
# WTI Crude Oil Trading Strategy Experiments
# =============================================================================
#
# This script implements a systematic approach to building trading strategies:
#
# 1. TRADING SIGNAL CONSTRUCTION
#    - Model predictions → trading signals via threshold/percentile methods
#    - Evaluated on direction accuracy, not just MSE
#
# 2. MODEL SELECTION
#    - Compare multiple architectures (simple to complex)
#    - Key insight: Simple models often beat complex ones for noisy financial data
#
# 3. PRACTICAL ISSUES ADDRESSED
#    - Noise: Use regularization (dropout), smaller models
#    - Overfitting: Multiple iterations, early stopping, train/val/test split
#    - Transaction costs: Evaluated in post-analysis (not in training)
#    - Prediction horizon: Test multiple pred_len values
#
# Data assumption:
#   - daily_return = (price_{t+1} - price_t) / price_t (next-day return)
#   - This is the TARGET we want to predict
#
# =============================================================================

set -euo pipefail

# Activate virtual environment
source /home/rguan/project/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# CONFIGURATION - MODIFY THESE BASED ON YOUR DATA
# =============================================================================

DATA_NAME="WTI-log"
ROOT_PATH="./dataset/FBD/"
DATA_PATH="${DATA_NAME}.csv"
TARGET="daily_return"
FREQ="d"

# Number of input features (columns in your CSV excluding 'date' and 'target')
# Typical: date, open, high, low, close, volume, daily_return = 6 features
# Adjust this based on your actual data!
CHANNEL_NUM=6

# Experiment settings
ITERATIONS=3          # Run each config 3 times for statistical significance
TRAIN_EPOCHS=100      # Max epochs (early stopping will kick in)
PATIENCE=15           # Early stopping patience
BATCH_SIZE=32         # Batch size

# =============================================================================
# OUTPUT DIRECTORY - Timestamped for easy identification
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="WTI_trading_${TIMESTAMP}"
RESULTS_DIR="./results_${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints_${EXPERIMENT_NAME}"

# Create directories
mkdir -p ${RESULTS_DIR}
mkdir -p ${CHECKPOINT_DIR}

# Create a log file
LOG_FILE="${RESULTS_DIR}/experiment_log.txt"

# Log experiment info
echo "=============================================================="  | tee ${LOG_FILE}
echo "EXPERIMENT: ${EXPERIMENT_NAME}" | tee -a ${LOG_FILE}
echo "Started: $(date)" | tee -a ${LOG_FILE}
echo "Results will be saved to: ${RESULTS_DIR}" | tee -a ${LOG_FILE}
echo "==============================================================" | tee -a ${LOG_FILE}

echo "=============================================================="
echo "WTI CRUDE OIL TRADING STRATEGY EXPERIMENTS"
echo "=============================================================="
echo "Start Time: $(date)" | tee -a ${LOG_FILE}
echo "Data: ${ROOT_PATH}${DATA_PATH}" | tee -a ${LOG_FILE}
echo "Target: ${TARGET}" | tee -a ${LOG_FILE}
echo "Channels: ${CHANNEL_NUM}" | tee -a ${LOG_FILE}
echo "Iterations per config: ${ITERATIONS}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "OUTPUT DIRECTORIES:" | tee -a ${LOG_FILE}
echo "  Results: ${RESULTS_DIR}" | tee -a ${LOG_FILE}
echo "  Checkpoints: ${CHECKPOINT_DIR}" | tee -a ${LOG_FILE}
echo "  Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "=============================================================="


# =============================================================================
# EXPERIMENT 1: BASELINE & MODEL ARCHITECTURE COMPARISON
# =============================================================================
# Purpose: Find the best model architecture for WTI prediction
# 
# Models tested (from simple to complex):
# - Naive: Just repeats last value (if we can't beat this, no point trading)
# - DLinear: Linear model (surprisingly strong baseline for finance)
# - PatchTST: Patch-based Transformer
# - iTransformer: Inverted Transformer (good for cross-variate learning)
# - TimeMixer: Multi-scale temporal mixing
#
# Key insight: For noisy financial data, simpler models often work better
# =============================================================================

echo ""
echo "=============================================================="
echo "EXPERIMENT 1: Model Architecture Comparison"
echo "=============================================================="
echo "Testing 5 model architectures with pred_len=5 (1-week horizon)"
echo ""

# Fixed parameters for fair comparison
SEQ_LEN=512           # ~2 years of trading history
PRED_LEN=5            # Predict 5 days ahead (1 trading week)
LABEL_LEN=48          # Label length for decoder models

# --- 1.1 Naive Baseline ---
# CRITICAL: If your model can't beat Naive, it has NO predictive value
echo "[1/5] Naive Baseline (must beat this!)..."
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${DATA_NAME}_Naive_pl${PRED_LEN} \
    --model Naive \
    --data custom \
    --features MS \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${PRED_LEN} \
    --enc_in ${CHANNEL_NUM} \
    --dec_in ${CHANNEL_NUM} \
    --c_out 1 \
    --freq ${FREQ} \
    --target ${TARGET} \
    --train_epochs 1 \
    --itr ${ITERATIONS} \
    --num_workers 0 \
    --des 'ModelComparison'

# --- 1.2 DLinear ---
# Simple linear model - often surprisingly effective for financial data
# Less prone to overfitting than complex models
echo "[2/5] DLinear (simple but strong)..."
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${DATA_NAME}_DLinear_pl${PRED_LEN} \
    --model DLinear \
    --data custom \
    --features MS \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${PRED_LEN} \
    --enc_in ${CHANNEL_NUM} \
    --dec_in ${CHANNEL_NUM} \
    --c_out 1 \
    --freq ${FREQ} \
    --target ${TARGET} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --learning_rate 0.001 \
    --batch_size ${BATCH_SIZE} \
    --itr ${ITERATIONS} \
    --num_workers 0 \
    --des 'ModelComparison'

# --- 1.3 PatchTST ---
# Patch-based Transformer - good at capturing local patterns
# Using small model size to prevent overfitting
echo "[3/5] PatchTST (patch-based attention)..."
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${DATA_NAME}_PatchTST_pl${PRED_LEN} \
    --model PatchTST \
    --data custom \
    --features MS \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${PRED_LEN} \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in ${CHANNEL_NUM} \
    --dec_in ${CHANNEL_NUM} \
    --c_out 1 \
    --d_model 64 \
    --d_ff 128 \
    --n_heads 4 \
    --dropout 0.3 \
    --freq ${FREQ} \
    --target ${TARGET} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --learning_rate 0.0005 \
    --batch_size ${BATCH_SIZE} \
    --itr ${ITERATIONS} \
    --num_workers 0 \
    --des 'ModelComparison'

# --- 1.4 iTransformer ---
# Inverted Transformer - explicitly models cross-variate dependencies
# Good when features interact with each other
echo "[4/5] iTransformer (cross-variate attention)..."
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${DATA_NAME}_iTransformer_pl${PRED_LEN} \
    --model iTransformer \
    --data custom \
    --features MS \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${PRED_LEN} \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in ${CHANNEL_NUM} \
    --dec_in ${CHANNEL_NUM} \
    --c_out 1 \
    --d_model 64 \
    --d_ff 64 \
    --n_heads 4 \
    --dropout 0.3 \
    --freq ${FREQ} \
    --target ${TARGET} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --learning_rate 0.0005 \
    --batch_size ${BATCH_SIZE} \
    --itr ${ITERATIONS} \
    --num_workers 0 \
    --des 'ModelComparison'

# --- 1.5 TimeMixer ---
# Multi-scale temporal mixing - captures patterns at different time scales
echo "[5/5] TimeMixer (multi-scale mixing)..."
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model_id ${DATA_NAME}_TimeMixer_pl${PRED_LEN} \
    --model TimeMixer \
    --data custom \
    --features MS \
    --seq_len ${SEQ_LEN} \
    --label_len 0 \
    --pred_len ${PRED_LEN} \
    --e_layers 2 \
    --enc_in ${CHANNEL_NUM} \
    --dec_in ${CHANNEL_NUM} \
    --c_out 1 \
    --d_model 32 \
    --d_ff 64 \
    --dropout 0.3 \
    --freq ${FREQ} \
    --target ${TARGET} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --learning_rate 0.001 \
    --batch_size 64 \
    --down_sampling_layers 3 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --channel_independence 0 \
    --itr ${ITERATIONS} \
    --num_workers 0 \
    --des 'ModelComparison'


# =============================================================================
# EXPERIMENT 2: PREDICTION HORIZON ANALYSIS
# =============================================================================
# Purpose: Find the optimal forecast horizon for trading
#
# Trade-off:
# - Shorter horizon: Easier to predict, but more trading = more costs
# - Longer horizon: Harder to predict, but less trading = lower costs
#
# We test: 1, 3, 5, 10, 21 days (1 day to 1 month)
# =============================================================================

echo ""
echo "=============================================================="
echo "EXPERIMENT 2: Prediction Horizon Analysis"
echo "=============================================================="
echo "Testing different prediction horizons with DLinear"
echo ""

# Use DLinear for horizon analysis (stable, fast)
for PRED_LEN in 1 3 5 10 21
do
    echo "[Horizon ${PRED_LEN}d] Running DLinear..."
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --model_id ${DATA_NAME}_DLinear_horizon${PRED_LEN} \
        --model DLinear \
        --data custom \
        --features MS \
        --seq_len ${SEQ_LEN} \
        --label_len ${LABEL_LEN} \
        --pred_len ${PRED_LEN} \
        --enc_in ${CHANNEL_NUM} \
        --dec_in ${CHANNEL_NUM} \
        --c_out 1 \
        --freq ${FREQ} \
        --target ${TARGET} \
        --train_epochs ${TRAIN_EPOCHS} \
        --patience ${PATIENCE} \
        --learning_rate 0.001 \
        --batch_size ${BATCH_SIZE} \
        --itr ${ITERATIONS} \
        --num_workers 0 \
        --des "Horizon${PRED_LEN}"
done


# =============================================================================
# EXPERIMENT 3: HYPERPARAMETER TUNING
# =============================================================================
# Purpose: Find optimal hyperparameters for best model
#
# Key parameters for financial data:
# - Learning rate: Controls training stability and convergence
# - Dropout: Critical for preventing overfitting on noisy data
# - Sequence length: How much history to consider
# - Model capacity (d_model): Larger isn't always better for noisy data
# =============================================================================

echo ""
echo "=============================================================="
echo "EXPERIMENT 3: Hyperparameter Tuning"
echo "=============================================================="

PRED_LEN=5  # Fixed horizon for hyperparameter comparison

# --- 3.1 Learning Rate Tuning ---
echo ""
echo "[HP-1] Learning Rate Search..."
for LR in 0.0001 0.0005 0.001 0.005
do
    echo "  Testing LR=${LR}..."
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --model_id ${DATA_NAME}_HP_lr${LR} \
        --model DLinear \
        --data custom \
        --features MS \
        --seq_len ${SEQ_LEN} \
        --label_len ${LABEL_LEN} \
        --pred_len ${PRED_LEN} \
        --enc_in ${CHANNEL_NUM} \
        --dec_in ${CHANNEL_NUM} \
        --c_out 1 \
        --freq ${FREQ} \
        --target ${TARGET} \
        --train_epochs ${TRAIN_EPOCHS} \
        --patience ${PATIENCE} \
        --learning_rate ${LR} \
        --batch_size ${BATCH_SIZE} \
        --itr ${ITERATIONS} \
        --num_workers 0 \
        --des "HP_LR"
done

# --- 3.2 Dropout Tuning (for Transformer models) ---
echo ""
echo "[HP-2] Dropout Search (PatchTST)..."
for DROPOUT in 0.1 0.2 0.3 0.4 0.5
do
    echo "  Testing Dropout=${DROPOUT}..."
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --model_id ${DATA_NAME}_HP_dropout${DROPOUT} \
        --model PatchTST \
        --data custom \
        --features MS \
        --seq_len ${SEQ_LEN} \
        --label_len ${LABEL_LEN} \
        --pred_len ${PRED_LEN} \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in ${CHANNEL_NUM} \
        --dec_in ${CHANNEL_NUM} \
        --c_out 1 \
        --d_model 64 \
        --d_ff 128 \
        --n_heads 4 \
        --dropout ${DROPOUT} \
        --freq ${FREQ} \
        --target ${TARGET} \
        --train_epochs ${TRAIN_EPOCHS} \
        --patience ${PATIENCE} \
        --learning_rate 0.0005 \
        --batch_size ${BATCH_SIZE} \
        --itr ${ITERATIONS} \
        --num_workers 0 \
        --des "HP_Dropout"
done

# --- 3.3 Sequence Length Tuning ---
echo ""
echo "[HP-3] Sequence Length Search..."
for SL in 128 256 512
do
    echo "  Testing SeqLen=${SL}..."
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --model_id ${DATA_NAME}_HP_seqlen${SL} \
        --model DLinear \
        --data custom \
        --features MS \
        --seq_len ${SL} \
        --label_len ${LABEL_LEN} \
        --pred_len ${PRED_LEN} \
        --enc_in ${CHANNEL_NUM} \
        --dec_in ${CHANNEL_NUM} \
        --c_out 1 \
        --freq ${FREQ} \
        --target ${TARGET} \
        --train_epochs ${TRAIN_EPOCHS} \
        --patience ${PATIENCE} \
        --learning_rate 0.001 \
        --batch_size ${BATCH_SIZE} \
        --itr ${ITERATIONS} \
        --num_workers 0 \
        --des "HP_SeqLen"
done

# --- 3.4 Model Capacity Tuning (for Transformer) ---
echo ""
echo "[HP-4] Model Capacity Search (d_model)..."
for D_MODEL in 32 64 128
do
    echo "  Testing d_model=${D_MODEL}..."
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ${ROOT_PATH} \
        --data_path ${DATA_PATH} \
        --model_id ${DATA_NAME}_HP_dmodel${D_MODEL} \
        --model PatchTST \
        --data custom \
        --features MS \
        --seq_len ${SEQ_LEN} \
        --label_len ${LABEL_LEN} \
        --pred_len ${PRED_LEN} \
        --e_layers 2 \
        --d_layers 1 \
        --enc_in ${CHANNEL_NUM} \
        --dec_in ${CHANNEL_NUM} \
        --c_out 1 \
        --d_model ${D_MODEL} \
        --d_ff $((D_MODEL * 2)) \
        --n_heads 4 \
        --dropout 0.3 \
        --freq ${FREQ} \
        --target ${TARGET} \
        --train_epochs ${TRAIN_EPOCHS} \
        --patience ${PATIENCE} \
        --learning_rate 0.0005 \
        --batch_size ${BATCH_SIZE} \
        --itr ${ITERATIONS} \
        --num_workers 0 \
        --des "HP_DModel"
done


# =============================================================================
# SUMMARY & COLLECT RESULTS
# =============================================================================

echo ""
echo "=============================================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================================="
echo "End Time: $(date)" | tee -a ${LOG_FILE}

# Copy all results to the timestamped directory
echo "" | tee -a ${LOG_FILE}
echo "Collecting results to ${RESULTS_DIR}..." | tee -a ${LOG_FILE}

# Copy results (pred.npy, true.npy, metrics.npy)
if [ -d "./results" ]; then
    cp -r ./results/* ${RESULTS_DIR}/ 2>/dev/null || true
fi

# Copy any result txt files
cp result_long_term_forecast*.txt ${RESULTS_DIR}/ 2>/dev/null || true

# List what was collected
echo "" | tee -a ${LOG_FILE}
echo "Files collected:" | tee -a ${LOG_FILE}
ls -la ${RESULTS_DIR}/ | tee -a ${LOG_FILE}

echo ""
echo "=============================================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================================="
echo ""
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo ""
echo "All results saved to: ${RESULTS_DIR}/"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Experiments completed:"
echo "  1. Model Comparison: Naive, DLinear, PatchTST, iTransformer, TimeMixer"
echo "  2. Horizon Analysis: pred_len = 1, 3, 5, 10, 21 days"
echo "  3. Hyperparameter Tuning:"
echo "     - Learning Rate: 0.0001, 0.0005, 0.001, 0.005"
echo "     - Dropout: 0.1, 0.2, 0.3, 0.4, 0.5"
echo "     - Seq Length: 128, 256, 512"
echo "     - Model Capacity: d_model = 32, 64, 128"
echo ""
echo "Next step: Run the analysis script"
echo "  python WTI_trading_analysis.py --results_dir ${RESULTS_DIR}"
echo ""
echo "==============================================================" | tee -a ${LOG_FILE}