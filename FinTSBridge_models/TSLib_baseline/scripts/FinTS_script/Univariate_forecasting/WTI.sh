#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
source /home/rguan/project/venv/bin/activate

# 控制 CPU 线程，避免进程风暴
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# =========================
# Dataset & common configs
# =========================
data=WTI
pred_len=6
target=daily_return

# 说明：由于 Loader 将 target 也当作输入特征，实际输入通道=6
# （open/high/low/volume/close + daily_return）
# 为了先跑通，这里把 enc/dec 统一设为 6；输出 c_out 保持 1（只预测 daily_return）
encdec_ch=6
c_out=1

freq=d
seq_len=512

echo "[INFO] data=${data} pred_len=${pred_len} target=${target} enc/dec=${encdec_ch} c_out=${c_out} freq=${freq}"

# =========================
# TimeMixer  (多变量输入 -> 单变量输出)
# =========================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/FBD/ \
  --data_path ${data}.csv \
  --model_id ${data}_${seq_len}_${pred_len}_TimeMixer \
  --model TimeMixer \
  --data custom \
  --features MS \
  --seq_len ${seq_len} \
  --label_len 0 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --enc_in ${encdec_ch} \
  --dec_in ${encdec_ch} \
  --c_out ${c_out} \
  --freq ${freq} \
  --des 'Exp' \
  --target ${target} \
  --itr 1 \
  --num_workers 0 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10 \
  --batch_size 64 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --channel_independence 0

# =========================
# PatchTST  (多变量输入 -> 单变量输出)
# =========================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/FBD/ \
  --data_path ${data}.csv \
  --model_id ${data}_${seq_len}_${pred_len}_PatchTST \
  --model PatchTST \
  --data custom \
  --features MS \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in ${encdec_ch} \
  --dec_in ${encdec_ch} \
  --c_out ${c_out} \
  --freq ${freq} \
  --des 'Exp' \
  --target ${target} \
  --itr 1 \
  --num_workers 0 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.0005

# =========================
# TiDE  (multivariate inputs -> single target)
# =========================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/FBD/ \
  --data_path ${data}.csv \
  --model_id ${data}_512_${pred_len}_TiDE \
  --model TiDE \
  --data custom \
  --features MS \
  --seq_len 512 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in ${channel_num} \
  --dec_in ${channel_num} \
  --c_out 1 \
  --freq d \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --target ${target} \
  --itr 5 \
  --num_workers 0


# =========================
# PSformer (TSLib version, if registered)
# =========================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/FBD/ \
  --data_path ${data}.csv \
  --model_id ${data}_512_${pred_len}_PSformer \
  --model PSformer \
  --data custom \
  --features MS \
  --seq_len 512 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in ${channel_num} \
  --dec_in ${channel_num} \
  --c_out 1 \
  --freq d \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --target ${target} \
  --itr 5 \
  --num_workers 0
