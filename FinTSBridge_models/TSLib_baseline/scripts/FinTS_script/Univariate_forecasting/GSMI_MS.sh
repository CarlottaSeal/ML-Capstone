export CUDA_VISIBLE_DEVICES=0

# model_name=DLinear
data=GSMI
pred_len=5
target=close.SPX
channel_num=100

# DLinear
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model DLinear \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --target ${target} \
    --itr 5

# FEDformer
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model FEDformer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --target ${target} \
    --itr 5


# iTransformer

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model iTransformer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --target ${target} \
    --itr 5


# Koopa
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model Koopa \
    --data custom \
    --features MS \
    --seq_len 512 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --learning_rate 0.001 \
    --target ${target} \
    --itr 5


# Nonstationary_Transformer
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model Nonstationary_Transformer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --p_hidden_dims 256 256 \
    --p_hidden_layers 2 \
    --d_model 128 \
    --target ${target} \
    --itr 5



#TiDE
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/FBD/ \
  --data_path ${data}.csv \
  --model_id ${data}_512_${pred_len} \
  --model TiDE \
  --data custom \
  --features MS \
  --seq_len 512 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 512 \
  --learning_rate 0.01 \
  --target ${target} \
  --itr 5


# TimeMixer
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model TimeMixer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 0 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --enc_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --target ${target} \
    --itr 5 \
    --d_model 16 \
    --d_ff 32 \
    --learning_rate 0.01 \
    --train_epochs 50 \
    --patience 10 \
    --batch_size 128 \
    --down_sampling_layers 3 \
    --down_sampling_method avg \
    --down_sampling_window 2


# model_name=PatchTST
# PatchTST
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/FBD/ \
  --data_path ${data}.csv \
  --model_id ${data}_512_${pred_len} \
  --model PatchTST \
  --data custom \
  --features MS \
  --seq_len 512 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --target ${target} \
  --itr 5 


# Naive
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model Naive \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --train_epochs 1\
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --p_hidden_dims 256 256 \
    --p_hidden_layers 2 \
    --d_model 128 \
    --target ${target} \
    --itr 5


#Autoformer
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model Autoformer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --target ${target} \
    --itr 5


#Crossformer
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model Crossformer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --des 'Exp' \
    --target ${target} \
    --itr 5


# Informer
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model Informer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --target ${target} \
    --itr 5


# TimesNet
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model TimesNet \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --des 'Exp' \
    --target ${target} \
    --itr 5


#Transformer

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model Transformer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --target ${target} \
    --itr 5



# TSMixer
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model TSMixer \
    --data custom \
    --features MS \
    --seq_len 512 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${channel_num} \
    --dec_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --target ${target} \
    --itr 5