export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer





seq_len=512
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10

for data in OPTION BTCF
do
for pred_len in 5 21 63 126
do


if [ ${data} = GSMI ]; then
    target=volume.SZSE
    channel_num=100
elif [ ${data} = OPTION ]; then
    target=t
    channel_num=22
elif [ ${data} = BTCF ]; then
    target=taker_buy_volume_spot
    channel_num=12
fi



python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/FBD/ \
    --data_path ${data}.csv \
    --model_id ${data}_512_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_len} \
    --e_layers $e_layers \
    --enc_in ${channel_num} \
    --c_out ${channel_num} \
    --des 'Exp' \
    --target ${target} \
    --itr 5 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 128 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window

done
done
