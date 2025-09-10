export CUDA_VISIBLE_DEVICES=0

model_name=DLinear


for data in GSMI OPTION BTCF
do
    for pred_len in 5 21 63 126
    do


        if [ ${data} = GSMI ]; then
            target=volume.SZSE
        elif [ ${data} = OPTION ]; then
            target=t
        elif [ ${data} = BTCF ]; then
            target=taker_buy_volume_spot
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


    done
done