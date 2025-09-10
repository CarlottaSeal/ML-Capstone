

python -u run.py --is_training 1 --model_id GSMI --model PSformer --root_path ./dataset/FBD/ --data_path GSMI.csv --data custom --target close.SPX --features M --pos_emb 0 --seq_len 512 --pred_len 5 --itr 5 --train_epochs 300 --batch_size 16 --patience 30 --learning_rate 2e-4 --lradj constant --dropout 0.1 --num_encoder 1 --num_seg 16 --rho 0.0 --norm_window 512

python -u run.py --is_training 1 --model_id GSMI --model PSformer --root_path ./dataset/FBD/ --data_path GSMI.csv --data custom --target close.SPX --features M --pos_emb 0 --seq_len 512 --pred_len 21 --itr 5 --train_epochs 300 --batch_size 16 --patience 30 --learning_rate 2e-4 --lradj constant --dropout 0.1 --num_encoder 1 --num_seg 16 --rho 0.4 --norm_window 512

python -u run.py --is_training 1 --model_id GSMI --model PSformer --root_path ./dataset/FBD/ --data_path GSMI.csv --data custom --target close.SPX --features M --pos_emb 0 --seq_len 512 --pred_len 63 --itr 5 --train_epochs 300 --batch_size 16 --patience 30 --learning_rate 2e-4 --lradj constant --dropout 0.1 --num_encoder 1 --num_seg 16 --rho 0.3 --norm_window 512

python -u run.py --is_training 1 --model_id GSMI --model PSformer --root_path ./dataset/FBD/ --data_path GSMI.csv --data custom --target close.SPX --features M --pos_emb 0 --seq_len 512 --pred_len 126 --itr 5 --train_epochs 300 --batch_size 16 --patience 30 --learning_rate 2e-4 --lradj constant --dropout 0.1 --num_encoder 1 --num_seg 16 --rho 0.2 --norm_window 512
