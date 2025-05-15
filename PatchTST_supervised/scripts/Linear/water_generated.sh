# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path node_nname_581_SBW_simtime_28_02_2024_09_46_43.csv \
  --model_id waterG_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'WaterG_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path node_nname_581_SBW_simtime_28_02_2024_09_46_43.csv \
  --model_id waterG_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'WaterG_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path node_nname_581_SBW_simtime_28_02_2024_09_46_43.csv \
  --model_id waterG_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'WaterG_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path node_nname_581_SBW_simtime_28_02_2024_09_46_43.csv \
  --model_id waterG_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'WaterG_$seq_len'_'720.log
