if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# data_version=10_07_2024_16_06_02

for target_node in wv max_wv wd rain raining SWDR PAR max_PAR Tlog OT
#p T Tpot Tdew rh VPmax VPact VPdef sh H2OC rho wv max_wv wd rain raining SWDR PAR max_PAR Tlog OT

do
    if [ ! -d "./logs/LongForecastingWeather" ]; then
        mkdir ./logs/LongForecastingWeather
    fi
    
    if [ ! -d "./logs/LongForecastingWeather/"$target_node ]; then
        mkdir ./logs/LongForecastingWeather/$target_node
    fi
    seq_len=336
    model_name=PatchTST

    root_path_name=./dataset/
    data_path_name=weather_noOutlier.csv
    model_id_name=weather_noOutlier
    data_name=custom

    features=MS
    enc_in=21

    random_seed=2021
    for pred_len in 96
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features $features \
        --target $target_node \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --e_layers 1 \
        --n_heads 4 \
        --d_model 16 \
        --d_ff 128 \
        --dropout 0.5\
        --fc_dropout 0.5\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --patience 20\
        --gpu 1\
        --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecastingWeather/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr14'$features'all'.log
    done
done