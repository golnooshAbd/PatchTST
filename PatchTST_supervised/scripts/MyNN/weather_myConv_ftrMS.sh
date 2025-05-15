if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# data_version=10_07_2024_16_06_02

for target_node in p T Tpot Tdew rh VPact sh H2OC rho wv max_wv Tlog OT
#maxplanck: p T Tpot Tdew rh VPmax VPact VPdef sh H2OC rho wv max_wv wd
#public: p T Tpot Tdew rh VPmax VPact VPdef sh H2OC wv max_wv wd rain raining SWDR PAR max_PAR Tlog OT
do
    if [ ! -d "./logs/Weather_MyConv" ]; then
        mkdir ./logs/Weather_MyConv
    fi
    
    if [ ! -d "./logs/Weather_MyConv/"$target_node ]; then
        mkdir ./logs/Weather_MyConv/$target_node
    fi
    seq_len=96
    model_name=MyConvLinear

    root_path_name=./dataset/
    data_path_name=weather_noOutlier_limited13.csv
    model_id_name=weather_noOutlier_limited13
    data_name=custom

    features=MS
    enc_in=13

    pred_len=96

    random_seed=2021

    d_model=32
    for pred_len in 96
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name \
        --model $model_name \
        --data $data_name \
        --features $features \
        --target $target_node \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --d_model $d_model \
        --dropout 0.5\
        --kernel_size 5\
        --stride 3\
        --des 'Exp' \
        --learning_rate 0.0001\
        --train_epochs 50\
        --patience 10\
        --gpu 1\
        --itr 1 --batch_size 32 >logs/Weather_MyConv/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
    done
done