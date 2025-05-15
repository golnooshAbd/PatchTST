if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# data_version=10_07_2024_16_06_02

for target_node in OT p T Tpot Tdew rh VPmax VPact VPdef sh H2OC rho wv max_wv wd rain raining SWDR PAR max_PAR Tlog
# wv max_wv wd rain raining SWDR PAR max_PAR Tlog OT
do
    if [ ! -d "./logs/Weather_MyTransformer" ]; then
        mkdir ./logs/Weather_MyTransformer
    fi
    
    if [ ! -d "./logs/Weather_MyTransformer/"$target_node ]; then
        mkdir ./logs/Weather_MyTransformer/$target_node
    fi
    seq_len=96
    model_name=MyTransformer

    root_path_name=./dataset/
    data_path_name=weather_noOutlier.csv
    model_id_name=weather_noOutlier
    data_name=custom

    features=S
    enc_in=1

    pred_len=96

    random_seed=2021
    # for e_layers in 1 2 4 6 8
    for e_layers in 6
    do  
        for n_heads in 1
        do
            for d_model in 16
            do
                for d_ff in 16
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
                    --e_layers $e_layers \
                    --n_heads $n_heads \
                    --d_model $d_model \
                    --d_ff $d_ff \
                    --dropout 0.5\
                    --stride 2\
                    --des 'Exp' \
                    --lradj 'constant'\
                    --learning_rate 0.0001\
                    --train_epochs 50\
                    --patience 50\
                    --gpu 1\
                    --itr 1 --batch_size 32 >logs/Weather_MyTransformer/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_el'$e_layers'_nh'$n_heads'_dm'$d_model'_dff'$d_ff.log
                done
            done
        done
    done
done