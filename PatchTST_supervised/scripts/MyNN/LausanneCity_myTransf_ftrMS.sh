if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

data_version=1_3_3

for target_node in Hydrique_31_002_hauteur #Hydrique_hydrique-Vuachere_Vallon_hauteur STEP_DВbitcanalEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09 STEP_DВbitcanalOUEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09
do
    if [ ! -d "./logs/LausanneCity_"$data_version ]; then
        mkdir ./logs/LausanneCity_$data_version
    fi
    
    if [ ! -d "./logs/LausanneCity_"$data_version"/"$target_node ]; then
        mkdir ./logs/LausanneCity_"$data_version"/$target_node
    fi
    seq_len=96
    model_name=MyTransformer

    root_path_name=/home/abgo/Data/LausanneCity/
    data_path_name=LausanneCity_v"$data_version".csv
    model_id_name=LausanneCity_v"$data_version"
    data_name=custom

    features=MS
    enc_in=6

    pred_len=96

    random_seed=2021

    e_layers=3
    n_heads=2
    d_model=32
    d_ff=128
    for pred_len in 96
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 0 \
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
        --exo \
        --exo_future \
        --e_layers $e_layers \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout 0.5\
        --stride 2\
        --des 'Exp' \
        --lradj 'constant'\
        --learning_rate 0.0001\
        --train_epochs 100\
        --patience 10\
        --gpu 0\
        --itr 1 --batch_size 32 >logs/LausanneCity_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
    done
done

data_version=1_3_1

for target_node in Hydrique_31_002_hauteur Hydrique_hydrique-Vuachere_Vallon_hauteur STEP_DВbitcanalEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09 STEP_DВbitcanalOUEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09
do
    if [ ! -d "./logs/LausanneCity_"$data_version ]; then
        mkdir ./logs/LausanneCity_$data_version
    fi
    
    if [ ! -d "./logs/LausanneCity_"$data_version"/"$target_node ]; then
        mkdir ./logs/LausanneCity_"$data_version"/$target_node
    fi
    seq_len=96
    model_name=MyTransformer

    root_path_name=/home/abgo/Data/LausanneCity/
    data_path_name=LausanneCity_v"$data_version".csv
    model_id_name=LausanneCity_v"$data_version"
    data_name=custom

    features=MS
    enc_in=5

    pred_len=96

    random_seed=2021

    e_layers=3
    n_heads=2
    d_model=32
    d_ff=128
    for pred_len in 96
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 0 \
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
        --exo \
        --no-exo_future \
        --e_layers $e_layers \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout 0.5\
        --stride 2\
        --des 'Exp' \
        --lradj 'constant'\
        --learning_rate 0.0001\
        --train_epochs 100\
        --patience 10\
        --gpu 0\
        --itr 1 --batch_size 32 >logs/LausanneCity_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
    done
done

data_version=1_2

for target_node in Hydrique_31_002_hauteur Hydrique_hydrique-Vuachere_Vallon_hauteur STEP_DВbitcanalEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09 STEP_DВbitcanalOUEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09
do
    if [ ! -d "./logs/LausanneCity_"$data_version ]; then
        mkdir ./logs/LausanneCity_$data_version
    fi
    
    if [ ! -d "./logs/LausanneCity_"$data_version"/"$target_node ]; then
        mkdir ./logs/LausanneCity_"$data_version"/$target_node
    fi
    seq_len=96
    model_name=MyTransformer

    root_path_name=/home/abgo/Data/LausanneCity/
    data_path_name=LausanneCity_v"$data_version".csv
    model_id_name=LausanneCity_v"$data_version"
    data_name=custom

    features=MS
    enc_in=4

    pred_len=96

    random_seed=2021

    e_layers=3
    n_heads=2
    d_model=32
    d_ff=128
    for pred_len in 96
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 0 \
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
        --no-exo \
        --no-exo_future \
        --e_layers $e_layers \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout 0.5\
        --stride 2\
        --des 'Exp' \
        --lradj 'constant'\
        --learning_rate 0.0001\
        --train_epochs 100\
        --patience 10\
        --gpu 0\
        --itr 1 --batch_size 32 >logs/LausanneCity_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
    done
done