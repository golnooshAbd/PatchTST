if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# for target_node in node_47a_output

target_node=node_47a_output

# for data_version in 1_5_0 1_6_0 1_7_0
# do
#     if [ ! -d "./logs/synthesized_"$data_version ]; then
#         mkdir ./logs/synthesized_$data_version
#     fi
    
#     if [ ! -d "./logs/synthesized_"$data_version"/"$target_node ]; then
#         mkdir ./logs/synthesized_"$data_version"/$target_node
#     fi
#     seq_len=96
#     model_name=MyTransformer

#     root_path_name=/home/abgo/Data/synthesized/
#     data_path_name=synthesized_v"$data_version".csv
#     model_id_name=synthesized_v"$data_version"
#     data_name=custom

#     features=MS
#     enc_in=440
#     # exo=False
#     # exo_future=False

#     pred_len=96

#     random_seed=2021

#     e_layers=3
#     n_heads=2
#     d_model=32
#     d_ff=128

#     for pred_len in 96
#     do
#         python -u run_longExp.py \
#         --random_seed $random_seed \
#         --is_training 0 \
#         --root_path $root_path_name \
#         --data_path $data_path_name \
#         --model_id $model_id_name'_'$seq_len'_'$pred_len \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --target $target_node \
#         --seq_len $seq_len \
#         --pred_len $pred_len \
#         --label_len 0 \
#         --enc_in $enc_in \
#         --no-exo \
#         --no-exo_future \
#         --e_layers $e_layers \
#         --n_heads $n_heads \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --dropout 0.5\
#         --stride 2\
#         --des 'Exp' \
#         --lradj 'constant'\
#         --learning_rate 0.0001\
#         --train_epochs 100\
#         --patience 10\
#         --gpu 0\
#         --itr 1 --batch_size 32 >logs/synthesized_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
#     done
# done

# for data_version in 1_5_1 1_6_1 1_7_1
# do
#     if [ ! -d "./logs/synthesized_"$data_version ]; then
#         mkdir ./logs/synthesized_$data_version
#     fi
    
#     if [ ! -d "./logs/synthesized_"$data_version"/"$target_node ]; then
#         mkdir ./logs/synthesized_"$data_version"/$target_node
#     fi
#     seq_len=96
#     model_name=MyTransformer

#     root_path_name=/home/abgo/Data/synthesized/
#     data_path_name=synthesized_v"$data_version".csv
#     model_id_name=synthesized_v"$data_version"
#     data_name=custom

#     features=MS
#     enc_in=441
#     # exo=True
#     # exo_future=False

#     pred_len=96

#     random_seed=2021

#     e_layers=3
#     n_heads=2
#     d_model=32
#     d_ff=128

#     for pred_len in 96
#     do
#         python -u run_longExp.py \
#         --random_seed $random_seed \
#         --is_training 0 \
#         --root_path $root_path_name \
#         --data_path $data_path_name \
#         --model_id $model_id_name'_'$seq_len'_'$pred_len \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --target $target_node \
#         --seq_len $seq_len \
#         --pred_len $pred_len \
#         --label_len 0 \
#         --enc_in $enc_in \
#         --exo \
#         --no-exo_future \
#         --e_layers $e_layers \
#         --n_heads $n_heads \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --dropout 0.5\
#         --stride 2\
#         --des 'Exp' \
#         --lradj 'constant'\
#         --learning_rate 0.0001\
#         --train_epochs 100\
#         --patience 10\
#         --gpu 0\
#         --itr 1 --batch_size 32 >logs/synthesized_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
#     done
# done

for data_version in 1_5_2 1_6_2 1_7_2
do
    if [ ! -d "./logs/synthesized_"$data_version ]; then
        mkdir ./logs/synthesized_$data_version
    fi
    
    if [ ! -d "./logs/synthesized_"$data_version"/"$target_node ]; then
        mkdir ./logs/synthesized_"$data_version"/$target_node
    fi
    seq_len=96
    model_name=MyTransformer

    root_path_name=/home/abgo/Data/synthesized/
    data_path_name=synthesized_v"$data_version".csv
    model_id_name=synthesized_v"$data_version"
    data_name=custom

    features=MS
    enc_in=442
    exo=True
    exo_future=Ture

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
        --label_len 0 \
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
        --itr 1 --batch_size 32 >logs/synthesized_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
    done
done