if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# data_version=10_07_2024_16_06_02

for target_node in L_CCT_Amont_VuLink_P_Temperature L_BAM_VuLink_P_Temperature L_4:_CCTAmont_P_Temperature L_1:_Amont_P_Temperature L_Aval_VuLink_P_Temperature L_2:_BAM_P_Temperature L_6:_Aval_P_Temperature L_5:_CCTAval_P_Temperature L_4:_CCTAmont_P_Actual_Conductivity L_1:_Amont_P_Actual_Conductivity L_2:_BAM_P_Actual_Conductivity L_6:_Aval_P_Actual_Conductivity L_5:_CCTAval_P_Actual_Conductivity L_4:_CCTAmont_P_Specific_Conductivity L_1:_Amont_P_Specific_Conductivity L_2:_BAM_P_Specific_Conductivity L_6:_Aval_P_Specific_Conductivity L_5:_CCTAval_P_Specific_Conductivity L_1:_Amont_P_Salinity L_2:_BAM_P_Salinity L_6:_Aval_P_Salinity L_5:_CCTAval_P_Salinity L_4:_CCTAmont_P_Total_Dissolved_Solids L_5:_CCTAval_P_Total_Dissolved_Solids L_4:_CCTAmont_P_Density L_1:_Amont_P_Density L_2:_BAM_P_Density
do
    if [ ! -d "./logs/hydrovu_MyTransf" ]; then
        mkdir ./logs/hydrovu_MyTransf
    fi
    
    if [ ! -d "./logs/hydrovu_MyTransf/"$target_node ]; then
        mkdir ./logs/hydrovu_MyTransf/$target_node
    fi
    seq_len=96
    model_name=MyTransformer

    root_path_name=./dataset/
    data_path_name=hydrovu_v1_0.csv
    model_id_name=hydrovu_v1_0
    data_name=custom

    features=MS
    enc_in=27

    pred_len=96

    random_seed=2021

    e_layers=6
    n_heads=1
    d_model=8
    d_ff=8
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
        --gpu 1\
        --itr 1 --batch_size 16 >logs/hydrovu_MyTransf/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model.log
    done
done