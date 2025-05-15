if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# data_version=10_07_2024_16_06_02

for target_node in L_CCT_Amont_VuLink_P_Temperature L_BAM_VuLink_P_Temperature L_4:_CCTAmont_P_Temperature L_1:_Amont_P_Temperature L_Aval_VuLink_P_Temperature L_2:_BAM_P_Temperature L_6:_Aval_P_Temperature L_5:_CCTAval_P_Temperature L_4:_CCTAmont_P_Actual_Conductivity L_1:_Amont_P_Actual_Conductivity L_2:_BAM_P_Actual_Conductivity L_6:_Aval_P_Actual_Conductivity L_5:_CCTAval_P_Actual_Conductivity L_4:_CCTAmont_P_Specific_Conductivity L_1:_Amont_P_Specific_Conductivity L_2:_BAM_P_Specific_Conductivity L_6:_Aval_P_Specific_Conductivity L_5:_CCTAval_P_Specific_Conductivity L_1:_Amont_P_Salinity L_2:_BAM_P_Salinity L_6:_Aval_P_Salinity L_5:_CCTAval_P_Salinity L_4:_CCTAmont_P_Total_Dissolved_Solids L_5:_CCTAval_P_Total_Dissolved_Solids L_4:_CCTAmont_P_Density L_1:_Amont_P_Density L_2:_BAM_P_Density

do
    if [ ! -d "./logs/hydrovu_MyConv" ]; then
        mkdir ./logs/hydrovu_MyConv
    fi
    
    if [ ! -d "./logs/hydrovu_MyConv/"$target_node ]; then
        mkdir ./logs/hydrovu_MyConv/$target_node
    fi
    seq_len=96
    model_name=MyConvLinear

    root_path_name=./dataset/
    data_path_name=hydrovu_v1_0.csv
    model_id_name=hydrovu_v1_0
    data_name=custom

    features=S
    enc_in=1

    pred_len=96

    random_seed=2021

    d_model=8
    for pred_len in 96
    do
        for random_seed in 2021 42 2040
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
            --kernel_size 7\
            --stride 3\
            --des 'Exp' \
            --learning_rate 0.0001\
            --train_epochs 100\
            --patience 10\
            --gpu 1\
            --itr 1 --batch_size 16 >logs/hydrovu_MyConv/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model'_rnd'$random_seed.log
        done
    done
done

#  other pred_len
for target_node in L_CCT_Amont_VuLink_P_Temperature L_BAM_VuLink_P_Temperature L_4:_CCTAmont_P_Temperature L_1:_Amont_P_Temperature L_Aval_VuLink_P_Temperature L_2:_BAM_P_Temperature L_6:_Aval_P_Temperature L_5:_CCTAval_P_Temperature L_4:_CCTAmont_P_Actual_Conductivity L_1:_Amont_P_Actual_Conductivity L_2:_BAM_P_Actual_Conductivity L_6:_Aval_P_Actual_Conductivity L_5:_CCTAval_P_Actual_Conductivity L_4:_CCTAmont_P_Specific_Conductivity L_1:_Amont_P_Specific_Conductivity L_2:_BAM_P_Specific_Conductivity L_6:_Aval_P_Specific_Conductivity L_5:_CCTAval_P_Specific_Conductivity L_1:_Amont_P_Salinity L_2:_BAM_P_Salinity L_6:_Aval_P_Salinity L_5:_CCTAval_P_Salinity L_4:_CCTAmont_P_Total_Dissolved_Solids L_5:_CCTAval_P_Total_Dissolved_Solids L_4:_CCTAmont_P_Density L_1:_Amont_P_Density L_2:_BAM_P_Density

do
    if [ ! -d "./logs/hydrovu_MyConv" ]; then
        mkdir ./logs/hydrovu_MyConv
    fi
    
    if [ ! -d "./logs/hydrovu_MyConv/"$target_node ]; then
        mkdir ./logs/hydrovu_MyConv/$target_node
    fi
    seq_len=96
    model_name=MyConvLinear

    root_path_name=./dataset/
    data_path_name=hydrovu_v1_0.csv
    model_id_name=hydrovu_v1_0
    data_name=custom

    features=S
    enc_in=1

    # pred_len=96

    random_seed=2021

    d_model=8
    for pred_len in 48 32
    do
        for random_seed in 2021 42 2040
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
            --kernel_size 7\
            --stride 3\
            --des 'Exp' \
            --learning_rate 0.0001\
            --train_epochs 100\
            --patience 10\
            --gpu 1\
            --itr 1 --batch_size 16 >logs/hydrovu_MyConv/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_dm'$d_model'_rnd'$random_seed.log
        done
    done
done