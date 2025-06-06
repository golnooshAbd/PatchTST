if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

data_version=10_07_2024_16_06_02

for target_node in  601 602 603 603A_SBW 604 605 606 607_SBW 608 609 6a 6b 7 7a 8 801_SBW 801b 801c 801d 801e 801f 801g 801h 801i 802 803_SBW_1 804 805_SBW 805a 805b 805c 805d 805e 805f 805g 805h 805i 805j 805k 8a 9 KNO4983 L43 L45 Node_1 Node_12 P366 RA_40a_SBW RK59_MW R�B_128 SK_102_Steuerschacht SK102_SBW VS_22 ARA_Fehraltorf R�B+PW80_Industrie
# rain_gauge 101 102a 102b 102c 102d 102e 102f 102g 102i 102k 104a 104b 104c 104d 104e 105 106 1102 1103 1104 1105 1106 112 113 114 115 115a 116 11d 11e 11f 11g 11h 12 121 122 123 123a 124 124a 125_SBW 126_SBW 127 127.2 128a_Blindsch. 12a 12b 13 134a_SW 134b_ 134c_SW 134SW 135MW 136MW 137 138_SBW 138a_Mess-Sch. 13a 13d 13e 14 142aSW 142SW 144 145 146 147 147a 148 149 14a 14b 15 150 150a 150b 150c 153 154_SBW 155SW 156a 156SW 157_SBW 157a_SBW 159c 159d 159MW 15a 160MW 161 162 163 164 165 166 167_SBW 17 17b 18 19 206 206a 206b 207 207a 207b 207c 208 209 210 211 22a 23 26_SBW 26a 27 27a 27b 28 284 284a 284b 285 286 287 28a 28b 28c 28d 33a 33b 34 35a 35b 36 36a 37 374 375 376 376a 377 377a 377b 377C 378 379 37a 38 381 381b 382 382a 382b 384 385 386 387 388 389 38a 38b 39 39a 40 40b 40d 41 41a 41b 42 42a 446 446a 446b 446c 447 447d 448 448a 448b 448c 448d 45 45a 47 47a 48a_SBW 48c_SBW 50_SBW 508_SW 509_SW 510_SW 52_SW 523_SW 525_SW 526_SW 526a_SW 531_SW 532_SW 533_SW 534_SW 565 566 567 56a 571 572 573 574 575 576 576a 578 579 579a 579b 579c 579d 58_SBW 58_Spezialsch._ 581_SBW 582_SBW 583_SBW 584_SBW 585_SBW 591 591a 591e 591f 591g 591h 591i 591j 591m 591n 591o 591p 591q 592 593 594 596 596__Spez.sch. 596a 596d 596e 597.2_SBW 597_SBW 597a 597b 597c 597d 597e 597f 597g 597h 597i 597j 597k 597m 597n 598 5fSWRW 5g_SBW 5h 6_SBW 600_SBW 601 602 603 603A_SBW 604 605 606 607_SBW 608 609 6a 6b 7 7a 8 801_SBW 801b 801c 801d 801e 801f 801g 801h 801i 802 803_SBW_1 804 805_SBW 805a 805b 805c 805d 805e 805f 805g 805h 805i 805j 805k 8a 9 KNO4983 L43 L45 Node_1 Node_12 P366 RA_40a_SBW RK59_MW R�B_128 SK_102_Steuerschacht SK102_SBW VS_22 ARA_Fehraltorf R�B+PW80_Industrie
do
    if [ ! -d "./logs/LongForecasting_"$data_version ]; then
        mkdir ./logs/LongForecasting_$data_version
    fi
    
    if [ ! -d "./logs/LongForecasting_"$data_version"/"$target_node ]; then
        mkdir ./logs/LongForecasting_"$data_version"/$target_node
    fi
    seq_len=96
    model_name=PatchTST

    root_path_name=./dataset/
    data_path_name=waterG_allNodes.csv
    model_id_name=waterG_allNodes
    data_name=custom

    features=S
    enc_in=1

    random_seed=2021
    for pred_len in 96 192 336 720
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
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --patience 20\
        --itr 1 --batch_size 32 --learning_rate 0.0001 --gpu 1 > logs/LongForecasting_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features.log 
    done
done
#original name of waterG_allNodes is /single_file_simtime_05_07_2024_09_25_01.csv, generated by /home/abgo/Desktop/Codes/UTWIN_proj, SWMM, physical model from 1 year(s) rain dataset for all nodes (node name '101' changed to 'OT')
#simulation_start_date="2015/08/19 23:50",simulation_end_date="2016/08/19 23:50"