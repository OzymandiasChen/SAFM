EXP=CL_GEN
source ../environment
source ../orders

cd $CODE_DIR/SAFM

SEED=777
ID=05140200
CUDA_ID="0"
CUDA_VISIBLE_DEVICES=$CUDA_ID bash train_myadaptor.sh \
    --id $ID \
    --cm_lamol_flag --gen_lm_sample_percentage 0.2 \
    --cm_mu 0.11 --mix_ini 0.08 \
    --cm_unshared_adapter_layer_list 4 \
    --z_max_batch_size 8 --cm_test_batch_size 8 \
    --whole_mix_step 6 --warm_mix_step 3 \
    --tasks $nlg_order01 \
    --z_train_epochs 12 12 12 12 12 12 12 12 12 12 12 12 12 12 \
    --z_train_lrs 1.75e-4 1.75e-4 1.75e-4 1.75e-4 1.75e-4 1.75e-4 1.75e-4 1.75e-4 1.75e-4 1.75e-4 \
    1.75e-4 1.75e-4 1.75e-4 1.75e-4 \
    --fit_epoch 0 --random_replay_batch --clear_model --clear_model \
    --lamaml --entropy_coe 0.01 --fp32 --adam_epsilon 1e-6 --learning_rate 1.75e-4 \
    --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED \
    > $LOG_DIR/log.$ID.train.NLG.$SEED 2>&1
sleep 30
CUDA_VISIBLE_DEVICES=$CUDA_ID bash test_myadaptor.sh \
    --id $ID \
    --gen_lm_sample_percentage 0.2 \
    --cm_fwt_dwt_flag \
    --cm_bleu_for_nlg \
    --cm_test_batch_size 8 \
    --cm_test_tasks $test_nlg_order01 --tasks $nlg_order01 \
    --fp32 --seq_train_type lll --model_name gpt2 \
    --add_task_tokens --seed $SEED \
    > $LOG_DIR/log.$ID.test.NLG.$SEED 2>&1
