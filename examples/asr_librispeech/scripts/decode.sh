#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/stek/lconcina/SLAM-LLM-DVC-/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

#speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Large.pt
speech_encoder_path=$1
#llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
llm_path=$2

#output_dir=/root/tmp/vicuna-7b-v1.5-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-20240426
output_dir=$3
ckpt_path=$output_dir/asr_epoch_4_step_4729  #TODO find a way to expose this parameter
#split=librispeech_test_clean
split=$4
#val_data_path=/nfs/maziyang.mzy/data/librispeech/${split}.jsonl
test_data_path=$5/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam4

llm_name=$6
llm_dim=$7
encoder_name=$8
encoder_dim=$9
encoder_projector=${10}
num_epochs=${11}
val_batch_size=${12}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=$llm_name \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=$encoder_name \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=$encoder_projector \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$test_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=$num_epochs \
        ++train_config.val_batch_size=$val_batch_size \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        ++log_config.log_file=$output_dir/test.log 
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
