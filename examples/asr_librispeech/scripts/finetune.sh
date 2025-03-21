#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/stek/lconcina/SLAM-LLM-DVC-/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=$1
llm_path=$2
train_data_path=$3
val_data_path=$4

exp_data=speechMassive
learn_rate=$5

#config data coming from params.yaml
llm_name=$6
llm_dim=$7
encoder_name=$8
encoder_projector_ds_rate=$9
encoder_dim=${10}
encoder_projector=${11}


num_epochs=${12}
#echo "num_epochs: ${12}"
warmup_steps=${13}
total_steps=${14}
batch_size_training=${15}
val_batch_size=${16}
output_dir=${17}

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=$encoder_name \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=$encoder_projector \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=$num_epochs \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=$warmup_steps \
++train_config.total_steps=$total_steps \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=$val_batch_size \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
++log_config.log_file=$output_dir/train.log \
"

# -m debugpy --listen 5678 --wait-for-client
## -m debugpy --listen 5678 --wait-for-client
#if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
#    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
#        --config-path "conf" \
#        --config-name "prompt.yaml" \
#        $hydra_args
#else
#    torchrun \
#        --nnodes 1 \
#        --nproc_per_node 2 \
#        --master_port=29503 \
#        $code_dir/finetune_asr.py \
#        --config-path "conf" \
#        --config-name "prompt.yaml" \
#        ++train_config.enable_fsdp=false \
#        ++train_config.enable_ddp=true \
#        ++train_config.use_fp16=true \
#        $hydra_args
#fi

# Removed the condition and always run torchrun
# set nproc per node = 1 if only 1 GPU
torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    $code_dir/finetune_asr.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=true \
    $hydra_args