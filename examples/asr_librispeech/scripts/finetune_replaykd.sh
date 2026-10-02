#!/bin/bash
# ============================================================================
# Replay + Eq.2 ablation (the experiment your boss asked for).
#
# FULL fine-tuning of a plain linear projector + LoRA, but trained with the SVR
# rehearsal loss (per-step memory minibatch + KD against the base model, Eq.2)
# instead of plain data-mixing cross-entropy. This inserts the missing middle
# point of the chain, so each step isolates ONE variable:
#
#     Replay (plain CE)  -->  Replay+Eq.2 (THIS)  -->  SVR (gated + Eq.2)
#     |------- the loss -------|                   |------- the gating -------|
#
# Differences vs the SVR finetune.sh:
#   * encoder_projector   = linear   (NOT linear-svr)  -> the full projector trains
#   * ckpt_path           = BASE MEUSLI (28-lang)      -> start = teacher = theta_0
#   * svr_full_ft_teacher = true     (snapshot the base projector for the teacher)
#   * lr                  = 1e-4     (original replay recipe; whole proj + LoRA)
# Everything else (buffer, lambda, KD temp, batch size, epochs, decoding) MUST
# stay identical to your reported SVR runs, so the ONLY change vs SVR is the
# gating, and the ONLY change vs Replay is the loss.
# ============================================================================

export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

run_dir=/stek/lconcina/SLAM-LLM-DVC-/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

# ============================ EDIT THESE ============================
# Base 28-language MEUSLI checkpoint (projector + base LoRA) = theta_0.
# Same file you pass to build_svr_projector.py as --base.
base_ckpt=/stek/lconcina/SLAM-LLM-DVC-/train_output/meusli_backup/EuroLLM-1.7B-Instruct-lora8r32a-multilingual-linear/only_checkpoint/model.pt        # <-- FILL IN

# Rehearsal buffer (per-base-language, NO Ukrainian). Use the SAME file as the
# matching SVR run: the 25-sample buffer for the 25 row, the 500 for the 500 row.
memory_data_path=/stek/lconcina/SLAM-LLM-DVC-/data/cv_17_ukranian_continuallearning/500_samples_data_reply/data_reply_500_train_random.jsonl    # <-- FILL IN

# Output dir (one per buffer size, e.g. .../replaykd/uk_25 and .../uk_500).
output_dir=/stek/lconcina/SLAM-LLM-DVC-/train_output/SVR_experiments/replaykd_ablation/uk_500  # <-- FILL IN

# MUST equal the lambda (svr_mem_weight) of your reported SVR runs.
# NOTE: finetune.sh currently has 5 but Appendix A of the paper says 2 --
# settle which value produced the reported SVR numbers and use it here.
lambda_mem=5                                                                # <-- CONFIRM
# ===================================================================

# New task = Ukrainian only (from params.yaml).
train_data_path=/stek/lconcina/SLAM-LLM-DVC-/data/cv17_ukranian_data/cv_17_train.jsonl
val_data_path=/stek/lconcina/SLAM-LLM-DVC-/data/cv17_ukranian_data/cv_17_validation.jsonl

# Model / encoder config (from params.yaml -- do not change for fairness).
speech_encoder_path=large-v3-turbo
llm_path=/stek/lconcina/SLAM-LLM-DVC-/models/eurollm-1.7b
llm_name=eurollm-1.7b
llm_dim=2048
encoder_name=whisper
encoder_projector_ds_rate=5
encoder_dim=1280
encoder_projector=linear         # plain projector -> full fine-tuning (no gates)
num_epochs=5
warmup_steps=1000
total_steps=1000000
batch_size_training=2            # keep identical to your SVR runs
val_batch_size=2

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++ckpt_path=$base_ckpt \
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
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++train_config.model_name=asr \
++train_config.num_epochs=$num_epochs \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=$warmup_steps \
++train_config.total_steps=$total_steps \
++train_config.lr=1e-4 \
++train_config.validation_interval=12568 \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=$val_batch_size \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++train_config.freeze_peft=false \
++train_config.max_grad_norm=1.0 \
++train_config.svr_kd=true \
++train_config.svr_full_ft_teacher=true \
++train_config.svr_mem_weight=$lambda_mem \
++train_config.svr_kd_temperature=1.0 \
++dataset_config.memory_data_path=$memory_data_path \
++metric=acc \
++log_config.log_file=$output_dir/train.log \
"

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
