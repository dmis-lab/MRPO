#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate MRPO
PYTHON_BIN="$CONDA_PREFIX/bin/python"
TORCHRUN_BIN="$CONDA_PREFIX/bin/torchrun"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export TOKENIZERS_PARALLELISM=false

echo "Checking GPU..."
nvidia-smi
"$PYTHON_BIN" -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

cd "$(dirname "${BASH_SOURCE[0]}")/VLM-R1/src/open-r1-multimodal/src/open_r1"
export PYTHONPATH=${PYTHONPATH}:$(dirname "$(pwd)")

export DEBUG_MODE="true"


### ========== IMPORTANT : Set the file paths and model paths ==========

export DATA_DIR="<DATA_DIR>"
export MODEL_DIR="<MODEL_DIR>"
export BIOMEDBERT_PATH="<BIOMEDBERT_PATH>"
export OUTPUT_DIR="<OUTPUT_DIR>"
export WANDB_API_KEY="<WANDB_API_KEY>"
export WANDB_PROJECT="<WANDB_PROJECT>"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PROCESS_REWARD_MODEL="gpt-5-mini"

### ====================================================================




RUN_NAME="${MODEL_NAME}-MRPO"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

case "${MODEL_NAME,,}" in
    *internvl*) DEEPSPEED_CONFIG="../../local_scripts/zero3_internvl3.json" ;;
    *)          DEEPSPEED_CONFIG="../../local_scripts/zero3.json" ;;
esac


"$TORCHRUN_BIN" --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    grpo_vqa_MRPO.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --train_file "$DATA_DIR/Data_Preprocessed/train_open_ended.json" \
    --test_file "$DATA_DIR/Data_Preprocessed/test_open_ended.json" \
    --gold_reasoning_file "$DATA_DIR/Medthink/medthink_train.json" \
    --process_reward_model "$PROCESS_REWARD_MODEL" \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --max_pixels 802816 \
    --num_generations 8 \
    --num_iterations 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --ignore_data_skip false \
    --reward_funcs "accuracy" "process" "step_count"
