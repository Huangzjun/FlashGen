export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY=
DATA_DIR=mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset/
NUM_GPUS=8
CONFIG=${CONFIG:-scripts/distill/wan_dmd_npu.yaml}
export FLASHGEN_ATTENTION_BACKEND=TORCH_SDPA
export FLASHGEN_TARGET_DEVICE=npu
export TOKENIZERS_PARALLELISM=false

# make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    -m flashgen.train.entrypoint.train \
    --config "$CONFIG" \
    --training.data.data_path "$DATA_DIR" \
    --training.distributed.num_gpus "$NUM_GPUS" \
    --training.distributed.hsdp_replicate_dim "$NUM_GPUS"
