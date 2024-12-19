# Gemma2-2b model.
# Not yet verified to work.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
#
# Example to invoke this script:
# bash MaxText/configs/v4/gemma2_2b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>"
#

# Stop execution if any command exits with error
set -e

export EXECUTABLE="train.py" # or train_compile.py
export RUN_PREFLIGHT="true"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# The setup accommodates two cases:
# 1) Passing the 'RUN_NAME' variable at runtime
# 2) Propagating the 'M_RUN_NAME' variable within an Airflow sweeping workflow
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

# Set up network optimizations
if [ "$RUN_PREFLIGHT" = "true" ]; then
    bash preflight.sh
fi

MODEL_NAME="gemma2-2b"
MAX_TARGET_LENGTH=8192

REMAT_POLICY="full"
FSDP=-1
FSDP_TRANSPOSE=256

# is this a good VMEM_LIMIT value for 2B? not yet known
VMEM_LIMIT=114688
PER_DEVICE_BATCH_SIZE=3
BLOCK_SIZE=2048

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_scoped_vmem_limit_kib=${VMEM_LIMIT} --xla_tpu_enable_async_collective_fusion=true --xla_tpu_assign_all_reduce_scatter_layout --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/$EXECUTABLE MaxText/configs/base.yml\
        model_name=${MODEL_NAME}\
        base_output_directory=${BASE_OUTPUT_DIR}\
        dataset_path=${DATASET_PATH}\
        tokenizer_path=assets/tokenizer.gemma\
        max_target_length=${MAX_TARGET_LENGTH}\
        ici_fsdp_parallelism=${FSDP}\
        ici_fsdp_transpose_parallelism=${FSDP_TRANSPOSE}\
        per_device_batch_size=${PER_DEVICE_BATCH_SIZE}\
        remat_policy=${REMAT_POLICY}\
        steps=30\
        enable_checkpointing=false\
        use_iota_embed=true\
        gcs_metrics=true\
        dataset_type=synthetic\
        reuse_example_batch=1\
        enable_checkpointing=False\
        profiler=xplane\
        attention=flash\
        sa_block_q=${BLOCK_SIZE}\
        sa_block_q_dkv=${BLOCK_SIZE}\
        sa_block_q_dq=${BLOCK_SIZE}\
        base_output_directory=$OUTPUT_PATH\
        dataset_path=$DATASET_PATH 
