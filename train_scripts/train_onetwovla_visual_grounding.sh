logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

num_devices=$(nvidia-smi --list-gpus | wc -l)
single_batch_size=20
batch_size=$((num_devices * single_batch_size))
echo batch_size $batch_size

single_val_batch_size=12
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

# the json file can be downloaded from https://huggingface.co/datasets/Richard-Nai/onetwovla-dataset/tree/main/wild_move_to
# ensure the dataset's path is $LEROBOT_HOME/umi/wild_move_to
reasoning_json_path=/path/to/your/cot.json

# normalization stats
# this can only run on a single GPU.
# this code only needs to run once.
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py onetwovla_visual_grounding --exp-name=computing-norm \
--create_train_val_split --val_ratio=0.05 \
--reasoning_json_path $reasoning_json_path \
--is_computing_norm_stats

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py onetwovla_visual_grounding  --exp-name=${now_date}/${now_seconds}/one-two-vla-visual-grounding --batch-size=$batch_size \
--reasoning_json_path $reasoning_json_path
