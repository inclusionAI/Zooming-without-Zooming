# The paper uses the post-trained Qwen/Qwen3-8B checkpoint, not a base model.
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-Qwen/Qwen3-8B}"

CUDA_VISIBLE_DEVICES=12,13,14,15 \
vllm serve "$REWARD_MODEL_PATH" \
    --port 18901 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --data-parallel-size 4 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests
