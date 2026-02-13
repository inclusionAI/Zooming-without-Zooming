CUDA_VISIBLE_DEVICES=12,13,14,15 \
vllm serve /mnt/r-contentsecurity-p/common/checkpoints/opensources/Qwen3-VL-4B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --data-parallel-size 4 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests