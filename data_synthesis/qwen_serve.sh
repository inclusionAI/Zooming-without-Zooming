vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --mm-encoder-tp-mode data \
  --served-model-name "Qwen3-VL-8B-Instruct" \
  --async-scheduling \
  --port 18902