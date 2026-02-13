export MKL_SERVICE_FORCE_INTEL=1

benchmarks=('zoom-bench' 'zoom-bench-crop' "mmstar" "hrbench-4k" "hrbench-8k" "cvbench-2d" "cvbench-3d" "vstar" "countqa" "colorbench"  "babyvision" "mme-realworld" "mme-realworld-cn")

model_path="Qwen3/Qwen3-VL-8B-Instruct"
model_name="baseline_8B"
model_names="${model_name}_seed42"

echo "=========================================="
echo "Processing step: $step"
echo "Model path: $model_path"
echo "Model name: $model_name"
echo "=========================================="

for benchmark in "${benchmarks[@]}"; do
    echo "Running inference for model: $model_name on benchmark: $benchmark"
    python infer_without_tool.py \
        --benchmark "$benchmark" \
        --model "$model_name" \
        --model_path "$model_path" \
        --gpus 4
    
    echo "Running judge for model: $model_name on benchmark: $benchmark"
    python judge_qwenlm.py \
        --benchmark "$benchmark" \
        --model "$model_names"
    
    echo "---------- Completed $benchmark for $model_name ----------"
done

echo "========== Completed all benchmarks for $model_name =========="
echo ""


