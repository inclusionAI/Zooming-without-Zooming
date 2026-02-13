import json
import os

def calculate_accuracy(file_path):

    total_records = 0
    correct_records = 0

    # try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for record in data:
        total_records += 1
        if record.get("judge") == "Yes":
            correct_records += 1
    if total_records == 0:
        return 0.0
    else:
        accuracy = correct_records / total_records
        return accuracy

# 示例使用
if __name__ == "__main__":
    model_name = f"baseline_8B"
    full_name = f"{model_name}_seed42_answer.json"
    benchmarks = ["zoom-bench", "zoom-bench-crop","mmstar", "hrbench-4k", "hrbench-8k", "vstar", "cvbench-2d", "cvbench-3d", "countqa", "colorbench", "mathvision","babyvision",'loki','mme-realworld', 'mme-realworld-cn']
    for benchmark in benchmarks:
        jsonl_file = os.path.join("./judge", benchmark, full_name)
        try:
            accuracy = calculate_accuracy(jsonl_file)

            if accuracy is not None:
                print(f"{model_name}:{benchmark}:acc:{accuracy*100:.2f}")
        except:
            continue