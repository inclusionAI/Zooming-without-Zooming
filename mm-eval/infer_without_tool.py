from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from concurrent.futures import ThreadPoolExecutor # Import multithreading
import argparse
import re, json, torch
from tqdm import tqdm
import time
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch._dynamo

torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default='mathverse', type=str)
parser.add_argument("--model_path", default='Qwen/Qwen2.5-VL-7B-Instruct', type=str)
parser.add_argument("--model", default='qwen2.5vl-7b', type=str)
parser.add_argument("--out_dir", default='model_answer', type=str)
parser.add_argument("--temperature", default=0.7, type=float)
parser.add_argument("--gpus", default=1, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--batch_size", default=1024, type=int, help="Number of samples per batch, adjust according to memory")
args = parser.parse_args()

def preprocess_item(item, processor, benchmark):
    try:
        if benchmark == 'zoom-bench-crop':
            img_path = item['crop_images'][0]
            # print(img_path)
        else:
            img_path = item['images'][0]
        
        inst = item['query'].replace('<image>', '')
        messages = [
            {"role": "system", "content": "You are a helpful assistant.\n The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The answer are enclosed within <answer> </answer> tags, respectively, i.e., reasoning process here <answer> answer here </answer>"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "min_pixels": 65536, "max_pixels": 16777216},
                    {"type": "text", "text": inst},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        mm_data = {}
        if image_inputs is not None: mm_data["image"] = image_inputs
        if video_inputs is not None: mm_data["video"] = video_inputs
        
        return {"prompt": prompt, "multi_modal_data": mm_data, "raw_item": item}
    except Exception as e:
        print(f"{e}")
        return None




if __name__ == '__main__':
    
    MODEL_PATH = args.model_path
    if args.benchmark == 'countqa':
        BENCHMARK_PATH = 'test.json'
    if args.benchmark == 'hrbench-4k':
        BENCHMARK_PATH = 'hr_bench_4k.json'
    if args.benchmark == 'hrbench-8k':
        BENCHMARK_PATH = 'hr_bench_8k.json'
    if args.benchmark == 'vstar':
        BENCHMARK_PATH = 'vstar.json'
    if args.benchmark == 'cvbench-2d':
        BENCHMARK_PATH = 'test_2d.json'
    if args.benchmark == 'cvbench-3d':
        BENCHMARK_PATH = 'test_3d.json'
    if args.benchmark == 'mmstar':
        BENCHMARK_PATH = 'MMStar.json'
    if args.benchmark == 'babyvision':
        BENCHMARK_PATH = 'babyvision.json'
    if args.benchmark == 'mme-realworld':
        BENCHMARK_PATH = 'MME_RealWorld.json'
    if args.benchmark == 'mme-realworld-cn':
        BENCHMARK_PATH = 'MME_RealWorld_CN.json'
    if args.benchmark == 'perception_bench_1':
        BENCHMARK_PATH = 'hallusion.json'
    if args.benchmark == 'colorbench':
        BENCHMARK_PATH = 'test.json'
    if args.benchmark == 'zoom-bench':
        BENCHMARK_PATH = 'zoombench.json'
    if args.benchmark == 'zoom-bench-crop':
        BENCHMARK_PATH = 'zoombench.json'
    if args.benchmark == "fakeclue":
        BENCHMARK_PATH = 'fakeclue.json'
    if args.benchmark == "forensicsbench":
        BENCHMARK_PATH = 'forensicsbench.json'
    if args.benchmark == "loki":
        BENCHMARK_PATH = 'loki_image.json'



    tensor_parallel_size = args.gpus
    gpu_memory_utilization = 0.8
    out_dir = os.path.join(args.out_dir, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)
    OUT_PATH = f'{out_dir}/{args.model}_seed{args.seed}_answer.json'
    
    # 2. Resume from checkpoint logic: count processed samples
    done_sample_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    identifier = data.get('images')[0] + data.get('query')
                    # print(identifier)
                    done_sample_ids.add(identifier)
                except:
                    continue
        print(f"Checkpoint detected, {len(done_sample_ids)} samples already completed.")

    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        total_data = json.load(f)
    
    todo_data = [
        item for item in total_data 
        if (item.get('images')[0] + item.get('query')) not in done_sample_ids
    ]
    
    if not todo_data:
        print("All samples have been processed.")
        exit()

    print(f"Remaining samples to process: {len(todo_data)}")

    # 4. Initialize model
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 1, "video": 1},
        dtype=torch.bfloat16, 
        gpu_memory_utilization=0.8, 
        tensor_parallel_size=args.gpus,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=args.temperature,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        seed=args.seed,
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, max_pixels=16777216)

    total_latency = 0
    total_output_tokens = 0
    processed_count = 0

    # 5. Batch inference loop
    batch_size = args.batch_size
    for i in range(0, len(todo_data), batch_size):
        chunk = todo_data[i : i + batch_size]
        prompt_list = []
        valid_chunk_items = []

        print(f"Preprocessing images in parallel... (Batch {i//batch_size})")
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Adjust max_workers according to your CPU core count
            results = list(executor.map(lambda x: preprocess_item(x, processor, args.benchmark), chunk))
        
        # Filter out failed samples
        valid_results = [r for r in results if r is not None]
        prompt_list = [{"prompt": r["prompt"], "multi_modal_data": r["multi_modal_data"]} for r in valid_results]
        valid_chunk_items = [r["raw_item"] for r in valid_results]

        if not prompt_list:
            continue
            
        # --- Timing: Model inference ---
        gen_start = time.time()
        outputs = llm.generate(prompt_list, sampling_params=sampling_params)
        gen_end = time.time()
        
        batch_latency = gen_end - gen_start
        total_latency += batch_latency
        
        # Write results and count tokens
        with open(OUT_PATH, 'a', encoding='utf-8') as f_out:
            for j, output in enumerate(outputs):
                response = output.outputs[0].text
                # Get the number of generated tokens
                out_tokens = len(output.outputs[0].token_ids)
                total_output_tokens += out_tokens
                
                result_item = valid_chunk_items[j].copy()
                result_item['model_answer'] = response
                
                # Calculate average token latency for this batch (approximate value)
                # vllm is parallel, so single-item latency is usually described as batch_time / batch_size
                result_item['gen_time_sec'] = round(batch_latency, 3) 
                result_item['output_tokens'] = out_tokens
                result_item['tokens_per_sec'] = round(out_tokens / (batch_latency / len(outputs)), 2) if batch_latency > 0 else 0
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
        
        processed_count += len(prompt_list)
        print(f"Completed: {processed_count} / {len(todo_data)} | "
              f"Batch time: {batch_latency:.2f}s | "
              f"Average speed: {total_output_tokens / total_latency:.2f} tokens/s")

    # 7. Print final summary report
    print("\n" + "="*50)
    print("Inference task completion statistics:")
    print(f"Total samples processed: {processed_count}")
    print(f"Total inference time: {total_latency:.2f} seconds")
    print(f"Total tokens generated: {total_output_tokens}")
    if total_latency > 0:
        print(f"System throughput: {total_output_tokens / total_latency:.2f} tokens/s")
        print(f"Average inference time per sample: {total_latency / processed_count:.3f} seconds")
    print(f"Results saved to: {OUT_PATH}")
    print("="*50)
