from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import multimodal_typewriter_print
import argparse
import json
import os
import time
from typing import Dict, Any
from tqdm import tqdm
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default='zoom-bench', type=str)
parser.add_argument("--model", default='qwen3-4b', type=str)
parser.add_argument("--out_dir", default='model_answer', type=str)
parser.add_argument("--temperature", default=0.7, type=float)
parser.add_argument("--api_base", default='http://localhost:18901/v1', type=str, help="vLLM service address")
parser.add_argument("--api_key", default='EMPTY', type=str, help="API key")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--max_samples", default=None, type=int, help="Maximum number of samples to process, None means all")
parser.add_argument("--use_tools", default=True, type=bool, help="Whether to use agent tools")
args = parser.parse_args()


def get_benchmark_path(benchmark_name: str) -> str:
    """Return data path based on benchmark name"""
    benchmark_paths = {
        'countqa': 'test.json',
        'hrbench-4k': 'hr_bench_4k.json',
        'hrbench-8k': 'hr_bench_8k.json',
        'vstar': 'vstar.json',
        'cvbench-2d': 'test_2d.json',
        'cvbench-3d': 'test_3d.json',
        'mmstar': 'MMStar.json',
        'babyvision': 'babyvision.json',
        'mme-realworld': 'MME_RealWorld.json',
        'mme-realworld-cn': 'MME_RealWorld_CN.json',
        'perception_bench_1': 'hallusion.json',
        'colorbench': 'test.json',
        'zoom-bench': 'zoombench.json',
        'zoom-bench-crop': 'zoombench.json',
        'fakeclue': 'fakeclue.json',
        'forensicsbench': 'forensicsbench.json',
        'loki': 'loki_image.json'
    }
    return benchmark_paths.get(benchmark_name, '')


def initialize_agent(
    api_base: str,
    api_key: str,
    temperature: float,
    model,
    use_tools: bool = False
) -> Assistant:
    """Initialize Agent, keeping interface consistent with script B"""
    llm_cfg = {
        'model_type': 'qwenvl_oai',
        'model': model,
        'model_server': api_base,
        'api_key': api_key,
        'generate_cfg': {
            "top_p": 0.8,
            "top_k": 20,
            "temperature": temperature,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "max_tokens": 8192,
        }
    }

    analysis_prompt = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""

    tools = ['image_zoom_in_tool'] if use_tools else []
    
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=analysis_prompt,
    )
    return agent


if __name__ == '__main__':
    
    BENCHMARK_PATH = get_benchmark_path(args.benchmark)
    if not BENCHMARK_PATH:
        print(f"Benchmark '{args.benchmark}' not found.")
        exit()
    
    # Set output path (keep format consistent with script A)
    out_dir = os.path.join(args.out_dir, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)
    OUT_PATH = f'{out_dir}/{args.model}_seed{args.seed}_answer.json'
    
    # Resume from checkpoint: count processed samples
    done_sample_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    identifier = data.get('images')[0] + data.get('query')
                    done_sample_ids.add(identifier)
                except:
                    continue
        print(f"Checkpoint detected, {len(done_sample_ids)} samples already completed.")

    # Load benchmark data
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        total_data = json.load(f)
    
    # Limit sample count
    if args.max_samples and len(total_data) > args.max_samples:
        total_data = total_data[:args.max_samples]
    
    # Filter out already processed samples
    todo_data = [
        item for item in total_data 
        if (item.get('images')[0] + item.get('query')) not in done_sample_ids
    ]
    
    if not todo_data:
        print("All samples have been processed.")
        exit()

    print(f"Remaining samples to process: {len(todo_data)}")

    # Initialize Agent
    agent = initialize_agent(
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        model = args.model,
        use_tools=args.use_tools
    )

    # Inference loop
    inference_times = []
    
    for idx, item in enumerate(tqdm(todo_data, desc=f"Processing {args.benchmark}")):
        try:
            # Get image path
            if args.benchmark == 'zoom-bench-crop':
                img_path = item['crop_images'][0]
            else:
                img_path = item['images'][0]
            
            # Get query text
            inst = item['query'].replace('<image>', '').strip()
            
            if not os.path.exists(img_path):
                print(f"Skip: Image does not exist {img_path}")
                continue
            
            # Construct message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": img_path},
                        {"text": inst},
                    ]
                }
            ]
            
            # Start timing
            start_time = time.time()
            
            # Call Agent inference
            response_plain_text = ''
            for ret_messages in agent.run(messages):
                response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
            
            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            inference_times.append(elapsed_time)
            
            # Save result (keep format consistent with script A)
            result_item = item.copy()
            result_item['model_answer'] = response_plain_text
            
            with open(OUT_PATH, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            
            # Print statistics every 10 samples
            if (idx + 1) % 10 == 0:
                current_avg = statistics.mean(inference_times)
                print(f"\nCompleted: {idx + 1} / {len(todo_data)}, Current average latency: {current_avg:.2f}s")
        
        except Exception as e:
            print(f"Processing failed (idx: {idx}): {e}")
            continue

    # Final statistics
    print(f"\n{'='*60}")
    print(f"Inference completed - Benchmark: {args.benchmark}")
    print(f"{'='*60}")
    print(f"Total samples: {len(total_data)}")
    print(f"Successfully processed: {len(inference_times)}")
    print(f"Results saved to: {OUT_PATH}")
    
    if inference_times:
        avg_latency = statistics.mean(inference_times)
        min_latency = min(inference_times)
        max_latency = max(inference_times)
        std_latency = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
        
        print(f"\nLatency statistics:")
        print(f"  Average latency (Mean):    {avg_latency:.2f}s")
        print(f"  Minimum latency (Min):     {min_latency:.2f}s")
        print(f"  Maximum latency (Max):     {max_latency:.2f}s")
        print(f"  Standard deviation (Std):  {std_latency:.2f}s")
        
        # Save statistics
        stats_path = f'{out_dir}/{args.model}_seed{args.seed}_stats.json'
        stats = {
            'benchmark': args.benchmark,
            'model': args.model,
            'total_samples': len(total_data),
            'processed_samples': len(inference_times),
            'avg_latency': avg_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'std_latency': std_latency,
            'inference_times': inference_times,
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  Statistics saved to: {stats_path}")
    else:
        print("No samples were successfully processed.")
