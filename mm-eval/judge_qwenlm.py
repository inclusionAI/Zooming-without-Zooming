import json
from tqdm import tqdm
import base64
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
import time
import argparse, os
import torch._dynamo
import re
from mathruler.grader import grade_answer

torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default='mathverse', type=str)
parser.add_argument("--model", default='qwen2.5vl-7b', type=str)
args = parser.parse_args()
mcq_benchmarks = ["mmstar", "hrbench-4k", "hrbench-8k","vstar", "cvbench-2d", "cvbench-3d", "colorbench", "mme-realworld", "mme-realworld-cn"]

def extract_first_option(text):
    if not text:
        return ""
    
    # Regex logic here:
    # 1. First match letters in parentheses, like (A)
    # 2. Then match A. or A) or isolated A
    # [A-Z] represents uppercase letters
    
    # Pattern 1: Match (A)
    match = re.search(r'\(([A-Z])\)', text)
    if match:
        return match.group(1)
    
    # Pattern 2: Match A. or A) or A followed by space
    match = re.search(r'([A-Z])[\.\)\s]', text)
    if match:
        return match.group(1)

    # Pattern 3: Directly find the first uppercase letter (fallback)
    match = re.search(r'([A-Z])', text)
    if match:
        return match.group(1)
        
    return ""

def extract_mcq_option(answer):
    """
    Determine if the answer is in multiple choice format (e.g.: A, A., (A), A xxx)
    While excluding common words like Any, Apple, Area.
    """

    if not isinstance(answer, str) or not answer:
        return ''
    
    # Remove leading and trailing spaces
    text = answer.strip()
    
    pattern = r'^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))'
    match = re.match(pattern, text)

    if match:
        return match.group(1)  # Return the captured letter
    return ""
  
def first_letter_match(gt, answer):
    # gt_val = extract_first_option(gt)
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    if gt and pred_val and gt_val == pred_val:
      return True
    else:
      return False

if __name__ == '__main__':
    answer_path = f"model_answer/{args.benchmark}/{args.model}_answer.json"
    save_path = f"judge/{args.benchmark}/{args.model}_answer.json"
    os.makedirs(f"judge/{args.benchmark}", exist_ok=True)
    is_mcq = args.benchmark in mcq_benchmarks

    # with open(answer_path, 'r', encoding='utf-8') as f:
    #     data_list = json.load(f)
    data_list = []
    with open(answer_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    # Initialize model (only used when needed)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0)
    llm = LLM(model='/r-contentsecurity/share/checkpoints/opensources/Qwen3-30B-A3B-Instruct-2507',tensor_parallel_size=1,dtype=torch.bfloat16, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained('/r-contentsecurity/share/checkpoints/opensources/Qwen3-30B-A3B-Instruct-2507')

    prompt_template = "Your task is to judge whether the response expresses the same meaning as the answer of a question.\nThe question is: {question}\nThe answer is: {gt}\nThe response is: {response}\nPlease check and compare them and then judge. If the response is correct, your output should be Yes. Otherwise, your output should be No. Directly give me your output."
    prompt_lists = []
    # Pre-define Prompt template
    prompt_template = "Your task is to judge whether the response expresses the same meaning as the answer of a question.\nThe question is: {question}\nThe answer is: {gt}\nThe response is: {response}\nPlease check and compare them and then judge. If the response is correct, your output should be Yes. Otherwise, your output should be No. Directly give me your output."

    # Step 1: Use mathruler for initial judgment
    to_llm_indices = []  # Record indices that need LLM intervention
    prompt_lists = []    # Store LLM prompts
    
    print("Step 1: Running MathRuler Grader...")
    for i, item in enumerate(tqdm(data_list)):
        # --- Answer extraction logic ---
        question = item['query'].replace('<image>', '')
        model_answer_raw = item['model_answer']
        
        if '<answer>' in model_answer_raw:
            extracted_answer = model_answer_raw[model_answer_raw.find('<answer>'):model_answer_raw.find('</answer>')].replace('<answer>', '').replace('</answer>', '')
        elif 'Answer:' in model_answer_raw:
            extracted_answer = model_answer_raw[model_answer_raw.find('Answer:'):]
        else:
            extracted_answer = '\n'.join(model_answer_raw.split('\n')[-3:])
        
        gt = item['response']
        item['extracted_answer'] = extracted_answer # Save the extracted answer
        
        # --- Try using MathRuler ---
        try:
            # grade_answer usually returns True/False or 1/0
            is_correct = grade_answer(gt, extracted_answer)
        except Exception as e:
            print(f"Grader error at index {i}: {e}")
            is_correct = False
        
        # --- For MCQ, use first uppercase letter matching ---
        is_letter_correct = False
        if not is_correct and is_mcq:
            try:
            # grade_answer usually returns True/False or 1/0
                is_letter_correct = first_letter_match(gt, extracted_answer)
            except Exception as e:
                print(f"Grader error at index {i}: {e}")
                is_letter_correct = False

        if is_correct:
            item['judge'] = 'Yes'
            item['judge_source'] = 'mathruler'
        elif is_letter_correct:
            item['judge'] = 'Yes'
            item['judge_source'] = 'first letter'
        else:
            # Only here we need to build chat template
            messages = [{"role": "user", "content": prompt_template.format(gt=gt, response=extracted_answer, question=question)}]
            text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            # Delay tokenizer initialization to save resources, or initialize outside the loop
            to_llm_indices.append(i)
            prompt_lists.append(text)

    # Step 2: Call LLM for failed cases
    if prompt_lists:
        print(f"Step 2: Calling LLM for {len(prompt_lists)} remaining cases...")
        
        
        # Batch generation
        outputs = llm.generate(prompt_lists, sampling_params)
        
        # Fill results back into data_list
        for idx_in_llm, output in enumerate(outputs):
            original_idx = to_llm_indices[idx_in_llm]
            response_text = output.outputs[0].text.strip()
            data_list[original_idx]['judge'] = response_text
            data_list[original_idx]['judge_source'] = 'llm'

    # Step 3: Statistics and saving
    correct_num = 0
    for item in data_list:
        if 'judge' in item and ('Yes' in item['judge'] or 'yes' in item['judge']):
            correct_num += 1

    print(f"Final Accuracy: {correct_num/len(data_list):.4f}")
    print(f"Total: {len(data_list)}, LLM used: {len(prompt_lists)}")

    with open(save_path, 'w', encoding='utf-8') as out_file:
        json.dump(data_list, out_file, ensure_ascii=False, indent=4)
