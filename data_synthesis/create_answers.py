import requests
import json
import base64
import os
import time
import argparse
from tqdm import tqdm
from PIL import Image
import io
from openai import OpenAI
import concurrent.futures
import threading, random
import re
from collections import Counter, defaultdict
from mathruler.grader import extract_boxed_content

# ==============================================================================
# API Configuration
# ==============================================================================

# Qwen3-VL-8B Client
mllm_client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:18902/v1",
    timeout=3600
)


# ==============================================================================
# Helper Functions 
# ==============================================================================

def load_local_image(image_path, image_type='PIL', max_retries=2):
    """
    从本地加载图片
    
    Args:
        image_path: 本地图片路径
        image_type: 'PIL' 返回PIL Image对象, 'base64' 返回base64编码字符串
        max_retries: 最大重试次数
    
    Returns:
        PIL Image 或 base64 字符串
    """
    retries = 0
    while retries <= max_retries:
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if image_type == 'PIL':
                image = Image.open(image_path)
                if image.mode == 'P':
                    image = image.convert('RGBA')
                else:
                    image = image.convert('RGB')
                return image
                
            elif image_type == 'base64':
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                return base64_image
                
            else:
                raise ValueError("Invalid image_type. Must be 'PIL' or 'base64'.")
                
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                print(f"Retrying load image ({retries}/{max_retries}): {image_path}")
                time.sleep(2)
            else:
                print(f"Failed to load image {image_path}: {e}")
                raise e
    
    return None


def expand_bbox(bbox, image_size, expand_ratio=0.05):
    """
    Expand bbox to avoid covering edge information
    
    Args:
        bbox: [x1, y1, x2, y2]
        image_size: (width, height)
        expand_ratio: expansion ratio, default 5%
    
    Returns:
        expanded_bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    width, height = image_size
    
    # Calculate bbox width and height
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Calculate expansion amount
    expand_w = bbox_width * expand_ratio
    expand_h = bbox_height * expand_ratio
    
    # Expand bbox
    new_x1 = max(0, x1 - expand_w)
    new_y1 = max(0, y1 - expand_h)
    new_x2 = min(width, x2 + expand_w)
    new_y2 = min(height, y2 + expand_h)
    
    return [new_x1, new_y1, new_x2, new_y2]


def draw_bbox_on_image(image_path, bbox, image_size, output_dir="/path/datasets/sa1b_bbox_images"):
    """Draw red bbox on image and save"""
    from PIL import ImageDraw
    import hashlib
    
    # Load image from local
    try:
        image = load_local_image(image_path, image_type='PIL')
        if image is None:
            return None
    except Exception as e:
        print(f"Failed to load image for bbox drawing: {image_path}, error: {e}")
        return None
    
    # Draw red bbox
    expanded_bbox = expand_bbox(bbox, image_size, 0.05)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = expanded_bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    
    # Generate unique filename
    os.makedirs(output_dir, exist_ok=True)
    hash_str = hashlib.md5(f"{image_path}_{bbox}_expand_0.05".encode()).hexdigest()
    output_path = os.path.join(output_dir, f"{hash_str}.png")
    
    # Save
    image.save(output_path, "PNG")
    
    return output_path


# ==============================================================================
# API Call Functions
# ==============================================================================

def call_mllm_api(payload: dict, api_url: str, api_key: str, model_name: str, 
                  max_retries: int = 300, base_retry_delay: float = 1.0):
    """Generic MLLM API call with retry"""
    for retry_count in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            
            # Ensure model name is in payload
            payload["model"] = model_name
            
            resp = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120,
                stream=False
            )
            
            if resp.status_code in (429, 503):
                time.sleep(base_retry_delay)
                continue
            
            if resp.status_code != 200:
                time.sleep(base_retry_delay)
                continue
            
            data = resp.json()
            return data
            
        except Exception as e:
            time.sleep(base_retry_delay)
    
    return None


def generate_mcq_with_mllm(bbox_image_path, question, correct_answer, api_url, api_key, 
                            model_name, num_distractors=3, max_retries=100):
    """
    Generate MCQ based on bbox_image_path + (question, correct_answer)
    """
    prompt = f"""You are an expert at creating multiple-choice questions. 

Given:
- Question: {question}
- Correct Answer: {correct_answer}
- Image: [provided]

Your task:
Generate {num_distractors} plausible but INCORRECT distractor options for this question based on the image content. The distractors should:
1. Be relevant to the image and question context
2. Be clearly different from the correct answer
3. Be plausible enough that they could seem correct at first glance
4. Not be obviously wrong
5. Have similar format and length as the correct answer

Important: Only provide the distractor options, NOT the correct answer.

Format your response as a JSON list:
["distractor1", "distractor2", "distractor3"]

Output the JSON list.
"""

    for _ in range(max_retries):
        try:
            # Load image from local
            image_b64 = load_local_image(bbox_image_path, image_type='base64')
            if not image_b64:
                time.sleep(1)
                continue

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]

            payload = {
                "stream": False,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 1024
            }

            resp_json = call_mllm_api(payload, api_url, api_key, model_name)
            if not resp_json:
                time.sleep(1)
                continue

            content = resp_json["choices"][0]["message"]["content"].strip()

            # Error tolerance: remove code block wrapping
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()

            distractors = json.loads(content)
            if not isinstance(distractors, list) or len(distractors) < num_distractors:
                time.sleep(1)
                continue

            distractors = distractors[:num_distractors]

            all_options = [correct_answer] + distractors
            random.shuffle(all_options)
            correct_idx = all_options.index(correct_answer)
            correct_option = chr(65 + correct_idx)

            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(all_options)])
            mcq_question = (
                question
                + "\n\n"
                + options_text
                + "\n\nAnswer with the option's letter from the given choices."
            )

            return {
                "mcq_question": mcq_question,
                "all_options": all_options,
                "correct_option": correct_option,
                "distractors": distractors
            }

        except Exception as e:
            print(f"Error in MCQ generation: {e}")
            time.sleep(1)

    return None


# ==============================================================================
# VQA Answer Generation
# ==============================================================================

def answer_question_with_mllm(image_b64, question, api_url, api_key, num_samples=8, model_name="Qwen3-VL-235B-A22B-Instruct"):
    """Use MLLM to answer question, sample multiple times"""
    answers = []
    
    for _ in range(num_samples):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": f"Question: {question}\n\nPlease reason step by step, and then put your answer within \\boxed{{}}."}
                ]
            }
        ]
        
        payload = {
            "stream": False,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        resp_json = call_mllm_api(payload, api_url, api_key, model_name)
        if resp_json:
            try:
                content = resp_json["choices"][0]["message"]["content"]
                content = extract_boxed_content(content) if 'boxed' in content else content
                answers.append(content.strip())
            except:
                answers.append("[ERROR: Parse failed]")
        else:
            answers.append("[ERROR: No response]")
    
    return answers


def answer_question_with_qwen8b(img_path, question, num_samples=4):
    """Use Qwen3-VL-8B to answer question, sample multiple times"""
    answers = []
    
    for _ in range(num_samples):
        try:
            # Load image from local
            image_encoded = load_local_image(img_path, image_type='base64')
            if not image_encoded:
                answers.append("[ERROR: Failed to load image]")
                continue

            if img_path.endswith(".mp4"):
                base64_image = f'data:video/mp4;base64,{image_encoded}'
                message_mm = {"type": "video_url", "video_url": {"url": base64_image}}
            else:
                base64_image = f"data:image;base64,{image_encoded}"
                message_mm = {"type": "image_url", "image_url": {"url": base64_image}}

            messages = [
                {
                    "role": "user",
                    "content": [
                        message_mm,
                        {"type": "text", "text": question + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}
                    ]
                }
            ]
            
            response = mllm_client.chat.completions.create(
                model="Qwen3-VL-8B-Instruct",
                messages=messages,
                temperature=1.0,
                max_tokens=20480
            )

            answer = response.choices[0].message.content
            if 'boxed' in answer:
                answer = extract_boxed_content(answer)
            elif '<|begin_of_box|>' in answer:
                answer = answer.split('<|begin_of_box|>')[1].split('<|end_of_box|>')[0]
            
            answers.append(answer.strip())
        
        except Exception as e:
            print(f"Error in Qwen8B answer generation: {str(e)}")
            time.sleep(1)
            answers.append("[ERROR: No response]")
    
    return answers


# ==============================================================================
# LLM Evaluation Functions
# ==============================================================================

def evaluate_consistency_with_llm(question, answers, client):
    """Use language model to evaluate answer consistency"""
    num_answers = len(answers)
    answers_text = "\n".join([f"Answer {i+1}: {ans}" for i, ans in enumerate(answers)])
    
    prompt = f"""You are an expert at evaluating the consistency of multiple answers to the same question.

Question: {question}

Here are {num_answers} different answers to this question:

{answers_text}

Your task:
1. Carefully analyze all {num_answers} answers
2. Group semantically similar or equivalent answers together
3. Identify the most common answer group (the mode)
4. Count how many answers belong to this most common group

Consider them equivalent if:
- Different phrasings of the same answer (e.g., "5 people" vs "five people")
- Slight variations in description that refer to the same entity
- Minor differences in precision that don't change the core answer (such as two colors that are similar: e.g., brown and dark brown)

Do NOT consider them equivalent if:
- They refer to different objects, numbers, or concepts
- They represent different interpretations of
the question

Please think step by step, then provide:
- The most common answer (normalized/standardized form)
- The count of how many answers are consistent with this most common answer (a number from 1 to {num_answers})

Format your response as:
Most Common Answer: [your answer here]
Consistency Count: [number]
"""

    max_retries = 2000
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="Kimi-K2-Instruct-0905",
                messages=[
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.3,
                max_tokens=20480,
            )
            
            response = completion.choices[0].message.content
            
            # Parse response
            most_common_answer = None
            consistency_count = 0
            
            for line in response.split('\n'):
                if 'Most Common Answer:' in line:
                    most_common_answer = line.split('Most Common Answer:')[1].strip()
                elif 'Consistency Count:' in line:
                    count_str = line.split('Consistency Count:')[1].strip()
                    numbers = re.findall(r'\d+', count_str)
                    if numbers:
                        consistency_count = int(numbers[0])
            
            return {
                "most_common_answer": most_common_answer,
                "consistency_count": consistency_count,
                "consistency_score": consistency_count / num_answers,
                "llm_response": response,
                "success": True,
                "error": None
            }
        
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "most_common_answer": None,
                    "consistency_count": 0,
                    "consistency_score": 0.0,
                    "llm_response": None,
                    "success": False,
                    "error": str(e)
                }
            time.sleep(2)


def check_correct_count(answers, ground_truth, client, question):
    """
    Check how many answers in the list match the ground truth
    
    Args:
        answers: list of answers
        ground_truth: the ground truth answer (majority_answer)
        client: LLM client
        question: the original question
    
    Returns:
        int: number of correct answers
    """
    num_answers = len(answers)
    answers_text = "\n".join([f"Answer {i+1}: {ans}" for i, ans in enumerate(answers)])
    
    prompt = f"""You are an expert at comparing answers for semantic equivalence.

Question: {question}

Ground Truth Answer: {ground_truth}

Here are {num_answers} candidate answers:
{answers_text}

Your task:
Count how many of these {num_answers} answers are semantically equivalent to the ground truth answer.
Consider them equivalent if:
- Different phrasings of the same answer (e.g., "5 people" vs "five people")
- Slight variations in description that refer to the same entity
- Minor differences in precision that don't change the core answer (such as two colors that are similar: e.g., brown and dark brown)

Do NOT consider them equivalent if:
- They refer to different objects, numbers, or concepts
- They represent different interpretations of the question

Please analyze each answer step by step, then provide your count.

Format your response as:
Correct Count: [0-{num_answers}]
Correct Answer Numbers: [list like "1, 3" or "None"]
Explanation: [Your reasoning here]
"""

    max_retries = 2000
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="Kimi-K2-Instruct-0905",
                messages=[
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.3,
                max_tokens=20480,
            )
            
            response = completion.choices[0].message.content
            
            # Parse response
            correct_count = 0
            
            for line in response.split('\n'):
                if 'Correct Count:' in line:
                    count_str = line.split('Correct Count:')[1].strip()
                    numbers = re.findall(r'\d+', count_str)
                    if numbers:
                        correct_count = int(numbers[0])
                    break
            
            return correct_count
        
        except Exception as e:
            if attempt == max_retries - 1:
                # If all retries fail, conservatively return 0
                return 0
            time.sleep(2)
    
    return 0


# ==============================================================================
# Main Processing Pipeline
# ==============================================================================

def process_single_vqa(original_image_path, crop_image_path, original_b64, question, bbox, image_size, args):
    """
    Process complete pipeline for single VQA question
    
    Returns: {
        "question": str,
        "status": "very easy vqa" | "valid vqa" | "invalid vqa",
        "majority_answer": str (crop answer - ground truth),
        "crop_eval": dict,
        "qwen8b_eval": dict,
        "bbox": list
    }
    """
    result = {
        "question": question,
        "status": "processing",
        "majority_answer": None,
        "crop_eval": None,
        "qwen8b_eval": None,
        "bbox": bbox,
        "image_size": image_size
    }
    
    # Step 1: Use MLLM to answer question on crop image (sample 8 times)
    try:
        # Load crop image from local
        crop_b64 = load_local_image(crop_image_path, image_type='base64')
        if not crop_b64:
            result["status"] = "invalid vqa"
            result["error"] = "Failed to load crop image"
            return result
        
        # Split into two groups (4 samples each)
        qwen_answers = answer_question_with_mllm(
            crop_b64, question, 
            args.api_url, args.api_key,
            num_samples=4, 
            model_name=args.model_name_qwen
        )
        glm_answers = answer_question_with_mllm(
            crop_b64, question,
            args.api_url, args.api_key,
            num_samples=4,
            model_name=args.model_name_glm
        )
        crop_answers = qwen_answers + glm_answers
    except Exception as e:
        result["status"] = "invalid vqa"
        result["error"] = f"Crop answering failed: {str(e)}"
        return result
    
    # Step 2: Evaluate crop answer consistency
    crop_eval = evaluate_consistency_with_llm(
        question, 
        crop_answers, 
        client=llm_client
    )
    result["crop_eval"] = crop_eval
    result["majority_answer"] = crop_eval.get("most_common_answer")
    
    # Step 3: Check consistency score
    consistency_score = crop_eval.get("consistency_score", 0.0)
    
    if consistency_score < 0.75:
        result["status"] = "invalid vqa"
        result["reason"] = f"Low consistency: {consistency_score}"
        return result
    
    # Step 4: High consistency, majority_answer as ground truth
    majority_answer = result["majority_answer"]
    
    # Step 5: Rejection sampling
    bbox_image_path = draw_bbox_on_image(original_image_path, bbox, image_size, args.bbox_output_dir)
    if not bbox_image_path:
        result["qwen8b_eval"] = {
            "error": "Failed to draw bbox image",
            "correct_count": 0
        }
        result["status"] = "invalid vqa"
        result["reason"] = "Failed to draw bbox image for MCQ generation"
        return result
    else:
        result["bbox_image_path"] = bbox_image_path  # Add to result
        
        try:
            question_with_bbox = question + "\n\nOnly focus on the objects inside the red bounding box in the image to answer this question."
            
            # Generate MCQ
            mcq_pack = generate_mcq_with_mllm(
                bbox_image_path=bbox_image_path,
                question=question_with_bbox,
                correct_answer=majority_answer,
                api_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model_name_qwen,
                num_distractors=3
            )
            if not mcq_pack:
                result["status"] = "invalid vqa"
                result["reason"] = "MCQ generation failed"
                return result
            
            question = mcq_pack["mcq_question"]
            majority_answer = mcq_pack["correct_option"]
            result["mcq"] = {
                "mcq_question": mcq_pack["mcq_question"],
                "all_options": mcq_pack["all_options"],
                "correct_option": mcq_pack["correct_option"],
                "distractors": mcq_pack["distractors"],
                "correct_answer_text": result["majority_answer"],  # Keep original majority_answer text
            }

            qwen8b_answers = answer_question_with_qwen8b(
                bbox_image_path,
                question, 
                num_samples=4
            )
            correct_count_8b = check_correct_count(
                qwen8b_answers,
                majority_answer,
                llm_client,
                question
            )
            
            result["qwen8b_eval"] = {
                "answers": qwen8b_answers,
                "correct_count": correct_count_8b,
                "ground_truth": majority_answer
            }
            
            if correct_count_8b >= 2:
                result["status"] = "very easy vqa"
                return result
            
        except Exception as e:
            # If Qwen8B call fails, log error but continue
            result["qwen8b_eval"] = {
                "error": str(e),
                "correct_count": 0
            }
            print(f"Warning: Qwen8B test failed for question '{question}': {str(e)}")
    
    # Step 6: If 8B correct count < 2, mark as valid vqa
    result["status"] = "valid vqa"
    
    return result


def process_single_image_task(line_data, args):
    """Process all valid questions for a single image"""
    original_image_path = line_data.get('image_path')
    if not original_image_path:
        return None
    
    # Load original image base64 from local
    try:
        original_b64 = load_local_image(original_image_path, image_type='base64')
        if not original_b64:
            return None
        
        original_image = load_local_image(original_image_path, image_type='PIL')
        if not original_image:
            return None
        
        image_size = original_image.size  # (width, height)
    except Exception as e:
        print(f"Failed to load original image {original_image_path}: {e}")
        return None
    
    generated_vqa = line_data.get('generated_vqa', [])
    
    results = []
    
    for vqa_item in generated_vqa:
        gen_question_status = vqa_item.get('gen_question_status')
        
        # Only process successfully generated data
        if gen_question_status != 'success':
            continue
        
        crop_path = vqa_item.get('crop_path')
        bbox = vqa_item.get('bbox')
        
        if not crop_path or not bbox:
            continue
        
        vqa_pairs = vqa_item.get('vqa_pairs', [])
        
        # Process each valid question
        for pair in vqa_pairs:
            validation_status = pair.get('validation_status')
            if validation_status != 'passed':
                continue
            
            question = pair.get('question')
            if not question:
                continue
            
            # Process this valid question
            try:
                vqa_result = process_single_vqa(
                    original_image_path,
                    crop_path,  # Pass crop_path directly
                    original_b64,
                    question,
                    bbox,
                    image_size,
                    args
                )
                
                # Add metadata
                vqa_result['original_image_path'] = original_image_path
                vqa_result['crop_path'] = crop_path
                vqa_result['mask_path'] = vqa_item.get('mask_path')
                vqa_result['bbox_image_path'] = vqa_result.get('bbox_image_path')
                results.append(vqa_result)
                
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                results.append({
                    "question": question,
                    "status": "error",
                    "error": str(e),
                    "original_image_path": original_image_path,
                    "crop_path": crop_path,
                    "bbox": bbox,
                    "image_size": image_size
                })
    
    if results:
        return {
            "image_path": original_image_path,
            "vqa_results": results
        }
    
    return None


# ==============================================================================
# Main Function
# ==============================================================================

def main(args):
    global llm_client
    llm_client = OpenAI(
        api_key=args.kimi_api_key,
        base_url=args.kimi_api_url,
    )
    # Create bbox output directory if it doesn't exist
    os.makedirs(args.bbox_output_dir, exist_ok=True)
    print(f"✓ Bbox images will be saved to: {args.bbox_output_dir}")
    
    # 1. Read input file and deduplicate
    print(f"✓ Reading input file: {args.input_file}")
    
    unique_all_lines = []
    seen_input_paths = set()
    duplicate_count = 0
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                path = data.get('image_path')
                
                if path is None:
                    continue
                
                if path not in seen_input_paths:
                    seen_input_paths.add(path)
                    unique_all_lines.append(data)
                else:
                    duplicate_count += 1
            except json.JSONDecodeError:
                continue

    print(f"✓ Total read {len(unique_all_lines) + duplicate_count} records")
    if duplicate_count > 0:
        print(f"  (Removed duplicates from input file: {duplicate_count} records)")
    
    # 2. Resume from checkpoint: read already processed data
    processed_paths = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    path = data.get('image_path')
                    if path:
                        processed_paths.add(path)
                except:
                    continue
        print(f"✓ Resumed from output file, already processed {len(processed_paths)} unique records")
    
    # 3. Filter tasks: only keep data not in output file and not in current task duplicate list
    tasks = [line for line in unique_all_lines if line.get('image_path') not in processed_paths]
    random.shuffle(tasks)
    if not tasks:
        print("✓ No new tasks to process")
        return
    
    print(f"✓ Tasks to process (excluding already processed): {len(tasks)}")
    
    # Statistics
    total_processed = 0
    total_very_easy = 0
    total_valid = 0
    total_invalid = 0
    
    write_lock = threading.Lock()
    
    # Concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor, \
         open(args.output_file, 'a', encoding='utf-8') as f_out:
        
        future_to_task = {
            executor.submit(process_single_image_task, task, args): task.get('image_path')
            for task in tasks
        }
        
        progress_bar = tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc="Processing VQA Validation",
            ncols=140
        )
        
        for future in progress_bar:
            task_id = future_to_task[future]
            
            try:
                result = future.result(timeout=600)  # 10 minutes timeout
                
                if result:
                    # Statistics
                    vqa_results = result.get('vqa_results', [])
                    for vqa in vqa_results:
                        total_processed += 1
                        status = vqa.get('status')
                        if status == 'very easy vqa':
                            total_very_easy += 1
                        elif status == 'valid vqa':
                            total_valid += 1
                        elif status == 'invalid vqa':
                            total_invalid += 1
                    
                    # Write to file
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'v_easy': total_very_easy,
                        'valid': total_valid,
                        'invalid': total_invalid
                    })
                
            except concurrent.futures.TimeoutError:
                print(f"\n⚠ Task {task_id} timeout")
                
            except Exception as exc:
                print(f"\n⚠ Task {task_id} exception: {exc}")
    
    # Final statistics
    total_all_valid = total_very_easy + total_valid
    print("\n" + "="*100)
    print("Processing completed! Final statistics:")
    
    print(f"  Total processed VQA: {total_processed}")
    print(f"  Total Valid VQA: {total_all_valid} ({total_all_valid/total_processed*100:.2f}%)" if total_processed > 0 else "  Total Valid VQA: 0")
    print(f"    ├─ Very Easy VQA: {total_very_easy} ({total_very_easy/total_processed*100:.2f}%)" if total_processed > 0 else "    ├─ Very Easy VQA: 0")
    print(f"    └─ Valid VQA: {total_valid} ({total_valid/total_processed*100:.2f}%)" if total_processed > 0 else "    └─ Valid VQA: 0")
    print(f"  Invalid VQA: {total_invalid} ({total_invalid/total_processed*100:.2f}%)" if total_processed > 0 else "  Invalid VQA: 0")
    print(f"  Bbox images saved to: {args.bbox_output_dir}")
    print("="*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQA Answer Validation Pipeline (Local Image Version)")
    
    # Input/Output
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True,
        help='Input JSONL file path (output from previous VQA generation step)'
    )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True,
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--bbox_output_dir',
        type=str,
        required=True,
        help='Directory to save bbox images'
    )
    
    # Concurrency control
    parser.add_argument(
        '--max_workers', 
        type=int, 
        default=48,
        help='Maximum number of concurrent workers'
    )
    
    # API configuration (all required, no defaults in code)
    parser.add_argument(
        '--api_key',
        type=str,
        required=True,
        help='API key for the service'
    )
    
    parser.add_argument(
        '--api_url',
        type=str,
        required=True,
        help='API base URL'
    )
    
    parser.add_argument(
        '--model_name_qwen',
        type=str,
        default="Qwen3-VL-235B-A22B-Instruct",
        help='Qwen model name to use'
    )
    
    parser.add_argument(
        '--model_name_glm',
        type=str,
        default="GLM-4.5V",
        help='GLM model name to use'
    )

    parser.add_argument(
        '--kimi_api_key',
        type=str,
        required=True,
        help='API key for Kimi LLM service'
    )

    parser.add_argument(
        '--kimi_api_url',
        type=str,
        required=True,
        help='API base URL for Kimi LLM service'
    )
    
    args = parser.parse_args()
    
    main(args)
