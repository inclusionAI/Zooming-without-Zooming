"""
Unified Pipeline: For each image, detect objects, generate bboxes, create questions,
generate & validate answers, and collect results — all in one pass.

Usage:
    python create_vqa.py \
        --api_key "$MLLM_KEY" \
        --api_url "$MLLM_URL" \
        --kimi_api_key "$KIMI_KEY" \
        --kimi_api_url "$KIMI_URL" \
        --image_folders "/path/images/sa1b" \
        --crop_output_dir "/path/images/crops" \
        --bbox_output_dir "/path/images/bbox_images" \
        --output_parquet "validated_vqa.parquet" \
        --output_jsonl "validated_vqa.jsonl"
"""

import base64
from openai import OpenAI
from PIL import Image, ImageDraw
import io
import os
import json
import re
import random
import argparse
import numpy as np
from tqdm import tqdm
from mathruler.grader import extract_boxed_content
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import threading
import torch
import hashlib
from collections import Counter, defaultdict
from datasets import Dataset

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ==============================================================================
# Qwen3-VL-8B Client (local vLLM server for rejection sampling)
# ==============================================================================
mllm_client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:18902/v1",
    timeout=3600
)


# ==============================================================================
# Helper Functions (from step 1)
# ==============================================================================

def pil_image_to_base64(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr = byte_arr.getvalue()
    base64_str = base64.b64encode(byte_arr).decode('utf-8')
    return base64_str


def get_image_dimensions(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
    return width, height


def extract_objects_from_response(response_text):
    pattern1 = r'\d+\.\s*([^\n]+)'
    matches1 = re.findall(pattern1, response_text)
    pattern2 = r'(?:objects?|items?)[:\s]+([^\n\.]+)'
    matches2 = re.findall(pattern2, response_text, re.IGNORECASE)
    objects = []
    if matches1 and len(matches1) > 0:
        objects = [obj.strip().rstrip(',.;:') for obj in matches1]
    elif matches2 and len(matches2) > 0:
        objects = [obj.strip() for obj in matches2[0].split(',')]
    objects = [obj for obj in objects if obj and len(obj) > 1 and len(obj) < 50]
    return objects


def get_images_from_folder(folder_path):
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file.lower())[1] in supported_formats:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return sorted(image_paths)


def get_images_from_folders(folder_list):
    all_image_paths = []
    folder_stats = {}
    for folder_path in folder_list:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder does not exist: {folder_path}")
            folder_stats[folder_path] = 0
            continue
        if not os.path.isdir(folder_path):
            print(f"Warning: Not a directory: {folder_path}")
            folder_stats[folder_path] = 0
            continue
        images = get_images_from_folder(folder_path)
        folder_stats[folder_path] = len(images)
        all_image_paths.extend(images)
        print(f"Found {len(images)} images in: {folder_path}")
    return all_image_paths, folder_stats


# ==============================================================================
# Helper Functions (from step 2)
# ==============================================================================

def parse_bbox_string(bbox_str):
    try:
        bbox_str = bbox_str.strip("()")
        coords = [float(x) for x in bbox_str.split(",")]
        if len(coords) == 4:
            return tuple(coords)
    except:
        pass
    return None


def extract_json_from_text(text):
    if not text:
        return []
    json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_block_match:
        content = json_block_match.group(1).strip()
    else:
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            content = text[start_idx:end_idx + 1]
        else:
            content = text.strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return []


def is_crop_valid(original_image, bbox):
    w, h = original_image.size
    try:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            return False, "skipped_filter_bbox (invalid bbox)"
        object_area = crop_w * crop_h
        area_ratio = object_area / (w * h)
        if area_ratio >= 0.1:
            return False, f"skipped_filter_mask_ratio (is {area_ratio:.2%})"
        return True, "valid"
    except Exception as e:
        return False, f"skipped_filter_bbox (error: {e})"


def save_crop_image_to_local(pil_image, crop_output_dir, filename):
    os.makedirs(crop_output_dir, exist_ok=True)
    full_path = os.path.join(crop_output_dir, filename)
    pil_image.save(full_path, format='JPEG', quality=95)
    return full_path


def get_vqa_generation_prompt(seed_questions):
    sample_size = min(15, len(seed_questions))
    selected_examples = random.sample(seed_questions, sample_size)
    examples_str = "\n".join([f"- {q}" for q in selected_examples])
    return f"""You are an expert specialist in generating Visual Question Answering (VQA) datasets. Your task is to generate three high-quality and valid questions based solely on the provided image.

**Reference Examples (use these for inspiration on questioning angles):**
{examples_str}

**Core Generation Rules:**
1. **Image-based Questions:** All questions must be answerable by examining the image provided. The answer to each question must be identical, accurate, and concise. It can be a **short, factual, and concrete string** (e.g., a number, a noun, or text).
2. **Content Relevance:** Question types include, but are not limited to:
    * **Object Identification:** Identify the exact sub-component or item. (e.g., 'What is the person holding in their hand?' Answer: 'Apple').
    * **OCR:** Recognizing text within the image.
3. **Quality Control:** If the image is of such low quality that it is impossible to generate meaningful questions, return an empty JSON list: `[]`.
5. **Diversity of Questions:** Aim for a diverse range of questions. This includes counting, spatial relationships, scene recognition, anomaly detection, shape, material, structure, etc.

**Please carefully observe the image and generate your response in the following JSON format:**

```json
[
  {{"question": "Question 1"}},
  {{"question": "Question 2"}},
  {{"question": "Question 3"}}
]
```"""


# ==============================================================================
# Helper Functions (from step 3)
# ==============================================================================

def load_local_image(image_path, image_type='PIL', max_retries=2):
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
                time.sleep(2)
            else:
                print(f"Failed to load image {image_path}: {e}")
                raise e
    return None


def expand_bbox(bbox, image_size, expand_ratio=0.05):
    x1, y1, x2, y2 = bbox
    width, height = image_size
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    expand_w = bbox_width * expand_ratio
    expand_h = bbox_height * expand_ratio
    new_x1 = max(0, x1 - expand_w)
    new_y1 = max(0, y1 - expand_h)
    new_x2 = min(width, x2 + expand_w)
    new_y2 = min(height, y2 + expand_h)
    return [new_x1, new_y1, new_x2, new_y2]


def draw_bbox_on_image(image_path, bbox, image_size, output_dir):
    try:
        image = load_local_image(image_path, image_type='PIL')
        if image is None:
            return None
    except Exception as e:
        print(f"Failed to load image for bbox drawing: {image_path}, error: {e}")
        return None

    expanded_bbox = expand_bbox(bbox, image_size, 0.05)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = expanded_bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    os.makedirs(output_dir, exist_ok=True)
    hash_str = hashlib.md5(f"{image_path}_{bbox}_expand_0.05".encode()).hexdigest()
    output_path = os.path.join(output_dir, f"{hash_str}.png")
    image.save(output_path, "PNG")
    return output_path


# ==============================================================================
# Generic API Call
# ==============================================================================

def call_api(payload, api_url, api_key, model_name, stream=False,
             max_retries=3000, base_retry_delay=1.0):
    for retry_count in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload["model"] = model_name
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120, stream=stream)
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


# ==============================================================================
# Step 1: Object Detection + SAM3 BBox Generation
# ==============================================================================

def detect_objects_and_bboxes(img_path, api_url, api_key, sam3_processor, sam3_lock, model_name="Qwen3-VL-235B-A22B-Instruct"):
    """
    Step 1: Use Qwen3-VL to list objects, then SAM3 to get bboxes.
    Returns list of bbox_results: [{detected_object, bbox, sam3_score, success}, ...]
    """
    width, height = get_image_dimensions(img_path)

    object_detection_prompt = f"""You are an expert at analyzing images and identifying objects.

Your task:
1. Carefully observe the given image
2. List visible distinct and clear objects (such as people, animals, and things) you can see in the image. Only list objects that are clearly identifiable and only appear once. Avoid guessing blurry or ambiguous shapes.

Please provide a numbered list of objects you can identify. The example output format is:
\\boxed{{1. person
2. coffee cup
3. laptop
4. plant
5. clock}}"""

    max_attempts = 200
    for attempts in range(max_attempts):
        try:
            with open(img_path, "rb") as f:
                image_encoded = base64.b64encode(f.read()).decode("utf-8")

            base64_image = f"data:image;base64,{image_encoded}"
            message_mm = {"type": "image_url", "image_url": {"url": base64_image}}

            messages = [{
                "role": "user",
                "content": [message_mm, {"type": "text", "text": object_detection_prompt}]
            }]

            data = {
                'stream': False,
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            if response.status_code in (429, 503):
                time.sleep(1)
                continue
            response.raise_for_status()

            response_data = response.json()
            full_response = response_data['choices'][0]['message']['content']
            res_content = extract_boxed_content(full_response)
            objects = extract_objects_from_response(res_content)

            if not objects or len(objects) <= 2:
                return [], width, height

            # SAM3 bbox extraction
            image = Image.open(img_path).convert('RGB')
            bbox_results = []

            for detected_object in objects:
                try:
                    with sam3_lock:
                        inference_state = sam3_processor.set_image(image)
                        inference_state = sam3_processor.set_text_prompt(
                            state=inference_state,
                            prompt=detected_object
                        )
                        masks = inference_state.get("masks")
                        boxes = inference_state.get("boxes")
                        scores = inference_state.get("scores")

                        if boxes is None or len(boxes) == 0:
                            bbox_results.append({
                                "detected_object": detected_object,
                                "bbox": None, "sam3_score": 0.0, "success": False,
                                "error": "SAM3 found no boxes"
                            })
                            continue
                        if len(boxes) > 1:
                            bbox_results.append({
                                "detected_object": detected_object,
                                "bbox": None, "sam3_score": 0.0, "success": False,
                                "error": f"Multiple instances found: {len(boxes)}"
                            })
                            continue

                        best_box = boxes[0]
                        best_score = scores[0].item() if scores is not None and len(scores) > 0 else 0.0
                        if isinstance(best_box, torch.Tensor):
                            best_box = best_box.cpu().tolist()

                        bbox_str = f"({best_box[0]:.1f},{best_box[1]:.1f},{best_box[2]:.1f},{best_box[3]:.1f})"
                        bbox_results.append({
                            "detected_object": detected_object,
                            "bbox": bbox_str,
                            "sam3_score": best_score,
                            "success": True,
                            "error": None
                        })
                except Exception as sam_error:
                    bbox_results.append({
                        "detected_object": detected_object,
                        "bbox": None, "sam3_score": 0.0, "success": False,
                        "error": str(sam_error)
                    })

            return bbox_results, width, height

        except Exception as e:
            time.sleep(1)
            continue

    return [], width, height


# ==============================================================================
# Step 2: VQA Question Generation & Validation
# ==============================================================================

def generate_vqa(crop_b64_image, seed_questions, api_url, api_key, model_name):
    prompt = get_vqa_generation_prompt(seed_questions)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "This is the Image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64_image}"}},
            {"type": "text", "text": prompt}
        ]
    }]
    payload = {"stream": False, "messages": messages, "temperature": 0.9, "top_p": 0.8}
    resp_json = call_api(payload, api_url=api_url, api_key=api_key, model_name=model_name, stream=False, max_retries=3000)
    if not resp_json:
        return None
    try:
        content = resp_json["choices"][0]["message"]["content"]
        return extract_json_from_text(content)
    except:
        return None


def validate_single_vqa(original_b64_image, crop_b64_image, question, api_url, api_key, model_name):
    validation_prompt = f"""
You are an expert at validating whether a question is appropriate for Visual Question Answering (VQA) under a crop-consistency setting.

You will be shown two images:
- Image 1: The original image (full context)
- Image 2: A cropped region taken from Image 1

Question: {question}

Your task: Determine whether the question is VALID according to ALL criteria below. If ANY criterion fails, the question is NOT valid.

CRITERIA (ALL must be satisfied):
1) Crop-answerable:
   - The question MUST be answerable using both two images.
   - If answering requires information outside the crop, mark INVALID.

2) Unique and unambiguous (in the original image):
   - In Image 1, there must be exactly ONE clearly correct answer.
   - If multiple instances/objects in the original image could produce different correct answers, mark INVALID.

3) Consistent with the original image:
   - The answer derived from Image 2 MUST match the answer that would be obtained from Image 1.
   - If Image 1 allows additional valid answers or changes the interpretation, mark INVALID even if Image 2 looks unambiguous.

4) Clear question:
   - The question must specify the target unambiguously.
   - Avoid unclear references like "it", "this", "the object" when multiple candidates exist.

OUTPUT FORMAT (strict):
VALID: Yes/No
REASON: A brief explanation referencing which criterion/criteria passed or failed.

Now please reason step by step, and then evaluate the given Question using the rules above.
"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "This is the Original Image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64_image}"}},
            {"type": "text", "text": "This is the Cropped Image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64_image}"}},
            {"type": "text", "text": validation_prompt}
        ]
    }]
    payload = {"stream": False, "messages": messages, "temperature": 0.3, "top_p": 0.9}
    resp_json = call_api(payload, api_url=api_url, api_key=api_key, model_name=model_name, stream=False, max_retries=3000)
    if not resp_json:
        return {"is_valid": False, "reason": "API call failed"}
    try:
        full_content = resp_json["choices"][0]["message"]["content"]
    except:
        return {"is_valid": False, "reason": "Response parsing failed"}

    is_valid = False
    reason = "No reason provided."
    valid_match = re.search(r'VALID:\s*(Yes|No)', full_content, re.IGNORECASE)
    if valid_match:
        is_valid = valid_match.group(1).upper() == "YES"
    reason_match = re.search(r'REASON:\s*(.+?)(?:\n\n|\Z)', full_content, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    return {"is_valid": is_valid, "reason": reason}


def validate_vqa_pairs(original_b64_image, crop_b64_image, vqa_pairs, api_url, api_key, model_name, max_validation_workers=6):
    if not vqa_pairs:
        return []
    temp_results = [None] * len(vqa_pairs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_validation_workers) as executor:
        future_to_idx = {}
        for idx, item in enumerate(vqa_pairs):
            question = item.get("question", "")
            if not question:
                temp_results[idx] = {"is_valid": False, "reason": "Empty question"}
                continue
            future = executor.submit(validate_single_vqa, original_b64_image, crop_b64_image, question, api_url, api_key, model_name)
            future_to_idx[future] = idx
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                temp_results[idx] = future.result()
            except Exception as e:
                temp_results[idx] = {"is_valid": False, "reason": f"Validation exception: {str(e)}"}
    return temp_results


# ==============================================================================
# Step 3: Answer Generation, Consistency Check, MCQ, Rejection Sampling
# ==============================================================================

def answer_question_with_mllm(image_b64, question, api_url, api_key, num_samples=8, model_name="Qwen3-VL-235B-A22B-Instruct"):
    answers = []
    for _ in range(num_samples):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": f"Question: {question}\n\nPlease reason step by step, and then put your answer within \\boxed{{}}."}
            ]
        }]
        payload = {"stream": False, "messages": messages, "temperature": 0.7, "top_p": 0.9}
        resp_json = call_api(payload, api_url=api_url, api_key=api_key, model_name=model_name, stream=False, max_retries=3000)
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
    answers = []
    for _ in range(num_samples):
        try:
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
            messages = [{
                "role": "user",
                "content": [
                    message_mm,
                    {"type": "text", "text": question + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}
                ]
            }]
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


def evaluate_consistency_with_llm(question, answers, client):
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
- They represent different interpretations of the question

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
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=20480,
            )
            response = completion.choices[0].message.content
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
                "success": True, "error": None
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "most_common_answer": None, "consistency_count": 0,
                    "consistency_score": 0.0, "success": False, "error": str(e)
                }
            time.sleep(2)


def check_correct_count(answers, ground_truth, client, question):
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
- Different phrasings of the same answer
- Slight variations in description that refer to the same entity
- Minor differences in precision that don't change the core answer

Do NOT consider them equivalent if:
- They refer to different objects, numbers, or concepts

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
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3, max_tokens=20480,
            )
            response = completion.choices[0].message.content
            for line in response.split('\n'):
                if 'Correct Count:' in line:
                    count_str = line.split('Correct Count:')[1].strip()
                    numbers = re.findall(r'\d+', count_str)
                    if numbers:
                        return int(numbers[0])
            return 0
        except Exception as e:
            if attempt == max_retries - 1:
                return 0
            time.sleep(2)
    return 0


def generate_mcq_with_mllm(bbox_image_path, question, correct_answer, api_url, api_key,
                            model_name, num_distractors=3, max_retries=100):
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
            payload = {"stream": False, "messages": messages, "temperature": 0.8, "max_tokens": 1024}
            resp_json = call_api(payload, api_url=api_url, api_key=api_key, model_name=model_name, stream=False, max_retries=3000)
            if not resp_json:
                time.sleep(1)
                continue
            content = resp_json["choices"][0]["message"]["content"].strip()
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
                question + "\n\n" + options_text +
                "\n\nAnswer with the option's letter from the given choices."
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
# Core: Process Single VQA (Step 3 answer pipeline for one question)
# ==============================================================================

def process_single_vqa_answer(original_image_path, crop_image_path, original_b64, question,
                               bbox, image_size, args, llm_client):
    """
    Full answer pipeline for a single validated question:
    1. Answer on crop (8 samples from 2 models)
    2. Consistency check
    3. If consistent: MCQ generation + rejection sampling with 8B
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

    # Step 1: Answer on crop image
    try:
        crop_b64 = load_local_image(crop_image_path, image_type='base64')
        if not crop_b64:
            result["status"] = "invalid vqa"
            result["error"] = "Failed to load crop image"
            return result

        qwen_answers = answer_question_with_mllm(
            crop_b64, question, args.api_url, args.api_key,
            num_samples=4, model_name=args.model_name_qwen
        )
        glm_answers = answer_question_with_mllm(
            crop_b64, question, args.api_url, args.api_key,
            num_samples=4, model_name=args.model_name_glm
        )
        crop_answers = qwen_answers + glm_answers
    except Exception as e:
        result["status"] = "invalid vqa"
        result["error"] = f"Crop answering failed: {str(e)}"
        return result

    # Step 2: Evaluate consistency
    crop_eval = evaluate_consistency_with_llm(question, crop_answers, client=llm_client)
    result["crop_eval"] = crop_eval
    result["majority_answer"] = crop_eval.get("most_common_answer")

    consistency_score = crop_eval.get("consistency_score", 0.0)
    if consistency_score < 0.75:
        result["status"] = "invalid vqa"
        result["reason"] = f"Low consistency: {consistency_score}"
        return result

    majority_answer = result["majority_answer"]

    # Step 3: MCQ generation + rejection sampling
    bbox_image_path = draw_bbox_on_image(original_image_path, bbox, image_size, args.bbox_output_dir)
    if not bbox_image_path:
        result["status"] = "invalid vqa"
        result["reason"] = "Failed to draw bbox image"
        return result

    result["bbox_image_path"] = bbox_image_path

    try:
        question_with_bbox = question + "\n\nOnly focus on the objects inside the red bounding box in the image to answer this question."

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

        mcq_question = mcq_pack["mcq_question"]
        mcq_correct_option = mcq_pack["correct_option"]
        result["mcq"] = {
            "mcq_question": mcq_pack["mcq_question"],
            "all_options": mcq_pack["all_options"],
            "correct_option": mcq_pack["correct_option"],
            "distractors": mcq_pack["distractors"],
            "correct_answer_text": majority_answer,
        }

        qwen8b_answers = answer_question_with_qwen8b(
            bbox_image_path, mcq_question, num_samples=4
        )
        correct_count_8b = check_correct_count(
            qwen8b_answers, mcq_correct_option, llm_client, mcq_question
        )

        result["qwen8b_eval"] = {
            "answers": qwen8b_answers,
            "correct_count": correct_count_8b,
            "ground_truth": mcq_correct_option
        }

        if correct_count_8b >= 2:
            result["status"] = "very easy vqa"
            return result

    except Exception as e:
        result["qwen8b_eval"] = {"error": str(e), "correct_count": 0}
        print(f"Warning: Qwen8B test failed: {str(e)}")

    result["status"] = "valid vqa"
    return result


# ==============================================================================
# Core: Process Single Image End-to-End
# ==============================================================================

def process_single_image_e2e(img_path, img_idx, args, sam3_processor, sam3_lock, seed_questions, llm_client):
    """
    End-to-end pipeline for a single image:
    Step 1: Detect objects + get SAM3 bboxes
    Step 2: For each valid bbox, generate questions + validate
    Step 3: For each valid question, generate answers + MCQ + rejection sampling
    
    Returns a result dict or None.
    """
    print(f"\n[Image {img_idx}] Processing: {img_path}")

    # ========== Step 1: Object detection + SAM3 bboxes ==========
    bbox_results, width, height = detect_objects_and_bboxes(
        img_path, args.api_url, args.api_key,
        sam3_processor, sam3_lock,
        model_name=args.model_name_qwen
    )

    if not bbox_results:
        print(f"  [Image {img_idx}] No objects detected or no bboxes found.")
        return None

    successful_bboxes = [r for r in bbox_results if r.get("success")]
    print(f"  [Image {img_idx}] {len(successful_bboxes)}/{len(bbox_results)} successful bboxes")

    # Load original image once
    try:
        original_pil_image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"  [Image {img_idx}] Failed to load image: {e}")
        return None

    image_size = original_pil_image.size
    buffered = io.BytesIO()
    original_pil_image.save(buffered, format="JPEG")
    original_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    all_vqa_results = []

    for bbox_item in bbox_results:
        # Filter: must be successful and score >= 0.75
        if not bbox_item.get("success"):
            continue
        sam3_score = bbox_item.get("sam3_score", 0)
        if sam3_score < 0.75:
            continue

        detected_object = bbox_item.get("detected_object", "unknown")
        bbox_str = bbox_item.get("bbox", "")
        bbox = parse_bbox_string(bbox_str)
        if not bbox:
            continue

        # Validate crop
        is_valid, reason = is_crop_valid(original_pil_image, bbox)
        if not is_valid:
            print(f"    [{detected_object}] Crop invalid: {reason}")
            continue

        # ========== Step 2: Crop, generate questions, validate ==========
        try:
            x1, y1, x2, y2 = bbox
            crop_pil_image = original_pil_image.crop((x1, y1, x2, y2))
        except Exception as e:
            print(f"    [{detected_object}] Crop failed: {e}")
            continue

        # 2x upscale
        orig_w, orig_h = crop_pil_image.size
        resized_crop = crop_pil_image.resize((orig_w * 2, orig_h * 2), Image.LANCZOS)

        # Save crop
        filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{detected_object}_{sam3_score:.4f}.jpg"
        try:
            crop_local_path = save_crop_image_to_local(resized_crop, args.crop_output_dir, filename)
        except Exception as e:
            print(f"    [{detected_object}] Failed to save crop: {e}")
            continue

        # Crop to base64
        buffered_crop = io.BytesIO()
        resized_crop.save(buffered_crop, format="JPEG")
        crop_b64 = base64.b64encode(buffered_crop.getvalue()).decode('utf-8')

        # Generate VQA questions (crop only)
        candidate_vqa_pairs = generate_vqa(
            crop_b64, seed_questions,
            args.api_url, args.api_key,
            args.model_name_qwen
        )

        if not candidate_vqa_pairs or not isinstance(candidate_vqa_pairs, list):
            print(f"    [{detected_object}] No questions generated")
            continue

        # Validate questions (original + crop)
        validation_results = validate_vqa_pairs(
            original_b64, crop_b64,
            candidate_vqa_pairs,
            args.api_url, args.api_key,
            args.model_name_qwen,
            max_validation_workers=args.validation_workers
        )

        # Filter valid questions
        valid_questions = []
        for j, pair in enumerate(candidate_vqa_pairs):
            if j < len(validation_results) and validation_results[j].get("is_valid", False):
                valid_questions.append(pair.get("question", ""))

        if not valid_questions:
            print(f"    [{detected_object}] No valid questions after validation")
            continue

        print(f"    [{detected_object}] {len(valid_questions)} valid question(s)")

        # ========== Step 3: For each valid question, run answer pipeline ==========
        for question in valid_questions:
            if not question:
                continue

            try:
                vqa_result = process_single_vqa_answer(
                    img_path, crop_local_path, original_b64,
                    question, list(bbox), image_size,
                    args, llm_client
                )
                vqa_result['original_image_path'] = img_path
                vqa_result['crop_path'] = crop_local_path
                vqa_result['detected_object'] = detected_object
                vqa_result['sam3_score'] = sam3_score

                status = vqa_result.get('status', 'unknown')
                print(f"      Q: \"{question[:60]}...\" → {status}")

                all_vqa_results.append(vqa_result)

            except Exception as e:
                print(f"      Error for question '{question[:50]}': {e}")
                all_vqa_results.append({
                    "question": question,
                    "status": "error",
                    "error": str(e),
                    "original_image_path": img_path,
                    "crop_path": crop_local_path,
                    "bbox": list(bbox),
                    "image_size": image_size,
                    "detected_object": detected_object,
                    "sam3_score": sam3_score
                })

    if all_vqa_results:
        return {
            "image_path": img_path,
            "image_index": img_idx,
            "image_width": width,
            "image_height": height,
            "vqa_results": all_vqa_results
        }

    return None


# ==============================================================================
# Step 4: Convert results to parquet
# ==============================================================================

def convert_results_to_parquet(jsonl_path, output_parquet_path):
    """Read the output JSONL and convert valid VQA to parquet for training."""
    print(f"\nConverting {jsonl_path} to parquet...")

    valid_vqas = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing JSONL for parquet"):
            try:
                data = json.loads(line)
                vqa_results = data.get('vqa_results', [])
                for vqa in vqa_results:
                    status = vqa.get('status')
                    if status not in ['valid vqa']:
                        continue
                    mcq = vqa.get('mcq')
                    if not mcq:
                        continue
                    majority_answer = vqa.get('majority_answer')
                    if majority_answer is None or len(majority_answer) == 0:
                        continue
                    bbox_image_path = vqa.get('bbox_image_path')
                    original_image_path = vqa.get('original_image_path')
                    if not bbox_image_path or not original_image_path:
                        continue
                    valid_vqas.append({
                        'bbox_image_path': bbox_image_path,
                        'original_image_path': original_image_path,
                        'problem': mcq.get('mcq_question'),
                        'answer': mcq.get('correct_option'),
                        'bbox': vqa.get('bbox'),
                        'status': status
                    })
            except:
                continue

    print(f"✓ Extracted {len(valid_vqas)} valid VQA data")

    if not valid_vqas:
        print("No valid VQA data to convert.")
        return

    images = []
    problems = []
    answers = []
    bboxes = []
    statuses = []

    for vqa in tqdm(valid_vqas, desc="Building dataset"):
        images.append([vqa['bbox_image_path']])
        problems.append(vqa['problem'])
        answers.append(vqa['answer'])
        bboxes.append(vqa['bbox'])
        statuses.append(vqa['status'])

    data = {
        'images': images,
        'problem': problems,
        'answer': answers,
        'bbox': bboxes,
        'status': statuses
    }

    dataset = Dataset.from_dict(data)
    print(f"\nTotal training samples: {len(dataset)}")
    dataset.to_parquet(output_parquet_path)
    print(f"✓ Parquet saved to: {output_parquet_path}")

    if len(dataset) > 0:
        print(f"\nTraining set example:")
        print(dataset[0])


# ==============================================================================
# Resume Support
# ==============================================================================

def load_completed_from_jsonl(output_jsonl):
    """Load already-processed image paths from JSONL for resume."""
    if not os.path.exists(output_jsonl):
        return set()
    completed = set()
    try:
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    img_path = result.get('image_path')
                    if img_path:
                        completed.add(img_path)
                except json.JSONDecodeError:
                    continue
        print(f"✓ Loaded {len(completed)} completed images from {output_jsonl}")
    except Exception as e:
        print(f"Warning: Error loading existing results: {e}")
    return completed


def append_result_to_jsonl(result, output_jsonl, file_lock):
    """Thread-safe append result to JSONL file."""
    with file_lock:
        with open(output_jsonl, 'a', encoding='utf-8') as f:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified VQA Pipeline: detect objects → generate bboxes → create questions → generate answers → parquet'
    )

    # Image input
    parser.add_argument('--image_folders', type=str, nargs='+', required=True,
                        help='Paths to folders containing images (can specify multiple)')

    # Output
    parser.add_argument('--output_jsonl', type=str, default='validated_vqa.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--output_parquet', type=str, default='validated_vqa.parquet',
                        help='Output parquet file path')
    parser.add_argument('--crop_output_dir', type=str, required=True,
                        help='Directory to save cropped images')
    parser.add_argument('--bbox_output_dir', type=str, required=True,
                        help='Directory to save bbox images')

    # API configuration
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for MLLM service (e.g., Qwen3-VL)')
    parser.add_argument('--api_url', type=str, required=True,
                        help='API URL for MLLM service')
    parser.add_argument('--kimi_api_key', type=str, required=True,
                        help='API key for Kimi LLM service')
    parser.add_argument('--kimi_api_url', type=str, required=True,
                        help='API URL for Kimi LLM service')

    # Model names
    parser.add_argument('--model_name_qwen', type=str, default='Qwen3-VL-235B-A22B-Instruct',
                        help='Qwen model name')
    parser.add_argument('--model_name_glm', type=str, default='GLM-4.5V',
                        help='GLM model name')

    # Concurrency
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Max concurrent image-level workers')
    parser.add_argument('--validation_workers', type=int, default=6,
                        help='Max concurrent validation workers per bbox')

    # Image selection
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max number of images to process (default: all)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for image selection')

    # Seed questions
    parser.add_argument('--seed_json', type=str, default='./seed.json',
                        help='Seed questions JSON file path')

    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing JSONL output file')

    args = parser.parse_args()

    # ========== Print banner ==========
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         Unified VQA Pipeline (End-to-End per Image)         ║
╟──────────────────────────────────────────────────────────────╢
║  Image workers:       {args.max_workers:4d}                                  ║
║  Validation workers:  {args.validation_workers:4d}                                  ║
║  Qwen model:          {args.model_name_qwen:40s}║
║  GLM model:           {args.model_name_glm:40s}║
║  API URL:             {args.api_url:40s}║
║  Crop output:         {args.crop_output_dir:40s}║
║  Bbox output:         {args.bbox_output_dir:40s}║
║  Output JSONL:        {args.output_jsonl:40s}║
║  Output parquet:      {args.output_parquet:40s}║
╚══════════════════════════════════════════════════════════════╝
""")

    # ========== Load SAM3 model ==========
    print("Loading SAM3 model...")
    try:
        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)
        sam3_lock = threading.Lock()
        print("✓ SAM3 model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading SAM3 model: {e}")
        print("Please make sure sam3 is properly installed and sam3.pt is available.")
        exit(1)

    # ========== Load LLM client (Kimi) ==========
    global llm_client
    llm_client = OpenAI(
        api_key=args.kimi_api_key,
        base_url=args.kimi_api_url,
    )
    print("✓ Kimi LLM client initialized")

    # ========== Load seed questions ==========
    print(f"Loading seed questions from: {args.seed_json}")
    with open(args.seed_json, "r", encoding="utf-8") as f:
        seed_questions = json.load(f)
    print(f"✓ Loaded {len(seed_questions)} seed questions")

    # ========== Create output directories ==========
    os.makedirs(args.crop_output_dir, exist_ok=True)
    os.makedirs(args.bbox_output_dir, exist_ok=True)

    # ========== Discover images ==========
    print(f"\nDiscovering images from {len(args.image_folders)} folder(s)...")
    print("=" * 60)
    all_image_paths, folder_stats = get_images_from_folders(args.image_folders)
    print("=" * 60)
    print(f"Total images found: {len(all_image_paths)}")
    for folder, count in folder_stats.items():
        print(f"  {folder}: {count} images")

    if len(all_image_paths) == 0:
        print("✗ No images found!")
        exit(1)

    # ========== Validate and filter by resolution ==========
    print("\nValidating images and filtering by resolution...")
    MIN_W, MIN_H = 1200, 1200
    valid_image_paths = []
    invalid_count = 0
    filtered_small_count = 0

    for img_path in tqdm(all_image_paths, desc="Validating images"):
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                w, h = img.size
            if w <= MIN_W or h <= MIN_H:
                filtered_small_count += 1
                continue
            valid_image_paths.append(img_path)
        except Exception as e:
            invalid_count += 1
            tqdm.write(f"Warning: Invalid image {img_path}: {str(e)}")

    print(f"Valid images (>{MIN_W}x{MIN_H}): {len(valid_image_paths)}")
    print(f"Filtered (too small): {filtered_small_count}")
    print(f"Invalid: {invalid_count}")

    # ========== Random selection ==========
    if args.max_images is not None and args.max_images < len(valid_image_paths):
        print(f"\nRandomly selecting {args.max_images} images (seed={args.random_seed})...")
        random.seed(args.random_seed)
        valid_image_paths = random.sample(valid_image_paths, args.max_images)
        print(f"Selected {len(valid_image_paths)} images")

    # ========== Resume ==========
    completed_paths = set()
    if args.resume:
        completed_paths = load_completed_from_jsonl(args.output_jsonl)

    # ========== Prepare tasks ==========
    tasks = []
    skipped = 0
    for img_idx, img_path in enumerate(valid_image_paths):
        if img_path in completed_paths:
            skipped += 1
            continue
        tasks.append((img_path, img_idx))

    print(f"\nTotal images: {len(valid_image_paths)}")
    print(f"Already completed: {skipped}")
    print(f"Remaining: {len(tasks)}")

    if not tasks:
        print("✓ All images already processed!")
        # Still convert to parquet
        if os.path.exists(args.output_jsonl):
            convert_results_to_parquet(args.output_jsonl, args.output_parquet)
        exit(0)

    # ========== Process images ==========
    file_lock = threading.Lock()
    processed_count = 0
    total_valid = 0
    total_very_easy = 0
    total_invalid = 0

    def worker_fn(task_tuple):
        img_path, img_idx = task_tuple
        return process_single_image_e2e(
            img_path, img_idx, args,
            sam3_processor, sam3_lock,
            seed_questions, llm_client
        )

    print(f"\nStarting processing with {args.max_workers} workers...")
    print(f"Results will be streamed to: {args.output_jsonl}\n")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(worker_fn, t): t for t in tasks}

        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for future in as_completed(future_to_task):
                task_tuple = future_to_task[future]
                img_path, img_idx = task_tuple

                try:
                    result = future.result(timeout=1800)  # 30 min timeout per image

                    if result:
                        # Write to JSONL
                        append_result_to_jsonl(result, args.output_jsonl, file_lock)

                        # Update statistics
                        vqa_results = result.get('vqa_results', [])
                        for vqa in vqa_results:
                            status = vqa.get('status')
                            if status == 'very easy vqa':
                                total_very_easy += 1
                            elif status == 'valid vqa':
                                total_valid += 1
                            elif status == 'invalid vqa':
                                total_invalid += 1

                    processed_count += 1
                    pbar.set_postfix({
                        'valid': total_valid,
                        'v_easy': total_very_easy,
                        'invalid': total_invalid
                    })
                    pbar.update(1)

                except concurrent.futures.TimeoutError:
                    print(f"\n⚠ Timeout for image {img_path}")
                    processed_count += 1
                    pbar.update(1)

                except Exception as exc:
                    print(f"\n⚠ Exception for image {img_path}: {exc}")
                    processed_count += 1
                    pbar.update(1)

    # ========== Final Statistics ==========
    total_all_valid = total_very_easy + total_valid
    total_processed_vqa = total_very_easy + total_valid + total_invalid

    print("\n" + "=" * 80)
    print("Pipeline completed! Final statistics:")
    print(f"  Images processed: {processed_count}")
    if total_processed_vqa > 0:
        print(f"  Total VQA processed: {total_processed_vqa}")
        print(f"  Total Valid VQA: {total_all_valid} ({total_all_valid/total_processed_vqa*100:.2f}%)")
        print(f"    ├─ Very Easy VQA: {total_very_easy} ({total_very_easy/total_processed_vqa*100:.2f}%)")
        print(f"    └─ Valid VQA: {total_valid} ({total_valid/total_processed_vqa*100:.2f}%)")
        print(f"  Invalid VQA: {total_invalid} ({total_invalid/total_processed_vqa*100:.2f}%)")
    else:
        print(f"  Total VQA processed: 0")
    print(f"  JSONL output: {args.output_jsonl}")
    print(f"  Crop images: {args.crop_output_dir}")
    print(f"  Bbox images: {args.bbox_output_dir}")
    print("=" * 80)

    # ========== Step 4: Convert to parquet ==========
    if os.path.exists(args.output_jsonl):
        convert_results_to_parquet(args.output_jsonl, args.output_parquet)
    else:
        print("No JSONL output found, skipping parquet conversion.")

    print("\n✓ All done!")


if __name__ == '__main__':
    main()



