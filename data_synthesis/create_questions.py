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
import random
import threading
import shutil
import queue
from collections import defaultdict
import re


# ==============================================================================
# Configuration
# ==============================================================================


# ==============================================================================
# Helper Functions
# ==============================================================================

def parse_bbox_string(bbox_str):
    """Parse bbox string, format like "(519.6,5.5,1854.7,2596.9)" """
    try:
        # Remove parentheses and split
        bbox_str = bbox_str.strip("()")
        coords = [float(x) for x in bbox_str.split(",")]
        if len(coords) == 4:
            return tuple(coords)
    except:
        pass
    return None


def extract_json_from_text(text):
    """Extract JSON content from model response string"""
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


def is_crop_valid(original_image: Image.Image, bbox: tuple) -> (bool, str):
    """Validate if crop region is valid"""
    w, h = original_image.size
    
    try:
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within image bounds
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


def encode_pil_image_to_base64(pil_image: Image.Image, image_format: str = "PNG") -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format=image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def save_crop_image_to_local(pil_image: Image.Image, crop_output_dir: str, filename: str) -> str:
    """
    Save cropped image to local directory
    
    Args:
        pil_image: PIL Image object to save
        crop_output_dir: Output directory for cropped images
        filename: Filename for the saved image
    
    Returns:
        Full path to saved image file
    """
    # Create output directory if it doesn't exist
    os.makedirs(crop_output_dir, exist_ok=True)
    
    # Full path for the saved image
    full_path = os.path.join(crop_output_dir, filename)
    
    # Save image
    pil_image.save(full_path, format='JPEG', quality=95)
    
    return full_path


def get_vqa_generation_prompt(seed_questions):
    """Generate VQA questions prompt based on crop image only"""
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
# API Call
# ==============================================================================

def call_api(payload: dict, api_url: str, api_key: str, model_name: str, stream: bool = False,
             max_retries: int = 3000, base_retry_delay: float = 1.0):
    """
    Generic API call with automatic retry
    payload: directly the body for requests.post(json=payload)
    """
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
                stream=stream
            )

            # Handle rate limiting/overload
            if resp.status_code in (429, 503):
                time.sleep(base_retry_delay)
                continue

            # Handle other non-200 status
            if resp.status_code != 200:
                time.sleep(base_retry_delay)
                continue

            data = resp.json()
            return data

        except Exception as e:
            time.sleep(base_retry_delay)

    return None


# ==============================================================================
# VQA Generation Function (Modified - Using crop image only)
# ==============================================================================

def generate_vqa(crop_b64_image, seed_questions, api_url, api_key, model_name):
    """Generate VQA using API - based on crop image only"""
    prompt = get_vqa_generation_prompt(seed_questions)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the Image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64_image}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    payload = {
        "stream": False,
        "messages": messages,
        "temperature": 0.9,
        "top_p": 0.8
    }

    resp_json = call_api(payload, api_url=api_url, api_key=api_key, model_name=model_name, stream=False, max_retries=3000)
    if not resp_json:
        return None

    try:
        content = resp_json["choices"][0]["message"]["content"]
        return extract_json_from_text(content)
    except Exception:
        return None


# ==============================================================================
# VQA Validation Function
# ==============================================================================

def validate_single_vqa(original_b64_image, crop_b64_image, question, api_url, api_key, model_name):
    """Validate single VQA question"""
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
   - If Image 1 allows additional valid answers or changes the interpretation (e.g., there are multiple relevant objects in the full image), mark INVALID even if Image 2 looks unambiguous.

4) Clear question:
   - The question must specify the target unambiguously (which object/person/instance).
   - Avoid unclear references like "it", "this", "the object" when multiple candidates exist in Image 1 or Image 2.
   - The question should not rely on unspecified perspective ("left/right" without a clear frame is okay if it's the image frame), or vague quantifiers ("a lot", "some").

OUTPUT FORMAT (strict):
VALID: Yes/No
REASON: A brief explanation referencing which criterion/criteria passed or failed.

EXAMPLE (illustrating a common INVALID case):
- Question: "What is the number on the side of the boat hull?"
- Image 1 (original): There are TWO boats, each with a different hull number (e.g., 12 and 18).
- Image 2 (crop): The crop shows only ONE boat with number 12.
Decision: INVALID
Reason: Although Image 2 yields a single answer (12), the original image allows multiple valid answers (12 or 18), so criterion #3 (consistency with original) fails and the question is ambiguous in the full context.

Now please reason step by step, and then evaluate the given Question using the rules above.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the Original Image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64_image}"}},
                {"type": "text", "text": "This is the Cropped Image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64_image}"}},
                {"type": "text", "text": validation_prompt}
            ]
        }
    ]

    payload = {
        "stream": False,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9
    }

    resp_json = call_api(payload, api_url=api_url, api_key=api_key, model_name=model_name, stream=False, max_retries=3000)
    if not resp_json:
        return {"is_valid": False, "reason": "API call failed"}

    try:
        full_content = resp_json["choices"][0]["message"]["content"]
    except Exception:
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


def validate_vqa_pairs(original_b64_image, crop_b64_image, vqa_pairs, api_url, api_key, model_name, max_validation_workers=6) -> list:
    """Concurrently validate multiple VQA questions"""
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

            future = executor.submit(
                validate_single_vqa,
                original_b64_image,
                crop_b64_image,
                question,
                api_url,
                api_key,
                model_name
            )
            future_to_idx[future] = idx

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                temp_results[idx] = future.result()
            except Exception as e:
                temp_results[idx] = {"is_valid": False, "reason": f"Validation exception: {str(e)}"}

    return temp_results


# ==============================================================================
# Main Processing Pipeline (Modified - Adapted for new JSON format)
# ==============================================================================

def process_line_task(line_data: dict, args, seed_questions: list) -> dict or None:
    """Process single task - Modified to handle new JSON format"""
    
    # Extract image path from new format
    original_image_path = line_data.get('image_path')
    if not original_image_path or not os.path.exists(original_image_path):
        return None

    # Read image directly from local path
    try:
        original_pil_image = Image.open(original_image_path).convert('RGB')
    except Exception as e:
        print(f"⚠ Unable to read image {original_image_path}: {e}")
        return None
    
    # Convert to base64 for API calls
    buffered = io.BytesIO()
    original_pil_image.save(buffered, format="JPEG")
    original_b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Get bbox results
    bbox_results = line_data.get('bbox_results', [])
    if not bbox_results:
        return None
    
    generated_vqa_results = []
    
    for bbox_item in bbox_results:
        # Filter bbox with sam3_score < 0.75
        sam3_score = bbox_item.get('sam3_score', 0)
        if sam3_score < 0.75:
            continue
        
        # Parse bbox
        bbox_str = bbox_item.get('bbox', '')
        bbox = parse_bbox_string(bbox_str)
        if not bbox:
            continue
        
        detected_object = bbox_item.get('detected_object', 'unknown')
        
        # Validate bbox validity
        is_valid, reason = is_crop_valid(original_pil_image, bbox)
        if not is_valid:
            generated_vqa_results.append({
                "detected_object": detected_object,
                "bbox": bbox,
                "sam3_score": sam3_score,
                "crop_path": None,
                "gen_question_status": reason,
                "vqa_pairs": []
            })
            continue

        # Crop image
        try:
            x1, y1, x2, y2 = bbox
            crop_pil_image = original_pil_image.crop((x1, y1, x2, y2))
        except Exception as e:
            generated_vqa_results.append({
                "detected_object": detected_object,
                "bbox": bbox,
                "sam3_score": sam3_score,
                "crop_path": None,
                "gen_question_status": f"crop_failed: {e}",
                "vqa_pairs": []
            })
            continue
        
        # Resize crop image (2x upscaling for better quality)
        original_width, original_height = crop_pil_image.size
        resized_crop_image = crop_pil_image.resize(
            (original_width * 2, original_height * 2), 
            Image.LANCZOS  # High quality interpolation
        )
        
        # Save cropped image to local directory
        filename = f"{os.path.splitext(os.path.basename(original_image_path))[0]}_{detected_object}_{sam3_score:.4f}.jpg"
        try:
            crop_local_path = save_crop_image_to_local(resized_crop_image, args.crop_output_dir, filename)
        except Exception as e:
            print(f"⚠ Failed to save crop image: {e}")
            generated_vqa_results.append({
                "detected_object": detected_object,
                "bbox": bbox,
                "sam3_score": sam3_score,
                "crop_path": None,
                "gen_question_status": f"crop_save_failed: {e}",
                "vqa_pairs": []
            })
            continue
        
        # Convert crop image to base64
        buffered_crop = io.BytesIO()
        resized_crop_image.save(buffered_crop, format="JPEG")
        crop_b64_image = base64.b64encode(buffered_crop.getvalue()).decode('utf-8')

        # Step 1: Generate VQA questions - using crop image only
        candidate_vqa_pairs = generate_vqa(
            crop_b64_image,
            seed_questions,
            args.api_url,
            args.api_key,
            args.model_name
        )

        if not candidate_vqa_pairs or not isinstance(candidate_vqa_pairs, list):
            generated_vqa_results.append({
                "detected_object": detected_object,
                "bbox": bbox,
                "sam3_score": sam3_score,
                "crop_path": crop_local_path,
                "gen_question_status": "skipped_by_generator",
                "vqa_pairs": []
            })
            continue

        # Step 2: Validate VQA questions (validation requires both original and crop images)
        validation_results = validate_vqa_pairs(
            original_b64_image, 
            crop_b64_image,
            candidate_vqa_pairs,
            args.api_url,
            args.api_key,
            args.model_name,
            max_validation_workers=args.validation_workers
        )

        # Integrate validation results
        validated_pairs = []
        for j, pair in enumerate(candidate_vqa_pairs):
            if j < len(validation_results):
                result = validation_results[j]
                pair['validation_status'] = 'passed' if result.get("is_valid", False) else 'failed'
                pair['validation_reason'] = result.get("reason", "No reason.")
            else:
                pair['validation_status'] = 'failed'
                pair['validation_reason'] = 'No validation result'
            validated_pairs.append(pair)

        final_status = "success" if any(p['validation_status'] == 'passed' for p in validated_pairs) else "all_failed_validation"
        generated_vqa_results.append({
            "detected_object": detected_object,
            "bbox": bbox,
            "sam3_score": sam3_score,
            "crop_path": crop_local_path,
            "gen_question_status": final_status,
            "vqa_pairs": validated_pairs
        })
    
    # Return result (keep field names consistent with original code)
    return {
        'image_path': original_image_path,  # Keep field name consistent
        'generated_vqa': generated_vqa_results
    }


# ==============================================================================
# Main Function
# ==============================================================================

def main(args):
    # Create crop output directory if it doesn't exist
    os.makedirs(args.crop_output_dir, exist_ok=True)
    print(f"✓ Crop images will be saved to: {args.crop_output_dir}")
    
    # --- 1. Resume from checkpoint: determine already processed IDs ---
    processed_ids = set()
    output_file_path = f"{args.output_file}.jsonl"
    
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f_out_read:
            for line in f_out_read:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed_ids.add(json.loads(line)['image_path'])
                except:
                    continue
        print(f"✓ Resumed from main output file, already processed {len(processed_ids)} records.")
    else:
        print(f"✓ Main output file {output_file_path} does not exist, will create new file.")

    # --- 2. Read input data (JSONL format) and seed questions ---
    print(f"✓ Reading {len(args.input_files)} input files...")

    all_data = []
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"⚠ File does not exist, skipping: {input_file}")
            continue
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                line_count = 0
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        print(f"  ✗ Failed to parse line in {input_file}: {e}")
                        continue
                
                print(f"  ✓ Read {line_count} records: {os.path.basename(input_file)}")
        except Exception as e:
            print(f"  ✗ Failed to read, skipping {input_file}: {e}")

    if not all_data:
        print("✗ No data successfully read")
        return

    print(f"✓ Total read {len(all_data)} records")
    
    with open(args.seed_json, "r", encoding="utf-8") as f:
        seed_questions = json.load(f)

    # Filter tasks
    tasks = []
    for data in all_data:
        image_path = data.get('image_path', '')
        if image_path and image_path not in processed_ids:
            tasks.append(data)
    
    if not tasks:
        print("✓ No new tasks to process.")
        return

    print(f"✓ Tasks to process: {len(tasks)} (skipped {len(processed_ids)} records)")
    
    # --- 3. Start concurrent processing ---
    processed_count_this_session = 0
    success_count = 0
    error_count = 0
    
    # Create a lock for file writing
    write_lock = threading.Lock()
    
    stats_interval = 50  # Print statistics every 50 tasks
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor, \
         open(output_file_path, 'a', encoding='utf-8') as f_out:

        # Submit tasks
        future_to_task = {
            executor.submit(process_line_task, task_data, args, seed_questions): task_data.get('image_path', 'Unknown')
            for task_data in tasks
        }
        
        progress_bar = tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc="Processing",
            ncols=100
        )
        
        for future in progress_bar:
            task_id = future_to_task.get(future, "Unknown")
            
            try:
                result = future.result(timeout=30000)  # 5 minutes timeout
                
                if result:
                    # Thread-safe file writing
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                    
                    success_count += 1
                    processed_count_this_session += 1
                else:
                    error_count += 1
                    processed_count_this_session += 1
                
                # Periodically print statistics
                if processed_count_this_session % stats_interval == 0:
                    progress_bar.set_postfix({
                        'success': success_count,
                        'error': error_count
                    })
                
            except concurrent.futures.TimeoutError:
                error_count += 1
                print(f"\n⚠ Task {task_id} timeout")
                
            except Exception as exc:
                error_count += 1
                print(f"\n⚠ Task {task_id} exception: {exc}")
    
    # Final statistics
    print("\n" + "="*60)
    print("Processing completed! Final statistics:")
    print(f"  Total processed: {processed_count_this_session}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Output file: {output_file_path}")
    print(f"  Crop images saved to: {args.crop_output_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQA Batch Generation & Validation (Read from JSONL)")
    
    # Input/Output
    parser.add_argument('--input_files', type=str, nargs='+', required=True,
                        help='Input JSONL file path list, supports multiple files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path (without extension, will add .jsonl)')
    parser.add_argument('--crop_output_dir', type=str, required=True,
                        help='Directory to save cropped images')
    parser.add_argument("--seed_json", type=str, default='./seed.json',
                        help='Seed questions JSON file path')
    
    # Concurrency control
    parser.add_argument('--max_workers', type=int, default=30,
                        help='Main task concurrency count')
    parser.add_argument('--validation_workers', type=int, default=30,
                        help='Validation sub-task concurrency count per task')
    
    # API configuration (all required, no defaults in code)
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for the service')
    parser.add_argument('--api_url', type=str, required=True,
                        help='API base URL')
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-235B-A22B-Instruct",
                        help='Model name to use')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║      VQA Generation & Validation System (JSONL)           ║
╟──────────────────────────────────────────────────────────╢
║  Input format: JSONL (DeepEyes format)                    ║
║  Main task concurrency: {args.max_workers:2d}                              ║
║  Validation sub-task concurrency: {args.validation_workers:2d}                     ║
║  Model: {args.model_name:48s} ║
║  API URL: {args.api_url:46s} ║
║  Crop output dir: {args.crop_output_dir:44s} ║
║  Generation method: Based on crop image only              ║
║  SAM3 filter threshold: > 0.75                            ║
╚══════════════════════════════════════════════════════════╝
""")
    
    main(args)
