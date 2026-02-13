import base64
from openai import OpenAI
from PIL import Image
import io
import os
import pandas as pd
import json
import csv
import re
import random
import argparse
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from mathruler.grader import extract_boxed_content
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch
from collections import Counter, defaultdict
import concurrent.futures
# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ==============================================================================
# Configuration
# ==============================================================================



def pil_image_to_base64(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr = byte_arr.getvalue()
    base64_str = base64.b64encode(byte_arr).decode('utf-8')
    return base64_str

def get_image_dimensions(img_path):
    """Get the width and height of the image"""
    with Image.open(img_path) as img:
        width, height = img.size
    return width, height

def extract_objects_from_response(response_text):
    """Extract object list from Qwen response"""
    # Try multiple patterns to extract object list
    # Pattern 1: Look for numbered list like "1. object1\n2. object2"
    pattern1 = r'\d+\.\s*([^\n]+)'
    matches1 = re.findall(pattern1, response_text)
    
    # Pattern 2: Look for comma-separated list
    pattern2 = r'(?:objects?|items?)[:\s]+([^\n\.]+)'
    matches2 = re.findall(pattern2, response_text, re.IGNORECASE)
    
    objects = []
    if matches1 and len(matches1) > 0:
        objects = [obj.strip().rstrip(',.;:') for obj in matches1]
    elif matches2 and len(matches2) > 0:
        # Split by comma
        objects = [obj.strip() for obj in matches2[0].split(',')]
    
    # Clean and filter
    objects = [obj for obj in objects if obj and len(obj) > 1 and len(obj) < 50]
    
    return objects

def generate_bboxes_for_image(img_path, api_url, img_idx, sam3_processor, sam3_lock, api_key):
    """Generate bounding box annotations for all objects in the image using Qwen3-VL to list objects + SAM3 to extract all bboxes"""
    
    # Get image dimensions
    width, height = get_image_dimensions(img_path)
    
    # Step 1: Use Qwen3-VL to list all objects in the image (only call once)
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
    attempts = 0
    last_error = None

    # Try to call Qwen3-VL to identify objects
    while attempts < max_attempts:
        attempts += 1
        try:
            # Step 1: Call Qwen3-VL to identify objects
            with open(img_path, "rb") as f:
                image_encoded = base64.b64encode(f.read()).decode("utf-8")

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
                        {"type": "text", "text": object_detection_prompt}
                    ]
                }
            ]

            data = {
                'stream': False,
                "model": "Qwen3-VL-235B-A22B-Instruct",
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
            
            # Extract object list
            objects = extract_objects_from_response(res_content)
            
            if not objects or len(objects) <= 2:
                return {
                    "image_path": img_path,
                    "image_width": width,
                    "image_height": height,
                    "total_objects_detected": 0,
                    "successful_bboxes": 0,
                    "success_rate": 0.0,
                    "bbox_results": [],
                    "success": False,
                    "full_response": None,
                    "error": "Failed. Last error: No object detected",
                    "image_index": img_idx,
                    "qwen_attempts": 1
                }
            
            print(f"\n[Image {img_idx}] Detected {len(objects)} objects: {objects[:10]}...")
            
            # Step 2: For each detected object, use SAM3 to extract bbox
            bbox_results = []
            
            # Load image once (for all SAM3 calls)
            image = Image.open(img_path).convert('RGB')
            
            for obj_idx, detected_object in enumerate(objects):
                try:
                    # Use SAM3 for segmentation (thread-safe)
                    with sam3_lock:
                        # Set image
                        inference_state = sam3_processor.set_image(image)
                        
                        # Use current object as prompt
                        inference_state = sam3_processor.set_text_prompt(
                            state=inference_state, 
                            prompt=detected_object
                        )
                        
                        # Get results
                        masks = inference_state.get("masks")
                        boxes = inference_state.get("boxes")
                        scores = inference_state.get("scores")
                        
                        if boxes is None or len(boxes) == 0:
                            print(f"  [{obj_idx+1}/{len(objects)}] SAM3 found no boxes for: {detected_object}")
                            bbox_results.append({
                                "detected_object": detected_object,
                                "bbox": None,
                                "sam3_score": 0.0,
                                "success": False,
                                "error": "SAM3 found no boxes"
                            })
                            continue
                        if len(boxes) > 1:
                            print(f"  [{obj_idx+1}/{len(objects)}] Skipping {detected_object}: found {len(boxes)} instances (not unique)")
                            bbox_results.append({
                                "detected_object": detected_object,
                                "bbox": None,
                                "sam3_score": 0.0,
                                "success": False,
                                "error": f"Multiple instances found: {len(boxes)}"
                            })
                            continue
                        
                        best_box = boxes[0]
                        best_score = scores[0].item() if scores is not None and len(scores) > 0 else 0.0
                        
                        # Convert bbox format: tensor -> (x_min, y_min, x_max, y_max)
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
                        
                        print(f"  [{obj_idx+1}/{len(objects)}] ✓ {detected_object}: {bbox_str} (score: {best_score:.3f})")
                        
                except Exception as sam_error:
                    print(f"  [{obj_idx+1}/{len(objects)}] ✗ SAM3 error for {detected_object}: {str(sam_error)}")
                    bbox_results.append({
                        "detected_object": detected_object,
                        "bbox": None,
                        "sam3_score": 0.0,
                        "success": False,
                        "error": str(sam_error)
                    })
            
            # Calculate success rate
            successful_bboxes = [r for r in bbox_results if r["success"]]
            success_rate = len(successful_bboxes) / len(bbox_results) if bbox_results else 0
            
            return {
                "image_path": img_path,
                "image_width": width,
                "image_height": height,
                "total_objects_detected": len(objects),
                "successful_bboxes": len(successful_bboxes),
                "success_rate": success_rate,
                "bbox_results": bbox_results,
                "success": True,
                "full_response": full_response,
                "error": None,
                "image_index": img_idx,
                "qwen_attempts": attempts
            }
                    
        except Exception as e:
            last_error = str(e)
            print(f"Attempt {attempts}/{max_attempts} failed for {img_path}: {last_error}")
            time.sleep(1)
            continue

    # If max attempts reached without success (Qwen3-VL call failed)
    return {
        "image_path": img_path,
        "image_width": width,
        "image_height": height,
        "total_objects_detected": 0,
        "successful_bboxes": 0,
        "success_rate": 0.0,
        "bbox_results": [],
        "success": False,
        "full_response": None,
        "error": f"Failed after {max_attempts} attempts. Last error: {last_error}",
        "image_index": img_idx,
        "qwen_attempts": max_attempts
    }


def process_single_task(task_data):
    img_path = task_data['img_path']
    img_idx = task_data['img_idx']
    api_url = task_data['api_url']
    sam3_processor = task_data['sam3_processor']
    sam3_lock = task_data['sam3_lock']
    api_key = task_data['api_key']
    
    result = generate_bboxes_for_image(
        img_path, api_url, img_idx, sam3_processor, sam3_lock, api_key
    )
    
    return result

def get_images_from_folder(folder_path):
    """Get all image file paths from a folder"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file.lower())[1] in supported_formats:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    return sorted(image_paths)

def get_images_from_folders(folder_list):
    """Get all image file paths from multiple folders"""
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

def load_completed_tasks_from_jsonl(output_jsonl):
    """Load completed tasks from JSONL file"""
    if not os.path.exists(output_jsonl):
        return set(), 0
    
    try:
        completed_tasks = set()
        line_count = 0
        
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    completed_tasks.add(result['image_index'])
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_count + 1}: {str(e)}")
                    continue
        
        print(f"Loaded {line_count} existing results from {output_jsonl}")
        print(f"Found {len(completed_tasks)} completed images")
        
        return completed_tasks, line_count
    
    except Exception as e:
        print(f"Error loading existing results: {str(e)}")
        print("Starting from scratch...")
        return set(), 0

def append_result_to_jsonl(result, output_jsonl, file_lock):
    """Thread-safe append result to JSONL file"""
    with file_lock:
        with open(output_jsonl, 'a', encoding='utf-8') as f:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate bounding boxes for images using Qwen3-VL + SAM3.')
    parser.add_argument(
        '--image_folders',
        type=str,
        nargs='+',
        default=['/path/images/sa1b'],
        help='paths to folders containing images (can specify multiple folders)'
    )
    parser.add_argument("--output_jsonl", type=str,
                        default="generated_bboxes_sa1b.jsonl",
                        help="Path to output JSONL file")
    parser.add_argument("--max_workers", type=int, default=8, 
                        help="Maximum number of concurrent workers")
    parser.add_argument("--resume", action='store_true',
                        help="Resume from existing JSONL file")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to randomly select (default: use all images")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducible image selection (default: 42)")
    parser.add_argument("--api_key", type=str, default="xxx",
                        help="API key for Qwen service")
    parser.add_argument("--api_url", type=str, default="https://xxx.com/v1/chat/completions",
                        help="API key for Qwen service")
    args = parser.parse_args()
    
    
    print("Loading SAM3 model...")
    try:
        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)
        sam3_lock = threading.Lock()
        print("SAM3 model loaded successfully!")
    except Exception as e:
        print(f"Error loading SAM3 model: {str(e)}")
        print("Please make sure sam3 is properly installed and sam3.pt is available.")
        exit(1)
    
    print(f"\nLoading images from {len(args.image_folders)} folder(s)...")
    print("="*60)
    
    all_image_paths, folder_stats = get_images_from_folders(args.image_folders)
    
    print("="*60)
    print(f"Total images found: {len(all_image_paths)}")
    print("\nFolder statistics:")
    for folder, count in folder_stats.items():
        print(f"  {folder}: {count} images")
    print("="*60)
    
    if len(all_image_paths) == 0:
        raise ValueError(f"No images found in any of the specified folders")
    
    print("\nValidating image paths (and filtering by resolution)...")
    valid_image_paths = []
    invalid_count = 0
    filtered_small_count = 0

    MIN_W, MIN_H = 1200, 1200

    for img_path in tqdm(all_image_paths, desc="Validating images"):
        try:
            with Image.open(img_path) as img:
                img.verify()

            # After verify(), need to reopen to read size
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
    print(f"Filtered by small resolution: {filtered_small_count}")
    print(f"Invalid images: {invalid_count}")
    
    if args.max_images is not None and args.max_images < len(valid_image_paths):
        print(f"\n{'='*60}")
        print(f"Randomly selecting {args.max_images} images from {len(valid_image_paths)} valid images...")
        print(f"Random seed: {args.random_seed}")
        
        random.seed(args.random_seed)
        valid_image_paths = random.sample(valid_image_paths, args.max_images)
        
        print(f"Selected {len(valid_image_paths)} images")
        print(f"{'='*60}")
    
    # Load completed tasks (if in resume mode)
    completed_tasks = set()
    existing_count = 0
    
    if args.resume:
        print("\n" + "="*60)
        print("Checking for existing results...")
        completed_tasks, existing_count = load_completed_tasks_from_jsonl(args.output_jsonl)
        print("="*60)
    
    print("\nPreparing tasks...")
    all_tasks = []
    skipped_count = 0
    
    for img_idx, img_path in enumerate(valid_image_paths):
        if img_idx in completed_tasks:
            skipped_count += 1
            continue
        
        task_data = {
            'img_path': img_path,
            'img_idx': img_idx,
            'api_url': args.api_url,
            'sam3_processor': sam3_processor,
            'sam3_lock': sam3_lock,
            'api_key': args.api_key
        }

        all_tasks.append(task_data)
    
    total_tasks = len(all_tasks)
    print(f"Total images: {len(valid_image_paths)}")
    print(f"Already completed: {skipped_count}")
    print(f"Remaining images: {total_tasks}")
    print(f"Using {args.max_workers} concurrent workers")
    
    if total_tasks == 0:
        print("\nAll images already processed!")
        print(f"Results are in: {args.output_jsonl}")
        exit(0)
    
    # Create file lock for thread-safe writing
    file_lock = threading.Lock()
    
    # Statistics
    processed_count = 0
    success_count = 0
    
    # Use thread pool for concurrent processing
    print(f"\nStarting concurrent processing...")
    print(f"Results will be written to: {args.output_jsonl}")
    print("Note: SAM3 operations are thread-safe via locking")
    print("Each image will be processed once - all detected objects will get bboxes\n")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_task, task): task for task in all_tasks}
        
        # Use tqdm to show progress
        with tqdm(total=total_tasks, desc="Processing images", initial=0) as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    
                    # Immediately write to JSONL file (thread-safe)
                    append_result_to_jsonl(result, args.output_jsonl, file_lock)
                    
                    # Update statistics
                    processed_count += 1
                    if result["success"]:
                        success_count += 1
                    
                    # Update progress bar
                    if result["success"]:
                        pbar.set_postfix({
                            "success": f"{success_count}/{processed_count}",
                            "objects": result.get('total_objects_detected', 0),
                            "bboxes": result.get('successful_bboxes', 0),
                            "rate": f"{result.get('success_rate', 0)*100:.0f}%"
                        })
                    else:
                        pbar.set_postfix({
                            "success": f"{success_count}/{processed_count}",
                            "error": result['error'][:30] if result.get('error') else "Unknown"
                        })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    tqdm.write(f"Task failed with exception: {str(e)}")
                    # Record even if failed
                    error_result = {
                        "image_path": task['img_path'],
                        "image_index": task['img_idx'],
                        "success": False,
                        "error": f"Exception: {str(e)}"
                    }
                    append_result_to_jsonl(error_result, args.output_jsonl, file_lock)
                    processed_count += 1
                    pbar.update(1)
    
    # Print final statistics
    print("\n" + "="*60)
    print("Processing completed!")
    print(f"Total processed: {processed_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {processed_count - success_count}")
    print(f"Success rate: {success_count/processed_count*100:.2f}%")
    print(f"Results saved to: {args.output_jsonl}")
    print(f"Total lines in file: {existing_count + processed_count}")
    print("="*60)

