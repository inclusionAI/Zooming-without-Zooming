import json
import argparse
from pathlib import Path
from collections import defaultdict
import random, os
from datasets import Dataset
from tqdm import tqdm


def main(args):
    # 1. Read JSONL and extract data
    print(f"Reading file: {args.input_file}")
    valid_vqas = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing JSONL"):
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
                    if majority_answer is None or len(majority_answer)==0:
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
    
    
    # 2. Build training dataset (using bbox images)
    def build_dataset(vqas):
        print(f"\nBuilding training dataset...")
        images = []
        problems = []
        answers = []
        bboxes = []
        statuses = []
        
        for vqa in tqdm(vqas):
            # Use bbox image for training
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
        
        return Dataset.from_dict(data)
    
    # Use all data for training
    train_dataset = build_dataset(valid_vqas)
    
    print(f"\nTotal training samples: {len(train_dataset)}")
    
    # 3. Save to parquet
    print(f"\nSaving parquet file to: {args.output_path}")
    train_dataset.to_parquet(args.output_path)
    
    print("\n✓ Done!")
    print(f"\nTraining set example:")
    print(train_dataset[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output parquet file')
    args = parser.parse_args()
    main(args)
