from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
import math
import json
from PIL import Image, ImageDraw
import numpy as np

def calculate_bbox_attention_ratio(att_map, bbox, image_size, output_shape):
    """
    Calculate the proportion of attention within the bbox region
    
    Args:
        att_map: attention map (H*W,) or (H, W)
        bbox: [x1, y1, x2, y2] original image coordinates
        image_size: (width, height) original image size
        output_shape: (H, W) shape of attention map
    
    Returns:
        ratio: proportion of attention within the bbox region
    """
    # Ensure att_map is 2D
    if len(att_map.shape) == 1:
        att_h, att_w = output_shape
        att_map = att_map.reshape(att_h, att_w)
    else:
        att_h, att_w = att_map.shape
    
    img_w, img_h = image_size
    
    # Convert bbox coordinates to attention map coordinate system
    x1, y1, x2, y2 = bbox
    x1_att = int(x1 / img_w * att_w)
    y1_att = int(y1 / img_h * att_h)
    x2_att = int(x2 / img_w * att_w)
    y2_att = int(y2 / img_h * att_h)
    
    # Ensure coordinates are within valid range
    x1_att = max(0, min(x1_att, att_w - 1))
    x2_att = max(0, min(x2_att, att_w - 1))
    y1_att = max(0, min(y1_att, att_h - 1))
    y2_att = max(0, min(y2_att, att_h - 1))
    
    # Ensure bbox has valid area
    if x2_att <= x1_att:
        x2_att = x1_att + 1
    if y2_att <= y1_att:
        y2_att = y1_att + 1
    
    # Calculate total attention within bbox region
    bbox_attention = att_map[y1_att:y2_att+1, x1_att:x2_att+1].sum()
    total_attention = att_map.sum()
    
    ratio = bbox_attention / total_attention if total_attention > 0 else 0
    
    return ratio, (x1_att, y1_att, x2_att, y2_att)

def visualize_attention_with_bbox(att_map, bbox_att_coords, output_shape, sample_idx, save_path=None):
    """
    Visualize attention map and mark bbox region
    """
    # Ensure att_map is 2D
    if len(att_map.shape) == 1:
        att_map = att_map.reshape(output_shape)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    im = ax.imshow(att_map, cmap="viridis", interpolation="nearest")
    
    # Draw bbox rectangle
    x1, y1, x2, y2 = bbox_att_coords
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    ax.set_title(f"Sample {sample_idx} - Layer 24")
    ax.axis("off")
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def process_benchmark(json_path, model, processor, device, output_dir='./attention_results_ours', target_layer=24):
    """
    Process benchmark dataset and calculate attention concentration
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    for idx, item in enumerate(data):
        print(f"Processing {idx+1}/{len(data)}...")
        
        image_path = item['images'][0]
        query = item['query']
        bbox = item['bbox']  # [x1, y1, x2, y2]
        
        # Get original image size
        img = Image.open(image_path)
        image_size = img.size  # (width, height)
        
        # Prepare input
        messages_query = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "max_pixels": 2048*1024},
                    {"type": "text", "text": f"{query}"},
                ],
            }
        ]
        
        image_inputs, _ = process_vision_info(messages_query)
        
        text_query = processor.apply_chat_template(
            messages_query,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_query],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # Get attention map shape
        image_inputs_aux = processor.image_processor(images=image_inputs)
        output_shape = image_inputs_aux["image_grid_thw"].numpy().squeeze(0)[1:]/2
        output_shape = output_shape.astype(int)
        
        with torch.no_grad():
            vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
            pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
            
            output = model(**inputs, output_attentions=True)
            
            # Only calculate attention for layer 24 (index 23, starting from 0)
            layer_idx = target_layer - 1
            attention = output.attentions[layer_idx]
            
            att = attention[0, :, -1, pos:pos_end].mean(dim=0)
            att = att.to(torch.float32).detach().cpu().numpy()
            
            # Calculate proportion of attention within bbox region
            ratio, bbox_att_coords = calculate_bbox_attention_ratio(
                att, bbox, image_size, output_shape
            )
            
            # Save visualization
            save_path = os.path.join(output_dir, f'sample_{idx}_layer_{target_layer}.png')
            visualize_attention_with_bbox(att, bbox_att_coords, output_shape, idx, save_path)
            
            result = {
                'sample_idx': idx,
                'image_path': image_path,
                'query': query,
                'bbox': bbox,
                'layer_24_ratio': float(ratio),
                'question_type': item['question_type']
            }
            results.append(result)
            
            print(f"  Layer 24 bbox attention ratio: {ratio:.4f}")
    
    # Save results
    results_path = os.path.join(output_dir, 'attention_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print statistics
    print("\n=== Overall Statistics (Layer 24) ===")
    all_ratios = [r['layer_24_ratio'] for r in results]
    print(f"Mean bbox attention ratio: {np.mean(all_ratios):.4f}")
    print(f"Std: {np.std(all_ratios):.4f}")
    print(f"Min: {np.min(all_ratios):.4f}")
    print(f"Max: {np.max(all_ratios):.4f}")
    print(f"Median: {np.median(all_ratios):.4f}")
    
    return results

# Main program
if __name__ == "__main__":
    device = 'cuda'
    model_path ="Qwen/Qwen3-VL-4B-Instruct"
    save_path = "4B"
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    ).eval()
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        padding_side='left', 
        use_fast=True,
        max_pixels=2048*2048,
    )
    
    # Process benchmark
    json_path = '../zoom-bench.json'  # Change to your json file path
    results = process_benchmark(json_path, model, processor, device, output_dir=save_path, target_layer=24)
    
    print("\nProcessing complete! Results saved to ./attention_results/")
