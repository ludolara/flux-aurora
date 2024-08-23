from __future__ import annotations

import random
import sys
from argparse import ArgumentParser
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import json
import os
import clip
from diffusers import DiffusionPipeline
# from edit_dataset import EditITMDataset
from dotenv import load_dotenv
load_dotenv()

def calculate_clip_similarity(generated_images, original_image, clip_model, preprocess, device):
    original_image_processed = preprocess(original_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        original_features = clip_model.encode_image(original_image_processed)

    similarities = []
    for img in generated_images:
        img_processed = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            generated_features = clip_model.encode_image(img_processed)
        similarity = torch.nn.functional.cosine_similarity(generated_features, original_features, dim=-1)
        similarities.append(similarity.item())

    dists = [1 - sim for sim in similarities]
    return dists

def load_model(device):
    print("Loading model from McGill-NLP/AURORA")
    pipeline = DiffusionPipeline.from_pretrained("McGill-NLP/AURORA", cache_dir=os.getenv("CACHE_DIR"))
    pipeline = pipeline.to(device)
    return pipeline

def calculate_accuracy(losses):
    correct_count = 0
    for loss in losses:
        if loss[0] < min(loss[1:]):
            correct_count += 1
    return correct_count, len(losses)  # Return counts for aggregation

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="aurora-mixratio-15-15-1-1-42k-steps.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--task", default='flickr_edit', type=str)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--samples", default=4, type=int)
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument('--targets', type=str, nargs='*', help="which target groups for mmbias",default='')
    parser.add_argument("--device", default=0, type=int, help="GPU device index")
    parser.add_argument("--log_imgs", action="store_true")
    parser.add_argument("--conditional_only", action="store_true")
    parser.add_argument("--metric", default="latent", type=str)
    parser.add_argument("--split", default='test', type=str)
    parser.add_argument("--skip", default=1, type=int)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    pipeline = load_model(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # dataset = EditITMDataset(split=args.split, task=args.task, min_resize_res=args.size, max_resize_res=args.size, crop_res=args.size)
    # dataloader = DataLoader(dataset, batch_size=args.batchsize, num_workers=1, worker_init_fn=None, shuffle=False, persistent_workers=True)

    # if os.path.exists(f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}_results.json'):
    #     with open(f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}_results.json', 'r') as f:
    #         results = json.load(f)
    #         results = defaultdict(dict, results)
    # else:
    #     results = defaultdict(dict)

    # for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    #     if len(batch['input'][0].shape) < 3:
    #         continue
    #     for j, prompt in enumerate(batch['texts']):
    #         img_id = batch['path'][0] + f'_{i}'

    #         with torch.no_grad():
    #             generated_images = []
    #             for _ in range(args.samples):
    #                 generated_image = pipeline(prompt[0], guidance_scale=args.cfg_text).images[0]
    #                 generated_images.append(generated_image)

    #         ######## LOG IMAGES ########
    #             input_image_pil = ((batch['input'][0] + 1) * 0.5).clamp(0, 1)
    #             input_image_pil = input_image_pil.permute(1, 2, 0)  # Change from CxHxW to HxWxC for PIL
    #             input_image_pil = (input_image_pil * 255).type(torch.uint8).cpu().numpy()

    #             for k, edited_image in enumerate(generated_images):
    #                 edited_image_np = np.array(edited_image)
    #                 both = np.concatenate((input_image_pil, edited_image_np), axis=1)
    #                 both = Image.fromarray(both)
    #                 out_base = f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}'
    #                 if not os.path.exists(out_base):
    #                     os.makedirs(out_base)
    #                 prompt_str = prompt[0].replace(' ', '_')[0:100]
    #                 both.save(f'{out_base}/{i}_{"correct" if j == 0 else "incorrect"}_sample{k}_{prompt_str}.png')
                
    #         ######## CLIP ########
    #             input_image_pil = Image.fromarray(input_image_pil)
    #             dists_clip = calculate_clip_similarity(generated_images, input_image_pil, clip_model, preprocess, device)

    #         ######## SAVE RESULTS ########
    #             results[img_id]['pos' if j == 0 else 'neg'] = {
    #                 "prompt": prompt[0],
    #                 "clip": dists_clip,
    #             }
    #             with open(f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}_results.json', 'w') as f:
    #                 json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
