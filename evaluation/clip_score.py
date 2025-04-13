from PIL import Image
import argparse
import torch
from torchvision import transforms
import os
import glob

import sys
import numpy as np
from evaluation.clip_eval import CustomCLIPEvaluator


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="running in time square",
        help="the prompt to render"
    )
    parser.add_argument(
        "--ref_img_dir_or_path",
        type=str,
        default='data/dataset2/dog8',
        help=""
    )

    parser.add_argument(
        "--gen_img_dir_or_path",
        type=str,
        default='outputs_others/outputs_dog8/dog8-running-in-time-square/0-1',
        help=""
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default='test_clip_score/clip_score_with_others_methods.txt',
        help=""
    )

    opt = parser.parse_args()

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = CustomCLIPEvaluator(device)
    
    # gen_img
    opt.gen_img_dir_or_path_samples = os.path.join(opt.gen_img_dir_or_path)
    if os.path.isdir(opt.gen_img_dir_or_path_samples):
        gen_file_list = glob.glob(os.path.join(opt.gen_img_dir_or_path_samples, '*.*'))
    else:
        gen_file_list = [opt.gen_img_dir_or_path]

    # # masked gen_img
    # opt.gen_mask_img_dir_or_path_samples = os.path.join(opt.gen_mask_img_dir_or_path)
    # if os.path.isdir(opt.gen_mask_img_dir_or_path_samples):
    #     mask_gen_file_list = glob.glob(os.path.join(opt.gen_mask_img_dir_or_path_samples, '*.*'))
    # else:
    #     mask_gen_file_list = [opt.gen_mask_img_dir_or_path]
    
    # ref_img   
    opt.ref_img_dir_or_path_samples = os.path.join(opt.ref_img_dir_or_path)
    if os.path.isdir(opt.ref_img_dir_or_path_samples):
        ref_file_list = glob.glob(os.path.join(opt.ref_img_dir_or_path_samples, '*.*'))
    else:
        ref_file_list = [opt.ref_img_dir_or_path]



    clip_img_score = []
    clip_text_score = []


    for ref_file in ref_file_list:
        ref_img = Image.open(ref_file)
        clip_img_score_per_ref = []
        for simple in gen_file_list:
            gen_img = Image.open(simple)
            clip_img_score_per_ref.append(evaluator.img_to_img_similarity(ref_img, gen_img).cpu())
        clip_img_score.append(clip_img_score_per_ref)


    for simple in gen_file_list:
        gen_img = Image.open(simple)
        clip_text_score_per_gen = evaluator.txt_to_img_similarity(opt.prompt, gen_img).cpu()
        clip_text_score.append(clip_text_score_per_gen)



    clip_text_score = np.array(clip_text_score)
    # avg_clip_img_score = np.mean(clip_img_score)
    avg_clip_text_score = np.mean(clip_text_score)

    clip_img_score = np.array(clip_img_score)
    avg_clip_img_score = np.mean(clip_img_score)

    avg_clip_scores = avg_clip_text_score * avg_clip_img_score
    avg_clip_scores = avg_clip_scores ** 0.5  # geometric mean
    
    log_file = opt.log_file
    with open(log_file, 'a') as f:
        f.write(f'{opt.prompt}\n')
        f.write(f'ref_img: {opt.ref_img_dir_or_path}, gen_img: {opt.gen_img_dir_or_path}, clip img score: {avg_clip_img_score:.4f} \n')
        f.write(f'gen_img: {opt.gen_img_dir_or_path}, prompt: {opt.prompt}, clip text score: {avg_clip_text_score:.4f} \n')
        f.write(f'avg clip score: {avg_clip_scores:.4f} \n')
        # f.write('\n')



