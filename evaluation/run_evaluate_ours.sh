# python scripts/clip_score.py --ref_img_dir_or_path data_ours/dataset3/cat \
#                              --gen_img_dir_or_path outputs_ours/cat-wear-sunglasses_9-11_249 \
#                              --prompt 'wear sunglasses' \
#                              --log_file test_clip_score/clip_score_with_ours_methods_action_edit.txt \



python scripts/clip_score.py --ref_img_dir_or_path other_methods/ref/datasets3/tortoise_plushy \
                             --gen_img_dir_or_path other_methods/dreambooth/datasets3/tortoise_plushy_wooden_floor \
                             --prompt 'wear sunglasses' \
                             --log_file test_clip_score/appendix.txt \