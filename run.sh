python scripts/txt2img_generator.py --ddim_eta 0.0  \
                          --n_samples 1  \
                          --n_iter 1 \
                          --scale 10  \
                          --ddim_steps 50   \
                          --embedding_path logs/colorful_teapot/checkpoints/adapter_gs-999 \
                          --ckpt_path models/sd/v1-5.ckpt   \
                          --prompt "colorful_teapot bicycle"  \
                          --seed 857


python scripts/txt2img_generator.py --ddim_eta 0.0  \
                          --n_samples 1  \
                          --n_iter 1 \
                          --scale 10  \
                          --ddim_steps 50   \
                          --embedding_path logs/colorful_teapot/checkpoints/adapter_gs-999 \
                          --ckpt_path models/sd/v1-5.ckpt   \
                          --prompt "colorful_teapot cake"  \
                          --seed 545

python scripts/txt2img_generator.py --ddim_eta 0.0  \
                          --n_samples 1  \
                          --n_iter 1 \
                          --scale 10  \
                          --ddim_steps 50   \
                          --embedding_path logs/teddybear/checkpoints/adapter_gs-999 \
                          --ckpt_path models/sd/v1-5.ckpt   \
                          --prompt "teddybear on swimming pool"  \
                          --seed 231

python scripts/txt2img_generator.py --ddim_eta 0.0  \
                          --n_samples 1  \
                          --n_iter 1 \
                          --scale 10  \
                          --ddim_steps 50   \
                          --embedding_path logs/teddybear/checkpoints/adapter_gs-999 \
                          --ckpt_path models/sd/v1-5.ckpt   \
                          --prompt "teddybear on the beach"  \
                          --seed 78


