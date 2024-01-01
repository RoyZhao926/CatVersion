import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
os.environ["CUDA_VISIBLE_DEVICES"]='1'
sys.path.append(os.getcwd())
print(os.getcwd())
# sys.path.append('..')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="Beach and seaside in the style of  *",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=10,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/models/sd/v1-5.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        default="/logs/style2023-05-11T19-45-48_test1/checkpoints/embeddings.pt",
        help="Path to a pre-trained embedding manager checkpoint")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=8888,
        help="init manual seed for Xt")    
    # parser.add_argument(
    #     "-b",
    #     "--base",
    #     nargs="*",
    #     metavar="base_config.yaml",
    #     help="paths to base configs. Loaded from left-to-right. "
    #          "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    #     default=list(),
    # )
    
    opt = parser.parse_args()


    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    # model.embedding_manager.load(opt.embedding_path)
    # model.cond_stage_model.load('logs/style12023-05-12T00-00-38_test1/checkpoints/lora_gs-4999.pt')
    model.cond_stage_model.adapter_load(opt.embedding_path)
    # model.cond_stage_model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt
    torch.manual_seed(opt.seed)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                #uc = model.get_learned_conditioning(opt.n_samples * [""])
                uc = torch.load("models/uc.pt")
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                # torch.save(c,"models/mycondition.pt")
                shape = [4, opt.H//8, opt.W//8]
                
                # Fixed sampling xt
                C, H, W = shape
                size = (opt.n_samples, C, H, W)
                x_T = torch.randn(size, device=device)
                
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 x_T=x_T,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.jpg"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
