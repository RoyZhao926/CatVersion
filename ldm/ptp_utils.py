
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from einops import rearrange, repeat
import cv2
from tqdm import tqdm, trange
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)



@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    sampler,
    num_inference_steps: int = 50,
    guidance_scale: float = 10,
    iter_seed: float = 8888,

):
    register_attention_control(model, controller) #修改了model中含Cross attention的层的forward() 1. 在controller里添加cross attention的attention map；2. count添加attention map的个数
    height = width = 512
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    uc = torch.load("models/uc.pt")
  
    torch.manual_seed(iter_seed)
    cond=model.get_learned_conditioning([prompt])
    shape = [4, height//8, width//8]
    all_samples=[]
    # Fixed sampling xt
    C, H, W = shape
    size = (1, C, H, W)
    x_T = torch.randn(size, device=device)
                
    samples_ddim, _ = sampler.sample(S=num_inference_steps,
                                    conditioning=cond,
                                    batch_size=1,
                                    shape=shape,
                                    verbose=False,
                                    x_T=x_T,
                                    unconditional_guidance_scale=guidance_scale,
                                    unconditional_conditioning=uc,
                                    eta=0)

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            all_samples.append(x_sample)
    print("out len+  "+str(len(all_samples)))
    iter_seed+=1
    return all_samples


# def register_attention_control(model, controller):
#     def ca_forward(self, place_in_unet):
#         to_out = self.to_out
#         if type(to_out) is torch.nn.modules.container.ModuleList:
#             to_out = self.to_out[0]
#         else:
#             to_out = self.to_out

#         def forward(x, context=None, mask=None): 
#             batch_size, sequence_length, dim = x.shape
#             h = self.heads
#             q = self.to_q(x)
#             is_cross = context is not None
#             context = context if is_cross else x
#             k = self.to_k(context)
#             v = self.to_v(context)
#             q = self.reshape_heads_to_batch_dim(q)
#             k = self.reshape_heads_to_batch_dim(k)
#             v = self.reshape_heads_to_batch_dim(v)

#             sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

#             if mask is not None:
#                 mask = mask.reshape(batch_size, -1)
#                 max_neg_value = -torch.finfo(sim.dtype).max
#                 mask = mask[:, None, :].repeat(h, 1, 1)
#                 sim.masked_fill_(~mask, max_neg_value)

#             # attention, what we cannot get enough of
#             attn = sim.softmax(dim=-1)
#             attn = controller(attn, is_cross, place_in_unet) ## 在CrossAttention forward里统计attention map； 
#             out = torch.einsum("b i j, b j d -> b i d", attn, v)
            
#             out = self.reshape_batch_dim_to_heads(out)
#             return to_out(out)

#         return forward

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out

        def forward(x, context=None, mask=None): 
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                
              
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet) ## 在CrossAttention forward里统计attention map； 
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        for name, child in net_.named_children():
            if name == 'attn2':
                child.forward = ca_forward(child, place_in_unet)
                return count + 1
            elif hasattr(child, 'children'): # 统计所有子模块包含cross attention，递归
                count = register_recr(child, count, place_in_unet)
        return count
    
            
    cross_att_count = 0
    sub_nets = model.model.diffusion_model.named_children()
    for net in sub_nets:
        if "input_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count # 统计添加的层数

    