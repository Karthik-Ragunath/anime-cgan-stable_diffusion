from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from types import SimpleNamespace

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def edit_image(input: str, edit: str, output: str):
    parser = ArgumentParser()
    args = SimpleNamespace(
        resolution=512,
        steps=100,
        config="configs/generate.yaml",
        ckpt="instruct_pix2pix_checkpoints/instruct-pix2pix-00-22000.ckpt",
        vae_ckpt=None,
        input=input,
        output=output,
        edit=edit,
        cfg_text=7.5,
        cfg_image=1.5,
        seed=None
    )
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""]) # torch.Size([1, 77, 768])

    seed = random.randint(0, 100000) if args.seed is None else args.seed # 45819
    input_image = Image.open(args.input).convert("RGB")
    width, height = input_image.size # 512, 512
    factor = args.resolution / max(width, height) # 1.0
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height) # 1.0
    width = int((width * factor) // 64) * 64 # 512
    height = int((height * factor) // 64) * 64 # 512
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if args.edit == "":
        input_image.save(args.output)
        return

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])] # args.edit - 'turn him into a cyborg' # cond["c_crossattn"][0].shape - torch.Size([1, 77, 768])
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1 # torch.Size([512, 512, 3])
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device) # torch.Size([1, 3, 512, 512])
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()] # torch.Size([1, 4, 64, 64])

        uncond = {}
        uncond["c_crossattn"] = [null_token] # uncond["c_crossattn"][0].shape - torch.Size([1, 77, 768])
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])] # uncond["c_concat"][0].shape - torch.Size([1, 4, 64, 64])

        sigmas = model_wrap.get_sigmas(args.steps) # args.steps - 100;  # sigmas.shape - torch.Size([101])

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text, # 1.5
            "image_cfg_scale": args.cfg_image, # 7.5
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0] # torch.Size([1, 4, 64, 64]) # sigmas[0] - tensor(14.6146, device='cuda:0')
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args) # torch.Size([1, 4, 64, 64])
        x = model.decode_first_stage(z) # torch.Size([1, 3, 512, 512])
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0) # torch.Size([1, 3, 512, 512])
        x = 255.0 * rearrange(x, "1 c h w -> h w c") # torch.Size([512, 512, 3])
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)

def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="instruct_pix2pix_checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""]) # torch.Size([1, 77, 768])

    seed = random.randint(0, 100000) if args.seed is None else args.seed # 45819
    input_image = Image.open(args.input).convert("RGB")
    width, height = input_image.size # 512, 512
    factor = args.resolution / max(width, height) # 1.0
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height) # 1.0
    width = int((width * factor) // 64) * 64 # 512
    height = int((height * factor) // 64) * 64 # 512
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if args.edit == "":
        input_image.save(args.output)
        return

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])] # args.edit - 'turn him into a cyborg' # cond["c_crossattn"][0].shape - torch.Size([1, 77, 768])
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1 # torch.Size([512, 512, 3])
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device) # torch.Size([1, 3, 512, 512])
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()] # torch.Size([1, 4, 64, 64])

        uncond = {}
        uncond["c_crossattn"] = [null_token] # uncond["c_crossattn"][0].shape - torch.Size([1, 77, 768])
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])] # uncond["c_concat"][0].shape - torch.Size([1, 4, 64, 64])

        sigmas = model_wrap.get_sigmas(args.steps) # args.steps - 100;  # sigmas.shape - torch.Size([101])

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text, # 1.5
            "image_cfg_scale": args.cfg_image, # 7.5
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0] # torch.Size([1, 4, 64, 64]) # sigmas[0] - tensor(14.6146, device='cuda:0')
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args) # torch.Size([1, 4, 64, 64])
        x = model.decode_first_stage(z) # torch.Size([1, 3, 512, 512])
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0) # torch.Size([1, 3, 512, 512])
        x = 255.0 * rearrange(x, "1 c h w -> h w c") # torch.Size([512, 512, 3])
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)


if __name__ == "__main__":
    main()

'''
sigmas:
tensor([14.6146, 13.7525, 12.9524, 12.2094, 11.5188, 10.8763, 10.2782,  9.7208,
         9.2012,  8.7162,  8.2632,  7.8399,  7.4439,  7.0731,  6.7258,  6.4002,
         6.0946,  5.8077,  5.5380,  5.2845,  5.0458,  4.8210,  4.6092,  4.4093,
         4.2207,  4.0425,  3.8740,  3.7146,  3.5637,  3.4207,  3.2851,  3.1565,
         3.0344,  2.9183,  2.8079,  2.7029,  2.6029,  2.5076,  2.4167,  2.3299,
         2.2470,  2.1678,  2.0921,  2.0196,  1.9502,  1.8836,  1.8198,  1.7585,
         1.6996,  1.6431,  1.5886,  1.5362,  1.4858,  1.4371,  1.3902,  1.3449,
         1.3012,  1.2589,  1.2180,  1.1784,  1.1400,  1.1028,  1.0667,  1.0317,
         0.9976,  0.9646,  0.9324,  0.9010,  0.8705,  0.8407,  0.8116,  0.7832,
         0.7555,  0.7283,  0.7017,  0.6757,  0.6501,  0.6250,  0.6003,  0.5760,
         0.5521,  0.5285,  0.5051,  0.4820,  0.4592,  0.4364,  0.4138,  0.3912,
         0.3687,  0.3460,  0.3231,  0.2999,  0.2763,  0.2520,  0.2267,  0.2000,
         0.1712,  0.1386,  0.0987,  0.0292,  0.0000], device='cuda:0')
'''
