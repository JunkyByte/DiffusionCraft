import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import h5py
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat

# from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import sys
sys.path.append('./src/imagebind')

torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


if __name__ == '__main__':
    seed_everything(42)
    config = OmegaConf.load('src/configs/v2-1-stable-unclip-h-bind-inference.yaml')
    device_name = 'cuda'
    device = torch.device(device_name) # if opt.device == 'cuda' else torch.device('cpu')
    model = load_model_from_config(config, '/home/adryw/dataset/imagecraft/sd21-unclip-h.ckpt', device)
    # model = load_model_from_config(config, '/home/adryw/dataset/imagecraft/v2-1_512-ema-pruned.ckpt', device)
    # model = load_model_from_config(config, '/u/dssc/adonninelli/scratch/sd21-unclip-h.ckpt', device)

    # TODO: Add negative prompts

    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    # https://stable-diffusion-art.com/samplers/
    sampler = DDIMSampler(model, device=device)
    ddim_eta = 0  # "ddim eta (eta=0.0 corresponds to deterministic sampling"

    # Out folders
    outpath = './output/'
    os.makedirs(outpath, exist_ok=True)

    # Hardcoded batches and prompts (can be read from file)
    batch_size = 1
    n_rows = 1
    prompt = 'a photo of sunset with highway in the distance, canyon style, vivid colors, western, sunset, chill mood, detailed, orange'

    prompt_addition = ', best quality, extremely detailed'
    prompt = prompt + prompt_addition

    prompts_data = [batch_size * [prompt]]

    image_paths = ["samples/car_image.jpg"]
    audio_paths = ["samples/rain.wav"]
    # depth_paths = "samples/01441.h5"

    # with h5py.File(depth_paths, 'r') as f:
    #     depth = np.array(f['depth'])
    #     rgb = np.array(f['rgb'])
    # depth = ((depth / depth.max()) * 255).astype(np.uint8)
    # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
        # ModalityType.DEPTH: data.load_and_transform_thermal_data([depth], device),
    }

    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # Can be different
    C = 4  # Latent channels
    H = 768
    W = 768
    f = 8  # Downsampling factor
    shape = [C, H // f, W // f]
    diff_steps = 50

    # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    scale = 9

    # Prepare embeddings cond
    with torch.no_grad():  # TODO: In main_bind2 they use norm=True for audio
        embeddings_imagebind = model.embedder(inputs, normalize=False)
    strength = 0.5
    noise_level = 0.25

    alpha = 0.5
    # embeddings_imagebind = 0.1 * embeddings_imagebind[ModalityType.DEPTH]
    # embeddings_imagebind = alpha * embeddings_imagebind[ModalityType.AUDIO] + (1-alpha) * embeddings_imagebind[ModalityType.VISION]
    embeddings_imagebind = embeddings_imagebind[ModalityType.VISION]

    og_c_adm = repeat(embeddings_imagebind, '1 ... -> b ...', b=batch_size) * strength
    # og_c_adm = (og_c_adm / og_c_adm.norm()) * 20
    
    # fiuuu
    model.embedder.to('cpu')

    n_samples = 32
    for i in range(n_samples):
        start_code = torch.randn([batch_size, *shape], device=device)
        
        c_adm = og_c_adm
        if model.noise_augmentor is not None:
            c_adm, noise_level_emb = model.noise_augmentor(og_c_adm, noise_level=(
                    torch.ones(batch_size) * model.noise_augmentor.max_noise_level *
                    noise_level).long().to(c_adm.device))
            # assume this gives embeddings of noise levels
            c_adm = torch.cat((c_adm, noise_level_emb), 1)

        with torch.no_grad(), model.ema_scope():
            for prompts in tqdm(prompts_data, desc="data"):
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [n_prompt])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)

                uc = {'c_crossattn': [uc], 'c_adm': c_adm}
                c = {'c_crossattn': [c], 'c_adm': c_adm}

                samples, _ = sampler.sample(S=diff_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=start_code)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
                    sample_count += 1