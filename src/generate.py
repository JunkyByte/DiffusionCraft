import argparse, os
import yaml
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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )

    parser.add_argument(
        "--start-image",
        type=str,
        default=None,
        nargs="?",
        help=""
    )

    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        nargs="?",
        help=""
    )

    parser.add_argument(
        "--cond-image",
        type=str,
        nargs="*",
        help=""
    )

    parser.add_argument(
        "--cond-video",
        type=str,
        nargs="*",
        help=""
    )

    parser.add_argument(
        "--cond-audio",
        type=str,
        nargs="*",
        help=""
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./output/"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0,
        help="noise level",
    )
    parser.add_argument(
        "--cond-strength",
        type=float,
        default=1.0,
        help="strength of conditioning",
    )
    parser.add_argument(
        "--img-strength",
        type=float,
        default=0.8,
        help="strength of image transformation",
    )

    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=768,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=768,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="alpha for multi conditioning",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--startseed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    opt = parser.parse_args()
    return opt


def save_config_to_yaml(args, filename):
    config_dict = vars(args)
    with open(filename, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_img(img_file):
    image = Image.open(img_file).convert('RGB')
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = image.to(device.type)
    return image


def load_mask(mask_file, img_w, img_h, down_factor):
    w, h = img_w, img_h
    mask = Image.open(mask_file).convert('L')
    mask = mask.resize((w // down_factor, h // down_factor), resample=Image.Resampling.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = torch.from_numpy(mask).to(device.type)
    return mask


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
    opt = parse_args()

    print("Loading configuration files...")
    with open('src/configs/configs.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    config = OmegaConf.load('src/configs/v2-1-stable-unclip-h-bind-inference.yaml')
    device_name = 'cuda'
    device = torch.device(device_name) # if opt.device == 'cuda' else torch.device('cpu')
    print(f"Using device: {device}")

    model = load_model_from_config(config, config_data['sd_ckpt'], device)
    print("Model loaded successfully.")

    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    # https://stable-diffusion-art.com/samplers/
    if opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
        print("Using DPM Solver Sampler.")
    else:
        sampler = DDIMSampler(model, device=device)
        print("Using DDIM Sampler.")

    # Out folders
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Output directory created: {opt.outdir}")

    # Hardcoded batches and prompts (can be read from file)
    batch_size = 1
    prompt = opt.prompt
    prompt_addition = ', best quality, extremely detailed'
    prompt = prompt + prompt_addition
    print(f"Using prompt: {prompt}")
    prompts_data = [batch_size * [prompt]]

    n_prompt = '2D | | Low Quality | | text logos | | watermarks | | signatures | | out of frame | | jpeg artifacts | | ugly | | poorly drawn | | extra limbs | | extra hands | | extra feet | | backwards limbs | | extra fingers | | extra toes | | unrealistic, incorrect, bad anatomy | | cut off body pieces | | strange body positions | | impossible body positioning | | Mismatched eyes | | cross eyed | | crooked face | | crooked lips | | unclear | | undefined | | mutations | | deformities | | off center | | poor_composition | | duplicate faces, plastic, fake, human, humans, people, tiny, negativity, blurry, blurred, doll, unclear'

    image_paths = opt.cond_image if opt.cond_image else []
    video_paths = opt.cond_video if opt.cond_video else []
    audio_paths = opt.cond_audio if opt.cond_audio else []

    if image_paths:
        print(f"Conditioning on images: {image_paths}")
    if audio_paths:
        print(f"Conditioning on audio: {audio_paths}")
    if video_paths:
        print(f"Conditioning on videos: {video_paths}")

    sample_path = os.path.join(opt.outdir, "samples")
    save_config_to_yaml(opt, os.path.join(opt.outdir, 'gen_cfg.yaml'))
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(opt.outdir)) - 1

    # Can be different
    C = 4  # Latent channels
    H = opt.H
    W = opt.W
    f = 8  # Downsampling factor
    shape = [C, H // f, W // f]
    diff_steps = opt.steps

    print(f"Using shape: {shape}, diffusion steps: {diff_steps}")

    # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    scale = opt.scale
    print(f"Using guidance scale: {scale}")

    # Prepare embeddings cond
    num_conds = len(image_paths) + len(audio_paths) + len(video_paths)
    noise_level = opt.noise_level
    print(f"Number of conditions: {num_conds}, noise level: {noise_level}")

    if num_conds:
        inputs = {}
        if len(image_paths):
            inputs[ModalityType.VISION] = data.load_and_transform_vision_data(image_paths, device)
        if len(audio_paths):
            inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data(audio_paths, device)
        if len(video_paths):
            if ModalityType.VISION in inputs:
                inputs[ModalityType.VISION] = torch.cat(data.load_and_transform_video_data(video_paths, device), dim=0)
            else:
                inputs[ModalityType.VISION] = data.load_and_transform_video_data(video_paths, device)

        with torch.no_grad():  # TODO: In main_bind2 they use norm=True for audio
            embeddings_imagebind = model.embedder(inputs, normalize=False)
        cond_strength = opt.cond_strength
        print(f"Conditioning strength: {cond_strength}")

        if num_conds == 2:
            alpha = opt.alpha
            if len(image_paths) and len(audio_paths):
                embeddings_imagebind = alpha * embeddings_imagebind[ModalityType.VISION] + (1 - alpha) * embeddings_imagebind[ModalityType.AUDIO]
                print(f"Using alpha blending: alpha={alpha}")
            else:
                key = list(inputs.keys())[0]
                embeddings_imagebind = (alpha * embeddings_imagebind[key][0] + (1 - alpha) * embeddings_imagebind[key][1]).unsqueeze(0)
        else:
            alpha = 1 / num_conds
            out_embeddings = torch.zeros((1024,), device=device)
            if len(image_paths):
                out_embeddings += alpha * embeddings_imagebind[ModalityType.VISION].sum(0)
            elif len(audio_paths):
                out_embeddings += alpha * embeddings_imagebind[ModalityType.AUDIO].sum(0)
            embeddings_imagebind = out_embeddings.unsqueeze(0)
        og_c_adm = repeat(embeddings_imagebind, '1 ... -> b ...', b=batch_size) * cond_strength
        # og_c_adm = (og_c_adm / og_c_adm.norm()) * 20
    else:
        og_c_adm = torch.zeros((1, 1024), device=device)
        print("No conditioning used.")

    # REMOVE ME hard coded depth loading (overwrites other conditioning)
    # def convert_depth_to_disparity(depth, focal_length, sensor_type, min_depth=0.01, max_depth=50):
    #     baseline = 0.075 # sensor_to_params[sensor_type]["baseline"]
    #     # depth_in_meters = depth / 1000.
    #     depth_in_meters = depth
    #     if min_depth is not None:
    #         depth_in_meters = depth_in_meters.clip(min=min_depth, max=max_depth)
    #     disparity = baseline * focal_length / depth_in_meters
    #     return torch.from_numpy(disparity).float()
    # from scipy.ndimage import zoom
    # depth_files = ['./samples/depth/01441.h5']
    # with h5py.File(depth_files[0], 'r') as f:
    #     depth = np.array(f['depth'])
    #     depth = zoom(depth, (224/depth.shape[0], 224/depth.shape[1]), order=1)
    #     disparity = convert_depth_to_disparity(depth, 518.85790117450188, 'kv1', min_depth=0.01, max_depth=50).unsqueeze_(dim=0).to(device)
    # inputs = {
    #     ModalityType.DEPTH: disparity[None],
    # }
    # with torch.no_grad():
    #     embeddings = model.embedder(inputs, normalize=False)
    #     embeddings_imagebind = embeddings[ModalityType.DEPTH]
    # cond_strength = opt.cond_strength
    # og_c_adm = repeat(embeddings_imagebind, '1 ... -> b ...', b=batch_size) * cond_strength
    #####
    
    if opt.start_image is None and opt.mask is not None:
        raise RuntimeError("If mask is passed also start image must be passed")

    if opt.start_image is not None:
        init_image_file = load_img(opt.start_image)
        print(f'>>> Loaded img with shape {init_image_file.shape}')

    if opt.mask is not None:
        mask = load_mask(opt.mask, init_image_file.shape[3], init_image_file.shape[2], f)
        print(f'>>> Loaded mask, reshaped to {init_image_file.shape}')

    # fiuuu
    model.embedder.to('cpu')

    for i in range(opt.n_iter):
        seed_everything(opt.startseed + i)

        if opt.start_image is None:
            start_code = torch.randn([batch_size, *shape], device=device)
        else:
            init_image = repeat(init_image_file, '1 ... -> b ...', b=batch_size).to(device)  # Propag. over batch
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
            sampler.make_schedule(ddim_num_steps=diff_steps, ddim_eta=opt.ddim_eta, verbose=False)
            t_enc = int(opt.img_strength * diff_steps)
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
            if opt.mask is not None:
                random = torch.randn(mask.shape, device=model.device)
                z_enc = (mask * random) + ((1 - mask) * z_enc)
        
        c_adm = og_c_adm
        if model.noise_augmentor is not None:
            c_adm, noise_level_emb = model.noise_augmentor(og_c_adm, noise_level=(
                    torch.ones(batch_size) * model.noise_augmentor.max_noise_level *
                    noise_level).long().to(c_adm.device))
            c_adm = torch.cat((c_adm, noise_level_emb), 1)

        with torch.no_grad(), model.ema_scope():
            for prompts in tqdm(prompts_data, desc="data"):
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [n_prompt])
                c = model.get_learned_conditioning(prompts)

                uc = {"c_crossattn": [uc], "c_adm": torch.zeros_like(c_adm)}
                c = {'c_crossattn': [c], 'c_adm': c_adm}

                if opt.start_image is None:
                    samples, _ = sampler.sample(S=diff_steps,
                                                conditioning=c,
                                                batch_size=batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code)
                else:
                    if opt.mask is None:
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc)
                    else:
                        samples = sampler.decode_inpaint(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                         z_mask=mask, x0=init_latent, unconditional_conditioning=uc)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}_{opt.startseed + i}.png"))
                    base_count += 1
                    sample_count += 1