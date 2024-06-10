import argparse, os
import cv2
import torch
import PIL
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
# from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def load_img_mask(img_file, mask_file, down_factor):
    image = Image.open(img_file).convert('RGB')
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = image.to(device.type)

    mask = Image.open(mask_file)
    mask = mask.resize((w, h), resample=Image.Resampling.LANCZOS).convert('L')
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device.type)
    return image, mask

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

    config = OmegaConf.load('src/configs/v2-inference-v_inpaint.yaml')
    device_name = 'cuda'
    device = torch.device(device_name) # if opt.device == 'cuda' else torch.device('cpu')
    model = load_model_from_config(config, '/home/adryw/dataset/imagecraft/v2_512-inpainting-ema.ckpt', device)

    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    # https://stable-diffusion-art.com/samplers/
    sampler = DDIMSampler(model, device=device)
    # sampler = PLMSSampler(model, device=device)
    # sampler = DPMSolverSampler(model, device=device)
    ddim_eta = 0  # "ddim eta (eta=0.0 corresponds to deterministic sampling"

    # Out folders
    outpath = 'output/'
    os.makedirs(outpath, exist_ok=True)

    # Watermark?
    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # Hardcoded batches and prompts (can be read from file)
    batch_size = 1
    prompt = 'yellow cat face on a bench at the partk'

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # Can be different
    C = 4  # Latent channels
    H = 512
    W = 512
    f = 8  # Downsampling factor
    shape = [C, H // f, W // f]
    diff_steps = 100

    # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    # it should be higher -> closer to prompt
    scale = 9

    # Init image and load mask :)
    init_image, mask = load_img_mask('samples/sample_inpaint.png', 'samples/sample_inpaint_mask.png', f)
    print(f'>>> Loaded img and mask with shape {init_image.shape}, {mask.shape}')

    # We save masked image
    masked_image = init_image * (mask < 0.5)
    masked_image = repeat(masked_image.to(device=device), "1 ... -> n ...", n=batch_size)
    mask = repeat(mask.to(device=device), "1 ... -> n ...", n=batch_size)

    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size).to(device)  # Propag. over batch

    start_code = np.random.randn(batch_size, *shape)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    sampler.make_schedule(ddim_num_steps=diff_steps, ddim_eta=ddim_eta, verbose=False)

    batch = {
        "image": init_image,
        "txt": batch_size * [prompt],
        "mask": mask,
        "masked_image": masked_image,
    }

    # Warmup
    uc = None
    with torch.no_grad(), model.ema_scope():
        c = model.get_learned_conditioning(batch['txt'])

        # Setup the rest of the conditioning
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [batch_size, *shape]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                # This might not be relevant anymore, re test
                # TODO: They will be concat so if wrong dim it won't work!!!
                # everything but masked_image is reshaped to output dim :O
                # i will interpolate also image here before applying encoding.
                # todo check what's the correct way :)
                # TODO ?
                # cc = torch.nn.functional.interpolate(cc, size=(H, W))

                cc = model.get_first_stage_encoding(
                            model.encode_first_stage(cc))

            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # conditioning!
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        uc_cross = None
        if scale != 1.0:
            uc_cross = model.get_learned_conditioning(batch_size * [""])
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        samples = sampler.decode(start_code, cond, diff_steps,
                                 unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc_full)
                                 
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1
            sample_count += 1
    # =======