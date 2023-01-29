import argparse, os
import cv2
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import clear_feature_dic,get_feature_dic
from ldm.models.seg_module import Segmodule

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
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

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def plot_mask(img, masks, colors=None, alpha=0.8,indexlist=[0,1]) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        corlor for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    H,W= masks.shape[0],masks.shape[1]
    color_list=[[255,97,0],[128,42,42],[220,220,220],[255,153,18],[56,94,15],[127,255,212],[210,180,140],[221,160,221],[255,0,0],[255,128,0],[255,255,0],[128,255,0],[0,255,0],[0,255,128],[0,255,255],[0,128,255],[0,0,255],[128,0,255],[255,0,255],[255,0,128]]*6
    final_color_list=[np.array([[i]*512]*512) for i in color_list]
    
    background=np.ones(img.shape)*255
    count=0
    colors=final_color_list[indexlist[count]]
    for mask, color in zip(masks, colors):
        color=final_color_list[indexlist[count]]
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha,background*0.4+img*0.6 )
        count+=1
    return img.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of a lion on a mountain top at sunset",
        help="the prompt to render"
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="?",
        default="lion",
        help="the category to ground"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
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
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
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
        default=1,
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
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="stable_diffusion.ckpt",
        help="path to checkpoint of stable diffusion model",
    )
    parser.add_argument(
        "--grounding_ckpt",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    
    tic = time.time()
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.sd_ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    toc = time.time()
    seg_module=Segmodule().to(device)
        
    seg_module.load_state_dict(torch.load(opt.grounding_ckpt, map_location="cpu"), strict=True)
    print('load time:',toc-tic)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    batch_size = opt.n_samples
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                prompt = opt.prompt
                text = opt.category
                trainclass = text
                if not opt.from_file:
                    assert prompt is not None
                    data = [batch_size * [prompt]]

                else:
                    print(f"reading prompts from {opt.from_file}")
                    with open(opt.from_file, "r") as f:
                        data = f.read().splitlines()
                        data = list(chunk(data, batch_size))

                sample_path = os.path.join(outpath, "samples")
                os.makedirs(sample_path, exist_ok=True)

                start_code = None
                if opt.fixed_code:
                    print('start_code')
                    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        clear_feature_dic()
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim,_, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        diffusion_features = get_feature_dic()
                
                        
                        x_sample = torch.clamp((x_samples_ddim[0] + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        
                        Image.fromarray(x_sample.astype(np.uint8)).save("demo/demo.png")    
                        img = x_sample.astype(np.uint8)
                        
                        class_name = trainclass
                        
                        query_text ="a "+prompt.split()[1]+" of a "+class_name
                        c_split = model.cond_stage_model.tokenizer.tokenize(query_text)
                        
                        sen_text_embedding = model.get_learned_conditioning(query_text)
                        class_embedding = sen_text_embedding[:, 5:len(c_split)+1, :]
                        
                        if class_embedding.size()[1] > 1:
                            class_embedding = torch.unsqueeze(class_embedding.mean(1), 1)
                        text_embedding = class_embedding
                    
                        text_embedding = text_embedding.repeat(batch_size, 1, 1)
                        
                                    
                        pred_seg_total = seg_module(diffusion_features, text_embedding)
                    
                        
                        pred_seg = torch.unsqueeze(pred_seg_total[0,0,:,:], 0).unsqueeze(0)
                            
                        label_pred_prob = torch.sigmoid(pred_seg)
                        label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
                        label_pred_mask[label_pred_prob > 0.5] = 1
                        annotation_pred = label_pred_mask[0][0].cpu()
                        
                        mask = annotation_pred.numpy()
                        mask = np.expand_dims(mask, 0)
                        done_image_mask = plot_mask(img, mask, alpha=0.9, indexlist=[0])
                        cv2.imwrite(os.path.join("demo/demo_mask.png"), done_image_mask)
                        
                        torchvision.utils.save_image(annotation_pred, os.path.join("demo/demo_segresult.png"), normalize=True, scale_each=True)



if __name__ == "__main__":
    main()
