from mmdet.apis import init_detector, inference_detector
import warnings
from pytorch_lightning import seed_everything
warnings.filterwarnings("ignore")
import argparse, os
import PIL
import torch
from datetime import datetime

import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import torchvision
import random
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torch.optim as optim
from ldm.modules.diffusionmodules.openaimodel import clear_feature_dic,get_feature_dic
from ldm.models.seg_module import Segmodule


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

class VOCColorize(object):
    def __init__(self, n):
        self.cmap = color_map(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
    
def color_map(N, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(h,w)
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
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
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
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
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
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
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
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
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
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
    parser.add_argument(
        "--save_name",
        type=str,
        help="the save dir name",
        default=""
    )
    
    parser.add_argument(
        "--class_split",
        type=int,
        help="the class split: 1,2,3",
        default=1
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="the type of training data: single, two, random",
        default="random"
    )
    
    opt = parser.parse_args()
    seed_everything(opt.seed)

    class_coco={}
    f=open("mmdetection/demo/coco_80_class.txt","r")
    count=0
    for line in f.readlines():
        c_name=line.split("\n")[0]
        class_coco[c_name]=count
        count+=1
    print("class_coco=",class_coco)
    
    pascal_file="VOC/class_split"+str(opt.class_split)+".csv"
    
    class_total=[]
    f=open(pascal_file,"r")
    count=0
    for line in f.readlines():
        count+=1
        class_total.append(line.split(",")[0])

    class_train=class_total[:15]
    class_test=class_total[15:]
    print("class_train=",class_train)
    print("class_test=",class_test)
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config_file = 'mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
    checkpoint_file = 'mmdetection/checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'

    pretrain_detector = init_detector(config_file, checkpoint_file, device=device)
    starttime = datetime.now()
    seg_module=Segmodule().to(device)
    model = load_model_from_config(config, f"{opt.ckpt}")
    endtime = datetime.now()
    print ("the time of load_model_from_config is",(endtime - starttime).seconds)
    
    model = model.to(device)

    
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    print('***********************   begin   **********************************')
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    learning_rate = 1e-5 
    total_epoch = 500000
    
    ckpt_dir = os.path.join(save_dir, opt.save_name+'run-'+current_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    
    g_optim = optim.Adam(
                [{"params": seg_module.parameters()},],
                lr=learning_rate
              )

    loss_fn = nn.BCEWithLogitsLoss()
    
    print("Start training with maximum {0} iterations.".format(total_epoch))
    
    start_code = None
    if opt.fixed_code:
        print('start_code')
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)  

    
    batch_size = opt.n_samples
    train_data_choice=opt.train_data
    for j in range(total_epoch):
        print('Epoch ' +  str(j) + '/' + str(total_epoch))
        if not opt.from_file:
            
            trainclass_list=[]
            text="a photograph of a "

            if train_data_choice=="random":
                single_or_two=random.randint(0,1)

                if single_or_two==0:
                    select_class_index1=random.randint(0,len(class_train)-1)
                    select_class1=class_train[select_class_index1]
                    prompt = text+select_class1
                    trainclass_list.append(select_class1)
                    
                elif single_or_two==1:
                    select_class_index1=random.randint(0,len(class_train)-1)
                    select_class_index2=random.randint(0,len(class_train)-1)
                    select_class1=class_train[select_class_index1]
                    select_class2=class_train[select_class_index2]
                    trainclass_list.append(select_class1)
                    trainclass_list.append(select_class2)
                    prompt = text+select_class1+" and a "+select_class2
                    
            elif train_data_choice=="two":
                
                select_class_index1=random.randint(0,len(class_train)-1)
                select_class_index2=random.randint(0,len(class_train)-1)
                select_class1=class_train[select_class_index1]
                select_class2=class_train[select_class_index2]
                trainclass_list.append(select_class1)
                trainclass_list.append(select_class2)
                prompt = text+select_class1+" and a "+select_class2
    
            elif train_data_choice=="single":
            
                select_class_index1=random.randint(0,len(class_train)-1)
                select_class1=class_train[select_class_index1]
                trainclass_list.append(select_class1)
                prompt = text+select_class1
               
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
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
                diffusion_features=get_feature_dic()
                
                x_sample_list=[]
                for i in range(x_samples_ddim.size()[0]):
                    x_sample = torch.clamp((x_samples_ddim[i] + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    x_sample_list.append(x_sample)
                
                result = inference_detector(pretrain_detector, x_sample_list)
                seg_result_list=[]
                for i in range(len(result)):
                    _,seg_result=result[i]
                    seg_result_list.append(seg_result)
                
                loss=[]

                
                text_embeddding_list=[]
                for trainclass in trainclass_list:

                    class_index=class_coco[trainclass]
                    class_name=trainclass
                    query_text="a photograph of a "+class_name
                    c_split = model.cond_stage_model.tokenizer.tokenize(query_text)
                    
                    sen_text_embedding=model.get_learned_conditioning(query_text)
                    
                    class_embedding=sen_text_embedding[:,5:len(c_split)+1,:]
                    
                    if class_embedding.size()[1] > 1:
                        class_embedding = torch.unsqueeze(class_embedding.mean(1),1)
                  
                    text_embeddding_list.append(class_embedding)
                text_embedding=torch.cat(text_embeddding_list,1)
                text_embedding=text_embedding.repeat(batch_size,1,1)
               
                total_pred_seg=seg_module(diffusion_features,text_embedding)
                
                for b_index in range(batch_size):
                    if b_index==0 and j%200 ==0:
                        Image.fromarray(x_sample_list[b_index].astype(np.uint8)).save(os.path.join(ckpt_dir, 
                                                                        'training/'+ str(b_index)+'viz_sample_{0:05d}.png'.format(j)))
                    for train_class_index in range(len(trainclass_list)):
                        trainclass=trainclass_list[train_class_index]
                        class_index=class_coco[trainclass]
                        class_name=trainclass
                        pred_seg=torch.unsqueeze(total_pred_seg[b_index,train_class_index,:,:],0).unsqueeze(0)
                       
                        label_pred_prob = torch.sigmoid(pred_seg)
                        label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
                        label_pred_mask[label_pred_prob>0.5] = 1
                        annotation_pred = label_pred_mask[0][0].cpu()
                        if b_index==0 and j%200 ==0:
                            torchvision.utils.save_image(annotation_pred, os.path.join(ckpt_dir, 
                                                            'training/'+ str(b_index)+'viz_sample_{0:05d}_pred_seg'.format(j)+class_name+'.png'), normalize=True, scale_each=True)
                            
                        if len(seg_result_list[b_index][class_index])==0:
                            print("pretrain detector fail to detect the object in the class:",class_name)
                        else:
                            seg=torch.from_numpy(seg_result_list[b_index][class_index][0].astype(int))
                            seg=seg.float().unsqueeze(0).unsqueeze(0).cuda()
                            
                            loss.append(loss_fn(pred_seg, seg))
                            
                            annotation_pred_gt = seg[0][0].cpu()
                            
                            if b_index==0 and j%200 ==0:
                                viz_tensor2 = torch.cat([annotation_pred_gt, annotation_pred], axis=1)
                                
                                torchvision.utils.save_image(viz_tensor2, os.path.join(ckpt_dir, 
                                                                    'training/'+ str(b_index)+'viz_sample_{0:05d}_seg'.format(j)+class_name+'.png'), normalize=True, scale_each=True)
                                
                if len(loss)==0:
                    pass
                else:
                    total_loss=0
                    for i in range(len(loss)):
                        total_loss+=loss[i]
                    total_loss/=batch_size
                    g_optim.zero_grad()
                    total_loss.backward()
                    g_optim.step()
                    
                    writer.add_scalar('train/loss', total_loss.item(), global_step=j)
                    print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}".format(j, total_epoch, total_loss))
        ##save checkpoint
        if j%200 ==0 and j!=0:
            print("Saving latest checkpoint to",ckpt_dir)
            torch.save(seg_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_latest.pth'))
        if j%5000==0  and j!=0:
            print("Saving latest checkpoint to",ckpt_dir)
            torch.save(seg_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))  
        
if __name__ == "__main__":
    main()
