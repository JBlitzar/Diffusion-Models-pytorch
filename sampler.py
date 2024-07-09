import torch
import torch.nn as nn
from utils import *
from ddpm import Diffusion
from modules import UNet
import time
device = "mps"

diffusion = Diffusion(img_size=64, device=device, noise_steps=1000)

model = UNet().to(device)

model.load_state_dict(torch.load("models/flowers_DDPM_Uncondtional_run2/ckpt.pt"))

sampled_images = diffusion.sample(model, n=10)
save_images(sampled_images, os.path.join("results", "flowers_DDPM_Uncondtional_run2", f"sampled_{int(time.time())}.jpg"))