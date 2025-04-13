
import argparse
from math import prod
import math
import os
import random
import comet_ml
from dotenv import load_dotenv
import einops
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn.utils as utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from vector_quantize_pytorch import FSQ
from zuko.utils import odeint

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import cv2

from emb import TimeEmbed


# Initialize distributed context
dist.init_process_group(backend='nccl')
rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(rank)


def make_beta_schedule(T=100, beta_start=1e-4, beta_end=0.02):
    """
    Return (betas, alphas, alpha_bars) each of shape (T,).
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_image(image_path='image.png'):
    """
    Load an image as a PyTorch tensor with shape B x C x H x W
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Image tensor with shape [1, C, H, W]
    """
    # Open the image file
    img = Image.open(image_path)
    
    # Define transformation to convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
    ])
    
    # Apply transformation
    img_tensor = transform(img)
    
    # Add batch dimension [B, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor

def crop_to_divisible(image_tensor, divisor=16):
    """
    Crop an image tensor to dimensions that are divisible by the specified divisor.
    
    Args:
        image_tensor (torch.Tensor): Image tensor with shape [B, C, H, W]
        divisor (int): The number both height and width should be divisible by
        
    Returns:
        torch.Tensor: Cropped image tensor with shape [B, C, H', W'] where H' and W' are divisible by divisor
    """
    # Get current dimensions
    _, _, height, width = image_tensor.shape
    
    # Calculate new dimensions
    new_height = (height // divisor) * divisor
    new_width = (width // divisor) * divisor
    
    # Crop the image (centered crop)
    h_start = (height - new_height) // 2
    w_start = (width - new_width) // 2
    
    cropped_image = image_tensor[:, :, h_start:h_start+new_height, w_start:w_start+new_width]
    
    return cropped_image

# Load and crop the image
image_tensor = load_image('image.png')
# Resize the image to a width of 1024
print(f"Image tensor shape: {image_tensor.shape}")

_, _, h, w = image_tensor.shape
new_width = 1024
new_height = int(h * (new_width / w))
image_tensor = F.interpolate(image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
#image_tensor = crop_to_divisible(image_tensor, divisor=16)
print(f"Image tensor shape, after resizing: {image_tensor.shape}")

def normalize_image(image_tensor, return_stats=True):
    """
    Normalize an image tensor to have zero mean and unit variance.
    
    Args:
        image_tensor (torch.Tensor): Image tensor with shape [B, C, H, W]
        return_stats (bool): Whether to return normalization statistics for later denormalization
        
    Returns:
        torch.Tensor: Normalized image tensor
        dict (optional): Dictionary containing mean and std for denormalization
    """
    # Calculate mean and std across spatial dimensions (H, W) for each channel
    # Keep batch dimension intact
    mean = image_tensor.mean(dim=(2, 3), keepdim=True)
    std = image_tensor.std(dim=(2, 3), keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
    
    # Normalize the image
    normalized_image = (image_tensor - mean) / std
    
    if return_stats:
        stats = {'mean': mean, 'std': std}
        return normalized_image, stats
    return normalized_image

def denormalize_image(normalized_image, stats):
    """
    Denormalize an image tensor using the provided statistics.
    
    Args:
        normalized_image (torch.Tensor): Normalized image tensor with shape [B, C, H, W]
        stats (dict): Dictionary containing mean and std used for normalization
        
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    # Denormalize the image
    denormalized_image = normalized_image * stats['std'] + stats['mean']
    return denormalized_image

# Normalize the image and save normalization statistics
normalized_image, norm_stats = normalize_image(image_tensor)
print(f"Normalized image tensor shape: {normalized_image.shape}")
print(f"Normalized image mean: {normalized_image.mean():.4f}, std: {normalized_image.std():.4f}")

class CNN(nn.Module):
    def __init__(self, patch_levels=4):
        super().__init__()
        # Input channels (assuming RGB images)
        self.in_channels = 4
        self.out_channels = 3
        self.patch_levels = patch_levels
        # Calculate patch size based on patch_levels
        self.patch_size = 2 ** patch_levels
        
        # Initial number of filters
        initial_filters = 32

        self.time_embed = TimeEmbed(embed_dim=128, time_emb_dim=128)
        # --- Set up beta schedule ---
        # For example, linearly from 1e-4 to 0.02
        self.T = 100
        betas, alphas, alpha_bars = make_beta_schedule(T=self.T)
        # Register them as buffers so they're moved to GPU with .to(device)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        # Time embedding dimension
        self.time_embed = TimeEmbed(embed_dim=128, time_emb_dim=128)

        # Example: your UNet layers
        initial_filters = 32
        
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels + 128, initial_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters, initial_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(initial_filters, initial_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*2, initial_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(initial_filters*2, initial_filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*4, initial_filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(initial_filters*4, initial_filters*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*8, initial_filters*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(initial_filters*8, initial_filters*16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*16, initial_filters*16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv4 = nn.ConvTranspose2d(initial_filters*16, initial_filters*8, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(initial_filters*16, initial_filters*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*8, initial_filters*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(initial_filters*8, initial_filters*4, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(initial_filters*8, initial_filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*4, initial_filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(initial_filters*4, initial_filters*2, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(initial_filters*4, initial_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters*2, initial_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(initial_filters*2, initial_filters, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(initial_filters*2, initial_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters, initial_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(initial_filters, self.out_channels, kernel_size=1)
    
    def forward(self, x, t, list_of_patches=False):
        """
        x shape: (B, 4, H, W) where channel-3 is the mask.
        t shape: (B,) of integers in [0..T-1].
        We embed t and concat it as extra channels. Then do the UNet.
        """
        # If the input is a full image, unfold into patches
        if not list_of_patches:
            x = self.to_patches(x)
            B, C, Hp, Wp, pH, pW = x.shape
            x = einops.rearrange(x, 'b c h w p1 p2 -> (b h w) c p1 p2')
            process_as_patches = True
        else:
            process_as_patches = False

        # (B*, 4, pH, pW)
        # Time embedding
        t_emb = self.time_embed(t)  # shape (B*, 128)
        # broadcast to 2D
        t_emb_2d = t_emb.unsqueeze(-1).unsqueeze(-1)  # (B*, 128, 1, 1)
        t_emb_2d = t_emb_2d.expand(-1, -1, x.shape[-2], x.shape[-1]) # (B*, 128, pH, pW)

        # Concat
        x = torch.cat([x, t_emb_2d], dim=1)  # => (B*, 4+128, pH, pW)

        # Standard UNet forward
        enc1 = self.enc_conv1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool4(enc4)
        
        bottleneck = self.bottleneck(pool4)
        
        up4 = self.upconv4(bottleneck)
        merge4 = torch.cat([enc4, up4], dim=1)
        dec4 = self.dec_conv4(merge4)
        
        up3 = self.upconv3(dec4)
        merge3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec_conv3(merge3)
        
        up2 = self.upconv2(dec3)
        merge2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec_conv2(merge2)
        
        up1 = self.upconv1(dec2)
        merge1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec_conv1(merge1)
        
        out = self.final_conv(dec1)
        # out shape: (B*, 3, pH, pW)
        
        # Re-fold if needed
        if process_as_patches:
            out = einops.rearrange(out, '(b h w) c p1 p2 -> b c h w p1 p2',
                                   b=B, h=Hp, w=Wp)
            out = self.from_patches(out)
        return out

    # ------------------------------------------------------------------
    #                              TRAIN
    # ------------------------------------------------------------------
    def loss(self, x, goal):
        """
        x shape: (B,4,H,W), channel-3 is mask in [0,1].
        We pick a random t, do forward diffusion on masked pixels, 
        then train the model to predict the added noise.
        """
        mask = x[:, 3:4, :, :]
        x[:, :3, :, :][mask == 1] = goal[mask == 1]
        # Unfold
        x_patches = self.to_patches(x)  # (B,4,Hp,Wp,pH,pW)
        B_, C, Hp, Wp, pH, pW = x_patches.shape
        # Flatten patches => (B'* = B*Hp*Wp, 4, pH, pW)
        x_patches = einops.rearrange(x_patches, 'b c h w p1 p2 -> (b h w) c p1 p2')
        batch_size = x_patches.shape[0]
        
        # random t in [0..T-1] for each patch
        t = torch.randint(0, self.T, (batch_size,), device=x_patches.device)

        # Original image portion:
        img_0 = x_patches[:, :3]  # (B', 3, pH, pW)
        mask = x_patches[:, 3:4]  # (B', 1, pH, pW), 1 => "masked" => will be noised

        # forward diffusion at step t:
        #   x_t = sqrt(alpha_bar[t])*img_0 + sqrt(1 - alpha_bar[t])*noise
        # But only for masked pixels, unmasked remain = x_0
        noise = torch.randn_like(img_0)

        alpha_bar_t = self.alpha_bars[t].view(-1,1,1,1)  # shape (B',1,1,1)
        sqrt_ab = alpha_bar_t.sqrt()
        sqrt_1mab = (1.0 - alpha_bar_t).sqrt()

        # noised version for masked pixels
        x_t_masked = sqrt_ab * img_0 + sqrt_1mab * noise
        # keep unmasked
        x_t = mask * x_t_masked + (1 - mask) * img_0

        # Combine back with mask => shape (B',4,pH,pW)
        x_patches_noise = torch.cat([x_t, mask], dim=1)

        # Predict noise
        noise_pred = self.forward(x_patches_noise, t, list_of_patches=True)

        # MSE loss on the masked region
        # Option 1: average MSE over all pixels in the patch (common approach)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    # ------------------------------------------------------------------
    #                              SAMPLE
    # ------------------------------------------------------------------
    def ddpm_inpaint(self, x_0):
        """
        Inpaint the masked region of x_0 by reverse diffusion.
        x_0: shape (B,4,H,W) with channel-3 as mask (1=masked).
        We'll do T steps from t=T-1 down to 0, updating only masked pixels.
        """
        self.eval()
        with torch.no_grad():
            # 1) Unfold
            x_patches = self.to_patches(x_0)
            B_, C, Hp, Wp, pH, pW = x_patches.shape
            x_patches = einops.rearrange(x_patches, 'b c h w p1 p2 -> (b h w) c p1 p2')
            # x_patches: shape (B',4,pH,pW)

            img_0 = x_patches[:, :3]  # original content
            mask = x_patches[:, 3:4]  # 1=masked

            # We'll start from pure noise in the masked region
            x_t = mask * torch.randn_like(img_0) + (1-mask)*img_0

            # Reverse loop
            for i in reversed(range(self.T)):
                t_batch = torch.tensor([i]*x_t.shape[0], device=x_t.device)
                # Combine x_t with mask
                combined = torch.cat([x_t, mask], dim=1)  # (B',4,pH,pW)

                # model predicts noise
                eps_theta = self.forward(combined, t_batch, list_of_patches=True)  # (B',3,pH,pW)

                # Coeffs
                alpha_i = self.alphas[i]
                alpha_bar_i = self.alpha_bars[i]
                # For i>0, alpha_bar_(i-1), etc.
                if i > 0:
                    alpha_bar_prev = self.alpha_bars[i-1]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=x_t.device)  # alpha_bar_-1 = 1

                one_over_sqrt_alpha = 1.0 / alpha_i.sqrt()
                coef_eps = (1.0 - alpha_i)/( (1.0 - alpha_bar_i).sqrt() )

                # x_{t-1} (before adding noise)
                x_noiseless = one_over_sqrt_alpha * (x_t - coef_eps * eps_theta)

                if i > 0:
                    # sigma_t
                    beta_i = self.betas[i]
                    # eq. var term => ...
                    # in typical DDPM: 
                    sigma_i = ((1.0 - alpha_bar_prev)/(1.0 - alpha_bar_i)*beta_i).sqrt()
                    # sample random z
                    z = torch.randn_like(x_t)
                    x_next = x_noiseless + sigma_i * z
                else:
                    x_next = x_noiseless

                # only update the masked portion
                x_t = mask * x_next + (1.0 - mask)*img_0  # keep the unmasked part = x_0

            # Reshape back to full image
            x_t = einops.rearrange(x_t, '(b h w) c p1 p2 -> b c h w p1 p2',
                                   b=B_, h=Hp, w=Wp)
            x_out = self.from_patches(x_t)
        self.train()
        return x_out
    
    def to_patches(self, x):
        assert len(x.shape) == 4
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        return x

    def from_patches(self, x):
        B, C, Hp, Wp, pH, pW = x.shape
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hp * pH, Wp * pW)
        return x



import comet_ml
from tqdm import tqdm


# Create model and move to device
model = CNN(patch_levels=4).to(device)
model = DDP(model, device_ids=[rank])

if rank == 0:
    # Initialize comet.ml experiment
    experiment = comet_ml.Experiment(
        api_key="ZFi08G3WImS7t3E560rj6BTHs",
        project_name="arcy"
    )


max_patch_size = 32

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
def make_sample():
    # Apply random shift
    shift_h = random.randint(0, max_patch_size - 1)
    shift_w = random.randint(0, max_patch_size - 1)
    
    # Roll the image instead of shifting
    rolled_image = torch.roll(normalized_image, shifts=(-shift_h, -shift_w), dims=(2, 3))
    
    # Crop the image to a multiple of the patch size
    cropped_image = rolled_image[:, :, 
                    :max_patch_size * (normalized_image.shape[2] // max_patch_size - 1), 
                    :max_patch_size * (normalized_image.shape[3] // max_patch_size - 1)]
    
    # Apply random flipping
    if random.random() > 0.5:
        cropped_image = torch.flip(cropped_image, dims=[2])  # Horizontal flip
    if random.random() > 0.5:
        cropped_image = torch.flip(cropped_image, dims=[3])  # Vertical flip

    # Apply random color palette swaps/mutations
    if random.random() > 0:
        # Randomly permute color channels
        perm = torch.randperm(3)
        cropped_image = cropped_image[:, perm]
    
    if random.random() > 0:
        # Apply random color shift
        color_shift = torch.randn(1, 3, 1, 1, device=device) * 0.1
        cropped_image = cropped_image + color_shift
    
    if random.random() > 0:
        # Apply random color scaling
        color_scale = torch.rand(1, 3, 1, 1, device=device) * 0.5 + 0.75  # Scale between 0.75 and 1.25
        cropped_image = cropped_image * color_scale
    
    # Add a 4th channel for mask (all zeros)
    batch_size, _, height, width = cropped_image.shape
    mask_channel = torch.zeros(batch_size, 1, height, width, device=device)
    cropped_image_with_mask = torch.cat([cropped_image, mask_channel], dim=1)

    # Generate random mask with 16x16 squares removed with probability 0.3
    batch_size, _, height, width = cropped_image.shape
    
    # Calculate number of 16x16 squares in height and width
    num_squares_h = height // 16
    num_squares_w = width // 16
    
    # Create a mask where each 16x16 square has a 30% chance of being masked
    square_mask = torch.rand(batch_size, 1, num_squares_h, num_squares_w, device=device) < 0.3
    
    # Upsample the mask to the full image size (each square becomes 16x16)
    mask = square_mask.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)
    
    # If the image dimensions aren't divisible by 16, adjust the mask size
    if mask.shape[2] != height or mask.shape[3] != width:
        mask = mask[:, :, :height, :width]
    # Update the mask channel in the image
    cropped_image_with_mask[:, 3:4, :, :] = mask.float()
    # Remove color information where the mask is applied
    cropped_image_with_mask[:, :3, :, :] = cropped_image_with_mask[:, :3, :, :] * (1 - mask.float())
    
    return cropped_image_with_mask, cropped_image

def make_batch(batch_size=16):
    samples = [make_sample() for _ in range(batch_size)]
    batch, goal = zip(*samples)
    batch = torch.cat(batch, dim=0)
    goal = torch.cat(goal, dim=0)
    return batch, goal

def add_right_mask(goal):
    batch_size, channels, height, width = goal.shape
    
    # Create a new wider image with an additional 32 pixels on the right
    extended_width = width + 32
    extended_image = torch.zeros(batch_size, channels, height, extended_width, device=goal.device)
    
    # Copy the original image to the left part of the extended image
    extended_image[:, :, :, :width] = goal
    
    # Create a mask for the right 32 pixels (1 = masked region)
    mask = torch.zeros(batch_size, 1, height, extended_width, device=goal.device)
    mask[:, :, :, -32:] = 1.0
    
    # Add the mask as the 4th channel
    if channels == 3:
        input_image_with_mask = torch.cat([extended_image, mask], dim=1)
    else:
        # If already has 4 channels, update the mask channel
        extended_image[:, 3:4, :, :] = mask
        input_image_with_mask = extended_image
    
    # Remove color information where the mask is applied
    input_image_with_mask[:, :3, :, :] = input_image_with_mask[:, :3, :, :] * (1 - mask)
    return input_image_with_mask


def generate_one_step(input_image_with_mask):
    # Get dimensions of the input image
    batch_size, channels, height, width = input_image_with_mask.shape
    
    # Extract the mask channel
    mask = input_image_with_mask[:, 3:4, :, :]
    
    # Sample random patches until we find ones that meet our criteria
    patch_size = 16
    max_attempts = 10000
    selected_patch = None
    selected_patch_position = None
    
    #print(f"Searching for patches in {height}x{width} image")
    
    # Define valid range for patch coordinates
    max_y = height - patch_size
    max_x = width - patch_size
    
    # Sample random patches until we find enough or reach max attempts
    attempt = 0
    while selected_patch is None and attempt < max_attempts:
        # Sample random position
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        
        # Extract patch and its mask
        patch = input_image_with_mask[:, :, y:y+patch_size, x:x+patch_size]
        patch_mask = mask[:, :, y:y+patch_size, x:x+patch_size]
        
        # Calculate percentage of masked pixels
        masked_percentage = torch.mean(patch_mask.reshape(patch_mask.shape[0], -1))
        
        # Check if patch meets criteria (>30% masked, <70% masked)
        if 0.3 < masked_percentage < 0.7:
            selected_patch = patch
            selected_patch_position = (y, x)
        
        attempt += 1
    
    #print(f"Found 1 patch in {attempt} attempts")
    assert selected_patch is not None, "No suitable patches found"
    assert selected_patch_position is not None, "No suitable patch position found"
    
    with torch.no_grad():
        output = model.module.ddpm_inpaint(selected_patch)

    # Get the position of the selected patch
    y_pos, x_pos = selected_patch_position
    # Create a copy of the input image to modify
    result_image = input_image_with_mask.clone()
    
    # Get the mask for the selected patch
    patch_mask = result_image[:, 3:4, y_pos:y_pos+patch_size, x_pos:x_pos+patch_size]
    
    # Replace the patch in the original image with the model's output, but only where mask is 1
    # Create a masked version of the output
    masked_output = output[:, 0:3, :, :] * patch_mask
    # Create a masked version of the original content
    masked_original = result_image[:, 0:3, y_pos:y_pos+patch_size, x_pos:x_pos+patch_size] * (1 - patch_mask)
    # Combine them
    result_image[:, 0:3, y_pos:y_pos+patch_size, x_pos:x_pos+patch_size] = masked_output + masked_original
    
    # For visualization purposes, set the mask to zero in the area we've filled
    result_image[:, 3:4, y_pos:y_pos+patch_size, x_pos:x_pos+patch_size] = 0
    
    # Update output to be the full image with the patch replaced
    output = result_image
    return output
def generate(goal):
    model.eval()
    start_time = time.time()
    input_image_with_mask = add_right_mask(goal)
    
    # Create a list to store frames for the video
    frames = []
    
    # Add the initial masked image as the first frame
    initial_frame = denormalize_image(input_image_with_mask[:, :3, :, :], norm_stats)
    initial_frame = initial_frame[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    frames.append((initial_frame * 255).astype(np.uint8))
    
    for _ in tqdm(range(100)):
        input_image_with_mask = generate_one_step(input_image_with_mask)
        
        # Add the current state to the video frames
        current_frame = denormalize_image(input_image_with_mask[:, :3, :, :], norm_stats)
        current_frame = current_frame[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        frames.append((current_frame * 255).astype(np.uint8))
    
    # Create a video from the frames
    height, width, _ = frames[0].shape
    video_filename = f'generation_video_{time.time()}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, 10, (width, height))
    
    for frame in frames:
        # OpenCV uses BGR format, so convert from RGB
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video.release()
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    return input_image_with_mask, video_filename

# Make sure model is in training mode
model.train()

# Training loop
for epoch in range(100000):
    # Zero gradients
    optimizer.zero_grad()

    batch, goal = make_batch(batch_size=32)
    loss = model.module.loss(batch, goal)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
        
    # Log to comet.ml
    if epoch % 10 == 0 and rank == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")
        experiment.log_metric("loss", loss.item(), step=epoch)
        
        with torch.no_grad():
            # Convert tensors to images (assuming values in [0,1])
            cropped_image = denormalize_image(batch[:, :3, :, :], norm_stats)
            goal_image = denormalize_image(goal[:, :3, :, :], norm_stats)
            output = model.module.ddpm_inpaint(batch[:1])
            output_img = denormalize_image(output[:, :3, :, :], norm_stats)

            cropped_image = cropped_image[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            goal_image = goal_image[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            output_img = output_img[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            
            experiment.log_image(cropped_image, name=f"input_epoch_{epoch}")
            experiment.log_image(goal_image, name=f"goal_epoch_{epoch}")
            experiment.log_image(output_img, name=f"reconstruction_epoch_{epoch}")
    if epoch % 500 == 0 and rank == 0:
        final_image, video_path = generate(goal)
        # Log the final generated image
        final_output = denormalize_image(final_image[:, :3, :, :], norm_stats)
        final_output = final_output[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        experiment.log_image(final_output, name=f"generated_epoch_{epoch}")
        
        # Log the generation video
        experiment.log_video(video_path, name=f"generation_process_epoch_{epoch}", step=epoch)
        
        # Clean up the video file after logging
        try:
            os.remove(video_path)
        except:
            pass

# End the experiment
experiment.end()
