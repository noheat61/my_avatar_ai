import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from CartoonStyleGAN.utils import tensor2image, save_image
import CartoonStyleGAN.lpips
from CartoonStyleGAN.model import Generator, Encoder


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


class kwargs:
    pass


def main(factor, ckpt, e_ckpt, files):

    setattr(kwargs, "factor_name", factor)
    setattr(kwargs, "ckpt", ckpt)
    setattr(kwargs, "e_ckpt", e_ckpt)
    setattr(kwargs, "files", files)
    setattr(kwargs, "size", 256)
    setattr(kwargs, "truncation", 0.7)
    setattr(kwargs, "lr", 0.01)
    setattr(kwargs, "noise", 0.05)
    setattr(kwargs, "noise_ramp", 0.75)
    setattr(kwargs, "step", 1000)
    setattr(kwargs, "noise_regularize", 1e5)
    setattr(kwargs, "w_plus", True)
    setattr(kwargs, "mse", 0)
    setattr(kwargs, "vgg", 1.0)
    setattr(kwargs, "project_name", "project")
    setattr(kwargs, "lr_rampup", 0.05)
    setattr(kwargs, "lr_rampdown", 0.25)
    setattr(kwargs, "device", "cuda" if torch.cuda.is_available() else "cpu")

    if kwargs.files is None:
        return False

    n_mean_latent = 10000

    # Load Real Images
    resize = min(kwargs.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    for imgfile in kwargs.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(kwargs.device)

    # -------------
    # Generator
    # -------------

    g_ema = Generator(kwargs.size, 512, 8).to(kwargs.device)
    g_ema.load_state_dict(kwargs.ckpt["g_ema"], strict=False)
    g_ema.eval()

    trunc = g_ema.mean_latent(4096).detach().clone()

    # -------------
    # Encoder
    # -------------

    if kwargs.e_ckpt is not None:
        e_ckpt = kwargs.e_ckpt

        encoder = Encoder(kwargs.size, 512).to(kwargs.device)
        encoder.load_state_dict(e_ckpt["e"])
        encoder.eval()

    # -------------
    # Latent vector
    # -------------

    if kwargs.e_ckpt is not None:
        with torch.no_grad():
            latent_init = encoder(imgs)
        latent_in = latent_init.detach().clone()
    else:
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=kwargs.device)
            latent_out = g_ema.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = (
                (latent_out - latent_mean).pow(2).sum() / n_mean_latent
            ) ** 0.5

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if kwargs.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    # -------------
    # Noise
    # -------------

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    for noise in noises:
        noise.requires_grad = True

    # -------------
    # Loss
    # -------------

    # PerceptualLoss
    percept = CartoonStyleGAN.lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=kwargs.device.startswith("cuda")
    )

    # Optimizer
    if kwargs.e_ckpt is not None:
        optimizer = optim.Adam([latent_in], lr=kwargs.lr)
    else:
        optimizer = optim.Adam([latent_in] + noises, lr=kwargs.lr)

    pbar = tqdm(range(kwargs.step))
    latent_path = []
    proj_images = []

    # Training !

    for i in pbar:

        t = i / kwargs.step
        lr = get_lr(t, kwargs.lr)

        optimizer.param_groups[0]["lr"] = lr

        # fake image
        if kwargs.e_ckpt is not None:
            img_gen, _ = g_ema(
                [latent_in],
                input_is_latent=True,
                truncation=kwargs.truncation,
                truncation_latent=trunc,
                randomize_noise=False,
            )
        else:
            noise_strength = (
                latent_std * kwargs.noise * max(0, 1 - t / kwargs.noise_ramp) ** 2
            )
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        #
        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        # latent
        if kwargs.e_ckpt is not None:
            latent_hat = encoder(img_gen)

        # Loss
        p_loss = percept(img_gen, imgs).sum()
        r_loss = torch.mean((img_gen - imgs) ** 2)
        mse_loss = F.mse_loss(img_gen, imgs)

        n_loss = noise_regularize(noises)

        if kwargs.e_ckpt is not None:
            style_loss = F.mse_loss(latent_hat, latent_init)
            loss = kwargs.vgg * p_loss + r_loss + style_loss + kwargs.mse * mse_loss
        else:
            style_loss = 0.0
            loss = (
                kwargs.vgg * p_loss
                + r_loss
                + kwargs.mse * mse_loss
                + kwargs.noise_regularize * n_loss
            )

        # update
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())
            proj_images.append(img_gen)

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f}; "
                f"reconstruction: {r_loss:.4f}; "
                f"mse_img: {mse_loss.item():.4f}; mse_latent: {style_loss:.4f}; lr: {lr:.4f} |"
            )
        )

    # =============================================

    # -----------------------------------
    # Save image, latent, noise
    # -----------------------------------

    # final generated image
    if kwargs.e_ckpt is not None:
        img_gen, _ = g_ema(
            [latent_path[-1]],
            input_is_latent=True,
            truncation=kwargs.truncation,
            truncation_latent=trunc,
            randomize_noise=None,
        )
    else:
        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

    filename = f"{kwargs.project_name}.pt"
    img_ar = make_image(img_gen)

    images = []
    for i in range(len(proj_images)):
        img = proj_images[i][0]
        for k in range(1, len(proj_images[0])):
            # img : torch.Size([3, 256*num_img, 256])
            img = torch.cat([img, proj_images[i][k]], dim=1)
        images.append(img)

    result_file = {}
    for i, input_name in enumerate(kwargs.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise)

        name = os.path.splitext(os.path.basename(input_name))[0]
        result_file[name] = {
            "r_img": tensor2image(imgs[i]),
            "f_img": tensor2image(img_gen[i]),
            "p_img": tensor2image(torch.cat(images, dim=2)),
            "latent": latent_in[i].unsqueeze(0),
            "noise": noise_single,
            "args": kwargs,
        }

        img_name = os.path.splitext(input_name)[0] + "-project.png"
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

        # img_name = (
        #     os.path.splitext(os.path.basename(input_name))[0]
        #     + "-project-interpolation.png"
        # )
        # save_image(tensor2image(torch.cat(images, dim=2)), size=20, out=img_name)

    torch.save(result_file, filename)
    return True
