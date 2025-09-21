import os
import random
from collections import OrderedDict

import config
import numpy as np
import torch
from dataset import PairedImageDataset
from loss import L1Loss, OCRFeatureExtractor, PerceptualLoss
from realesrgan.model import (
    Generator,
    UNetDiscriminatorSN,
    VGGStyleDiscriminator,
)
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_checkpoint, plot_examples, save_checkpoint

torch.backends.cudnn.benchmark = True


def train_generator(disc, opt_gen, pix, fake, loss_dict, vgg_loss, high_res_to_device, bce):
    opt_gen.zero_grad()

    for p in disc.parameters():
        p.requires_grad = False

    fake_g_pred = disc(
        fake
    )

    l_g_total = 0

    l_g_pix = pix(fake, high_res_to_device)
    l_g_total = l_g_total + l_g_pix
    loss_dict['l_g_pix'] = l_g_pix

    l_g_percep = vgg_loss(
        fake, high_res_to_device
    )
    l_g_total = l_g_total + l_g_percep
    loss_dict['l_g_percep'] = l_g_percep

    l_g_gan = config.GAN_LOSS_WEIGHT * bce(
        fake_g_pred, torch.ones_like(fake_g_pred)
    )
    l_g_total = l_g_total + l_g_gan
    loss_dict['l_g_gan'] = l_g_gan

    l_g_total.backward()
    opt_gen.step()

    print(f'l_g_pix: {loss_dict["l_g_pix"].item()} ; \
          l_g_percep: {loss_dict["l_g_percep"].item()} ; \
            l_g_gan: {loss_dict["l_g_gan"].item()}'
    )

    return loss_dict


def train_discriminator(disc, opt_disc, high_res_to_device, bce, loss_dict, fake):
    opt_disc.zero_grad()

    for p in disc.parameters():
        p.requires_grad = True

    # Real
    real_d_pred = disc(
        high_res_to_device
    )
    l_d_real = bce(
        real_d_pred,
        torch.ones_like(real_d_pred) - 0.1 * torch.rand_like(real_d_pred),
    )
    loss_dict['l_d_real'] = l_d_real
    loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
    l_d_real.backward()

    # Fake
    fake_d_pred = disc(
        fake.detach().clone()
    )
    l_d_fake = bce(
        fake_d_pred, torch.zeros_like(fake_d_pred)
    )
    loss_dict['l_d_fake'] = l_d_fake
    loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
    l_d_fake.backward()

    opt_disc.step()

    print(f'l_d_real: {loss_dict["l_d_real"].item()} ; \
          out_d_real: {loss_dict["out_d_real"].item()} ; \
            l_d_fake: {loss_dict["l_d_fake"].item()} ; \
                out_d_fake: {loss_dict["out_d_fake"].item()}'
    )

    return loss_dict, real_d_pred, fake_d_pred


def train_fn(
    loader, 
    disc,
    gen,
    opt_gen,
    opt_disc,
    mse,
    pix,
    bce,
    vgg_loss,
    history_loss
):
    loop = tqdm(loader, leave=True)

    for idx, (high_res, low_res) in enumerate(loop):
        high_res_to_device = high_res.to(config.DEVICE)
        low_res_to_device = low_res.to(config.DEVICE)

        fake = gen(low_res_to_device)

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        loss_dict = OrderedDict()
        # if (idx % config.NET_D_ITERS == 0 and idx > config.NET_D_INIT_ITERS):

        # Train Discriminator: max log(D(x)) + log(1-D(G(z)))
        loss_dict, real_d_pred, fake_d_pred = train_discriminator(
            disc,
            opt_disc,
            high_res_to_device,
            bce,
            loss_dict,
            fake
        )

        loss_dict = train_generator(
            disc,
            opt_gen,
            pix,
            fake,
            loss_dict,
            vgg_loss,
            high_res_to_device,
            bce
        )
        history_loss.append(loss_dict)

    return gen, disc, history_loss


def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if config.DEVICE == 'cuda':
        torch.cuda.manual_seed_all(seed)

    dataset = PairedImageDataset(
        gt_folder=config.GT_FOLDER_TRAINING,
        lq_folder=config.LQ_FOLDER_TRAINING
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    gen = Generator(scale=2, num_block=23).to(config.DEVICE)
    disc = VGGStyleDiscriminator(
        num_in_ch=config.IMG_CHANNELS,
        num_feat=64
    ).to(config.DEVICE)

    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999)
    )
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999)
    )

    mse = nn.MSELoss()
    pix = L1Loss(loss_weight=1.0, reduction='mean')
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = PerceptualLoss(layer_weights={'conv1_2': 0.1,
      'conv2_2': 0.1,
      'conv3_4': 1,
      'conv4_4': 1,
      'conv5_4': 1})

    history_loss = []
    initial_epoch = 0
    if config.LOAD_MODEL:
        gen, opt_gen, initial_epoch, history_loss = load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        disc, opt_disc, initial_epoch, history_loss = load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE
        )
    if config.TRANSFER_LEARNING_BASELINE:
        gen.load_state_dict(
            torch.load(os.path.join('..', '..', 'RealESRGAN_x2plus.pth'))['params_ema'],
            strict=True
        )

    for epoch in range(config.NUM_EPOCHS):
        if epoch <= initial_epoch:
            continue
        gen, disc, history_loss = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            mse,
            pix,
            bce,
            vgg_loss,
            history_loss
        )

        plot_examples(config.LQ_FOLDER_VALIDATION, gen, epoch)

        if config.SAVE_MODEL:
            save_checkpoint(
                gen,
                opt_gen,
                epoch,
                history_loss,
                filename=config.CHECKPOINT_GEN
            )
            save_checkpoint(
                disc,
                opt_disc,
                epoch,
                history_loss,
                filename=config.CHECKPOINT_DISC
            )


if __name__ == '__main__':
    main()
