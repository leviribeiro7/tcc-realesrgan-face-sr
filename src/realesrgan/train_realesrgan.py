import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
import wandb

import config
from dataset import PairedImageDataset
from loss import L1Loss, PerceptualLoss
from model import Generator, UNetDiscriminatorSN
from utils import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    setup_experiment_dirs,
    validate_and_log_samples,
)


def train_one_epoch(loader, disc, gen, opt_gen, opt_disc, pix, bce, vgg_loss):
    gen.train()
    disc.train()
    loop = tqdm(loader, leave=True)

    metrics_acc = {"l_g_pix": 0.0, "l_g_percep": 0.0, "l_g_gan": 0.0, "l_d_real": 0.0, "l_d_fake": 0.0}
    num_batches = len(loader)

    for high_res, low_res in loop:
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        fake = gen(low_res)

        # 1. Treinar Discriminador
        opt_disc.zero_grad()
        for p in disc.parameters():
            p.requires_grad = True

        real_d_pred = disc(high_res)
        l_d_real = bce(real_d_pred, torch.ones_like(real_d_pred) - 0.1 * torch.rand_like(real_d_pred))
        l_d_real.backward()

        fake_d_pred = disc(fake.detach())
        l_d_fake = bce(fake_d_pred, torch.zeros_like(fake_d_pred))
        l_d_fake.backward()
        opt_disc.step()

        # 2. Treinar Gerador
        opt_gen.zero_grad()
        for p in disc.parameters():
            p.requires_grad = False

        fake_g_pred = disc(fake)
        l_g_pix = pix(fake, high_res)
        l_g_percep = vgg_loss(fake, high_res)
        l_g_gan = config.GAN_LOSS_WEIGHT * bce(fake_g_pred, torch.ones_like(fake_g_pred))

        l_g_total = l_g_pix + l_g_percep + l_g_gan
        l_g_total.backward()
        opt_gen.step()

        metrics_acc["l_g_pix"] += l_g_pix.item() / num_batches
        metrics_acc["l_g_percep"] += l_g_percep.item() / num_batches
        metrics_acc["l_g_gan"] += l_g_gan.item() / num_batches
        metrics_acc["l_d_real"] += l_d_real.item() / num_batches
        metrics_acc["l_d_fake"] += l_d_fake.item() / num_batches

        loop.set_postfix(
            pix=f"{l_g_pix.item():.4f}",
            percep=f"{l_g_percep.item():.4f}",
            gan=f"{l_g_gan.item():.4f}",
        )

    return metrics_acc


def main():
    seed_everything(config.SEED)
    setup_experiment_dirs()

    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            name=config.EXPERIMENT_NAME,
            config={
                "dataset_size": config.DATASET_SIZE,
                "degradation_mode": config.DEGRADATION_MODE,
                "high_res": config.HIGH_RES,
                "low_res": config.LOW_RES,
                "num_epochs": config.NUM_EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "seed": config.SEED,
            },
        )

    train_dataset = PairedImageDataset(config.GT_FOLDER_TRAINING, config.LQ_FOLDER_TRAINING)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    gen = Generator(scale=config.SCALE, num_block=23).to(config.DEVICE)
    disc = UNetDiscriminatorSN(num_in_ch=config.IMG_CHANNELS).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))

    pix = L1Loss(loss_weight=config.PIXEL_LOSS_WEIGHT)
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = PerceptualLoss(
        layer_weights={"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1.0, "conv4_4": 1.0, "conv5_4": 1.0},
        perceptual_weight=config.PERCEPTUAL_LOSS_WEIGHT,
    )
    lpips_fn = lpips.LPIPS(net="alex").to(config.DEVICE).eval()

    history = []
    start_epoch = 0
    best_val_lpips = float("inf")

    if config.LOAD_MODEL and os.path.exists(config.CHECKPOINT_GEN_LATEST):
        gen, opt_gen, last_epoch, history = load_checkpoint(
            config.CHECKPOINT_GEN_LATEST, gen, opt_gen, config.LEARNING_RATE
        )
        disc, opt_disc, _, _ = load_checkpoint(
            config.CHECKPOINT_DISC_LATEST, disc, opt_disc, config.LEARNING_RATE
        )
        start_epoch = last_epoch + 1

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n--- Época [{epoch + 1}/{config.NUM_EPOCHS}] | {config.EXPERIMENT_NAME} ---")
        train_metrics = train_one_epoch(train_loader, disc, gen, opt_gen, opt_disc, pix, bce, vgg_loss)
        val_metrics = validate_and_log_samples(gen, lpips_fn, epoch)

        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            "val_psnr": val_metrics["val_psnr"],
            "val_ssim": val_metrics["val_ssim"],
            "val_lpips": val_metrics["val_lpips"],
        }
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(os.path.join(config.RESULTS_DIR, "training_history.csv"), index=False)

        print(
            f"Validação -> PSNR: {val_metrics['val_psnr']:.2f} dB | "
            f"SSIM: {val_metrics['val_ssim']:.4f} | LPIPS: {val_metrics['val_lpips']:.4f}"
        )

        if config.USE_WANDB:
            log_payload = {k: v for k, v in epoch_record.items() if k != "epoch"}
            if val_metrics["preview_image"] is not None:
                log_payload["val_comparison (Bicubic | SR | HR)"] = wandb.Image(val_metrics["preview_image"])
            wandb.log(log_payload, step=epoch)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, epoch, history, config.CHECKPOINT_GEN_LATEST)
            save_checkpoint(disc, opt_disc, epoch, history, config.CHECKPOINT_DISC_LATEST)

            if val_metrics["val_lpips"] < best_val_lpips:
                best_val_lpips = val_metrics["val_lpips"]
                save_checkpoint(gen, opt_gen, epoch, history, config.CHECKPOINT_GEN_BEST)
                print(f"-> Novo Melhor Modelo salvo (LPIPS: {best_val_lpips:.4f})")

    if config.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
