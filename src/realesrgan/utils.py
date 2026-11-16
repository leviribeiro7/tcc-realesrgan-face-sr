import os
import re
import random
import shutil
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import config


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def setup_experiment_dirs():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    shutil.copy2(os.path.abspath(config.__file__), os.path.join(config.EXPERIMENT_DIR, "config_snapshot.py"))


def save_checkpoint(model, optimizer, epoch, history_loss, filename):
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history_loss": history_loss,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return model, optimizer, checkpoint["epoch"], checkpoint["history_loss"]


@torch.inference_mode()
def validate_and_log_samples(gen, lpips_fn, epoch: int):
    gen.eval()
    lq_files = sorted(os.listdir(config.LQ_FOLDER_VALIDATION), key=natural_sort_key)

    psnr_list, ssim_list, lpips_list = [], [], []
    epoch_dir = os.path.join(config.SAMPLES_DIR, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    wandb_preview_grid = None

    for idx, fname in enumerate(lq_files):
        lq_bgr = cv2.imread(os.path.join(config.LQ_FOLDER_VALIDATION, fname), cv2.IMREAD_COLOR)
        hr_bgr = cv2.imread(os.path.join(config.GT_FOLDER_VALIDATION, fname), cv2.IMREAD_COLOR)

        lq_rgb = cv2.cvtColor(lq_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        lq_tensor = torch.from_numpy(np.transpose(lq_rgb, (2, 0, 1))).unsqueeze(0).to(config.DEVICE)
        hr_tensor = torch.from_numpy(np.transpose(hr_rgb, (2, 0, 1))).unsqueeze(0).to(config.DEVICE)

        sr_tensor = gen(lq_tensor).clamp(0.0, 1.0)

        lpips_val = lpips_fn(sr_tensor * 2.0 - 1.0, hr_tensor * 2.0 - 1.0).item()
        lpips_list.append(lpips_val)

        sr_rgb_u8 = (sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        hr_rgb_u8 = (hr_rgb * 255.0).round().astype(np.uint8)

        psnr_list.append(compute_psnr(hr_rgb_u8, sr_rgb_u8, data_range=255))
        ssim_list.append(compute_ssim(hr_rgb_u8, sr_rgb_u8, channel_axis=2, data_range=255))

        if idx < config.MAX_IMAGES_EXAMPLES:
            bicubic_bgr = cv2.resize(lq_bgr, (config.HIGH_RES, config.HIGH_RES), interpolation=cv2.INTER_CUBIC)
            sr_bgr_u8 = cv2.cvtColor(sr_rgb_u8, cv2.COLOR_RGB2BGR)
            comparison = np.concatenate([bicubic_bgr, sr_bgr_u8, hr_bgr], axis=1)
            cv2.imwrite(os.path.join(epoch_dir, fname), comparison)
            if idx == 0:
                wandb_preview_grid = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)

    gen.train()
    return {
        "val_psnr": float(np.mean(psnr_list)),
        "val_ssim": float(np.mean(ssim_list)),
        "val_lpips": float(np.mean(lpips_list)),
        "preview_image": wandb_preview_grid,
    }
