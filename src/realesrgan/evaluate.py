import os
import cv2
import numpy as np
import pandas as pd
import torch
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import config
from model import Generator
from utils import natural_sort_key, seed_everything


def extract_image_attributes(hr_bgr: np.ndarray) -> dict:
    """
    Extrai atributos visuais quantitativos da imagem Ground Truth
    para análise estratificada de erros na fase de Análise de Dados.
    """
    gray = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "gt_brightness": brightness,
        "gt_contrast": contrast,
        "gt_sharpness": sharpness_laplacian,
    }


@torch.inference_mode()
def main():
    seed_everything(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    test_grids_dir = os.path.join(config.RESULTS_DIR, "test_visual_grids")
    os.makedirs(test_grids_dir, exist_ok=True)

    gen = Generator(scale=config.SCALE, num_block=23).to(config.DEVICE)
    ckpt_path = config.CHECKPOINT_GEN_BEST if os.path.exists(config.CHECKPOINT_GEN_BEST) else config.CHECKPOINT_GEN_LATEST
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()
    print(f"Checkpoint carregado: {ckpt_path} (Época {checkpoint['epoch']})")

    lpips_fn = lpips.LPIPS(net="alex").to(config.DEVICE).eval()
    test_files = sorted(os.listdir(config.LQ_FOLDER_TESTING), key=natural_sort_key)

    records = []
    for idx, fname in enumerate(tqdm(test_files, desc=f"Avaliando Teste Cego ({config.EXPERIMENT_NAME})")):
        lq_bgr = cv2.imread(os.path.join(config.LQ_FOLDER_TESTING, fname), cv2.IMREAD_COLOR)
        hr_bgr = cv2.imread(os.path.join(config.GT_FOLDER_TESTING, fname), cv2.IMREAD_COLOR)

        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hr_rgb_u8 = (hr_rgb * 255.0).round().astype(np.uint8)
        hr_tensor = torch.from_numpy(np.transpose(hr_rgb, (2, 0, 1))).unsqueeze(0).to(config.DEVICE)

        # 1. Baseline Determinístico: Interpolação Bicúbica
        bicubic_bgr = cv2.resize(lq_bgr, (config.HIGH_RES, config.HIGH_RES), interpolation=cv2.INTER_CUBIC)
        bicubic_rgb = cv2.cvtColor(bicubic_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bicubic_rgb_u8 = (bicubic_rgb * 255.0).round().astype(np.uint8)
        bicubic_tensor = torch.from_numpy(np.transpose(bicubic_rgb, (2, 0, 1))).unsqueeze(0).to(config.DEVICE)

        bic_psnr = compute_psnr(hr_rgb_u8, bicubic_rgb_u8, data_range=255)
        bic_ssim = compute_ssim(hr_rgb_u8, bicubic_rgb_u8, channel_axis=2, data_range=255)
        bic_lpips = lpips_fn(bicubic_tensor * 2.0 - 1.0, hr_tensor * 2.0 - 1.0).item()

        # 2. Modelo Generativo: Real-ESRGAN
        lq_rgb = cv2.cvtColor(lq_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lq_tensor = torch.from_numpy(np.transpose(lq_rgb, (2, 0, 1))).unsqueeze(0).to(config.DEVICE)
        sr_tensor = gen(lq_tensor).clamp(0.0, 1.0)
        sr_rgb_u8 = (sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

        sr_psnr = compute_psnr(hr_rgb_u8, sr_rgb_u8, data_range=255)
        sr_ssim = compute_ssim(hr_rgb_u8, sr_rgb_u8, channel_axis=2, data_range=255)
        sr_lpips = lpips_fn(sr_tensor * 2.0 - 1.0, hr_tensor * 2.0 - 1.0).item()

        attrs = extract_image_attributes(hr_bgr)
        records.append({
            "experiment": config.EXPERIMENT_NAME,
            "degradation": config.DEGRADATION_MODE,
            "dataset_size": config.DATASET_SIZE,
            "filename": fname,
            "bicubic_psnr": bic_psnr,
            "bicubic_ssim": bic_ssim,
            "bicubic_lpips": bic_lpips,
            "realesrgan_psnr": sr_psnr,
            "realesrgan_ssim": sr_ssim,
            "realesrgan_lpips": sr_lpips,
            **attrs,
        })

        if idx < 30:
            sr_bgr_u8 = cv2.cvtColor(sr_rgb_u8, cv2.COLOR_RGB2BGR)
            grid = np.concatenate([bicubic_bgr, sr_bgr_u8, hr_bgr], axis=1)
            cv2.imwrite(os.path.join(test_grids_dir, fname), grid)

    df = pd.DataFrame(records)
    csv_path = os.path.join(config.RESULTS_DIR, f"test_metrics_{config.EXPERIMENT_NAME}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nMétricas por imagem salvas em: {csv_path}")
    print(f"Bicúbica Média    -> PSNR: {df['bicubic_psnr'].mean():.2f} | SSIM: {df['bicubic_ssim'].mean():.4f} | LPIPS: {df['bicubic_lpips'].mean():.4f}")
    print(f"Real-ESRGAN Média -> PSNR: {df['realesrgan_psnr'].mean():.2f} | SSIM: {df['realesrgan_ssim'].mean():.4f} | LPIPS: {df['realesrgan_lpips'].mean():.4f}")


if __name__ == "__main__":
    main()
