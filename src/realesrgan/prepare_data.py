import os
import glob
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
import config


def apply_degradation(hr_img: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """
    Gera uma imagem LR 64x64 a partir de uma HR 128x128 (BGR).
    - 'light': Desfoque Gaussiano + Downsampling Bicúbico.
    - 'severe': Desfoque Gaussiano + Downsampling + Ruído Gaussiano + Compressão JPEG.
    """
    kernel_size = 5 if mode == "severe" else 3
    sigma = float(rng.uniform(1.0, 2.2)) if mode == "severe" else 1.0
    blurred = cv2.GaussianBlur(hr_img, (kernel_size, kernel_size), sigmaX=sigma)

    lr = cv2.resize(blurred, (config.LOW_RES, config.LOW_RES), interpolation=cv2.INTER_CUBIC)

    if mode == "severe":
        noise_sigma = float(rng.uniform(5.0, 15.0))
        noise = rng.normal(0, noise_sigma, lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        jpeg_quality = int(rng.integers(35, 70))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode(".jpg", lr, encode_param)
        lr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    return lr


def collect_all_images(raw_dir: str) -> list:
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))
    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {raw_dir}. Verifique o download do Kaggle.")
    return image_paths


def process_split(file_list: list, gt_out: str, lq_out: str, mode: str, split_seed: int):
    os.makedirs(gt_out, exist_ok=True)
    os.makedirs(lq_out, exist_ok=True)
    rng = np.random.default_rng(split_seed)

    for idx, img_path in enumerate(tqdm(file_list, desc=f"Preparando {os.path.basename(os.path.dirname(gt_out))}")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        hr = cv2.resize(img, (config.HIGH_RES, config.HIGH_RES), interpolation=cv2.INTER_AREA)
        lr = apply_degradation(hr, mode=mode, rng=rng)

        filename = f"face_{idx:05d}.png"
        cv2.imwrite(os.path.join(gt_out, filename), hr)
        cv2.imwrite(os.path.join(lq_out, filename), lr)


def main():
    if os.path.exists(config.PROCESSED_DATASET_DIR):
        shutil.rmtree(config.PROCESSED_DATASET_DIR)

    all_images = collect_all_images(config.RAW_DATASET_DIR)
    print(f"Total de imagens encontradas no dataset bruto: {len(all_images)}")

    py_rng = random.Random(config.SEED)
    py_rng.shuffle(all_images)

    # 1. Separa primeiro o conjunto cego de teste (sempre as mesmas 300 imagens)
    test_files = all_images[:config.TEST_SET_SIZE]
    remaining_files = all_images[config.TEST_SET_SIZE:]

    if len(remaining_files) < config.DATASET_SIZE:
        raise ValueError(
            f"DATASET_SIZE={config.DATASET_SIZE} excede as {len(remaining_files)} imagens disponíveis após separar o teste."
        )

    # 2. Fatia o volume do experimento (500, 1500 ou 3000) e divide em treino/validação
    exp_pool = remaining_files[:config.DATASET_SIZE]
    val_count = int(config.DATASET_SIZE * config.VAL_RATIO)
    val_files = exp_pool[:val_count]
    train_files = exp_pool[val_count:]

    print(f"Experimento: {config.EXPERIMENT_NAME} | Degradação: {config.DEGRADATION_MODE}")
    print(f"Divisão -> Treino: {len(train_files)} | Validação: {len(val_files)} | Teste (Fixo): {len(test_files)}")

    process_split(train_files, config.GT_FOLDER_TRAINING, config.LQ_FOLDER_TRAINING, config.DEGRADATION_MODE, config.SEED + 1)
    process_split(val_files, config.GT_FOLDER_VALIDATION, config.LQ_FOLDER_VALIDATION, config.DEGRADATION_MODE, config.SEED + 2)
    process_split(test_files, config.GT_FOLDER_TESTING, config.LQ_FOLDER_TESTING, config.DEGRADATION_MODE, config.SEED + 3)
    print("Preparação do dataset concluída com sucesso.")


if __name__ == "__main__":
    main()
