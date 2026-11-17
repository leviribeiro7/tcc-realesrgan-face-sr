import os
import torch

# 1. PAINEL DE CONTROLE DO EXPERIMENTO
EXPERIMENT_NAME = "E3_1500imgs_severe"
DATASET_SIZE = 1500
DEGRADATION_MODE = "severe"

# 2. REPRODUTIBILIDADE E PARTICIONAMENTO FIXO
SEED = 42
TEST_SET_SIZE = 300
VAL_RATIO = 0.15

# 3. ARQUITETURA E HIPERPARAMETROS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIGH_RES = 128
LOW_RES = HIGH_RES // 2
SCALE = 2
IMG_CHANNELS = 3

LOAD_MODEL = False
SAVE_MODEL = True
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 2

PIXEL_LOSS_WEIGHT = 1.0
PERCEPTUAL_LOSS_WEIGHT = 1.0
GAN_LOSS_WEIGHT = 0.1

# 4. DETECCAO AUTOMATICA DE AMBIENTE (KAGGLE OU COLAB)
IS_KAGGLE = os.path.exists("/kaggle/working")

RAW_DATASET_DIR = "/kaggle/input" if IS_KAGGLE else "/content/dataset_original"
PROCESSED_DATASET_DIR = "/kaggle/working/dataset_processed" if IS_KAGGLE else "/content/dataset_processed"
DRIVE_BASE_DIR = "/kaggle/working/TCC_RealESRGAN" if IS_KAGGLE else "/content/drive/MyDrive/TCC_RealESRGAN"

GT_FOLDER_TRAINING = os.path.join(PROCESSED_DATASET_DIR, "training", "hr")
LQ_FOLDER_TRAINING = os.path.join(PROCESSED_DATASET_DIR, "training", "lr")
GT_FOLDER_VALIDATION = os.path.join(PROCESSED_DATASET_DIR, "validation", "hr")
LQ_FOLDER_VALIDATION = os.path.join(PROCESSED_DATASET_DIR, "validation", "lr")
GT_FOLDER_TESTING = os.path.join(PROCESSED_DATASET_DIR, "testing", "hr")
LQ_FOLDER_TESTING = os.path.join(PROCESSED_DATASET_DIR, "testing", "lr")

EXPERIMENT_DIR = os.path.join(DRIVE_BASE_DIR, EXPERIMENT_NAME)
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(EXPERIMENT_DIR, "validation_samples")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

CHECKPOINT_GEN_LATEST = os.path.join(CHECKPOINT_DIR, "gen_latest.pth.tar")
CHECKPOINT_DISC_LATEST = os.path.join(CHECKPOINT_DIR, "disc_latest.pth.tar")
CHECKPOINT_GEN_BEST = os.path.join(CHECKPOINT_DIR, "gen_best_lpips.pth.tar")

MAX_IMAGES_EXAMPLES = 16

# 5. TELEMETRIA WEIGHTS & BIASES
USE_WANDB = True
WANDB_PROJECT = "tcc-realesrgan-face-sr"
