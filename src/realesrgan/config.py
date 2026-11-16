import os
import torch

# ==============================================================================
# 1. PAINEL DE CONTROLE DO EXPERIMENTO (ALTERE APENAS ESTES PARA E1, E2, E3, E4)
# ==============================================================================
EXPERIMENT_NAME = "E1_500imgs_light"   # E1_500imgs_light | E2_500imgs_severe | E3_1500imgs_severe | E4_3000imgs_severe
DATASET_SIZE = 500                     # Volume treino+validação: 500 (E1/E2), 1500 (E3), 3000 (E4)
DEGRADATION_MODE = "light"             # "light" (Blur + Bicúbica) ou "severe" (Blur + Bicúbica + Ruído + JPEG)

# ==============================================================================
# 2. REPRODUTIBILIDADE E PARTICIONAMENTO FIXO
# ==============================================================================
SEED = 42
TEST_SET_SIZE = 300                    # Conjunto de teste cego fixo e idêntico em TODOS os experimentos
VAL_RATIO = 0.15                       # 15% de DATASET_SIZE para validação, 85% para treinamento

# ==============================================================================
# 3. ARQUITETURA E HIPERPARÂMETROS (FIXOS ENTRE EXPERIMENTOS)
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIGH_RES = 128
LOW_RES = HIGH_RES // 2                # 64x64
SCALE = 2
IMG_CHANNELS = 3

LOAD_MODEL = False                     # Sempre False para treinar do zero a cada experimento
SAVE_MODEL = True
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 2                        # Ideal para as 2 vCPUs do Google Colab

# Pesos da Função de Perda (Gerador)
PIXEL_LOSS_WEIGHT = 1.0
PERCEPTUAL_LOSS_WEIGHT = 1.0
GAN_LOSS_WEIGHT = 0.1

# ==============================================================================
# 4. CAMINHOS (SSD LOCAL DO COLAB PARA DADOS, GOOGLE DRIVE PARA ARTEFATOS)
# ==============================================================================
RAW_DATASET_DIR = "/content/dataset_original"
PROCESSED_DATASET_DIR = "/content/dataset_processed"

GT_FOLDER_TRAINING = os.path.join(PROCESSED_DATASET_DIR, "training", "hr")
LQ_FOLDER_TRAINING = os.path.join(PROCESSED_DATASET_DIR, "training", "lr")
GT_FOLDER_VALIDATION = os.path.join(PROCESSED_DATASET_DIR, "validation", "hr")
LQ_FOLDER_VALIDATION = os.path.join(PROCESSED_DATASET_DIR, "validation", "lr")
GT_FOLDER_TESTING = os.path.join(PROCESSED_DATASET_DIR, "testing", "hr")
LQ_FOLDER_TESTING = os.path.join(PROCESSED_DATASET_DIR, "testing", "lr")

# Diretório Persistente no Google Drive
DRIVE_BASE_DIR = "/content/drive/MyDrive/TCC_RealESRGAN"
EXPERIMENT_DIR = os.path.join(DRIVE_BASE_DIR, EXPERIMENT_NAME)
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(EXPERIMENT_DIR, "validation_samples")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

CHECKPOINT_GEN_LATEST = os.path.join(CHECKPOINT_DIR, "gen_latest.pth.tar")
CHECKPOINT_DISC_LATEST = os.path.join(CHECKPOINT_DIR, "disc_latest.pth.tar")
CHECKPOINT_GEN_BEST = os.path.join(CHECKPOINT_DIR, "gen_best_lpips.pth.tar")

MAX_IMAGES_EXAMPLES = 16

# ==============================================================================
# 5. TELEMETRIA WEIGHTS & BIASES
# ==============================================================================
USE_WANDB = True
WANDB_PROJECT = "tcc-realesrgan-face-sr"
