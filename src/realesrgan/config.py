import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

GT_FOLDER_TRAINING = os.path.join('..', '..', 'dataset', 'training', 'hr')
LQ_FOLDER_TRAINING = os.path.join('..', '..', 'dataset', 'training', 'lr')
GT_FOLDER_VALIDATION = os.path.join('..', '..', 'dataset', 'validation', 'hr')
LQ_FOLDER_VALIDATION = os.path.join('..', '..', 'dataset', 'validation', 'lr')
GT_FOLDER_TESTING = os.path.join('..', '..', 'dataset', 'testing', 'hr')
LQ_FOLDER_TESTING = os.path.join('..', '..', 'dataset', 'testing', 'lr')

OCR_PATH = os.path.join('..', '..', 'saved_models', 'RodoSol-SR')

EXPERIMENT_FOLDER = os.path.join('..', '..', 'experiments')
os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

SEED = 431313
TRANSFER_LEARNING_BASELINE = False
LOAD_MODEL = True
SAVE_MODEL = True
os.makedirs(os.path.join('..', '..', 'checkpoints'), exist_ok=True)
CHECKPOINT_GEN = os.path.join('..', '..', 'checkpoints', 'gen.pth')
CHECKPOINT_DISC = os.path.join('..', '..', 'checkpoints', 'disc.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 160
BATCH_SIZE = 16
NUM_WORKERS = 16
HIGH_RES = 160
LOW_RES = HIGH_RES // 2
IMG_CHANNELS = 3
MAX_IMAGES_EXAMPLES = 20

NET_D_ITERS = 1
NET_D_INIT_ITERS = 0

PERCEPTUAL_LOSS_WEIGHT = 1.0
GAN_LOSS_WEIGHT = 1e-1

highres_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

lowres_transform = A.Compose([
    A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
])

both_transform = A.Compose([
    A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])
