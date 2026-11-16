import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    def __init__(self, gt_folder: str, lq_folder: str):
        super().__init__()
        self.gt_folder = gt_folder
        self.lq_folder = lq_folder

        gt_files = sorted(os.listdir(self.gt_folder))
        lq_files = sorted(os.listdir(self.lq_folder))
        assert gt_files == lq_files, "Inconsistência entre arquivos HR e LR."

        self.data = [
            (os.path.join(gt_folder, fname), os.path.join(lq_folder, fname))
            for fname in gt_files
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gt_path, lq_path = self.data[index]

        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        lq_img = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # BGR -> RGB e HWC -> CHW
        gt_tensor = torch.from_numpy(np.transpose(gt_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        lq_tensor = torch.from_numpy(np.transpose(lq_img[:, :, [2, 1, 0]], (2, 0, 1))).float()

        return gt_tensor, lq_tensor
