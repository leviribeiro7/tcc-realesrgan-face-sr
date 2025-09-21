import os

import config
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PairedImageDataset(Dataset):
    def __init__(self, gt_folder, lq_folder):
        super().__init__()
        self.gt_folder = gt_folder
        self.lq_folder = lq_folder

        gt_images_names_and_suffix = os.listdir(self.gt_folder)
        lq_images_names_and_suffix = os.listdir(self.lq_folder)
        gt_images_names_and_suffix.sort()
        lq_images_names_and_suffix.sort()

        assert gt_images_names_and_suffix == lq_images_names_and_suffix, "Mismatch between gt images and lq images"

        self.gt_image_paths = [
            os.path.join(gt_folder, image_name_and_suffix)
            for image_name_and_suffix in gt_images_names_and_suffix
        ]
        self.lq_image_paths = [
            os.path.join(lq_folder, image_name_and_suffix)
            for image_name_and_suffix in lq_images_names_and_suffix
        ]

        self.data = list(zip(self.gt_image_paths, self.lq_image_paths))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gt_path, lq_path = self.data[index]

        gt_image = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        lq_image = cv2.imread(lq_path, cv2.IMREAD_COLOR)

        gt_image = gt_image.astype('float32') / 255.
        lq_image = lq_image.astype('float32') / 255.

        gt_image_tensor = torch.from_numpy(
            np.transpose(
                gt_image[:, :, [2, 1, 0]], (2, 0, 1)
            )
        ).float()
        lq_image_tensor = torch.from_numpy(
            np.transpose(
                lq_image[:, :, [2, 1, 0]], (2, 0, 1)
            )
        ).float()

        return gt_image_tensor, lq_image_tensor


def test():
    dataset = PairedImageDataset(
        gt_folder=config.GT_FOLDER_TRAINING,
        lq_folder=config.LQ_FOLDER_TRAINING
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == '__main__':
    test()
