import os
import pandas as pd
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

from core.data._config import PEDIATRIC_PNEUMONIA_TRANSFORMS, DATA_DIR


class Pediatric_Pneumonia_Dataset(Dataset):
    """
    Dataset for Pediatric Chest X-ray Pneumonia from Kaggle.
    """

    classes = ['NORMAL', 'PNEUMONIA']

    def __init__(self, dir_path, split='train', transform=None):
        """
        Args:
        dir_path: path to the base folder of the dataset (which contains train/ and test/)
        split: 'train', or 'test'
        transform: torchvision transforms to apply to images
        """
        self.transform = transform
        self.images = []

        split_dir = os.path.join(dir_path, split)

        for i in range(len(self.classes)):
            cls_folder = os.path.join(split_dir, self.classes[i])
            self.images += [(os.path.join(cls_folder, fname), i) for fname in os.listdir(cls_folder)]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
        return img, label
    

def load_pediatric_pneumonia(
    batch_size: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loading method for Peadiatric Chest X-ray Pneumonia dataset.
    Expects the images to be saved inside the directory designed for data (`DATA_DIR`) under the name "Peadiatric Chest X-ray Pneumonia"

    Args:
        batch_size: batch size. Defaults to 128.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """
    
    full_train_dataset = Pediatric_Pneumonia_Dataset(
        os.path.join(DATA_DIR, "Pediatric Chest X-ray Pneumonia"),
        split="train",
        transform=PEDIATRIC_PNEUMONIA_TRANSFORMS
    )
    test_dataset = Pediatric_Pneumonia_Dataset(
        os.path.join(DATA_DIR, "Pediatric Chest X-ray Pneumonia"),
        split="test",
        transform=PEDIATRIC_PNEUMONIA_TRANSFORMS
    )

    train_size = int(len(full_train_dataset) * 0.8)
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return [train_loader, val_loader, test_loader]