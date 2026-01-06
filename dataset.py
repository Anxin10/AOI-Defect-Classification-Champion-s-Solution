import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config
from tqdm import tqdm

class AOIDataset(Dataset):
    def __init__(self, df, transforms=None, mode='train'):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.cache_images = Config.CACHE_IMAGES_TO_RAM
        self.images = []

        # Ensure 'filepath' column exists or construct it
        # Based on previous context, images are in TRAIN_IMAGES_DIR or TEST_IMAGES_DIR.
        # Let's assume ID is filename for now or construct full path.
        # The path construction will be handled within the caching loop and __getitem__.

        if self.cache_images:
            print(f"Loading {len(df)} images into RAM for {mode} (Threads: {Config.NUM_WORKERS})...")
            
            from joblib import Parallel, delayed

            def load_single_image(row):
                img_dir_for_mode = Config.TRAIN_IMAGES_DIR if self.mode in ['train', 'valid', 'student'] else Config.TEST_IMAGES_DIR
                img_path = os.path.join(img_dir_for_mode, row['ID'])
                if not os.path.exists(img_path) and not img_path.endswith('.png'):
                     img_path += '.png'
                
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img

            # Parallel loading
            self.images = Parallel(n_jobs=Config.NUM_WORKERS, backend='threading')(
                delayed(load_single_image)(row) for _, row in tqdm(df.iterrows(), total=len(df))
            )
            print("RAM Cache Loaded!")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.cache_images:
            image = self.images[idx]
        else:
            row = self.df.iloc[idx]
            img_name = row['ID']
            
            # Determine image path
            img_dir_for_mode = Config.TRAIN_IMAGES_DIR if self.mode in ['train', 'valid', 'student'] else Config.TEST_IMAGES_DIR
            img_path = os.path.join(img_dir_for_mode, img_name)
            # If ID doesn't have extension, try adding .png
            if not os.path.exists(img_path) and not img_path.endswith('.png'):
                img_path += '.png'
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        # No else block needed here, as default transform logic is handled by `get_transforms`
            
        if self.mode in ['train', 'valid', 'student']:
            label = self.df.iloc[idx]['Label']
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image

def get_transforms(img_size, mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Replaced ShiftScaleRotate with Affine as per warning
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=0.5),
            # Simulate "Dual Stream" (Sharpen/Blur)
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            # Updated CoarseDropout API
            A.CoarseDropout(num_holes_limit=8, hole_height_range=(1, img_size//10), hole_width_range=(1, img_size//10), p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif mode == 'student': # Stronger augmentations for Student
         return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Replaced ShiftScaleRotate with Affine
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-45, 45), p=0.7),
            A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3, val_shift_limit=0.3, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.OneOf([
                A.GaussianBlur(p=0.5),
                A.GaussNoise(p=0.5),
            ], p=0.5),
            # Updated CoarseDropout API
            A.CoarseDropout(num_holes_limit=16, hole_height_range=(1, img_size//8), hole_width_range=(1, img_size//8), p=0.7),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else: # Valid / Test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
