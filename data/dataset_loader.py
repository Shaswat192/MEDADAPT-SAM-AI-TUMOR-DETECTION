"""
Dataset Loader for BraTS 2021 PNG Dataset
Handles loading of FLAIR, RGB images and segmentation masks
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BraTSDataset(Dataset):
    """BraTS 2021 Dataset Loader for PNG images"""
    
    def __init__(
        self,
        data_path: str,
        manifest_csv: Optional[str] = None,
        image_size: int = 256,
        mode: str = 'train',
        transform: Optional[A.Compose] = None,
        use_cache: bool = False
    ):
        """
        Args:
            data_path: Path to brats_png directory
            manifest_csv: Path to CSV manifest file
            image_size: Target image size
            mode: 'train', 'val', or 'test'
            transform: Albumentations transform
            use_cache: Cache images in memory
        """
        self.data_path = data_path
        self.image_size = image_size
        self.mode = mode
        self.transform = transform
        self.use_cache = use_cache
        self.cache = {}
        
        # Load data paths
        self.data_list = self._load_data_list(manifest_csv)
        
    def _load_data_list(self, manifest_csv: Optional[str]) -> List[Dict]:
        """Load list of image paths"""
        data_list = []
        
        if manifest_csv and os.path.exists(manifest_csv):
            # Load from CSV manifest
            df = pd.read_csv(manifest_csv)
            data_list = df.to_dict('records')
        else:
            # Scan directory structure
            for patient_folder in sorted(os.listdir(self.data_path)):
                patient_path = os.path.join(self.data_path, patient_folder)
                
                if not os.path.isdir(patient_path):
                    continue
                
                # Get all slices for this patient
                files = os.listdir(patient_path)
                
                # Group by slice number
                slices = {}
                for f in files:
                    if '_FLAIR.png' in f:
                        slice_num = f.split('_slice_')[1].split('_')[0]
                        slices[slice_num] = {
                            'patient_id': patient_folder,
                            'slice_num': slice_num
                        }
                
                # Add full paths
                for slice_num, info in slices.items():
                    base_name = f"{patient_folder}_slice_{slice_num}"
                    
                    data_list.append({
                        'patient_id': patient_folder,
                        'slice_num': slice_num,
                        'flair_path': os.path.join(patient_path, f"{base_name}_FLAIR.png"),
                        'rgb_path': os.path.join(patient_path, f"{base_name}_RGB.png"),
                        'et_mask_path': os.path.join(patient_path, f"{base_name}_ET.png"),
                        'tc_mask_path': os.path.join(patient_path, f"{base_name}_TC.png"),
                        'wt_mask_path': os.path.join(patient_path, f"{base_name}_WT.png"),
                    })
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        
        # Check cache
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        item = self.data_list[idx]
        
        # Load images
        flair = np.array(Image.open(item['flair_path']).convert('L'))
        rgb = np.array(Image.open(item['rgb_path']).convert('RGB'))
        
        # Load masks
        et_mask = np.array(Image.open(item['et_mask_path']).convert('L'))
        tc_mask = np.array(Image.open(item['tc_mask_path']).convert('L'))
        wt_mask = np.array(Image.open(item['wt_mask_path']).convert('L'))
        
        # Combine masks into multi-channel
        mask = np.stack([et_mask, tc_mask, wt_mask], axis=-1)
        
        # Normalize masks to 0-1
        mask = (mask > 0).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=rgb, mask=mask)
            rgb = transformed['image']
            mask = transformed['mask']
        else:
            # Default: resize and convert to tensor
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        sample = {
            'image': rgb,
            'mask': mask,
            'patient_id': item['patient_id'],
            'slice_num': item['slice_num'],
            'flair': torch.from_numpy(flair).unsqueeze(0).float() / 255.0
        }
        
        # Cache if enabled
        if self.use_cache:
            self.cache[idx] = sample
        
        return sample


def get_transforms(image_size: int = 256, mode: str = 'train') -> A.Compose:
    """Get augmentation transforms"""
    
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def create_dataloaders(
    data_path: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create full dataset
    full_dataset = BraTSDataset(
        data_path=data_path,
        image_size=image_size,
        mode='train'
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_transforms(image_size, 'train')
    val_dataset.dataset.transform = get_transforms(image_size, 'val')
    test_dataset.dataset.transform = get_transforms(image_size, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loader
    data_path = "D:/major projrct PNG folder/brats_png"
    
    dataset = BraTSDataset(data_path=data_path, image_size=256)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Patient ID: {sample['patient_id']}")
    print(f"Slice number: {sample['slice_num']}")
