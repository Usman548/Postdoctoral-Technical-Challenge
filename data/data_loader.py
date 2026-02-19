"""
Data loading and preprocessing utilities for PneumoniaMNIST dataset.
"""

import medmnist
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PneumoniaMNISTDataset:
    """
    Wrapper class for PneumoniaMNIST dataset with preprocessing and augmentation.
    Implements DatasetProvider protocol (get_dataloaders, get_class_names, class_weights)
    for use with train/evaluate without coupling to this concrete class.
    """
    
    def __init__(self, 
                 batch_size: int = 32,
                 img_size: int = 28,
                 normalize: bool = True,
                 augment: bool = True,
                 validation_split: float = 0.15,
                 test_split: float = 0.15,
                 random_seed: int = 42,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        """
        Initialize the dataset wrapper.
        
        Args:
            batch_size: Batch size for dataloaders
            img_size: Image size (assumed square)
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation
            validation_split: Validation split ratio (not used - MedMNIST has fixed splits)
            test_split: Test split ratio (not used - MedMNIST has fixed splits)
            random_seed: Random seed for reproducibility
            num_workers: Number of workers for dataloader
            pin_memory: Whether to pin memory
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.normalize = normalize
        self.augment = augment
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.random_seed = random_seed
        
        # These are kept for compatibility but not used (MedMNIST has fixed splits)
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Define transformations
        self._setup_transforms()
        
        # Load dataset
        self._load_data()
        
    def _setup_transforms(self):
        """Setup image transformations and augmentations."""
        
        # Basic preprocessing
        if self.normalize:
            base_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        else:
            base_transforms = [
                transforms.ToTensor()
            ]
        
        # Training transforms with augmentation
        if self.augment:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.train_transform = transforms.Compose(base_transforms)
        
        # Validation/test transforms (no augmentation)
        self.val_transform = transforms.Compose(base_transforms)
        
    def _load_data(self):
        """Load PneumoniaMNIST dataset splits."""
        try:
            # Download and load data using medmnist
            self.train_dataset_raw = medmnist.PneumoniaMNIST(split='train', download=True, size=28)
            self.val_dataset_raw = medmnist.PneumoniaMNIST(split='val', download=True, size=28)
            self.test_dataset_raw = medmnist.PneumoniaMNIST(split='test', download=True, size=28)
            
            logger.info(f"Training set size: {len(self.train_dataset_raw)}")
            logger.info(f"Validation set size: {len(self.val_dataset_raw)}")
            logger.info(f"Test set size: {len(self.test_dataset_raw)}")
            
            # Check class balance
            train_labels = [label for _, label in self.train_dataset_raw]
            unique, counts = np.unique(train_labels, return_counts=True)
            # Convert to float32 and ensure 1D
            self.class_weights = torch.tensor([1.0 / count for count in counts], dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create and return dataloaders for train, validation, and test sets.
        """
        
        # Create datasets with appropriate transforms
        train_dataset = PneumoniaMNISTSplit(
            self.train_dataset_raw, 
            transform=self.train_transform
        )
        val_dataset = PneumoniaMNISTSplit(
            self.val_dataset_raw, 
            transform=self.val_transform
        )
        test_dataset = PneumoniaMNISTSplit(
            self.test_dataset_raw, 
            transform=self.val_transform
        )
        
        # Create dataloaders with the provided parameters
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_names(self) -> List[str]:
        """Return class names."""
        return ['Normal', 'Pneumonia']
    
    def visualize_sample_batch(self, dataloader: DataLoader, num_samples: int = 8):
        """
        Visualize a batch of samples.
        
        Args:
            dataloader: DataLoader to visualize
            num_samples: Number of samples to show
        """
        images, labels = next(iter(dataloader))
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        class_names = self.get_class_names()
        
        for i in range(min(num_samples, len(images))):
            img = images[i].squeeze().numpy()
            if self.normalize:
                img = (img * 0.5) + 0.5  # Denormalize
            label = labels[i].item()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{class_names[label]}')
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(min(num_samples, len(images)), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


class PneumoniaMNISTSplit(Dataset):
    """
    Custom Dataset wrapper for PneumoniaMNIST with transformations.
    """
    
    def __init__(self, base_dataset, transform=None):
        """
        Initialize dataset split.
        
        Args:
            base_dataset: Original medmnist dataset split
            transform: Transformations to apply
        """
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Convert numpy array to PIL Image
        if isinstance(img, np.ndarray):
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            
            img = Image.fromarray(img, mode='L')  # 'L' for grayscale
        
        if self.transform:
            img = self.transform(img)
        
        # Convert label to scalar tensor
        if isinstance(label, (list, np.ndarray)):
            label = label[0] if len(label) > 0 else label
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label