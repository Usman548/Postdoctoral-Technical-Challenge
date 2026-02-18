#!/usr/bin/env python3
"""
Training script for pneumonia classification from chest X-ray images.
Optimized for CPU execution with comprehensive monitoring and error handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
import json
import yaml
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from data.data_loader import PneumoniaMNISTDataset
from models.cnn_model import create_model
from utils.logger import setup_logger
from utils.visualization import plot_training_history

# Configure logging
logger = setup_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration dataclass for type safety and serialization."""
    
    # Model parameters
    model_name: str = 'custom'
    num_classes: int = 2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    patience: int = 10
    weight_decay: float = 1e-4
    
    # Data parameters
    augment: bool = True
    random_seed: int = 42
    
    # System parameters
    device: str = 'cpu'
    num_workers: int = 0
    pin_memory: bool = False
    verbose: bool = True
    
    # Paths
    output_dir: str = 'reports/task1'
    models_dir: str = 'models/saved'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class CPUTrainer:
    """
    Trainer class for pneumonia classification optimized for CPU execution.
    """
    
    def __init__(self, config: Union[Dict[str, Any], TrainingConfig, str]):
        """
        Initialize the CPU trainer.
        
        Args:
            config: Training configuration (dict, TrainingConfig, or path to YAML)
        """
        # Load configuration
        if isinstance(config, str) and config.endswith(('.yaml', '.yml')):
            self.config = TrainingConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = TrainingConfig(**config)
        elif isinstance(config, TrainingConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # Ensure CPU usage
        self.config.device = 'cpu'
        self.device = torch.device('cpu')
        
        logger.info(f"Initializing CPU Trainer with config: {self.config}")
        
        # Setup directories
        self._setup_directories()
        
        # Load dataset
        self._setup_data()
        
        # Create model
        self._setup_model()
        
        # Setup loss function, optimizer, and scheduler
        self._setup_training_components()
        
        # Initialize training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        logger.info("Trainer initialization complete")
    
    def _setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        self.output_dir = Path(self.config.output_dir)
        self.figures_dir = self.output_dir / 'figures'
        self.models_dir = Path(self.config.models_dir)
        
        for directory in [self.output_dir, self.figures_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _setup_data(self) -> None:
        """Load and prepare dataset with CPU-optimized settings."""
        logger.info("Loading PneumoniaMNIST dataset...")
        
        try:
            self.dataset = PneumoniaMNISTDataset(
                batch_size=self.config.batch_size,
                augment=self.config.augment,
                random_seed=self.config.random_seed,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            self.train_loader, self.val_loader, self.test_loader = self.dataset.get_dataloaders()
            self.class_names = self.dataset.get_class_names()
            self.class_weights = self.dataset.class_weights
            
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  - Training samples: {len(self.train_loader.dataset)}")
            logger.info(f"  - Validation samples: {len(self.val_loader.dataset)}")
            logger.info(f"  - Test samples: {len(self.test_loader.dataset)}")
            logger.info(f"  - Class names: {self.class_names}")
            logger.info(f"  - Class weights: {self.class_weights}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def _setup_model(self) -> None:
        """Create and initialize the model."""
        logger.info(f"Creating {self.config.model_name} model...")
        
        try:
            self.model = create_model(
                model_name=self.config.model_name,
                num_classes=self.config.num_classes,
                pretrained=False
            )
            self.model = self.model.to(self.device)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model created successfully:")
            logger.info(f"  - Total parameters: {total_params:,}")
            logger.info(f"  - Trainable parameters: {trainable_params:,}")
            logger.info(f"  - Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def _setup_training_components(self) -> None:
        """Setup loss function, optimizer, and learning rate scheduler."""
        
        # Loss function with class weights for imbalanced data
        if hasattr(self, 'class_weights'):
            weights = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            logger.info(f"Using weighted CrossEntropyLoss with weights: {weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropyLoss")
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        logger.info(f"Optimizer: Adam (lr={self.config.learning_rate}, weight_decay={self.config.weight_decay})")
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        logger.info("Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy percentage)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {self.current_epoch + 1}/{self.config.num_epochs} [Train]',
                   disable=not self.config.verbose)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Ensure labels are 1D
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader: Optional[torch.utils.data.DataLoader] = None) -> Tuple[float, float]:
        """
        Validate the model on given data loader.
        
        Args:
            loader: DataLoader to use (default: validation loader)
            
        Returns:
            Tuple of (average loss, accuracy percentage)
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, 
                       desc=f'Epoch {self.current_epoch + 1}/{self.config.num_epochs} [Val]',
                       disable=not self.config.verbose)
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if labels.dim() > 1:
                    labels = labels.squeeze()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop with early stopping and model checkpointing.
        
        Returns:
            Training history dictionary
        """
        logger.info("=" * 60)
        logger.info(f"Starting training on {self.device}")
        logger.info("=" * 60)
        
        # Save configuration
        self.config.save(self.output_dir / 'config.json')
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            try:
                # Training phase
                train_loss, train_acc = self.train_epoch()
                
                # Validation phase
                val_loss, val_acc = self.validate()
                
                # Update learning rate
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Save to history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rates'].append(current_lr)
                self.history['epoch_times'].append(epoch_time)
                
                # Log epoch summary
                logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs} Summary:")
                logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                logger.info(f"  Learning Rate: {current_lr:.6f}")
                logger.info(f"  Time: {epoch_time:.2f}s")
                
                # Save best model based on validation accuracy
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save model checkpoint
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"  ✓ New best model! Val Acc: {val_acc:.2f}%")
                    
                else:
                    self.patience_counter += 1
                    logger.info(f"  Patience: {self.patience_counter}/{self.config.patience}")
                
                # Early stopping check
                if self.patience_counter >= self.config.patience:
                    logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
                
                # Save periodic checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(epoch, is_best=False)
                
            except KeyboardInterrupt:
                logger.info("\nTraining interrupted by user. Saving checkpoint...")
                self._save_checkpoint(epoch, is_best=False)
                break
                
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                logger.exception("Detailed error trace:")
                raise
        
        # Training completed
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info("=" * 60)
        
        # Generate and save visualizations
        self._generate_visualizations()
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else 0,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else float('inf'),
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        if is_best:
            checkpoint_path = self.models_dir / f'best_model_{self.config.model_name}.pth'
        else:
            checkpoint_path = self.models_dir / f'checkpoint_epoch_{epoch + 1}_{self.config.model_name}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _generate_visualizations(self) -> None:
        """Generate and save training visualizations."""
        logger.info("Generating training visualizations...")
        
        try:
            # Plot training curves
            fig = plot_training_history(self.history)
            fig.savefig(self.figures_dir / 'training_curves.png', 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Plot additional metrics if available
            if len(self.history['epoch_times']) > 0:
                plt.figure(figsize=(10, 4))
                plt.plot(self.history['epoch_times'])
                plt.xlabel('Epoch')
                plt.ylabel('Time (seconds)')
                plt.title('Training Time per Epoch')
                plt.grid(True)
                plt.savefig(self.figures_dir / 'epoch_times.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Visualizations saved to {self.figures_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
    
    def _save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.output_dir / 'training_history.json'
        
        # Convert numpy values to Python types for JSON serialization
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded (epoch {checkpoint['epoch'] + 1}, "
                   f"val_acc={checkpoint['val_acc']:.2f}%)")


def main():
    """Main entry point for training script."""
    
    parser = argparse.ArgumentParser(
        description='Train pneumonia classifier (CPU-optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='custom',
                       choices=['custom', 'resnet18', 'efficientnet-b0', 'vit-tiny'],
                       help='Model architecture')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (reduce if running out of memory)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001,
                       dest='learning_rate', help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    
    # Data arguments
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                       help='Disable data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 = main process)')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                       help='Suppress verbose output')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to YAML configuration file (overrides other args)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='reports/task1',
                       help='Directory for outputs')
    parser.add_argument('--models_dir', type=str, default='models/saved',
                       help='Directory for saved models')
    
    args = parser.parse_args()
    
    # If config file is provided, load from it
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = TrainingConfig.from_yaml(args.config)
    else:
        # Create config from command line arguments
        config_dict = {
            'model_name': args.model,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.epochs,
            'patience': args.patience,
            'weight_decay': args.weight_decay,
            'augment': args.augment,
            'random_seed': args.seed,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
            'output_dir': args.output_dir,
            'models_dir': args.models_dir
        }
        config = TrainingConfig(**config_dict)
    
    # Log training start
    logger.info("=" * 60)
    logger.info("PNEUMONIA CLASSIFICATION TRAINING")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CPU cores available: {torch.get_num_threads()}")
    logger.info(f"Using device: CPU (as specified)")
    
    # Set number of threads for CPU optimization
    if args.num_workers == 0:
        torch.set_num_threads(min(4, torch.get_num_threads()))
        logger.info(f"Set number of CPU threads to {torch.get_num_threads()}")
    
    # Create trainer and train
    try:
        trainer = CPUTrainer(config)
        history = trainer.train()
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.exception("Detailed error trace:")
        sys.exit(1)


if __name__ == '__main__':
    main()
