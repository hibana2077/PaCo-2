"""Main training script for PaCo-2"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dataset.ufgvc import UFGVCDataset
from src.models import PaCoModel
from src.train_utils import (
    train_epoch, validate_epoch, MetricsTracker, 
    create_optimizer, create_scheduler, save_checkpoint, 
    load_checkpoint, save_config, save_metrics_csv,
    create_run_directory, AverageMeter, compute_metrics
)
from src.data_utils import create_ufgvc_transforms, TwoViewDataset


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_loaders(config: dict) -> tuple:
    """Create train and validation data loaders"""
    
    dataset_config = config['dataset']
    data_config = config['data']
    
    # Create transforms
    train_transform = create_ufgvc_transforms(
        image_size=data_config['image_size'],
        split='train',
        use_two_views=True  # Always use two views for training
    )
    
    val_transform = create_ufgvc_transforms(
        image_size=data_config['image_size'],
        split='val',
        use_two_views=False
    )
    
    # Create datasets
    train_dataset = UFGVCDataset(
        dataset_name=dataset_config['name'],
        root=dataset_config['root'],
        split='train',
        download=dataset_config.get('download', True)
    )
    
    val_dataset = UFGVCDataset(
        dataset_name=dataset_config['name'],
        root=dataset_config['root'],
        split='val',
        download=dataset_config.get('download', True)
    )
    
    # Wrap with two-view transform for training
    train_dataset = TwoViewDataset(train_dataset, train_transform)
    val_dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        drop_last=True  # Important for consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.dataset.classes)


def create_model(config: dict, num_classes: int) -> PaCoModel:
    """Create PaCo-2 model"""
    
    model_config = config['model']
    
    model = PaCoModel(
        backbone_name=model_config['backbone'],
        num_classes=num_classes,
        pretrained=model_config.get('pretrained', True),
        # Part sampling parameters
        K=model_config['K'],
        r=model_config['r'],
        d=model_config['d'],
        # Loss parameters
        lambda_pac=model_config['lambda_pac'],
        eta_soc=model_config['eta_soc'],
        alpha=model_config['alpha'],
        beta=model_config['beta'],
        gamma=model_config['gamma'],
        # Technical parameters
        epsilon=model_config.get('epsilon', 1e-5),
        tau=model_config.get('tau', 1e-5),
        metric=model_config.get('metric', 'fro'),
        use_mahalanobis_warmup=model_config.get('use_mahalanobis_warmup', True),
        warmup_epochs=model_config.get('warmup_epochs', 3),
        # Optional features
        use_weighted_ce=model_config.get('use_weighted_ce', True),
        use_semi_hard=model_config.get('use_semi_hard', True),
        use_class_proto=model_config.get('use_class_proto', True),
        proto_momentum=model_config.get('proto_momentum', 0.9)
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train PaCo-2 Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda:0, etc.)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(config['training']['seed'])
    
    # Create run directory
    run_dir = create_run_directory(
        Path(config['training']['save_dir']),
        config['dataset']['name'],
        config['model']['backbone']
    )
    
    print(f"Run directory: {run_dir}")
    
    # Save configuration
    save_config(config, run_dir / 'config.yaml')
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, num_classes = create_data_loaders(config)
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config, num_classes)
    model = model.to(device)
    
    print("Model info:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config['optimizer'])
    scheduler = create_scheduler(optimizer, config['scheduler'])
    
    # Initialize tracking
    metrics_tracker = MetricsTracker()
    start_epoch = 0
    best_val_acc = 0.0
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(model, optimizer, Path(args.resume))
        start_epoch = checkpoint_info['epoch'] + 1
        best_val_acc = checkpoint_info['metrics'].get('val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.3f}")
    
    # Training loop
    print("Starting training...")
    num_epochs = config['training']['epochs']
    print_freq = config['training'].get('print_freq', 100)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs-1}")
        print("-" * 50)
        
def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                optimizer,
                device: torch.device,
                epoch: int,
                print_freq: int = 100) -> dict:
    """Train for one epoch - modified for two-view data"""
    
    model.train()
    model.set_epoch(epoch)
    
    # Initialize meters
    meters = {
        'loss': AverageMeter(),
        'ce_loss': AverageMeter(),
        'pac_loss': AverageMeter(),
        'soc_loss': AverageMeter(),
        'acc': AverageMeter(),
        'balanced_acc': AverageMeter()
    }
    
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle two-view data format
        if isinstance(batch_data[0], tuple):
            # Two-view format: ((view1, view2), targets)
            (x1, x2), targets = batch_data
        else:
            # Single view format: (images, targets) - create duplicate
            images, targets = batch_data
            x1 = x2 = images
        
        # Move to device
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = x1.size(0)
        
        optimizer.zero_grad()
        
        # Forward pass with two views
        outputs = model(x1, x2, targets)
        
        # Extract losses
        total_loss = outputs['total']
        ce_loss = outputs.get('ce', torch.tensor(0.0))
        pac_loss = outputs.get('pac', torch.tensor(0.0))
        soc_loss = outputs.get('soc', torch.tensor(0.0))
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            from src.train_utils import compute_metrics
            metrics = compute_metrics(outputs['logits'], targets, 
                                    model.classifier.out_features)
        
        # Update meters
        meters['loss'].update(total_loss.item(), batch_size)
        meters['ce_loss'].update(ce_loss.item(), batch_size)
        meters['pac_loss'].update(pac_loss.item(), batch_size)
        meters['soc_loss'].update(soc_loss.item(), batch_size)
        meters['acc'].update(metrics['acc'], batch_size)
        meters['balanced_acc'].update(metrics['balanced_acc'], batch_size)
        
        # Print progress
        if batch_idx % print_freq == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                  f'Time: {elapsed:.1f}s '
                  f'Loss: {meters["loss"].avg:.4f} '
                  f'CE: {meters["ce_loss"].avg:.4f} '
                  f'PaC: {meters["pac_loss"].avg:.4f} '
                  f'SoC: {meters["soc_loss"].avg:.4f} '
                  f'Acc: {meters["acc"].avg:.3f} '
                  f'Bal-Acc: {meters["balanced_acc"].avg:.3f}')
    
    return {
        'loss': meters['loss'].avg,
        'ce': meters['ce_loss'].avg,
        'pac': meters['pac_loss'].avg,
        'soc': meters['soc_loss'].avg,
        'acc': meters['acc'].avg,
        'balanced_acc': meters['balanced_acc'].avg
    }
        
        # Validation phase
        val_metrics = validate_epoch(model, val_loader, device, epoch)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Track metrics
        metrics_tracker.update('train', train_metrics)
        metrics_tracker.update('val', {
            'loss': val_metrics['loss'],
            'acc': val_metrics['acc'],
            'balanced_acc': val_metrics['balanced_acc']
        })
        
        # Print epoch summary
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['acc']:.3f}, "
              f"Bal-Acc: {train_metrics['balanced_acc']:.3f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['acc']:.3f}, "
              f"Bal-Acc: {val_metrics['balanced_acc']:.3f}")
        
        # Save checkpoint
        is_best = val_metrics['acc'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['acc']
            print(f"New best validation accuracy: {best_val_acc:.3f}")
        
        checkpoint_metrics = {
            'train_acc': train_metrics['acc'],
            'val_acc': val_metrics['acc'],
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        }
        
        save_checkpoint(
            model, optimizer, epoch, checkpoint_metrics,
            run_dir / 'checkpoints' / f'epoch_{epoch}.pth',
            is_best=is_best
        )
        
        # Save best checkpoint separately
        if is_best:
            save_checkpoint(
                model, optimizer, epoch, checkpoint_metrics,
                run_dir / 'best_model.pth',
                is_best=True
            )
    
    # Save final results
    print("\nSaving final results...")
    
    # Save training curves
    metrics_tracker.save_curves(run_dir / 'curves')
    
    # Save metrics CSV
    save_metrics_csv(metrics_tracker, run_dir / 'metrics.csv')
    
    # Save final model info
    final_info = {
        'config': config,
        'model_info': model.get_model_info(),
        'best_val_acc': best_val_acc,
        'total_epochs': num_epochs,
        'run_dir': str(run_dir)
    }
    
    import json
    with open(run_dir / 'final_info.json', 'w') as f:
        json.dump(final_info, f, indent=2)
    
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Results saved to: {run_dir}")


if __name__ == '__main__':
    main()
