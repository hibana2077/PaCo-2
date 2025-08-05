"""Training utilities for PaCo-2"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track training and validation metrics"""
    
    def __init__(self):
        self.history = {
            'train': {'loss': [], 'ce': [], 'pac': [], 'soc': [], 'acc': [], 'balanced_acc': []},
            'val': {'loss': [], 'acc': [], 'balanced_acc': []}
        }
    
    def update(self, phase: str, metrics: Dict[str, float]):
        """Update metrics for given phase"""
        for key, value in metrics.items():
            if key in self.history[phase]:
                self.history[phase][key].append(value)
    
    def get_best_metric(self, phase: str, metric: str, mode: str = 'max') -> Tuple[float, int]:
        """Get best metric value and epoch"""
        values = self.history[phase][metric]
        if not values:
            return float('-inf') if mode == 'max' else float('inf'), -1
        
        if mode == 'max':
            best_val = max(values)
            best_epoch = values.index(best_val)
        else:
            best_val = min(values)
            best_epoch = values.index(best_val)
        
        return best_val, best_epoch
    
    def save_curves(self, save_dir: Path):
        """Save training curves"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss curves
        plt.figure(figsize=(15, 5))
        
        # Total loss
        plt.subplot(1, 3, 1)
        if self.history['train']['loss']:
            plt.plot(self.history['train']['loss'], label='Train')
        if self.history['val']['loss']:
            plt.plot(self.history['val']['loss'], label='Val')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Component losses
        plt.subplot(1, 3, 2)
        if self.history['train']['ce']:
            plt.plot(self.history['train']['ce'], label='CE', alpha=0.7)
        if self.history['train']['pac']:
            plt.plot(self.history['train']['pac'], label='PaC', alpha=0.7)
        if self.history['train']['soc']:
            plt.plot(self.history['train']['soc'], label='SoC', alpha=0.7)
        plt.title('Component Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(1, 3, 3)
        if self.history['train']['acc']:
            plt.plot(self.history['train']['acc'], label='Train Acc')
        if self.history['val']['acc']:
            plt.plot(self.history['val']['acc'], label='Val Acc')
        if self.history['train']['balanced_acc']:
            plt.plot(self.history['train']['balanced_acc'], label='Train Balanced Acc', alpha=0.7)
        if self.history['val']['balanced_acc']:
            plt.plot(self.history['val']['balanced_acc'], label='Val Balanced Acc', alpha=0.7)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                   num_classes: int) -> Dict[str, float]:
    """Compute classification metrics"""
    pred = outputs.argmax(dim=1)
    pred_np = pred.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Basic metrics
    acc = accuracy_score(targets_np, pred_np)
    balanced_acc = balanced_accuracy_score(targets_np, pred_np)
    
    # Top-5 accuracy (if num_classes >= 5)
    top5_acc = 0.0
    if num_classes >= 5:
        _, top5_pred = torch.topk(outputs, 5, dim=1)
        top5_correct = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))
        top5_acc = top5_correct.float().sum().item() / targets.size(0)
    
    return {
        'acc': acc,
        'balanced_acc': balanced_acc,
        'top5_acc': top5_acc
    }


def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                print_freq: int = 100) -> Dict[str, float]:
    """Train for one epoch with two-view support"""
    
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
    
    # Create progress bar for batches
    train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                     desc=f"Epoch {epoch}", leave=False, unit="batch")
    
    for batch_idx, batch_data in train_pbar:
        # Handle two-view data format
        if isinstance(batch_data[0], tuple):
            # Two-view format: ((view1, view2), targets)
            (x1, x2), targets = batch_data
        elif isinstance(batch_data[0], list):
            # Handle list format from DataLoader
            if len(batch_data[0]) == 2 and hasattr(batch_data[0][0], 'shape'):
                # Two views in list format
                x1, x2 = batch_data[0]
                targets = batch_data[1]
            else:
                # Single view in list format
                images = batch_data[0]
                targets = batch_data[1]
                x1 = x2 = images
        else:
            # Single view format: (images, targets) - create duplicate for testing
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
            metrics = compute_metrics(outputs['logits'], targets, 
                                    model.classifier.out_features)
        
        # Update meters
        meters['loss'].update(total_loss.item(), batch_size)
        meters['ce_loss'].update(ce_loss.item(), batch_size)
        meters['pac_loss'].update(pac_loss.item(), batch_size)
        meters['soc_loss'].update(soc_loss.item(), batch_size)
        meters['acc'].update(metrics['acc'], batch_size)
        meters['balanced_acc'].update(metrics['balanced_acc'], batch_size)
        
        # Update progress bar with current metrics
        train_pbar.set_postfix({
            'Loss': f"{meters['loss'].avg:.4f}",
            'CE': f"{meters['ce_loss'].avg:.4f}",
            'PaC': f"{meters['pac_loss'].avg:.4f}",
            'SoC': f"{meters['soc_loss'].avg:.4f}",
            'Acc': f"{meters['acc'].avg:.3f}"
        })
        
        # Optional: Still print detailed progress at intervals
        if batch_idx % print_freq == 0:
            elapsed = time.time() - start_time
            tqdm.write(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
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


def validate_epoch(model: nn.Module,
                  val_loader: DataLoader,
                  device: torch.device,
                  epoch: int) -> Dict[str, float]:
    """Validate for one epoch"""
    
    model.eval()
    
    meters = {
        'loss': AverageMeter(),
        'acc': AverageMeter(),
        'balanced_acc': AverageMeter()
    }
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation", leave=False, unit="batch")
        
        for images, targets in val_pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            batch_size = images.size(0)
            
            # Single view for validation
            outputs = model(images)
            logits = outputs['logits']
            
            # Simple CE loss for validation
            loss = nn.CrossEntropyLoss()(logits, targets)
            
            # Compute metrics
            metrics = compute_metrics(logits, targets, model.classifier.out_features)
            
            # Update meters
            meters['loss'].update(loss.item(), batch_size)
            meters['acc'].update(metrics['acc'], batch_size)
            meters['balanced_acc'].update(metrics['balanced_acc'], batch_size)
            
            # Update progress bar
            val_pbar.set_postfix({
                'Val_Loss': f"{meters['loss'].avg:.4f}",
                'Val_Acc': f"{meters['acc'].avg:.3f}"
            })
            
            # Store for confusion matrix
            all_outputs.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    return {
        'loss': meters['loss'].avg,
        'acc': meters['acc'].avg,
        'balanced_acc': meters['balanced_acc'].avg,
        'outputs': torch.cat(all_outputs, dim=0),
        'targets': torch.cat(all_targets, dim=0)
    }


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   save_path: Path,
                   is_best: bool = False):
    """Save model checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': model.get_model_info()
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)


def load_checkpoint(model: nn.Module,
                   optimizer: Optional[optim.Optimizer],
                   checkpoint_path: Path) -> Dict[str, Any]:
    """Load model checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'model_config': checkpoint.get('model_config', {})
    }


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer based on config"""
    
    optimizer_name = config.get('name', 'adamw').lower()
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', True)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                             momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Optional[Any]:
    """Create learning rate scheduler"""
    
    if not config.get('use_scheduler', True):
        return None
    
    scheduler_name = config.get('name', 'cosine').lower()
    
    if scheduler_name == 'cosine':
        T_max = config.get('T_max', 100)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'multistep':
        milestones = config.get('milestones', [60, 80])
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma)
    else:
        print(f"Unknown scheduler: {scheduler_name}, using None")
        return None
    
    return scheduler


def save_config(config: Dict[str, Any], save_path: Path):
    """Save configuration to YAML file"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics_csv(metrics_tracker: MetricsTracker, save_path: Path):
    """Save metrics to CSV file"""
    import pandas as pd
    
    # Prepare data
    data = []
    max_epochs = max(
        len(metrics_tracker.history['train']['loss']),
        len(metrics_tracker.history['val']['loss'])
    )
    
    for epoch in range(max_epochs):
        row = {'epoch': epoch}
        
        # Training metrics
        for key, values in metrics_tracker.history['train'].items():
            if epoch < len(values):
                row[f'train_{key}'] = values[epoch]
            else:
                row[f'train_{key}'] = None
        
        # Validation metrics
        for key, values in metrics_tracker.history['val'].items():
            if epoch < len(values):
                row[f'val_{key}'] = values[epoch]
            else:
                row[f'val_{key}'] = None
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def create_run_directory(base_dir: Path, dataset_name: str, model_name: str) -> Path:
    """Create run directory with timestamp"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / dataset_name / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir
