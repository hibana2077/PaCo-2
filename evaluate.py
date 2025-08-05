"""Evaluation script for PaCo-2"""

import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dataset.ufgvc import UFGVCDataset
from src.models import PaCoModel
from src.train_utils import load_checkpoint, compute_metrics
from src.data_utils import create_ufgvc_transforms


def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: torch.device,
                  class_names: list = None) -> dict:
    """Evaluate model on test set"""
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_logits = []
    
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Single view inference
            outputs = model(images)
            logits = outputs['logits']
            
            predictions = logits.argmax(dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits.append(logits.cpu())
            
            # Update counts
            total_samples += targets.size(0)
            correct_predictions += (predictions == targets).sum().item()
    
    # Calculate metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_targets_tensor = torch.tensor(all_targets)
    
    metrics = compute_metrics(all_logits, all_targets_tensor, model.classifier.out_features)
    
    # Additional metrics
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Classification report
    if class_names:
        target_names = class_names
    else:
        target_names = [f'Class_{i}' for i in range(len(np.unique(all_targets)))]
    
    report = classification_report(
        all_targets, all_predictions, 
        target_names=target_names,
        output_dict=True
    )
    
    return {
        'accuracy': metrics['acc'],
        'balanced_accuracy': metrics['balanced_acc'],
        'top5_accuracy': metrics['top5_acc'],
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'targets': all_targets,
        'logits': all_logits.numpy()
    }


def save_confusion_matrix(cm: np.ndarray, 
                         class_names: list,
                         save_path: Path,
                         title: str = "Confusion Matrix"):
    """Save confusion matrix plot"""
    
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names[:len(cm)],
                yticklabels=class_names[:len(cm)])
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(report: dict, save_path: Path):
    """Save classification report to CSV"""
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Save to CSV
    df.to_csv(save_path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate PaCo-2 Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['val', 'test'], default='test',
                       help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset_config = config['dataset']
    data_config = config['data']
    
    test_transform = create_ufgvc_transforms(
        image_size=data_config['image_size'],
        split=args.dataset,
        use_two_views=False
    )
    
    test_dataset = UFGVCDataset(
        dataset_name=dataset_config['name'],
        root=dataset_config['root'],
        split=args.dataset,
        transform=test_transform,
        download=False  # Assume data already downloaded
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Number of classes: {len(test_dataset.classes)}")
    
    # Create model
    model_config = config['model']
    num_classes = len(test_dataset.classes)
    
    model = PaCoModel(
        backbone_name=model_config['backbone'],
        num_classes=num_classes,
        pretrained=False,  # Will load from checkpoint
        **{k: v for k, v in model_config.items() 
           if k not in ['backbone', 'pretrained']}
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_info = load_checkpoint(model, None, Path(args.checkpoint))
    model = model.to(device)
    
    print("Model loaded successfully")
    print(f"Checkpoint epoch: {checkpoint_info['epoch']}")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, test_dataset.classes)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    
    # Save results
    print(f"\nSaving results to {output_dir}")
    
    # Save confusion matrix
    save_confusion_matrix(
        results['confusion_matrix'],
        test_dataset.classes,
        output_dir / 'confusion_matrix.png',
        f"Confusion Matrix - {dataset_config['name']}"
    )
    
    # Save classification report
    save_classification_report(
        results['classification_report'],
        output_dir / 'classification_report.csv'
    )
    
    # Save numerical results
    np.save(output_dir / 'predictions.npy', results['predictions'])
    np.save(output_dir / 'targets.npy', results['targets'])
    np.save(output_dir / 'logits.npy', results['logits'])
    
    # Save summary
    summary = {
        'dataset': dataset_config['name'],
        'split': args.dataset,
        'model': model_config['backbone'],
        'num_samples': len(test_dataset),
        'num_classes': num_classes,
        'accuracy': float(results['accuracy']),
        'balanced_accuracy': float(results['balanced_accuracy']),
        'top5_accuracy': float(results['top5_accuracy']),
        'checkpoint_epoch': checkpoint_info['epoch']
    }
    
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()
