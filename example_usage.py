"""Example usage of PaCo-2 implementation"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dataset.ufgvc import UFGVCDataset
from src.models import PaCoModel
from src.data_utils import create_ufgvc_transforms, TwoViewDataset


def example_dataset_usage():
    """Example of using UFGVC dataset"""
    print("=== Dataset Usage Example ===")
    
    # Create transforms
    train_transform = create_ufgvc_transforms(
        image_size=224, 
        split='train', 
        use_two_views=True
    )
    
    # Create dataset - will download automatically
    dataset = UFGVCDataset(
        dataset_name="cotton80",
        root="./data",
        split="train",
        download=True
    )
    
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Split: {dataset.split}") 
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {len(dataset.classes)}")
    print(f"First few classes: {dataset.classes[:5]}")
    
    # Wrap with two-view transform
    two_view_dataset = TwoViewDataset(dataset, train_transform)
    
    # Create dataloader
    loader = DataLoader(two_view_dataset, batch_size=4, shuffle=True)
    
    # Get a batch
    for (view1, view2), targets in loader:
        print(f"View1 shape: {view1.shape}")
        print(f"View2 shape: {view2.shape}")
        print(f"Targets: {targets}")
        break
    
    return len(dataset.classes)


def example_model_usage(num_classes):
    """Example of using PaCo-2 model"""
    print("\n=== Model Usage Example ===")
    
    # Create model
    model = PaCoModel(
        backbone_name='resnet18',  # Use ResNet-18 for quick testing
        num_classes=num_classes,
        K=4,           # 4 parts
        r=5,           # 5x5 part window  
        d=64,          # Reduce to 64 dimensions
        lambda_pac=1.0,  # PaC loss weight
        eta_soc=0.1,     # SoC loss weight
        alpha=0.2,       # Triplet margin
        beta=0.05,       # Prototype regularization
    )
    
    print("Model created successfully!")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    B, C, H, W = 2, 3, 224, 224
    x1 = torch.randn(B, C, H, W)  # First view
    x2 = torch.randn(B, C, H, W)  # Second view
    targets = torch.randint(0, num_classes, (B,))
    
    # Training mode (with losses)
    model.train()
    outputs = model(x1, x2, targets)
    print(f"\nTraining output keys: {list(outputs.keys())}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Total loss: {outputs['total']:.4f}")
    print(f"CE loss: {outputs['ce']:.4f}")
    print(f"PaC loss: {outputs['pac']:.4f}")
    print(f"SoC loss: {outputs['soc']:.4f}")
    
    # Inference mode (logits only)
    model.eval()
    with torch.no_grad():
        outputs = model(x1)  # Single view for inference
        print(f"Inference output keys: {list(outputs.keys())}")
        print(f"Predicted classes: {outputs['logits'].argmax(dim=1)}")


def example_configuration():
    """Example of using configuration files"""
    print("\n=== Configuration Example ===")
    
    config_path = Path("configs/ufg_base.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Configuration loaded:")
        print(f"  Dataset: {config['dataset']['name']}")
        print(f"  Backbone: {config['model']['backbone']}")
        print(f"  Batch size: {config['data']['batch_size']}")
        print(f"  Learning rate: {config['optimizer']['lr']}")
        print(f"  Epochs: {config['training']['epochs']}")
        
        # You can modify config programmatically
        config['model']['backbone'] = 'convnext_tiny'
        config['training']['epochs'] = 50
        print(f"  Modified backbone: {config['model']['backbone']}")
        print(f"  Modified epochs: {config['training']['epochs']}")
        
    else:
        print("Configuration file not found!")


def example_training_loop():
    """Example of a simple training loop"""
    print("\n=== Training Loop Example ===")
    
    # Create dataset
    dataset = UFGVCDataset(
        dataset_name="cotton80",
        root="./data", 
        split="train",
        download=False  # Already downloaded
    )
    
    transform = create_ufgvc_transforms(224, 'train', use_two_views=True)
    two_view_dataset = TwoViewDataset(dataset, transform)
    loader = DataLoader(two_view_dataset, batch_size=2, shuffle=True)
    
    # Create model
    model = PaCoModel(
        backbone_name='resnet18',
        num_classes=len(dataset.classes),
        K=2, r=3, d=32  # Small parameters for demo
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("Starting mini training loop...")
    model.train()
    
    for epoch in range(2):  # Just 2 epochs for demo
        total_loss = 0
        num_batches = 0
        
        for batch_idx, ((x1, x2), targets) in enumerate(loader):
            if batch_idx >= 3:  # Only process 3 batches for demo
                break
                
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x1, x2, targets)
            loss = outputs['total']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            print(f"  Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    print("Mini training completed!")


def main():
    print("PaCo-2 Implementation Usage Examples")
    print("=" * 60)
    
    try:
        # Dataset example
        num_classes = example_dataset_usage()
        
        # Model example
        example_model_usage(num_classes)
        
        # Configuration example
        example_configuration()
        
        # Training example
        example_training_loop()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nTo start full training, run:")
        print("python train_clean.py --config configs/ufg_base.yaml")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
