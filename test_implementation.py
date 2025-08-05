"""Quick test script to verify PaCo-2 implementation"""

import sys
from pathlib import Path
import torch
import timm

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.dataset.ufgvc import UFGVCDataset
        print("✓ UFGVC Dataset import successful")
    except Exception as e:
        print(f"✗ UFGVC Dataset import failed: {e}")
    
    try:
        from src.models import PaCoModel
        print("✓ PaCoModel import successful")
    except Exception as e:
        print(f"✗ PaCoModel import failed: {e}")
    
    try:
        from src.models.losses import PaCoLoss, SoCLoss, PaCLoss
        print("✓ Loss functions import successful")
    except Exception as e:
        print(f"✗ Loss functions import failed: {e}")
    
    try:
        from src.models.utils import PartSampler, HungarianMatcher, CovarUtils
        print("✓ Utilities import successful")
    except Exception as e:
        print(f"✗ Utilities import failed: {e}")
    
    try:
        from src.data_utils import create_ufgvc_transforms, TwoViewDataset
        print("✓ Data utils import successful")
    except Exception as e:
        print(f"✗ Data utils import failed: {e}")


def test_model_creation():
    """Test model creation with different backbones"""
    print("\nTesting model creation...")
    
    from src.models import PaCoModel
    
    backbones = ['resnet18', 'convnext_tiny', 'efficientnet_b0']
    
    for backbone in backbones:
        try:
            # Check if backbone is available in timm
            available_models = timm.list_models(backbone.split('_')[0] + '*')
            if backbone not in available_models:
                print(f"⚠ {backbone} not available in timm, skipping...")
                continue
            
            model = PaCoModel(
                backbone_name=backbone,
                num_classes=10,
                K=2,  # Small for testing
                r=3,
                d=32
            )
            print(f"✓ {backbone} model created successfully")
            
            # Test forward pass
            x1 = torch.randn(2, 3, 224, 224)
            x2 = torch.randn(2, 3, 224, 224)
            targets = torch.randint(0, 10, (2,))
            
            with torch.no_grad():
                outputs = model(x1, x2, targets)
                print(f"  - Forward pass successful, output keys: {list(outputs.keys())}")
                
        except Exception as e:
            print(f"✗ {backbone} model creation failed: {e}")


def test_loss_functions():
    """Test loss function computations"""
    print("\nTesting loss functions...")
    
    from src.models.losses import PaCoLoss
    
    try:
        criterion = PaCoLoss()
        print("✓ PaCoLoss created successfully")
        
        # Mock data for testing
        B, K, d = 2, 4, 32
        num_classes = 10
        
        logits = torch.randn(B, num_classes)
        targets = torch.randint(0, num_classes, (B,))
        Z1t = torch.randn(B, K, d)
        Z2t = torch.randn(B, K, d)
        matching = torch.randint(0, K, (B, K))
        negatives_pool = torch.randn(10, d)  # Some negative samples
        Sigma1 = torch.eye(d).unsqueeze(0).expand(B, -1, -1) + 0.1 * torch.randn(B, d, d)
        Sigma2 = torch.eye(d).unsqueeze(0).expand(B, -1, -1) + 0.1 * torch.randn(B, d, d)
        Sigma_plus = 0.5 * (Sigma1 + Sigma2) + 1e-3 * torch.eye(d).unsqueeze(0).expand(B, -1, -1)
        
        with torch.no_grad():
            losses = criterion(
                logits=logits,
                targets=targets,
                Z1t=Z1t,
                Z2t=Z2t,
                matching=matching,
                negatives_pool=negatives_pool,
                Sigma1=Sigma1,
                Sigma2=Sigma2,
                Sigma_plus=Sigma_plus
            )
            print(f"  - Loss computation successful: {losses}")
            
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")


def test_transforms():
    """Test data transforms"""
    print("\nTesting data transforms...")
    
    from src.data_utils import create_ufgvc_transforms
    from PIL import Image
    import numpy as np
    
    try:
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test single view transform
        transform = create_ufgvc_transforms(image_size=224, split='train', use_two_views=False)
        single_output = transform(dummy_img)
        print(f"✓ Single view transform: {single_output.shape}")
        
        # Test two view transform
        two_view_transform = create_ufgvc_transforms(image_size=224, split='train', use_two_views=True)
        view1, view2 = two_view_transform(dummy_img)
        print(f"✓ Two view transform: {view1.shape}, {view2.shape}")
        
    except Exception as e:
        print(f"✗ Transform test failed: {e}")


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    import yaml
    
    config_path = Path("configs/ufg_base.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration loaded successfully")
            print(f"  - Dataset: {config['dataset']['name']}")
            print(f"  - Backbone: {config['model']['backbone']}")
            print(f"  - Parameters: K={config['model']['K']}, d={config['model']['d']}")
        except Exception as e:
            print(f"✗ Configuration loading failed: {e}")
    else:
        print("⚠ Configuration file not found")


def main():
    print("PaCo-2 Implementation Test")
    print("=" * 50)
    
    test_imports()
    test_model_creation()
    test_loss_functions()
    test_transforms()
    test_configuration()
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == '__main__':
    main()
