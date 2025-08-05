"""Data augmentation utilities for PaCo-2"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import numpy as np
from typing import Tuple, Optional


class TwoViewTransform:
    """Create two augmented views of the same image for contrastive learning"""
    
    def __init__(self, base_transform, view1_transform, view2_transform):
        self.base_transform = base_transform
        self.view1_transform = view1_transform  
        self.view2_transform = view2_transform
    
    def __call__(self, x):
        # Apply base transform first
        if self.base_transform:
            x = self.base_transform(x)
        
        # Create two views
        view1 = self.view1_transform(x)
        view2 = self.view2_transform(x)
        
        return view1, view2


class LightOcclusionAndShuffle:
    """
    Light occlusion and patch shuffling following CLE-ViT methodology
    For part-aware learning with controlled corruption
    """
    
    def __init__(self, 
                 occlusion_prob: float = 0.3,
                 occlusion_ratio: float = 0.1,  # Small ratio as recommended
                 shuffle_prob: float = 0.3,
                 patch_size: int = 16,
                 max_patches_to_shuffle: int = 4):
        self.occlusion_prob = occlusion_prob
        self.occlusion_ratio = occlusion_ratio
        self.shuffle_prob = shuffle_prob
        self.patch_size = patch_size
        self.max_patches_to_shuffle = max_patches_to_shuffle
    
    def __call__(self, img):
        """Apply light occlusion and patch shuffling"""
        
        if not isinstance(img, torch.Tensor):
            # Convert PIL to tensor if needed
            img = TF.to_tensor(img)
        
        C, H, W = img.shape
        
        # Light occlusion
        if random.random() < self.occlusion_prob:
            img = self._apply_occlusion(img)
        
        # Patch shuffling
        if random.random() < self.shuffle_prob:
            img = self._apply_patch_shuffling(img)
        
        return img
    
    def _apply_occlusion(self, img):
        """Apply small rectangular occlusions"""
        C, H, W = img.shape
        
        # Calculate occlusion size
        total_pixels = H * W
        occlude_pixels = int(total_pixels * self.occlusion_ratio)
        
        # Square occlusion
        occlude_h = occlude_w = int(np.sqrt(occlude_pixels))
        occlude_h = min(occlude_h, H // 4)  # Limit size
        occlude_w = min(occlude_w, W // 4)
        
        # Random position
        start_h = random.randint(0, H - occlude_h)
        start_w = random.randint(0, W - occlude_w)
        
        # Apply occlusion (set to mean value)
        img_mean = img.mean()
        img[:, start_h:start_h+occlude_h, start_w:start_w+occlude_w] = img_mean
        
        return img
    
    def _apply_patch_shuffling(self, img):
        """Apply patch shuffling to small regions"""
        C, H, W = img.shape
        
        # Calculate number of patches
        patches_h = H // self.patch_size
        patches_w = W // self.patch_size
        
        if patches_h < 2 or patches_w < 2:
            return img  # Skip if too few patches
        
        # Extract patches
        patches = []
        positions = []
        
        for i in range(patches_h):
            for j in range(patches_w):
                start_h = i * self.patch_size
                end_h = min((i + 1) * self.patch_size, H)
                start_w = j * self.patch_size
                end_w = min((j + 1) * self.patch_size, W)
                
                patch = img[:, start_h:end_h, start_w:end_w].clone()
                patches.append(patch)
                positions.append((start_h, end_h, start_w, end_w))
        
        # Shuffle a subset of patches
        num_to_shuffle = min(self.max_patches_to_shuffle, len(patches) // 2)
        indices_to_shuffle = random.sample(range(len(patches)), num_to_shuffle * 2)
        
        # Swap patches in pairs
        for i in range(0, len(indices_to_shuffle), 2):
            if i + 1 < len(indices_to_shuffle):
                idx1, idx2 = indices_to_shuffle[i], indices_to_shuffle[i + 1]
                patches[idx1], patches[idx2] = patches[idx2], patches[idx1]
        
        # Reconstruct image
        result = img.clone()
        for patch, (start_h, end_h, start_w, end_w) in zip(patches, positions):
            result[:, start_h:end_h, start_w:end_w] = patch
        
        return result


class RandAugmentLight:
    """Lightweight RandAugment for UFGVC as recommended in docs"""
    
    def __init__(self, n_ops: int = 2, magnitude: int = 5):
        self.n_ops = n_ops
        self.magnitude = magnitude
        
        # Define operations with controlled magnitude for fine-grained tasks
        self.ops = [
            ('rotate', lambda img, mag: TF.rotate(img, random.uniform(-mag*3, mag*3))),
            ('brightness', lambda img, mag: TF.adjust_brightness(img, 1 + random.uniform(-mag*0.05, mag*0.05))),
            ('contrast', lambda img, mag: TF.adjust_contrast(img, 1 + random.uniform(-mag*0.05, mag*0.05))),
            ('saturation', lambda img, mag: TF.adjust_saturation(img, 1 + random.uniform(-mag*0.05, mag*0.05))),
            ('hue', lambda img, mag: TF.adjust_hue(img, random.uniform(-mag*0.02, mag*0.02))),
            ('sharpness', lambda img, mag: TF.adjust_sharpness(img, 1 + random.uniform(-mag*0.1, mag*0.1))),
        ]
    
    def __call__(self, img):
        """Apply random augmentations"""
        ops = random.choices(self.ops, k=self.n_ops)
        
        for op_name, op_func in ops:
            try:
                img = op_func(img, self.magnitude)
            except:
                continue  # Skip if operation fails
        
        return img


def create_ufgvc_transforms(image_size: int = 224, 
                           split: str = 'train',
                           use_two_views: bool = False) -> transforms.Compose:
    """
    Create transforms for UFGVC datasets following docs/exp_data.md recommendations
    
    Args:
        image_size: Target image size (default 224)
        split: 'train', 'val', or 'test'
        use_two_views: Whether to create two views for contrastive learning
    
    Returns:
        Transform compose or TwoViewTransform
    """
    
    # ImageNet normalization (standard for timm models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        # Base preprocessing
        base_transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # Long edge scaling
            transforms.CenterCrop(image_size),  # Short edge alignment
        ])
        
        # Standard training augmentations
        standard_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            RandAugmentLight(n_ops=2, magnitude=5),  # Light augmentation
            transforms.ToTensor(),
            normalize
        ])
        
        if use_two_views:
            # View 1: Standard augmentation
            view1_transform = standard_augment
            
            # View 2: Standard + light occlusion/shuffle (following CLE-ViT)
            view2_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                RandAugmentLight(n_ops=2, magnitude=5),
                transforms.ToTensor(),
                LightOcclusionAndShuffle(
                    occlusion_prob=0.3,
                    occlusion_ratio=0.1,  # Small ratio to preserve matchable structure
                    shuffle_prob=0.3,
                    max_patches_to_shuffle=4
                ),
                normalize
            ])
            
            return TwoViewTransform(base_transform, view1_transform, view2_transform)
        else:
            return transforms.Compose([
                base_transform,
                standard_augment
            ])
    
    else:  # val or test
        # Simple preprocessing for validation/test
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),  # Center crop only
            transforms.ToTensor(),
            normalize
        ])


def create_timm_transforms(model_name: str, split: str = 'train', 
                          use_two_views: bool = False) -> transforms.Compose:
    """
    Create transforms using timm's data config following docs/timm_req.md
    
    Args:
        model_name: Name of the timm model
        split: 'train', 'val', or 'test'  
        use_two_views: Whether to create two views for contrastive learning
    
    Returns:
        Transform compose or TwoViewTransform
    """
    import timm
    
    # Create model to get data config
    model = timm.create_model(model_name, pretrained=True)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    
    if split == 'train':
        # Get base transform from timm
        base_transform = timm.data.create_transform(**data_cfg, is_training=True)
        
        if use_two_views:
            # Create modified transforms for two views
            # Extract components
            image_size = data_cfg.get('input_size', [3, 224, 224])[-1]
            
            # View 1: Standard timm training transform
            view1_transform = base_transform
            
            # View 2: Modified with light occlusion/shuffle
            view2_components = []
            
            # Add base preprocessing
            view2_components.extend([
                transforms.Resize(int(image_size * 1.14)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                LightOcclusionAndShuffle(),
                transforms.Normalize(
                    mean=data_cfg.get('mean', [0.485, 0.456, 0.406]),
                    std=data_cfg.get('std', [0.229, 0.224, 0.225])
                )
            ])
            
            view2_transform = transforms.Compose(view2_components)
            
            return TwoViewTransform(None, view1_transform, view2_transform)
        else:
            return base_transform
    else:
        # Validation/test transform from timm
        return timm.data.create_transform(**data_cfg, is_training=False)


# Utility class for handling two-view datasets
class TwoViewDataset:
    """Wrapper to convert single-view dataset to two-view dataset"""
    
    def __init__(self, dataset, two_view_transform):
        self.dataset = dataset
        self.two_view_transform = two_view_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Apply two-view transform
        if self.two_view_transform:
            view1, view2 = self.two_view_transform(image)
            return (view1, view2), label
        else:
            return image, label
    
    @property
    def classes(self):
        return getattr(self.dataset, 'classes', None)
    
    @property
    def class_to_idx(self):
        return getattr(self.dataset, 'class_to_idx', None)
