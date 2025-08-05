"""
PaCo-2 Main Model Implementation
Integrates timm CNN backbones with Part-aware Contrast and Second-order Consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple, Any
from .utils import PartSampler, HungarianMatcher, CovarUtils
from .losses import PaCoLoss


class PaCoModel(nn.Module):
    """
    PaCo-2 Model: Part-aware Contrast with Second-order Consistency
    
    Integrates with timm CNN backbones following docs/timm_req.md requirements
    """
    
    def __init__(self,
                 backbone_name: str = 'resnet50',
                 num_classes: int = 1000,
                 pretrained: bool = True,
                 # Part sampling parameters
                 K: int = 4,
                 r: int = 5,
                 d: int = 64,
                 # Loss parameters
                 lambda_pac: float = 1.0,
                 eta_soc: float = 0.1,
                 alpha: float = 0.2,
                 beta: float = 0.05,
                 gamma: float = 0.1,
                 # Technical parameters
                 epsilon: float = 1e-5,
                 tau: float = 1e-5,
                 metric: str = "fro",
                 use_mahalanobis_warmup: bool = True,
                 warmup_epochs: int = 3,
                 # Optional features
                 use_weighted_ce: bool = True,
                 use_semi_hard: bool = True,
                 use_class_proto: bool = True,
                 proto_momentum: float = 0.9):
        
        super().__init__()
        
        # Store hyperparameters
        self.K = K
        self.r = r
        self.d = d
        self.epsilon = epsilon
        self.tau = tau
        self.use_mahalanobis_warmup = use_mahalanobis_warmup
        self.warmup_epochs = warmup_epochs
        self.use_class_proto = use_class_proto
        self.proto_momentum = proto_momentum
        self.current_epoch = 0
        
        # Create timm backbone following docs/timm_req.md
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone.forward_features(dummy_input)
            self.feature_dim = features.shape[1]  # C
            self.feature_size = features.shape[2:]  # (H, W)
        
        # Remove classifier head from backbone to use forward_features
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features if hasattr(self.backbone.classifier, 'in_features') else self.backbone.classifier.in_channels
        elif hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features if hasattr(self.backbone.head, 'in_features') else self.backbone.head.in_channels
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
        else:
            in_features = self.feature_dim
        
        # Part sampling module
        self.part_sampler = PartSampler(K=K, r=r)
        
        # Dimension reduction: 1x1 conv C -> d
        self.dim_reduction = nn.Conv2d(self.feature_dim, d, 1, bias=False)
        nn.init.kaiming_normal_(self.dim_reduction.weight)
        
        # Hungarian matcher
        self.matcher = HungarianMatcher(use_cosine=True)  # Start with cosine
        
        # Global classifier head
        self.classifier = nn.Linear(in_features, num_classes)
        
        # Loss function
        self.criterion = PaCoLoss(
            lambda_pac=lambda_pac,
            eta_soc=eta_soc,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            metric=metric,
            use_weighted_ce=use_weighted_ce,
            use_semi_hard=use_semi_hard
        )
        
        # Class prototype covariances (optional)
        if use_class_proto:
            self.register_buffer('class_prototypes', torch.zeros(num_classes, d, d))
            self.register_buffer('class_counts', torch.zeros(num_classes))
        
        print(f"PaCoModel initialized:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Feature dim: {self.feature_dim} -> {d}")
        print(f"  Feature size: {self.feature_size}")
        print(f"  Classes: {num_classes}")
        print(f"  Parts (K): {K}, Window (r): {r}")
    
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None, 
                return_features: bool = False) -> Dict[str, Any]:
        """
        Forward pass for PaCo-2 model
        
        Args:
            x1: First view images (B, 3, H, W)
            x2: Second view images (B, 3, H, W), optional for inference
            targets: Ground truth labels (B,), optional
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing logits, losses, and optional features
        """
        if self.training and x2 is None:
            raise ValueError("Second view x2 is required during training")
        
        # Extract backbone features
        F1 = self.backbone.forward_features(x1)  # (B, C, H, W)
        
        # Global features for classification
        g1 = torch.nn.functional.adaptive_avg_pool2d(F1, 1).flatten(1)  # (B, C)
        
        if x2 is not None:
            F2 = self.backbone.forward_features(x2)  # (B, C, H, W)
            g2 = torch.nn.functional.adaptive_avg_pool2d(F2, 1).flatten(1)  # (B, C)
            g = (g1 + g2) / 2  # Average global features
        else:
            g = g1
        
        # Classification logits
        logits = self.classifier(g)
        
        result = {'logits': logits}
        
        # Training mode: compute PaCo losses
        if self.training and x2 is not None:
            # Part sampling
            Z1, peaks1 = self.part_sampler(F1)  # (B, K, C), (B, K, 2)
            Z2, peaks2 = self.part_sampler(F2)  # (B, K, C), (B, K, 2)
            
            # Dimension reduction
            Z1t = self._reduce_dimensions(Z1)  # (B, K, d)
            Z2t = self._reduce_dimensions(Z2)  # (B, K, d)
            
            # Covariance computation
            Sigma1, mu1 = CovarUtils.compute_covariance(Z1t, self.epsilon)
            Sigma2, mu2 = CovarUtils.compute_covariance(Z2t, self.epsilon)
            
            # Sigma_plus with additional regularization
            Sigma_plus = 0.5 * (Sigma1 + Sigma2)
            I = torch.eye(self.d, device=Sigma_plus.device).unsqueeze(0).expand_as(Sigma_plus)
            Sigma_plus = Sigma_plus + self.tau * I
            
            # Part matching (use Mahalanobis after warmup)
            if self.use_mahalanobis_warmup and self.current_epoch >= self.warmup_epochs:
                self.matcher.use_cosine = False
                matching = self.matcher(Z1t, Z2t, Sigma_plus)
            else:
                self.matcher.use_cosine = True
                matching = self.matcher(Z1t, Z2t)
            
            # Create negatives pool from other samples in batch
            negatives_pool = self._create_negatives_pool(Z1t, Z2t)
            
            # Get class prototypes if using them
            Sigma_proto = None
            if self.use_class_proto and targets is not None:
                Sigma_proto = self._get_class_prototypes(targets)
            
            # Compute saliency map for weighted CE (optional)
            saliency_map = None
            peaks = None
            if hasattr(self.criterion.ce_loss, 'gamma') and self.criterion.ce_loss.gamma > 0:
                # Simple saliency: L2 norm across channels
                saliency_map = torch.norm(F1, p=2, dim=1)  # (B, H, W)
                peaks = peaks1
            
            # Compute losses
            if targets is not None:
                losses = self.criterion(
                    logits=logits,
                    targets=targets,
                    Z1t=Z1t,
                    Z2t=Z2t,
                    matching=matching,
                    negatives_pool=negatives_pool,
                    Sigma1=Sigma1,
                    Sigma2=Sigma2,
                    Sigma_plus=Sigma_plus,
                    Sigma_proto=Sigma_proto,
                    saliency_map=saliency_map,
                    peaks=peaks
                )
                
                result.update(losses)
                
                # Update class prototypes
                if self.use_class_proto:
                    self._update_class_prototypes(targets, Sigma1, Sigma2)
        
        # Return additional features if requested
        if return_features:
            features = {'global_features': g}
            if self.training and x2 is not None:
                features.update({
                    'part_features_1': Z1t,
                    'part_features_2': Z2t,
                    'covariances_1': Sigma1,
                    'covariances_2': Sigma2,
                    'peaks_1': peaks1,
                    'peaks_2': peaks2,
                    'matching': matching
                })
            result['features'] = features
        
        return result
    
    def _reduce_dimensions(self, Z: torch.Tensor) -> torch.Tensor:
        """Apply 1x1 conv dimension reduction to part features"""
        B, K, C = Z.shape
        Z_reshaped = Z.view(B * K, C, 1, 1)  # Reshape for conv2d
        Zt_reshaped = self.dim_reduction(Z_reshaped)  # (B*K, d, 1, 1)
        Zt = Zt_reshaped.view(B, K, self.d)  # (B, K, d)
        return Zt
    
    def _create_negatives_pool(self, Z1t: torch.Tensor, Z2t: torch.Tensor) -> torch.Tensor:
        """Create pool of negative part features from other samples in batch"""
        B, K, d = Z1t.shape
        
        # Combine all parts from all samples except current
        all_parts = []
        for b in range(B):
            # Get parts from other samples
            other_indices = [i for i in range(B) if i != b]
            if len(other_indices) > 0:
                other_Z1 = Z1t[other_indices]  # (B-1, K, d)
                other_Z2 = Z2t[other_indices]  # (B-1, K, d)
                other_parts = torch.cat([other_Z1, other_Z2], dim=1)  # (B-1, 2K, d)
                all_parts.append(other_parts.view(-1, d))  # ((B-1)*2K, d)
        
        if len(all_parts) > 0:
            negatives_pool = torch.cat(all_parts, dim=0)  # (N_neg, d)
        else:
            # Fallback: empty pool
            negatives_pool = torch.empty(0, d, device=Z1t.device)
        
        return negatives_pool
    
    def _get_class_prototypes(self, targets: torch.Tensor) -> torch.Tensor:
        """Get class prototype covariances for given targets"""
        B = targets.shape[0]
        proto_covs = torch.zeros(B, self.d, self.d, device=targets.device)
        
        for b, target in enumerate(targets):
            if self.class_counts[target] > 0:
                proto_covs[b] = self.class_prototypes[target]
            else:
                # Initialize with identity if no prototype yet
                proto_covs[b] = torch.eye(self.d, device=targets.device)
        
        return proto_covs
    
    def _update_class_prototypes(self, targets: torch.Tensor, 
                                Sigma1: torch.Tensor, Sigma2: torch.Tensor):
        """Update class prototype covariances using EMA"""
        Sigma_bar = 0.5 * (Sigma1 + Sigma2)  # (B, d, d)
        
        for b, target in enumerate(targets):
            target = target.item()
            
            if self.class_counts[target] == 0:
                # First sample for this class
                self.class_prototypes[target] = Sigma_bar[b].detach()
                self.class_counts[target] = 1
            else:
                # EMA update
                self.class_prototypes[target] = (
                    self.proto_momentum * self.class_prototypes[target] +
                    (1 - self.proto_momentum) * Sigma_bar[b].detach()
                )
                self.class_counts[target] += 1
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling"""
        self.current_epoch = epoch
        
        # Switch to Mahalanobis distance after warmup
        if self.use_mahalanobis_warmup and epoch >= self.warmup_epochs:
            print(f"Epoch {epoch}: Switching to Mahalanobis distance for matching")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone.__class__.__name__,
            'num_classes': self.classifier.out_features,
            'feature_dim': self.feature_dim,
            'reduced_dim': self.d,
            'num_parts': self.K,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'current_epoch': self.current_epoch,
            'using_mahalanobis': not self.matcher.use_cosine,
            'class_prototypes_initialized': self.class_counts.sum().item() if self.use_class_proto else 0
        }
