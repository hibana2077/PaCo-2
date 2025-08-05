"""
PaCo-2 Loss Functions: PaC, SoC, and Combined Loss
Following the pseudo code and mathematical formulations from docs/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .utils import CovarUtils


class PaCLoss(nn.Module):
    """
    Part-aware Contrast Loss (PaC)
    Triplet-style contrastive loss on matched parts with Mahalanobis distance
    """
    
    def __init__(self, alpha: float = 0.2, use_semi_hard: bool = True):
        super().__init__()
        self.alpha = alpha
        self.use_semi_hard = use_semi_hard
    
    def forward(self, Z1t: torch.Tensor, Z2t: torch.Tensor, 
                matching: torch.Tensor, negatives_pool: torch.Tensor,
                Sigma_plus: torch.Tensor) -> torch.Tensor:
        """
        Compute PaC loss
        
        Args:
            Z1t, Z2t: Part features (B, K, d)
            matching: Part matching indices (B, K)
            negatives_pool: Negative part features from other samples (N_neg, d)
            Sigma_plus: Shared covariance (B, d, d)
            
        Returns:
            loss: PaC loss value
        """
        B, K, d = Z1t.shape
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            for k in range(K):
                # Get positive pair
                p = Z1t[b, k]  # (d,)
                q = Z2t[b, matching[b, k]]  # (d,)
                
                # Compute positive distance
                d_pos = CovarUtils.mahalanobis_distance(p, q, Sigma_plus[b])
                
                # Compute negative distances
                d_negs = []
                for neg in negatives_pool:
                    d_neg = CovarUtils.mahalanobis_distance(p, neg, Sigma_plus[b])
                    d_negs.append(d_neg)
                
                if len(d_negs) == 0:
                    continue
                
                d_negs = torch.stack(d_negs)
                
                # Select negative based on strategy
                if self.use_semi_hard:
                    # Semi-hard negative: d_neg > d_pos and smallest such distance
                    valid_negs = d_negs[d_negs > d_pos]
                    if len(valid_negs) > 0:
                        d_neg = valid_negs.min()
                    else:
                        d_neg = d_negs.min()  # Hardest negative if no semi-hard
                else:
                    # Hardest negative
                    d_neg = d_negs.min()
                
                # Compute triplet loss
                loss_k = torch.clamp(d_pos - d_neg + self.alpha, min=0.0)
                total_loss += loss_k
                count += 1
        
        return total_loss / max(1, count)


class SoCLoss(nn.Module):
    """
    Second-order Consistency Loss (SoC)
    Minimizes covariance difference between two views with optional prototype regularization
    """
    
    def __init__(self, beta: float = 0.05, metric: str = "fro"):
        super().__init__()
        self.beta = beta
        self.metric = metric
    
    def forward(self, Sigma1: torch.Tensor, Sigma2: torch.Tensor,
                Sigma_proto: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SoC loss
        
        Args:
            Sigma1, Sigma2: Covariance matrices from two views (B, d, d)
            Sigma_proto: Prototype covariances per class (B, d, d), optional
            
        Returns:
            loss: SoC loss value
        """
        # Main term: consistency between two views
        d12 = CovarUtils.spd_distance(Sigma1, Sigma2, self.metric)
        loss = (d12 ** 2).mean()
        
        # Optional prototype regularization
        if Sigma_proto is not None and self.beta > 0:
            Sigma_bar = 0.5 * (Sigma1 + Sigma2)
            d_proto = CovarUtils.spd_distance(Sigma_bar, Sigma_proto, self.metric)
            loss = loss + self.beta * (d_proto ** 2).mean()
        
        return loss


class WeightedCELoss(nn.Module):
    """
    Part consistency weighted Cross-Entropy Loss
    Weights CE loss based on part saliency consistency
    """
    
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                saliency_map: Optional[torch.Tensor] = None,
                peaks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted CE loss
        
        Args:
            logits: Classification logits (B, num_classes)
            targets: Ground truth labels (B,)
            saliency_map: Saliency maps (B, H, W), optional
            peaks: Peak locations (B, K, 2), optional
            
        Returns:
            loss: Weighted CE loss
        """
        base_loss = self.ce_loss(logits, targets)
        
        if saliency_map is None or peaks is None or self.gamma == 0:
            return base_loss
        
        # Compute part consistency weights
        B, K, _ = peaks.shape
        weights = torch.ones(B, device=logits.device)
        
        for b in range(B):
            total_weight = 0.0
            for k in range(K):
                h, w = peaks[b, k]
                # Simple window average around peak
                window_size = 5
                h_start = max(0, h - window_size // 2)
                h_end = min(saliency_map.shape[1], h + window_size // 2 + 1)
                w_start = max(0, w - window_size // 2)
                w_end = min(saliency_map.shape[2], w + window_size // 2 + 1)
                
                window_weight = saliency_map[b, h_start:h_end, w_start:w_end].mean()
                total_weight += window_weight
            
            weights[b] = total_weight / K
        
        # Apply weighting
        weighted_loss = (1 - self.gamma) * base_loss + self.gamma * (weights * base_loss).mean()
        return weighted_loss


class PaCoLoss(nn.Module):
    """
    Combined PaCo-2 Loss: CE + PaC + SoC
    """
    
    def __init__(self, 
                 lambda_pac: float = 1.0,
                 eta_soc: float = 0.1,
                 alpha: float = 0.2,
                 beta: float = 0.05,
                 gamma: float = 0.1,
                 metric: str = "fro",
                 use_weighted_ce: bool = True,
                 use_semi_hard: bool = True):
        super().__init__()
        
        self.lambda_pac = lambda_pac
        self.eta_soc = eta_soc
        
        # Component losses
        if use_weighted_ce:
            self.ce_loss = WeightedCELoss(gamma=gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.pac_loss = PaCLoss(alpha=alpha, use_semi_hard=use_semi_hard)
        self.soc_loss = SoCLoss(beta=beta, metric=metric)
        
        self.use_weighted_ce = use_weighted_ce
    
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                Z1t: torch.Tensor,
                Z2t: torch.Tensor,
                matching: torch.Tensor,
                negatives_pool: torch.Tensor,
                Sigma1: torch.Tensor,
                Sigma2: torch.Tensor,
                Sigma_plus: torch.Tensor,
                Sigma_proto: Optional[torch.Tensor] = None,
                saliency_map: Optional[torch.Tensor] = None,
                peaks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined PaCo loss
        
        Returns:
            Dictionary containing individual and total losses
        """
        # CE Loss
        if self.use_weighted_ce and saliency_map is not None and peaks is not None:
            L_ce = self.ce_loss(logits, targets, saliency_map, peaks)
        else:
            if isinstance(self.ce_loss, WeightedCELoss):
                L_ce = self.ce_loss(logits, targets)
            else:
                L_ce = self.ce_loss(logits, targets)
        
        # PaC Loss
        L_pac = self.pac_loss(Z1t, Z2t, matching, negatives_pool, Sigma_plus)
        
        # SoC Loss
        L_soc = self.soc_loss(Sigma1, Sigma2, Sigma_proto)
        
        # Total Loss
        L_total = L_ce + self.lambda_pac * L_pac + self.eta_soc * L_soc
        
        return {
            'total': L_total,
            'ce': L_ce,
            'pac': L_pac,
            'soc': L_soc
        }
