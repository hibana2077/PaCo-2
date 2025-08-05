"""
Optimized PaCo-2 Model Implementation: Integrating High Priority Optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple, Any
from .utils import PartSampler, HungarianMatcher, CovarUtils
from .optimized_losses import OptimizedPaCoLoss


class OptimizedPaCoModel(nn.Module):
    """
    Optimized PaCo-2 Model: Integrating High Priority Optimizations
    
    High Priority Optimizations Integrated:
    1. Forced small parameters: K=4, d=64 (keep small dimensions and small parts)
    2. SoC fixed to use Frobenius distance
    3. Vectorized whitening PaC computation
    4. Two-stage negative sampling
    """
    
    def __init__(self,
                 backbone_name: str = 'resnet50',
                 num_classes: int = 1000,
                 pretrained: bool = True,
                 # Forced small parameter settings (high priority optimization)
                 K: int = 4,        # Max number of parts
                 r: int = 5,        # Sampling window size
                 d: int = 64,       # Max feature dimension
                 # Loss parameters
                 lambda_pac: float = 1.0,
                 eta_soc: float = 0.1,
                 alpha: float = 0.2,
                 beta: float = 0.05,
                 gamma: float = 0.1,
                 # Optimization parameters
                 top_m_candidates: int = 32,  # Candidate count for two-stage filtering
                 epsilon: float = 1e-5,
                 tau: float = 1e-5,
                 # Technical parameters
                 use_mahalanobis_warmup: bool = True,
                 warmup_epochs: int = 3,
                 # Optional features
                 use_weighted_ce: bool = True,
                 use_semi_hard: bool = True,
                 use_class_proto: bool = True,
                 proto_momentum: float = 0.9):
        
        super().__init__()
        
        # 強制最優配置（高優先級優化）
        if K > 4:
            print(f"Warning: K={K} > 4, forcing K=4 for optimal performance")
            K = 4
        if d > 64:
            print(f"Warning: d={d} > 64, forcing d=64 for optimal performance")
            d = 64
        
        # 存儲超參數
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
        self.top_m_candidates = top_m_candidates
        
        # 創建 timm backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        
        # 獲取特徵維度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone.forward_features(dummy_input)
            self.feature_dim = features.shape[1]  # C
            self.feature_size = features.shape[2:]  # (H, W)
        
        # 獲取分類器輸入維度
        if hasattr(self.backbone, 'classifier'):
            in_features = getattr(self.backbone.classifier, 'in_features', 
                                getattr(self.backbone.classifier, 'in_channels', self.feature_dim))
        elif hasattr(self.backbone, 'head'):
            in_features = getattr(self.backbone.head, 'in_features', 
                                getattr(self.backbone.head, 'in_channels', self.feature_dim))
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
        else:
            in_features = self.feature_dim
        
        # 部位採樣模組
        self.part_sampler = PartSampler(K=K, r=r)
        
        # 維度縮減：1x1 conv C -> d
        self.dim_reduction = nn.Conv2d(self.feature_dim, d, 1, bias=False)
        nn.init.kaiming_normal_(self.dim_reduction.weight)
        
        # 匈牙利匹配器
        self.matcher = HungarianMatcher(use_cosine=True)  # 熱身期使用餘弦
        
        # 全域分類器
        self.classifier = nn.Linear(in_features, num_classes)
        
        # 使用優化版損失函數
        self.criterion = OptimizedPaCoLoss(
            lambda_pac=lambda_pac,
            eta_soc=eta_soc,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            use_weighted_ce=use_weighted_ce,
            use_semi_hard=use_semi_hard,
            top_m_candidates=top_m_candidates
        )
        
        # 類別原型協方差（可選）
        if use_class_proto:
            self.register_buffer('class_prototypes', torch.zeros(num_classes, d, d))
            self.register_buffer('class_counts', torch.zeros(num_classes))
        
        print(f"OptimizedPaCoModel initialized:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Feature dim: {self.feature_dim} -> {d}")
        print(f"  Feature size: {self.feature_size}")
        print(f"  Classes: {num_classes}")
        print(f"  Parts (K): {K}, Window (r): {r}")
        print(f"  Optimizations: K<=4, d<=64, Frobenius SoC, Vectorized PaC")
        print(f"  Top-M candidates: {top_m_candidates}")
    
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None, 
                return_features: bool = False) -> Dict[str, Any]:
        """
        Optimized forward pass
        """
        if self.training and x2 is None:
            raise ValueError("Second view x2 is required during training")
        
        # 提取 backbone 特徵
        F1 = self.backbone.forward_features(x1)  # (B, C, H, W)
        
        # 全域特徵用於分類
        g1 = torch.nn.functional.adaptive_avg_pool2d(F1, 1).flatten(1)  # (B, C)
        
        if x2 is not None:
            F2 = self.backbone.forward_features(x2)  # (B, C, H, W)
            g2 = torch.nn.functional.adaptive_avg_pool2d(F2, 1).flatten(1)  # (B, C)
            g = (g1 + g2) / 2  # 平均全域特徵
        else:
            g = g1
        
        # 分類 logits
        logits = self.classifier(g)
        
        result = {'logits': logits}
        
        # 訓練模式：計算優化版 PaCo 損失
        if self.training and x2 is not None:
            # 部位採樣
            Z1, peaks1 = self.part_sampler(F1)  # (B, K, C), (B, K, 2)
            Z2, peaks2 = self.part_sampler(F2)  # (B, K, C), (B, K, 2)
            
            # 維度縮減
            Z1t = self._reduce_dimensions(Z1)  # (B, K, d)
            Z2t = self._reduce_dimensions(Z2)  # (B, K, d)
            
            # 協方差計算
            Sigma1, mu1 = CovarUtils.compute_covariance(Z1t, self.epsilon)
            Sigma2, mu2 = CovarUtils.compute_covariance(Z2t, self.epsilon)
            
            # Sigma_plus 附加正則化
            Sigma_plus = 0.5 * (Sigma1 + Sigma2)
            I = torch.eye(self.d, device=Sigma_plus.device).unsqueeze(0).expand_as(Sigma_plus)
            Sigma_plus = Sigma_plus + self.tau * I
            
            # 部位匹配（熱身後使用 Mahalanobis）
            if self.use_mahalanobis_warmup and self.current_epoch >= self.warmup_epochs:
                self.matcher.use_cosine = False
                matching = self.matcher(Z1t, Z2t, Sigma_plus)
            else:
                self.matcher.use_cosine = True
                matching = self.matcher(Z1t, Z2t)
            
            # 創建負對池
            negatives_pool = self._create_optimized_negatives_pool(Z1t, Z2t)
            
            # 獲取類別原型（如果使用）
            Sigma_proto = None
            if self.use_class_proto and targets is not None:
                Sigma_proto = self._get_class_prototypes(targets)
            
            # 計算顯著性圖（可選）
            saliency_map = None
            peaks = None
            if hasattr(self.criterion.ce_loss, 'gamma') and self.criterion.ce_loss.gamma > 0:
                saliency_map = torch.norm(F1, p=2, dim=1)  # (B, H, W)
                peaks = peaks1
            
            # 計算優化版損失
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
                
                # 更新類別原型
                if self.use_class_proto:
                    self._update_class_prototypes(targets, Sigma1, Sigma2)
        
        # 返回額外特徵（如果需要）
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
        """對部位特徵應用 1x1 conv 維度縮減"""
        B, K, C = Z.shape
        Z_reshaped = Z.view(B * K, C, 1, 1)  # 重塑為 conv2d 格式
        Zt_reshaped = self.dim_reduction(Z_reshaped)  # (B*K, d, 1, 1)
        Zt = Zt_reshaped.view(B, K, self.d)  # (B, K, d)
        return Zt
    
    def _create_optimized_negatives_pool(self, Z1t: torch.Tensor, Z2t: torch.Tensor) -> torch.Tensor:
        """
        創建優化版負對池
        針對大 batch 進行優化，避免過多負對
        """
        B, K, d = Z1t.shape
        
        # 限制每個樣本的負對數量，避免計算爆炸
        max_negatives_per_sample = min(2 * K, 16)  # 每個樣本最多 16 個負對
        
        all_parts = []
        for b in range(B):
            # 獲取其他樣本的部位
            other_indices = [i for i in range(B) if i != b]
            if len(other_indices) > 0:
                # 隨機選擇其他樣本以控制負對數量
                if len(other_indices) > max_negatives_per_sample // (2 * K):
                    selected_indices = torch.randperm(len(other_indices))[:max_negatives_per_sample // (2 * K)]
                    other_indices = [other_indices[i] for i in selected_indices]
                
                other_Z1 = Z1t[other_indices]  # (N_other, K, d)
                other_Z2 = Z2t[other_indices]  # (N_other, K, d)
                other_parts = torch.cat([other_Z1, other_Z2], dim=1)  # (N_other, 2K, d)
                
                # 限制負對數量
                other_parts_flat = other_parts.view(-1, d)  # (N_other*2K, d)
                if other_parts_flat.shape[0] > max_negatives_per_sample:
                    # 隨機採樣
                    perm = torch.randperm(other_parts_flat.shape[0])[:max_negatives_per_sample]
                    other_parts_flat = other_parts_flat[perm]
                
                all_parts.append(other_parts_flat)
        
        if len(all_parts) > 0:
            negatives_pool = torch.cat(all_parts, dim=0)  # (N_neg, d)
        else:
            negatives_pool = torch.empty(0, d, device=Z1t.device)
        
        return negatives_pool
    
    def _get_class_prototypes(self, targets: torch.Tensor) -> torch.Tensor:
        """獲取給定標籤的類別原型協方差"""
        B = targets.shape[0]
        proto_covs = torch.zeros(B, self.d, self.d, device=targets.device)
        
        for b, target in enumerate(targets):
            if self.class_counts[target] > 0:
                proto_covs[b] = self.class_prototypes[target]
            else:
                # 如果還沒有原型，初始化為單位矩陣
                proto_covs[b] = torch.eye(self.d, device=targets.device)
        
        return proto_covs
    
    def _update_class_prototypes(self, targets: torch.Tensor, 
                                Sigma1: torch.Tensor, Sigma2: torch.Tensor):
        """使用 EMA 更新類別原型協方差"""
        Sigma_bar = 0.5 * (Sigma1 + Sigma2)  # (B, d, d)
        
        for b, target in enumerate(targets):
            target = target.item()
            
            if self.class_counts[target] == 0:
                # 該類別的第一個樣本
                self.class_prototypes[target] = Sigma_bar[b].detach()
                self.class_counts[target] = 1
            else:
                # EMA 更新
                self.class_prototypes[target] = (
                    self.proto_momentum * self.class_prototypes[target] +
                    (1 - self.proto_momentum) * Sigma_bar[b].detach()
                )
                self.class_counts[target] += 1
    
    def set_epoch(self, epoch: int):
        """設置當前 epoch 用於熱身調度"""
        self.current_epoch = epoch
        
        # 熱身後切換到 Mahalanobis 距離
        if self.use_mahalanobis_warmup and epoch >= self.warmup_epochs:
            print(f"Epoch {epoch}: Switching to Mahalanobis distance for matching")
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """獲取優化信息和統計"""
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
            'class_prototypes_initialized': self.class_counts.sum().item() if self.use_class_proto else 0,
            'optimizations': {
                'small_params': f'K={self.K}<=4, d={self.d}<=64',
                'vectorized_pac': '一次白化向量化',
                'two_stage_negatives': f'兩段式篩選(M={self.top_m_candidates})',
                'frobenius_soc': 'Frobenius 距離 SoC',
                'limited_negatives_pool': '限制負對池大小'
            }
        }
