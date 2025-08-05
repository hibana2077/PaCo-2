"""
Optimized PaCo-2 Loss Functions: High Priority Optimizations Implementation
Implements vectorized whitening, two-stage negative sampling, and other optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from .utils import CovarUtils


class OptimizedPaCLoss(nn.Module):
    """
    優化版 Part-aware Contrast Loss (PaC)
    
    高優先級優化：
    1. 一次白化後向量化計算所有 Mahalanobis 距離
    2. 兩段式負對篩選：餘弦預篩 -> Mahalanobis 精算
    """
    
    def __init__(self, alpha: float = 0.2, use_semi_hard: bool = True, 
                 top_m_candidates: int = 32):
        super().__init__()
        self.alpha = alpha
        self.use_semi_hard = use_semi_hard
        self.top_m_candidates = top_m_candidates  # 兩段式篩選的候選數量
    
    def forward(self, Z1t: torch.Tensor, Z2t: torch.Tensor, 
                matching: torch.Tensor, negatives_pool: torch.Tensor,
                Sigma_plus: torch.Tensor) -> torch.Tensor:
        """
        優化版 PaC loss 計算
        
        Args:
            Z1t, Z2t: Part features (B, K, d)
            matching: Part matching indices (B, K)
            negatives_pool: Negative part features from other samples (N_neg, d)
            Sigma_plus: Shared covariance (B, d, d)
            
        Returns:
            loss: PaC loss value
        """
        if negatives_pool.shape[0] == 0:
            return torch.tensor(0.0, device=Z1t.device, requires_grad=True)
        
        B, K, d = Z1t.shape
        N_neg = negatives_pool.shape[0]
        
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            # 一次性白化：對當前樣本的 Sigma_plus 做 Cholesky 分解
            # 增加基本正則化以確保數值穩定性
            epsilon = 1e-5  # 使用固定的正則化參數
            base_reg = max(1e-4, epsilon * 10)  # 使用更強的基本正則化
            reg_Sigma = Sigma_plus[b] + base_reg * torch.eye(d, device=Sigma_plus.device)
            
            # 使用更穩健的 Cholesky 分解
            try:
                L = torch.linalg.cholesky(reg_Sigma)  # (d, d)
            except RuntimeError:
                # 如果仍然失敗，使用更強的正則化
                strong_reg = base_reg * 100  # 1e-2 或更大
                reg_Sigma = Sigma_plus[b] + strong_reg * torch.eye(d, device=Sigma_plus.device)
                try:
                    L = torch.linalg.cholesky(reg_Sigma)
                except RuntimeError:
                    # 最後備用方案：使用對角陣近似
                    diag_values = torch.diag(Sigma_plus[b]) + strong_reg
                    L = torch.diag(torch.sqrt(diag_values))
                    print(f"Warning: Using diagonal approximation for batch {b}")
            
            # 準備所有需要白化的向量
            batch_vectors = []  # 收集該 batch 的所有向量
            vector_info = []    # 記錄每個向量的類型和索引
            
            for k in range(K):
                # 正對
                p = Z1t[b, k]  # (d,)
                q = Z2t[b, matching[b, k]]  # (d,)
                
                batch_vectors.extend([p, q])
                vector_info.extend([('pos', k, 0), ('pos', k, 1)])
                
                # 負對候選（先加入所有負對）
                for neg_idx in range(N_neg):
                    neg = negatives_pool[neg_idx]  # (d,)
                    batch_vectors.append(neg)
                    vector_info.append(('neg', k, neg_idx))
            
            if len(batch_vectors) == 0:
                continue
                
            # 一次性白化所有向量
            all_vectors = torch.stack(batch_vectors)  # (N_vectors, d)
            try:
                # 解線性方程組 L @ Y = all_vectors.T，得到白化後的向量
                whitened_vectors = torch.linalg.solve_triangular(
                    L, all_vectors.T, upper=False
                ).T  # (N_vectors, d)
            except:
                # 備用方案：使用 torch.solve（較慢但更穩定）
                whitened_vectors = torch.linalg.solve(L, all_vectors.T).T
            
            # 重新組織白化後的向量
            vector_idx = 0
            for k in range(K):
                # 獲取正對的白化向量
                p_white = whitened_vectors[vector_idx]      # 正對 p
                q_white = whitened_vectors[vector_idx + 1]  # 正對 q
                vector_idx += 2
                
                # 計算正對距離（白化後就是歐氏距離）
                d_pos = torch.norm(p_white - q_white, p=2)
                
                # 獲取所有負對的白化向量
                neg_whites = []
                neg_originals = []
                
                for neg_idx in range(N_neg):
                    neg_white = whitened_vectors[vector_idx]
                    neg_whites.append(neg_white)
                    neg_originals.append(negatives_pool[neg_idx])
                    vector_idx += 1
                
                if len(neg_whites) == 0:
                    continue
                
                # 兩段式負對篩選
                if len(neg_whites) > self.top_m_candidates:
                    # 第一階段：使用餘弦距離快速預篩選 Top-M 候選
                    p_norm = F.normalize(Z1t[b, k].unsqueeze(0), p=2, dim=1)  # (1, d)
                    neg_stack = torch.stack(neg_originals)  # (N_neg, d)
                    neg_norm = F.normalize(neg_stack, p=2, dim=1)  # (N_neg, d)
                    
                    cosine_sim = torch.mm(p_norm, neg_norm.T).squeeze(0)  # (N_neg,)
                    cosine_dist = 1 - cosine_sim  # 餘弦距離
                    
                    # 選擇 Top-M 最近的候選（餘弦距離最小）
                    _, top_indices = torch.topk(cosine_dist, k=min(self.top_m_candidates, len(neg_whites)), largest=False)
                    
                    # 只計算被選中候選的 Mahalanobis 距離
                    selected_neg_whites = [neg_whites[i] for i in top_indices]
                else:
                    # 負對數量少，直接使用全部
                    selected_neg_whites = neg_whites
                
                # 第二階段：計算 Mahalanobis 距離（已白化，直接算歐氏距離）
                d_negs = []
                for neg_white in selected_neg_whites:
                    d_neg = torch.norm(p_white - neg_white, p=2)
                    d_negs.append(d_neg)
                
                if len(d_negs) == 0:
                    continue
                
                d_negs = torch.stack(d_negs)
                
                # 選擇負對策略
                if self.use_semi_hard:
                    # 半難負對：d_neg > d_pos 且最小的距離
                    valid_negs = d_negs[d_negs > d_pos]
                    if len(valid_negs) > 0:
                        d_neg = valid_negs.min()
                    else:
                        d_neg = d_negs.min()  # 最難負對
                else:
                    # 最難負對
                    d_neg = d_negs.min()
                
                # 計算 triplet loss
                loss_k = torch.clamp(d_pos - d_neg + self.alpha, min=0.0)
                total_loss += loss_k
                count += 1
        
        return total_loss / max(1, count)


class OptimizedSoCLoss(nn.Module):
    """
    優化版 Second-order Consistency Loss (SoC)
    
    高優先級優化：
    - 固定使用 Frobenius 距離（最省計算）
    """
    
    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta
        # 固定使用 Frobenius 距離
        self.metric = "fro"
    
    def forward(self, Sigma1: torch.Tensor, Sigma2: torch.Tensor,
                Sigma_proto: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        計算優化版 SoC loss
        
        Args:
            Sigma1, Sigma2: Covariance matrices from two views (B, d, d)
            Sigma_proto: Prototype covariances per class (B, d, d), optional
            
        Returns:
            loss: SoC loss value
        """
        # 主項：兩視圖間的一致性，使用 Frobenius 距離
        d12 = self._frobenius_distance(Sigma1, Sigma2)
        loss = (d12 ** 2).mean()
        
        # 可選的原型正則化
        if Sigma_proto is not None and self.beta > 0:
            Sigma_bar = 0.5 * (Sigma1 + Sigma2)
            d_proto = self._frobenius_distance(Sigma_bar, Sigma_proto)
            loss = loss + self.beta * (d_proto ** 2).mean()
        
        return loss
    
    def _frobenius_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """快速計算 Frobenius 距離"""
        diff = A - B
        return torch.norm(diff.view(diff.shape[0], -1), p='fro', dim=1)


class OptimizedPaCoLoss(nn.Module):
    """
    優化版組合 PaCo-2 Loss: CE + 優化PaC + 優化SoC
    
    高優先級優化整合：
    1. 使用優化版 PaC（一次白化 + 兩段式篩選）
    2. 使用優化版 SoC（固定 Frobenius 距離）
    3. 強制小維度 d=64, K=4
    """
    
    def __init__(self, 
                 lambda_pac: float = 1.0,
                 eta_soc: float = 0.1,
                 alpha: float = 0.2,
                 beta: float = 0.05,
                 gamma: float = 0.1,
                 use_weighted_ce: bool = True,
                 use_semi_hard: bool = True,
                 top_m_candidates: int = 32):
        super().__init__()
        
        self.lambda_pac = lambda_pac
        self.eta_soc = eta_soc
        
        # 元件損失函數
        if use_weighted_ce:
            from .losses import WeightedCELoss
            self.ce_loss = WeightedCELoss(gamma=gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # 使用優化版 PaC 和 SoC
        self.pac_loss = OptimizedPaCLoss(
            alpha=alpha, 
            use_semi_hard=use_semi_hard,
            top_m_candidates=top_m_candidates
        )
        self.soc_loss = OptimizedSoCLoss(beta=beta)
        
        self.use_weighted_ce = use_weighted_ce
        
        print(f"OptimizedPaCoLoss initialized:")
        print(f"  - PaC: Vectorized whitening + Two-stage negative sampling (M={top_m_candidates})")
        print(f"  - SoC: Fixed Frobenius distance")
        print(f"  - Weights: lambda_pac={lambda_pac}, eta_soc={eta_soc}")
    
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
        計算優化版組合 PaCo loss
        
        Returns:
            Dictionary containing individual and total losses
        """
        # CE Loss
        if self.use_weighted_ce and saliency_map is not None and peaks is not None:
            L_ce = self.ce_loss(logits, targets, saliency_map, peaks)
        else:
            if hasattr(self.ce_loss, 'forward') and len(self.ce_loss.forward.__code__.co_varnames) > 3:
                L_ce = self.ce_loss(logits, targets)
            else:
                L_ce = self.ce_loss(logits, targets)
        
        # 優化版 PaC Loss
        L_pac = self.pac_loss(Z1t, Z2t, matching, negatives_pool, Sigma_plus)
        
        # 優化版 SoC Loss  
        L_soc = self.soc_loss(Sigma1, Sigma2, Sigma_proto)
        
        # 總損失
        L_total = L_ce + self.lambda_pac * L_pac + self.eta_soc * L_soc
        
        return {
            'total': L_total,
            'ce': L_ce,
            'pac': L_pac,
            'soc': L_soc
        }
