"""
PaCo-2 Utilities for Part Sampling, Hungarian Matching, and Covariance Computations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Tuple, Optional, List


class PartSampler(nn.Module):
    """
    Extract K salient parts from feature maps using top-K peak detection
    Following the pseudo code in docs/pesudo_code.md
    """
    
    def __init__(self, K: int = 4, r: int = 5, nms_threshold: float = 0.1):
        super().__init__()
        self.K = K
        self.r = r
        self.nms_threshold = nms_threshold
        
        # Simple 1x1 conv for spatial attention
        self.spatial_conv = nn.Conv2d(1, 1, 1, bias=False)
        nn.init.ones_(self.spatial_conv.weight)
        
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract parts from feature map F
        
        Args:
            feat: Feature map of shape (B, C, H, W)
            
        Returns:
            Z: Part features (B, K, C)
            peaks: Peak locations (B, K, 2) - (h, w) coordinates
        """
        B, C, H, W = feat.shape
        
        # Channel attention: Global Average Pooling + softmax
        a = feat.mean(dim=[2, 3])  # (B, C)
        a = F.softmax(a, dim=1)  # (B, C)
        
        # Spatial attention: mean over channels + 1x1 conv + softmax
        S = feat.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        S = self.spatial_conv(S)  # (B, 1, H, W)
        S = F.softmax(S.view(B, -1), dim=1).view(B, 1, H, W)
        
        # Combine for saliency map M
        a_expanded = a.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        weighted_F = feat * a_expanded  # (B, C, H, W)
        M = (weighted_F.sum(dim=1, keepdim=True) * S).squeeze(1)  # (B, H, W)
        
        # Extract K peaks with NMS
        peaks = self._extract_peaks_with_nms(M)  # (B, K, 2)
        
        # Extract part features using r×r windows
        Z = self._extract_part_features(feat, peaks)  # (B, K, C)
        
        return Z, peaks
    
    def _extract_peaks_with_nms(self, M: torch.Tensor) -> torch.Tensor:
        """
        Extract K peaks from saliency map with NMS
        
        Args:
            M: Saliency map (B, H, W)
            
        Returns:
            peaks: Peak coordinates (B, K, 2)
        """
        B, H, W = M.shape
        peaks = torch.zeros(B, self.K, 2, device=M.device, dtype=torch.long)
        
        for b in range(B):
            saliency = M[b].detach().cpu().numpy()
            peak_coords = []
            
            # Simple peak detection with NMS
            for _ in range(self.K):
                # Find maximum
                flat_idx = np.argmax(saliency)
                h, w = np.unravel_index(flat_idx, saliency.shape)
                peak_coords.append([h, w])
                
                # Apply NMS: suppress neighborhood
                h_start = max(0, h - int(self.nms_threshold * H))
                h_end = min(H, h + int(self.nms_threshold * H) + 1)
                w_start = max(0, w - int(self.nms_threshold * W))
                w_end = min(W, w + int(self.nms_threshold * W) + 1)
                
                saliency[h_start:h_end, w_start:w_end] = -np.inf
            
            peaks[b] = torch.tensor(peak_coords, device=M.device)
            
        return peaks
    
    def _extract_part_features(self, F: torch.Tensor, peaks: torch.Tensor) -> torch.Tensor:
        """
        Extract part features using r×r windows around peaks
        
        Args:
            F: Feature map (B, C, H, W)
            peaks: Peak coordinates (B, K, 2)
            
        Returns:
            Z: Part features (B, K, C)
        """
        B, C, H, W = F.shape
        Z = torch.zeros(B, self.K, C, device=F.device)
        
        for b in range(B):
            for k in range(self.K):
                h, w = peaks[b, k]
                
                # Define window bounds
                h_start = max(0, h - self.r // 2)
                h_end = min(H, h + self.r // 2 + 1)
                w_start = max(0, w - self.r // 2)
                w_end = min(W, w + self.r // 2 + 1)
                
                # Average pool the window
                window = F[b, :, h_start:h_end, w_start:w_end]
                Z[b, k] = window.mean(dim=[1, 2])
        
        return Z


class HungarianMatcher(nn.Module):
    """
    Hungarian matching for part correspondence between two views
    """
    def __init__(self, use_cosine: bool = True):
        super().__init__()
        self.use_cosine = use_cosine

    def forward(self, Z1: torch.Tensor, Z2: torch.Tensor, 
                Sigma_plus: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Match parts between two views using Hungarian algorithm
        
        Args:
            Z1, Z2: Part features (B, K, d)
            Sigma_plus: Covariance for Mahalanobis distance (B, d, d), optional
            
        Returns:
            matching: Permutation indices (B, K) for Z2
        """
        B, K, d = Z1.shape
        matching = torch.zeros(B, K, dtype=torch.long, device=Z1.device)
        
        for b in range(B):
            if self.use_cosine or Sigma_plus is None:
                # Cosine distance cost matrix
                z1_norm = F.normalize(Z1[b], p=2, dim=1)
                z2_norm = F.normalize(Z2[b], p=2, dim=1)
                cost = 1 - torch.mm(z1_norm, z2_norm.t())
            else:
                # Mahalanobis distance cost matrix
                cost = torch.zeros(K, K, device=Z1.device)
                for i in range(K):
                    for j in range(K):
                        diff = Z1[b, i] - Z2[b, j]
                        # Use Cholesky for numerical stability
                        L = torch.linalg.cholesky(Sigma_plus[b])
                        y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
                        cost[i, j] = torch.norm(y, p=2)
            
            # Solve assignment problem
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            matching[b] = torch.tensor(col_ind, device=Z1.device)
        
        return matching


class CovarUtils:
    """
    Utilities for covariance computation and SPD manifold distances
    """
    
    @staticmethod
    def compute_covariance(Zt: torch.Tensor, epsilon: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sample covariance with diagonal loading
        
        Args:
            Zt: Part features (B, K, d)
            epsilon: Diagonal loading factor
            
        Returns:
            Sigma: Covariance matrices (B, d, d)
            mu: Mean vectors (B, d)
        """
        B, K, d = Zt.shape
        
        # Compute mean
        mu = Zt.mean(dim=1)  # (B, d)
        
        # Center the data
        Zh = Zt - mu.unsqueeze(1)  # (B, K, d)
        
        # Compute covariance
        Sigma = torch.bmm(Zh.transpose(-1, -2), Zh) / max(1, K - 1)  # (B, d, d)
        
        # Add diagonal loading for numerical stability
        I = torch.eye(d, device=Zt.device).unsqueeze(0).expand(B, -1, -1)
        Sigma = Sigma + epsilon * I
        
        return Sigma, mu
    
    @staticmethod
    def spd_distance(Sigma1: torch.Tensor, Sigma2: torch.Tensor, 
                     metric: str = "fro") -> torch.Tensor:
        """
        Compute distance between SPD matrices
        
        Args:
            Sigma1, Sigma2: Covariance matrices (B, d, d)
            metric: Distance metric - "fro", "log-euclidean", or "stein"
            
        Returns:
            distances: SPD distances (B,)
        """
        if metric == "fro":
            # Frobenius norm
            diff = Sigma1 - Sigma2
            return torch.norm(diff.view(diff.shape[0], -1), p='fro', dim=1)
        
        elif metric == "log-euclidean":
            # Log-Euclidean metric
            try:
                L1 = torch.logm(Sigma1)
                L2 = torch.logm(Sigma2)
                diff = L1 - L2
                return torch.norm(diff.view(diff.shape[0], -1), p='fro', dim=1)
            except:
                # Fallback to Frobenius if logm fails
                print("Warning: logm failed, falling back to Frobenius")
                return CovarUtils.spd_distance(Sigma1, Sigma2, "fro")
        
        elif metric == "stein":
            # Stein/Jensen-Bregman LogDet distance
            try:
                Sigma_m = 0.5 * (Sigma1 + Sigma2)
                logdet_m = torch.logdet(Sigma_m)
                logdet_1 = torch.logdet(Sigma1)
                logdet_2 = torch.logdet(Sigma2)
                
                v = logdet_m - 0.5 * (logdet_1 + logdet_2)
                return torch.clamp(v, min=0.0)
            except:
                # Fallback to Frobenius if logdet fails
                print("Warning: logdet failed, falling back to Frobenius")
                return CovarUtils.spd_distance(Sigma1, Sigma2, "fro")
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def mahalanobis_distance(u: torch.Tensor, v: torch.Tensor, 
                           Sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance using Cholesky decomposition
        
        Args:
            u, v: Vectors (B, d) or (d,)
            Sigma: Covariance matrix (B, d, d) or (d, d)
            
        Returns:
            distances: Mahalanobis distances (B,) or scalar
        """
        diff = u - v
        
        if len(Sigma.shape) == 3:  # Batched
            B = Sigma.shape[0]
            distances = torch.zeros(B, device=u.device)
            
            for b in range(B):
                try:
                    L = torch.linalg.cholesky(Sigma[b])
                    y = torch.linalg.solve_triangular(L, diff[b], upper=False)
                    distances[b] = torch.norm(y, p=2)
                except:
                    # Fallback to regularized version
                    reg_Sigma = Sigma[b] + 1e-6 * torch.eye(Sigma.shape[-1], device=Sigma.device)
                    L = torch.linalg.cholesky(reg_Sigma)
                    y = torch.linalg.solve_triangular(L, diff[b], upper=False)
                    distances[b] = torch.norm(y, p=2)
            
            return distances
        else:  # Single matrix
            try:
                L = torch.linalg.cholesky(Sigma)
                y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
                return torch.norm(y, p=2)
            except:
                # Fallback to regularized version
                reg_Sigma = Sigma + 1e-6 * torch.eye(Sigma.shape[-1], device=Sigma.device)
                L = torch.linalg.cholesky(reg_Sigma)
                y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
                return torch.norm(y, p=2)
