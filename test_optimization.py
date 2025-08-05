#!/usr/bin/env python3
"""
Optimization Performance Test Script: Measure and compare training speed before and after optimization
"""

import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# 添加 src 到路徑
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.paco_model import PaCoModel
from src.models.optimized_paco_model import OptimizedPaCoModel


class SpeedBenchmark:
    """Speed benchmark testing"""
    
    def __init__(self, device='cuda', batch_size=16, num_classes=1000):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        print(f"Using device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Number of classes: {num_classes}")
    
    def create_dummy_data(self):
        """Create dummy data"""
        x1 = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        x2 = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        targets = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)
        
        # Enable channels_last (if supported)
        try:
            x1 = x1.contiguous(memory_format=torch.channels_last)
            x2 = x2.contiguous(memory_format=torch.channels_last)
        except:
            pass
        
        return x1, x2, targets
    
    def benchmark_model(self, model, name, num_iterations=10, warmup_iterations=3):
        """Benchmark a single model"""
        print(f"\n{'='*50}")
        print(f"Testing model: {name}")
        print(f"{'='*50}")
        
        model = model.to(self.device)
        model.train()
        
        # Enable channels_last (if supported)
        try:
            model = model.to(memory_format=torch.channels_last)
        except:
            pass
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        # Timers
        forward_times = []
        backward_times = []
        total_times = []
        
        print(f"Warming up {warmup_iterations} iterations...")
        
        # Warmup
        for i in range(warmup_iterations):
            x1, x2, targets = self.create_dummy_data()
            
            optimizer.zero_grad()
            outputs = model(x1, x2, targets)
            loss = outputs['total']
            loss.backward()
            optimizer.step()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        print(f"Measuring performance for {num_iterations} iterations...")
        
        # Actual measurement
        for i in range(num_iterations):
            x1, x2, targets = self.create_dummy_data()
            
            # Measure forward pass time
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            optimizer.zero_grad()
            outputs = model(x1, x2, targets)
            loss = outputs['total']
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            forward_time = time.time() - start_time
            
            # Measure backward pass time
            start_time = time.time()
            
            loss.backward()
            optimizer.step()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            backward_time = time.time() - start_time
            total_time = forward_time + backward_time
            
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            total_times.append(total_time)
            
            # Deconstruct outputs to check detailed information
            if i == 0:  # Only print detailed info for first iteration
                print(f"  First iteration loss details:")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:  # Only scalars can be converted
                            print(f"    {key}: {value.item():.6f}")
                        else:
                            print(f"    {key}: shape {value.shape}")
                    else:
                        print(f"    {key}: {value}")
        
        # Calculate statistics
        avg_forward = np.mean(forward_times)
        avg_backward = np.mean(backward_times)
        avg_total = np.mean(total_times)
        
        std_forward = np.std(forward_times)
        std_backward = np.std(backward_times)
        std_total = np.std(total_times)
        
        print(f"\nPerformance statistics ({num_iterations} iterations):")
        print(f"  Forward pass:  {avg_forward:.4f} ± {std_forward:.4f} seconds")
        print(f"  Backward pass: {avg_backward:.4f} ± {std_backward:.4f} seconds")
        print(f"  Total:         {avg_total:.4f} ± {std_total:.4f} seconds")
        print(f"  Throughput:    {self.batch_size / avg_total:.2f} samples/sec")
        
        return {
            'forward_time': avg_forward,
            'backward_time': avg_backward,
            'total_time': avg_total,
            'forward_std': std_forward,
            'backward_std': std_backward,
            'total_std': std_total,
            'throughput': self.batch_size / avg_total
        }
    
    def compare_models(self):
        """Compare original and optimized models"""
        print("PaCo-2 Optimization Performance Comparison Test")
        print("=" * 60)
        
        # Create original model (larger parameters)
        print("Creating original model...")
        original_model = PaCoModel(
            backbone_name='resnet50',
            num_classes=self.num_classes,
            pretrained=False,  # No pretrained weights needed for testing
            K=6,               # Larger number of parts
            d=128,             # Larger feature dimension
            metric="log-euclidean"  # More expensive distance metric
        )
        
        # Create optimized model
        print("Creating optimized model...")
        optimized_model = OptimizedPaCoModel(
            backbone_name='resnet50',
            num_classes=self.num_classes,
            pretrained=False,
            K=4,               # Forced small parameters
            d=64,              # Forced small parameters
            top_m_candidates=32  # Two-stage filtering
        )
        
        # Test original model
        original_stats = self.benchmark_model(original_model, "Original PaCoModel", num_iterations=5)
        
        # Clean GPU memory
        del original_model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Test optimized model
        optimized_stats = self.benchmark_model(optimized_model, "Optimized OptimizedPaCoModel", num_iterations=5)
        
        # Calculate improvement ratios
        self.print_comparison(original_stats, optimized_stats)
    
    def print_comparison(self, original_stats, optimized_stats):
        """Print comparison results"""
        print(f"\n{'='*60}")
        print("Optimization Results Summary")
        print(f"{'='*60}")
        
        # Calculate speedup ratios
        forward_speedup = original_stats['forward_time'] / optimized_stats['forward_time']
        backward_speedup = original_stats['backward_time'] / optimized_stats['backward_time']
        total_speedup = original_stats['total_time'] / optimized_stats['total_time']
        throughput_improvement = optimized_stats['throughput'] / original_stats['throughput']
        
        print(f"Forward speedup:      {forward_speedup:.2f}x  ({original_stats['forward_time']:.4f}s -> {optimized_stats['forward_time']:.4f}s)")
        print(f"Backward speedup:     {backward_speedup:.2f}x  ({original_stats['backward_time']:.4f}s -> {optimized_stats['backward_time']:.4f}s)")
        print(f"Total speedup:        {total_speedup:.2f}x  ({original_stats['total_time']:.4f}s -> {optimized_stats['total_time']:.4f}s)")
        print(f"Throughput improvement: {throughput_improvement:.2f}x  ({original_stats['throughput']:.2f} -> {optimized_stats['throughput']:.2f} samples/sec)")
        
        print(f"\nHigh Priority Optimizations:")
        print(f"[OK] K=6->4, d=128->64: Reduce computation complexity O(K*d^2)")
        print(f"[OK] Vectorized whitening: Avoid repeated Cholesky decomposition")
        print(f"[OK] Two-stage negative sampling: Reduce Mahalanobis computation")
        print(f"[OK] Fixed Frobenius SoC: Avoid eigen decomposition and logdet")
        print(f"[OK] Limited negatives pool: Control memory and computation")
        
        if total_speedup >= 2.0:
            print(f"\n[EXCELLENT] Significant optimization effect! Total speedup {total_speedup:.1f}x")
        elif total_speedup >= 1.5:
            print(f"\n[GOOD] Optimization effective! Total speedup {total_speedup:.1f}x")
        else:
            print(f"\n[LIMITED] Optimization effect limited, total speedup {total_speedup:.1f}x")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PaCo-2 Optimization Performance Test')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--iterations', type=int, default=5, help='Test iterations')
    
    args = parser.parse_args()
    
    # Create benchmark tester
    benchmark = SpeedBenchmark(
        device=args.device,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    # Run comparison test
    benchmark.compare_models()


if __name__ == '__main__':
    main()
