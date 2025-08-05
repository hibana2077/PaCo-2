#!/usr/bin/env python3
"""
優化版 PaCo-2 訓練腳本：高優先級優化實作
整合所有高優先級優化項目，專注於訓練速度提升
"""

import os
import sys
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import wandb
from pathlib import Path

# 添加 src 到路徑
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.optimized_paco_model import OptimizedPaCoModel
from src.data_utils import get_dataloaders
from src.train_utils import AverageMeter, accuracy, save_checkpoint


class OptimizedTrainer:
    """優化版 PaCo-2 訓練器"""
    
    def __init__(self, config_path: str):
        """初始化訓練器"""
        # 載入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 系統級優化設置
        self._setup_system_optimizations()
        
        # 建立模型
        self.model = self._build_model()
        
        # 建立資料載入器
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # 建立優化器和調度器
        self.optimizer, self.scheduler = self._build_optimizer()
        
        # 混合精度訓練
        self.scaler = torch.amp.GradScaler('cuda') if self.config['training']['mixed_precision'] else None
        
        # 統計指標
        self.best_acc1 = 0
        self.train_time_meter = AverageMeter('TrainTime', ':6.3f')
        self.loss_meter = AverageMeter('Loss', ':.4e')
        self.pac_meter = AverageMeter('PaC', ':.4e')
        self.soc_meter = AverageMeter('SoC', ':.4e')
        
        # 實驗追蹤
        if wandb.run is None:
            # 設置 WANDB 為離線模式，避免互動式選擇
            os.environ['WANDB_MODE'] = 'offline'
            wandb.init(
                project="paco2-optimization",
                name=self.config['experiment']['name'],
                tags=self.config['experiment']['tags'],
                config=self.config
            )
    
    def _setup_system_optimizations(self):
        """設置系統級優化"""
        # 啟用 channels_last 記憶體格式
        if self.config['training'].get('channels_last', False):
            print("Enabling channels_last memory format")
            torch.backends.cudnn.benchmark = True
        
        # 其他 CUDA 優化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
    
    def _build_model(self):
        """建立優化版模型"""
        model_config = self.config['model']
        
        # 強制高優先級優化參數
        if model_config['K'] > 4:
            print(f"Warning: Forcing K from {model_config['K']} to 4 for optimization")
            model_config['K'] = 4
        if model_config['d'] > 64:
            print(f"Warning: Forcing d from {model_config['d']} to 64 for optimization")
            model_config['d'] = 64
        
        model = OptimizedPaCoModel(
            backbone_name=model_config['backbone_name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained'],
            K=model_config['K'],
            r=model_config['r'],
            d=model_config['d'],
            lambda_pac=model_config['lambda_pac'],
            eta_soc=model_config['eta_soc'],
            alpha=model_config['alpha'],
            beta=model_config['beta'],
            gamma=model_config['gamma'],
            top_m_candidates=model_config['top_m_candidates'],
            epsilon=model_config['epsilon'],
            tau=model_config['tau'],
            use_mahalanobis_warmup=model_config['use_mahalanobis_warmup'],
            warmup_epochs=model_config['warmup_epochs'],
            use_weighted_ce=model_config['use_weighted_ce'],
            use_semi_hard=model_config['use_semi_hard'],
            use_class_proto=model_config['use_class_proto'],
            proto_momentum=model_config['proto_momentum']
        )
        
        model = model.to(self.device)
        
        # 啟用 channels_last
        if self.config['training'].get('channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        
        # torch.compile 優化（PyTorch 2.x）
        if self.config['training'].get('compile', False):
            try:
                print("Compiling model with torch.compile...")
                model = torch.compile(model, dynamic=True, disable=False)
                print("Model compilation successful")
            except Exception as e:
                print(f"Model compilation failed: {e}, continuing without compilation")
                # Disable compilation in config to avoid repeated attempts
                self.config['training']['compile'] = False
        
        # 列印優化信息
        if self.config['debug']['print_model_info']:
            opt_info = model.get_optimization_info()
            print("\n=== 優化版模型信息 ===")
            for key, value in opt_info.items():
                if key != 'optimizations':
                    print(f"{key}: {value}")
            print("\n=== 啟用的優化 ===")
            for opt_name, opt_desc in opt_info['optimizations'].items():
                print(f"✓ {opt_name}: {opt_desc}")
            print("=" * 50)
        
        return model
    
    def _build_dataloaders(self):
        """建立資料載入器"""
        dataset_config = self.config['dataset']
        training_config = self.config['training']
        
        train_loader, val_loader = get_dataloaders(
            dataset_name=dataset_config['name'],
            data_root=dataset_config['root'],
            batch_size=training_config['batch_size'],
            num_workers=dataset_config['num_workers'],
            pin_memory=dataset_config['pin_memory'],
            augmentation_config=training_config['augmentation']
        )
        
        print(f"Dataset: {dataset_config['name']}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"Batch size: {training_config['batch_size']}")
        
        return train_loader, val_loader
    
    def _build_optimizer(self):
        """建立優化器和調度器"""
        train_config = self.config['training']
        
        # 優化器
        if train_config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                momentum=0.9,
                weight_decay=train_config['weight_decay']
            )
        
        # 調度器
        if train_config['scheduler'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_config['epochs'],
                eta_min=train_config['min_lr']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        
        return optimizer, scheduler
    
    def train_epoch(self, epoch: int):
        """訓練一個 epoch"""
        self.model.train()
        self.model.set_epoch(epoch)  # 設置 epoch 用於熱身
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        pac_losses = AverageMeter('PaC', ':.4e')
        soc_losses = AverageMeter('SoC', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        end = time.time()
        
        for i, (views, targets) in enumerate(self.train_loader):
            # 測量資料載入時間
            data_time.update(time.time() - end)
            
            # Extract two views
            x1, x2 = views
            
            # 移到設備
            x1 = x1.to(self.device, non_blocking=True, memory_format=torch.channels_last if self.config['training'].get('channels_last') else torch.contiguous_format)
            x2 = x2.to(self.device, non_blocking=True, memory_format=torch.channels_last if self.config['training'].get('channels_last') else torch.contiguous_format)
            targets = targets.to(self.device, non_blocking=True)
            
            batch_size = x1.shape[0]
            
            # 前向傳播
            with torch.amp.autocast(device_type='cuda') if self.scaler else torch.amp.autocast(device_type='cuda', enabled=False):
                try:
                    outputs = self.model(x1, x2, targets)
                    
                    logits = outputs['logits']
                    total_loss = outputs['total']
                    pac_loss = outputs['pac']
                    soc_loss = outputs['soc']
                except Exception as e:
                    if "not positive-definite" in str(e) or "cholesky" in str(e).lower():
                        print(f"Warning: Numerical instability detected at batch {i}, skipping batch")
                        print(f"Error: {e}")
                        continue
                    else:
                        # Re-raise other exceptions
                        raise e
            
            # 反向傳播
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # 計算準確率
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            
            # 更新統計
            losses.update(total_loss.item(), batch_size)
            pac_losses.update(pac_loss.item(), batch_size)
            soc_losses.update(soc_loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            
            # 測量時間
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 記錄
            if i % self.config['training']['log_frequency'] == 0:
                print(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'PaC {pac_losses.val:.4f} ({pac_losses.avg:.4f})\t'
                      f'SoC {soc_losses.val:.4f} ({soc_losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
                
                # WandB 記錄
                wandb.log({
                    'train/loss': losses.val,
                    'train/pac_loss': pac_losses.val,
                    'train/soc_loss': soc_losses.val,
                    'train/acc1': top1.val,
                    'train/batch_time': batch_time.val,
                    'train/data_time': data_time.val,
                    'epoch': epoch,
                    'step': epoch * len(self.train_loader) + i
                })
        
        return losses.avg, pac_losses.avg, soc_losses.avg, top1.avg
    
    def validate(self, epoch: int):
        """驗證"""
        self.model.eval()
        
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        with torch.no_grad():
            end = time.time()
            for i, (images, targets) in enumerate(self.val_loader):  # 驗證時只需要一個視圖
                # Handle single view for validation
                if isinstance(images, tuple):
                    x1 = images[0]  # Use first view for validation
                else:
                    x1 = images
                x1 = x1.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # 前向傳播
                with torch.amp.autocast(device_type='cuda') if self.scaler else torch.amp.autocast(device_type='cuda', enabled=False):
                    outputs = self.model(x1)  # 推理模式
                    logits = outputs['logits']
                    loss = nn.CrossEntropyLoss()(logits, targets)
                
                # 測量準確率
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                
                losses.update(loss.item(), x1.shape[0])
                top1.update(acc1[0], x1.shape[0])
                top5.update(acc5[0], x1.shape[0])
                
                # 測量時間
                batch_time.update(time.time() - end)
                end = time.time()
        
        print(f'Val: [{epoch}] Time {batch_time.avg:.3f}\t'
              f'Loss {losses.avg:.4f}\t'
              f'Acc@1 {top1.avg:.3f}\t'
              f'Acc@5 {top5.avg:.3f}')
        
        # WandB 記錄
        wandb.log({
            'val/loss': losses.avg,
            'val/acc1': top1.avg,
            'val/acc5': top5.avg,
            'val/batch_time': batch_time.avg,
            'epoch': epoch
        })
        
        return top1.avg
    
    def train(self):
        """主訓練循環"""
        print("開始優化版 PaCo-2 訓練...")
        print(f"總 epochs: {self.config['training']['epochs']}")
        
        for epoch in range(self.config['training']['epochs']):
            # 訓練
            train_loss, train_pac, train_soc, train_acc = self.train_epoch(epoch)
            
            # 驗證
            if epoch % self.config['training']['validate_frequency'] == 0:
                val_acc = self.validate(epoch)
                
                # 調整學習率 (移到驗證後)
                self.scheduler.step()
                
                # 記錄最佳結果
                is_best = val_acc > self.best_acc1
                self.best_acc1 = max(val_acc, self.best_acc1)
                
                # 保存檢查點
                if epoch % self.config['training']['save_frequency'] == 0 or is_best:
                    checkpoint_path = Path(f'checkpoint_epoch_{epoch}.pth.tar')
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_acc1': self.best_acc1,
                        'config': self.config,
                        'metrics': {
                            'val_acc1': val_acc,
                            'train_loss': train_loss,
                            'train_pac': train_pac,
                            'train_soc': train_soc,
                            'train_acc': train_acc
                        }
                    }
                    torch.save(checkpoint, checkpoint_path)
                    
                    if is_best:
                        best_path = Path('best_model.pth')
                        torch.save(checkpoint, best_path)
                        print(f"New best model saved with accuracy: {val_acc:.3f}%")
                
                # WandB 記錄
                wandb.log({
                    'epoch_summary/train_loss': train_loss,
                    'epoch_summary/train_pac_loss': train_pac,
                    'epoch_summary/train_soc_loss': train_soc,
                    'epoch_summary/train_acc': train_acc,
                    'epoch_summary/val_acc': val_acc,
                    'epoch_summary/best_acc': self.best_acc1,
                    'epoch_summary/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        print(f"訓練完成！最佳驗證準確率: {self.best_acc1:.3f}%")


def main():
    parser = argparse.ArgumentParser(description='優化版 PaCo-2 訓練')
    parser.add_argument('--config', default='configs/optimized_high_priority.yaml',
                       help='配置文件路徑')
    parser.add_argument('--resume', default='', help='恢復訓練檢查點路徑')
    
    args = parser.parse_args()
    
    # 創建訓練器並開始訓練
    trainer = OptimizedTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
