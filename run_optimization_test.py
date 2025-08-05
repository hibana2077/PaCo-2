#!/usr/bin/env python3
"""
快速運行腳本：一鍵測試優化效果
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """運行命令並處理輸出"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"執行命令: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print(f"錯誤輸出: {result.stderr}")
        
        if result.returncode != 0:
            print(f"❌ 命令執行失敗，返回碼: {result.returncode}")
            return False
        else:
            print(f"✅ 命令執行成功")
            return True
    
    except Exception as e:
        print(f"❌ 執行命令時出錯: {e}")
        return False


def check_requirements():
    """檢查依賴是否安裝"""
    print("🔍 檢查依賴套件...")
    
    # 套件名稱映射：(導入名稱, 顯示名稱)
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('timm', 'timm'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('yaml', 'pyyaml'),  # PyYAML 的導入名稱是 yaml
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    
    for import_name, display_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} (缺失)")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n⚠️  缺少以下套件: {', '.join(missing_packages)}")
        print("請執行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依賴套件已安裝")
    return True


def main():
    """主函數"""
    print("🎯 PaCo-2 高優先級優化 - 快速測試")
    print("本腳本將依序執行以下測試：")
    print("1. 檢查依賴套件")
    print("2. 運行使用範例")
    print("3. 執行性能基準測試")
    
    # 1. 檢查依賴
    if not check_requirements():
        print("\n❌ 依賴檢查失敗，請先安裝所需套件")
        return
    
    # 2. 運行使用範例
    success = run_command(
        "python example_optimized_usage.py",
        "執行優化版使用範例"
    )
    
    if not success:
        print("❌ 使用範例執行失敗")
        return
    
    # 3. 運行性能測試（小規模測試）
    success = run_command(
        "python test_optimization.py --batch-size 8 --iterations 3",
        "執行性能基準測試（小規模）"
    )
    
    if not success:
        print("❌ 性能測試執行失敗")
        return
    
    print(f"\n{'='*60}")
    print("🎉 所有測試完成！")
    print("="*60)
    print("📊 如需詳細性能測試，請執行:")
    print("   python test_optimization.py --batch-size 16 --iterations 10")
    print()
    print("🚀 如需開始訓練，請執行:")
    print("   python train_optimized.py --config configs/optimized_high_priority.yaml")
    print()
    print("📖 詳細說明請參考: README_OPTIMIZATION.md")


if __name__ == '__main__':
    main()
