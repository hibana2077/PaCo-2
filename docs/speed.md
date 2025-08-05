哇！測試結果很有趣，讓我總結一下：

PaCo-2 模型速度測試結果
推理模式 (Inference)
Forward 時間: 22.59 ms
每樣本時間: 2.82 ms/sample
吞吐量: 354 samples/sec
訓練模式 (Training)
Forward 時間: 4,982.15 ms (~5秒)
Backward 時間: 8,284.00 ms (~8.3秒)
總時間: 13,266.15 ms (~13.3秒)
每樣本總時間: 1,658.27 ms/sample (~1.66秒)
關鍵發現
訓練 vs 推理: 訓練模式比推理模式慢了 220倍！
Backward vs Forward: Backward 比 Forward 慢 1.66倍
與你之前的 Loss 測試對比:
你測試的 Loss: ~1000 ms
實際訓練 Forward: ~5000 ms (包含 Loss 計算)
這很合理，因為訓練模式包含了複雜的 PaCo 計算
性能瓶頸分析
訓練模式這麼慢的原因：

Part sampling 和特徵提取
協方差計算 (Covariance computation)
匈牙利匹配 (Hungarian matching)
複雜的 Loss 計算 (PaC + SoC losses)
Mahalanobis 距離計算
這解釋了為什麼你的 loss 測試大約是 1000ms，而完整的訓練 forward 是 5000ms 左右。