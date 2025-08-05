## （二）要跑的實驗與要收集的數據

### 1. 資料集與分割

* **UFG 基準**：Cotton80, SoyLocal, SoyGene, SoyAgeing, SoyGlobal（沿用既有 train/val/test split）。
* **資料前處理**：長邊縮放、短邊對齊、RandAug（輕量）；Valid/Test 僅中心裁切。
* **重現性**：固定 seed（e.g., 3407）、3 次獨立重複取平均；記錄 splits 與 seeds。

---

### 2. Backbone 與設定（timm）

* **核心**：ResNet-50、ConvNeXt-T、EfficientNet-B3。
* **擴充（選做）**：ResNet-101、RegNetY-8GF、MobileNetV3-L。
* **訓練**：AdamW / SGD+Nesterov、Cosine LR、Warmup 5 epochs、Label smoothing 0.1。
* **PaC/SoC 超參初值**：K=4、r=5、d=64、α=0.2、λ=1.0、η=0.1、β=0.05、ε=1e-5、τ=1e-5。
* **馬氏切換**：第 3–5 個 epoch 起改用馬氏距離。

---

### 3. 實驗矩陣

#### 3.1 核心比較（先做）

| 類別        | 方法             | 損失組合                    | 跑法                |
| --------- | -------------- | ----------------------- | ----------------- |
| Baseline  | CE             | CE                      | 各 backbone × 各資料集 |
| +Contrast | CE+SupCon      | CE + SupCon (instance)  | 同上                |
| CLE-style | CE+Triplet     | 影像級 triplet             | 同上                |
| **Ours**  | **CE+PaC+SoC** | **CE + PaC + SoC(fro)** | **同上（主結果）**       |

#### 3.2 消融（主消融）

* **度量選擇**：SoC 距離 = Fro vs Log-Euclidean vs Stein。
* **PaC 構成**：餘弦 vs 馬氏；半難負對 vs 最難負對；是否用 Σplus。
* **K, d, r 掃描**：K ∈ {2,4,6}, d ∈ {32,64,128}, r ∈ {3,5,7}。
* **λ, η, β**：權重掃描（對主資料集至少做 3×3 網格）。
* **類原型 Σ\_c**：啟用/關閉；全矩陣 vs 對角近似。
* **Weighted CE**：啟用/關閉（γ∈{0,0.05,0.1}）。
* **Warmup**：有/無；切換 epoch ∈ {0,3,5}。
* **增強策略**：遮擋比例（小/中）、打亂程度（on/off）。

#### 3.3 擴充研究（選做）

* **不同 batch size / 記憶體足跡**：檢測可擴展性。
* **跨資料集泛化**：在資料集 A 訓練、B 上測試。
* **不同輸入解析度**：224/256/320 對精度與效率的影響。

---

### 4. 收集的數據與指標

#### 4.1 性能與泛化

* **Top-1 / Top-5 / Class-balanced Acc（CB-Acc）**
* **Confusion Matrix**（含易混類對）
* **mCE / ECE（Calibration）**（選做，溫度縮放前後）

#### 4.2 表徵品質（理論對應）

* **類間距離 / 類內散佈**：

  * $\operatorname{tr}(S_w)$、$\operatorname{tr}(S_b)$、Fisher 比 $J=\operatorname{tr}(S_w^{-1}S_b)$
  * **PaC margin 分佈**：$d_{\text{neg}}-d_{\text{pos}}$ 的均值、5/50/95 分位
* **SoC 對齊程度**：$\|\Sigma^{(1)}-\Sigma^{(2)}\|_F$、或 SPD 距離（log-euclidean/Stein）
* **嵌入可視化**：t-SNE/UMAP（**部位嵌入 Zt** 與 **全域嵌入 g** 各一），存圖與原始 2D 座標
* **部位穩定性**：跨視角匹配誤差（像素偏移分佈）、匹配成功率（IoU>閾值的比例）

#### 4.3 效率與資源

* **GPU 記憶體峰值（MB）**、**每 epoch 時間（min）**、**吞吐量（img/s）**
* **參數量（M）**、**FLOPs（G）**（backbone + 輕量 head，PaC/SoC 額外開銷）
* **訓練/驗證 I/O 利用率**（選做）

#### 4.4 訓練動態

* **Loss 曲線**：CE / PaC / SoC / Total
* **學習率、權重衰減** 追蹤
* **Cholesky 失敗率**（理論上應 0；如>0 代表 Σ 非 SPD 或 ε 太小）

---

### 5. 輸出產物與紀錄（強烈建議）

* `runs/{dataset}/{model}/{date_time}/`

  * `config.yaml`（超參、seed、資料路徑）
  * `metrics.csv`（逐 epoch：Top-1、Top-5、CB-Acc、Fisher、SoC距離、loss等）
  * `curves/`（訓練/驗證曲線 png）
  * `embeddings/`（t-SNE/UMAP npy + png）
  * `checkpoints/`（最佳與最後權重）
  * `perf.json`（記憶體、吞吐、FLOPs、params）
  * `ablation/`（各掃描表格）

---

### 6. 推薦的執行順序（兩週打底）

1. **核心比較**（單一 backbone×全資料集）→ 得到主表。
2. **主消融**（SoC 距離、K/d/λ/η/β、馬氏 vs 餘弦、Warmup）→ 產學術圖。
3. **效率評估**（記憶體/吞吐 vs baseline/CLE-style）。
4. **可視化與分析**（部位嵌入、Fisher、PaC margin 分佈）。
5. **擴充研究**（跨資料集、解析度/BS 變更）。

---

需要的話，我可以幫你把這兩份文件輸出成：

* **Mermaid 渲染好的 PNG/SVG**（或 PowerPoint 一頁一圖），
* **`metrics.csv` 樣板與 `ablation.xlsx` 表格模板**，
* **`config.yaml` 範本**與**實驗腳本（bash + Python）**。
  告訴我你偏好哪種格式，我直接生出檔案給你下載。

