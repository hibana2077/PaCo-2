以下提供兩份文件：

1. **Training + Validation 流程圖**（含簡要步驟說明）；
2. **要跑的實驗與要收集的數據**（含核心/擴充實驗矩陣與指標清單）。
   皆以你先前的 PaC/SoC 設計與 timm-CNN 為前提撰寫。

---

## （一）Training + Validation 流程圖

> **說明**：用 Mermaid 描述主流程；若你的環境不支援 Mermaid，請把程式碼貼到支援的 Markdown/diagram 工具即可渲染。

### A. Training 流程圖

```mermaid
flowchart TD
    A[Start: Load batch (x, y)] --> B[Augment: view1, view2\n(標準增強 + 輕遮擋/打亂)]
    B --> C[Backbone (timm CNN)\nforward_features]
    C --> C1[F1 (C×H×W) from view1]
    C --> C2[F2 (C×H×W) from view2]
    C1 --> D1[Part Sampling on F1\n顯著度M, Top-K峰值, r×r池化 → Z1(K×C)]
    C2 --> D2[Part Sampling on F2\n顯著度M, Top-K峰值, r×r池化 → Z2(K×C)]
    D1 --> E[Hungarian Matching m(k)]
    D2 --> E
    E --> F1[1×1 Conv 降維 W: C→d\nZ1t(K×d), Z2t(K×d)]
    F1 --> G1[Covariance Σ1, μ1\n(ε I對角加載)]
    F1 --> G2[Covariance Σ2, μ2\n(ε I對角加載)]
    G1 --> H[Σplus = 0.5(Σ1+Σ2)+τI\n(Cholesky cache)]
    G2 --> H
    H --> I[PaC: Triplet on Parts\nMahalanobis 距離 with Σplus]
    G1 --> J[SoC: 協方差一致性\n||Σ1-Σ2||² 或 SPD距離²]
    G2 --> J
    C1 --> K[GAP → g1]
    C2 --> K2[GAP → g2]
    K --> L[Classifier on (g1+g2)/2 → logits]
    K2 --> L
    L --> M[CE 或 部位加權 CE]
    I --> N[Loss Combine:\nL = CE + λ·PaC + η·SoC]
    J --> N
    N --> O[Backprop + Update (SGD/AdamW)]
    O --> P[Update class proto Σ_c (EMA)]
    P --> Q[Log metrics & save ckpt]
    Q --> R{Next batch?}
    R -- Yes --> A
    R -- No --> S[End Epoch]
```

**要點（落地細節）**

* **數值穩定**：協方差加 $\epsilon I$、Σplus 再加 $\tau I$；馬氏距離用 **Cholesky+solve**，避免顯式逆。
* **效率**：K、d 小常數；小批次內負對池替代記憶庫；Σplus 的 Cholesky 分解做 cache。
* **對應穩定**：前幾個 epoch 先用餘弦距進行匹配與對比，warmup 後切馬氏度量。
* **原型協方差**：以 EMA 維護，每類一個 $d\times d$（類別很多時可用對角近似）。

---

### B. Validation/Test 流程圖

```mermaid
flowchart TD
    A[Start: Load val batch (x, y)] --> B[Preprocess: center/resize crop\n(無強增強)]
    B --> C[Backbone forward_features → F]
    C --> D[GAP → g]
    D --> E[Classifier(g) → logits]
    E --> F[Compute Top-1/Top-5/CB-Acc]
    C --> G[可選: Part Sampling + 1×1降維\nZt(K×d)]
    G --> H[Optional: Σ, 部位嵌入保存\n(分析用，不回傳導)]
    H --> I[計算類內/類間散佈、Fisher比、\nSoC距離、t-SNE/UMAP資料]
    F --> J[Aggregate metrics by class/fold]
    I --> J
    J --> K[Log & Save best ckpt by metric]
    K --> L[End]
```

**要點**

* Valid 不做 SoC / PaC 反向傳遞，只**計算分類指標**；部位/協方差僅用於**分析與可視化**。
* 若需 TTA（多裁切），以少量水平翻轉/多尺度平均，不引入遮擋/打亂。