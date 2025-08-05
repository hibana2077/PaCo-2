下面把**核心且困難**的部分整理成「可直接對照實作」的**虛擬碼**（Pseudo）。包含：**損失函數**、**流形距離計算**（特徵空間的馬氏距離 + SPD 協方差流形距離的兩種高效選項）、以及**必要但易遺漏**的穩定化與效率細節。所有模組皆為**可插拔**，能套在 **timm** 的 CNN backbone 最末層特徵圖上。

---

## 共用符號與介面

* `F ∈ R^{C×H×W}`：backbone 最末層特徵圖
* `K`：部位數；`r`：部位視窗大小；`d`：降維後通道數（如 64）
* `W ∈ R^{d×C}`：1×1 conv 權重（可學習）
* `ε, τ`：對角加載（diagonal loading）穩定常數（如 1e-5）
* `α`：Triplet margin；`λ, η`：損失權重
* 兩視角以標準增強 + 輕度遮擋/打亂產生：`view1, view2`

---

## 1) 部位抽樣（無標註；峰值選點 + 視窗池化）

```pseudo
FUNCTION GET_PARTS(F, K, r):
    # 空間與通道注意力（極簡版；可替換任意輕量顯著度）
    a = softmax(GAP(F))                            # a: R^{C}
    S = softmax(Conv1x1(F))                        # S: R^{1×H×W}  或用 mean over channels
    M = sum_c (a[c] * F[c,:,:]) * S                # M: R^{H×W}    （可只用 S 或 F 的 L2 以簡化）

    peaks = TOPK_PEAKS_WITH_NMS(M, K)              # 取 K 個局部最大（搭配 NMS）
    parts = []
    FOR p IN peaks:
        window = CROP_AROUND(F, center=p, size=r)  # R^{C×r×r}
        z = MEAN_POOL(window)                      # R^{C}
        parts.append(z)
    RETURN STACK(parts)                            # Z: R^{K×C}
```

---

## 2) 部位對應（匈牙利匹配，成本小：K×K）

```pseudo
FUNCTION MATCH_PARTS(Z1, Z2):
    # 以餘弦距離或 (簡化) 馬氏距離為代價，這裡先用餘弦避免早期不穩
    cost = PAIRWISE_COSINE_DISTANCE(Z1, Z2)        # R^{K×K}
    matching = HUNGARIAN(cost)                     # permutation m: {1..K}→{1..K}
    RETURN matching                                # m(k) -> Z2 的對應索引
```

---

## 3) 低秩降維與協方差估計（數值穩定＋線性時間）

```pseudo
FUNCTION LOW_RANK(Z, W):                           # Z: R^{K×C}
    Zt = (W @ Z^T)^T                               # Zt: R^{K×d}, 1×1 conv 等價線性降維

FUNCTION COVARIANCE(Zt, ε):
    # Zt: R^{K×d}; 使用無偏估計；避免顯式逆，後續以 Cholesky/解線性方程處理
    μ = MEAN(Zt, axis=0)                           # R^{d}
    Zh = Zt - μ
    Σ = (Zh^T @ Zh) / max(1, K-1)                  # R^{d×d}
    Σ = Σ + ε * I_d                                # 對角加載，保 SPD
    RETURN Σ, μ
```

---

## 4) 流形距離計算

### 4.1 特徵對的「馬氏距離」（度量來自部位協方差）

> 計算 `d_M(u, v) = sqrt((u-v)^T Σ^{-1} (u-v))` 時，**切勿顯式反矩陣**；改用 **Cholesky** 或 **共軛梯度解線性方程**。

```pseudo
FUNCTION MAHALANOBIS_DISTANCE(u, v, Σ):
    # u,v ∈ R^{d}; Σ ∈ R^{d×d}是 SPD
    L = CHOLESKY(Σ)                                # Σ = L L^T
    x = u - v
    y = SOLVE_LOWER(L, x)                          # 解 L y = x
    RETURN NORM(y, 2)                              # sqrt(x^T Σ^{-1} x)
```

### 4.2 協方差之間的 SPD 流形距離（SoC 的強化版，可三選一）

* **(A) Frobenius（最省）**：`||Σ1 - Σ2||_F`（論文主配方；最快）
* **(B) Log-Euclidean**（中等成本；較符 SPD 幾何）：`||log(Σ1) - log(Σ2)||_F`
* **(C) Stein/Jensen-Bregman LogDet**（免對數特徵分解；穩定快速）

```pseudo
FUNCTION SPD_DISTANCE(Σ1, Σ2, metric):
    IF metric == "fro":
        RETURN FROBENIUS_NORM(Σ1 - Σ2)

    ELSE IF metric == "log-euclidean":
        # d small (e.g., 64)，特徵分解可接受；更大可改 Newton-Schulz 近似 logm
        U1, Λ1 = EIGH(Σ1)                          # Σ=UΛU^T
        U2, Λ2 = EIGH(Σ2)
        L1 = U1 @ LOG(Λ1) @ U1^T                   # matrix log
        L2 = U2 @ LOG(Λ2) @ U2^T
        RETURN FROBENIUS_NORM(L1 - L2)

    ELSE IF metric == "stein":                     # Jensen-Bregman LogDet 距離
        Σm = 0.5 * (Σ1 + Σ2)
        # 使用 Cholesky + logdet，避免特徵分解
        v = LOGDET(Σm) - 0.5 * (LOGDET(Σ1) + LOGDET(Σ2))
        # Stein 距離常用 sqrt( v ) 或 2 * v 的變體，這裡回傳 v 的正比例版本
        RETURN MAX(0, v)                           # 數值上 v≥0（理論上）
```

> **實務建議**：主線 SoC 用 `fro`；若你要加強理論幾何，可切到 `log-euclidean`，但保持 `d≤64`。

---

## 5) 二階一致性損失（SoC）

```pseudo
FUNCTION LOSS_SOC(Σ1, Σ2, Σ_proto=None, beta=0.0, metric="fro"):
    # 主項：兩視角協方差接近
    d12 = SPD_DISTANCE(Σ1, Σ2, metric)             # 建議 metric="fro" 以省算
    L_soc = d12^2

    # 可選：往「類原型」協方差收斂，抑制類內散布
    IF Σ_proto IS NOT None AND beta > 0:
        Σ_bar = 0.5 * (Σ1 + Σ2)
        dproto = SPD_DISTANCE(Σ_bar, Σ_proto, metric)
        L_soc = L_soc + beta * dproto^2
    RETURN L_soc
```

---

## 6) 部位感知對比損失（PaC, 小批次內半難負對）

```pseudo
FUNCTION LOSS_PAC(Z1t, Z2t, matching, negatives_pool, Σplus, α):
    # Z1t, Z2t: R^{K×d} ；matching: m(k) → Z2t 的索引
    # negatives_pool：小批次其他影像的所有部位（R^{(B-1)·K×d}），省記憶庫
    total = 0
    count = 0
    FOR k IN [1..K]:
        p = Z1t[k]
        q = Z2t[ matching[k] ]                     # matched positive
        d_pos = MAHALANOBIS_DISTANCE(p, q, Σplus)  # Σplus = 0.5*(Σ1+Σ2) + τI

        # 半難負對：選擇 d_neg > d_pos 且差距最小者；若無則取最難
        d_negs = [ MAHALANOBIS_DISTANCE(p, n, Σplus) for n in negatives_pool ]
        d_negs_filtered = [d for d in d_negs if d > d_pos]
        IF d_negs_filtered NOT EMPTY:
            d_neg = MIN(d_negs_filtered)
        ELSE:
            d_neg = MIN(d_negs)                    # 最難負對（避免全易負）

        loss_k = MAX(d_pos - d_neg + α, 0)
        total += loss_k
        count += 1
    RETURN total / MAX(1, count)
```

> **備註**：`Σplus` 每張影像對應一次（視角平均 + τI），以 **Cholesky** 方式重用分解，避免多次逆。

---

## 7) 部位一致性加權的交叉熵（可選，但常有效）

```pseudo
FUNCTION LOSS_CE_WEIGHTED(logits, y, M, peaks, γ=0.1):
    # logits: 影像級分類器輸出；y: 標籤
    # M: 顯著度圖；peaks: K 個部位座標
    # 以部位周邊的平均顯著度作為權重 w，重加權 CE，鼓勵依賴穩定部位
    w = 0
    FOR p IN peaks:
        w += MEAN_IN_WINDOW(M, center=p, size=r)
    w = w / K
    base = CROSS_ENTROPY(logits, y)
    RETURN (1 - γ) * base + γ * w * base
```

---

## 8) 總損失與訓練步驟（可直接映射到 PyTorch/timm）

```pseudo
FUNCTION TRAIN_STEP(x, y, model, W, hyper):
    # === 兩視角 ===
    x1 = AUGMENT_VIEW1(x)                          # 含標準增強
    x2 = AUGMENT_VIEW2(x)                          # 輕度遮擋/打亂

    # === backbone 前向（timm CNN）===
    F1 = model.forward_features(x1)                # R^{C×H×W}; 不改 backbone
    F2 = model.forward_features(x2)

    # === 部位抽樣 ===
    Z1 = GET_PARTS(F1, K=hyper.K, r=hyper.r)       # R^{K×C}
    Z2 = GET_PARTS(F2, K=hyper.K, r=hyper.r)

    # === 對應與降維 ===
    m  = MATCH_PARTS(Z1, Z2)                       # 匈牙利；K×K
    Z1t = LOW_RANK(Z1, W)                          # R^{K×d}
    Z2t = LOW_RANK(Z2, W)

    # === 協方差與 Σplus ===
    Σ1, μ1 = COVARIANCE(Z1t, ε=hyper.epsilon)
    Σ2, μ2 = COVARIANCE(Z2t, ε=hyper.epsilon)
    Σplus  = 0.5*(Σ1 + Σ2) + hyper.tau * I_d       # 確保 SPD

    # === CE（影像級分類頭）===
    g1 = GAP(F1); g2 = GAP(F2)                     # 可接線性分類頭
    logits = CLASSIFIER( (g1 + g2)/2 )
    L_ce = CROSS_ENTROPY(logits, y)                # 或用 LOSS_CE_WEIGHTED(logits, y, ...)

    # === 小批次內負對池（其它樣本的部位特徵）===
    negatives_pool = GATHER_OTHER_PARTS_IN_BATCH(Z1t, Z2t, exclude_current=True)

    # === 部位感知對比 ===
    L_pac = LOSS_PAC(Z1t, Z2t, m, negatives_pool, Σplus, α=hyper.alpha)

    # === 二階一致性 ===
    L_soc = LOSS_SOC(Σ1, Σ2, Σ_proto=LOOKUP_CLASS_COV(y), beta=hyper.beta, metric=hyper.metric)

    # === 總損失 ===
    L = L_ce + hyper.lambda * L_pac + hyper.eta * L_soc

    BACKPROP(L)
    UPDATE(model, W, CLASSIFIER)
    # 可選：以指數移動平均更新類別協方差原型
    UPDATE_CLASS_PROTO_COV(y, Σ_bar=0.5*(Σ1+Σ2), momentum=hyper.mproto)

    RETURN { loss:L, L_ce:L_ce, L_pac:L_pac, L_soc:L_soc }
```

---

## 9) 重要但易遺漏的穩定化與效率細節

```pseudo
# (1) 以 Cholesky / solve 避免顯式逆；Σplus 重用分解（per-sample cache）。
# (2) 對角加載：Σ ← Σ + εI；Σplus ← Σplus + τI，確保 SPD、避免奇異。
# (3) K、d 小常數（如 K=4, d=64），確保 O(K·d^2)；d 過大才考慮近似對數（Newton-Schulz）。
# (4) 匹配早期不穩：先用餘弦距離；待 SoC 收斂後切換馬氏度量（warm-up T_warm）。
# (5) 半難負對選取避免全易負；若全為易負，退化到最難負對。
# (6) 部位落點可加「鄰域幾何一致性」檢查：若匹配點偏離>δ，給低權重或重匹配。
# (7) 兩視角的剪裁/遮擋幅度保持小，以免破壞可匹配結構；避免大面積隨機擦除。
# (8) 類別協方差原型 Σ_proto 用 EMA 維護（記憶體僅存每類一個 d×d；類別多時可降維或採對角近似）。
# (9) 計算圖節省：匹配與部位選點可 detach（不回傳導）以避免高階導數開銷。
# (10) 多尺度可選：若 backbone 輸出多 stage，僅在最後一層做 PaC/SoC 以省算。
```

---

## 10) 總結：最小可用超參（建議起點）

```pseudo
K=4, r=5, d=64, α=0.2, λ=1.0, η=0.1, beta=0.05,
epsilon=1e-5, tau=1e-5, metric="fro", mproto=0.9,
warmup_for_mahalanobis=3~5 epochs
```

---