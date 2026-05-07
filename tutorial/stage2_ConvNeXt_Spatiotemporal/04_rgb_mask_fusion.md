# 第 4 章 — RGB feature 與 stage 1 mask 的融合機制

> 撰寫日期:2026-05-05
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:解釋 ConvNeXt 抽出的 RGB feature 怎麼跟 stage 1 SAM3 推論出的 mask 結合成「一張同時懂影像又懂解剖位置」的特徵地圖

---

## 為什麼需要把 RGB 跟 mask「融合」?

stage 2 的目標是預測 CBD bbox 的座標。它有兩個資訊源:

1. **RGB frame**:原始影像本身——所有視覺線索(器械、組織、顏色、紋理、邊緣)
2. **stage 1 SAM3 推論的 mask**:`gallbladder` 與 `liver` 的位置(2 通道 binary mask)

如果只有 RGB,模型必須**從零學會「膽囊在哪、肝在哪」**才能推論「CBD 應該在它們之間」——這太難。

如果只有 mask,模型只知道「膽囊在這個區塊、肝在那個區塊」,但**完全看不到影像細節**(出血、ICG 螢光的程度、組織質感)——也不夠。

**兩個資訊融合**=「模型同時看到影像本身,以及解剖學上『該關心哪些結構』」=精確判斷的基礎。

**外科類比**:就像 ICG 螢光腹腔鏡——術者同時看「白光影像」(看到組織的紋理顏色)與「螢光通道」(看到膽道結構顯影)。**單獨任何一個都不夠**,要疊加才有價值。stage 2 的 fusion 做的是同樣的事。

---

## stage 1 mask 是怎麼進來的?

`bsafe_cbd.yaml` 第 33-36 行:

```yaml
stage1_sam3:
  easy_prompts:
    - gallbladder
    - liver
```

stage 1 的 SAM3+LoRA 模型被叫進來,對每段 clip 跑出 **2 個 mask**(對應兩個 prompt:gallbladder + liver),壓縮存成 `.npz` 檔。stage 2 訓練時,從快取直接讀取——不用每次重跑 SAM3。

這個快取邏輯寫在 `src/cbd/cache.py`,讀取邏輯寫在 `src/cbd/common.py:207-227`(`load_mask_cache_tensor`)。從 dataset 出來時,mask 形狀是 `(T, 2, H, W)`,2 個通道分別代表 gallbladder mask 與 liver mask。

進入 backbone 前,mask 透過 `F.interpolate` 被 resize 成跟 RGB 同樣的 `(image_size, image_size)`(`common.py:201-204`)。

**為什麼 mask 通道數是 2?**——因為 `easy_prompts` 列了 2 個。如果改成 3 個 prompts,mask 通道數就變 3,但這時 mask encoder 的 `nn.Conv2d(2, 32, ...)` 就要跟著改。本專案目前寫死 2,改 prompt 數量會是個 breaking change。

---

## v1 的融合方式(簡單但會丟空間資訊)

`src/cbd/model.py` 第 63-77 行:

### v1 的 mask encoder

```python
self.mask_encoder = nn.Sequential(
    nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, self.mask_channels, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(1),    # ★ 整個 spatial pool 掉
)
```

注意最後一行 `nn.AdaptiveAvgPool2d(1)`——它把整個空間 grid 壓成 1×1。所以 mask encoder 的輸出形狀是 `(B*T, 128, 1, 1)`,flatten 後變 `(B*T, 128)`。

### v1 的 fusion

`model.py:73-77`:

```python
self.fusion = nn.Sequential(
    nn.Linear(self.rgb_dim + self.mask_channels, self.d_model),
    nn.ReLU(inplace=True),
    nn.Dropout(self.dropout),
)
```

整個 fusion 就是把 768 維 RGB 向量 + 128 維 mask 向量 concat 成 896 維 → MLP 投影成 256 維(`d_model`)。

```
RGB feature   (B*T, 768)  ─┐
                            ├─► concat (B*T, 896) ─► Linear → (B*T, 256)
Mask feature  (B*T, 128)  ─┘
```

**特性**:這個 fusion 是「**frame-level**」的——每一張 frame 變成單一個 256 維向量,完全沒有空間資訊。後續 temporal transformer 看的是 25 個這種向量(對應 25 張 frame)。

**外科類比**:像是「術中總覽鏡頭+語音標記」——每一秒給術者「整體狀況概述」,但不告訴你「具體哪個位置該注意」。

---

## v2 的融合方式(保留空間網格)

`src/cbd/model.py` 第 149-161 行:

### v2 的 mask encoder

```python
self.mask_encoder = nn.Sequential(
    nn.Conv2d(2, 32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, self.mask_channels, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    # 注意:沒有 AdaptiveAvgPool!
)
```

**v2 的 mask encoder 沒有 pool**——它保留輸入的空間尺寸。但這時 mask 還是 `(image_size, image_size)` = 512×512,跟 ConvNeXt 出來的 16×16 對不齊!

### v2 的關鍵步驟:把 mask resize 到跟 RGB feature 同尺寸

`model.py:213`:

```python
mask_flat = F.interpolate(mask_flat, size=(grid_h, grid_w), mode="nearest")
```

這行把 mask 從 `(B*T, 2, 512, 512)` 直接降到 `(B*T, 2, 16, 16)`——使用 **nearest neighbor 插值**,確保 mask 仍然是 binary-like 的(不會被線性插值模糊掉)。

### v2 的 fusion projection(1×1 conv)

`model.py:157-161`:

```python
self.fusion_projection = nn.Sequential(
    nn.Conv2d(self.rgb_dim + self.mask_channels, self.d_model, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.Dropout2d(self.dropout),
)
```

關鍵是 **1×1 卷積**——它只在通道維度做線性投影,**不動空間 grid**。所以:

```
RGB feature   (B*T, 768, 16, 16)  ─┐
                                    ├─► concat (B*T, 896, 16, 16) ─► 1×1 conv → (B*T, 256, 16, 16)
Mask feature  (B*T, 128, 16, 16)  ─┘
```

每個 16×16 的 grid cell 都被獨立投影——同一個 1×1 conv 的權重套用到所有 256 個 cell。

**特性**:這個 fusion 是「**spatial-level**」的——每張 frame 變成 256 個 token(每個 token 對應原圖一個 32×32 區塊),每個 token 同時編碼了「**這個區塊的 RGB 視覺特徵**」+「**這個區塊有沒有膽囊/肝**」。

**外科類比**:像是「ICG 螢光疊加白光影像」——你**逐個區塊**看「這裡的紋理是什麼?這裡有沒有 ICG 顯影?」結合成一個融合視野。

---

## 為什麼 mask 要先過 mask_encoder,不直接 concat 上去?

這是個合理的疑問——既然 mask 已經是 binary,為什麼不直接把 `(2, 16, 16)` 的 mask resize 後跟 RGB feature concat?中間多一個 mask_encoder 是不是多餘?

設計理由(從 model.py 寫法反推):

1. **讓 mask 也有 learnable representation**——直接 binary mask 是「0/1 的硬訊號」,過了 conv 之後變成「**128 維的軟訊號**」,可以表達「邊緣的不確定性」「兩個 mask 重疊區域」這種更細的資訊
2. **匹配 channel 級別**——如果 RGB 是 768 通道、mask 只有 2 通道,concat 後 mask 訊號被 RGB「淹沒」(資訊稀釋)。把 mask 也升到 128 通道,讓它在融合時有相當的份量
3. **學習特徵組合**——mask_encoder 內的多層 conv 可以**學習怎麼把兩個 mask 通道(gb + liver)組合成更有用的訊號**(例如「兩 mask 中間的薄條狀區域」)

**外科類比**:不是把「螢光訊號的明暗」直接疊上影像,而是先讓影像處理機**對螢光做預處理**(放大、去噪、強化邊緣),再疊加。處理後的訊號比原始 binary 更有用。

---

## v1 mask encoder vs v2 mask encoder 的差別

兩者長得很像,但有 3 個關鍵差別:

| 項目 | v1(`model.py:63-71`) | v2(`model.py:149-156`) |
|---|---|---|
| stride | 每層 stride=2(三次下採樣) | 每層 stride=1(不改變空間) |
| 結尾 | `AdaptiveAvgPool2d(1)` 整體 pool | 沒有 pool,保留 spatial |
| 輸出形狀 | `(B*T, 128)` | `(B*T, 128, H, W)` |

**為什麼設計上分歧?**——因為兩個 variant 對 mask 資訊的需求不同:

- v1:後面是 frame-level pool,反正空間會丟,mask encoder 不如直接 pool 省事
- v2:後面要保留空間,mask encoder 自然也要保留空間

這個設計很「對稱」——v1 從頭到尾都是 frame-level、v2 從頭到尾都是 spatial-level。

---

## resize mask 為何用 nearest 而不是 bilinear?

`model.py:213`:

```python
mask_flat = F.interpolate(mask_flat, size=(grid_h, grid_w), mode="nearest")
```

選 `nearest` 而不是預設的 `bilinear` 是有原因的:

- mask 本質上是 **0/1 的指示訊號**(這個位置有沒有膽囊?)。bilinear 會把邊緣模糊成 0.5、0.3、0.7 這種小數值,**改變了 mask 的語意**
- nearest 直接取最近的 pixel 值,保持 0/1 性質,**邊緣會階梯狀但語意不變**
- 後面的 mask_encoder 用 conv 自己會處理邊緣的軟硬,不需要在 resize 階段就模糊

**外科類比**:像是「**地圖上的行政區界線**」——州界要嘛屬於 A 州、要嘛屬於 B 州,不可能「30% A 州、70% B 州」。模糊處理會破壞這個離散性質。

---

## 為什麼 1×1 conv 是 fusion 的好選擇?

v2 用 1×1 conv 做最終融合(`model.py:158`)。為什麼是 1×1 而不是 3×3 或 5×5?

- **1×1 = 純通道混合**:每個 grid cell 內,把 896 個 channel 線性組合成 256 個 channel。**完全不混合鄰近 cell**。
- 3×3 = 通道混合 + 空間混合:會把鄰近 cell 的訊號也吸進來
- 既然 ConvNeXt 已經對 RGB 做了大量空間處理、mask_encoder 也對 mask 做了大量空間處理,**fusion 階段不需要再混空間了**——1×1 是最節能的選擇
- 後面 temporal transformer 會處理跨 cell + 跨 frame 的注意力,fusion 階段只負責「點對點」融合就好

**外科類比**:像是「最後一道調味」——只負責把已經煮好的兩種食材調勻(同一格內的 RGB+mask),不用再切碎或拌入鄰桌的食材(那是後面 transformer 的工作)。

---

## fusion 的位置編碼(positional encoding)細節

`model.py:163-165`(v2):

```python
self.temporal_position = nn.Parameter(torch.zeros(1, self.clip_len, 1, 1, self.d_model))
self.row_position = nn.Parameter(torch.zeros(1, 1, self.grid_size, 1, self.d_model))
self.col_position = nn.Parameter(torch.zeros(1, 1, 1, self.grid_size, self.d_model))
```

fusion 之後,v2 還會把這 3 種 learnable position embedding 「**加上去**」:

- `temporal_position`:對 25 個 frame 各自一組 256 維向量
- `row_position`:對 16 個 row 各自一組
- `col_position`:對 16 個 col 各自一組

`model.py:217-220`:

```python
fused = fused.view(batch_size, clip_len, self.d_model, grid_h, grid_w).permute(0, 1, 3, 4, 2)
fused = fused + self.temporal_position[:, :clip_len]
fused = fused + self.row_position[:, :, :grid_h]
fused = fused + self.col_position[:, :, :, :grid_w]
```

這三個位置編碼是**相加而不是 concat**——讓 token 同時帶有「我是第幾個 frame、第幾個 row、第幾個 col」的訊號。下游 transformer 因此知道每個 token 的時空座標。

**外科類比**:就像在每段手術錄影上印**時間碼+鏡頭位置**——讓回放時能精準對到「術中第幾分幾秒、視野的哪個區域」。沒有這個,影像就只是「一堆混在一起的畫面」,順序與位置都不見了。

---

## 這一章你需要帶走的重點

1. fusion 的目的:**讓模型同時看到 RGB 視覺特徵 + stage 1 SAM3 給的解剖位置 mask**
2. mask 來自 stage 1 SAM3+LoRA 用 `gallbladder` 與 `liver` 兩個 prompt 跑出的快取(`bsafe_cbd.yaml:33-36`)
3. **v1 fusion 是 frame-level**(整張 pool 成一個 256 維向量),**v2 fusion 是 spatial-level**(每個 16×16 grid cell 一個 256 維向量)
4. v2 把 mask 從 512×512 用 **nearest interpolation** 降到 16×16,保持 binary 語意
5. fusion 用 **1×1 conv**——只混通道、不混空間,因為空間混合留給後面的 transformer
6. mask 不直接 concat 上 RGB,而是先過 mask_encoder——讓 mask 也有 learnable representation,並把通道升到 128 與 RGB 平衡
7. v2 在 fusion 之後加 3 種 positional embedding(temporal、row、col),讓 transformer 知道每個 token 的時空座標

---

## 進一步深挖的線索

- 經典「multi-modal fusion」討論:Ramachandran et al. *Stand-Alone Self-Attention*(2019)——討論 1×1 conv 在 multi-modal 場景的選擇邏輯
- Late fusion vs early fusion 對比:醫療影像領域常見討論,本專案屬於「**中間融合(intermediate fusion)**」
- 為何 segmentation mask 拿來當輔助訊號(prior)而不是直接當答案:Lin et al. *Focal Loss for Dense Object Detection* 提供啟發

---

## 對話脈絡記錄

- **2026-05-05**:第 4 章是 ConvNeXt 教學的「最後 backbone-中心的章節」——下一章開始重心會移到 temporal transformer。本章特別強調「為什麼 mask 也要過 encoder」這種看起來多餘但實際必要的設計選擇。

---

## 程式碼速查總表

### Mask 從 stage 1 進到 stage 2 的路徑

| 步驟 | 檔案行號 | 內容 |
|---|---|---|
| stage 1 prompt 設定 | `configs/bsafe_cbd.yaml:33-36` | `easy_prompts: [gallbladder, liver]` |
| Mask 快取讀取函式 | `src/cbd/common.py:207-227` | `load_mask_cache_tensor` |
| Resize 到 image_size | `src/cbd/common.py:201-204` | `resize_mask_sequence` 用 nearest |
| Dataset 回傳 | `src/cbd/dataset.py:303-307` | `masks, payload = load_mask_cache_tensor(...)` |
| Collate 堆疊 | `src/cbd/dataset.py:336` | `torch.stack([sample["masks"] ...])` 變 (B,T,2,H,W) |
| 進 model | `src/cbd/engine.py:265` | `self.model(batch["rgb"], batch["masks"])` |

### v1 fusion(frame-level)

| 步驟 | model.py 行號 | 形狀變化 |
|---|---|---|
| mask_encoder | 63-71 | (B*T, 2, H, W) → (B*T, 128, 1, 1) |
| flatten | 102 | (B*T, 128, 1, 1) → (B*T, 128) |
| concat | 113 | (B*T, 768) ⊕ (B*T, 128) → (B*T, 896) |
| MLP fusion | 73-77 | (B*T, 896) → (B*T, 256) |
| Reshape 為 frame 序列 | 114 | (B*T, 256) → (B, T, 256) |

### v2 fusion(spatial-level)

| 步驟 | model.py 行號 | 形狀變化 |
|---|---|---|
| ConvNeXt features | 211 | (B*T, 3, H, W) → (B*T, 768, 16, 16) |
| Resize mask | 213 | (B*T, 2, H, W) → (B*T, 2, 16, 16) |
| mask_encoder | 149-156 + 214 | (B*T, 2, 16, 16) → (B*T, 128, 16, 16) |
| concat 在 channel | 215 | concat → (B*T, 896, 16, 16) |
| 1×1 conv fusion | 157-161 + 215 | (B*T, 896, 16, 16) → (B*T, 256, 16, 16) |
| 重組為 (B, T, H, W, d) | 217 | view + permute |
| 加 3 種 position embedding | 218-220 | broadcast 加總 |
| 攤成 token 序列 | 223 | (B, T*16*16, 256) |

### 1×1 conv 為何選 1×1 的證據

| 觀察 | 行號 |
|---|---|
| `nn.Conv2d(self.rgb_dim + self.mask_channels, self.d_model, kernel_size=1)` | `model.py:158` |
| 配套的 `nn.Dropout2d(self.dropout)` | `model.py:160` |
| 不同於 mask_encoder 內部用的 `kernel_size=3`(空間混合) | `model.py:150, 152, 154` |

### 快速驗證命令

```bash
# 確認 mask resize 用 nearest 而非 bilinear
grep -n "interpolate" src/cbd/model.py src/cbd/common.py

# 確認 v2 fusion 是 1×1 conv
grep -n "kernel_size=1" src/cbd/model.py

# 確認 mask 快取路徑(stage 1 的快取在哪)
grep -n "easy_mask_cache_root\|masks.npz" src/cbd/dataset.py src/cbd/sources.py
```
