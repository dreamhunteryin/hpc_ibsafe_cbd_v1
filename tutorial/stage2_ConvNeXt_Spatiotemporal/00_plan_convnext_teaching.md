# Stage 2 起點 — ConvNeXt 教學規劃

> 撰寫日期:2026-05-05
> 目的:規劃 stage 2 的學習路線,起點是「ConvNeXt-Small backbone 在這個 pipeline 中的角色與功能」。
> 此文件是**大綱**,不含實際教學內容。等使用者確認大綱後再展開撰寫。

---

## 為什麼從 ConvNeXt 開始

`cbd_v1/src/cbd/model.py` 是 stage 2 的核心,僅 269 行。讀完它你會發現:**整個 stage 2 只用了一個視覺 backbone,就是 ConvNeXt-Small**。所有 spatiotemporal 設計、box query、type head、heatmap head,全都建在這個 backbone 抽出來的 feature map 上。

換句話說:**先搞懂 ConvNeXt 在輸入輸出層級做了什麼,後面所有 stage 2 模組才有立足點**。

外科類比:就像在學一個「複合手術 pipeline」前,要先知道每個器械(backbone)是做什麼用的。ConvNeXt 在這個 pipeline 中相當於「視覺 sensor」——把每張 frame 變成電腦看得懂的「結構化特徵地圖」(feature map),後面的 transformer 才有東西吃。

---

## 已掃描到的關鍵事實(planning 階段先列,後面分章詳述)

從 `cbd_v1/src/cbd/model.py` 與 `configs/bsafe_cbd.yaml` 直接觀察:

1. **backbone 來源**:`from torchvision.models import convnext_small, ConvNeXt_Small_Weights`——直接用 torchvision 的官方實作,**不是自己刻的**,也不是 SAM3 內建的。
2. **預訓練權重**:`ConvNeXt_Small_Weights.DEFAULT`(ImageNet-1K)。**完全沒有用醫療資料預訓練**。
3. **被用在兩個 variant**(model.py 第 41 / 120 行):
   - `CBDV1GlobalPool`:把 ConvNeXt 抽出的 feature map 整張 average pool 成一個向量,只看「整張 frame 是什麼」
   - `CBDV2Spatiotemporal`:**保留 ConvNeXt 的空間網格**(16×16),不 pool,直接餵給 temporal transformer
4. **預設 variant 是 v2**(`bsafe_cbd.yaml` 第 45 行 `variant: v2_spatiotemporal`)
5. **三種凍結模式**(model.py 第 32-38 行):`freeze_all`、`last_stage`、`full`;v2 預設 `last_stage`(只解凍 `features[6:]`)
6. **input size**:`bsafe_cbd.yaml` 第 48 行 `input_size: 512`。ConvNeXt stride = 32 → grid = 16×16
7. **feature 維度**:`self.rgb_dim = 768`(ConvNeXt-Small 最後一個 stage 的 channel 數)
8. **與 mask feature 融合**:RGB 768 dim concat mask 128 dim → 1×1 conv 投影到 `d_model=256`(model.py 第 157-161 行)
9. **不同學習率**(`bsafe_cbd.yaml` 第 62-63 行):`backbone_lr: 1e-5` vs `new_layers_lr: 1e-4`——backbone 學得慢、新層學得快

這些事實本身已經夠豐富,可以拆成 4-5 章說清楚。

---

## 教學章節規劃

我建議把 ConvNeXt 教學切成 **4 章 + 1 章後續銜接**,每章對焦一個你會想問的具體問題:

### 第 1 章 — `01_convnext_architecture_basics.md`
**核心問題**:ConvNeXt 是什麼?為什麼是「現代化的 CNN」?Small / Base / Large 的差別?

**會講的內容**:
- ConvNeXt 的歷史定位:Meta 2022 ConvNeXt 論文,挑戰「ViT 一定贏 CNN」的看法
- 與 ResNet、Swin Transformer 的關係(ConvNeXt = 用 transformer 設計哲學重新煉的 CNN)
- 4 個 stage、stride 32、channel 漸增(96→192→384→768)的金字塔結構
- Small vs Tiny vs Base vs Large 的 trade-off(計算量 vs 表達力)
- 為什麼這個專案選 **Small** 而不是 Tiny / Base?(我會用 ImageNet acc + GPU memory 估算反推工程師的決策)

**外科類比**:像各種型號的腹腔鏡——5 mm / 10 mm / 30 度,挑哪個取決於手術難度與可用空間。

**會引用的程式碼位置**:`model.py` 第 53-56 行、`bsafe_cbd.yaml` 第 46 行

---

### 第 2 章 — `02_convnext_role_in_cbdv2.md`
**核心問題**:在 CBD v2 pipeline 裡,ConvNeXt **吃什麼、吐什麼**?它的 feature map 給誰用?

**會講的內容**:
- 輸入:`(B, T=25, 3, 512, 512)` 的 RGB clip → reshape 成 `(B*T, 3, 512, 512)` 餵給 backbone
- 輸出:`(B*T, 768, 16, 16)` 的 feature map(stride 32)
- `self.rgb_backbone.classifier = nn.Identity()`——把 ImageNet 分類頭拔掉,只取 features
- v1 vs v2 的差別:**v1 是 mean pool 變向量,v2 是保留 grid**——這是整個 stage 2 設計的核心 fork
- 為什麼 stage 2 不繼續用 SAM3 image encoder?(SAM3 太大、推論成本高、且空間 grid 已經 down-sample 過,反而不利下游用)
- ConvNeXt 抽出的 feature map 不是直接給 box head 看,而是先跟 stage 1 給的 mask 做 fusion(預告第 4 章)

**外科類比**:ConvNeXt 像「術中影像增強層」——它把 raw 影像轉成「凸顯結構邊緣 / 紋理 / 對比」的版本,後面的演算法才看得清楚。

**會引用的程式碼位置**:`model.py` 第 95-117 行(v1 forward)、第 206-252 行(v2 forward)、`bsafe_cbd.yaml` 第 44-56 行

---

### 第 3 章 — `03_convnext_freeze_strategies.md`
**核心問題**:為什麼工程師選 `last_stage` 凍結?三種模式各適合什麼場景?

**會講的內容**:
- 三種模式的精確定義(從 model.py 第 32-38、139-148 行直接讀)
  - `freeze_all`:整個 backbone 凍結,只訓新層
  - `last_stage`:只解凍 `features[6:]`(最後一個 stage)
  - `full`:整個 backbone 一起訓
- 為什麼 v1 預設是 `freeze_all`、v2 預設是 `last_stage`?(從 `resolve_backbone_mode` 第 35-36 行的條件分支推測工程師的設計理由)
- ImageNet 預訓練的 representation 為什麼可以直接用在內視鏡影像?——transfer learning 在低層特徵(邊緣、紋理)上的可遷移性
- `backbone_lr=1e-5` vs `new_layers_lr=1e-4`(差 10×)的意義——partial unfreezing 配合差別學習率,避免破壞預訓練 representation
- 對應外科決策:像「重新訓練老主治」vs「派 fellow 上手新術式」——主治(backbone)只在最後關鍵步驟微調,fellow(新層)從零學。

**會引用的程式碼位置**:`model.py` 第 32-38、59-61、139-148、203-204 行、`bsafe_cbd.yaml` 第 50、62-63 行

---

### 第 4 章 — `04_rgb_mask_fusion.md`
**核心問題**:ConvNeXt 抽的 RGB feature 怎麼跟 stage 1 SAM3 給的 mask 結合?

**會講的內容**:
- 輸入端的 mask 是哪來的?——是 stage 1 SAM3+LoRA 推論出來的 gallbladder + liver mask(2 channel,所以 `nn.Conv2d(2, 32, ...)`)
- v1 的融合方式:RGB pool 成 768 dim 向量 + mask pool 成 128 dim 向量 → concat → MLP → 256 dim
- v2 的融合方式:**保留空間 grid**——RGB feature `(768, 16, 16)` + mask feature `(128, 16, 16)` → concat 在 channel 維度 → 1×1 conv 投影成 `(256, 16, 16)`
- 為什麼 mask 要先過 mask encoder 再融合,而不是直接 concat?(讓 mask 也有 learnable representation,不是 raw binary map)
- 為什麼 v2 要先把 mask resize 成跟 RGB feature map 同尺寸?(`F.interpolate` 第 213 行)
- 這個融合決定了 stage 2 的「視野」——它同時看到「frame 長什麼樣(RGB)」與「stage 1 認為解剖在哪(mask)」

**外科類比**:就像在 ICG 螢光影像上同時疊上白光 RGB 影像——兩個資訊源融合後,術者(下游模組)才能做精準判斷。

**會引用的程式碼位置**:`model.py` 第 63-77 行(v1 mask encoder + fusion)、第 149-161 行(v2 mask encoder + fusion projection)、第 213 行(F.interpolate)

---

### 第 5 章 — `05_handoff_to_temporal_transformer.md`(銜接章)
**核心問題**:ConvNeXt + mask 融合後的 feature,怎麼交給 temporal transformer?ConvNeXt 教學在這裡告一段落,後續會展開 spatiotemporal transformer 章節。

**會講的內容**:
- v1:`(B, T, 256)` 序列 → temporal transformer → 取最後一個 timestamp → box head
- v2:`(B, T*16*16+1, 256)` token 序列(含一個 CLS token)→ temporal transformer → 解出 CLS token + spatial tokens → 多頭(box / type / center cell / heatmap)
- 為什麼 v2 要混合 spatial + temporal token 到同一個 transformer?(不是先空間再時間的 factorized 設計,是 joint attention)
- ConvNeXt 教學的下一站:**spatiotemporal transformer 內部**(這就是 stage 2 的下一個章節主題)

**會引用的程式碼位置**:`model.py` 第 78-93、107-117 行(v1)、第 162-183、217-243 行(v2)

---

## 教學內容的撰寫風格(沿用 stage 1 的慣例)

從 stage 1 的 02_Notes、06_Notes 學到的撰寫模式,我會在每章遵守:

1. **文字優先,程式碼放尾巴** — 每章前 70% 用敘述+類比解釋觀念,程式碼在「程式碼速查總表」段落集中列出
2. **外科類比** — 每個關鍵抽象至少給一個外科手術 / 解剖 / 病房作業的對照
3. **帶走的重點** — 章末用 5-10 條 bullet 凝練成可直接記憶的結論
4. **進一步深挖的線索** — 列出可選的延伸閱讀(原論文 / 相關 benchmark)
5. **對話脈絡記錄** — 章末記錄當次討論的問題,讓未來回頭看能還原當時的脈絡

---

## 這份規劃需要你確認的決策點

1. **章節數量**:5 章夠不夠?太多?要不要把第 4、5 章合併?
2. **要不要從第 1 章「ConvNeXt 是什麼」開始**,還是直接跳到第 2 章「在這個 pipeline 的位置」?(第 1 章對你來說可能太「教科書」)
3. **要不要在某一章補上「v1 vs v2 的選擇邏輯」**——也就是工程師為什麼設計兩個 variant?是 ablation 嗎?(從 git log 可以查證)
4. **每一章預計長度**:stage 1 的 02_Notes 大概 51 KB、06_Notes 大概 36 KB,你希望 stage 2 的章節維持這個量級,還是更精簡?

確認以上四點後,就可以開始展開第 1 章(或你決定的起始章)。
