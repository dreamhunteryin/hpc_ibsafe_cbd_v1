# 02_Notes — SAM3 架構深解:對話補充筆記

> 對應原始筆記:[[02_sam3_architecture_deep]]
> 系列脈絡:[[01_sam3_lora_overview]] → **本份** → [[03_lora_principles_and_freeze]] → [[04_data_pipeline_class_prompted_box]] → [[05_training_loop_and_slurm]] → [[06_inference_eval_handson_recipe]]
> 索引:[[README]]
> 撰寫日期:2026-05-04(持續更新)

---

## 為什麼有這份筆記

[[02_sam3_architecture_deep]] 已經把五大組件講過,但有幾個概念在原文「點到為止」,這份筆記負責補。目前累積了四個主題:

1. **ViT-L 內部魔改**:RoPE、windowed attention 是怎麼回事
2. **LoRA 注入點抉擇**:7 個旗標為什麼挑這 3 個
3. **YOLO 格式 ICG 資料 → SAM3 finetune 資料**:逐步操作指南
4. **SAM3 的空間推理能力邊界**:text_encoder 能不能聽懂「在 liver 下方」?多個 bbox 同時餵 geometry_encoder 能不能學相對位置?

---

## 主題 1:SAM3 的 ViT-L vs 最原始的 ViT

### 1.1 原始 ViT 的兩個致命限制(對偵測/分割而言)

教科書版本 ViT(Dosovitskiy et al. 2020)的設定:224×224、patch 16×16、每層 full global attention、absolute positional embedding。對分類任務夠,但對偵測/分割有兩個痛點:

- **解析度被綁死**:換大圖必須重新插值 positional embedding,效果折損
- **patch 一多就跑不動**:1008/14 = 72,72×72 = **5184 個 token**,O(N²) 計算量會把 VRAM 撐爆

→ 所以偵測/分割任務的 ViT 必須有兩個改造:**位置編碼能彈性處理任何解析度**(RoPE)+ **attention 不要每層都 global**(windowed)。SAM3 的 ViT-L 走這條路。

### 1.2 SAM3 的 ViT-L 規格 vs 原始 ViT-L

SAM3 的 ViT 是 **ViT-Det 風格**(Li et al. 2022, *Exploring Plain Vision Transformer Backbones for Object Detection*,arXiv:2203.16527)。

| 參數 | 原始 ViT-L | SAM3 ViT-L | 差別意義 |
|---|---|---|---|
| `img_size` | 224 | **1008** | 4.5 倍邊長,保留手術影像細節 |
| `patch_size` | 16 | **14** | 14 整除 1008,patch 較小 → 細節保留更多 |
| Patch 數 | 196 | **5184** | 26 倍長度,需要 windowed attention 才跑得動 |
| `embed_dim` | 1024 | 1024 | 一樣 |
| `depth` | 24 | **32** | 比原版 L 更深,容量更大 |
| `num_heads` | 16 | 16 | 一樣 |
| `mlp_ratio` | 4.0 | **4.625** | FFN 拓寬一點點(SAM3 的微調) |
| 位置編碼 | absolute(學習式) | **RoPE(2D axial)** | 解析度可變、外推友善 |
| Attention | 每層都 global | **第 7、15、23、31 層 global,其餘 window=24** | 32 層裡只 4 層全局,計算量降到接近線性 |
| 輸出 | 單一尺度 | 經 neck 出 **多尺度 FPN-like 金字塔** | 偵測/分割需要不同尺度 |

**為什麼是 ViT-L 不是 B/H**:L 是「容量 × 計算成本」的甜蜜點。B 不夠承載 SAM3 預訓練、H 太重 VRAM 翻倍但 mAP 提升有限。Meta 在 SAM2/SAM3 系列都選 L。

### 1.3 一句話總結

> **SAM3 的 ViT-L = 原始 ViT-L + ViT-Det 兩項魔改(RoPE 取代 absolute PE、windowed attention 取代全層 global)+ FPN-like neck(多尺度輸出)**。前兩項是為了「能跑得動高解析度」,第三項是為了「能餵下游 transformer」。

---

## 主題 1A:RoPE 是什麼?

### 2.1 一句話心法

> **RoPE = 不在 token 上「加」位置 embedding,而是在 attention 計算時把 Q、K 向量「旋轉」一個跟位置成正比的角度。**

### 2.2 為什麼需要 RoPE

原始 absolute PE:`token_with_pos = patch_embedding + pos_embedding`,有兩個痛點:
- 換解析度就要插值 pos embedding,效果折損
- attention 看到的是「絕對位置」而非「相對距離」,模型必須間接從兩個 absolute embedding 推

### 2.3 RoPE 的做法

把每個 head 的 d 維向量切成 d/2 對(每對 2 維)。對位置 m 的 token,把 Q、K 在每對 2 維平面上旋轉角度 m·θ:

```
對位置 m,  q_rotated = R(m·θᵢ) · Q[2i:2i+2]
        →  k_rotated = R(n·θᵢ) · K[2i:2i+2]

attention score = q_rotated · k_rotated
                = Q · R((n−m)·θᵢ) · K   ← 自然出現「相對位置 (n−m)」
```

**數學上自動得到「分數只跟位置差有關」這個性質**。模型不用學「絕對位置」,直接看「兩個 token 距離多遠」。

### 2.4 為什麼好

| 特性 | absolute PE | RoPE |
|---|---|---|
| 換解析度 | 必須插值 pos embedding | **直接用,只要重算頻率** |
| 注入方式 | 加在 token 上 | **乘進 Q、K 裡**(V 不動) |
| 學什麼 | 位置 embedding 是參數 | **完全沒有可學參數**,純函數 |
| 長 sequence 外推 | 一般 | **強** |

對 SAM3 這種「訓練 1008、推論可能換解析度」的 vision model,RoPE 是天然首選。

### 2.5 SAM3 的 RoPE 是 **2D axial**

影像是 2D 的,所以 RoPE 也要 2D:
- 把 head 維度切成兩半:**前半用 x 座標旋轉,後半用 y 座標旋轉**
- 一個 patch 在 (x=3, y=5) 位置 → Q/K 前半根據 x=3 旋轉、後半根據 y=5 旋轉
- attention score 自然反映「2D 平面上的相對位移」

對應論文:*RoPE-ViT*(Heo et al. 2024)首次把 axial RoPE 系統化用在 vision transformer。

---

## 主題 1B:Windowed Attention 是什麼?

### 3.1 痛點:全局 attention 對 5184 token 太貴

每層 attention 計算量 ≈ N² × d。
- 全局:5184² ≈ 26.9 × 10⁶ 個 dot product / head / layer
- 32 層 × 16 head ≈ **138 × 10⁹ 次 dot product**

→ 訓練時 VRAM 直接爆。

### 3.2 Windowed attention 一句話

> **把 72×72 的 patch 網格切成不重疊的 24×24 小窗,每個窗內各自做 attention,跨窗的 token 看不到彼此。**

```
原本:每個 token 跟全部 5184 個 token attend  → O(N²) = 26.9M
改成:每個 token 只跟同一 24×24 窗內的 575 個 token attend  → O(N·W²) = 5184 × 576 = 2.98M
                                                                    降到原來的 1/9
```

### 3.3 純 windowed 的副作用 + 解法

跨窗 token 永遠看不到對方 → 模型無法捕捉「肝臟左葉與右葉是同一器官」這類**遠距離關聯**。

→ 解法:**夾層設計**(ViT-Det 提出):大部分層用 windowed(便宜)、少數幾層用 global(貴但能跨窗連線),交替使用。SAM3 設定:32 層中第 7、15、23、31 層 global,其他 windowed=24。等同「每打 8 局籃球換一次教練視角」。

### 3.4 RoPE 與 windowed attention 配合的細節

SAM3 的 RoPE 是針對**當前 attention 的座標範圍**算的:
- 在 windowed 層:RoPE 編碼「視窗內位置」(`input_size = window_size × window_size`)
- 在 global 層:RoPE 編碼「整張圖位置」

→ 兩者都是「相對位置」思維,不會互相打架。

---

## 主題 2:LoRA 注入點選擇 — 為什麼是 vision encoder?

### 4.1 七個旗標(LoRA 注入點全集)

`LoRAConfig` 共 7 個獨立開關,每個都對應到 [[02_sam3_architecture_deep]] 五大組件中的一個位置:

| # | 旗標 | 控制什麼 | 對應組件 |
|---|---|---|---|
| 1 | `apply_to_vision_encoder` | ViT-L 內所有 attention 與 FFN 線性層 | 組件 1:Vision Backbone |
| 2 | `apply_to_text_encoder` | CLIP-style text transformer 線性層 | 組件 2:Text Encoder |
| 3 | `apply_to_geometry_encoder` | Box prompt 編碼層 | 組件 3:Geometry Encoder |
| 4 | `apply_to_detr_encoder` | TransformerEncoderFusion 內線性層 | 組件 4 上半 |
| 5 | `apply_to_detr_decoder` | DETR decoder 的 self/cross-attention | 組件 4 下半 |
| 6 | `apply_to_decoder_text_attention` | Decoder 內專門對 text 的 cross-attention | 組件 4 子模塊 |
| 7 | `apply_to_mask_decoder` | UniversalSegmentationHead 的 MLP | 組件 5:Segmentation Head |

理論上 2⁷ = 128 種組合。實務上由設計哲學收斂到少數幾個合理組合。

### 4.2 ICG stage 1 的實際配置

3 個開、4 個關:

| 旗標 | 值 | 結果 |
|---|---|---|
| `apply_to_vision_encoder` | **true** | ✅ 套 LoRA |
| `apply_to_text_encoder` | false | ❌ 完全凍結 |
| `apply_to_geometry_encoder` | false | ❌ 完全凍結 |
| `apply_to_detr_encoder` | **true** | ✅ 套 LoRA |
| `apply_to_detr_decoder` | **true** | ✅ 套 LoRA |
| `apply_to_decoder_text_attention` | false | ❌ 完全凍結 |
| `apply_to_mask_decoder` | false | ❌ 完全凍結 |

### 4.3 為什麼開 vision encoder?(本題核心)

**簡答:domain shift 最大、訊號最弱、效益最高的就是視覺端。**

詳細推理:

#### (a) ICG 螢光影像 vs SAM3 預訓練影像 — 視覺分布天差地別

| 維度 | SAM3 預訓練資料(SA-1B 等) | ICG 螢光腹腔鏡影像 |
|---|---|---|
| 主要色彩 | 全色域 | **以綠色為主**(ICG 在 760-820nm 激發、830nm 發射,經 NIR camera 轉成偏綠的 false color) |
| 對比度 | 高 | **低**(背景組織訊號弱、ICG 訊號集中) |
| 紋理 | 多樣化 | **單一**(器官表面、血管、脂肪、組織液) |
| 形狀分布 | 物體輪廓清晰 | **邊界模糊**(發光擴散、組織液包覆) |

→ 把 SAM3 預訓練的 vision encoder 直接餵 ICG 影像,attention 會被「綠色為主、低對比」這種異常統計帶歪。**vision encoder 必須重新校準**。

#### (b) 從「mask = query · pixel_embedding」反推

```
mask[i, h, w] = sigmoid( decoder_query[i] · vision_pixel_embedding[h, w] )
                                              ↑
                                    這個來自 vision encoder
```

→ 即便 mask decoder 凍結,只要 `vision_pixel_embedding` 改變,mask 自然跟著變。**動 vision encoder = 改變 mask 的「素材」,是最廉價的方法去影響最終輸出**。

#### (c) 投資報酬率

ViT-L 是 SAM3 中最大的元件(約佔總參數 30-40%)。LoRA 注入到這裡:
- trainable 參數還是只有 0.1-0.5%(rank=16)
- 但對「視覺特徵分布」的影響面最廣
- B 矩陣初始化為 0 → 訓練起點等同原模型 → 安全

### 4.4 為什麼**不是**其他六個?

| 旗標 | 為什麼 stage 1 不選 |
|---|---|
| `apply_to_text_encoder` | `"gallbladder"`、`"liver"` 的語意 SAM3 已經學得很好。動它反而會破壞 prompt 識別能力,且 ICG dataset 太小不足以重訓語意 |
| `apply_to_geometry_encoder` | Stage 1 走「文字 prompt 模式」,不餵 box prompt。整條路徑根本沒被使用 |
| `apply_to_detr_encoder` | **是 true**,跟 vision encoder 一起改:vision 特徵變了,fusion cross-attention 必須重新校準 |
| `apply_to_detr_decoder` | **是 true**,理由同上:object query 表徵需對映到 ICG 下的物體分布 |
| `apply_to_decoder_text_attention` | Decoder 內對文字的 cross-attention 是「query 怎麼解讀 prompt」。語意理解部分繼承自 text encoder(已凍結),所以這條子路徑也維持原樣 |
| `apply_to_mask_decoder` | Universal segmentation head 是 object-agnostic 神器,動它風險高、收益小。再加上 ICG 沒有 mask 標註,根本沒監督訊號 |

### 4.5 對照組:`endoscapes_lora.yaml` 為什麼可以開 mask decoder?

唯一差別:**EndoScapes 有 mask 標註,ICG 只有 bbox**。
- 有監督訊號 → 可以打開 mask decoder
- 沒監督訊號 → 必須凍

→ 「開哪些旗標」根本由資料集有什麼標註來決定。

### 4.6 LoRA 哲學:「改中間,不改兩端」

不選 box head / class head 的另一個原因:LoRA 是「對中間表徵做低秩微調」,head 直接產出最終預測,通常用全參數訓練更穩。本專案連 head 都沒額外 unfreeze,因為 `apply_to_detr_decoder=true` 已經改了 query 表徵 → head 看到的輸入分布變了 → 輸出自動跟著變。

---

## 主題 3:YOLO 格式 ICG 資料 → SAM3 finetune 資料(實作步驟)

### 5.1 你手上有什麼 vs SAM3 期待什麼

| 維度 | 你手上的(YOLO 格式) | SAM3 期待的(本 codebase) |
|---|---|---|
| 影像 | `images/xxx.png` 或 `.jpg` | `{root}/{dataset}/{split}/images/xxx.png` |
| 標註 | 每張影像對應一個 `labels/xxx.txt`,每行 `class_id cx_norm cy_norm w_norm h_norm`(都是 0-1 normalized) | 整個 split 共一個 `annotation_coco.json`(COCO 格式) |
| Bbox 單位 | normalized [0, 1] | **像素單位**(SAM3 的 dataset code 自己會 normalize) |
| Bbox 錨點 | YOLO 是 **center**(cxcywh) | ICG-LC-EAES 設定也是 **center**(`bbox_anchor: center`)→ 與 YOLO 一致,**換算只需要乘上原圖尺寸** |
| Class 編號 | 從 0 開始(YOLO 慣例) | 從 1 開始(COCO 慣例,但不強制 — 程式只要求對應一致) |
| Split | YOLO 通常已經有 `train.txt / val.txt / test.txt` 或目錄分好 | 需 `train/`、`val/`、`test/` 三個資料夾,各自一份 `annotation_coco.json` |
| 時間資訊 | 你說「不含」 | SAM3 image 模式**本來也不需要時間資訊**(每張獨立)。stage 1 訓練 = 純圖片任務 |

→ **核心工作 = 寫一個 YOLO → COCO 轉換腳本,並把資料放進 SAM3 期待的目錄結構**。

### 5.2 SAM3 期待的目錄結構(以 ICG 為例)

```
{dataset_root}/                                ← 你選一個本機路徑,例如 /data/luc/sam3_data
└── ICG-LC-EAES/                               ← ⚠ 這個名字是程式硬寫死的白名單之一,不能亂改
    ├── train/
    │   ├── images/
    │   │   ├── frame_0001.png
    │   │   ├── frame_0002.png
    │   │   └── ...
    │   └── annotation_coco.json               ← 整個 train split 一份 JSON
    ├── val/
    │   ├── images/
    │   │   └── ...
    │   └── annotation_coco.json
    └── test/
        ├── images/
        │   └── ...
        └── annotation_coco.json
```

**幾個鐵則:**
1. **`dataset_name` 必須是 `ICG-LC-EAES`**(程式 `SUPPORTED_DATASETS` 白名單,改其他名字會直接 raise ValueError;真要新增名字得動 codebase,違反 [[CLAUDE]] 唯讀原則)
2. **三個 split 都要有 `annotation_coco.json`**,即使 test 沒有標註也得放一份(content 可以是空 annotations + 完整 images)
3. **影像副檔名** PNG/JPG/JPEG 都可以,只要 PIL 能讀就行(`array_to_pil_rgb` 會處理灰階、RGBA 等多種輸入)
4. **`file_name`** 在 JSON 裡只寫檔名(不含資料夾路徑),程式會自動到 `{split}/images/` 找

### 5.3 COCO JSON 的最小規格(SAM3 必要欄位)

```json
{
  "images": [
    {
      "id": 1,                    // ← 不重複的整數;後續 annotation 用這個對應
      "file_name": "frame_0001.png",
      "width":  1920,             // ← 原圖寬高(像素)
      "height": 1080
    },
    ...
  ],
  "annotations": [
    {
      "id":          1,           // ← 不重複的整數(annotation 自己的 id)
      "image_id":    1,           // ← 對應 images.id
      "category_id": 1,           // ← 對應 categories.id
      "bbox":        [cx, cy, w, h],   // ← 像素單位、center 錨點(因為 ICG-LC-EAES 是 center)
      "area":        w * h,
      "iscrowd":     0,
      "segmentation": []          // ← 空 list 就可以(YOLO 沒 mask、stage 1 也不需要)
    },
    ...
  ],
  "categories": [
    {"id": 1, "name": "gallbladder"},
    {"id": 2, "name": "liver"}
  ]
}
```

**注意四件事:**
- **`bbox` 是像素單位、center 錨點**(對應 `bbox_anchor: center`)。YOLO 的 normalized cxcywh 要乘回去:`cx_pixel = cx_norm * orig_w`、`cy_pixel = cy_norm * orig_h`、`w_pixel = w_norm * orig_w`、`h_pixel = h_norm * orig_h`
- **`category_id` 從 1 開始**:YOLO 從 0 開始,你要在轉換時 `+1`
- **`name` 必須是 SAM3 文字 prompt 識別得出的詞**:`"gallbladder"`、`"liver"`(英文小寫)。中文或自創縮寫會讓 text encoder 編出語意不準的向量
- **`segmentation` 給 `[]` 而不是 `null`**:程式用 `if segmentation:` 判斷,空 list 跟 None 都會被跳過,但空 list 比較安全

### 5.4 步驟一覽(從 YOLO 到能跑訓練)

```
Step 1  確認 YOLO 資料的 split 分法(train/val/test 各幾張)
Step 2  決定本機 dataset_root 路徑,建好 ICG-LC-EAES/{train,val,test}/images/ 6 個資料夾
Step 3  把影像複製到對應的 images/(或用 symlink 省空間)
Step 4  寫一個 yolo2coco.py,逐 split 把 .txt 轉成 annotation_coco.json
Step 5  驗證:用 pycocotools 讀檢查 + 隨機抽一張畫 bbox 看對不對
Step 6  改 cbd_v1/configs/icglceaes_lora.yaml 的 dataset_root 指向本機路徑
Step 7  跑 smoke test(本機 1-epoch),確認 dataloader 能正常餵資料
Step 8  正式上 HPC 訓練(若有 GPU 資源)
```

### 5.5 Step 1-3:目錄與影像就位

假設你的 YOLO 資料現在長這樣:

```
~/icg_yolo/
├── images/
│   ├── train/  (1000 張)
│   ├── val/    (200 張)
│   └── test/   (200 張)
├── labels/
│   ├── train/  (1000 個 .txt)
│   ├── val/
│   └── test/
└── classes.txt   (內容:第 0 行 gallbladder、第 1 行 liver)
```

決定一個本機根目錄(範例用 `/data/luc/sam3_data`),建立目標結構:

```bash
DATASET_ROOT=/data/luc/sam3_data
mkdir -p $DATASET_ROOT/ICG-LC-EAES/{train,val,test}/images
```

複製或 symlink 影像(symlink 省硬碟):

```bash
for split in train val test; do
    ln -s ~/icg_yolo/images/$split/* $DATASET_ROOT/ICG-LC-EAES/$split/images/
done
```

> **Tip**:用 symlink 而非 copy,省空間,且改原始檔不用重複。但 HPC 跨檔系統時要 copy。

### 5.6 Step 4:寫 yolo2coco.py(本機放在 `cbd_v1/tutorial/scripts/yolo2coco.py` 即可,**這是新增檔不違反唯讀規則**;或者放在你個人 `~/scripts/`)

下面是一份可以直接執行的草稿(每個區塊都註解過,可逐段跑):

```python
#!/usr/bin/env python3
"""
YOLO format → COCO JSON (for SAM3 ICG-LC-EAES finetuning).

Input layout:
    {yolo_root}/images/{split}/*.png        # 影像
    {yolo_root}/labels/{split}/*.txt        # 每張影像一個 .txt,每行 'cls cx cy w h'(normalized 0-1)
    {yolo_root}/classes.txt                  # 每行一個類別名,順序對應 cls id(從 0 開始)

Output:
    {dataset_root}/ICG-LC-EAES/{split}/annotation_coco.json
"""

import argparse
import json
from pathlib import Path
from PIL import Image


def load_class_names(classes_txt: Path) -> list[str]:
    with open(classes_txt) as f:
        return [line.strip() for line in f if line.strip()]


def yolo_line_to_pixel_cxcywh(line: str, img_w: int, img_h: int) -> tuple[int, list[float]]:
    """Parse 'cls cx_norm cy_norm w_norm h_norm' → (cls, [cx_px, cy_px, w_px, h_px])."""
    parts = line.strip().split()
    cls_id = int(parts[0])
    cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
    return cls_id, [cx_n * img_w, cy_n * img_h, w_n * img_w, h_n * img_h]


def build_split_coco(
    yolo_root: Path,
    split: str,
    class_names: list[str],
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> dict:
    images_dir = yolo_root / "images" / split
    labels_dir = yolo_root / "labels" / split

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_exts
    )

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            # COCO id 從 1 開始,YOLO id 從 0 開始 → 轉換時 +1
            {"id": idx + 1, "name": name} for idx, name in enumerate(class_names)
        ],
    }

    ann_id = 1
    for img_id, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,    # 只放檔名,不放路徑
            "width": w,
            "height": h,
        })

        # 找對應的 .txt
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            # 沒標註的影像也保留(視為「這張沒物體」),include_negatives=true 時會當負樣本
            continue

        with open(label_path) as f:
            for line in f:
                if not line.strip():
                    continue
                cls_id, bbox_pixel = yolo_line_to_pixel_cxcywh(line, w, h)
                cw, _, bw, bh = bbox_pixel
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id + 1,        # YOLO 0-indexed → COCO 1-indexed
                    "bbox": bbox_pixel,                # [cx, cy, w, h] 像素、center 錨點
                    "area": bw * bh,
                    "iscrowd": 0,
                    "segmentation": [],                # YOLO 沒 mask
                })
                ann_id += 1

    return coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-root", required=True, type=Path,
                        help="YOLO root containing images/, labels/, classes.txt")
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="SAM3 dataset root (will create ICG-LC-EAES/ inside)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    class_names = load_class_names(args.yolo_root / "classes.txt")
    print(f"Classes: {class_names}")

    for split in args.splits:
        out_dir = args.dataset_root / "ICG-LC-EAES" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        coco = build_split_coco(args.yolo_root, split, class_names)
        out_path = out_dir / "annotation_coco.json"
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"[{split}] {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations → {out_path}")


if __name__ == "__main__":
    main()
```

執行:

```bash
python yolo2coco.py \
    --yolo-root    ~/icg_yolo \
    --dataset-root /data/luc/sam3_data \
    --splits train val test
```

預期輸出範例:

```
Classes: ['gallbladder', 'liver']
[train] 1000 images, 1820 annotations → /data/luc/sam3_data/ICG-LC-EAES/train/annotation_coco.json
[val]   200  images,  365 annotations → /data/luc/sam3_data/ICG-LC-EAES/val/annotation_coco.json
[test]  200  images,  370 annotations → /data/luc/sam3_data/ICG-LC-EAES/test/annotation_coco.json
```

### 5.7 Step 5:驗證資料正確性(很重要)

寫一個小腳本,隨機抽 5 張畫 bbox 看對不對:

```python
import json, random
from PIL import Image, ImageDraw
from pathlib import Path

DATASET_ROOT = Path("/data/luc/sam3_data/ICG-LC-EAES")
SPLIT = "train"

with open(DATASET_ROOT / SPLIT / "annotation_coco.json") as f:
    coco = json.load(f)

img_id_to_meta = {im["id"]: im for im in coco["images"]}
ann_by_img = {}
for ann in coco["annotations"]:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

sampled = random.sample(coco["images"], 5)
for meta in sampled:
    img_path = DATASET_ROOT / SPLIT / "images" / meta["file_name"]
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for ann in ann_by_img.get(meta["id"], []):
        # bbox 是 center 錨點 → 畫 box 要先換成 topleft
        cx, cy, w, h = ann["bbox"]
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 12), cat_id_to_name[ann["category_id"]], fill="red")
    img.save(f"/tmp/check_{meta['id']}.png")
    print(f"Saved /tmp/check_{meta['id']}.png with {len(ann_by_img.get(meta['id'], []))} boxes")
```

→ 打開 5 張 PNG,看 bbox 是否準確包住器官。如果偏移 → 多半是「center vs topleft」搞反了,回去檢查 `yolo_line_to_pixel_cxcywh()`。

### 5.8 Step 6:改 config 指向你的本機路徑

打開 `cbd_v1/configs/icglceaes_lora.yaml`(這是 [[CLAUDE]] 唯讀規則允許讀但不能改的檔)。

→ **不要直接改既有 config**,而是**複製一份新的**(例如 `icglceaes_lora_local.yaml`)放在 `cbd_v1/configs/` 或你個人目錄,改動以下幾欄:

```yaml
data:
  dataset_root: /data/luc/sam3_data       # ← 你的本機路徑
  dataset_name: ICG-LC-EAES                # ← 不能改
  annotation_file: annotation_coco.json    # ← 不能改(預設值)
  train_split: train
  val_split: val
  test_split: test
  include_negatives: true
  bbox_anchor: center                       # ← 不能改(YOLO cxcywh 對應 center)
  class_names:
    - gallbladder
    - liver

# 其餘 lora、training、output 段落維持不動
```

> **嚴格說來,新增 config 算是「修改既有目錄結構」嗎?**
> [[CLAUDE]] 規定不可修改既有檔案,但**新增**檔案是允許的。新 config 是新檔,沒違反規則。但保險起見,放在 `cbd_v1/tutorial/configs/icglceaes_lora_local.yaml`(自己的 tutorial 目錄底下),最不會跟工程師原版衝突。

### 5.9 Step 7:Smoke test(確認資料管線通)

最簡單的驗證方法:

```bash
cd /mnt/d/Camma_project/BSAFE_code_luc/cbd_v1
python -c "
from sam3.data.endoscapes import CammaSam3Dataset
ds = CammaSam3Dataset(
    dataset_root='/data/luc/sam3_data',
    dataset_name='ICG-LC-EAES',
    split='train',
    selected_class_names=['gallbladder', 'liver'],
    include_negatives=True,
    augment=False,
    resolution=1008,
    bbox_anchor='center',
)
print('len:', len(ds))
sample = ds[0]
print('queries:', [q.query_text for q in sample.find_queries])
print('image shape:', sample.images[0].data.shape)
print('objects per image:', len(sample.images[0].objects))
"
```

預期輸出類似:

```
len: 1000
queries: ['gallbladder', 'liver']
image shape: torch.Size([3, 1008, 1008])
objects per image: 2
```

如果這通了,代表資料管線 OK。下一步才是真正跑訓練(需要 GPU,且不在本筆記範圍 — 看 [[05_training_loop_and_slurm]])。

### 5.10 常見坑

| 症狀 | 原因 | 解法 |
|---|---|---|
| `Unsupported dataset_name=...` | `dataset_name` 不在白名單 | 必須用 `ICG-LC-EAES` |
| `Missing COCO json: ...` | 路徑不對或 JSON 沒生成 | 確認 `{root}/{name}/{split}/annotation_coco.json` 存在 |
| 訓練 loss 不收斂、bbox 偏移半個物體大小 | center vs topleft 搞反 | 檢查 `bbox_anchor: center`,且 JSON 寫的是像素 cxcywh |
| `Unknown class names: ['gallbladder']` | category 名稱大小寫不對 | 程式比對前會 lowercase,所以 JSON 裡寫 `"Gallbladder"` 也 OK,但保持小寫最穩 |
| 推論時 prompt `"gallbladder"` 找不到東西 | text encoder 沒 fine-tune,但你用了奇怪的詞 | 用標準英文器官名(gallbladder、liver、cystic duct)比較穩 |
| 影像讀取慢 | symlink 跨檔系統 | HPC 上改成 copy,或用 fast storage |

### 5.11 為什麼這個流程能運作 — 把關鍵概念串起來

從前兩個主題回顧:
- **SAM3 是 class-prompted 模型**(主題 2 + [[04_data_pipeline_class_prompted_box]]):每張影像 + 每個 selected category → 一個 `FindQueryLoaded`,query_text 是類別名,object_ids 是該類在這張影像的 box list
- **YOLO 標註 = 每行 (class, cxcywh)**,本質就是「這張影像裡這個類別有這個 box」,跟 SAM3 期待的 supervision 結構**完全對應**
- **轉換 = 重新打包成 COCO JSON 容器**,不丟資訊也不加資訊

→ 沒有時間資訊不是問題,因為 stage 1 是純圖片任務,**時間連續性是 stage 2 (`infer_video_lora.py` 用 SAM3 video tracker) 才用到的**(見 [[06_Notes_inference_eval_handson_recipe]])。

---

## 主題 4:SAM3 的空間推理能力邊界

### 4.1 兩個直觀問題的短答

把直覺與實際對照清楚:

| 問題 | 答案 |
|---|---|
| text_encoder 能不能理解「common bile duct, 位於圖像中 liver 下方、gallbladder 右方,且幾乎不可能與這兩者重疊」這種空間語句? | ❌ **幾乎不能** |
| geometry_encoder 同時餵多個 bbox,訓練時會自動學到「相對位置」嗎? | ❌ **架構支援資訊流通,但訓練目標沒獎勵這件事**,所以不會自然學到 |
| 那有辦法讓 SAM3 理解 spatial constraint 嗎? | ⚠️ 有,但要動 codebase 結構或換模型,**不是 ICG-LC-EAES stage 1 的路線**(見 4.4) |

下面分四節展開。

---

### 4.2 為什麼 text_encoder 不擅長 spatial language

#### 4.2.1 SAM3 的 text encoder 是 CLIP-style — 訓練資料的限制

SAM3 的 text encoder 屬於 CLIP-style(`VETextEncoder`,跟 OpenAI CLIP 同源)。這類文字編碼器的訓練資料是**網路爬下來的圖文對**(SAM3 用的是 Meta 內部更大規模的版本,但對「空間語句」的訓練分布跟 CLIP 一樣稀疏)。對比學習(contrastive learning)的目標是「**把圖跟描述拉近**」,主要受益的是:

| 語意類別 | 學得如何 | 例子 |
|---|---|---|
| **名詞** | 學得很好 | `gallbladder`、`liver`、`scissors`、`grasper` |
| **形容詞** | 還可以 | `green`、`shiny`、`small`、`fluorescent` |
| **空間介詞與相對關係** | **學得很差** | `below`、`right of`、`overlapping with`、`between A and B` |

→ 這是 CLIP 文獻有名的 weakness。代表 benchmark:
- **Winoground**(Thrush et al. 2022, arXiv:2204.03162):測 vision-language model 對「組合性語意」的理解,CLIP 系列接近隨機
- **VL-CheckList**(Zhao et al. 2022):分項測試,空間關係是 CLIP 表現最差的子類

→ 直白說:CLIP 看到 `"a cat on a mat"` 跟 `"a mat on a cat"` 給出的 embedding 幾乎一樣。這個弱點是訓練 paradigm 帶來的,不是參數量問題。

#### 4.2.2 即便 text encoder 編出某種「下方」的方向感,下游也不會用

退一萬步,假設 text encoder 真的能把 `"below liver"` 編成一個帶方向資訊的 256 維向量,SAM3 後續模組仍然**不會用**,原因有二:

1. **接收端是「文字 token 跟 vision token 做 cross-attention」**:attention 是「query 跟 key 的 dot product」 — **沒有任何機制告訴模型「這個 token 代表的是空間限制」**。它只會被當成另一個語意 token,跟其他 token 一起被 query 拉去看。沒有 verifier、沒有 spatial filter。
2. **DETR-style decoder 的 N=200 個 query 沒有「驗證限制」的能力**:它們各自獨立預測 box+mask+score,**沒有「如果 query A 預測在某位置就過濾 query B」的這種跨 query 推理**。Hungarian matcher 也是逐 query 對 GT 匹配,完全沒有「組合限制」的概念。

→ 簡單說:**SAM3 的設計是「prompt = 我想找什麼」,不是「prompt = 物體要符合什麼限制」**。把限制塞進 prompt 字串,模型不會解析,只會把它當一團雜訊和 `gallbladder` 一起 mush 起來。

#### 4.2.3 實證觀察與 prompt 設計實務

在 ICG-LC-EAES 這種小 dataset 上 fine-tune,即便把 prompt 改成 `"common bile duct below liver right of gallbladder"`,效果幾乎跟單純 `"common bile duct"` 沒差(甚至可能更差,因為長 prompt 在 BPE 後變成多 token,平均下來反而稀釋了 `"common bile duct"` 的訊號)。

→ **prompt 設計實務**:
- **簡短、單一名詞、英文小寫**:`gallbladder` / `liver` / `cystic duct`
- 這也是工程師 config 跟 training data 的選擇原因
- 不要嘗試把任何空間限制、否定句、條件子句塞進 prompt
- 同義詞建議跟訓練時的 class_names 一致(`gallbladder` 比 `bile sac` 穩),雖然 text encoder 沒被 fine-tune,但 vision encoder 已經適應了「特定 prompt 字串配 ICG 影像」的對應關係

---

### 4.3 geometry_encoder 餵多個 bbox 能學相對位置嗎?

#### 4.3.1 架構面:資訊「能流通」

看 `geometry_encoders.py` 的 `Prompt` dataclass — `box_embeddings` 形狀是 `(4, num_prompts, 1)`,**架構上支援多個 box**。多個 box 進去後會被編成多個 geo token,跟 text token 一起串接成 `prompt`,送進 `TransformerEncoderFusion`。

關鍵:**在 fusion encoder 裡,多個 geo token 之間會做 self-attention**。從架構面看,「box A 在哪、box B 在哪」這種相對資訊,**有機會**被 attention 機制捕捉到 — 因為 self-attention 計算 dot product 時,Q/K 帶 RoPE 位置編碼,理論上「box A 中心 (100, 200) 跟 box B 中心 (300, 400) 的相對位置 (200, 200)」這種訊號是可以流通的。

→ 「架構是否支援」答案是 yes。

#### 4.3.2 但「訓練時是否真的學到」?

**沒有**。原因是 SAM3 預訓練的 task 設計:

- **SAM 1/2/3 系列的 geometry encoder 從一開始就是 interactive segmentation 的工具**:使用者畫**一個**框,模型 segment 框內的物體。每次 forward 一個 query 對應**一個 box prompt**,模型學的是「在這個框內把該物體的精細邊界切出來」。
- **多個 box 同時餵進去**也是支援的,但訓練資料裡的多 box 場景是「**多個獨立 instance 各自需要 segment**」(例如一張圖有 3 顆蘋果,各畫一框),而**不是**「**用 box A、B 當『環境參考』,求位置約束下的 box C**」。
- **Loss function 不獎勵跨 box 推理**:Hungarian matching 是把每個 query 各自匹配到一個 GT,然後算 box loss(L1 + GIoU)+ mask loss(focal + dice)。**沒有任何 loss 項在懲罰「query A 跟 query B 的相對位置不對」**。

→ 結論:**架構讓資訊流通是「必要條件」,訓練目標獎勵這件事才是「充分條件」**。SAM3 沒有後者,所以即便資訊通過 attention 流動了,也不會凝結成「相對位置 → 約束預測」的能力。

#### 4.3.3 「我自己 fine-tune 時餵多 bbox,會不會在 ICG 上學到相對位置?」

**幾乎不會**,而且風險還很大:

| 風險 | 細節 |
|---|---|
| 沒對應的 loss | 即使 dataloader 餵多 box,Hungarian matcher 仍是逐 query 獨立匹配,模型學不到跨 box 推理 |
| 資料量不夠 | ICG dataset 通常幾百到幾千張,**學「common bile duct 在 liver 下方」這種高階 prior 需要至少數萬個正反樣本對比**,小資料根本撐不起 |
| LoRA 容量不夠 | LoRA 是低秩微調(rank 16-32),設計初衷是「微調已有能力」,**不是「教模型新的推理範式」**。要學 spatial reasoning 是 paradigm shift,LoRA 容量不足 |
| Geometry encoder 在 ICG config 凍結 | 想開也得改 config + 重訓,違反目前唯讀協議 |

→ 直白說:即便你把多 bbox 餵進去,模型只會把它們當成「multiple independent prompts」處理,**不會**自動湧現「common bile duct 必須在某處」這種推理。

---

### 4.4 真正注入空間先驗的四條路徑

如果這個臨床需求很強(對 CBD 來說常見:**它跟 cystic duct、common hepatic duct 形成 Calot 三角,位置高度可預測**),目前學界較常見的做法有四條:

#### 4.4.1 改架構:Multi-task / Multi-class joint detection(本專案 stage 2 走的路)

**同時偵測多個解剖結構,讓 model 內部自然學到它們的相對分布**。這正是 cbd_v1 stage 2 的設計:**ConvNeXt + spatiotemporal transformer 同時看 RGB 影像 + SAM3 anatomical mask(gallbladder / liver),預測 CBD bbox**。位置先驗是**透過 mask 視覺特徵注入**,不是文字。

→ 這是為什麼工程師設計成「stage 1 用 SAM3 切 gallbladder/liver,stage 2 再吃這些 mask 預測 CBD」 — **空間先驗的注入點不是 text encoder,是 stage 2 的視覺輸入**。

#### 4.4.2 改訓練 paradigm:Referring Expression Segmentation

專門設計用「`the duct between gallbladder and common hepatic duct`」這種長句來定位的模型。代表作:
- **GLIP / GLIPv2**(Li et al. 2022, arXiv:2112.03857):open-vocabulary detection 加強空間語句處理
- **GRES**(Liu et al. 2023, arXiv:2306.00968):多 referent 與 no-target 處理(generalized referring expression segmentation)
- **PolyFormer**(Liu et al. 2023, arXiv:2302.07387):referring segmentation 直接 polygon 化

→ 這條路要**換模型**,不是 fine-tune SAM3 能做的。在臨床手術場景目前還沒有公開預訓練 weights,自己從頭訓需要大量 referring expression 標註。

#### 4.4.3 注入 anatomical heatmap 當輸入(spatial prior fusion)

把「典型 CBD 位置的高斯分布 heatmap」當第 4 個 channel 拼到 RGB 輸入,讓 vision encoder 直接看到位置先驗。**簡單但有效**,只是需要重新訓練(LoRA 仍可行,但要重設 input projection,因為 ViT 的 patch embedding 是 3-channel 的)。

→ 適合「**位置高度可預測**」的解剖結構(像 CBD 在 Calot 三角附近),不適合姿勢變化大的器官。

#### 4.4.4 後處理過濾(最務實)

**最務實也最常用**:讓 SAM3 預測完所有候選,再在後處理階段用「gallbladder mask 跟 CBD mask 的相對 centroid 位置」過濾誤報。

```
SAM3 預測 gallbladder mask、CBD candidate mask
        ↓
post-processing(自己寫 .py 腳本):
  - 計算 gallbladder centroid (gx, gy)
  - 計算每個 CBD candidate centroid (cx, cy)
  - 過濾掉 cy < gy(在膽囊上方的 candidate,通常是誤報)
  - 過濾掉跟 gallbladder mask IoU > 閾值的(重疊代表錯位)
  - 過濾掉距離 gallbladder centroid > 閾值的(太遠,不可能是 CBD)
```

→ 這條路**不動模型**,符合本專案唯讀規則,而且效果通常很好。stage 2 的 `compute_cbd_prediction_metrics.py` 跟 inference 後處理都在這個邏輯範圍內。

---

### 4.5 對 CBD pipeline 的實際意涵

把整個 cbd_v1 pipeline 重新看一次,你會發現**整個 pipeline 的設計就是繞過 SAM3 在 spatial reasoning 上的弱點**:

1. **Stage 1(SAM3 + LoRA)只負責「找解剖學上有明確視覺特徵的 anatomy」**:gallbladder、liver 是大且色彩/紋理特徵明顯的器官,SAM3 預訓練 + ICG fine-tune 就能搞定
2. **Stage 1 不直接找 CBD**:CBD 細小、被組織覆蓋、視覺特徵弱、**位置只能由 anatomical prior(在 gallbladder 後下方)推得**,SAM3 不擅長這個
3. **Stage 2(ConvNeXt + spatiotemporal transformer)專門處理 CBD**:吃 RGB + stage 1 的 anatomical mask + 時間序列(25 frames),用視覺 + 時序 + spatial mask 三重信號定位 CBD
4. **空間先驗在 stage 2 是「視覺特徵」而非「文字描述」**:stage 1 mask 直接當 channel 拼接,讓 stage 2 的 backbone 自然學到「CBD 通常出現在 gallbladder 周圍」

→ 工程師設計 two-stage 的核心理由,正是「**SAM3 不會空間推理,所以不要要求它做空間推理;讓它專心找它擅長的 anatomy,空間整合留給 stage 2**」。

---

### 4.6 一張圖總結:能力 vs 任務的對應

```
任務類型                                        SAM3 擅長嗎?    CBD pipeline 怎麼解
────────────────────                            ─────────────    ──────────────────
找視覺特徵清楚的 anatomy
(gallbladder, liver)                            ✅ 擅長            stage 1(SAM3 + LoRA)直接搞定

找視覺特徵弱、靠位置推斷的結構
(common bile duct, cystic duct)                 ❌ 不擅長         stage 2(ConvNeXt + spatiotemporal)
                                                                    吃 stage 1 mask + RGB + 時序

理解 prompt 中的空間限制
("below liver", "between A and B")              ❌ 完全做不到     不要嘗試 — 訊號注入改用 mask 或後處理

跨 instance 推理
(query A 與 B 不能重疊、A 必須在 B 下方)        ❌ 完全做不到     後處理過濾(centroid 距離、IoU)

時序一致性(同一物體跨 frames)
                                                ✅ tracker 擅長  infer_video_lora.py 用 SAM3 官方 tracker
                                                                    propagate(不掛 LoRA,沿用預訓練)
```

---

## 帶走的重點

### 主題 1-3 的 5 件事

1. ✅ **SAM3 ViT-L = 原始 ViT-L + ViT-Det 魔改**(RoPE + windowed/sparse-global attention + FPN-like neck)
2. ✅ **RoPE 用「旋轉 Q/K 向量」注入位置**,自動產生「分數只跟相對位置有關」性質,且沒有可學參數
3. ✅ **Windowed attention 的代價是看不到跨窗 token**,SAM3 用「每 8 層插一次 global 層」解決
4. ✅ **LoRA 7 旗標,ICG stage 1 開 3 個**(vision encoder + detr encoder + detr decoder),理由:domain shift 集中在視覺端,改源頭就要連動下游
5. ✅ **YOLO → SAM3 = 寫個 yolo2coco.py 把每張的 normalized cxcywh 乘回像素 cxcywh,包進 COCO JSON,放到 `{root}/ICG-LC-EAES/{split}/`**;dataset_name 必須是白名單裡的 `ICG-LC-EAES`,bbox_anchor 用 center

### 主題 4 額外的 5 件事

6. ✅ **CLIP-style text encoder 對空間語句近乎無感**;長 prompt 反而稀釋有用名詞訊號,prompt 一定要簡短、單一名詞、英文小寫
7. ✅ **SAM3 的 prompt 設計哲學是「想找什麼」而非「物體要符合什麼限制」**;塞限制進去模型不會解析(沒有 verifier、沒有跨 query 推理)
8. ✅ **geometry_encoder 架構支援多 box,但訓練 paradigm 是 interactive single-box**,Hungarian matcher 逐 query 獨立匹配,沒學跨 box 相對位置;LoRA 容量也不夠重新教
9. ✅ **空間先驗的正確注入點是「視覺特徵或後處理」**(mask 當 channel、後處理 centroid 距離/IoU 過濾),**不是 text/geometry encoder**
10. ✅ **cbd_v1 two-stage 設計就是繞過 SAM3 spatial weakness**:stage 1 找 anatomy(SAM3 擅長),stage 2 用 RGB+mask+時序整合空間 prior 找 CBD(SAM3 不擅長的部分)

---

## 進一步深挖的線索

### 主題 1-3 相關論文

- ViT-Det:Li et al. 2022, *Exploring Plain Vision Transformer Backbones for Object Detection*,arXiv:2203.16527
- RoPE 原論文:Su et al. 2021, *RoFormer*,arXiv:2104.09864
- 2D/Axial RoPE 應用到 ViT:Heo et al. 2024, *Rotary Position Embedding for Vision Transformer*,arXiv:2403.13298
- Windowed attention 起源:Liu et al. 2021, *Swin Transformer*,arXiv:2103.14030
- LoRA 原論文:Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models*,arXiv:2106.09685

### 主題 4(空間推理弱點)相關論文

- CLIP 弱點 benchmark:Thrush et al. 2022, *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*,arXiv:2204.03162
- VL composition 弱點分析:Zhao et al. 2022, *VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations*,arXiv:2207.00221
- Open-vocabulary detection 加強空間語句:Li et al. 2022, *GLIP: Grounded Language-Image Pre-training*,arXiv:2112.03857
- Generalized Referring Expression Segmentation:Liu et al. 2023, *GRES: Generalized Referring Expression Segmentation*,arXiv:2306.00968
- Polygon-based referring segmentation:Liu et al. 2023, *PolyFormer: Referring Image Segmentation as Sequential Polygon Generation*,arXiv:2302.07387

---

## 對話脈絡記錄

本份筆記由 2026-05-04、2026-05-05 對話累積,使用者已主動詢問:
1. SAM3 使用的 ViT-L 跟最原始的 ViT 有什麼不同?什麼是 RoPE 跟 windowed attention?(主題 1、1A、1B,2026-05-04)
2. 用 LoRA fine-tune SAM3 有哪些選擇?為什麼選 vision encoder 而不是其他?(主題 2,2026-05-04)
3. 用 YOLO 格式 ICG dataset 怎麼當 SAM3 finetune 的資料?要做哪些前處理、資料結構、路徑放哪?(主題 3,2026-05-04)
4. SAM3 的 text_encoder 或 geometry_encoder 是否具有物件或 mask 相對位置的理解功能?譬如給 prompt「common bile duct, 位於 liver 下方、gallbladder 右方,且幾乎不可能與兩者重疊」,或同時餵多個 bbox 給 geometry_encoder 訓練,推論時會更精準嗎?(主題 4,2026-05-05)

下次對話可從這幾條線延伸:
- ViT-Det 的「plain backbone」哲學(為何放棄 hierarchical 設計如 Swin)
- LoRA rank/alpha 怎麼選定;stage 1 為何選 16/32
- decoder text attention 何時應該打開
- YOLO → COCO 轉換在 segmentation mask(stage 2 用)該怎麼擴充
- 多個 video 來源混合訓練時 image_id 怎麼避免衝突(目前腳本只支援單一來源)
- 主題 4 延伸:stage 2 的 ConvNeXt + spatiotemporal transformer 如何用 stage 1 mask 當 channel 注入空間先驗(細節留給 cbd_v1 後續筆記)
- 主題 4 延伸:後處理 centroid / IoU 過濾的具體實作(在哪一個推論腳本接最自然?可從 `infer_cbd_clip.py` 的 post-processing 區段看起)
- 主題 4 延伸:若真要讓 SAM3 學跨 box 相對位置,需要怎樣的 loss(例如加入 pairwise position loss)、需要多大資料量、與直接用 referring expression segmentation 模型的取捨

---

## 程式碼速查總表(集中區)

> 前面段落不再夾雜行號,所有程式碼引用集中在這裡。看完概念回頭找位置時用。

### ViT-L 與 ViT-Det 改造

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| ViT-L 規格(`_create_vit_backbone`) | `cbd_v1/src/sam3/model_builder.py` | 57-83 |
| Neck 多尺度輸出(`_create_vit_neck`) | `cbd_v1/src/sam3/model_builder.py` | 86-97 |
| ViT 主類別與 Block | `cbd_v1/src/sam3/model/vitdet.py` | 616-779 |
| Block 內 windowed/global 切換邏輯 | `cbd_v1/src/sam3/model/vitdet.py` | 597-613 |
| `window_partition` / `window_unpartition` | `cbd_v1/src/sam3/model/vitdet.py` | 93-139 |
| ViT 論文出處註解 | `cbd_v1/src/sam3/model/vitdet.py` | 618-621 |

### RoPE 實作

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| 2D 軸向頻率計算(`compute_axial_cis`) | `cbd_v1/src/sam3/model/vitdet.py` | 41-57 |
| 套用 RoPE 到 Q/K(`apply_rotary_enc`,複數乘法) | `cbd_v1/src/sam3/model/vitdet.py` | 68-90 |
| Attention 模塊內呼叫 RoPE(`_apply_rope`) | `cbd_v1/src/sam3/model/vitdet.py` | 459-464 |
| RoPE 通用實作(SAM 子模組) | `cbd_v1/src/sam3/sam/rope.py` | 整檔 |

### LoRA 注入點(主題 2)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `LoRAConfig` 7 旗標定義 | `cbd_v1/src/sam3/lora/lora_layers.py` | 274-371 |
| `should_apply_lora_to_component` 凍結邏輯 | `cbd_v1/src/sam3/lora/lora_layers.py` | 397-431 |
| ICG stage 1 的 7 旗標真實值 | `cbd_v1/configs/icglceaes_lora.yaml` | 19-44 |

### YOLO → SAM3 資料管線(主題 3)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| Dataset 主類別 `CammaSam3Dataset` | `cbd_v1/src/sam3/data/endoscapes.py` | 111-265 |
| `__getitem__` 主流程 | `cbd_v1/src/sam3/data/endoscapes.py` | 166-261 |
| `bbox_anchor` 預設(`ICG-LC-EAES = center`) | `cbd_v1/src/data/dataset_camma.py` | 19-22 |
| 支援的 `dataset_name` 白名單 | `cbd_v1/src/data/dataset_camma.py` | 14-17 |
| COCO JSON 解析(`CammaContext.build`) | `cbd_v1/src/data/dataset_camma.py` | 157-234 |
| `coco_bbox_to_normalized_cxcywh`(根據 anchor 換算) | `cbd_v1/src/sam3/image_utils.py` | (utility) |
| ICG config 全文 | `cbd_v1/configs/icglceaes_lora.yaml` | 1-50 |

### 空間推理弱點(主題 4)

> 主題 4 是「能力邊界」的觀念釐清,不是程式碼操作主題。但要驗證「為什麼 SAM3 不會空間推理」,可以從以下幾個關鍵程式位置看出端倪。

#### text_encoder 是 CLIP-style(主題 4.2)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `VETextEncoder` 主類別(CLIP-style transformer)| `cbd_v1/src/sam3/model/text_encoder_ve.py` | 整檔 |
| BPE tokenizer | `cbd_v1/src/sam3/model/tokenizer_ve.py` | 整檔 |
| BPE vocab | `cbd_v1/src/sam3/assets/bpe_simple_vocab_16e6.txt.gz` | — |
| `forward_text` 包裝(把字串列表編成 language_features) | `cbd_v1/src/sam3/model/vl_combiner.py` | 121-170 |
| Text encoder 在 forward 中如何進入 prompt | `cbd_v1/src/sam3/model/sam3_image.py` | 167-210(`_encode_prompt`)|

→ 重點是看 `text_encoder` 的訓練分布(SAM3 預訓練 + 凍結),以及 `_encode_prompt` 把 txt_feats 跟 vision token 串接的方式 — 沒有任何「verifier 機制」,只有 dot product attention。

#### geometry_encoder 多 box 介面(主題 4.3.1)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `Prompt` dataclass(`box_embeddings: (4, num_prompts, 1)` 支援多 box)| `cbd_v1/src/sam3/model/geometry_encoders.py` | 82-249 |
| `SequenceGeometryEncoder` 主類別(編碼器本體)| `cbd_v1/src/sam3/model/geometry_encoders.py` | 整檔 |
| `_encode_prompt` 中 geo_feats 與 txt_feats 串接後送 fusion encoder | `cbd_v1/src/sam3/model/sam3_image.py` | 188-209 |
| `_get_dummy_prompt`(沒 box prompt 時建空 Prompt)| `cbd_v1/src/sam3/model/sam3_image.py` | 493-499 |
| `forward` 中 `geometric_prompt` 從 `find_input.input_boxes` 構造(stage 1 = 空) | `cbd_v1/src/sam3/model/sam3_image.py` | 522-526 |
| ICG config 凍結 geometry_encoder(`apply_to_geometry_encoder: false`) | `cbd_v1/configs/icglceaes_lora.yaml` | 19-44 |

→ 「架構支援多 box」從 `Prompt.box_embeddings` 的形狀 `(4, num_prompts, 1)` 看得很清楚;但「stage 1 訓練/推論預設都餵空」從 `_get_dummy_prompt` 跟 `infer_lora.py:103-132`(沒填 input_boxes)印證。

#### 訓練 loss 沒有跨 box 推理項(主題 4.3.2)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| Hungarian matching(逐 query 獨立匹配) | `cbd_v1/src/sam3/model/sam3_image.py` | 549-552(`_compute_matching`)|
| Matcher 主類別(實作逐 query 一對一匹配) | `cbd_v1/src/sam3/model/matcher.py` | 整檔 |
| Loss 計算入口(L1 + GIoU + focal/dice,**沒有 pairwise position loss**) | `cbd_v1/train/train_lora.py` | (loss 組合區段,看 Trainer 主迴圈) |
| 訓練 config 的 loss 權重(`use_mask_loss: false` for ICG) | `cbd_v1/configs/icglceaes_lora.yaml` | 1-50 |

→ 重點在 `_compute_matching` 跟 matcher 的設計:**每個 query 各自配一個 GT,沒有「query A 跟 query B 的相對位置」這種 loss 項**。要驗證「SAM3 沒被獎勵學跨 box 推理」,看這裡。

#### 真正注入空間先驗的位置(主題 4.4)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| Stage 2 整合 stage 1 mask 的 entry(若 codebase 有)| `cbd_v1/configs/bsafe_cbd.yaml` | 29-42(`stage1_sam3` 段落)|
| Stage 2 訓練腳本(用 stage 1 mask 當輸入的關鍵設計)| `cbd_v1/train/train_cbd.py` | 整檔 |
| Stage 2 推論 + 後處理(centroid/IoU 過濾邏輯若有)| `cbd_v1/infer/infer_cbd_clip.py` | 整檔 |
| Detection metrics(IoU、mAP — 後處理用同一套邏輯)| `cbd_v1/compute_cbd_prediction_metrics.py` | 128-232 |

→ 注意 stage 2 的細節留給後續筆記(本份只到 stage 1 架構)。但**「空間先驗在 stage 2 是視覺特徵而非文字描述」** 這件事,看 `bsafe_cbd.yaml` 的 `stage1_sam3` 設計 + stage 2 model 怎麼吃 mask 就能驗證。

### 自己驗證指令(快速複製)

```bash
# 主題 1-3
# ViT-L 規格
sed -n '57,83p'   cbd_v1/src/sam3/model_builder.py
# RoPE
sed -n '41,90p'   cbd_v1/src/sam3/model/vitdet.py
# Windowed attention 切換
sed -n '93,139p'  cbd_v1/src/sam3/model/vitdet.py
sed -n '597,613p' cbd_v1/src/sam3/model/vitdet.py
# LoRA 旗標
sed -n '274,371p' cbd_v1/src/sam3/lora/lora_layers.py
sed -n '20,44p'   cbd_v1/configs/icglceaes_lora.yaml
# Dataset 主類別 + bbox_anchor 設定
sed -n '111,261p' cbd_v1/src/sam3/data/endoscapes.py
sed -n '14,22p'   cbd_v1/src/data/dataset_camma.py

# 主題 4(空間推理弱點驗證)
# text encoder + 進入 prompt
sed -n '121,170p' cbd_v1/src/sam3/model/vl_combiner.py
sed -n '167,210p' cbd_v1/src/sam3/model/sam3_image.py
# geometry encoder 多 box 介面
sed -n '82,249p'  cbd_v1/src/sam3/model/geometry_encoders.py
sed -n '493,499p' cbd_v1/src/sam3/model/sam3_image.py
sed -n '522,526p' cbd_v1/src/sam3/model/sam3_image.py
# Hungarian matching 逐 query 獨立(沒有跨 box loss)
sed -n '549,568p' cbd_v1/src/sam3/model/sam3_image.py
# Stage 2 注入點(空間先驗的真正去處)
sed -n '29,42p'   cbd_v1/configs/bsafe_cbd.yaml
```
