# 02_Notes — SAM3 架構深解:對話補充筆記

> 對應原始筆記:[[02_sam3_architecture_deep]]
> 系列脈絡:[[01_sam3_lora_overview]] → **本份** → [[03_lora_principles_and_freeze]] → [[04_data_pipeline_class_prompted_box]] → [[05_training_loop_and_slurm]] → [[06_inference_eval_handson_recipe]]
> 索引:[[README]]
> 撰寫日期:2026-05-04(持續更新)

---

## 為什麼有這份筆記

[[02_sam3_architecture_deep]] 已經把五大組件講過,但有幾個概念在原文「點到為止」,這份筆記負責補。目前累積了三個主題:

1. **ViT-L 內部魔改**:RoPE、windowed attention 是怎麼回事
2. **LoRA 注入點抉擇**:7 個旗標為什麼挑這 3 個
3. **YOLO 格式 ICG 資料 → SAM3 finetune 資料**:逐步操作指南

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

## 帶走的重點

1. ✅ **SAM3 ViT-L = 原始 ViT-L + ViT-Det 魔改**(RoPE + windowed/sparse-global attention + FPN-like neck)
2. ✅ **RoPE 用「旋轉 Q/K 向量」注入位置**,自動產生「分數只跟相對位置有關」性質,且沒有可學參數
3. ✅ **Windowed attention 的代價是看不到跨窗 token**,SAM3 用「每 8 層插一次 global 層」解決
4. ✅ **LoRA 7 旗標,ICG stage 1 開 3 個**(vision encoder + detr encoder + detr decoder),理由:domain shift 集中在視覺端,改源頭就要連動下游
5. ✅ **YOLO → SAM3 = 寫個 yolo2coco.py 把每張的 normalized cxcywh 乘回像素 cxcywh,包進 COCO JSON,放到 `{root}/ICG-LC-EAES/{split}/`**;dataset_name 必須是白名單裡的 `ICG-LC-EAES`,bbox_anchor 用 center

---

## 進一步深挖的線索

- ViT-Det:Li et al. 2022, *Exploring Plain Vision Transformer Backbones for Object Detection*,arXiv:2203.16527
- RoPE 原論文:Su et al. 2021, *RoFormer*,arXiv:2104.09864
- 2D/Axial RoPE 應用到 ViT:Heo et al. 2024, *Rotary Position Embedding for Vision Transformer*,arXiv:2403.13298
- Windowed attention 起源:Liu et al. 2021, *Swin Transformer*,arXiv:2103.14030
- LoRA 原論文:Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models*,arXiv:2106.09685

---

## 對話脈絡記錄

本份筆記由 2026-05-04 起對話累積,使用者已主動詢問:
1. SAM3 使用的 ViT-L 跟最原始的 ViT 有什麼不同?什麼是 RoPE 跟 windowed attention?(主題 1、1A、1B)
2. 用 LoRA fine-tune SAM3 有哪些選擇?為什麼選 vision encoder 而不是其他?(主題 2)
3. 用 YOLO 格式 ICG dataset 怎麼當 SAM3 finetune 的資料?要做哪些前處理、資料結構、路徑放哪?(主題 3)

下次對話可從這幾條線延伸:
- ViT-Det 的「plain backbone」哲學(為何放棄 hierarchical 設計如 Swin)
- LoRA rank/alpha 怎麼選定;stage 1 為何選 16/32
- decoder text attention 何時應該打開
- YOLO → COCO 轉換在 segmentation mask(stage 2 用)該怎麼擴充
- 多個 video 來源混合訓練時 image_id 怎麼避免衝突(目前腳本只支援單一來源)

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

### 自己驗證指令(快速複製)

```bash
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
```
