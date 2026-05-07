# 第 1 章 — ConvNeXt 架構基本

> 撰寫日期:2026-05-05
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:在進入 stage 2 細節之前,先用最少的字搞懂「ConvNeXt 是什麼」「為什麼選 Small」

---

## 一句話定位

**ConvNeXt 就是「用 Transformer 設計理念重新打磨過的 CNN」**——它仍然是 CNN(卷積神經網路),但每個設計細節都被重新檢視、向 Transformer 看齊。

這個 backbone 是 Meta AI 在 2022 年發表的(論文:*A ConvNet for the 2020s*)。在那之前,大家以為「Vision Transformer (ViT) 一定打敗 CNN」;ConvNeXt 出來後告訴大家:**只要 CNN 認真現代化,效能可以跟 ViT 平起平坐,但運算更輕、訓練更快、部署更簡單**。

**外科類比**:像是腹腔鏡器械的演進——傳統「直線剪刀」(老式 CNN / ResNet)被 ViT 這種「機械臂式器械」取代後,有人重新設計了「現代直線剪刀」(ConvNeXt),保留器械原本好上手、便宜、可靠的優點,但加入新的人體工學與材質。最終結果:在很多情境下不輸機械臂,還更實用。

---

## 它跟 ResNet、Swin Transformer 的關係

這三個東西是同一條時間線上的「親戚」:

| 模型 | 年份 | 一句話描述 |
|---|---|---|
| **ResNet** | 2015 | 經典 CNN。引入 residual connection,解決深層網路訓練困難。 |
| **Swin Transformer** | 2021 | 視窗化的 Vision Transformer。在影像辨識任務上首次明確贏過 ResNet。 |
| **ConvNeXt** | 2022 | 「把 Swin 的設計哲學搬回 CNN」——逐項對齊,看 CNN 還能多強。 |

ConvNeXt 論文做的事很學術但也很有趣:他們從一個標準的 ResNet-50 開始,**一條一條套用** Swin 的設計細節:換成更大的 kernel、減少 activation、用 LayerNorm 取代 BatchNorm、調整 stage 比例……每改一條就跑一次 ImageNet。最後他們證明,光是這些「小改」累積起來,CNN 就能追上 Swin。

**對你的意義**:ConvNeXt 是個「特別好懂的現代 backbone」——觀念上跟 ResNet 一脈相承(都是 CNN),不必先學會 Transformer 才能用。

---

## ConvNeXt 的內部結構(極簡版)

ConvNeXt 跟 ResNet 一樣是「**金字塔結構**」:影像進去後,空間尺寸逐步縮小、通道數逐步增加。

```
輸入 image  (3, 512, 512)
   │
   ▼  stem (4×4 conv, stride 4)
Stage 1:  (96,  128, 128)   ← 解析度 1/4
   │
   ▼  downsample (2×2 conv, stride 2)
Stage 2:  (192,  64,  64)   ← 解析度 1/8
   │
   ▼  downsample
Stage 3:  (384,  32,  32)   ← 解析度 1/16
   │
   ▼  downsample
Stage 4:  (768,  16,  16)   ← 解析度 1/32  ★ 這個就是 stage 2 用的 feature map
   │
   ▼  global average pool + classifier
ImageNet 1000 類分數
```

幾個關鍵數字記住就好:

- **4 個 stage**,每過一個 stage 解析度減半、通道數加倍
- **總 stride = 32**,所以 input 512×512 → output 16×16
- **最後一層通道數 = 768**(這是 Small 版本的)
- 最後本來會接「全域平均池化 + 分類器」做 ImageNet 1000 類分類——但本專案會把分類器拿掉(下一章詳述)

**外科類比**:像是內視鏡的影像處理 pipeline——原始 1080p 影像進來後,系統會逐層做「縮圖+特徵抽取」,最後得到一張小但資訊濃縮的「特徵圖」,給後端演算法判讀。

---

## 4 個 stage 各做什麼

不需要記得很細,但理解「越深層特徵越抽象」就夠:

- **Stage 1(早期)**:邊緣、紋理、亮度梯度。視野小,類似「鏡頭剛掃過時看到的局部紋路」。
- **Stage 2**:小型局部結構(器械尖端、組織皺褶、鏡面反光)
- **Stage 3**:中型語意(器械本體、解剖區域片段)
- **Stage 4(深層)**:整體語意(這是膽囊?是肝臟?場景在哪個術式階段?)

**本專案實際使用**:CBD v2 只取最深的 **Stage 4** 輸出(`(B*T, 768, 16, 16)`)——這是已經高度語意化的 feature map。stage 1 SAM3 推論的 mask 也會在這個尺寸跟 ConvNeXt feature 融合。

---

## Tiny / Small / Base / Large 怎麼選

ConvNeXt 有 4 個官方尺寸,差別純粹是「**更大版 = 更多層 + 更多通道**」:

| 版本 | 參數量 | 最後通道數 | ImageNet Top-1 (大約) | GPU 記憶體吃量 |
|---|---|---|---|---|
| ConvNeXt-Tiny  |  29M | 768 | 82.1% | 最少 |
| **ConvNeXt-Small** |  50M | 768 | 83.1% | 中等 ★ 本專案用這個 |
| ConvNeXt-Base  |  89M | 1024 | 83.8% | 較多 |
| ConvNeXt-Large | 198M | 1536 | 84.3% | 大量 |

**為什麼工程師選 Small 而不是其他三個?**(這是我從程式碼+config 反推的合理推論,沒有工程師親口說明)

我猜有 3 個理由:

1. **這是 stage 2 的 backbone,不是 stage 1**——stage 1 已經有 SAM3(極大模型)做語意分割,stage 2 不需要再扛一個超大 backbone,有「中等就好」的特徵抽取能力即可。
2. **clip_len = 25**——一個 batch 要同時處理 25 張 frame,backbone 越大、單 batch GPU 吃量爆炸越快。Small 是 GPU 記憶體與表達力的合理 trade-off。
3. **Small 與 Base 的最後通道數一樣是 768**(只差在中間 block 數量)——換成 Base 不會讓下游的 fusion / transformer 維度變動,但訓練成本上升不少。在「下游模組才是創新主角」的前提下,backbone 用 Small 是合理選擇。

**外科類比**:就像選腹腔鏡的 trocar 大小——5 mm / 10 mm / 12 mm 各有適用場合。對「要塞進更多器械、但又不能傷組織」的情境,大家通常選中等的 10 mm。Small 在 ConvNeXt 家族就是這個位置。

---

## 它在這個專案中的「外觀」(不是內部細節)

下一章會詳細講 ConvNeXt 在 CBD v2 forward pass 中的具體流向。這裡只先建立**最外層的輸入輸出印象**,讓你知道後續討論時的座標系:

```
        ┌──────────────┐
RGB   ─▶│              │─▶ feature map
(3,512,512) │ ConvNeXt-Small│   (768, 16, 16)
            │  (預訓練自    │
            │  ImageNet)   │
            └──────────────┘
              ▲
              │
    分類器被換成 nn.Identity()
    (不做 ImageNet 分類,只取特徵)
```

關鍵點:

- **輸入**:單一 frame `(3, 512, 512)`,但實際使用時會把 batch 跟 time 維度攤平成 `(B*T, 3, 512, 512)` 一起餵
- **輸出**:`(768, 16, 16)` 的特徵地圖——把整張 frame 壓縮成 16×16 個「結構化的特徵向量」,每個向量描述對應位置的 32×32 像素區塊
- **權重來源**:torchvision 的 `ConvNeXt_Small_Weights.DEFAULT`(ImageNet-1K 預訓練)
- **修改**:本專案把分類器(原本接 1000 類 softmax)換成 `nn.Identity()`,不分類、只提供 feature

---

## 為什麼可以拿 ImageNet 預訓練的權重直接用在內視鏡影像?

這是一個你大概會問的問題。簡短回答:**因為 backbone 學到的低層特徵(邊緣、紋理、形狀梯度)是跨領域通用的**。

外科類比:就像影像辨識的「基本功」——辨認「邊緣在哪」「組織紋理長什麼樣」這些基本能力,不管你看的是「貓的照片」還是「腹腔鏡的肝門」,本質上需要的視覺基礎是一樣的。ImageNet 預訓練讓 backbone 早就把這些基本功練到一定水準,本專案只需要「微調最後幾層」讓它適應內視鏡場景就好。

這也是為什麼 stage 2 預設 backbone 凍結策略是 `last_stage`(只解凍 stage 4)——前 3 個 stage 的「基本功」直接沿用,只重訓最深層的「語意層」讓它認識膽囊/肝/總膽管這些醫療類別。**第 3 章會詳細展開凍結策略**。

---

## 這一章你需要帶走的重點

1. ConvNeXt = 「用 Transformer 設計理念重新打磨過的 CNN」,2022 Meta 提出
2. 結構是 4 個 stage 的金字塔,stride 32,Small 版最後通道數 768
3. 本專案選 **Small** 不是 Tiny / Base / Large,這是 backbone 表達力與計算成本的平衡點
4. ConvNeXt 在這個專案的職責很單純:**把 RGB frame 轉成 (768, 16, 16) 的特徵地圖**,後續模組才有東西吃
5. 預訓練權重來自 ImageNet-1K——靠的是「低層視覺特徵跨領域通用」這個 transfer learning 的基本原理
6. 分類器被 `nn.Identity()` 換掉了,代表只用 backbone 的特徵抽取能力,不用它的分類能力

---

## 進一步深挖的線索

只在你想深入時看,**不是必修**:

- 原論文:Liu et al., *A ConvNet for the 2020s*, CVPR 2022(`stage1_SAM3+LoRA_Fine-tuning/docs/` 沒放這篇,有興趣再 arXiv 查)
- ConvNeXt 與 Swin Transformer 的逐項比較表(原論文 Table 5)
- Modern CNN 與 ViT 的計算量/精度 trade-off 圖(看 ConvNeXt 為何在「中等規模」打贏 ViT)

---

## 對話脈絡記錄

- **2026-05-05**:使用者要求從 ConvNeXt 起手 stage 2 教學;確認章節風格「精簡淺白」「以實際運行版本為主」「每章長度向 stage 1 看齊」。git log 顯示 `src/cbd/model.py` 只有一個 initial commit,所以教學版本就是現行版本。
  - 延伸思考:Tiny / Small / Base / Large 的選擇沒在 codebase 註解中說明,本章用「stage 2 不需要更大 backbone」+「clip_len=25 的 GPU 限制」+「最後通道數同 Base」三點反推

---

## 程式碼速查總表

> 這一章是觀念章,程式碼引用最少。具體 forward pass 行為留到第 2 章。

### ConvNeXt 在本專案被建立的位置

| 檔案 | 行號 | 作用 |
|---|---|---|
| `src/cbd/model.py` | 8 | `from torchvision.models import ConvNeXt_Small_Weights, convnext_small` —— 從 torchvision 直接 import,不是自刻 |
| `src/cbd/model.py` | 53-54 | `weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None`<br>`self.rgb_backbone = convnext_small(weights=weights)` —— v1 建立 backbone |
| `src/cbd/model.py` | 55 | `self.rgb_backbone.classifier = nn.Identity()` —— 拿掉 ImageNet 分類器 |
| `src/cbd/model.py` | 56 | `self.rgb_dim = 768` —— ConvNeXt-Small 最後通道數 |
| `src/cbd/model.py` | 133-135 | v2 同樣建立 backbone(複製 v1 的三行) |
| `configs/bsafe_cbd.yaml` | 46 | `backbone_name: convnext_small` —— config 標明 backbone 名稱 |
| `configs/bsafe_cbd.yaml` | 47 | `pretrained: true` —— 用 ImageNet 預訓練 |
| `configs/bsafe_cbd.yaml` | 48-49 | `input_size: 512` / `image_size: 512` —— 輸入解析度 |

### 快速驗證命令

```bash
# 確認 ConvNeXt-Small 的 import 路徑(在 cbd_v1/ 下)
grep -n "convnext\|ConvNeXt" src/cbd/model.py

# 確認哪些 config 用了 convnext_small
grep -rn "convnext" configs/

# 用 Python 確認 ConvNeXt-Small 的最後通道數(無需訓練,純結構檢查)
python -c "from torchvision.models import convnext_small; m = convnext_small(); print(m.features[-1])"
```

執行第三個命令後,你會看到最後一個 stage 的 `Conv2dNormActivation` 輸出 channels=768——印證程式碼裡 `self.rgb_dim = 768` 的數字從哪來。
