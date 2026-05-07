# 第 5 章 — 從 ConvNeXt 交棒到 Temporal Transformer

> 撰寫日期:2026-05-05
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:**收尾 ConvNeXt 教學**——把 fusion 後的特徵怎麼丟給 temporal transformer 講清楚,並預告下一章節「spatiotemporal transformer 內部」會展開的內容

---

## 這一章的位置

ConvNeXt 教學到第 4 章其實就講完了——backbone 的職責、凍結策略、跟 mask 的融合都已經拆解過。但 stage 2 的 forward pass 沒結束:fusion 後的特徵還要進 **temporal transformer**,然後才走多個 head 出最終預測。

這一章的目的是**讓你理解 ConvNeXt 的「下游接口」**——它丟出來的東西長什麼樣、被誰接走、那個下游模組大概在做什麼。詳細的 transformer 內部運作會在 stage 2 的下一個章節主題展開,**這裡只給接口層面的鳥瞰**。

---

## v1 跟 v2 的交棒方式對比

### v1:25 個 frame embedding 進 transformer

`src/cbd/model.py` 第 78-93 行 + 107-117 行:

```
fusion 完成後 (B, T, 256)  ─ 加 temporal position embedding
   │
   ▼  Temporal Transformer(2 層,8 個 head)
   ▼
(B, T, 256)
   │
   ▼  取最後一個 frame 的 embedding [:, -1]
   ▼  Box head: 3 層 MLP → sigmoid → (B, 4)
pred_box (cx, cy, w, h)
```

v1 的 transformer 看的是「**25 個 frame-level 描述**」——每個 token 是一張 frame 的整體 embedding,token 之間的差異就是「時間上的演變」。它做的是**純 temporal attention**(沒有空間)。

最終只取**最後一個 frame** 的 embedding 做 box 預測——意思是「整段 clip 整合後,輸出最後一刻 CBD 在哪」。

### v2:1 個 CLS token + T×16×16 個 spatial token 進 transformer

`src/cbd/model.py` 第 169-183 行 + 217-243 行:

```
fusion 完成後 (B, T, 16, 16, 256)  ─ 加 temporal/row/col position embedding
   │
   ▼  攤平成 token 序列 (B, T*16*16, 256)
   ▼  在最前面 prepend 一個 CLS token (B, 1, 256)
最終 token 序列 (B, 1 + T*256, 256)
   │
   ▼  Temporal Transformer(2 層,8 個 head)
   ▼  spatial × temporal joint attention
(B, 1 + T*256, 256)
   │
   ├─► [:, 0]   → CLS token  ──► Type head      → (B, 2)  [soft / hard]
   │
   └─► [:, 1:]  → spatial tokens
        │
        ▼  取最後一 frame 的 spatial tokens (B, 256, 256)
        │
        ├─► box query 透過 cross-attention 抽取  ──► Box head  → (B, 4)
        ├─► center cell head                    ──► (B, 256)
        └─► center heatmap head                 ──► (B, 16, 16)
```

v2 的 transformer 看的是「**全部時空 token 的聯合序列**」——每個 token 是「**第 t frame 的第 (r, c) 區塊**」,token 之間的關係涵蓋時間 + 空間兩個維度。它做的是 **spatiotemporal joint attention**。

---

## v2 的 CLS token 是做什麼的?

`src/cbd/model.py:166`:

```python
self.clip_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
```

這是個 learnable 的 token——形狀 `(1, 1, 256)`,在 forward 時被 broadcast 成 `(B, 1, 256)` 然後 prepend 到 token 序列最前面。

**它的角色**:

- 整段 clip 在 transformer 中流動時,attention 會把所有時空 token 的資訊**匯聚到 CLS token 上**
- 訓練結束後,CLS token 變成「**整段 clip 的全域語意摘要**」
- 用它來做「**clip-level 分類**」——`type_head` 預測 CBD 顯影品質是 soft 還是 hard(2 類)

**外科類比**:像是手術錄影的「**全片摘要鏡頭**」——醫師看完整段手術後,會在報告上寫「整體手術品質 = soft / hard / good / poor」。CLS token 學的就是這種「整段判斷」的概念。

---

## v2 的 box query 是怎麼運作的?

這是 stage 2 最巧妙的設計之一,值得拆解。

`src/cbd/model.py:167`:

```python
self.box_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
```

跟 CLS token 一樣,box_query 也是個 learnable 的 256 維向量。但**它不進 transformer**——它另外用一個 cross-attention 去「**問**」最後一 frame 的 spatial tokens:

`model.py:236-237`:

```python
query = self.box_query.expand(batch_size, -1, -1)
box_feature, attention_map = self.box_attention(query, last_tokens, last_tokens, need_weights=True)
```

這是個典型的 DETR 風格設計:

- `query` = 那個 learnable 的 box_query
- `last_tokens` = 最後一個 frame 的 spatial tokens(也就是 transformer 處理過後的時空 grid)
- 用 cross-attention 讓 box_query「**從 256 個 spatial tokens 中加權抽取自己關心的位置**」

**意義**:訓練後,box_query 學到的就是「**找出 CBD 大致在哪個區域**」的「眼神」——它會自動把 attention 集中在 CBD 周圍的 grid cell,從那些 cell 的 feature 中抽出 CBD 的位置/形狀資訊。

**外科類比**:就像影像識別師看到一張腹腔鏡截圖,**第一眼會自動把目光鎖定 Calot's triangle**——這個「目光鎖定」的能力是訓練出來的,不是寫死的規則。`box_query` 就是這個「目光」,訓練讓它學會去哪看。

額外好處:`attention_map`(`model.py:250`)是 box_query 對 256 個 cell 的權重,**可以視覺化成 heatmap**,讓你看模型「正在看哪裡」——這是個內建的可解釋性工具。

---

## center cell head 與 center heatmap head 是輔助訊號

`src/cbd/model.py:200-201`:

```python
self.center_cell_head = nn.Linear(self.d_model, 1)
self.center_heatmap_head = nn.Linear(self.d_model, 1)
```

這兩個 head 各自把 256 維 spatial token 投到 1 維 logit:

- `center_cell_logits`:形狀 `(B, 256)`,每個 grid cell 一個分數——訓練目標是「**CBD 中心在哪個 cell?**」(類似分類問題)
- `center_heatmap_logits`:形狀 `(B, 16, 16)`,每個位置一個分數——訓練目標是「**CBD 中心的 heatmap(高斯分布)**」

從 `src/cbd/engine.py:120-129`:

```python
if model_output.center_cell_logits is not None and model_output.grid_size is not None:
    center_indices, center_heatmaps = build_center_targets(...)
    center_ce = F.cross_entropy(model_output.center_cell_logits, center_indices)
    heatmap_bce = F.binary_cross_entropy_with_logits(model_output.center_heatmap_logits, center_heatmaps)
```

這兩個 head 的 loss 跟 box loss 一起加總(權重在 `bsafe_cbd.yaml:73-78`):

```yaml
loss_weights:
  box_l1: 5.0
  box_giou: 2.0
  center_ce: 1.0
  heatmap_bce: 1.0
  type_ce: 1.0
```

**為什麼要這兩個輔助 head?**——這是 deep learning 圈常用的「**multi-task auxiliary loss**」技巧:

- **加強訓練訊號**:bbox 預測是回歸任務(連續座標),訊號比較弱;center cell 是分類任務(分到 256 個 grid cell 哪一個),訊號更直接,可以**穩定訓練**
- **強迫 transformer 學空間語意**:如果只訓 bbox,transformer 可能「偷懶」用其他特徵推 box;加上 cell + heatmap loss 後,模型必須**讓每個 grid cell 的 feature 都帶有正確空間位置的資訊**
- **提供 inference 時的備援訊號**:推論時可以同時看 bbox + heatmap,兩個訊號交叉驗證

**外科類比**:像是訓練 fellow 不只看「他預測的切除範圍對不對」,還同時測「他能不能指出關鍵解剖中心」「他能不能畫出該關注的區域熱圖」——多重檢核讓訓練訊號更扎實。

---

## type head 預測什麼?

`src/cbd/model.py:184-189`:

```python
self.type_head = nn.Sequential(
    nn.LayerNorm(self.d_model),
    nn.Linear(self.d_model, self.d_model),
    nn.ReLU(inplace=True),
    nn.Linear(self.d_model, 2),
)
```

吃 CLS token,輸出 2 類 logits。從 `src/cbd/common.py:24-25`:

```python
TARGET_TYPE_ORDER = ("soft", "hard")
TARGET_TYPE_TO_LABEL = {name: index for index, name in enumerate(TARGET_TYPE_ORDER)}
```

兩類:`soft` / `hard`。這是 **CBD 的 ICG 顯影品質分類**——軟性顯影(模糊、間接)vs 硬性顯影(清晰、直接)。

**為什麼要分類顯影品質?**——因為這個資訊**回頭餵給 box head**(`model.py:190-199`):

```python
self.type_conditioner = nn.Sequential(
    nn.Linear(2, self.d_model),
    nn.ReLU(inplace=True),
)
self.box_head = nn.Sequential(
    nn.LayerNorm(2 * self.d_model),     # ★ 注意是 2 * d_model
    nn.Linear(2 * self.d_model, self.d_model),
    ...
    nn.Linear(self.d_model, 4),
)
```

`forward` 中(`model.py:236-240`):

```python
type_probs = type_logits.softmax(dim=-1)
type_feature = self.type_conditioner(type_probs)
...
conditioned_feature = torch.cat([box_feature, type_feature], dim=-1)
pred_boxes = self.box_head(conditioned_feature).sigmoid()
```

意義:**bbox 預測同時看「box_query 從 spatial tokens 抽出的特徵」+「整段 clip 的顯影品質判斷」**——顯影品質會影響 box 預測的精度與信心。

**外科類比**:外科醫師看到 ICG 影像時,**先判斷「螢光顯影品質好不好」,再決定「能多大膽地依賴這個顯影資訊去定位 CBD」**。本專案的 type head 模仿了這個決策流程。

---

## 最終輸出長什麼樣

stage 2 forward 最終回傳 `CBDModelOutput` dataclass(`model.py:14-21`):

```python
@dataclass
class CBDModelOutput:
    pred_boxes: torch.Tensor                       # (B, 4)  cx, cy, w, h(normalized)
    type_logits: torch.Tensor | None = None        # (B, 2)  soft / hard
    center_cell_logits: torch.Tensor | None = None # (B, H*W)
    center_heatmap_logits: torch.Tensor | None = None  # (B, H, W)
    attention_map: torch.Tensor | None = None      # (B, H, W) box_query 的 attention
    grid_size: tuple[int, int] | None = None       # (H, W) = (16, 16)
```

對 v1,只有 `pred_boxes` 是 non-None;其他四個都是 None(`model.py:117`)。
對 v2,五個都有(`model.py:245-252`)。

---

## ConvNeXt 教學在這裡告一段落

回顧整個 stage 2 forward pass(v2 路徑):

```
RGB clip ──► ConvNeXt-Small ──┐
                              ├─► fusion(1×1 conv)─► +position embedding
mask clip ─► mask_encoder ────┘
   │
   ▼
spatiotemporal token 序列 + CLS token
   │
   ▼  ← 你已經理解到這裡
Temporal Transformer(下個章節主題)
   │
   ▼
CLS / spatial tokens
   │
   ├─► type_head        → type_logits
   ├─► box_query + box_attention → box_feature ─► box_head → pred_boxes
   ├─► center_cell_head      → center_cell_logits
   └─► center_heatmap_head   → center_heatmap_logits
```

**ConvNeXt 教學的範圍是「最左邊三個方塊 + fusion」**——backbone 是什麼、做什麼、怎麼凍結、怎麼跟 mask 融合。下個章節主題會展開「**Temporal Transformer 是怎麼處理 spatiotemporal token 的**」,以及「**box_query 為什麼能學到自動定位**」。

---

## 這一章你需要帶走的重點

1. fusion 後的特徵會被送進 **Temporal Transformer**,但 v1 跟 v2 餵的東西不一樣
2. v1 餵 25 個 frame embedding,做純 temporal attention,只取最後 frame 預測 box
3. v2 餵 (1 + T×16×16) 個 token(含 1 個 learnable CLS token),做 spatiotemporal joint attention
4. v2 的 **CLS token** 學「整段 clip 的全域摘要」,用來預測 ICG 顯影品質(`type_head`)
5. v2 的 **box_query** 透過 cross-attention 從 spatial tokens 中「鎖定 CBD 位置」,類似 DETR 設計
6. v2 額外有 **center cell head** 與 **center heatmap head** 作為 multi-task auxiliary loss,穩定訓練
7. type_head 的輸出會反向餵給 box_head——「**先判斷顯影品質,再決定 box 預測**」,模仿外科醫師決策流
8. 最終回傳 `CBDModelOutput` dataclass,v1 只有 pred_boxes、v2 有 5 個欄位都齊全
9. **ConvNeXt 教學在這裡結束**——下一章節主題會深入 transformer 內部

---

## 進一步深挖的線索

- DETR(Carion et al. *End-to-End Object Detection with Transformers*, ECCV 2020):object query 的概念來源
- 為何 multi-task auxiliary loss 有效:Caruana, *Multitask Learning*(1997)的開山論文
- 想看 box_query 真實的 attention map 視覺化,可以利用 `model_output.attention_map`(`model.py:250`)畫 heatmap

---

## 對話脈絡記錄

- **2026-05-05**:第 5 章是 ConvNeXt 教學的「銜接章」——介紹下游 transformer 的接口跟它在做什麼,但**不深入 transformer 內部**(留給下一章節主題)。重點在 box_query / CLS token / type_head → box_head 反饋這三個設計。

---

## 程式碼速查總表

### v1 vs v2 的 transformer 輸入

| 項目 | v1 | v2 |
|---|---|---|
| Token 形狀 | (B, T, 256) | (B, 1 + T*256, 256) |
| 包含 CLS token? | 否 | 是(`model.py:166, 224-225`) |
| Position embedding | `model.py:78` 1D temporal | `model.py:163-165` 3 個(temporal/row/col) |
| Transformer 設定 | 2 層 / 8 head / d_model=256 | 同 v1 |

### v2 五個輸出 head

| Head | model.py 位置 | 輸入 | 輸出形狀 | 對應 loss |
|---|---|---|---|---|
| Type head | 184-189 | CLS token | (B, 2) | `type_ce`(`engine.py:111-114`) |
| Box query attention | 178-183, 237 | 最後 frame 的 spatial tokens | (B, 256) feature | — |
| Box head | 194-199, 240 | concat(box_feature, type_feature) | (B, 4) | `box_l1`+`box_giou`(`engine.py:54-68`) |
| Center cell head | 200, 242 | spatial tokens | (B, 256) | `center_ce`(`engine.py:128`) |
| Center heatmap head | 201, 243 | spatial tokens | (B, 16, 16) | `heatmap_bce`(`engine.py:129`) |

### Output dataclass

| 欄位 | 型別 | v1 | v2 |
|---|---|---|---|
| `pred_boxes` | Tensor (B, 4) | ✓ | ✓ |
| `type_logits` | Tensor (B, 2) | None | ✓ |
| `center_cell_logits` | Tensor (B, H*W) | None | ✓ |
| `center_heatmap_logits` | Tensor (B, H, W) | None | ✓ |
| `attention_map` | Tensor (B, H, W) | None | ✓ |
| `grid_size` | tuple | None | (16, 16) |

### Loss 權重(`configs/bsafe_cbd.yaml:73-78`)

| Loss | 權重 |
|---|---|
| box_l1 | 5.0 |
| box_giou | 2.0 |
| center_ce | 1.0 |
| heatmap_bce | 1.0 |
| type_ce | 1.0 |

box loss 加總後權重 7.0,輔助 loss 加總後 3.0——主任務佔約 70% 訓練訊號。

### 快速驗證命令

```bash
# 印出 CBDV2Spatiotemporal 五個 head 的 trainable 參數量
python -c "
import yaml
from cbd.model import CBDBoxModel
config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
model = CBDBoxModel(config['model']).impl
for name in ['type_head', 'box_attention', 'box_head', 'center_cell_head', 'center_heatmap_head']:
    module = getattr(model, name)
    n = sum(p.numel() for p in module.parameters())
    print(f'{name:25s} {n:>10,}')
"

# 確認 v2 的 forward 流向
grep -n "def forward\|clip_token\|box_query\|type_head\|box_head\|center_cell\|center_heatmap" src/cbd/model.py | head -30
```

---

> 下一階段:**stage 2 的 Temporal Transformer 內部解析**——spatiotemporal joint attention 是怎麼工作的、box_query 為何能學會「鎖定」位置、為何 CLS token 能彙整整段 clip 語意。這些是進入下一章節主題的問題。
