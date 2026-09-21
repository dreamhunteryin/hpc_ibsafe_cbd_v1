# 第 6 章 — Spatiotemporal Transformer 內部解析

> 撰寫日期:2026-05-27
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:把 ConvNeXt 教學(00~05)收尾後的下游模組打開——**fusion 後的 spatiotemporal token 進到 transformer 之後到底發生什麼**、**learnable 的 box_query 怎麼學會「鎖定」CBD**、**CLS token 怎麼變成整段 clip 的全域摘要**。

---

## 這一章的位置

回顧 stage 2 forward pass:

```
RGB clip ──► ConvNeXt-Small ──┐
                              ├─► fusion(1×1 conv)─► +position embedding
mask clip ─► mask_encoder ────┘
   │
   ▼
spatiotemporal token 序列 + CLS token
   │
   ▼  ◄── 這一章的範圍
Temporal Transformer
   │
   ▼
CLS / spatial tokens
   │
   ├─► type_head        → type_logits
   ├─► box_query + box_attention → box_feature ─► box_head → pred_boxes
   ├─► center_cell_head      → center_cell_logits
   └─► center_heatmap_head   → center_heatmap_logits
```

第 5 章把「下游模組的接口」鳥瞰過——CLS token、box_query、5 個 head 的角色。**這一章打開那個 transformer 黑盒子**:joint attention 怎麼運作、為何 bidirectional、box_query 的訓練動力學。

---

## 工程師原文(Stage 2 相關段落)

> For each target frame, a 5-second video clip preceding the annotated frame was sampled at 5 frames per second, yielding 25 RGB frames and corresponding SAM3-derived masks. RGB features were extracted using a ConvNeXt-Small backbone, whereas liver and gallbladder masks were encoded by a shallow convolutional branch and fused with RGB representations. **The fused spatiotemporal features were processed by a lightweight bidirectional transformer.** A clip-level classification head predicted poor versus good CBD fluorescence visualization, **and a learned query regressed the CBD bounding box in the final frame.**

逐句對照本章重點:

| 原文片段 | 本章節 |
|---|---|
| "5-second video clip ... 5 fps ... 25 RGB frames" | §「25 frames clip 的結構」 |
| "lightweight bidirectional transformer" | §「為何 lightweight / bidirectional」、§「joint attention 內部」 |
| "clip-level classification head" | §「CLS token 怎麼吸收整段 clip 語意」 |
| "learned query regressed the CBD bounding box in the final frame" | §「box_query 是怎麼學會鎖定的」 |

---

## 25 frames clip 的結構

`src/cbd/common.py:22, 102-111`:

```python
DEFAULT_CLIP_LEN = 25

def sample_clip_frame_indices(target_frame_id, fps, clip_len=25, target_fps=5):
    step = max(1, int(round(int(fps) / int(target_fps))))
    end_idx = int(target_frame_id)
    start_idx = end_idx - step * (int(clip_len) - 1)
    return list(range(start_idx, end_idx + 1, step))
```

意義:取標注 frame 的「**過去 5 秒**」並以 5 fps 重採樣,得到 **25 個 frame**(`5 秒 × 5 fps = 25`)。最後一個 frame 才是被標注的目標 frame,前面 24 個是「**手術行為的歷史脈絡**」。

**外科類比**:你要判斷「現在 Calot's triangle 看起來怎麼樣」,不會只看現在這一秒——你會回想「**過去 5 秒外科醫師做了什麼**(夾、拉、剝離?)」。模型也是,它需要看時間上下文才能判斷此刻 CBD 在哪。

token 形狀演進:

```
25 frames × ConvNeXt 編碼 → (B, 25, 768, 16, 16)  ← spatial grid 16×16
                ├ fusion ─► (B, 25, 16, 16, 256)
                ├ flatten ─► (B, 25*256, 256) = (B, 6400, 256)
                └ + CLS ──► (B, 6401, 256)
```

**6401 個 token** 一次進 transformer——這就是「spatiotemporal joint」的物理意義:**所有時間 × 所有空間位置同時參與 attention**。

> input_size=512 → ConvNeXt stride 32 → grid 16×16 = 256 spatial cells per frame。
> `src/cbd/model.py:137` `self.grid_size = max(1, self.input_size // 32)`。

---

## 為何「lightweight」「bidirectional」?

`src/cbd/model.py:169-177`:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=self.d_model,        # 256
    nhead=self.num_heads,        # 8
    dim_feedforward=self.d_model * 4,  # 1024
    dropout=self.dropout,        # 0.1
    batch_first=True,
    norm_first=True,
)
self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)  # 2 層
```

**Lightweight** 的具體含義:

| 超參 | 本專案值 | BERT-base 對比 |
|---|---|---|
| `d_model` | 256 | 768 |
| `num_heads` | 8 | 12 |
| `num_layers` | 2 | 12 |
| `dim_feedforward` | 1024 | 3072 |
| 可訓練參數量 | ~2.5 M | ~110 M |

只有 **2 層**,d_model 縮成 256——刻意做小,因為:

1. **token 數已經很大**(6401),attention 是 O(N²),不能再用大模型疊上去
2. **下游有強訊號的 head**(5 個 head 共同分擔學習任務),transformer 只負責「**讓 token 之間彼此交換上下文**」,不負責直接出答案
3. **訓練資料有限**(臨床標注昂貴),大模型容易過擬合

**Bidirectional** 的具體含義:用的是 `nn.TransformerEncoder`(BERT/ViT 風格),**沒有 causal mask**——每個 token 都能看見所有其他 token。對應 decoder-only / GPT 風格的「只能看左邊」是相反的。

**為什麼要 bidirectional?**——因為這不是「預測下一 frame」的任務,是「**整段 clip 同時看,在最後一 frame 上預測 CBD 位置**」。要讓 frame 25 的 token 能往回看 frame 1~24 的演變;也要讓 frame 1 的 token 能往前看 frame 24 的「結局」——這種互看才能學到時間因果。

**外科類比**:醫師術後 review 一段錄影做標注時,**會反覆前後切換**——「這一刻為什麼這樣?要看下一秒怎麼演變才知道」「這一秒的判斷是不是因為前面 3 秒鋪好的?」。Bidirectional attention 就是這種「**前後雙向看**」的能力。

---

## Joint attention 內部:在 6401 個 token 上怎麼算?

`src/cbd/model.py:223-226`:

```python
spatial_tokens = fused.reshape(batch_size, clip_len * grid_h * grid_w, self.d_model)
clip_token = self.clip_token.expand(batch_size, -1, -1)
tokens = torch.cat([clip_token, spatial_tokens], dim=1)
tokens = self.temporal_transformer(tokens)
```

一次 forward 內每一個 token 都會跟另外 6400 個 token 做 attention(self-attention with no mask)。看每個 token 的「身份」決定了 attention 的物理意義:

| Token 種類 | 索引 | 數量 | 內容 |
|---|---|---|---|
| CLS token | `[0]` | 1 | learnable,初始為 0 |
| spatial token | `[1:]` | 25 × 256 = 6400 | 第 t frame 的第 (r, c) 位置的 ConvNeXt+mask fused 特徵 |

attention 過程拆解:

```
spatial_token[t, r, c] 的 attention key/query/value 來自所有 6401 個 token:

(a) 跨 frame  → 同 (r, c) 的 token 在 t'=1..25      → 學「時間演變」
(b) 跨位置    → 同 t,其他 (r', c')                  → 學「同一刻的空間關係」
(c) 跨時空    → 任意 (t', r', c')                    → 學「不同時刻的不同位置」
(d) 與 CLS    → CLS token 收集所有,並回頭影響所有    → 學「全域語意」
```

**(a) 跨 frame** 是「**這個區塊隨時間怎麼變**」——例如 Calot's triangle 區塊在 5 秒內被夾子推開的演變。
**(b) 跨位置** 是「**同一刻不同器官間的相對關係**」——例如肝臟邊緣往右走會碰到膽囊。
**(c) 跨時空** 是 spatiotemporal 的本體——例如「**前 3 秒右上角出現夾子**,所以**現在中下方一定有 CBD 被夾住**」這種時空因果。

> attention 機制不會「自動」這樣分工——它**理論上能學到**,實際上學到什麼取決於 loss + 資料。下游的 box_l1/giou/center_ce/heatmap_bce/type_ce 5 個 loss 共同把 transformer 推向「**最後一 frame 的空間 token 要能定位 CBD**」、「**CLS token 要能判斷顯影品質**」這兩個方向。

---

## Position embedding 怎麼讓「6401 個 token」有時空感

`src/cbd/model.py:163-165` 三個獨立 position embedding:

```python
self.temporal_position = nn.Parameter(torch.zeros(1, self.clip_len, 1, 1, self.d_model))   # (1, 25, 1, 1, 256)
self.row_position      = nn.Parameter(torch.zeros(1, 1, self.grid_size, 1, self.d_model))   # (1, 1, 16, 1, 256)
self.col_position      = nn.Parameter(torch.zeros(1, 1, 1, self.grid_size, self.d_model))   # (1, 1, 1, 16, 256)
```

forward 中用 broadcasting 加上去(`model.py:218-220`):

```python
fused = fused + self.temporal_position[:, :clip_len]
fused = fused + self.row_position[:, :, :grid_h]
fused = fused + self.col_position[:, :, :, :grid_w]
```

**意義**:transformer 本身是 permutation-invariant(打亂 token 順序輸出相同)。要讓它知道「**這個 token 是第 t frame 的第 (r, c) 位置**」,就把「**第 t 個的 256 維 vector + 第 r 個的 256 維 vector + 第 c 個的 256 維 vector**」加進去。

**為什麼分成 3 個獨立 embedding?**——比起單一 25×16×16=6400 個獨立 embedding,**分解(factorized)** position embedding 有兩個好處:

1. **參數量小**:`25*256 + 16*256 + 16*256 = 14,592` vs 直接 `6400*256 = 1,638,400`——少 100 倍
2. **泛化更好**:模型學到「**時間第 t**」「**行第 r**」「**列第 c**」三個獨立座標系,推論時容易遷移到不同 clip 長度

`temporal_position[:, :clip_len]` 的切片是為了**允許推論時用更短的 clip**(雖然訓練固定 25,但設計上保留彈性)。

---

## CLS token 怎麼吸收整段 clip 語意

`src/cbd/model.py:166, 224-225, 228, 232`:

```python
self.clip_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
...
clip_token = self.clip_token.expand(batch_size, -1, -1)
tokens = torch.cat([clip_token, spatial_tokens], dim=1)
tokens = self.temporal_transformer(tokens)

clip_feature = tokens[:, 0]    # ← CLS token 經過 transformer 後的版本
type_logits = self.type_head(clip_feature)
```

**機制**:

1. 初始 CLS token 是 zeros + 0(沒有 position embedding,因為它「不屬於任何時空位置」)
2. 進 transformer 後,經過 2 層 self-attention,CLS token 「**主動關注所有 6400 個 spatial token**」並把資訊聚合進自己
3. 出來後,`tokens[:, 0]` 就是「**整段 clip 的全域摘要 vector**」
4. 用它丟給 `type_head` 預測 ICG 顯影品質(soft / hard)

**為什麼這個機制 work?**——因為 `type_head` 的 loss(`type_ce`)只回傳給 CLS token 那一條梯度路徑。為了讓 type_head 能正確分類,CLS token 必須**主動學會去 attention 到「對顯影品質判斷有意義的 spatial token」**——例如有 ICG 螢光的區塊、有膽囊形狀的區塊等。

訓練後 attention pattern:CLS token 對 `tokens[:, 1:]` 的 attention 權重會集中在「**對顯影品質有判別力的時空位置**」上,這是 emergent 的行為。

**外科類比**:醫師看完一段 5 秒錄影,寫報告上的「整體顯影品質 = poor」這個結論——他沒有「平均每秒給分再平均」,而是**自動把目光鎖定在最有判別力的瞬間**(例如螢光最亮那一刻),然後下結論。CLS token 學到的就是這種「**全片摘要時自動挑關鍵時刻**」的能力。

---

## box_query 是怎麼學會「鎖定」的(DETR 動力學)

第 5 章已經介紹過 box_query 的「**功能角色**」,這一節**展開它的訓練動力學**——為何一個 zeros 初始化的 vector 最後能學到「自動定位 CBD」?

### 設計回顧

`src/cbd/model.py:167, 178-183, 236-240`:

```python
self.box_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
self.box_attention = nn.MultiheadAttention(
    embed_dim=self.d_model, num_heads=self.num_heads,
    dropout=self.dropout, batch_first=True,
)
...
query = self.box_query.expand(batch_size, -1, -1)         # (B, 1, 256)
box_feature, attention_map = self.box_attention(
    query, last_tokens, last_tokens, need_weights=True,
)
```

注意:`box_query` **不進 temporal_transformer**(它沒有出現在 `torch.cat` 那一行)。它是一個獨立 vector,**只透過 box_attention 跟「最後一 frame 的 256 個 spatial token」cross-attention**。

### 訓練動力學

box_attention 公式:

```
attention_weights = softmax( (box_query · last_tokens^T) / sqrt(d) )   # 形狀 (1, 256)
box_feature       = attention_weights @ last_tokens                    # (1, 256)
```

**反向傳播時**,梯度走哪條路?

1. `pred_boxes = sigmoid(box_head(concat(box_feature, type_feature)))`
2. box loss(l1 + giou)算出 `pred_boxes` 跟 `target_box` 的差距
3. 梯度回傳到 `box_feature`
4. 再回傳到 `attention_weights` 跟 `last_tokens`
5. 最後傳到 `box_query`——**它每一步都更新「該怎麼 query last_tokens 才能把 box_feature 拉到對的位置」**

**為什麼能學到「鎖定」?**:

- 訓練時每個 batch 的 GT box 位置都在變(因為不同 clip 的 CBD 位置不同),但 `box_query` 始終是同一個 vector
- 為了應付各種位置,box_query 必須學會「**依據 last_tokens 的內容,動態生成不同的 attention weight**」
- 經過大量 batch 後,box_query 收斂到「**一個能挑出「CBD 樣的 spatial token」的 query 向量**」

這就是 DETR 的 object query 機制——**不寫死 anchor / proposal**,讓模型自己學「**想找什麼樣的東西**」這個概念。

> 補充:DETR 原本是 N 個 object query(N=100),這裡只用 **1 個**(`(1, 1, 256)`)。因為本任務是「**單目標、單 box**」——一個 clip 只預測一個 CBD bbox,所以一個 query 就夠。

### 為何只 query 最後一 frame?

`src/cbd/model.py:229-230`:

```python
spatial_tokens = tokens[:, 1:].view(batch_size, clip_len, grid_h, grid_w, self.d_model)
last_tokens = spatial_tokens[:, -1].reshape(batch_size, grid_h * grid_w, self.d_model)
```

只取 `spatial_tokens[:, -1]`——也就是第 25 個(最後一個)frame 的 spatial tokens 給 box_attention。

**為什麼?**——任務定義是「**預測標注 frame(也就是最後一 frame)上的 CBD bbox**」,所以 query 對象是最後一 frame 的空間結構。

**但是注意**:那些 `last_tokens` **已經經過 transformer 處理過**,所以它們**已經吸收了前 24 個 frame 的時間上下文**。也就是說,box_query query 的是「**經過 5 秒時間脈絡薰陶後的最後一 frame 空間表徵**」,不是純粹的 ConvNeXt 編碼。

**外科類比**:醫師判斷此刻 CBD 在哪,看的是「**現在這一幀**」,但他的判斷力來自「**過去 5 秒看到了什麼**」——他不會純粹看靜止圖。box_query 從 last_tokens 抽取資訊,但 last_tokens 已經是「**被時間薰陶過的當下**」。

---

## attention_map 是內建的可解釋性訊號

`src/cbd/model.py:237, 250`:

```python
box_feature, attention_map = self.box_attention(query, last_tokens, last_tokens, need_weights=True)
...
attention_map=attention_map.squeeze(1).view(batch_size, grid_h, grid_w),  # (B, 16, 16)
```

`attention_map` 就是 `box_query` 對 256 個 cell 的 softmax 權重——拿出來 reshape 成 16×16 就是個 heatmap。

`infer_cbd.py:46-65, 248-265` 把它畫成 overlay:

```python
def add_heatmap_overlay(image, heatmap, color=(255, 170, 0), max_alpha=144):
    ...
```

**意義**:臨床部署時,光是 bbox 還不夠——醫師會問「**為什麼模型認為 CBD 在這?它是看了哪些區域得出這個結論?**」。`attention_map` 提供了答案——「**模型看了這些位置才下這個 bbox**」。

這是 **post-hoc 解釋(explainability)** 的內建出口。比起 GradCAM 之類事後外掛,這是模型本來就有的 attention 權重,**沒有近似誤差**。

**外科類比**:就像醫師說「**我覺得這裡是 CBD,因為我看到 (a) 膽囊管走向 (b) 肝門 (c) ICG 顯影集中在這條結構上**」——他能講出依據。`attention_map` 是模型版的「**講依據**」。

---

## 為什麼 v2 把 spatial 拆開,v1 卻沒有?

對照 v1 的 forward(`model.py:107-117`)只有一個 256 維的 frame-level embedding 過 temporal transformer。v1 沒有 box_query、沒有 CLS token、沒有 spatial attention。**為什麼會有這種設計差異?**

| 方面 | v1_global_pool | v2_spatiotemporal |
|---|---|---|
| ConvNeXt 輸出處理 | global average pool 成 768 維 | 保留 16×16 spatial grid |
| Token 數 | 25(每 frame 一個) | 6401(1 CLS + 25×256) |
| Attention 模式 | 純 temporal | spatial × temporal joint |
| Box 預測 | 直接從 frame embedding MLP | DETR query cross-attention |
| Auxiliary loss | 無 | center_ce + heatmap_bce + type_ce |

**設計取捨**:

- v1 訓練快、參數少,但**只能學「時間上 box 從哪移到哪」的趨勢**,無法在最後一 frame 上精準定位
- v2 訓練重、token 數爆,但**能從 spatial 結構直接 query 出 box**——這才是「**外科醫師看靜止圖也能指出 CBD**」的能力

v2 預設(`bsafe_cbd.yaml:45`)——因為 CBD 定位需要空間精準度,v1 只是個 baseline。

**外科類比**:v1 像是「**只憑時間趨勢猜下一秒**」——適合動作預測;v2 像是「**看到的當下就能指出**」——適合定位任務。CBD 識別屬於後者,所以 v2 是主力。

---

## 完整 v2 forward 拆解

把所有片段拼起來,`src/cbd/model.py:206-252` 的完整流程:

```
輸入:
  rgb_clip   (B, 25, 3, 512, 512)
  mask_clip  (B, 25, 2, 512, 512)   ← 2 個 channel = liver + gallbladder

Step 1. 攤平時間軸到 batch
  rgb_flat   (B*25, 3, 512, 512)
  mask_flat  (B*25, 2, 512, 512)

Step 2. ConvNeXt + mask encoder
  rgb_features  (B*25, 768, 16, 16)        ← grid_h = grid_w = 16
  mask_flat → interpolate 到 (B*25, 2, 16, 16) → mask_features (B*25, 128, 16, 16)

Step 3. Fusion + 攤平
  cat → (B*25, 896, 16, 16)
  1×1 conv → (B*25, 256, 16, 16)
  view → (B, 25, 16, 16, 256)
  + temporal_position + row_position + col_position
  + LayerNorm

Step 4. 進 transformer
  reshape → spatial_tokens  (B, 25*256, 256) = (B, 6400, 256)
  prepend CLS              → tokens (B, 6401, 256)
  temporal_transformer     → tokens (B, 6401, 256)

Step 5. 拆出 CLS / 取最後 frame
  clip_feature  = tokens[:, 0]           (B, 256)
  spatial_tokens = tokens[:, 1:]         (B, 25, 16, 16, 256)
  last_tokens   = spatial_tokens[:, -1]  reshape→ (B, 256, 256)

Step 6. type_head 路徑
  type_logits  = type_head(clip_feature)            (B, 2)
  type_probs   = softmax(type_logits)
  type_feature = type_conditioner(type_probs)       (B, 256)

Step 7. box_query 路徑
  query        = box_query.expand(B, -1, -1)        (B, 1, 256)
  box_feature, attention_map = box_attention(query, last_tokens, last_tokens)
  box_feature  = box_feature.squeeze(1)              (B, 256)

Step 8. box_head 路徑(concat 兩個來源)
  conditioned  = cat([box_feature, type_feature])    (B, 512)
  pred_boxes   = sigmoid(box_head(conditioned))      (B, 4)

Step 9. center 路徑(從 last_tokens 直接出)
  center_cell_logits    = center_cell_head(last_tokens).squeeze(-1)    (B, 256)
  center_heatmap_logits = center_heatmap_head(last_tokens).squeeze(-1).view(B, 16, 16)

回傳 CBDModelOutput
```

---

## 這一章你需要帶走的重點

1. 6401 個 token = 1 個 CLS + 25 frames × 16×16 spatial tokens——這是 stage 2 的 attention 對象
2. transformer 設定刻意小(2 層 / d_model=256 / ~2.5M 參數),因為 token 數已經很大、下游有強訊號的 head 分擔
3. **bidirectional** 不是 BERT 風格的隨機 mask,是「**沒有 causal mask**」——所有 token 互看,適合「在最後一 frame 上預測」的任務型態
4. position embedding 拆成 **temporal / row / col 三組獨立** vector,參數省 100 倍且泛化更好
5. **CLS token** 經過 transformer 後吸收整段 clip 語意,丟給 type_head 預測 ICG 顯影品質
6. **box_query** 是 learnable 256 維 vector,**不進 transformer**,只透過 cross-attention 從「**最後一 frame 的時空薰陶過 spatial tokens**」抽取 CBD 位置特徵
7. 反向傳播動力學讓 box_query 收斂到「**能挑出 CBD 樣 spatial token 的 query 方向**」——DETR object query 概念
8. `attention_map` 是 box_query 對 256 個 cell 的權重 → reshape 成 16×16 heatmap,就是內建的可解釋性訊號
9. v1 跟 v2 的核心差別:v1 把 spatial pool 掉只看時間,v2 保留 spatial 做 joint attention——CBD 定位需要 spatial 精準度,所以 v2 是預設

---

## 進一步深挖的線索

- **DETR 原始論文**:Carion et al., *End-to-End Object Detection with Transformers*(ECCV 2020)——object query 的開山設計
- **ViT 原始論文**:Dosovitskiy et al., *An Image is Worth 16×16 Words*(ICLR 2021)——CLS token + spatial token 的設計來源
- **TimeSformer / ViViT**:把 ViT 推廣到 video,spatial × temporal joint attention 的設計參考——本專案的 transformer 結構接近 TimeSformer 的 joint divided attention 的 joint 路徑
- **想實際看 attention map**:取一個 checkpoint,跑 `infer/infer_cbd.py --clip-id <id> --split test --output overlay.png`——overlay 圖右半部就是 attention map
- **想實驗 token 數對速度的影響**:把 `model.input_size` 從 512 降到 384,grid 變 12×12,token 數變 1 + 25×144 = 3601(~一半)

---

## 補充問答(2026-05-28):stage 1 SAM-3 的 mask 有貢獻 temporal information 嗎?

外科醫師會問:既然 mask 也是每 frame 都有一張(25 張 mask),它是否也提供「時間上的變化資訊」?

**短答**:mask 本身**不直接**帶 temporal 訊號(mask_encoder 是 frame-by-frame 處理),但 mask 的**時序變化**會被 temporal transformer 觀察到,所以它**間接**貢獻 temporal modeling。

### 證據 1:mask_encoder 沒有 temporal 維度

`src/cbd/model.py:209` (v2):

```python
mask_flat = mask_clip.reshape(batch_size * clip_len, *mask_clip.shape[2:])
```

forward 一開始就**把 T 維度攤平到 batch**——`(B, T, 2, H, W)` → `(B*T, 2, H, W)`。後續 mask_encoder 內部全是 2D conv(`model.py:149-156`),**沒有任何跨 frame 的卷積或 attention**。每張 mask 都被獨立編碼成一張 spatial feature map。

意義:**從 mask 自己單獨來看,模型不知道「frame t 跟 frame t+1 的 mask 怎麼變化」**——這個資訊在 mask_encoder 階段是看不到的。

### 證據 2:temporal 訊號在 fusion 之後才出現

mask 在進 mask_encoder 後變成 128 通道 spatial feature,跟 RGB 在 channel 維度 concat、過 1×1 conv 變成 256 維 token。這些 token 帶上 temporal/row/col position embedding 後,被攤平成 6400 個 spatiotemporal token 進入 transformer。

**transformer 看到的是「25 frame × 256 cell」的時空 token 序列,每個 token 都帶有「該 frame 該 cell 的 mask 訊號」**——所以 joint attention 自然會比較「frame 1 的某 cell」與「frame 25 的同 cell」的 mask + RGB 訊號差異。

### 因此 mask 的「時序變化」確實會被學進去

- 鏡頭推進讓膽囊範圍變大 → mask 的 spatial 分布隨時間擴張 → token 之間的差異被 transformer 觀察
- 器械擋住膽囊 → 某 frame 的 mask 缺失 → token 序列出現「不連續」,transformer 學到那是遮擋
- ICG 顯影逐漸增強 → RGB 變化伴隨 mask 變化 → 兩個訊號的「同步性」被學進去

**這些都不是 mask 自己「產生」的 temporal 資訊,而是「多張 mask 一起被 transformer 看才會浮現的差異」**。

### 一個重要的補充警示:mask 的時序一致性靠不住

SAM-3 是**每一 frame 獨立**推論(沒有 temporal smoothing),所以「mask 的時序變化」**可能來自真實解剖動態,也可能來自 SAM-3 的 per-frame 預測噪聲**:

- frame t 預測膽囊邊界往左 5 像素、frame t+1 往右 3 像素——可能是器械輕微移動,也可能只是 SAM-3 對 ambiguous edge 的隨機抖動
- transformer 沒辦法區分「真實動態」與「預測噪聲」,只能把兩種訊號都當特徵學

意義:**mask 提供的 temporal 訊號是「噪聲很大的 weak signal」**——比 RGB 自己的運動(光流)弱很多。RGB 才是 temporal 訊號的**主要**來源,mask 提供的是「per-frame 解剖位置 anchor」,主要作用仍在 spatial 維度。

### 外科類比

mask = 一連串靜態「解剖標籤照片」,每張都是該 frame 的快照
- 一張一張看 = 靜態標籤,沒有時間資訊
- 25 張連著看 = 標籤的演變(放大、縮小、消失、出現),這才是 temporal signal
- 但這個 signal 不純(SAM-3 抖動會混進去),所以模型主要靠 RGB 抓動態,mask 提供位置 anchor

### 設計上想加強 mask 的 temporal 訊號可以怎麼做?(延伸思考,非當前實作)

- mask_encoder 改成 3D conv(`(T, H, W)` 三維卷積)——讓 mask 自己先做 temporal smoothing
- 或在進 fusion 前用 ConvLSTM 處理 mask 序列
- 或在 stage 1 SAM-3 端就加 temporal consistency loss,讓 mask 本身更穩
- 取捨:都會增加參數量與訓練成本,目前 v2 的設計選擇是「**讓 mask 維持 frame-level、把 temporal 工作全部交給 transformer**」,簡潔且足夠

---

## 對話脈絡記錄

- **2026-05-27**:第 6 章承接 ConvNeXt 教學(00~05),展開 temporal transformer 內部。重點不在「**transformer 是什麼**」(那是通識),而在「**為何選擇這個配置**」「**box_query 學到的東西**」「**joint attention 在 surgical video 上的物理意義**」。
- 工程師原文「lightweight bidirectional transformer」「learned query regressed the CBD bounding box in the final frame」這兩句的具體實作對應到本章 §「為何 lightweight / bidirectional」與 §「box_query 是怎麼學會鎖定的」。
- **2026-05-28**:外科醫師問「stage 1 SAM-3 的 mask 是否也貢獻 temporal information?」——上方「補充問答」段拆解 mask_encoder 的 frame-by-frame 性質、transformer 才產生 temporal 訊號的機制、SAM-3 per-frame 推論帶來的噪聲限制。配對問答在 stage 2 第 4 章(mask 怎麼輔助 bbox 預測)。

---

## 程式碼速查總表

### Transformer 超參(`src/cbd/model.py:121-128`)

| 名稱 | 預設值 | 配置 key |
|---|---|---|
| `clip_len` | 25 | `model.clip_len` |
| `d_model` | 256 | `model.d_model` |
| `num_layers` | 2 | `model.num_layers` |
| `num_heads` | 8 | `model.num_heads` |
| `dropout` | 0.1 | `model.dropout` |
| `dim_feedforward` | 4 × d_model = 1024 | 寫死(`model.py:172`) |

### Token 數計算

| input_size | grid (h, w) | spatial tokens per frame | total tokens |
|---|---|---|---|
| 384 | (12, 12) | 144 | 1 + 25×144 = 3601 |
| 512(本專案預設) | (16, 16) | 256 | 1 + 25×256 = 6401 |
| 768 | (24, 24) | 576 | 1 + 25×576 = 14401 |

stride 32 是 ConvNeXt 設計決定(`src/cbd/model.py:137`),不能改。

### Position embedding 參數量

| 名稱 | 形狀 | 參數量 |
|---|---|---|
| `temporal_position` | (1, 25, 1, 1, 256) | 6,400 |
| `row_position` | (1, 1, 16, 1, 256) | 4,096 |
| `col_position` | (1, 1, 1, 16, 256) | 4,096 |
| 三項合計 | — | 14,592 |
| 假設直接用 (1, 25, 16, 16, 256) | — | 1,638,400 |

分解(factorized)位置編碼少 100 倍參數。

### 快速驗證命令

```bash
# 印出 v2 transformer 與兩個 attention 模組的參數量
python -c "
import yaml
from cbd.model import CBDBoxModel
config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
model = CBDBoxModel(config['model']).impl
for name in ['temporal_transformer', 'box_attention',
             'clip_token', 'box_query',
             'temporal_position', 'row_position', 'col_position']:
    module_or_param = getattr(model, name)
    if hasattr(module_or_param, 'parameters'):
        n = sum(p.numel() for p in module_or_param.parameters())
    else:
        n = module_or_param.numel()
    print(f'{name:25s} {n:>12,}')
"

# 確認 forward 流向
grep -n "tokens = \|clip_token\|box_query\|last_tokens\|box_attention\|type_logits\|center_" src/cbd/model.py | head -30
```

---

> 下一章節主題:**訓練主迴圈 + multi-task loss**——5 個 loss 怎麼加總、optimizer/scheduler/DDP 怎麼設、HPC 提交流程怎麼跑。
