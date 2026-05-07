# 第 2 章 — ConvNeXt 在 CBD v2 中的角色

> 撰寫日期:2026-05-05
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:把 ConvNeXt 在 stage 2 forward pass 中的具體位置畫清楚——它吃什麼、吐什麼、給誰用,以及 v1 / v2 兩個 variant 為何給它不同任務

---

## 兩個 variant 的關鍵差別,一張圖看懂

`src/cbd/model.py` 裡有兩個 stage 2 模型:`CBDV1GlobalPool`(第 41 行)與 `CBDV2Spatiotemporal`(第 120 行)。**它們都用同一個 ConvNeXt-Small,但處理 ConvNeXt 輸出的方式完全不同**:

```
v1_global_pool(較簡單,作為 baseline)
─────────────────────────────────────────
RGB clip (T 張 frame)
   │
   ▼  ConvNeXt-Small
feature map (T, 768, 16, 16)
   │
   ▼  整張 spatial 平均池化(1 個 frame → 1 個向量)
frame embedding (T, 768)
   │
   ▼  + mask embedding (T, 128) → fusion → (T, 256)
   ▼  Temporal Transformer
   ▼  取最後一張 frame 的 embedding
   ▼  Box head → pred_box (4 個座標)


v2_spatiotemporal(主要版本,主訓練設定)
─────────────────────────────────────────
RGB clip (T 張 frame)
   │
   ▼  ConvNeXt-Small
feature map (T, 768, 16, 16)   ★ 不 pool,保留空間
   │
   ▼  + mask feature (T, 128, 16, 16) → 1×1 conv fusion
fused (T, 256, 16, 16)
   │
   ▼  攤平成 (T*16*16) 個 token + 1 個 CLIP token
   ▼  Temporal Transformer(spatial + temporal joint attention)
   ▼  CLIP token → type head(2 類)
   ▼  最後一 frame 的 spatial tokens → box query 抽取 → box head
   ▼  + center cell head + center heatmap head
```

`bsafe_cbd.yaml` 第 45 行 `variant: v2_spatiotemporal` 顯示:**主訓練設定走的是 v2**。v1 留作對照組(baseline)。

**外科類比**:

- v1 像是「只看影片每一秒的縮圖」,把空間細節壓縮成單一印象
- v2 像是「保留每秒的完整影格,連『畫面哪一格出現了什麼』都記得」——時間上的判斷可以結合空間位置

---

## ConvNeXt 在這個 pipeline 的「**唯一**」職責

不論 v1 或 v2,ConvNeXt 都只做一件事:**把 RGB frame 變成 (768, 16, 16) 的特徵地圖**。

它不負責:

- 文字理解(那是 stage 1 SAM3 的事)
- 時序整合(那是 temporal transformer 的事)
- mask 處理(mask 走另一個小型 mask encoder)
- 座標預測(那是 box head 的事)

所以 ConvNeXt 在這個 pipeline 中是「**最純粹的視覺特徵抽取器**」——一個專責的視覺感官模組。

**外科類比**:像是手術中的「**鏡頭+顯示器+影像增強處理**」這一段——它只負責把光學訊號轉成可判讀的影像,不參與決策。決策(切哪裡、夾哪裡)是後面的腦袋(transformer + heads)做的。

---

## 輸入端的細節

### 輸入張量的形狀

從 `src/cbd/engine.py` 第 265 行 `model_output = self.model(batch["rgb"], batch["masks"])` 往回追:

- `batch["rgb"]`:形狀 `(B, T, 3, H, W)`,其中:
  - `B` = batch size(`bsafe_cbd.yaml` 第 59 行 `batch_size: 2`)
  - `T` = clip 長度(`bsafe_cbd.yaml` 第 51 行 `clip_len: 25`)
  - `H = W = 512`(`bsafe_cbd.yaml` 第 48 行 `input_size: 512`)
  - 通道是 RGB 三色

- 每個值是已經 normalize 過的浮點數(用 ImageNet 的 mean / std——`src/cbd/common.py` 第 20-21 行 `RGB_MEAN, RGB_STD`)

### 進入 backbone 之前先 reshape

ConvNeXt 是個影像模型,只看「單張圖」。但我們有 `(B, T, 3, H, W)` 的 5 維張量——怎麼辦?

`src/cbd/model.py` 第 109、208 行的標準作法:**把 batch 跟 time 攤成一維**,當成「B×T 張獨立的圖片」一起餵進 backbone:

```python
rgb_flat = rgb_clip.reshape(batch_size * clip_len, *rgb_clip.shape[2:])
# (B, T, 3, 512, 512) → (B*T, 3, 512, 512)
```

這代表:**ConvNeXt 看 frame 時是「彼此獨立」的,不知道時間關係**。時間關係是後面的 temporal transformer 才開始建模的。

**外科類比**:像是把整段手術影片切成一張張靜態截圖,讓某個專家(ConvNeXt)個別看每一張寫下「這張看到什麼結構」,然後把結構描述彙整給另一個人(temporal transformer)做時間順序的綜合判斷。

---

## 輸出端的細節

### 輸出張量的形狀

ConvNeXt-Small 的 `features` 模組會回傳 `(B*T, 768, 16, 16)`。這是因為:

- 通道數 768 = ConvNeXt-Small 最後一個 stage 的 channels(第 1 章已說明)
- 16×16 = 輸入 512 / stride 32

每個「16×16 中的 1 格」對應原圖一塊 32×32 的像素區塊,通道方向有 768 個浮點數,描述那塊區塊的視覺語意。

### v1 怎麼用這個輸出

`src/cbd/model.py` 第 95-98 行:

```python
def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
    features = self.rgb_backbone.features(rgb)
    features = features.mean(dim=(-2, -1))   # ★ 這行就是 global pool
    return self.rgb_norm(features)
```

`features.mean(dim=(-2, -1))` 把 H 跟 W 維度平均掉,從 `(B*T, 768, 16, 16)` 變成 `(B*T, 768)`。

**意義**:把整張 frame 壓縮成「一個 768 維的描述向量」——這個向量回答的問題是「整張 frame 大概是什麼樣?」,但已經失去「**那個結構在 frame 哪個位置**」的資訊。

### v2 怎麼用這個輸出

`src/cbd/model.py` 第 211 行:

```python
rgb_features = self.rgb_backbone.features(rgb_flat)
# 形狀 (B*T, 768, 16, 16),不做 pool
```

v2 直接保留完整的 16×16 空間網格。這意味著:

- 每張 frame 變成 256 個 token(16×16 = 256)
- 每個 token 對應原圖一個 32×32 像素的區塊
- 通道數 768(後面會被 1×1 conv 投影成 256)

**為什麼 v2 要保留空間網格?**——因為**最終任務是預測 bbox 座標**,模型必須知道「目標出現在 frame 哪個區塊」。如果像 v1 一樣 pool 掉,座標資訊就被平均掉了,只能靠後續模組「腦補」回來。

**外科類比**:

- v1 「整張影像的氣氛感」(像問你「這張看起來是 Calot's triangle 還是肝下緣?」)
- v2 「每塊區域的 16×16 區域標籤地圖」(像問你「左上角看到什麼?中央看到什麼?右下角看到什麼?」)

對「找出 CBD 在哪裡」這種需要精確空間定位的任務,**v2 必勝**。這也是為什麼工程師把 v2 設成預設。

---

## 為什麼 stage 2 不直接用 stage 1 SAM3 的 image encoder?

這是個合理的疑問。stage 1 的 SAM3 已經有一個非常強大的 vision encoder(就是 SAM3 文件裡的 ViT),它甚至已經在內視鏡影像上 fine-tune 過了。為什麼 stage 2 不沿用,而要再放一個 ConvNeXt?

我從程式碼+設計推測有幾個原因(沒有工程師親口說明,但這幾點站得住腳):

1. **記憶體成本**:SAM3 image encoder 是 ViT-Det,光是它就佔很多 GB。v2 一個 batch 要處理 25 張 frame——若再加 SAM3,GPU 直接 OOM。ConvNeXt-Small 只有 50M 參數,記憶體輕得多。
2. **解析度差別**:SAM3 在 stage 1 用 1008×1008(`bsafe_cbd.yaml` 第 41 行)。stage 2 改用 512×512(`bsafe_cbd.yaml` 第 48 行)——這個對 box prediction 來說已經夠精細,但 SAM3 的高解析度設計用不上。
3. **解耦**:stage 1 跟 stage 2 是兩個獨立任務,各自最佳化。stage 1 的 SAM3 已經把它能做的(語意 mask)做完了,結果以 mask 形式傳給 stage 2;stage 2 要的是「**RGB 影像本身的結構特徵**」,跟 stage 1 的 mask 是互補資訊。
4. **stage 1 的 SAM3 也凍結了**:即使 stage 2 想用 SAM3 features,但 SAM3 是 stage 1 訓完後就凍結的,要重新接上 stage 2 訓練 pipeline 很彆扭。直接用一個獨立的、可凍結可微調的 ConvNeXt 反而乾淨。

**外科類比**:像是「分工」——影像增強處理(stage 1 SAM3)做完後輸出處理過的畫面跟標記,後面的判讀模組(stage 2)用「便宜可靠的標準器械」(ConvNeXt)做自己的事,沒必要把昂貴的影像處理機(SAM3)再叫醒一次。

---

## ConvNeXt 跟 stage 1 mask 的「會合點」

現在你已經知道 ConvNeXt 吃 RGB、吐 (768, 16, 16)。但 stage 2 還有一個重要輸入:**stage 1 SAM3 推論出來的 mask**。

從 `bsafe_cbd.yaml` 第 33-36 行:

```yaml
stage1_sam3:
  easy_prompts:
    - gallbladder
    - liver
```

stage 1 用這兩個 prompt 跑出 2 個 mask,壓縮存進 `easy_masks/...../masks.npz`。stage 2 讀進來時是 `(T, 2, H, W)` 的 2 通道 mask(2 個 channel = gallbladder mask + liver mask)。

這 2 通道 mask 會走另一個小型 `mask_encoder`(也是 CNN,但很小很簡單),抽出 mask 的 representation,然後**跟 ConvNeXt 抽出的 RGB feature 在 channel 維度 concat**,過 1×1 conv 融合。

詳細的融合機制是**第 4 章的主題**。第 2 章你只需要記得這個座標系:

```
RGB clip ──► ConvNeXt-Small ──► (T, 768, 16, 16) ─┐
                                                  ├─► concat ─► 1×1 conv ─► (T, 256, 16, 16)
mask clip ─► mask_encoder ────► (T, 128, 16, 16) ─┘                          │
                                                                              ▼
                                                                       Temporal Transformer
```

---

## 這一章你需要帶走的重點

1. ConvNeXt 在 CBD v2 的職責是「**單純的 RGB 視覺特徵抽取器**」,別的事情都不做
2. 兩個 variant(`v1_global_pool` / `v2_spatiotemporal`)都用 ConvNeXt-Small,但**處理輸出的方式相反**:v1 整張 pool 成向量、v2 保留 16×16 空間網格
3. 主訓練設定走 **v2**;v1 是 baseline 對照
4. 進入 backbone 前要把 `(B, T, 3, H, W)` reshape 成 `(B*T, 3, H, W)`——backbone 看 frame 是「彼此獨立」的,**時間關係留給 temporal transformer 處理**
5. v2 保留空間網格的根本原因:**最終任務要預測 bbox 座標**,空間位置不能 pool 掉
6. 為何不沿用 SAM3 image encoder:GPU 記憶體、解析度設計差別、stage 1/2 解耦
7. ConvNeXt feature 不是直接給 transformer,中間會跟 stage 1 的 mask feature 融合(第 4 章主題)

---

## 進一步深挖的線索

- 想知道 reshape `(B*T, …)` 的記憶體成本,可以用 `torch.cuda.memory_allocated()` 在 forward 前後監控
- 對 v1 vs v2 的「能不能找到 box 位置」做 ablation:把 v2 改回 pool,看 box error 變多少(這是 ablation study,不是教學重點)

---

## 對話脈絡記錄

- **2026-05-05**:第 2 章重點是把 ConvNeXt 在 forward pass 中的位置畫出來。延伸思考:v1 用 mean pool 完全丟空間資訊,box head 只能靠「整體 frame 描述」反推座標——理論上應該明顯比 v2 差;這也回答了「為什麼工程師預設 v2」。

---

## 程式碼速查總表

### v1 與 v2 各取什麼形狀的 feature

| variant | 取 feature 的呼叫 | 後續處理 | 最終形狀(進融合前) |
|---|---|---|---|
| v1 | `model.py:96` `self.rgb_backbone.features(rgb)` | `model.py:97` `features.mean(dim=(-2, -1))` global pool | `(B*T, 768)` |
| v2 | `model.py:211` `self.rgb_backbone.features(rgb_flat)` | 不做 pool | `(B*T, 768, 16, 16)` |

### Reshape 與重組空間

| 步驟 | 檔案行號 | 程式碼 | 說明 |
|---|---|---|---|
| Batch+Time 攤平 | `model.py:109` (v1) | `rgb_flat = rgb_clip.reshape(batch_size * clip_len, *rgb_clip.shape[2:])` | (B,T,3,H,W) → (B*T,3,H,W) |
| 同上(v2) | `model.py:208` | 同樣的 reshape | |
| v2 空間 token 攤平 | `model.py:217` | `fused.view(B, T, d, H, W).permute(0, 1, 3, 4, 2)` | (B*T,d,H,W) → (B,T,H,W,d) |
| v2 全域 token 化 | `model.py:223` | `fused.reshape(B, T*H*W, d_model)` | 攤成 (B, T*256, 256) 餵給 transformer |

### 輸入端的張量規格(從 config 確定)

| 項目 | 來源 | 值 |
|---|---|---|
| batch_size | `configs/bsafe_cbd.yaml:59` | 2 |
| clip_len | `configs/bsafe_cbd.yaml:51` | 25 |
| input_size | `configs/bsafe_cbd.yaml:48` | 512 |
| RGB normalize | `src/cbd/common.py:20-21` | ImageNet mean/std |
| mask 通道數 | `bsafe_cbd.yaml:33-36` 兩個 prompt | 2(gallbladder + liver) |

### 快速驗證命令

```bash
# 確認 engine 怎麼把 batch 餵給 model
grep -n "self.model(" src/cbd/engine.py

# 確認 dataset 回傳的 keys
grep -n "return {" src/cbd/dataset.py

# 用 python 印出 ConvNeXt-Small 對 (1, 3, 512, 512) 的輸出形狀
python -c "
import torch
from torchvision.models import convnext_small
m = convnext_small().eval()
m.classifier = torch.nn.Identity()
x = torch.randn(1, 3, 512, 512)
print('feature map shape:', m.features(x).shape)
"
```

第三個命令會印出 `feature map shape: torch.Size([1, 768, 16, 16])`——印證 stride 32 + 768 通道的事實。
