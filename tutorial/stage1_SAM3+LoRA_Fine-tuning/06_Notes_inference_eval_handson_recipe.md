# 06_Notes — SAM3 推論的輸入/輸出 + 「原生 vs 工程師包裝」的拆解

> 對應原始筆記:[[06_inference_eval_handson_recipe]]
> 系列脈絡:[[01_sam3_lora_overview]] → [[02_sam3_architecture_deep]] → [[02_Notes_sam3_architecture_deep]] → [[03_lora_principles_and_freeze]] → [[04_data_pipeline_class_prompted_box]] → [[05_training_loop_and_slurm]] → **本份**(對應 06)
> 索引:[[README]]
> 撰寫日期:2026-05-04(初稿)、2026-05-05(整合主題 1、2)

---

## 為什麼有這份筆記

[[06_inference_eval_handson_recipe]] 已經把三種推論模式的命令列旗標整理過,但對「**SAM3 在實作上到底怎麼把影像吃進去、把什麼東西吐出來**」缺少一個系統的觀念整理。本份筆記四個主題都是針對「容易誤會的觀念」做澄清:

1. **主題 1**:finetune 後的 SAM3 在推論時,**模型本身**到底吃什麼吐什麼?哪一段是 SAM3 原生格式,哪一段是工程師額外加的包裝?
2. **主題 2**:ICG-LC-EAES 標註裡的「類別名稱(class name)」是不是 text_encoder 的輸入?「座標(bounding box)」是不是 geometry_encoder 的輸入?
3. **主題 3**:能否讀一系列圖片直接吐 mask + label?(這份筆記的初稿問題)
4. **主題 4**:能否讀一段影片,每隔 5 秒輸出 image+mask 配對 + 帶 temporal information 的 label?(這份筆記的初稿問題)

本份筆記把所有「程式碼引用、檔案路徑、行號」集中放到最後的「程式碼速查總表」一節,正文盡量用文字解說觀念。

---

## 主題 1:Finetune 後的 SAM3,推論時讀什麼?吐什麼?是原生格式嗎?

### 1.1 一句話結論

> **「SAM3 模型本身」(`Sam3Image.forward()`)的輸入輸出是 SAM3 原生設計;但你執行 `python infer/infer_lora.py ...` 時看到的「指定器官 → mask + label」這個方便的字典結構,是工程師在 SAM3 原生輸出外面包了一層「前處理 + 後處理」做出來的。**

換句話說,把整條 inference pipeline 拆成「輸入端包裝 → SAM3 原生 forward → 輸出端包裝」三段:

| 段 | 是什麼 | 誰寫的 |
|---|---|---|
| 輸入端包裝 | PIL 影像 + prompt 字串 → 標準化張量 + `BatchedDatapoint` | 工程師(`infer_lora.py`) |
| **SAM3 原生 forward** | `BatchedDatapoint` → `pred_logits / pred_masks / pred_boxes / presence_logit_dec` 的 dict | Meta 的 SAM3 codebase |
| 輸出端包裝 | 把 N=200 個 query 過濾、排序、轉回原圖座標 → `[{prompt, detections: [{mask, bbox, score}, ...]}, ...]` | 工程師(`infer_lora.py`) |

理解這個分層很重要,因為:
- 你以後若要「批次處理一系列圖片」(主題 3)、或「拆出來嵌進 stage 2 pipeline」,要動的都是**外層包裝**,不是 SAM3 模型本身
- 你看到的最終輸出格式不是 SAM3 論文中描述的東西,是工程師為了下游方便使用而設計的字典結構

### 1.2 SAM3 模型「原生」吃進來的是什麼

進到 `Sam3Image.forward()` 的東西是 `BatchedDatapoint`,內含:

| 欄位 | 內容 | 訓練 vs 推論 |
|---|---|---|
| `img_batch` | 影像 batch,形狀 `(B, 3, 1008, 1008)`、float、值域 `[-1, 1]` | 兩者一樣 |
| `find_text_batch` | 去重後的 prompt 字串列表,例如 `["gallbladder", "liver"]` | 兩者一樣 |
| `find_inputs` | 每個 query 的索引資訊(text_ids、img_ids、可選的 input_boxes) | 兩者一樣 |
| `find_targets` | GT box / mask / object_ids,**用於計算 loss 的 supervision target** | 訓練用,推論不用 |
| `find_metadatas` | 後處理對回原圖用的中介資料 | 推論用較多 |

→ 這是 SAM3 codebase 設計的標準輸入容器,不是工程師另外發明的。

關鍵觀念:**模型不直接吃到「gallbladder」這個字串本身**。字串會在 collator 階段彙整去重(`find_text_batch`),然後在 forward 內第一步就被 text_encoder 編成 `language_features`(`(L, N, 256)` 的張量),之後 query 用 `text_ids` 索引去拿對應的 256 維向量。詳見主題 2。

### 1.3 SAM3 模型「原生」吐出來的是什麼

`Sam3Image.forward()` 回傳的是一個包裝在 `SAM3Output` iterator 裡的多 stage 多 step 結構,**最內層**是一個 dict,核心欄位:

| key | 形狀 | 意義 |
|---|---|---|
| `pred_logits[i]` | `(N_queries,)` | 第 i 個 prompt 的 N_queries(預設 200)個 query 各自的「跟此 prompt 匹配程度」分數 |
| `pred_masks[i]` | `(N_queries, H, W)` | 同上 N 個 query 的 mask logits(尚未 sigmoid)|
| `pred_boxes[i]` | `(N_queries, 4)` | 同上 N 個 query 的 normalized cxcywh box |
| `presence_logit_dec[i]` | scalar | 「這個 prompt 在影像中是否存在」的整體 logit(可選)|

關鍵觀念:**SAM3 的 forward 不直接告訴你「哪個 box 是膽囊」**。它每個 prompt 都吐 200 個候選 query,每個 query 都附 (mask, box, score)。**收斂到實際 detection 是後處理的工作**。這個 200 的數字、N_queries 結構、DETR 風格的 query 設計,都是 SAM3 原生。

### 1.4 工程師在外面包了什麼?(`infer_lora.py` 的兩端)

#### 輸入端包裝

- `preprocess_image()`:把任意尺寸 PIL 影像縮成 1008×1008、normalize 到 [-1, 1]
- `flatten_prompts()`:把 `--prompt gallbladder --prompt liver` 命令列攤平成 `["gallbladder", "liver"]`
- `build_inference_datapoint()`:每個 prompt 建一個 `FindQueryLoaded`,包成 `Datapoint`,**注意 stage 1 預設不傳 box prompt**(對應主題 2 的 geometry_encoder 觀念)
- `collate_fn_api(...)`:跟訓練同一個 collator 把 `Datapoint` 變成 `BatchedDatapoint`

→ 這層的目的就是「把使用者方便給的東西(PIL 影像 + 字串列表),轉成 SAM3 原生 forward 吃的容器」。

#### 輸出端包裝

進 `run_inference()` 後 `Sam3Image.forward()` 回傳了 200 個 query,工程師接著做:

1. **`filter_predictions()`** — 從 200 個 query 收斂到「真的 detection」
   - 用 sigmoid + presence logit 算每個 query 的 confidence score
   - mask 二值化(sigmoid > 0.5)
   - 跑 mask-NMS(IoU 閾值移除重疊 instance)
   - 用 prob_threshold 過濾分數太低的
   - 取 top-k(`max_detections`)
2. **`build_detection()`** — 把保留下來的每個 detection 從「模型內部尺寸」轉回原圖
   - mask 從 1008×1008 縮回 `(orig_h, orig_w)` 並轉 binary
   - normalized cxcywh box 轉成原圖像素 xywh
   - 拼成 `{mask, score, box_norm_cxcywh, bbox_xywh}` 字典
3. **`run_inference()` 的尾段** — 為每個 prompt 包裝成最終結構

最終 `predictions` 是這樣的列表(這就是工程師「自創」的便利結構):

```python
predictions = [
    {
        "prompt": "gallbladder",            # ← 工程師把它當 label 用
        "color":  (230, 57, 70),            # 視覺化用
        "detections": [
            {"mask": ..., "score": 0.92, "bbox_xywh": [...], ...},
            ...
        ],
    },
    {"prompt": "liver", "color": (...), "detections": [...]},
]
```

→ 這個「prompt → list of detections」的字典結構是工程師的**設計**,不是 SAM3 模型本身的輸出格式。但**字典裡每個 detection 的 mask、box、score** 都是 SAM3 原生的東西,只是經過後處理還原到原圖座標。

### 1.5 為什麼工程師要包這層?

三個原因:

1. **SAM3 原生輸出對下游不直觀**:200 個 query 的 mask logits 對外科醫師完全沒意義,要先做 NMS / threshold 才能說「這就是膽囊」
2. **座標系要轉換**:SAM3 內部都用 1008×1008 + normalized cxcywh,但你下游(寫 PowerPoint、餵 stage 2、計算 mAP)要的是原圖像素座標
3. **prompt 即 label 的便利結構**:在 ICG 場景下「prompt 字串本身就是器官名」這個事實太好用,工程師直接把它當 label 寫進輸出 dict,省一個 class id 對應表

### 1.6 一張圖總結

```
使用者                    輸入端包裝                 SAM3 原生 forward                  輸出端包裝                  使用者
───────                   (infer_lora.py)            (Sam3Image.forward)               (infer_lora.py)             ───────
PIL 影像          ─→  preprocess_image            BatchedDatapoint             pred_logits  (B, 200)         ─→  predictions
                     resize 1008+normalize                                                                        [{prompt, detections:
prompt 字串列表    ─→  build_inference_datapoint  + collate_fn_api      ─→     pred_masks   (B, 200, H, W)  ─→     [{mask, bbox_xywh,
                     FindQueryLoaded                                                                                  score}, ...]}, ...]
                                                                              pred_boxes   (B, 200, 4)
                                                                              presence_logit_dec
                                                                                       │
                                                                                       ↓
                                                                              filter_predictions
                                                                              (NMS + threshold + top-k)
                                                                                       ↓
                                                                              build_detection
                                                                              (回原圖大小)
```

→ 中間方塊裡的內容(`Sam3Image.forward()` 的輸入容器和輸出 dict)是**SAM3 原生**;左右兩側包裝(包成 `BatchedDatapoint`、把 200 個 query 收斂成可讀 detection)是**工程師寫的**。

---

## 主題 2:類別名 → text_encoder?座標 → geometry_encoder?

這個問題的直觀理解很容易混淆。先給結論再展開:

### 2.1 短答

| 問題 | 答案 |
|---|---|
| ICG-LC-EAES 中的 class 名稱(`gallbladder`、`liver`)是 **text_encoder** 的 input? | ✅ **是的**(透過 `FindQueryLoaded.query_text`,在 collate 後彙整去重餵進 text_encoder) |
| ICG-LC-EAES 中的 bbox 座標是 **geometry_encoder** 的 input? | ❌ **不是**。座標被當成 **supervision target**(計算 box loss 用),**不**進 geometry_encoder |
| 那 geometry_encoder 用來幹嘛? | 編碼**使用者畫的 box prompt**(interactive 模式)。stage 1 訓練 + stage 1 推論預設**完全沒用到**(餵空輸入) |

接下來把兩條路徑各自展開。

### 2.2 文字 prompt 的完整旅程(class 名 → text_encoder → query)

ICG-LC-EAES 的標註是 COCO 格式,`categories` 段裡有 `name` 欄位:

```json
"categories": [
  {"id": 1, "name": "gallbladder"},
  {"id": 2, "name": "liver"}
]
```

這個 `name` 是英文小寫的器官名。它的旅程是:

1. **資料端:`CammaSam3Dataset.__getitem__()`**
   - 對每張影像、每個 selected category,產生一個 `FindQueryLoaded`
   - 把 `name` 字串塞進 `FindQueryLoaded.query_text`
   - 同時把該類別在這張影像的所有 GT box id 列在 `object_ids_output`(注意:**id 列表**,不是 box 數值,box 數值另外存)

2. **collate 端:`collate_fn_api()`**
   - 同一個 batch 裡如果有兩張影像都有 `gallbladder`,**只把它加進 `text_batch` 一次**(去重)
   - 每個 query 額外存一個 `text_ids = text_batch.index(query_text)`,告訴模型「我這條 query 對應 text_batch 第幾條字串」
   - 結果:`find_text_batch = ["gallbladder", "liver"]`(去重後),配上每個 query 的 text_ids

3. **模型端:`Sam3Image.forward()` 第一步**
   - 呼叫 `self.backbone.forward_text(input.find_text_batch, device)` → 把 `["gallbladder", "liver"]` 餵進 **text encoder(CLIP-style)**
   - text encoder 對每個字串先做 BPE tokenize、再過 transformer,輸出 `language_features`(形狀 `(L, N=2, 256)`)
   - 這個 `language_features` 存在 `backbone_out` dict 裡備用

4. **使用端:`_encode_prompt()`**
   - 對每個 query,用該 query 的 `text_ids` 從 `language_features` 抓出對應的 256 維向量(`txt_feats = backbone_out["language_features"][:, txt_ids]`)
   - 把 txt_feats 跟 geo_feats 串接成最終 `prompt`,送進 transformer encoder

→ **觀念定錨**:類別名 → text encoder 是「整批 prompt 字串去重後一次性編碼」,不是「每張影像每次重編一次」。`text_ids` 是字串去重後的對應索引機制。

→ **推論時也是同一條路**:`infer_lora.py` 把命令列 `--prompt` 攤成字串列表後,經 `build_inference_datapoint()` 變成 `Datapoint`,經 `collate_fn_api()` 變成 `BatchedDatapoint`,接著也是同一個 `Sam3Image.forward()`。

### 2.3 座標的完整旅程(GT bbox → 哪裡?)

這部分容易誤會,因為「座標」聽起來像「幾何資訊」就直覺以為要送進 `geometry_encoder`。實際上不是。

**訓練時 GT bbox 的旅程**:

1. **資料端:`CammaSam3Dataset.__getitem__()`**
   - 每個 annotation 的 bbox 用 `coco_bbox_to_normalized_cxcywh()` 轉成 normalized cxcywh
   - 包進 `Object.bbox`,放進該影像的 `objects` 列表
   - `category_to_object_ids` 紀錄哪些 object 屬於哪個 category(這就是後面 `FindQueryLoaded.object_ids_output` 的來源)

2. **collate 端:`collate_fn_api()`**
   - 把所有影像所有 object 的 bbox 集中,放進 `find_targets[stage_id].boxes`
   - 同時包含 `boxes_padded`、`object_ids` 等對應結構

3. **模型端:`Sam3Image.forward()`**
   - **GT bbox 不進 forward 裡的任何 encoder**
   - 只在 forward 結束後,被 `_compute_matching()` 用來跟 `pred_boxes` 做 Hungarian matching(O2O)和 O2M matching
   - 之後在 loss 計算階段(`train_lora.py` 裡的 loss function)被當成 supervision target,跟 `pred_boxes` 算 box loss(L1 + GIoU)、跟 `pred_masks` 算 mask loss(若啟用)

→ **觀念定錨**:GT bbox 的角色是「告訴模型『正確答案是這個』,讓 loss 把模型往那邊推」。它**從不**作為輸入餵進 SAM3 的任何 encoder。模型訓練是「看影像 + 看文字 prompt → 自己預測 box」,**不是**「看影像 + 看 box → 預測 mask」。後者是 SAM 1/2 的 interactive 模式,SAM3 文字 prompt 模式跳過了。

### 2.4 那 geometry_encoder 是做什麼的?何時才會用?

`geometry_encoder` 的設計目的是處理**使用者畫的 box prompt**(就是 SAM 原始那個「框出物體 → 給 mask」的互動模式)。它的輸入不是 GT box,是**人類在畫面上畫的框**(或來自前一幀 tracker 的 box prediction)。

`Sam3Image.forward()` 內部幾乎每次都會呼叫 `self.geometry_encoder()`,但**輸入內容會根據 mode 不同**:

| 模式 | `find_input.input_boxes` 內容 | geometry_encoder 實際做什麼 |
|---|---|---|
| Stage 1 訓練(ICG-LC-EAES) | **空** | 跑空輸入,產生 0 個 geo token,`prompt = cat([txt_feats, geo_feats=空, vis=空])` 等同於 `txt_feats` 而已 |
| Stage 1 推論(`infer_lora.py` 預設) | **空** | 同上,完全不用 |
| 互動模式 / interactive prompt sampling | 有 box(從上一個 step 採樣) | 編碼 box → geo token → 跟 txt feats 串接,讓 query 同時受文字與框影響 |
| Video tracker 內部 | tracker 自己會傳 mask/box prompt | tracker 那邊另外處理,用的是 SAM3 官方的 tracker 模型,**不是**你 LoRA fine-tune 的 detector |

→ **stage 1 ICG-LC-EAES 完全沒用到 geometry_encoder**,所以 ICG config 設 `apply_to_geometry_encoder: false`(凍結它,反正用不到,動了也是浪費 trainable 參數)。

→ 這也解釋為什麼 [[02_sam3_architecture_deep]] 組件 3 說「stage 1 訓練不用」、「ICG config 預設不開 LoRA」:不是因為它不重要,是因為**這條路徑根本沒被啟用**。

### 2.5 一張圖看懂兩種「座標」與「文字」的去處

```
                                        [模型 INPUT 端]                        [模型輸出 vs Loss 端]
───────────────────────                 ─────────────────────                  ───────────────────────
COCO categories.name                                                            
"gallbladder"、"liver"      ───→        text_encoder                  ───→     language_features
                                       (CLIP-style transformer)                       │
                                                                                       ├── 串入 prompt
COCO annotation.bbox                                                                  │   參與 attention
[cx, cy, w, h]              ┐                                                          │
                            │                                                          ↓
                            ├──→  ❌ 不送 encoder,                            decoder 出 N=200 個 query
                            │      只當 supervision target                            │
                            │                                                          ├── pred_logits
                            ↓                                                          ├── pred_masks
                       find_targets.boxes  ───→  loss 函數                ←──         └── pred_boxes
                                                  (Hungarian matching                      │
                                                   + L1 + GIoU + mask loss)                │
                                                                                            ↓
                                                                                     反向傳播 → 更新 LoRA 權重
                                                                                     (vision/detr-encoder/decoder)


使用者畫的 box prompt 
(stage 1 沒這條路徑)        ───→        geometry_encoder              ───→     geo_feats(stage 1 為空)
                                                                                       │
                                                                                       └── 串入 prompt(若有)
```

→ 文字進 text_encoder、bbox 當 supervision target、geometry_encoder 在 stage 1 等於閒置。三條路徑各司其職。

---

## 主題 3:能否讀一系列圖片,直接吐 mask + label?

### 3.1 短答

> **可以,但 codebase 預設只處理「一張圖片 + 一組 prompts」**;一系列要自己寫 for-loop 包住 `run_inference()`,或拆出內部 model load + per-image forward。輸出**就是**「指定器官(prompt 字串)→ mask + bbox + score 列表」,所以「指定器官 mask + 對應 label」原生支援。

### 3.2 三條路線

#### 路線 A:**外層寫 for-loop**(最直接)

每張影像各自呼叫 `run_inference()`。**缺點**:每張都重新 load model + LoRA,非常慢。

#### 路線 B:**改寫 `run_inference()` 把 model load 跟 forward 拆開**

把 `build_sam3_image_model + apply_lora_to_model + load_lora_weights + model.eval()` 抽出來只做一次,然後對 `for image_path in image_paths` 逐張呼叫 `preprocess_image + build_inference_datapoint + collate + model + filter_predictions + build_detection`。

**這就是 `PromptedVideoTracker._detect_frame()` 的做法** — 它已經把 model 一次 load,對 frame 列表逐張呼叫 detector。可以直接拿來改寫成「批次圖片推論器」。

#### 路線 C:**真正 batch(同時餵多張)**

理論上 `Datapoint.images` 是 `list[Image]`,可以放多張,但需要驗證 `collate_fn_api` 與 `Sam3Image.forward()` 是否支援這個 batch 維度。**目前 codebase 沒提供範例**,風險較高,建議先用路線 A/B。

### 3.3 為什麼「指定器官 → mask + label」原生支援?

回顧主題 1.4 的最終 `predictions` 結構:每個元素有 `prompt`(就是器官名)+ `detections`(那個器官在這張影像的所有 mask + bbox + score)。**`prompt` 字串本身就是 label**,不需要分類層,不需要 class id 對應表。

### 3.4 一張圖總結 image 模式的輸入輸出

```
輸入                                         模型                          輸出
─────────                                    ─────                        ─────────
PIL.Image (任意尺寸)         ┐                                              ┌→ predictions = [
  → preprocess (1008×1008,   │                                              │     {
    [-1,1])                  ├→ Datapoint ─→ Sam3Image.forward ─→ filter ─→│       prompt: "gallbladder",
  list[str] prompts          │                                              │       detections: [{mask, bbox, score}, ...]
    ["gallbladder","liver"]  ┘                                              │     },
                                                                            │     {prompt: "liver", detections: [...]},
                                                                            └→  ]
```

---

## 主題 4:能否讀影片,每隔 5 秒輸出 image+mask + temporal label?

### 4.1 短答

> **可以做到,但需要一些 post-processing**。
> - **影片讀取與每 frame mask**:`PromptedVideoTracker` 原生支援,輸出 `tracks.json` 內含 `frame_index → mask_rle` 對應
> - **每隔 5 秒一張**:codebase 沒有「秒數抽樣」原生 API,但可以用 `--strategy stride --stride <fps × 5>` 或讀完 tracks 後自己每 N frames 取一筆
> - **Label 帶 temporal information**:**最基本的 temporal info(frame_index、object_id、prompt 一致性)原生有**;**進階的「進入/退出時間、可見區段、消失重現」需要自己後處理 `tracks.json`**

### 4.2 模型實際做什麼?(理解前提)

`PromptedVideoTracker` 同時用**兩個模型**:

| 模型 | 角色 | LoRA |
|---|---|---|
| **Detector** = SAM3-image | 在「prompt frames」上偵測初始 mask/box | **掛 LoRA**(你訓練出來的) |
| **Tracker** = SAM3-tracker(官方) | 在所有 frames 上 propagate mask,產生時間連續輸出 | **不掛 LoRA**(用 HF 預訓練) |

→ 這是 SAM3 設計的核心:**LoRA 學「如何在 ICG 影像找到器官」**,**「如何在連續 frames 上追蹤」沿用官方 SAM3 預訓練的 tracker memory**。後者是 SAM2/SAM3 系列在數百萬影片上學的能力,不需要也沒理由再訓。

### 4.3 輸出結構:`tracks.json`

```json
{
  "strategy": "adaptive",
  "num_frames": 120,
  "prompts": ["gallbladder", "liver"],
  "tracks": [
    {
      "frame_index": 0,
      "prompt": "gallbladder",
      "object_id": 1,
      "bbox_xywh": [x, y, w, h],
      "mask_rle": {"size": [H, W], "counts": "..."},
      "tracker_health": 0.87,
      "object_score": 0.92,
      "detector_prompted": true
    },
    ...
  ]
}
```

→ **每 frame × 每 prompt** 一筆,每筆是「一個 frame 上一個物體」。同一物體跨 frames 共享同一 `object_id` — 這就是 temporal identity。

### 4.4 「每隔 5 秒一張」— 推薦做法

直接過濾 `tracks.json`(不需要改 SAM3 程式碼,符合 [[CLAUDE]] 唯讀規則):

```
讀 tracks.json → 對每筆 track,判斷 frame_index % (fps × 5) == 0 → 留下來
                 → 從原影片取對應 frame → 對齊 mask_rle decode 出來 → 配對
                 → label 從 track["prompt"] + track["object_id"] + track["frame_index"] / fps 組
```

→ 出來的就是「image + mask 配對 + 帶 temporal info 的 label」。

### 4.5 「Label 帶 temporal information」— 原生有什麼、缺什麼

**原生有的**(每筆 track 自帶):
- `frame_index` — 第幾 frame(0-indexed)
- `object_id` — 跨 frames 同一物體的 ID(SAM3 tracker memory 自動維持)
- `prompt` — 該 object 對應的器官名
- `tracker_health` — 該 frame 上 tracker 對該物體的「掌握度」
- `object_score` — tracker 對「此物體在此 frame 確實存在」的信心
- `detector_prompted` — 該 frame 是否被 detector 重新 prompt 過

**原生沒有的高階 temporal label**(自己後處理):
- 「進入畫面時間」 → 對 `object_id` 取 `min(frame_index where mask_area > 閾值)`
- 「退出畫面時間」 → `max(frame_index where mask_area > 閾值)`
- 「連續可見區段」 → 掃 frames、以 `tracker_health > 閾值` group consecutive frames
- 「短暫消失再重現」 → 同上,找 gap 序列

→ 這些不是 SAM3 模型該做的,是 **post-processing 工程** — 從 `tracks.json` 寫個分析腳本就能得到。

### 4.6 一張圖總結

```
輸入                                                          模型                                                   輸出
─────────                                                     ─────                                                 ─────────
MP4 / frame dir   ┐                                                                                                  ┌→ tracks.json
  → cv2 / PIL 解  │                                           Detector  Tracker                                      │   per-frame:
  成 list[Image]  │                                              ↓        ↓                                          │     {frame_index, prompt,
                  ├→ 在「prompt frames」上跑 detector ─→ mask/box ─→ tracker.add_new_mask/box ─→ propagate_in_video ─┤      object_id, mask_rle,
prompts ["liver", │     ↑                                                                          ↓                 │      bbox_xywh,
  "gallbladder"]  │     └─ first / stride / adaptive 策略                                          ↓                 │      tracker_health,
                  │        決定哪些 frame 給 detector 跑                                            ↓                 │      object_score,
strategy +        │                                                                                ↓                 │      detector_prompted}
  stride / health │                                                                                ↓                 │
  threshold       ┘                                                                                ↓                 ├→ prompt_events.json
                                                                                                   ↓                 ├→ overlays/*.png
                                                                                                   ↓                 └→ overlay.mp4
                                                                                                  per-frame mask 序列
```

---

## 帶走的重點

1. ✅ **SAM3 模型「原生」forward** 吃 `BatchedDatapoint`、吐 `pred_logits / pred_masks / pred_boxes`(每個 prompt 各 N=200 個 query)。**最終 `predictions` 字典是工程師包裝**,不是 SAM3 原生格式。
2. ✅ **「prompt 字串本身就是 label」** — 工程師故意這樣設計,讓下游不需要 class id 對應表。
3. ✅ **類別名(`gallbladder`)→ text_encoder**(透過 `find_text_batch` + `text_ids`)。**GT bbox → 不進任何 encoder**,只當 supervision target 跟 `pred_boxes` 算 loss。
4. ✅ **geometry_encoder 在 stage 1 ICG 等同閒置**:沒人餵 box prompt,所以它收到空輸入。它的真正用途是 interactive 模式(使用者畫框) — stage 1 沒走那條路。
5. ✅ **Image 模式想批次處理一系列圖片**:寫 for-loop 包住 `run_inference()`,或仿 `PromptedVideoTracker._detect_frame()` 拆出 model load 跟 per-image forward。
6. ✅ **Video 模式輸入輸出**:detector(掛 LoRA)+ tracker(官方,不動)雙模型;`tracks.json` 每筆 = 一個 frame × 一個 object 的 mask + bbox + 健康度。
7. ✅ **「每隔 5 秒 + temporal label」靠後處理 `tracks.json`**:`frame_index / fps` 算秒數,`object_id` 跨 frame 一致就是基礎 temporal info,進階的進入/退出時間靠掃 `frame_index` 範圍取 min/max。

---

## 對話脈絡記錄

本份筆記由 2026-05-04、2026-05-05 對話累積,使用者已主動詢問:

1. SAM3 的實作上,輸入跟輸出是什麼?可否讀一系列圖片,直接輸出指定器官或結構的 segmentation mask 與對應的 label?(主題 3,2026-05-04)
2. 可否讀一段影片,每隔 5 秒鐘輸出 image + segmentation mask 的配對組合,且 label 中帶有 temporal information?(主題 4,2026-05-04)
3. 在這個專案中,SAM3 用 LoRA + ICG-LC-EAES 做 finetune 與訓練,但 inference 的時候,finetune 後的 SAM3 讀什麼?輸出什麼?這個輸出格式是 SAM3 原生還是工程師修改過的設計?(主題 1,2026-05-05)
4. 原本 ICG-LC-EAES 格式中的 class 名稱,是當作 text_encoder 的 input 嗎?座標資訊是 geometry_encoder 的輸入嗎?(主題 2,2026-05-05)

下次對話若要延伸這份筆記,可從:
- `Sam3Image.forward()` 內部的 `pred_logits` / `pred_masks` / `pred_boxes` 怎麼從 N=200 個 query 收斂到實際 detection(NMS、threshold、topk 細節)
- `tracker_health`、`object_score` 在 adaptive re-prompt 策略中怎麼觸發 detector 重跑(`_collect_adaptive_candidate_frames`)
- `prompt_events.json` 裡 `prompt_type` 是 `"mask"` 還是 `"box"` 的選擇邏輯(`choose_prompt_type`,以及為什麼 mask 比 box 優先)
- 若想改寫支援「真 batch 多張影像同時 forward」要動哪些介面(`collate_fn_api`、`Sam3Image.forward()` 的 image batch 維度)
- Interactive 模式(stage 1 沒用到的那條路徑)在 SAM3 codebase 怎麼啟用?何時才會餵 box 給 geometry_encoder?

— 從這幾個切角繼續延伸都合理。

---

## 程式碼速查總表(集中區)

> 前面段落不夾雜行號,所有程式碼引用集中在這裡。看完概念回頭找位置時用。

### 主題 1:推論 pipeline 三段拆解

#### 輸入端包裝(`infer_lora.py`)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `flatten_prompts`(命令列字串攤平) | `cbd_v1/infer/infer_lora.py` | 80-84 |
| `preprocess_image`(影像縮放 + normalize) | `cbd_v1/infer/infer_lora.py` | 87-91 |
| `build_inference_datapoint`(包成 Datapoint,**注意 `is_exhaustive=True`、不傳 box prompt**) | `cbd_v1/infer/infer_lora.py` | 103-132 |
| `collate_fn_api(..., with_seg_masks=False)` 推論變體 | `cbd_v1/infer/infer_lora.py` | 343-348 |

#### SAM3 原生 forward

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `Sam3Image.forward` 主入口 | `cbd_v1/src/sam3/model/sam3_image.py` | 501-547 |
| `forward_grounding`(encoder/decoder/seg 串起來)| `cbd_v1/src/sam3/model/sam3_image.py` | 440-491 |
| `_encode_prompt`(txt_feats + geo_feats + vis 合併) | `cbd_v1/src/sam3/model/sam3_image.py` | 167-210 |
| `_get_dummy_prompt`(stage 1 沒 box prompt 時用空 Prompt) | `cbd_v1/src/sam3/model/sam3_image.py` | 493-499 |
| Forward 裡 `geometric_prompt` 從 `find_input.input_boxes` 構建(stage 1 = 空) | `cbd_v1/src/sam3/model/sam3_image.py` | 522-526 |
| `pred_logits` / `pred_masks` / `pred_boxes` 取最後 stage | `cbd_v1/infer/infer_lora.py` | 350-354 |

#### 輸出端包裝(`infer_lora.py`)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `filter_predictions`(NMS + threshold + top-k) | `cbd_v1/infer/infer_lora.py` | 135-166 |
| `build_detection`(回原圖大小 + 字典化) | `cbd_v1/infer/infer_lora.py` | 169-188 |
| `run_inference` 主流程 | `cbd_v1/infer/infer_lora.py` | 307-405 |
| `predictions` 最終結構組裝 | `cbd_v1/infer/infer_lora.py` | 372-400 |
| `render_overlay`(視覺化) | `cbd_v1/infer/infer_lora.py` | 191-277 |

### 主題 2:文字與座標的去處

#### 文字 prompt 路徑

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| COCO categories 解析(class name 進入 dataset) | `cbd_v1/src/data/dataset_camma.py` | 157-234 |
| `FindQueryLoaded` 用 `name` 當 `query_text` | `cbd_v1/src/sam3/data/endoscapes.py` | 237-259 |
| collator 文字去重 + `text_ids` | `cbd_v1/src/sam3/train/data/collator.py` | 218-220 |
| forward 第一步餵 `find_text_batch` 進 text_encoder | `cbd_v1/src/sam3/model/sam3_image.py` | 508-509 |
| `_encode_prompt` 用 `text_ids` 抓 txt_feats | `cbd_v1/src/sam3/model/sam3_image.py` | 179-181 |
| Text encoder 主類別 | `cbd_v1/src/sam3/model/text_encoder_ve.py` | 整檔 |
| VL combiner 的 `forward_text` | `cbd_v1/src/sam3/model/vl_combiner.py` | 121-170 |

#### GT bbox 路徑(supervision 而非 input)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `coco_bbox_to_normalized_cxcywh` | `cbd_v1/src/sam3/image_utils.py` | (utility) |
| `Object.bbox` 與 `category_to_object_ids` | `cbd_v1/src/sam3/data/endoscapes.py` | 195-233 |
| `find_targets.boxes`(loss target) | `cbd_v1/src/sam3/model/data_misc.py` | 159-165 |
| `_compute_matching`(Hungarian 匹配) | `cbd_v1/src/sam3/model/sam3_image.py` | 549-552 |
| `back_convert`(target 整理給 matcher 用) | `cbd_v1/src/sam3/model/sam3_image.py` | 554-568 |

#### Geometry encoder 路徑(stage 1 閒置)

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `Prompt` dataclass(box_embeddings/box_mask/box_labels)| `cbd_v1/src/sam3/model/geometry_encoders.py` | 82-249 |
| `SequenceGeometryEncoder` 主類別 | `cbd_v1/src/sam3/model/geometry_encoders.py` | (整檔) |
| `apply_to_geometry_encoder: false`(ICG config) | `cbd_v1/configs/icglceaes_lora.yaml` | 19-44 |
| `forward` 中 `geometric_prompt` 由 `find_input.input_boxes` 構造 | `cbd_v1/src/sam3/model/sam3_image.py` | 522-526 |
| 推論時 `build_inference_datapoint` 沒填 box prompt(預設無) | `cbd_v1/infer/infer_lora.py` | 103-132 |

### 主題 3:image 模式延伸到一系列圖片

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `PromptedVideoTracker._detect_frame`(可參考的「批次圖片」做法) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 305-363 |
| Detector 一次 load,多次 forward(`_ensure_models`) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 196-228 |

### 主題 4:video 模式

| 想看什麼 | 檔案 | 行號 |
|---|---|---|
| `PromptedVideoTracker.__init__` + `_ensure_models` | `cbd_v1/src/sam3/video_prompt_tracker.py` | 165-228 |
| `_load_frames`(MP4 / 資料夾兩種讀法 + FPS) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 230-282 |
| `_run_tracking_pass`(產生 tracks 主迴圈) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 472-536 |
| `_collect_adaptive_candidate_frames`(adaptive 重 prompt 觸發) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 538-562 |
| `_write_outputs`(寫 tracks.json + overlays) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 622-679 |
| `run` 三種策略(first / stride / adaptive) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 681-782 |
| `encode_binary_mask` / `decode_binary_mask`(RLE 編碼) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 67-83 |
| `choose_prompt_type`(mask vs box 偏好) | `cbd_v1/src/sam3/video_prompt_tracker.py` | 103-111 |

### 自己驗證指令(快速複製)

```bash
# 主題 1:輸入端 + SAM3 原生 + 輸出端三段
sed -n '80,132p'   cbd_v1/infer/infer_lora.py            # flatten_prompts + preprocess + build_datapoint
sed -n '135,188p'  cbd_v1/infer/infer_lora.py            # filter_predictions + build_detection
sed -n '307,405p'  cbd_v1/infer/infer_lora.py            # run_inference 主流程
sed -n '501,547p'  cbd_v1/src/sam3/model/sam3_image.py   # SAM3 原生 forward 入口
sed -n '440,499p'  cbd_v1/src/sam3/model/sam3_image.py   # forward_grounding + _get_dummy_prompt

# 主題 2:文字路徑 + 座標路徑
sed -n '237,259p'  cbd_v1/src/sam3/data/endoscapes.py    # FindQueryLoaded with query_text
sed -n '218,220p'  cbd_v1/src/sam3/train/data/collator.py  # 文字去重 + text_ids
sed -n '167,210p'  cbd_v1/src/sam3/model/sam3_image.py   # _encode_prompt 用 text_ids 抓 feats
sed -n '522,526p'  cbd_v1/src/sam3/model/sam3_image.py   # geometric_prompt 從 input_boxes 構造
sed -n '549,568p'  cbd_v1/src/sam3/model/sam3_image.py   # _compute_matching + back_convert(GT 座標路徑)

# 主題 4:video 模式
sed -n '165,230p'  cbd_v1/src/sam3/video_prompt_tracker.py
sed -n '472,536p'  cbd_v1/src/sam3/video_prompt_tracker.py
sed -n '622,680p'  cbd_v1/src/sam3/video_prompt_tracker.py
sed -n '681,782p'  cbd_v1/src/sam3/video_prompt_tracker.py
```
