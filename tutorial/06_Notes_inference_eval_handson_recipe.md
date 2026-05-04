# 06_Notes — SAM3 推論的輸入/輸出與 video pairing:對話補充筆記

> 對應原始筆記:[[06_inference_eval_handson_recipe]]
> 系列脈絡:[[01_sam3_lora_overview]] → [[02_sam3_architecture_deep]] → [[02_Notes_sam3_architecture_deep]] → [[03_lora_principles_and_freeze]] → [[04_data_pipeline_class_prompted_box]] → [[05_training_loop_and_slurm]] → **本份**(對應 06)
> 索引:[[README]]
> 撰寫日期:2026-05-04

---

## 為什麼有這份筆記

[[06_inference_eval_handson_recipe]] 已經把三種推論模式的命令列旗標整理過,但對「**SAM3 在實作上到底怎麼把影像吃進去、把什麼東西吐出來**」缺少一個系統的輸入/輸出表。本份筆記補上:

1. **單張影像推論**(`infer_lora.py`)的輸入/輸出張量規格 → 回答「能否讀一系列圖片直接吐 mask + label」
2. **影片追蹤**(`infer_video_lora.py` + `PromptedVideoTracker`)的輸入/輸出 → 回答「能否每隔 5 秒輸出 image+mask 配對 + 時間 label」

底下所有結論都從程式碼確認,不是論文層級的描述。

---

## 問題 1:SAM3(image 模式)的輸入輸出是什麼?能讀一系列圖片直接吐 mask+label 嗎?

### 1.1 短答

> **可以,但 codebase 預設只處理「一張圖片 + 一組 prompts」**。
> 想處理「一系列圖片」要自己寫一個 for-loop 包住 `run_inference()`,或改 batching 邏輯。
> 輸出**就是**「**指定器官(prompt 字串)→ mask + bbox + score 列表**」,所以「指定器官 mask + 對應 label」這件事是原生支援的。

### 1.2 輸入規格(逐欄拆解)

來源:`cbd_v1/infer/infer_lora.py:307-345`(`run_inference()`)。

#### (a) 影像

```python
# infer_lora.py:339-341
image = PILImage.open(image_path).convert("RGB")     # 任意尺寸的 PIL Image
orig_w, orig_h = image.size
image_tensor = preprocess_image(image, args.resolution)
```

`preprocess_image()` @ `infer_lora.py:87-91` 做三件事:
1. `resize_image_to_square(pil_image, 1008)` — 直接縮成 1008×1008(會拉伸,不保留比例)
2. 轉 `numpy float32` 並除以 255 → 範圍 [0, 1]
3. permute 成 `(3, 1008, 1008)` 後做 `(x - 0.5) / 0.5` → 範圍 **[-1, 1]**

→ **單張影像最終 tensor shape = `(3, 1008, 1008)`,float [-1, 1]**。

#### (b) 文字 prompt(就是 label 來源)

```python
# infer_lora.py:80-84
def flatten_prompts(prompt_groups):
    prompts = [p.strip() for group in prompt_groups for p in group if p.strip()]
    ...
```

從命令列 `--prompt gallbladder --prompt liver` 拿到 `["gallbladder", "liver"]`,**就是純文字字串**。

→ 在 [[01_sam3_lora_overview]] 提到的「open-vocabulary 文字 prompt」就是這個。模型不需要重新訓練去支援新的器官名,只要該詞在 BPE vocab 裡就能編碼。

#### (c) 包成 Datapoint

`build_inference_datapoint()` @ `infer_lora.py:103-132`:

```python
queries = [FindQueryLoaded(query_text=p, ...) for p in prompts]
datapoint = Datapoint(
    find_queries=queries,                                # 一個 prompt = 一個 query
    images=[Image(data=image_tensor, ...)],              # 一張影像
)
```

每個 prompt 對應一個 `FindQueryLoaded`,代表「請在這張圖裡幫我找出符合這段文字的所有 instance」。

→ 注意:**stage 1 完全不餵 box prompt / point prompt**。`is_exhaustive=True`(`infer_lora.py:117`)告訴模型「找出所有」而非「只找一個」。

#### (d) 經 collator 進入模型

```python
# infer_lora.py:343-348
batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=False)
input_batch = move_to_device(batch["input"], device)
with torch.no_grad():
    with autocast_context(device, training_config):
        outputs_list = model(input_batch)
```

**注意 `with_seg_masks=False`**:推論時不需要餵 GT mask(訓練才需要)。

#### 輸入規格速查表

| 欄位 | 形狀/型別 | 來源 |
|---|---|---|
| 影像 tensor | `(3, 1008, 1008)`, float, [-1, 1] | `preprocess_image()` |
| Prompts | `list[str]`, 例如 `["gallbladder", "liver"]` | 命令列 `--prompt` |
| Datapoint.images | `list[Image]`,長度=1 | 一次一張 |
| Datapoint.find_queries | `list[FindQueryLoaded]`,長度=#prompts | 每 prompt 一個 |
| Box prompt | **不傳**(stage 1 純文字模式) | — |

### 1.3 輸出規格(逐欄拆解)

#### (a) 模型原始輸出 `final_outputs`(dict)

`final_outputs = list(outputs_iter)[-1][-1]`(`infer_lora.py:354`)— 取最後一個 stage 的最後一步。內含:

| key | 形狀 | 意義 |
|---|---|---|
| `pred_logits[i]` | `(N_queries,)` | 第 i 個 prompt 對應的 N 個 query 的分類 score |
| `pred_masks[i]`  | `(N_queries, H, W)` | N 個 query 的 mask logits(尚未 sigmoid),解析度通常 = SAM3 內部尺寸 |
| `pred_boxes[i]`  | `(N_queries, 4)` | N 個 query 的 normalized cxcywh box |
| `presence_logit_dec[i]` | scalar(可選) | "this prompt has matches" 的整體存在 logit |

→ N_queries 是 DETR-style 設計的固定數字(例:200)。每個 query 都是「一個可能的 instance」。

#### (b) 經 `filter_predictions()` 過濾

`infer_lora.py:135-166`:
1. 用 sigmoid + presence logit 算 `scores`(`compute_detection_scores`)
2. mask sigmoid 後二值化(`> 0.5`)
3. 跑 mask-NMS(`nms_masks`)用 IoU 閾值移除重複 instance
4. `prob_threshold` 過濾 score 太低的
5. 取 top-k(`max_detections`)

→ 從 200 個 query 收斂到「真正的 detection」(通常每 prompt 0~10 個)。

#### (c) 包成最終 detection dict(`build_detection()` @ `infer_lora.py:169-188`)

每個 detection 是:

```python
{
    "mask":            np.ndarray (orig_h, orig_w), bool      # 縮回原圖大小的 binary mask
    "score":           float                                   # confidence
    "box_norm_cxcywh": [cx, cy, w, h]   in [0, 1]              # normalized 座標
    "bbox_xywh":       [x, y, w, h]     像素單位、原圖座標
}
```

#### (d) `predictions` 最終結構(`infer_lora.py:394-400`)

```python
predictions = [
    {
        "prompt": "gallbladder",            # ← 這就是 label
        "color":  (230, 57, 70),            # 視覺化用
        "detections": [
            {"mask": ..., "score": 0.92, "bbox_xywh": [...], ...},
            {"mask": ..., "score": 0.78, "bbox_xywh": [...], ...},
        ],
    },
    {
        "prompt": "liver",
        "color":  (29, 78, 216),
        "detections": [...],
    },
]
```

→ **這就是「指定器官 + segmentation mask + 對應 label」的原生結構**。

### 1.4 「能讀一系列圖片嗎?」— 兩種解法

#### 解法 A:**外層寫 for-loop**(最直接)

```python
from infer_lora import run_inference  # 或拆出 run_inference 的內部 model+lora load 部分

# 假設 args.image 是一個資料夾
for image_path in sorted(Path(image_dir).iterdir()):
    args.image = str(image_path)
    args.output = str(output_dir / f"{image_path.stem}_overlay.png")
    predictions = run_inference(args)   # 每張獨立跑
```

**缺點**:每張影像都重新 load model + LoRA,非常慢。

#### 解法 B:**改寫 `run_inference()` 把 model load 跟 forward 拆開**

```python
# 偽碼
model = build_sam3_image_model(...)
model = apply_lora_to_model(model, lora_cfg)
load_lora_weights(model, weights_path, expected_config=lora_cfg)
model.eval()

results = []
for image_path in image_paths:
    image = PILImage.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image, 1008)
    datapoint = build_inference_datapoint(image_tensor, prompts, image.size[::-1], 1008)
    batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=False)
    input_batch = move_to_device(batch["input"], device)
    with torch.no_grad(), autocast_context(device, training_config):
        outputs_list = model(input_batch)
    # ... filter_predictions + build_detection ...
    results.append(per_image_predictions)
```

**這就是 `PromptedVideoTracker._detect_frame()`**(`cbd_v1/src/sam3/video_prompt_tracker.py:305-363`)的做法 — 它已經把 model 一次 load,然後對 frame 列表逐張呼叫 detector。可以直接拿來改寫成「批次圖片推論器」。

#### 解法 C:**真正 batch(同時餵多張)**

理論上 `Datapoint.images` 是 `list[Image]`,可以放多張,但需要驗證 `collate_fn_api` 與 `Sam3Image.forward()` 是否支援這個 batch 維度。**目前 codebase 沒提供範例**,風險較高,建議先用解法 A/B。

### 1.5 一張圖總結 image 模式的輸入輸出

```
輸入                                          模型                         輸出
─────────                                     ─────                       ─────────
PIL.Image (任意尺寸)         ┐                                              ┌→ predictions = [
  → preprocess (1008×1008,   │                                              │     {
    [-1,1])                  ├→ Datapoint ─→ Sam3Image.forward ─→ filter ─→│       prompt: "gallbladder",
  list[str] prompts          │                                              │       detections: [{mask, bbox, score}, ...]
    ["gallbladder","liver"]  ┘                                              │     },
                                                                            │     {prompt: "liver", detections: [...]},
                                                                            └→  ]
```

→ 「指定器官 → mask + label」**直接成立**,因為 `prompt` 字串本身就是 label。

---

## 問題 2:能讀一段影片,每隔 5 秒輸出 image+mask 配對組合,且 label 帶 temporal information 嗎?

### 2.1 短答

> **可以做到,但需要一些 post-processing**。
> - **影片讀取與每 frame mask**:`PromptedVideoTracker` 原生支援,輸出 `tracks.json` 內含 `frame_index → mask_rle` 對應。
> - **每隔 5 秒一張**:codebase 沒有「秒數抽樣」原生 API,但可以用 `--strategy stride --stride <fps × 5>` 或讀完 tracks 後自己每 N frames 取一筆。
> - **Label 帶 temporal information**:**最基本的 temporal info(frame_index、object_id、prompt 一致性)原生有**;**進階的「進入/退出時間、可見區段、消失重現」需要自己後處理 `tracks.json`**。

### 2.2 模型實際做什麼?(理解前提)

`PromptedVideoTracker` @ `cbd_v1/src/sam3/video_prompt_tracker.py:165-782` 同時用**兩個模型**:

| 模型 | 角色 | 載入 |
|---|---|---|
| **Detector** = SAM3-image + 你的 LoRA | 在「prompt frames」上偵測初始 mask/box | `build_sam3_image_model + apply_lora_to_model + load_lora_weights`(`L196-213`) |
| **Tracker** = SAM3-tracker(官方,**不掛 LoRA**) | 在所有 frames 上 propagate mask,產生時間連續輸出 | `build_sam3_tracker_model(load_from_HF=True)`(`L216-228`) |

→ 這是 SAM3 設計的核心:**LoRA 學「如何在 ICG 影像找到器官」**,**但「如何在連續 frames 上追蹤」沿用官方 SAM3 預訓練的 tracker memory**。後者是 SAM2/SAM3 系列在數百萬影片上學的能力,不需要也沒有理由再訓。

### 2.3 輸入規格

來源:`run()` 方法 @ `video_prompt_tracker.py:681-782`。

| 欄位 | 形狀/型別 | 來源 |
|---|---|---|
| `video_path` | MP4 檔 **或** 含有序圖片的資料夾 | `--video` |
| `prompts` | `list[str]` 文字 prompt(同 image 模式) | `--prompt` 或 config 的 `data.class_names` |
| `strategy` | `"first"` / `"stride"` / `"adaptive"` | `--strategy` |
| `stride` | int(僅 stride 策略用) | `--stride` |
| `output_dir` | 輸出資料夾 | `--output-dir` |

#### Frames 怎麼讀?

`_load_frames()` @ `L230-282`:
- 資料夾 → 按檔名排序的 PIL Image 列表,**FPS 用預設 10** (`DEFAULT_DIRECTORY_FPS`)
- MP4 → 用 `cv2.VideoCapture`,從中讀 FPS,逐 frame 解碼成 PIL Image

→ **重要**:資料夾路徑模式時 FPS 默認 10(`L37`),如果 frames 來源不是 10 FPS,你要嘛用 MP4 模式讀取(會自動讀真實 FPS),要嘛重寫 `_load_frames_from_directory` 接收正確 FPS。

### 2.4 輸出規格

來源:`_write_outputs()` @ `L622-679`。

```
output_dir/
├── tracks.json              # 每 frame、每物體的追蹤結果(含 mask + bbox + temporal metadata)
├── prompt_events.json       # 哪些 frame 被當 prompt frame、用什麼 prompt(mask/box)、reason
├── overlays/
│   ├── 00000.png            # frame 0 的 overlay 圖(原圖 + mask + bbox)
│   ├── 00001.png
│   └── ...
└── overlay.mp4              # 整段 overlay 影片(若 OpenCV 可用)
```

#### `tracks.json` 結構(這是回答問題 2 的關鍵)

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
      "mask_rle": {"size": [H, W], "counts": "..."},   // pycocotools RLE
      "tracker_health": 0.87,
      "object_score": 0.92,
      "detector_prompted": true
    },
    { "frame_index": 1, "prompt": "gallbladder", "object_id": 1, "mask_rle": {...}, ... },
    { "frame_index": 1, "prompt": "liver",       "object_id": 2, "mask_rle": {...}, ... },
    ...
  ]
}
```

→ **每 frame × 每 prompt** 一筆(一個 prompt 對應一個 `object_id`,從 1 開始遞增 — 見 `L382, L495`)。

#### Mask 是 RLE 編碼

`encode_binary_mask()` @ `L67-73`(用 `pycocotools.mask`)— 大幅減少 JSON 體積。讀取時用:

```python
import pycocotools.mask as mask_utils
mask_2d = mask_utils.decode(track["mask_rle"])   # → np.ndarray (H, W) uint8
```

或直接用 codebase 提供的 `decode_binary_mask()` @ `L76-83`。

### 2.5 「每隔 5 秒一張 image+mask」— 三條路線

#### 路線 A:**直接過濾 `tracks.json`**(推薦,最乾淨)

```python
import json
import pycocotools.mask as mask_utils
from PIL import Image

FPS_OF_VIDEO = 30                     # ← 知道你的 video FPS
SAMPLE_EVERY_SECONDS = 5
sample_every_frames = int(FPS_OF_VIDEO * SAMPLE_EVERY_SECONDS)  # 例:150

with open("output_dir/tracks.json") as f:
    data = json.load(f)

frames = load_video_frames("clip.mp4")                          # 你的影片解碼

samples = []
for track in data["tracks"]:
    if track["frame_index"] % sample_every_frames != 0:
        continue
    image = frames[track["frame_index"]]                        # PIL Image
    mask  = mask_utils.decode(track["mask_rle"]) if track["mask_rle"] else None
    samples.append({
        "image": image,
        "mask":  mask,
        "label": {
            "prompt":     track["prompt"],          # "gallbladder"
            "object_id":  track["object_id"],       # 1
            "frame_index": track["frame_index"],     # 150, 300, ...
            "time_sec":    track["frame_index"] / FPS_OF_VIDEO,
            "tracker_health": track["tracker_health"],
            "object_score":   track["object_score"],
        },
    })
```

→ 出來的 `samples` 就是你要的「image + mask 配對 + 帶 temporal info 的 label」。

#### 路線 B:**用 `--strategy stride` 但只做 prompt 抽樣**(注意:不是輸出抽樣)

`stride` 策略是控制「每幾 frames 重新給 detector prompt」,**輸出依然是每 frame 都有 mask**。所以 `--stride 150` 不會減少輸出數量,只會減少 detector 重新 prompt 的次數(用來控制計算成本與重新對焦頻率)。

→ **若你的目的是輸出抽樣,不能用這個**。要用路線 A。

#### 路線 C:**改 `_write_outputs()` 增加抽樣寫檔**

如果想要 `output_dir/sampled_overlays/` 只寫每 5 秒一張 overlay PNG,而不是每 frame 都寫,可以在 `_write_outputs()` `L654` 那個 for-loop 加條件判斷:

```python
for frame_index, image in enumerate(frames):
    if frame_index % sample_every_frames != 0:    # ← 新增
        continue
    ...
```

但這會違反 [[CLAUDE]] 的「不可修改既有 .py」規則。所以**仍以路線 A 為主**。

### 2.6 「Label 帶 temporal information」— 原生有什麼、缺什麼

#### 原生有的 temporal info(每筆 track 都自帶)

| 欄位 | 含義 | 從 SAM3 哪裡來 |
|---|---|---|
| `frame_index` | 第幾 frame(0-indexed) | tracker propagation 的時間索引 |
| `object_id` | 同一物體在不同 frame 是同一個 ID | tracker memory 自動維持(同一 prompt 在連續 frames 內 ID 不變,即「temporal identity」) |
| `prompt` | 該 object 對應的器官名 | 你的輸入 |
| `tracker_health` | 該 frame 上 tracker 對該物體的「掌握度」(`L114-121`) | mask 前景平均機率 × object score |
| `object_score` | tracker 對「此物體在此 frame 確實存在」的信心 | tracker 內部 head |
| `detector_prompted` | 該 frame 是否被 detector 重新 prompt 過 | 來自 `prompt_events` 比對 |

→ **`object_id` 的時間連續性**是 SAM3 video tracking 的核心 temporal info:同一物體跨 frames 共享同一 ID,讓你能「跟著一個器官看它在 5 秒、10 秒、15 秒的變化」。

#### 原生**沒有**的高階 temporal label(若需要要自己後處理)

| 想要的 label | 怎麼從 tracks.json 推導 |
|---|---|
| 「進入畫面時間」 | 對 `object_id` 的所有 entries 取 `min(frame_index where mask_area > 閾值)` |
| 「退出畫面時間」 | 對 `object_id` 取 `max(frame_index where mask_area > 閾值)` |
| 「連續可見區段(含起止)」 | 掃 `object_id` 的 `frame_index`,以 `tracker_health > 閾值` 為通過條件,group consecutive frames |
| 「短暫消失再重現」 | 同上,找 gap 序列 |
| 「相對該 clip 的時間百分比」 | `frame_index / num_frames` |

→ 這些不是 SAM3 模型該做的,是 **post-processing 工程** — 從 `tracks.json` 寫個分析腳本就能得到。

### 2.7 一張圖總結 video 模式的輸入輸出

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

→ 「每隔 5 秒一張 image+mask 配對 + temporal label」可以**從 tracks.json 後處理得到**,**不需要改 SAM3 程式碼**。

---

## 兩個問題的對照總結

| 問題 | 是否原生支援 | 怎麼做 |
|---|---|---|
| 一系列圖片 → 各自的 mask + 對應器官 label | **部分原生**(單張 OK,系列要寫 loop) | `infer_lora.py` 的 `run_inference()` 拆 model load 與 per-image forward,for-loop 餵圖片 |
| 影片 → 每隔 5 秒 image+mask + temporal label | **後處理就能做**(原生有 per-frame mask + frame_index) | `infer_video_lora.py --strategy adaptive` 跑完 → 讀 `tracks.json` → 用 `frame_index % (fps × 5) == 0` 篩出每 5 秒一筆 |

---

## 想自己驗證?三條檢查路線

1. **驗證 image 模式輸入輸出**
   ```bash
   sed -n '307,405p' cbd_v1/infer/infer_lora.py            # run_inference 主流程
   sed -n '87,132p'  cbd_v1/infer/infer_lora.py            # preprocess + datapoint 構造
   sed -n '169,188p' cbd_v1/infer/infer_lora.py            # build_detection 輸出格式
   ```

2. **驗證 video 模式輸入輸出**
   ```bash
   sed -n '165,230p' cbd_v1/src/sam3/video_prompt_tracker.py   # __init__ + _ensure_models
   sed -n '472,536p' cbd_v1/src/sam3/video_prompt_tracker.py   # _run_tracking_pass(產生 tracks)
   sed -n '622,680p' cbd_v1/src/sam3/video_prompt_tracker.py   # _write_outputs(寫 tracks.json)
   sed -n '681,782p' cbd_v1/src/sam3/video_prompt_tracker.py   # run() 三種策略
   ```

3. **驗證 detector 跟 tracker 是兩個獨立模型**
   ```bash
   sed -n '192,228p' cbd_v1/src/sam3/video_prompt_tracker.py   # _ensure_models 同時建 detector(掛 LoRA)和 tracker(不掛)
   ```

---

## 帶走的 5 件事

1. ✅ **Image 模式輸入** = 單張影像(縮成 1008×1008、normalize 到 [-1,1]) + 文字 prompt 列表;**輸出** = 每 prompt 對應一個 `detections` 列表,每筆有 `mask`(原圖大小 binary)、`bbox_xywh`、`score`
2. ✅ **「prompt 字串本身就是 label」** — 不需要分類層,輸出結構天然就帶器官名
3. ✅ **Video 模式用「detector(掛 LoRA)+ tracker(官方,不動)」雙模型架構**,detector 在 prompt frames 上抓 mask/box,tracker 在所有 frames 上 propagate
4. ✅ **`tracks.json` 每筆 = 一個 frame × 一個 object 的 mask + bbox + 健康度**,所有 temporal info 從 `frame_index` + `object_id` 推導
5. ✅ **「每隔 5 秒一張」+「帶 temporal label」靠後處理 `tracks.json`**,不需要改 SAM3 程式碼。秒數從 `frame_index / fps` 算,進入/退出時間從 `object_id` 的 `frame_index` 範圍取 min/max

---

## 對話脈絡記錄

本份筆記由 2026-05-04 的對話產出,使用者主動詢問:
1. SAM3 的實作上,輸入跟輸出是什麼?可否讀一系列圖片,直接輸出指定器官或結構的 segmentation mask 與對應的 label?
2. 可否讀一段影片,每隔 5 秒鐘輸出一段對 image + segmentation mask 的配對組合,且 label 中帶有 temporal information?

下次對話若要延伸這份筆記,可從:
- `Sam3Image.forward()` 內部的 `pred_logits` / `pred_masks` / `pred_boxes` 怎麼從 N=200 個 query 收斂到實際 detection(NMS、threshold、topk 細節)
- `tracker_health`、`object_score` 在 adaptive re-prompt 策略中怎麼觸發 detector 重跑(`_collect_adaptive_candidate_frames` @ `L538-562`)
- `prompt_events.json` 裡 `prompt_type` 是 `"mask"` 還是 `"box"` 的選擇邏輯(`choose_prompt_type` @ `L103-111`,以及為什麼 mask 比 box 優先)
- 若想改寫支援「真 batch 多張影像同時 forward」要動哪些介面(`collate_fn_api`, `Sam3Image.forward()` 的 image batch 維度)

— 從這幾個切角繼續延伸都合理。
