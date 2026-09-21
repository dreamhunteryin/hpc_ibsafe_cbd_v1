# 第 8 章 — 推論流程 + 評估指標 + 實操腳本

> 撰寫日期:2026-05-27
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:把訓練好的 `best_cbd.pt` 真的拿來用——兩種推論模式各自的腳本與輸出、`compute_cbd_prediction_metrics.py` 怎麼算 mAP/IoU/F1,以及一份從零跑通的 4 階段實操腳本。

---

## 這一章的位置

第 6 章拆 transformer 內部,第 7 章拆訓練主迴圈。**這一章把訓練完的權重變成「**可看的結果**」**——overlay 圖、JSON 預測、metric 表。

對應 stage 1 的第 6 章(`stage1_SAM3+LoRA_Fine-tuning/06_inference_eval_handson_recipe.md`),格式幾乎一樣:推論流程 + 評估指標 + 實操腳本。

---

## 兩種推論模式:cached vs external clip

stage 2 的推論有**兩條路徑**:

### A. Cached inference(`infer/infer_cbd.py`)

對 **已經納入 train/val/test split 的 clip** 做推論——資料路徑寫在 `bsafe_cbd.yaml` 的 `datasets`,clip + mask 已經存在 `clips_root/` 與 `easy_mask_cache_root/`。

適用情境:
- 模型 evaluation(對 test split 全跑)
- 視覺化檢查訓練後的預測品質
- 跟 ground truth 比 IoU

對應 Slurm 排程:`slurm/schedule_cbd_cached_inference.py`。

### B. External clip inference(`infer/infer_cbd_clip.py`)

對 **任意 clip**(一個資料夾 of frames、或一個 video file)做推論——**不依賴 dataset split**。stage 1 的 SAM3 LoRA 會即時跑、產 mask;然後 stage 2 接著吃。

適用情境:
- 部署到新醫院的影像
- 拿一段沒標注的手術錄影做 demo
- 不想被 dataset 框架綁住

對應 Slurm 排程:`slurm/schedule_cbd_inference.py`。

> 兩條路徑的「終極輸出」都是同一格式的 `*_overlay.png` + `*_overlay.json`,所以下一階段(metric 計算)是共用的。

---

## Cached 模式的兩種範圍:單 clip vs 全 split

`infer/infer_cbd.py:348-374` 的 `run_inference`:

```python
def run_inference(args, trainer_cls=CBDTrainer) -> dict:
    config = load_config(args.config)
    trainer = trainer_cls(config)
    weights_path = resolve_weights_path(config, args.weights)
    trainer.load_checkpoint(weights_path)

    if args.clip_id:
        # 單 clip 模式
        item = write_prediction_artifacts(
            trainer.predict_record(args.split, args.clip_id),
            args.output, show_gt=args.show_gt,
        )
        return {"mode": "single", "count": 1, "items": [item]}

    # 全 split 模式
    dataset = trainer.build_dataset(args.split, apply_augmentation=False)
    ...
    for sample_index, record in enumerate(dataset.records):
        output_path = output_dir / build_split_output_name(...)
        prediction = predict_dataset_index(trainer, dataset, sample_index, args.split)
        items.append(write_prediction_artifacts(prediction, output_path, show_gt=args.show_gt))
```

### 單 clip 模式

```bash
python infer/infer_cbd.py \
    --config configs/bsafe_cbd.yaml \
    --weights runs/cbd_v2/<ts>/run_1/best_cbd.pt \
    --split test \
    --clip-id <clip_id> \
    --output debug/my_clip_overlay.png \
    --show-gt
```

`--clip-id` 從 dataset 找到對應 record(`engine.py:423-426` 的 `predict_record`)→ 跑 forward → 渲染 overlay PNG + JSON。

### 全 split 模式

```bash
python infer/infer_cbd.py \
    --config configs/bsafe_cbd.yaml \
    --weights runs/cbd_v2/<ts>/run_1/best_cbd.pt \
    --split test \
    --output outputs/cbd_v2_test_overlays \
    --show-gt
```

不傳 `--clip-id`,腳本會跑**整個 test split**——每筆 record 一張 PNG + 一份 JSON。

---

## Overlay PNG 長什麼樣

`infer/infer_cbd.py:279-321` 的 `write_prediction_artifacts`:

```
+-----------------------------+-----------------------------+
|                             |                             |
|       overlay panel         |     center heatmap panel    |
|                             |                             |
| ┌──── pred bbox ─────┐      |                             |
| │  (cbd_pred soft 0.82)│    |   ← 16×16 heatmap upscale   |
| └────────────────────┘     |     成原圖大小,顏色由       |
|                             |     center_heatmap_logits   |
| ┌── gt bbox(--show-gt) ──┐ |     sigmoid 後映射          |
| │  (cbd_gt soft)         │ |                             |
| └────────────────────────┘ |                             |
|                             |                             |
+-----------------------------+-----------------------------+
        clip_id | pred=<type> | gt=<type> | IoU=0.4321    ← 上面條
```

左半:最後一 frame + pred bbox(+ 可選的 gt bbox)疊加。
右半:`center_heatmap_logits` sigmoid 後映射到 jet 色階,放大到原圖尺寸。
上方標題條:clip_id、預測 type、IoU。

**為什麼這樣設計?**——overlay 給「**位置對不對**」,heatmap 給「**模型認為中心應該在哪**」,兩個並列才能看出失敗模式:

- 框對了 heatmap 也亮 → 模型可信
- 框錯但 heatmap 在對的地方 → 模型「看到」CBD 但 box query 沒對齊
- 框錯 heatmap 也散 → 模型沒看出 CBD 在哪

---

## JSON 預測長什麼樣

`infer/infer_cbd.py:194-232` 的 `build_json_payload`:

```json
{
  "clip_id": "video_42_frame_1234",
  "split": "test",
  "pred_box_norm_cxcywh": [0.532, 0.418, 0.182, 0.124],
  "pred_bbox_xywh": [411.3, 339.2, 219.6, 119.5],
  "pred_type_name": "soft",
  "pred_type_probs": {"soft": 0.78, "hard": 0.22},
  "pred_type_confidence": 0.78,
  "pred_center_cell_confidence": 0.31,
  "annotation_id": 8821,
  "image_id": 1234,
  "target_box_norm_cxcywh": [0.521, 0.423, 0.195, 0.131],
  "target_bbox_xywh": [395.5, 343.1, 234.7, 126.2],
  "target_type": "soft",
  "iou": 0.7621
}
```

**重點欄位**:

| 欄位 | 形式 | 用途 |
|---|---|---|
| `pred_box_norm_cxcywh` | [cx, cy, w, h] in [0,1] | 模型原始輸出 |
| `pred_bbox_xywh` | [x, y, w, h] in 原始像素 | 視覺化、metric |
| `pred_type_name` | "soft" / "hard" | 顯影品質預測 |
| `pred_type_probs` | dict | 完整 softmax 分布 |
| `target_*` | 同 pred 結構 | 只有 cached 模式有(external 模式沒 GT) |
| `iou` | float in [0,1] | 預先算好放進來,metric script 不用再算 |

**為什麼把 GT 也寫進 JSON?**——讓 `compute_cbd_prediction_metrics.py` 可以**只吃這個資料夾**就完成 metric 計算,不需要再回 dataset 查標注。**自包含(self-contained)的 prediction artifact**——R6 + R7 的精神。

---

## External clip 推論(`infer_cbd_clip.py`)的特殊之處

跟 cached 模式不同,external 模式需要:

1. **自帶 stage 1 推論**:用 `bsafe_cbd.yaml:29-42` 的 `stage1_sam3` 配置即時跑 SAM3 LoRA 產 mask
2. **手動指定 clip / video**:`--clip <dir-or-video-file>`
3. **沒 GT 也能跑**:`--gt-bbox` / `--gt-type` 可選

```bash
# 對一個 video 跑 stage1 → stage2 完整 pipeline
python infer/infer_cbd_clip.py \
    --config configs/bsafe_cbd.yaml \
    --weights runs/cbd_v2/<ts>/run_1/best_cbd.pt \
    --clip /path/to/new_video.mp4 \
    --output demo/new_video_overlay.png

# 對一個資料夾的 frame 跑
python infer/infer_cbd_clip.py \
    --config configs/bsafe_cbd.yaml \
    --weights runs/cbd_v2/<ts>/run_1/best_cbd.pt \
    --clip /path/to/frames_dir \
    --output demo/frames_dir_overlay.png
```

> 這條路徑跑得慢(stage 1 要即時跑 25 frame 的 SAM3 推論),適合一次一段,不適合對全 split 跑。

---

## 評估指標:從 JSON 到 metric 表

`compute_cbd_prediction_metrics.py` 把一個資料夾的 `*_overlay.json` 變成 metric 表。

### 整體流程

```
infer_cbd.py --split test --output outputs/X
                       │
                       ▼ 產生大量 .png + .json
                       │
compute_cbd_prediction_metrics.py outputs/X --json-output metrics.json
                       │
                       ▼ 印出 detection + classification 表
                       │
                       └─► metrics.json(機器可讀)
```

### 計算邏輯

腳本兩大區塊:

#### 1. Detection metric(`compute_detection_metrics`,L128-151)

對每筆 record 拿 `iou`(若 JSON 沒有就用 bbox 重算),然後:

| 指標 | 定義 |
|---|---|
| `num_samples` | 樣本數 |
| `mean_iou` | 全部 IoU 的平均 |
| `mAP` | 對 IoU thresholds [0.5, 0.55, ..., 0.95] 算 AP 再平均 |
| `mAP50` | threshold=0.5 的 AP |
| `recall` | IoU ≥ 0.5 的比例(threshold=0.5 的 recall) |
| `mean_recall_50_95` | 10 個 threshold 的 recall 平均 |

**AP 怎麼算**(`compute_average_precision`,L94-125):
- 把所有樣本的 IoU 當「**分數**」降序排
- 用 threshold 判定每筆是 TP 或 FP(IoU ≥ threshold 為 TP)
- 算 precision-recall 曲線,做 11-point interpolated AP(實際是 101 點)
- 回傳 AP 值

> 注意:**這不是 COCO 風格的 AP**。COCO AP 要算「**對每個 detection 的 confidence 排序**」;這裡用「**IoU 本身當分數**」——因為每個 clip 只預測一個 box,沒有「**多 detection 排序**」的概念。

**分組評估**(`build_results_payload`,L235-258):
- 對 `soft` / `hard` 兩類**分別**算 detection metrics
- 對「**全體**」算一次 overall
- 對 soft / hard 兩組做 row-wise average

這呼應第 7 章 §「per-type 統計怎麼算」——**分組看不同臨床難度的表現**。

#### 2. Classification metric(`compute_binary_classification_metrics`,L167-232)

對 `pred_type_name` vs `target_type` 算:

| 指標 | 定義 |
|---|---|
| `accuracy` | 預測對的比例 |
| `macro_precision` | 各類 precision 取平均(每類等權) |
| `macro_recall` | 各類 recall 取平均 |
| `macro_f1` | 各類 F1 取平均 |
| `per_class.*` | 每類獨立的 precision / recall / f1 / support |
| `confusion_matrix` | rows=target, cols=prediction 的 2×2 矩陣 |

**為何 macro 而非 micro?**——資料不平衡(soft / hard 樣本數可能 差很多),macro 給「**每類同等重要**」。臨床上 hard 雖然多,但 soft 才是真正考驗模型——macro 才能正確反映「**對較難的類別表現好不好**」。

### Console 輸出長什麼樣

`compute_cbd_prediction_metrics.py:261-308`:

```
Detection metrics (score = IoU, recall = IoU >= 0.50)
subset         n   mean_iou        mAP      mAP50     recall  mr_50_95
soft          18     0.4321     0.3214     0.4500     0.5000    0.4123
hard          32     0.6134     0.5421     0.7188     0.7500    0.6234
average        -     0.5228     0.4318     0.5844     0.6250    0.5179

Type classification metrics (pred_type_name vs target_type)
accuracy        0.7600
macro_precision 0.7421
macro_recall    0.7350
macro_f1        0.7385

Per-class metrics
label       support  precision     recall         f1
soft             18     0.6842     0.7222     0.7027
hard             32     0.8000     0.7500     0.7742

Confusion matrix (rows = target, cols = prediction)
                 soft       hard
soft               13          5
hard                8         24
```

---

## 從零跑通的 4 階段實操腳本

對應 stage 1 第 6 章的「**從零跑通的 4 階段實操腳本**」,stage 2 給對應版本。

> 前提:已有 `best_cbd.pt`(訓練完成,在 `runs/<exp>/<ts>/run_1/`)。
> 假設用 cached 模式對 test split 全跑。

### Step 1. 本機驗證(CPU,不需要 GPU)

```bash
# 在本機(沒 GPU)只做 import + config 驗證
cd /home/shihminyin/CAMMA/iBSAFE_v1/iBSAFE_CBD_v1

# (a) 確認 yaml 解析
python -c "
import yaml
config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
print('model.variant   :', config['model']['variant'])
print('output.output_dir:', config.get('output', {}).get('output_dir'))
print('eval split would use:', config['data']['test_split'])
"

# (b) 確認 weights 路徑能 resolve
python -c "
import yaml, sys
sys.path.insert(0, 'src')
from pathlib import Path
config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
default = Path(config['output']['output_dir']) / 'best_cbd.pt'
print('default weights would be:', default)
print('exists?', default.exists())
"

# (c) 確認 metric script 能解析範例 JSON
echo '{"clip_id":"test","split":"test","pred_bbox_xywh":[10,10,50,50],"target_bbox_xywh":[12,12,48,48],"pred_type_name":"soft","target_type":"soft"}' > /tmp/test_overlay.json
mkdir -p /tmp/sanity_metric && mv /tmp/test_overlay.json /tmp/sanity_metric/x_overlay.json
python compute_cbd_prediction_metrics.py /tmp/sanity_metric
rm -rf /tmp/sanity_metric
```

### Step 2. HPC 端拉最新 code

```bash
ssh smyin@<HPC>
cd /home2020/home/icube/smyin/projects/ibsafe_cbd_v1
git pull
mamba activate py311cu118
```

### Step 3. 在 HPC 提交 cached inference job

```bash
# 用 schedule_cbd_cached_inference.py 提交 sbatch
# 對 test split 全跑(不傳 --clip-id)
python slurm/schedule_cbd_cached_inference.py \
    --experiment cbd_v2_test_eval \
    --config configs/bsafe_cbd.yaml \
    --weights runs/cbd_v2_baseline/<ts>/run_1/best_cbd.pt \
    --split test \
    --show-gt \
    --gpu h200 h100 a100hgx a100 l40s

# 查 job
squeue -u $USER

# tail 輸出(等 job 開始)
tail -f runs/cbd_v2_test_eval/<new_ts>/run_1/infer_cbd_run_1.out
```

> 注意:`schedule_cbd_cached_inference.py` 跟 `schedule_cbd_train.py` 結構相同,參數可能稍有不同。**先 `python slurm/schedule_cbd_cached_inference.py --help`** 確認 argparser。

### Step 4. 算 metric

```bash
# 找 inference 輸出目錄
ls runs/cbd_v2_test_eval/<new_ts>/run_1/

# 假設 PNG/JSON 在 outputs/cbd_v2_test_overlays/
python compute_cbd_prediction_metrics.py \
    runs/cbd_v2_test_eval/<new_ts>/run_1/outputs/cbd_v2_test_overlays/ \
    --json-output runs/cbd_v2_test_eval/<new_ts>/run_1/metrics.json

# 用 jq 看分組結果
jq '.detection.by_target_type' runs/cbd_v2_test_eval/<new_ts>/run_1/metrics.json
jq '.classification.confusion_matrix' runs/cbd_v2_test_eval/<new_ts>/run_1/metrics.json
```

### 完整四階段 cheatsheet

```bash
# 1. 本機 sanity
python -c "import yaml; print(yaml.safe_load(open('configs/bsafe_cbd.yaml'))['model']['variant'])"

# 2. HPC pull
ssh smyin@<HPC>; cd <project>; git pull; mamba activate py311cu118

# 3. Cached inference 提交
python slurm/schedule_cbd_cached_inference.py \
    --experiment <name> \
    --config configs/bsafe_cbd.yaml \
    --weights <best_cbd.pt> \
    --split test --show-gt

# 4. 算 metric
python compute_cbd_prediction_metrics.py <overlay_dir>/ --json-output metrics.json
jq '.detection.average_of_soft_and_hard' metrics.json
```

---

## 失敗模式速查(常見問題)

| 症狀 | 可能原因 | 排查 |
|---|---|---|
| Inference 跑完所有 PNG 都全黑 / 無 overlay | 模型輸出 NaN(訓練中 grad explode) | 看 train job 的 .out,搜 "nan" / "loss=nan" |
| metric script 報 `No *_overlay.json files found` | 輸出目錄錯了 | `find <runs_dir> -name "*_overlay.json"` |
| metric script 報 `Missing IoU and bbox data` | external mode 沒 GT 仍跑 metric | external mode 不該餵給這個 script |
| Confusion matrix 全部 hard 或全部 soft | 模型「**塌成單一類**」(常見的 overfit / underfit 表現) | 看 train_stats 的 type_accuracy 曲線 |
| Overlay 圖 attention map 是均勻色塊 | box_query 沒學到鎖定能力 | 訓練太短 / lr 設錯 / box loss 權重太小 |
| IoU 很高但 type accuracy 隨機 | type_head 沒在學 / type_ce 權重太小 | 提高 `loss_weights.type_ce` |

---

## 想做更多進階分析?

### 把 attention map 抽出來離線研究

```python
import yaml, torch, sys
sys.path.insert(0, 'src')
from cbd.engine import CBDTrainer, load_config

config = load_config('configs/bsafe_cbd.yaml')
trainer = CBDTrainer(config)
trainer.load_checkpoint('runs/.../best_cbd.pt')
prediction = trainer.predict_record('test', '<clip_id>')

attention_map = prediction['pred_attention_map']   # tensor (16, 16)
center_heatmap = prediction['pred_center_heatmap']  # tensor (16, 16)
# 拿去做 matplotlib heatmap、量化分析、cluster 等
```

### 對多個 checkpoint 做 ensemble(若你訓了 multi-run)

stage 2 目前**沒有內建 ensemble**,但結構允許:多個 run 各自 predict 同一 record,把 `pred_box_norm_cxcywh` 平均(或 weighted average by `pred_type_confidence`)。這是個合理的 future 工作。

### 跨 dataset evaluation

訓在 bsafe + icglceaes,但想單獨評估某個 source:

```yaml
# 改 configs/bsafe_cbd.yaml 暫時版本
data:
  test_sources: icglceaes      # 原本 bsafe,改成 icglceaes
```

然後跑 inference 就只跑 ICG-LC-EAES 的樣本——對「**域外泛化能力**」的單純評估。

---

## 這一章你需要帶走的重點

1. **兩種推論模式**:cached(走 dataset)/ external clip(走 stage1 即時 + 手動 clip 路徑)——終極輸出格式相同
2. **單 clip vs 全 split**:cached 模式可單跑(指定 `--clip-id`)或全跑(不指定,自動掃 split)
3. **Overlay PNG = 左 overlay + 右 heatmap + 上方標題條**,設計目的是**並排看「**框在哪**」跟「**模型認為中心在哪**」**
4. **JSON 自包含**——把 pred + GT + IoU + type prob 全寫進去,metric script 不再回查 dataset
5. **Detection metric** 用 IoU 當分數做 mAP,**分 soft/hard 兩類分組評估**——臨床難度的差異很重要
6. **Classification metric** 用 macro precision/recall/F1 + confusion matrix,因為類別不平衡
7. **AP 計算不是 COCO 風格**(沒有 confidence 排序)——單目標單 box 任務的合理簡化
8. **失敗模式優先看**:模型塌成單一類 / attention 均勻色塊 / NaN 全黑
9. **4 階段實操**:本機 sanity → HPC pull → schedule_cbd_cached_inference.py → compute_cbd_prediction_metrics.py
10. **進階用法**:attention map 離線分析、ensemble(目前無內建)、跨 dataset evaluation

---

## 進一步深挖的線索

- **COCO mAP 標準**:[COCO Detection Evaluation Spec](https://cocodataset.org/#detection-eval)——對比理解本專案 AP 為何簡化
- **想視覺化 train 過程的 attention map 演化**:寫個 epoch hook 在 `engine.py:332` 之後把 attention_map sample 出來存
- **想加入 ensemble**:在 `engine.py` 旁開個 `CBDEnsembleTrainer`,接收多個 checkpoint 路徑,predict 時平均 boxes
- **臨床效用 metric**(distance from CBD true center / bile duct injury risk reduction)目前沒實作,**這是 stage 2 評估的真正缺口**——本專案的 IoU/mAP 是「**ML 標準指標**」,不是「**臨床指標**」

---

## 對話脈絡記錄

- **2026-05-27**:第 8 章對應 stage 1 第 6 章,但 stage 2 因為多了 type classification + heatmap 輸出,**JSON 結構與 metric 分類**比 stage 1 複雜。重點放在「**為什麼這樣設計 metric**」(分組看 soft/hard、macro 而非 micro)而不是「**怎麼跑**」。
- 實操腳本壓在「**4 階段 cheatsheet**」一個 code block——可以直接複製貼上。
- 失敗模式速查表是新加的——前面章節沒有,這裡放是因為**推論結果是模型訓練品質的最直接 surface**,出問題的線索很多。

---

## 程式碼速查總表

### 推論腳本對照

| 用途 | 腳本 | Slurm 排程 |
|---|---|---|
| Cached(單 clip) | `infer/infer_cbd.py --clip-id ...` | `slurm/schedule_cbd_cached_inference.py` |
| Cached(全 split) | `infer/infer_cbd.py`(不傳 --clip-id) | `slurm/schedule_cbd_cached_inference.py` |
| External clip / video | `infer/infer_cbd_clip.py --clip ...` | `slurm/schedule_cbd_inference.py` |

### infer_cbd.py CLI

| 參數 | 必要 | 預設 | 意義 |
|---|---|---|---|
| `--config` | ✓ | — | YAML 配置 |
| `--weights` | 可選 | `<output_dir>/best_cbd.pt` | checkpoint |
| `--split` | 可選 | `test` | 用哪個 split 找 clip |
| `--clip-id` | 可選 | — | 單 clip 模式 |
| `--output` | ✓ | — | PNG 路徑或目錄 |
| `--show-gt` | flag | False | overlay GT box |

### Metric script CLI

| 參數 | 意義 |
|---|---|
| `predictions_dir`(positional) | 含 `*_overlay.json` 的目錄 |
| `--json-output` | metric 表存哪 |

### JSON payload 欄位(預期所有都有)

| 欄位 | 型 | 範圍 / 形式 |
|---|---|---|
| `clip_id` | str | dataset 給的 |
| `split` | str | train/val/test |
| `pred_box_norm_cxcywh` | [f, f, f, f] | [0,1] |
| `pred_bbox_xywh` | [f, f, f, f] | 像素 |
| `pred_type_name` | str | "soft" / "hard" |
| `pred_type_probs` | dict | {"soft": f, "hard": f} |
| `pred_type_confidence` | f | softmax 最高值 |
| `pred_center_cell_confidence` | f | center cell softmax 最高值 |
| `annotation_id` | int | 從 record 帶 |
| `image_id` | int | 從 record 帶 |
| `target_box_norm_cxcywh` | [f, f, f, f] | 只 cached 有 |
| `target_bbox_xywh` | [f, f, f, f] | 只 cached 有 |
| `target_type` | str | 只 cached 有 |
| `iou` | f | 只 cached 有 |

### 快速驗證命令

```bash
# 看 best checkpoint 的 epoch 跟 val loss
python -c "
import torch
ckpt = torch.load('runs/.../best_cbd.pt', map_location='cpu')
print('keys:', list(ckpt.keys()))
print('config.output:', ckpt['config'].get('output'))
"

# 拿一份 JSON 看完整欄位
jq '.' runs/cbd_v2_test_eval/<ts>/run_1/<overlay_dir>/<clip_id>_overlay.json

# 把 metric.json 印成表格
python -c "
import json
m = json.load(open('metrics.json'))
print('num samples:', m['num_samples'])
print('counts:', m['counts_by_target_type'])
print('overall detection:', m['detection']['overall'])
print('classification:', {k: m['classification'][k] for k in ('accuracy','macro_f1')})
"
```

---

> ConvNeXt + Spatiotemporal Transformer 教學的**第一條完整鏈路**就到這裡——backbone → fusion → transformer → multi-head → training → inference → metric。
>
> **下一階段(Stage 3)** 會展開:`src/cbd_rtdetrv4/` 的 RT-DETRv4 替代 backbone 線、它跟 ConvNeXt 線的對比、何時用哪一條、實驗結果如何。
