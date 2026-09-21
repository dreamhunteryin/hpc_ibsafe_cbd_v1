# 第 7 章 — 訓練主迴圈 + multi-task loss

> 撰寫日期:2026-05-27
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:把 stage 2 怎麼**真正訓練起來**講清楚——5 個 loss 各自的意義與權重、optimizer 與 scheduler、DDP / bf16 / gradient accumulation,以及 Slurm 提交流程。

---

## 這一章的位置

第 6 章把模型黑盒子打開——`CBDV2Spatiotemporal` 的 forward 已經拆解完。接下來就是**訓練動力學**:loss 怎麼算、optimizer 怎麼更新、整個 epoch loop 怎麼跑、HPC 上怎麼提交。

對應 stage 1 的第 5 章(`stage1_SAM3+LoRA_Fine-tuning/05_training_loop_and_slurm.md`),只是 stage 2 的細節不一樣——5 個 loss、差別學習率、bf16、cosine scheduler、Slurm 排程的設定都需要單獨拆。

---

## Trainer 鳥瞰

`src/cbd/engine.py:146-202` 的 `CBDTrainer.__init__` 做了五件事:

```
1. 解析配置(data/model/training/output/hardware)
2. 建模型(CBDBoxModel)並搬到 device
3. 拆 backbone 參數 vs 新增層參數 → 各自一個 param group
4. 建 AdamW + (可選的) CosineAnnealingLR
5. 設定 output_dir + val_stats.jsonl + best_val_loss 追蹤
```

接著 `train()` 方法(`engine.py:321-347`)是核心主迴圈:

```
for epoch in 1..num_epochs:
    train_stats = run_epoch(train_loader, train=True)
    val_stats   = run_epoch(val_loader,   train=False)
    log_val_stats(epoch, train_stats, val_stats)
    save_checkpoint("last_cbd.pt")
    if val_stats["loss"] < best_val_loss:
        save_checkpoint("best_cbd.pt")
    scheduler.step()
    若 early stopping patience 用完 → break
```

很直白——和大多數 vision tutorial 看到的 epoch loop 一樣。stage 2 的特殊之處在 `run_epoch` 的內容(尤其 5 個 loss)和**參數分組策略**。

---

## 工程師原文(本章相關)

> RGB features were extracted using a ConvNeXt-Small backbone, whereas liver and gallbladder masks were encoded by a shallow convolutional branch and fused with RGB representations. The fused spatiotemporal features were processed by a lightweight bidirectional transformer. **A clip-level classification head predicted poor versus good CBD fluorescence visualization, and a learned query regressed the CBD bounding box in the final frame.**

訓練面要落實「**分類 head + 回歸 query**」這個 multi-task 設定,就會用到 5 個 loss——本章重點。

---

## 5 個 loss 各自在訓練什麼?

`src/cbd/engine.py:41-51`(預設權重)+ `bsafe_cbd.yaml:73-78`:

```python
default = {
    "box_l1":      5.0,
    "box_giou":    2.0,
    "center_ce":   1.0,
    "heatmap_bce": 1.0,
    "type_ce":     1.0,
}
```

| Loss | 形式 | target | 監督對象 | 直覺意義 |
|---|---|---|---|---|
| `box_l1` | L1(逐分量絕對差) | (cx, cy, w, h) normalized | `pred_boxes` | 「**位置別差太遠**」——直接像素級對齊 |
| `box_giou` | 1 − GIoU | xyxy | `pred_boxes` | 「**框的形狀與位置都要對**」——比 IoU 更穩定的版本 |
| `center_ce` | cross-entropy | grid cell index | `center_cell_logits (B,256)` | 「**中心位於 16×16 哪一格**」(輔助 loss) |
| `heatmap_bce` | BCE-with-logits | 高斯 heatmap (σ=1) | `center_heatmap_logits (B,16,16)` | 「**中心位置的軟分布**」(輔助 loss) |
| `type_ce` | cross-entropy(可空缺) | soft / hard label | `type_logits (B,2)` | 「**整段 ICG 顯影品質判斷**」 |

### box_l1 vs box_giou 為什麼一起用?

L1 對「**位置誤差**」敏感但對「**大小尺度**」不夠精確——一個 0.1 的位置誤差,在大 box 跟小 box 上是完全不同的「**佔比**」。
GIoU 對「**框與框的重疊比例**」敏感,但訓練早期當預測完全跟 GT 沒重疊時 IoU=0、梯度消失——這時候 L1 還能提供「**至少把框往對的方向推**」的訊號。

所以 DETR 系列傳統:**L1 + GIoU 同時用,L1 給早期方向、GIoU 給後期精修**。

權重 `5.0 + 2.0 = 7.0` 給 box——明顯比 auxiliary 三個(各 1.0,合計 3.0)重——這就是「**主任務 vs 輔助任務的權重設計**」,主任務佔約 70% 訓練訊號。

### 為什麼還要 center_ce 跟 heatmap_bce?

第 5 章已經解釋過「**multi-task auxiliary loss**」的概念,這裡補上**訓練動力學的視角**:

- **梯度路徑不同**:`box_l1/box_giou` 的梯度經過 `box_head → box_attention → last_tokens`——只更新「**box query 路徑**」上的權重。`center_ce/heatmap_bce` 的梯度直接從 `last_tokens` 出發(`center_cell_head` 跟 `center_heatmap_head` 直接吃 `last_tokens`),會更新**所有 256 個 spatial cell 的特徵品質**
- **訊號類型不同**:bbox 是「**4 個連續值**」訊號,center cell 是「**256 類分類**」訊號——分類訊號更密、更穩定,有助訓練早期收斂
- **多任務一致性**:模型必須讓「**box 預測**」「**center 分類**」「**heatmap 預測**」三個出口都指向同一個 CBD 位置——這種「**多角度監督**」會抑制過擬合

`build_center_targets`(`common.py:428-449`)會根據 GT box 中心算出兩個 target:
- `center_indices`:把中心位置 (cx, cy) 投影到 16×16 grid,取那一格的 index(0~255)
- `heatmaps`:以那一格為中心,半徑 σ=1 的二維高斯分布(16×16)

```python
sigma = 1.0  # bsafe_cbd.yaml:72
dist_sq = (xx - cx).pow(2) + (yy - cy).pow(2)
heatmap = torch.exp(-0.5 * dist_sq / sigma_sq)
```

σ=1 表示「**中心格分數最高,周圍 1~2 格快速下降**」——這個寬度設計讓 BCE 訊號比硬分類(`center_ce`)更平滑,對訓練收斂有幫助。

### type_ce 的特殊處理:可缺值

`engine.py:108-118`:

```python
if model_output.type_logits is not None:
    valid_type_mask = batch["target_type_label"] >= 0
    if valid_type_mask.any():
        type_loss = F.cross_entropy(
            model_output.type_logits[valid_type_mask],
            batch["target_type_label"][valid_type_mask],
        )
```

注意這兩個檢查:

1. `model_output.type_logits is not None`——v1_global_pool 沒有 type_head,跳過
2. `target_type_label >= 0`——某些資料沒有 soft/hard 標注(`common.py:26` `UNLABELED_TARGET_TYPE = -1`),那些樣本不計入 type loss

**為什麼資料會缺標?**——`configs/bsafe_cbd.yaml:18-27` 的 `icglceaes` 資料來源沒有 soft/hard 標注(只有 bbox),它們會用 `UNLABELED_TARGET_TYPE`(`dataset.py:138-143`)。同個 batch 內可能混有「**bsafe 帶 type 標注 + icglceaes 不帶**」的樣本,所以要 mask 掉。

**外科類比**:就像研究多中心資料時,A 醫院記錄了顯影品質,B 醫院只記錄了 bbox——你不能因為 B 醫院缺一欄就丟掉它的整筆資料,而是「**有什麼標就學什麼**」。

### 總 loss 怎麼加總

`engine.py:131-134`:

```python
total_loss = loss_dict["loss"]                                # = 5.0 * box_l1 + 2.0 * box_giou
total_loss = total_loss + loss_weights["center_ce"]   * center_ce
total_loss = total_loss + loss_weights["heatmap_bce"] * heatmap_bce
total_loss = total_loss + loss_weights["type_ce"]     * type_loss
```

**簡單加權和**——沒有 GradNorm、沒有 uncertainty weighting、沒有 curriculum schedule。這是個保守的選擇:**權重靠 yaml 配置調**(`bsafe_cbd.yaml:73-78`),不依賴複雜的多任務權重學習機制。

> 想實驗 loss 權重調整:改 `configs/bsafe_cbd.yaml` 對應欄位,不要在 `engine.py` 寫死(R7 規則)。

---

## 參數分組:backbone vs 新增層用不同學習率

`src/cbd/engine.py:157-188` 是 stage 2 訓練最 subtle 的設計:

```python
weight_decay = float(self.training_config.get("weight_decay", 1e-4))
backbone_params = self.model.backbone_trainable_parameters()
backbone_ids = {id(parameter) for parameter in backbone_params}
new_params = [
    parameter for parameter in self.model.parameters()
    if parameter.requires_grad and id(parameter) not in backbone_ids
]
if backbone_params:
    optimizer_params = [
        {"params": new_params,      "lr": new_layers_lr, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": backbone_lr,   "weight_decay": weight_decay},
    ]
```

預設值(`bsafe_cbd.yaml:61-63`):

| 參數組 | learning rate | 例子 |
|---|---|---|
| backbone(`last_stage`) | `backbone_lr = 1e-5` | ConvNeXt 最後一個 stage 的 params |
| 新增層 | `new_layers_lr = 1e-4` | mask_encoder, fusion, transformer, 5 個 head, position embedding |

**比例 10:1**——新層學快 10 倍,pretrained backbone 學慢 10 倍。

### 為何要差別學習率?

`backbone_trainable_parameters()`(`model.py:203-204`)只回傳 `requires_grad=True` 的 backbone 參數。在 `last_stage` 模式下,只有 ConvNeXt 最後一個 stage(`features[6:]`)會被 unfreeze,前面的 stage 凍住。

對這個「**僅最後一階段被 unfreeze 的 pretrained backbone**」:

- 它本來在 ImageNet 上學得很好,**不該重新學**——只該「**微調」到本任務的 surgical 影像分布**
- 用 `1e-5` 這種小 lr 防止它「**忘掉 ImageNet 學到的視覺先驗**」(catastrophic forgetting)

對「**新增層**」(transformer, heads, mask_encoder...):

- 它們是隨機初始化,什麼都不會,**需要更大的 lr 才能在 30 epoch 內收斂**
- 用 `1e-4` 給它們充足的學習速率

這就是 transfer learning 圈的 **discriminative fine-tuning** 思路:**根據參數的「來歷」決定 lr**。

### 沒有 backbone trainable 怎麼辦?

`engine.py:181-188` 處理 `freeze_all` 模式:

```python
else:
    optimizer_params = [{"params": [parameter ...], "lr": learning_rate, ...}]
```

所有 trainable 參數(都是新層)用同一個 `learning_rate`(`bsafe_cbd.yaml:61` 預設 `1e-4`)。

---

## Optimizer + Scheduler

`engine.py:190-196`:

```python
self.optimizer = AdamW(optimizer_params)
scheduler_name = str(self.training_config.get("lr_scheduler", "cosine")).lower()
self.scheduler = (
    CosineAnnealingLR(self.optimizer, T_max=max(1, int(self.training_config.get("num_epochs", 20))))
    if scheduler_name == "cosine"
    else None
)
```

**AdamW**:Adam 的 weight decay 修正版,業界 transformer 訓練的標準選擇。`weight_decay = 1e-4`(`bsafe_cbd.yaml:64`)——溫和的正則化。

**CosineAnnealingLR**:lr 在 num_epochs 內從初始值平滑遞減到 0(cosine 曲線):

```
epoch 0:  lr = initial
epoch 15: lr ≈ initial / 2
epoch 30: lr ≈ 0
```

兩個 param group(backbone / new layers)會**各自獨立 cosine 遞減**——backbone 從 1e-5 → 0,new layers 從 1e-4 → 0。

**外科類比**:像是訓練 fellow 的學習進度——剛開始給高強度新挑戰,中段強度減半讓他鞏固,後段給最簡單的任務做最後微調。

---

## bf16 mixed precision

`engine.py:30-38`:

```python
def make_autocast(device, training_config):
    if device.type != "cuda":
        return contextlib.nullcontext()
    precision = str(training_config.get("mixed_precision", "bf16")).lower()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
```

預設 `bf16`(`bsafe_cbd.yaml:67`)。

**為何 bf16 不用 fp16?**:

| 屬性 | fp16 | bf16 |
|---|---|---|
| 數值範圍 | 受限(易 overflow) | 與 fp32 相同 |
| 精度 | 較高(尾數 10 bit) | 較低(尾數 7 bit) |
| 需要 GradScaler? | 是 | 否 |
| 硬體需求 | 大多數 GPU | A100/H100/L40s 等 |

bf16 在 surgical AI 領域 + 現代 GPU(H100/A100)成為新預設——**範圍大避免 overflow,且不用 grad scaler 的麻煩**。

`engine.py:264-268` 在 forward + loss 計算階段包進 autocast:

```python
with make_autocast(self.device, self.training_config):
    model_output = self.model(batch["rgb"], batch["masks"])
    loss_dict = compute_cbd_losses(model_output, batch, self.training_config)
    ...
    loss = loss_dict["loss_total"]
```

`backward()` **不在** autocast 內——梯度算回 fp32 主權重。這是標準寫法。

> 注意:Slurm 排程腳本(`schedule_cbd_train.py:18`)允許 v100,但 v100 不支援 bf16。若你想跑 v100,**要把 yaml 改成 `mixed_precision: fp16` 或省略 mixed precision**。

---

## Gradient accumulation + grad clip

`engine.py:257, 270-279`:

```python
accumulation_steps = max(1, int(self.training_config.get("gradient_accumulation_steps", 1)))
...
if train:
    (loss / accumulation_steps).backward()
    if step % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            float(self.training_config.get("max_grad_norm", 1.0)),
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
```

預設 `gradient_accumulation_steps = 1`(`bsafe_cbd.yaml:70`)——即**不累積**。每個 batch 都 step。

**為何提供 accumulation?**——stage 2 的 token 數爆(6401)+ batch_size=2 + clip_len=25 已經佔很多 VRAM。如果你想模擬「**等效 batch_size = 4 或 8**」但 VRAM 不夠,把 `gradient_accumulation_steps = 2` 或 `4`——backward 累積但 optimizer step 只做 1 次。

`max_grad_norm = 1.0`(`bsafe_cbd.yaml:68`)——**梯度範數超過 1.0 就 clip**。Transformer 訓練的標配,防止偶發大梯度炸 lr。

### 處理「**epoch 末殘留**」

`engine.py:297-303` 有個常被忽略的 detail:

```python
if train and count % accumulation_steps != 0:
    torch.nn.utils.clip_grad_norm_(...)
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
```

如果 epoch 結束時還剩 N < accumulation_steps 步沒 step,**手動補上一次 step**——避免那幾筆樣本的梯度白算。

---

## 為什麼這個版本「沒有」DDP?

stage 1 用了 DDP(`tutorial/stage1_SAM3+LoRA_Fine-tuning/05_training_loop_and_slurm.md`),stage 2 卻**單卡訓練**——schedule_cbd_train.py 的 sbatch 設定:

```bash
#SBATCH --gres=gpu:1     # 只要 1 張 GPU
srun python train/train_cbd.py --config <yaml>   # 不是 torchrun --nproc_per_node=N
```

`engine.py` 通篇沒有 `DistributedDataParallel` 或 `init_process_group`。

**為什麼?**——三個原因:

1. **單卡 VRAM 夠用**:batch_size=2, clip_len=25, input_size=512, bf16 → V100 32GB / A100 40GB 都能放
2. **資料量不大**:bsafe + ICG-LC-EAES 加起來規模不到 ImageNet 的零頭,單卡 30 epoch 一兩天能跑完
3. **stage 2 還在迭代**:現階段 multiple variants(rtdetr 系列、不同 backbone mode)要 A/B 比較,**單卡單 job 比 multi-GPU job 更靈活**

> 想擴成 DDP 也可以——`engine.py` 結構乾淨,加 `DDP(self.model)` + `DistributedSampler` 就能跑。但**不要為了 DDP 而 DDP**——確認單卡真的瓶頸了再說。

---

## Epoch loop 完整拆解

`engine.py:238-312` 的 `run_epoch`(刪掉一些 metric 累計細節):

```python
def run_epoch(self, loader, train):
    self.model.train(train)
    totals = {...}                # loss / metric 累加器
    per_type_sums = {soft/hard 分組統計}
    count = 0
    accumulation_steps = ...

    if train:
        self.optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader), start=1):
        batch = move_batch_to_device(batch, self.device)
        with make_autocast(self.device, self.training_config):
            model_output = self.model(batch["rgb"], batch["masks"])
            loss_dict = compute_cbd_losses(model_output, batch, self.training_config)
            metric_tensors = compute_box_metric_tensors(model_output.pred_boxes.detach(), batch["target_box"])
            loss = loss_dict["loss_total"]

        if train:
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0:
                clip_grad_norm_(...)
                self.optimizer.step()
                self.optimizer.zero_grad(...)

        # 累計 loss / metric / per-type 統計 ...

    # epoch 末殘留處理 (見前一節)

    return stats   # dict
```

### per-type 統計怎麼算

`engine.py:287-293`:

```python
for type_label, type_name in enumerate(("soft", "hard")):
    mask = batch["target_type_label"] == type_label
    if not mask.any():
        continue
    per_type_sums[type_name]["count"] += int(mask.sum().item())
    for metric_name in ("mean_iou", "center_error", "box_error"):
        per_type_sums[type_name][metric_name] += float(metric_tensors[metric_name][mask].sum().detach().item())
```

對每個 batch,把樣本按 type label(soft / hard)分開,**分組計算 IoU / center error / box error**。這樣 val 日誌會有:

```
val_soft_mean_iou: 0.43
val_hard_mean_iou: 0.61
val_soft_count:    18
val_hard_count:    32
```

**為什麼分組看?**——臨床上 soft / hard 兩種顯影品質的「**期待精度**」可能不同。模型若對 hard 預測得很好(顯影清楚的容易),但對 soft 預測得很差(顯影模糊才是真正考驗),這是個重要的失敗模式。**分組指標才能看出這個**。

---

## Checkpoint 與 early stopping

`engine.py:228-236`:

```python
def save_checkpoint(self, filename):
    path = self.output_dir / filename
    torch.save({"model": self.model.state_dict(), "config": self.config}, path)
    return path

def load_checkpoint(self, path):
    checkpoint = torch.load(path, map_location=self.device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    self.model.load_state_dict(state_dict)
```

每個 epoch 結束會儲存:

- `last_cbd.pt`:**最新一個 epoch** 的權重(總是覆寫)
- `best_cbd.pt`:`val_stats["loss"]` 至今最低的版本

兩個都帶上 `config`——之後 inference 時可以**自證自己用哪個 config 訓出來的**。

### Early stopping

`engine.py:325-326, 341-346`:

```python
patience = int(self.training_config.get("early_stopping_patience", num_epochs))
epochs_without_improvement = 0
...
if val_stats["loss"] < self.best_val_loss:
    self.best_val_loss = val_stats["loss"]
    self.best_epoch = epoch
    epochs_without_improvement = 0
    self.save_checkpoint("best_cbd.pt")
else:
    epochs_without_improvement += 1
...
if epochs_without_improvement >= patience:
    break
```

預設 `early_stopping_patience = 5`(`bsafe_cbd.yaml:66`)——**val_loss 連續 5 epoch 沒進步就停**。

注意:**指標是 val loss 不是 val IoU**。如果你的應用更在意 IoU 而非 loss,可以改 `engine.py:335` 那行,或者 yaml 加 `early_stopping_metric` 自訂(目前沒這個 hook,要加要改 engine)。

### val_stats.jsonl

`engine.py:200, 314-319`:

```python
self.val_stats_path = self.output_dir / "val_stats.jsonl"
...
def log_val_stats(self, epoch, train_stats, val_stats):
    payload = {"epoch": epoch}
    payload.update({f"train_{key}": value for key, value in train_stats.items()})
    payload.update({f"val_{key}": value for key, value in val_stats.items()})
    with open(self.val_stats_path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")
```

每 epoch 一行 JSON,append-mode 寫 `output_dir/val_stats.jsonl`。可以用 jq 或 pandas 解析做 loss curve。

```bash
# 看 epoch 跟兩種 type 的 val IoU
jq -r '[.epoch, .val_soft_mean_iou, .val_hard_mean_iou] | @tsv' outputs/<exp>/val_stats.jsonl
```

---

## HPC 提交流程

從本機改完 code → push → HPC 端 `git pull` → 提交 sbatch。`slurm/schedule_cbd_train.py` 的工作流:

```
schedule_cbd_train.py
   │
   ├─ 1. 解析 --experiment / --num-runs / --gpu / --config
   ├─ 2. 為每個 run 建 runs/<experiment>/<timestamp>/run_N/ 資料夾
   ├─ 3. 複製 config 到該資料夾(獨立的 config snapshot)
   ├─ 4. 改寫 config 內 output.output_dir 指到該 run 資料夾
   ├─ 5. 產生 sbatch 腳本 train_cbd_<run>.job
   └─ 6. sbatch 提交(除非 --no-submit)
```

### 為什麼每個 run 一個獨立 config copy?

`schedule_cbd_train.py:91-99`:

```python
shutil.copyfile(args.config, config_path)        # 複製
with open(config_path, "r") as handle:
    config = yaml.safe_load(handle)
config.setdefault("output", {})["output_dir"] = str(save_dir)
with open(config_path, "w") as handle:
    yaml.safe_dump(config, handle)
```

每個 run 都有**自己的 config copy + 改寫過的 output_dir**——確保:

1. 同一個 experiment 跑多 run 時,**輸出彼此不衝突**
2. 配置被改了你也能回頭找——`runs/<exp>/<timestamp>/run_1/<config>.yaml` 是當時跑的真正版本(不是現在 repo 裡那份)

這對 surgical AI 的「**可重現性**」很重要——半年後你問「**那個結果是哪個 config 跑的?**」,直接看 runs 資料夾就有答案。

### 多 GPU 類型 fallback

`schedule_cbd_train.py:17-18`:

```python
if args.gpu is not None:
    gpu = "|".join(f"gpu{x}" for x in args.gpu)
else:
    gpu = "gpuh200|gpuh100|gpua100hgx|gpua100|gpua40|gpul40s|gpuv100"
```

`#SBATCH --constraint=` 用 `|` 表示「**任一可用**」——Slurm 會挑空閒的 GPU 給你,**不挑剔型號**。

**注意**:如果你訓練配 bf16,卻被 Slurm 排到 v100(不支援 bf16),會在 train 開始時 crash。要嘛 `--gpu h200 h100 a100hgx a100 l40s` 限定 GPU,要嘛 yaml 改成 fp16。

### 提交命令範例

```bash
# 提交一個 30-epoch CBD v2 訓練 job
python slurm/schedule_cbd_train.py \
    --experiment cbd_v2_baseline \
    --num-runs 1 \
    --config configs/bsafe_cbd.yaml \
    --gpu h200 h100 a100hgx a100 l40s   # 排除 v100

# 提交 3 個獨立 run(不同 random seed 的多次訓練)
python slurm/schedule_cbd_train.py \
    --experiment cbd_v2_seed_sweep \
    --num-runs 3 \
    --config configs/bsafe_cbd.yaml

# 只生 sbatch 不真送
python slurm/schedule_cbd_train.py \
    --experiment debug \
    --num-runs 1 \
    --config configs/bsafe_cbd.yaml \
    --no-submit
```

### 監測訓練進度

```bash
# 看 job 狀態
squeue -u $USER

# tail 訓練輸出(tqdm + per-epoch json)
tail -f runs/cbd_v2_baseline/<timestamp>/run_1/train_cbd_run_1.out

# 把 val_stats.jsonl 印成表格
jq -r '[.epoch, .train_loss, .val_loss, .val_mean_iou] | @tsv' \
    runs/cbd_v2_baseline/<timestamp>/run_1/val_stats.jsonl
```

---

## 訓練前 sanity check

開大訓練前的本機 / debug 確認(R12 規則:本機只跑 CPU smoke test):

```bash
# 1. 確認 yaml 解析無誤
python -c "import yaml; print(yaml.safe_load(open('configs/bsafe_cbd.yaml'))['model'])"

# 2. 確認 model 能 instantiate(CPU OK,不需要 GPU)
python -c "
import yaml
from cbd.model import CBDBoxModel
config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
model = CBDBoxModel(config['model'])
print('Total params:', sum(p.numel() for p in model.parameters()))
print('Trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
"

# 3. 確認 Dataset 能 build(會檢查資料路徑)
python -c "
import yaml
from cbd.dataset import CBDDataset
config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
ds = CBDDataset.from_config(config, split='val')
print('val len:', len(ds))
"
```

在 HPC 端要做真正的 GPU sanity check,可以先用很小的 num_epochs:

```bash
# 改 yaml 暫時 num_epochs: 1 + early_stopping_patience: 1 + num_workers: 0
# 然後 schedule_cbd_train.py 提交 → 確認能跑完一個 epoch 不 crash
```

---

## 這一章你需要帶走的重點

1. **5 個 loss** 各管不同訊號:box_l1 + box_giou 是主任務(權重 7.0)、center_ce + heatmap_bce + type_ce 是輔助(各 1.0)
2. **box_l1 + box_giou 一起用** 因為兩者互補:L1 給早期方向、GIoU 給後期精修
3. **center_ce + heatmap_bce** 提供更密的分類訊號,促使 spatial token 學到正確位置資訊
4. **type_ce 可缺值**(`UNLABELED_TARGET_TYPE = -1`)——多來源資料時,缺顯影品質標注的樣本會被 mask 掉
5. **參數分組**:backbone(`last_stage` 模式下的最後一階段)用 `1e-5`、新層用 `1e-4`——discriminative fine-tuning,防止 catastrophic forgetting
6. **AdamW + cosine + bf16 + grad_clip=1.0 + accumulation_steps=1** 是 stage 2 預設組合
7. **單卡 DDP-free**——資料量 / VRAM 足夠,迭代靈活度更重要
8. **`val_stats.jsonl` 每 epoch 一行**,含 train/val 各自 loss/metric + per-type 統計
9. **best_cbd.pt = val_loss 最低**;指標想換 IoU 要改 engine.py
10. **HPC 流程**:`schedule_cbd_train.py` 每個 run 獨立 config copy + output_dir,可重現性的關鍵設計
11. **GPU 限制**:bf16 不支援 v100,排程時用 `--gpu` 排除或改 fp16

---

## 進一步深挖的線索

- **DETR set prediction loss**:Carion et al. 原論文的 L1 + GIoU 設計來源
- **Discriminative fine-tuning**:Howard & Ruder, *Universal Language Model Fine-tuning*(ULMFiT, 2018)——分層學習率的開山設計
- **bf16 vs fp16 在 transformer 訓練的對比**:NVIDIA 的 Mixed Precision Training Guide(自家文件)
- **想實驗 loss 權重**:改 `configs/bsafe_cbd.yaml:73-78`,跑多個 run 比較 val IoU 跟 type accuracy 的 trade-off
- **想看單一 loss curve**:val_stats.jsonl 用 pandas 讀出來畫
- **想換 scheduler**:目前只支援 cosine 跟 None,加 `step` / `warmup_cosine` 要改 `engine.py:191-196`

---

## 對話脈絡記錄

- **2026-05-27**:第 7 章對應 stage 1 的第 5 章(訓練主迴圈 + Slurm),但 stage 2 的 multi-task loss 是核心差異。**5 個 loss 的權重設計、param group 差別學習率、bf16 GPU 兼容性、val_stats.jsonl 的 per-type 分組統計**——這四個是 stage 2 訓練的特異點,優先講透。
- 不放完整 sbatch 腳本內文(stage 1 第 5 章已經有等價示範),只給命令範例 + 監測方式。

---

## 程式碼速查總表

### Loss 結構(`src/cbd/engine.py:94-143`)

| Loss | 形式 | target 來源 | 對應 head |
|---|---|---|---|
| `box_l1` | `F.l1_loss(pred_boxes, target_box)` | `target_box` (B, 4) | `box_head` |
| `box_giou` | `1 − GIoU(pred_xyxy, target_xyxy)` | `box_cxcywh_to_xyxy(target_box)` | `box_head` |
| `center_ce` | `F.cross_entropy(logits, indices)` | `build_center_targets(target_box).indices` | `center_cell_head` |
| `heatmap_bce` | `F.binary_cross_entropy_with_logits(logits, heatmaps)` | `build_center_targets(target_box).heatmaps` | `center_heatmap_head` |
| `type_ce` | `F.cross_entropy(logits[mask], labels[mask])` | `target_type_label`(mask `>= 0`) | `type_head` |

### 訓練配置(`configs/bsafe_cbd.yaml`)

| key | 預設 | 意義 |
|---|---|---|
| `batch_size` | 2 | 受 VRAM 限制(token 數大) |
| `num_workers` | 4 | DataLoader 子程序數 |
| `learning_rate` | 1.0e-4 | freeze_all 模式下唯一 lr |
| `backbone_lr` | 1.0e-5 | last_stage / full 模式下 backbone 的 lr |
| `new_layers_lr` | 1.0e-4 | last_stage / full 模式下新層的 lr |
| `weight_decay` | 1.0e-4 | AdamW 的 weight_decay |
| `num_epochs` | 30 | 訓練輪數 |
| `early_stopping_patience` | 5 | val_loss 連續沒進步幾 epoch 就停 |
| `mixed_precision` | bf16 | 也接受 fp16 / None |
| `max_grad_norm` | 1.0 | gradient clipping |
| `lr_scheduler` | cosine | 也接受 None |
| `gradient_accumulation_steps` | 1 | VRAM 不夠時調高 |
| `augmentation_level` | 2 | 0~3,影響旋轉/縮放/位移強度 |
| `center_heatmap_sigma` | 1.0 | heatmap 寬度(grid cell 單位) |
| `loss_weights.*` | 5/2/1/1/1 | 5 個 loss 的權重 |

### Sbatch 設定(`slurm/schedule_cbd_train.py`)

| 項目 | 設定 |
|---|---|
| Partition | `pri2021gpu`,account `qoscammagpu2` |
| GPU | 1 張,可指定型號或全選 |
| CPU | 8 cores |
| 記憶體 | 32G |
| 環境 | `mamba activate py311cu118` |

### 訓練輸出檔案結構

```
runs/<experiment>/<YYYY-MM-DD-HH-MM>/run_<N>/
├── <config>.yaml                # 此 run 用的 config snapshot
├── train_cbd_run_<N>.job        # sbatch 腳本
├── train_cbd_run_<N>.out        # stdout (tqdm + json)
├── best_cbd.pt                  # 最佳 val loss 權重
├── last_cbd.pt                  # 最新一 epoch 權重
└── val_stats.jsonl              # 每 epoch 一行 JSON 統計
```

### 快速驗證命令

```bash
# 解析配置 + 跑一個 train iteration 的偽 batch 確認 forward+backward 不 crash
python -c "
import yaml, torch
from cbd.engine import CBDTrainer

config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
trainer = CBDTrainer(config)
B, T, S = 1, 25, 512
batch = {
    'rgb':   torch.randn(B, T, 3, S, S),
    'masks': torch.randn(B, T, 2, S, S),
    'target_box':        torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    'target_type_label': torch.tensor([0], dtype=torch.long),
}
from cbd.engine import move_batch_to_device, compute_cbd_losses, compute_box_metric_tensors
batch = move_batch_to_device(batch, trainer.device)
model_output = trainer.model(batch['rgb'], batch['masks'])
loss_dict = compute_cbd_losses(model_output, batch, trainer.training_config)
print({k: float(v) for k, v in loss_dict.items() if hasattr(v, 'item')})
"
```

---

> 下一章節主題:**推論流程 + 評估指標**——`infer/infer_cbd.py` 怎麼跑、輸出長什麼樣、`compute_cbd_prediction_metrics.py` 怎麼把 JSON overlay 變成 mAP/IoU/F1。
