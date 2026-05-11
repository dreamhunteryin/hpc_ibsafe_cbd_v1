# CLAUDE.md — iBSAFE_CBD_v1 (Token-Efficient)

> Project: iBSAFE_CBD_v1 — 兩階段 CBD 偵測 surgical AI
> Stack: Python + PyTorch + SAM3 + LoRA + ConvNeXt/RT-DETRv4 + Spatiotemporal Transformer
> Runtime: Strasbourg icube HPC + Slurm + mamba env `py311cu118`
> Audience: 外科醫師（非工程師）／互動介面繁體中文
> Goal: 維持嚴格工程紀律的同時最小化 token 開銷

---

## 0) SESSION HANDSHAKE (ONE-TIME)

**新 session 的第一個任務（或本檔變動後），只回一行：**

`ACK:IBSAFE-CBD-V1`

之後**不重複握手或 checklist**，除非有規則違反風險。

---

## 1) RULE SET (REFERENCE BY ID)

### 通用工程紀律

### R1 — Root 不放新檔
新檔分流到 `train/`, `infer/`, `slurm/`, `src/<submodule>/`, `data_utils/`, `tutorial/`。
允許保留的 root 檔：`README.md`, `CLAUDE.md`, `.gitignore`, `compute_cbd_prediction_metrics.py`（legacy）。

### R2 — 訓練/推論輸出位置
必須寫到 `outputs/<exp_name>/`（既有慣例）或 yaml 中明寫的 `output_dir`。**禁止**寫到 root 或 `src/`。臨時檔用 `output/temp/` 或 `/tmp/`。

### R3 — 不主動新增 `.md`
除非使用者明確要求。`tutorial/basic_git_and_hpc_skills/` 案例集由 `/git-hpc-note` 維護，不要手動新增 markdown。

### R4 — 單一真相
相同職責的程式碼只能有一份。`src/cbd/` 與 `src/cbd_rtdetrv4/` 是**刻意的平行 backbone 實作**；新 backbone 建 `src/cbd_<name>/` 平行同構，不要融合。

### R5 — 禁止版本化檔名
不准 `*_v2.py`, `*_new.py`, `enhanced_*.py`, `train_cbd_final.py`。要改就改既有檔。

### R6 — 不抄貼
`cbd/` 與 `cbd_rtdetrv4/` 共通邏輯（dataset / metric / loss）若需共用，抽到 `src/data/` 或建 `src/cbd_common/`。不要兩邊各 fork 一份再各自打補丁。

### R7 — 可調值寫進 yaml
所有訓練/模型/資料參數（lr, batch, epoch, lora_rank, dataset path, augmentation 等）必須來自 `configs/*.yaml`，不准在 `.py` 內寫死。現存 LoRA configs 中殘留的 `/home2020/...` 絕對路徑屬技術債，動到時順手改成相對路徑或環境變數。

### R8 — 工具紀律
用 Read / Edit / Grep / Glob。不准用 shell `cat / head / tail / sed / awk / find`。LS、Write 視情況使用。

### R9 — Git checkpoint
有意義的里程碑後，**先問使用者**才 add / commit / push。push 後提醒使用者「HPC 端記得 `git pull`」。

### R10 — Subagent 使用紀律
什麼時候啟動子代理協助本專案：

| 情境 | 用哪一個 |
|---|---|
| 在 `src/`、`configs/`、`slurm/` 多處同時找 pattern；對既有實作做地圖式調查 | **Explore**（只一隻，範圍很大才用 2-3 隻平行） |
| 開放式研究、多步驟調查（例：比較 `cbd/` 與 `cbd_rtdetrv4/` metric 差異） | **general-purpose** |
| 跨多檔重構、新 backbone 串接、設計取向需驗證 | **Plan** |
| 中間工具輸出量大且用完即丟（大量 grep、跑 build、測一連串假設） | **Fork**（不指定 subagent_type） |
| 單檔編輯、一行 grep、Read 已知路徑、跑單一 sbatch | **不要用 subagent** |

- Fork 走背景，**不要**中途 Read fork 的 transcript（會把 noise 灌回 context）。
- 若你*本身就是 fork 出來的*，直接執行，不要再 fork。

### R11 — Search-first
新增 dataset loader / scheduler 腳本 / 新 backbone / 新 metric 之前，先 grep 既有實作能否擴充。
真的要新檔，標註 `NEW FILE REASON: <一句話>` 與正確 submodule 位置。

---

### 專案特有紀律

### R12 — HPC/Slurm 執行邊界
訓練 / 全集推論 / 快取預備預計 **> 5 分鐘** → 透過 `slurm/schedule_*.py` 提交 sbatch，**不在登入節點直接 `python train/...`**。
本機（無 GPU）只跑 import 驗證、yaml 解析、debug 模式 smoke test（`--debug` / `--num-iters 5`）。
改完程式後若需 GPU 驗證，明確列出「請在 HPC 端執行」的指令塊。

### R13 — HPC vs 本機環境分工
本機 = 編寫環境；HPC = 執行環境。
- 不假設本機跑得起來
- 不假設 HPC 已 `git pull` 最新版
- 改完 push 後提醒使用者去 HPC `git pull`

### R14 — 使用者面向繁中
互動語言、commit message、註解可英文，但**給使用者看的回覆、錯誤解釋、教學內容用繁體中文**。
優先給具體可貼上的指令（含完整路徑與旗標），不要只給抽象建議。

### R15 — 教學自動更新走 slash command
解決完 git/HPC 操作問題、Stop hook 注入提醒時，呼叫 `/git-hpc-note`，由它 append 案例到 `tutorial/basic_git_and_hpc_skills/`。
**不要直接編輯 tutorial 案例集**。

---

## 2) PRECHECK (MINIMAL OUTPUT)

開工前一行：
- `PRECHECK: OK` 或
- `PRECHECK: BLOCKED (R7) -> 把 lr 移進 yaml`

只在觸發 **Plan Mode**（見 §4）時才印長 checklist。

---

## 3) SEARCH-FIRST PROTOCOL (MANDATORY WHEN ADDING NEW CODE)

新增 feature / module / class / config / metric 前：

1. **Grep** 關鍵字、函數名、近似職責：
   - `Grep(pattern="<keyword>", glob="src/**/*.py")`
   - `Grep(pattern="<keyword>", glob="train/**/*.py")`
   - `Grep(pattern="<keyword>", glob="configs/**/*.yaml")`
2. **Read** 最相關的既有檔，理解 pattern。
3. 優先**擴充既有**實作。
4. 若真要新檔：
   - 放對 submodule
   - 一句話寫 `NEW FILE REASON: ...`

---

## 4) EXECUTION PATTERN

預設四步：
1. 一句話計畫（不寫小作文）
2. 實作
3. 測試/驗證（本機 smoke test 或 HPC sbatch）
4. Git checkpoint（R9，**問過再做**）

### Plan Mode Trigger (MANDATORY)

進入 **Plan Mode** 並印 Todo checklist，當以下任一條件成立：
- 跨 Stage 1 ↔ Stage 2 的修改
- 動到資料 pipeline / 新增 backbone
- 使用者一句話講出來但實際需多步驟順序
- 有 regression 風險（動到 `model.py`、`engine.py`、`dataset.py`、`cache.py`）

---

## 5) PROJECT STRUCTURE (REFERENCE)

```
configs/                yaml 訓練設定（*_lora.yaml = Stage 1, bsafe_cbd*.yaml = Stage 2）
slurm/                  HPC 排程腳本（schedule_*.py → 產 sbatch → 提交）
src/
  sam3/                 Stage 1: SAM3 + LoRA
  cbd/                  Stage 2: ConvNeXt 版（model/engine/dataset/cache/sources）
  cbd_rtdetrv4/         Stage 2: RT-DETRv4 版（與 cbd/ 同構平行）
  data/                 共用 dataset loader (dataset_bsafe.py / dataset_camma.py / viz.py)
  RT-DETRv4/            外部引入的 detection backbone
train/                  獨立 trainer (train_lora.py, train_cbd.py, train_cbd_rtdetrv4.py)
infer/                  推論 / 評估 (infer_lora.py, infer_cbd.py, eval_lora.py)
data_utils/             資料前處理
tutorial/               外科醫師教學（basic_git_and_hpc_skills/ 由 /git-hpc-note 維護）
doc_temp/               暫存文件
outputs/                訓練/推論輸出（git ignore）
```

不要在 root 新增頂層目錄，除非使用者明確要求（R1）。

---

## 6) PIPELINE 現況（流動中，非定案）

> ⚠️ 本專案正在嘗試多條 pipeline 設計。`README.md` 描述的只是**其中一條當前嘗試線**，不是已定型的工作流。
> 本節只做導航，不在 CLAUDE.md 中固化任何步驟順序 —— 流動中的設計不該被工程紀律檔鎖死。

- 想知道使用者目前跑哪條 pipeline → 讀 `README.md` 與最近動到的 `configs/*.yaml`
- 想看候選實作分支 → 比對 `src/cbd/` vs `src/cbd_rtdetrv4/` 與 `train/` 下的 trainers
- 收到「pipeline 該怎麼跑」類問題時，**不要逕自引述 README 步驟當權威答案**，先確認使用者問的是當前嘗試線還是想換一條
- 使用者改變 pipeline 設計時，真相來源是 README + configs，本節**不需要更新**

---

## 7) SKILLS & HOOKS（已建置）

| 機制 | 檔案 | 何時用 |
|---|---|---|
| Slash command | `.claude/commands/git-hpc-note.md` | 解決 git/HPC 操作問題後，append 案例到 tutorial |
| Stop hook | `.claude/hooks/detect-git-hpc-activity.sh` + `.claude/settings.json` | 自動偵測 git/HPC 活動，注入 system-reminder |
| Memory feedback | `~/.claude/projects/-home-shihminyin-CAMMA-iBSAFE-v1/memory/feedback_git_hpc_tutorial_autoupdate.md` | 跨 session 慣例（第三層備援） |

詳細路由規則與案例格式 → `tutorial/basic_git_and_hpc_skills/README.md`。

---

## 8) QUICK COMMANDS (OPTIONAL REFERENCE)

```bash
# === 本機 smoke test（CPU，僅驗 import / config 解析）===
python -c "import yaml; print(yaml.safe_load(open('configs/bsafe_cbd.yaml')))"

# === HPC 端常用 ===
ssh smyin@<HPC>
cd /home2020/home/icube/smyin/projects/ibsafe_cbd_v1
mamba activate py311cu118

# Stage 1 LoRA 訓練（提交 sbatch）
python slurm/schedule_train.py --config configs/icglceaes_lora.yaml --experiment <name>

# Stage 2 CBD 訓練
python slurm/schedule_cbd_train.py --config configs/bsafe_cbd.yaml --experiment <name>

# 查 job
squeue -u $USER
scontrol show job <jobid>
tail -f outputs/<name>/train_*.out
```

---

## 9) DEVELOPMENT STATUS

當下狀態：
- `README.md` 已就位（外科醫師操作手冊）
- 三層教學自動更新系統（slash command + Stop hook + memory）已就位並驗證
- `.claude/` 內 commands + hooks + settings 已驗證可用
- Stage 1 / Stage 2 兩條 backbone 線平行存在；新 backbone 沿用平行同構模式

最近常見任務：
- 改 Stage 2 模型（`src/cbd/model.py` 或 `src/cbd_rtdetrv4/`）
- 調整 LoRA config（`configs/*_lora.yaml`）
- HPC 排程除錯
- Tutorial 案例累積（自動由 `/git-hpc-note` 維護）
