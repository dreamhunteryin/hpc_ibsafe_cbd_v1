# HPC 基本操作教學

> 純 HPC 相關的概念、SSH 登入、Module、Slurm、檔案傳輸、案例集。
> Git 操作見 [`git_basic.md`](git_basic.md)，整合工作流見 [`README.md`](README.md)。
> 本叢集：Strasbourg icube HPC（Unistra）。官方文件：<https://hpc.pages.unistra.fr/doc/slurm>

---

## 目錄

- [Part 1：HPC 是什麼？（白話概念）](#part-1hpc-是什麼白話概念)
- [Part 2：SSH 登入與身分確認](#part-2ssh-登入與身分確認)
- [Part 3：Module 模組系統 + mamba 環境](#part-3module-模組系統--mamba-環境)
- [Part 4：目錄管理（含 /scratch 暫存磁碟）](#part-4目錄管理含-scratch-暫存磁碟)
- [Part 5：Slurm — 互動模式 vs 批次模式](#part-5slurm--互動模式-vs-批次模式)
- [Part 6：分區、GPU constraint、Preemption](#part-6分區gpu-constraintpreemption)
- [Part 7：提交與監控 Job](#part-7提交與監控-job)
- [Part 8：檔案傳輸（scp / rsync）](#part-8檔案傳輸scp--rsync)
- [Part 9：VS Code Remote SSH（推薦給非工程師）](#part-9vs-code-remote-ssh推薦給非工程師)
- [Part 10：常見 HPC 錯誤訊息與 Q&A](#part-10常見-hpc-錯誤訊息與-qa)
- [Part 11：HPC 案例集](#part-11hpc-案例集)

---

## Part 1：HPC 是什麼？（白話概念）

### 筆電 vs HPC

你的筆電像一台私家車：你一個人開、隨時能用、馬力有限。

HPC（高效能運算叢集）像一個有幾百台卡車的停車場：
- 卡車（**計算節點**）放在遠端機房，看不到摸不到
- 你透過 **登入節點（Login node）** 進去，就像走進管理室
- 要用卡車時要**排隊申請**（排程器叫 **Slurm**）
- 排到了你的程式**自動跑在卡車上**，你不必在場、可以關掉 terminal

### 重要概念對照表

| 名詞 | 白話意思 |
|---|---|
| Login node（登入節點）| 管理室，登入後在這裡，**不可在這裡跑訓練** |
| Compute node（計算節點）| 真正的卡車，有 GPU，訓練在這裡跑 |
| Slurm | 排程系統，決定輪到誰用哪台卡車 |
| Job（作業）| 你送出去排隊的一個任務（例：訓練模型）|
| Partition（分區）| 不同等級的停車區，`publicgpu` 是公用、24 小時 |
| Queue（佇列）| 排隊等候的清單 |
| Constraint（限制）| 指定卡車規格（例如要新型 Tensor Core GPU）|
| Preemption（搶佔）| 公用分區的 job 可能被有計畫的人擠掉、重排 |

### 為什麼登入後 `nvidia-smi` 看不到 GPU？

因為你在管理室（login node），GPU 在卡車上（compute node）。
**`nvidia-smi` 要在計算節點上才有效**，登入節點不會有 GPU。

### ⚠️ 本專案的關鍵原則

> **不要在登入節點直接跑 `python train/...`**。訓練、全集推論、快取預備都要透過 `slurm/schedule_*.py` 提交 sbatch，由 Slurm 分派到計算節點執行。
> 登入節點只能跑：import 驗證、yaml 解析、小型 smoke test。

---

## Part 2：SSH 登入與身分確認

### SSH 登入

```bash
ssh <帳號>@<HPC登入節點>
# 範例：ssh smyin@hpc-login1.icube.unistra.fr
```

登入後 prompt 顯示 `[<帳號>@<主機名> <當前目錄>]$`：
```
[smyin@hpc-login1 ~]$
```

`hpc-login1` 表示在登入節點。若日後進入計算節點，prompt 會變成 `[smyin@hpc-nXXX ...]$`（如 `hpc-n863`），用 prompt 區分自己在哪一台機器是重要習慣。

### 確認身分與位置

```bash
whoami                  # 看現在登入哪個帳號
pwd                     # 看完整路徑
hostname                # 看在哪一台機器
echo $HOME              # 看自己的 home 目錄路徑（如 /home2020/home/icube/smyin）
```

### 中斷與重連

```bash
exit                    # 或按 Ctrl+D，登出 SSH
```

若 SSH 連線斷掉但 **sbatch** job 還在跑：job 會繼續執行（Slurm 控制），重新 SSH 後 `squeue -u $USER` 可查狀態。
**但互動 session（`salloc` + `srun --pty bash`）不一樣 —— 互動 session 一斷線就終止**，要用 `tmux` 保護（見 Part 5）。

### 用 `~/.ssh/config` 簡化登入

把長指令收進設定檔，以後只要 `ssh unistra-hpc`：

```sshconfig
# ~/.ssh/config
Host unistra-hpc
    HostName hpc-login1.icube.unistra.fr
    User smyin
    IdentityFile ~/.ssh/id_ed25519
```

---

## Part 3：Module 模組系統 + mamba 環境

### Module 系統（HPC 軟體管理方式）

HPC 用 `module` 管理軟體版本（Python、CUDA、編譯器等）。**每次登入或進入計算節點後都要重新 `module load`**。

```bash
# 查可用模組
module avail python 2>&1 | grep -i python | head -20
module avail cuda 2>&1 | grep -i cuda | head -20

# 載入
module load python/3.12.8
module load cuda/cuda-12.1

# 確認載入了哪些
module list

# 卸載（很少用到）
module unload cuda/cuda-12.1
```

### mamba 環境（本專案使用 `py311cu118`）

本專案在使用者 home 中已建好 mamba env：

```bash
mamba env list           # 列出所有環境
# 預期看到：
# base                  /home2020/.../mambaforge
# py311cu118            /home2020/.../mambaforge/envs/py311cu118

mamba activate py311cu118
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

mamba deactivate         # 退出環境
```

> **Slurm 提交的 sbatch job 不需要手動 activate** —— `slurm/schedule_*.py` 產生的 sbatch 腳本內部會自己 `mamba activate py311cu118`。
> 手動 activate 是為了：本機 debug、互動 session 內測試、檢查 import 是否正常。

---

## Part 4：目錄管理（含 /scratch 暫存磁碟）

### 常用指令

```bash
mkdir -p <路徑>          # 建立資料夾（-p 中間目錄不存在也自動建）
cd <路徑>                # 進入
cd ..                    # 上一層
cd ~                     # 回 home
cd -                     # 回上一個
ls                       # 看當前資料夾
ls -la                   # 完整資訊（含權限、隱藏檔）
ls -lh                   # 大小用 K/M/G 顯示
du -sh <路徑>            # 看資料夾總大小
df -h $HOME              # 看自己 home 還剩多少空間
```

### icube 路徑慣例

| 路徑 | 用途 |
|---|---|
| `/home2020/home/icube/<帳號>/` | 你的 home，存程式碼、設定 |
| `/home2020/home/icube/<帳號>/projects/` | 建議放 git clone 的 repo |
| `/home2020/home/icube/camma_files/camma_data/` | **共用資料集（B-SAFE、ICG 等），唯讀** |
| `~/.cache/huggingface/` | HuggingFace 模型 cache |
| `/scratch/job.$SLURM_JOB_ID/` | **本地暫存磁碟（job 結束自動刪）** |

### /scratch — 本地高速暫存磁碟

家目錄（`/home2020/...`）是遠端儲存、讀寫較慢；`/scratch` 是計算節點本地 SSD、速度快很多。大資料集訓練可以先複製到 `/scratch` 加速 I/O。

**重點**：`/scratch/job.$SLURM_JOB_ID` 在 job 結束後**自動刪除**，必須在 sbatch 腳本結尾把結果複製回家目錄，否則會遺失。

範例（在 sbatch 腳本內）：
```bash
SCRATCH=/scratch/job.$SLURM_JOB_ID
cp -r ~/datasets/<small_subset> $SCRATCH/

# 跑訓練（config 內 dataset path 指向 $SCRATCH）
python train/train_cbd.py --config configs/bsafe_cbd.yaml

# 結束前把產出搬回 home
cp -r $SCRATCH/outputs/ ~/projects/ibsafe_cbd_v1/outputs/
```

> 本專案大多數情況下**不需要**動用 `/scratch`（資料集已在 `camma_files/`，讀取速度足夠）。只有特別需要極致 I/O 時才考慮。

---

## Part 5：Slurm — 互動模式 vs 批次模式

兩種使用 GPU 節點的方式：

| 方式 | 指令 | 適合情境 |
|---|---|---|
| **互動模式** | `salloc` + `srun --pty bash` | 除錯、確認 GPU 可用、跑短期實驗、看即時輸出 |
| **批次模式** | `sbatch <script>`（或本專案的 `python slurm/schedule_*.py`）| 正式訓練、長時間任務、提交完關 terminal 也不影響 |

### 互動模式：salloc + srun

```bash
# 第一步：申請 GPU 資源（排隊等）
salloc -p publicgpu -N 1 --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 04:00:00 --constraint=gputc

# 第二步：分配到節點後，進入該節點的 shell
srun --pty bash

# 進入後 prompt 變成 hpc-nXXX，可確認 GPU
echo $CUDA_VISIBLE_DEVICES   # 應顯示數字（如 0、3）
nvidia-smi                   # 確認 GPU 型號、可用記憶體
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ⚠️ **互動 session 要保持 terminal 開著**，關掉視窗或網路斷線 = job 結束、訓練中止。長時間訓練建議用 `tmux` 保護（見下）。

### `tmux` — 防斷線神器（互動模式必備）

```bash
# 進入 GPU 節點後先開 tmux
tmux new -s train

# 在 tmux 裡跑（在此你可以放心斷線）
python train/train_cbd.py --config configs/bsafe_cbd.yaml --debug

# 暫時離開但保持訓練：Ctrl+B 然後按 D
# 重新連回：
ssh unistra-hpc
tmux attach -t train

# 列出所有 tmux session
tmux ls
```

### 批次模式：本專案推薦走 `slurm/schedule_*.py`

本專案不直接呼叫 `sbatch`，而是用 Python wrapper 產生 sbatch 腳本並提交：

```bash
# Stage 1 LoRA
python slurm/schedule_train.py --config configs/icglceaes_lora.yaml --experiment <name>

# Stage 2 CBD 訓練
python slurm/schedule_cbd_train.py --config configs/bsafe_cbd.yaml --experiment <name>

# Stage 2 快取預備
python slurm/schedule_prepare_cbd_easy_masks.py --config ...

# 推論
python slurm/schedule_cbd_inference.py --config ...
```

這些 wrapper 會：
1. 讀 yaml config
2. 產生對應的 sbatch 腳本（含 `#SBATCH` 旗標、`mamba activate py311cu118`、訓練指令）
3. 用 `sbatch` 提交給 Slurm
4. 顯示 Job ID

提交後**可以關 terminal**，job 在計算節點繼續跑。

---

## Part 6：分區、GPU constraint、Preemption

### Unistra 分區

| 分區 | 對象 | 時間上限 | 說明 |
|---|---|---|---|
| `publicgpu` | 所有人 | 24 小時 | 公用 GPU，**可能被搶佔** |
| `grantgpu` | 需申請科學計畫 | 4 天 | 穩定不被搶佔，適合長訓練 |
| `public` | 所有人 | 24 小時 | 純 CPU |
| `grant` | 需申請計畫 | 4 天 | 純 CPU |

```bash
sinfo                                    # 看整體節點與分區
sinfo -p publicgpu -o "%N %G %t %C"      # 看 publicgpu 各節點狀態
# %t 狀態：idle=全空、alloc=全滿、mix=部分使用
```

### GPU constraint — 指定 GPU 型號

```bash
--constraint=gputc      # H200 / A100 / V100，Tensor Core，深度學習最快 ✓
--constraint=gpudp      # 雙精度 GPU（科學計算用）
# 不指定 → 可能分到舊型，較慢
```

### ⚠️ Preemption（搶佔）— `publicgpu` 必知

`publicgpu` 的 job 在有 `grantgpu`/私有分區用戶申請資源時**可能被暫停**：
- **互動 session**：直接中止，需重新申請
- **sbatch job**：退回佇列，等再分配時自動重跑（**前提是程式碼有 checkpoint 機制**，否則白跑）

長訓練（>幾小時）建議申請 `grantgpu`。

---

## Part 7：提交與監控 Job

### Job 狀態查詢

```bash
squeue -u $USER                          # 自己所有的 job
watch -n 30 squeue -u $USER              # 每 30 秒自動刷新（Ctrl+C 停）

scontrol show job <jobid>                # 詳細資訊（GPU 型號、節點、時間）
scontrol show job <jobid> | grep StartTime  # 預計開始時間（PD 中）

squeue -o "%S" -j <jobid>                # 預計開始時間（簡短）
sprio -l -u $USER                        # 看自己各 job 的優先權
```

### Job 狀態代碼

| 代碼 | 意思 | 要做什麼 |
|---|---|---|
| `PD` | Pending（排隊中）| 等，正常 |
| `R`  | Running（執行中）| 可看 log |
| `CG` | Completing（即將結束）| 等它自己完成 |
| `CD` | Completed（成功結束）| 檢查輸出 |
| `F`  | Failed（失敗）| 看 log 抓錯 |
| `PR` | Preempted（被搶佔）| 自動回 PD 重排（看是否有 checkpoint） |

### 取消 job

```bash
scancel <jobid>                          # 取消單一 job
scancel -u $USER                         # 取消自己所有 job（慎用）
```

### 即時看 log

本專案 sbatch 把 log 寫到 `runs/{experiment}/{timestamp}/run_1/train_*.out`：

```bash
tail -f runs/.../train_run_1.out         # 即時跟蹤（Ctrl+C 離開）
less runs/.../train_run_1.out            # 翻頁查看（按 q 退出）
grep "Epoch" runs/.../train_run_1.out    # 抓 epoch 進度
tail -50 runs/.../train_run_1.err        # 抓最近 50 行錯誤
```

### Email 通知（可選）

在 sbatch 腳本加入：
```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<你的 email>
```

設定後**可以關 terminal、收信再回來查結果**。本專案 `slurm/schedule_*.py` 若有支援 email 設定，可在對應 yaml 或 wrapper 旗標中啟用。

### 查 GPU 實際使用狀況

```bash
# 先找 job 跑在哪個節點
squeue -u $USER                          # 看 NODELIST 欄，例如 hpc-n863

# 登入該節點看 GPU 使用率（多數 HPC 允許 user SSH 到自己 job 的節點）
ssh hpc-n863 nvidia-smi
```

---

## Part 8：檔案傳輸（scp / rsync）

### scp（單次傳輸，簡單）

```bash
# 本機 → HPC
scp /local/path/file.txt <帳號>@<HPC>:/home2020/.../target/

# HPC → 本機
scp <帳號>@<HPC>:/home2020/.../outputs/<exp>/best.pt /local/path/

# 整個資料夾（-r）
scp -r /local/folder <帳號>@<HPC>:/home2020/.../target/
```

### rsync（推薦：續傳 + 增量同步）

```bash
# 本機 → HPC（已存在的檔只傳變動部分）
rsync -avzP /local/folder/ <帳號>@<HPC>:/home2020/.../target/

# HPC → 本機（拷大檔，斷線可續）
rsync -avzP <帳號>@<HPC>:/home2020/.../outputs/<exp>/best.pt /local/path/
```

旗標說明：
- `-a` archive 模式（保留權限、時間戳）
- `-v` verbose（看進度）
- `-z` 傳輸時壓縮
- `-P` 顯示進度 + 支援續傳

> 大檔（模型 checkpoint > 100 MB）建議用 rsync。

---

## Part 9：VS Code Remote SSH（推薦給非工程師）

不必每次 `ls`/`cat`，直接在本機 VS Code 瀏覽 HPC 上完整檔案樹、看 log、改 config。

### 安裝（只做一次）

1. VS Code → Extensions（`Ctrl+Shift+X`）→ 搜尋 `Remote - SSH`（Microsoft 官方）→ Install
2. `Ctrl+Shift+P` → `Remote-SSH: Add New SSH Host` → 輸入 `ssh unistra-hpc`
3. 選寫入哪個 SSH config 檔（通常 `~/.ssh/config`）

### 連線

1. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → 選 `unistra-hpc`
2. VS Code 開新視窗，等連線建立（首次會自動裝 VS Code Server 到 HPC）
3. `File > Open Folder` → 輸入 `~/projects/ibsafe_cbd_v1` → OK

連上後可以：
- 瀏覽完整 HPC 檔案樹
- 直接開啟 CSV / YAML / log 檔
- 看訓練曲線圖（PNG）
- 編輯 config 並儲存
- 開 terminal panel 在 HPC 上直接打指令

> 前提：本機 `~/.ssh/config` 已設好 `Host unistra-hpc`（見 Part 2）。

---

## Part 10：常見 HPC 錯誤訊息與 Q&A

> 此區隨遇到新問題逐步累積（見 [README.md 三層自動化系統](README.md#三層自動化系統)）。

### 「Permission denied」（對某個目錄）

❌ 沒有讀寫該目錄的權限。可能：
- 想存進別人的 home → 改寫到自己 `$HOME` 或 `projects/`
- 共用資料目錄唯讀（如 `camma_files/`）→ 不要寫入，改到自己 workspace
- 用 `ls -ld <目錄>` 看擁有者跟權限位元

### 「No space left on device」

❌ 磁碟滿了。`df -h $HOME` 看哪個分區滿。常見元兇：
- `~/.cache/huggingface/` 模型快取太多 → 清掉沒用的：`rm -rf ~/.cache/huggingface/hub/models--xxx`
- `outputs/` 或 `runs/` 累積太多 checkpoint → 刪舊的失敗 run

### 「(PD) Reason=Resources / Priority」

⚠️ 你的 job 在排隊，叢集 GPU 滿了。
- 等就好（通常 10-60 分鐘輪到）
- 等超過 2 小時可考慮拿掉 `--constraint=gputc` 讓 Slurm 分配任何可用 GPU
- 用 `sprio -l -u $USER` 看自己的優先權位置

### 「CUDA out of memory」

❌ GPU 記憶體不夠。
- 改小 yaml config 內的 `batch_size`
- 改用 mixed precision（`amp: true` 或對應旗標）
- `nvidia-smi` 看是不是有其他 process 佔住 GPU（互動 session 殘留）

### 「ModuleNotFoundError: No module named 'xxx'」

❌ Python 環境不對：
- 互動 session 內：忘了 `mamba activate py311cu118`
- sbatch job 內：sbatch 腳本內缺少 `mamba activate`
- 真的沒裝：在已啟動的 env 內 `pip install xxx`

### 「FileNotFoundError」資料集

❌ 路徑寫錯：
- yaml config 內 `data.root` 或 `dataset.path` 指向不存在的位置
- 本機 vs HPC 路徑沒切換（CLAUDE.md R13）
- `grep -r "/home" configs/*.yaml` 找寫死的路徑

### 「Preempted」（job 從 R 變成 PD）

⚠️ `publicgpu` 被搶佔。
- 短期：等它自動重排即可
- 中期：申請 `grantgpu` 計畫
- 長期：訓練程式碼加入 checkpoint，被搶後可從上次斷點繼續

### 「nvidia-smi: command not found」

❌ 你在登入節點。GPU 在計算節點上，要先 `salloc + srun --pty bash` 進入節點才能執行。

---

## Part 11：HPC 案例集

> 每個案例記錄一次實際操作中遇到的情境、執行的指令、結果。新案例由 Claude 自動附加在此區下方（見 [README.md 三層自動化說明](README.md#三層自動化系統)）。

---

### 案例 1：HPC 上已有同名資料夾的處理

**日期**：2026-05-07
**情境**：在 HPC 上要 clone 新 repo 時，發現 `/home2020/home/icube/smyin/projects/ibsafe_cbd_v1/` 資料夾已經存在（不確定是空資料夾、舊 clone、還是有未提交工作）。

**執行**（先診斷）：

```bash
pwd
ls -la
git remote -v 2>/dev/null || echo "(這不是 git repo)"
```

**處理 / 解法**：依輸出結果分三種狀況：

| 狀況 | 判斷依據 | 處理方式 |
|---|---|---|
| **A. 空資料夾或只有少數檔案** | `ls -la` 幾乎沒東西 | `cd ..` → `rm -rf ibsafe_cbd_v1` → `git clone <新URL> ibsafe_cbd_v1` |
| **B. 已是完整 repo 但 remote 指向舊 URL** | `git remote -v` 顯示舊 URL | `git remote set-url origin <新URL>` → `git pull origin main` |
| **C. 是舊 clone 但有未提交的工作** | `git status` 顯示 modified / untracked | `git stash` 暫存 → `git remote set-url origin <新URL>` → `git fetch origin` → `git reset --hard origin/main` → `git stash pop` 還原 |

**重點**：
- 刪除前一定要先 `ls` 確認沒有自己重要的工作
- `git stash` 是備份未提交變動的安全閥（→ 詳見 [`git_basic.md` Part 1 Stash 章節](git_basic.md#stash暫存未提交的變動)）
- 切換 remote 用 `git remote set-url`，不要 `remove` 再 `add`，避免操作中斷

---

## 本專案 HPC 環境快速參考

| 項目 | 值 |
|---|---|
| 登入指令 | `ssh smyin@hpc-login1.icube.unistra.fr`（或 `ssh unistra-hpc`） |
| 專案路徑 | `/home2020/home/icube/smyin/projects/ibsafe_cbd_v1` |
| Python 環境 | mamba env `py311cu118`（不是 venv） |
| 訓練分區 | `publicgpu`（24h，可被搶佔）／ `grantgpu`（4 天，需申請） |
| GPU 申請（互動，除錯用） | `salloc -p publicgpu -N 1 --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 04:00:00 --constraint=gputc` |
| 進入節點 | `srun --pty bash` |
| 防斷線 | `tmux new -s train` ／ `tmux attach -t train` |
| 正式訓練（推薦）| `python slurm/schedule_train.py --config configs/<name>.yaml --experiment <exp>` |
| 看 job 狀態 | `squeue -u $USER` ／ `watch -n 30 squeue -u $USER` |
| 看 log | `tail -f runs/<exp>/<ts>/run_1/train_run_1.out` |
| 輸出位置 | `outputs/<exp_name>/`（依 yaml `output_dir`）、`runs/<exp>/<ts>/`（sbatch log）|
| 共用資料集 | `/home2020/home/icube/camma_files/camma_data/`（唯讀）|
| 暫存磁碟 | `/scratch/job.$SLURM_JOB_ID/`（job 結束自動刪）|

---

*最後更新：2026-05-11*
