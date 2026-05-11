# 基本 Git 與 HPC 操作教學

> 外科醫師視角的 git 與 HPC 基本操作教學區。內容拆成兩篇 + 整合工作流，並設計了**三層自動化系統**讓案例可以持續累積。

---

## 教學文件結構

| 檔案 | 內容 | 何時看 |
|---|---|---|
| [`git_basic.md`](git_basic.md) | Git 三區概念、指令、Remote 管理、認證設定（SSH/HTTPS）、Git 錯誤訊息、**Git 案例集** | 想做版本控制、push/pull、設 SSH key |
| [`hpc_basic.md`](hpc_basic.md) | SSH 登入、目錄管理、conda/mamba、Slurm 指令、scp/rsync、HPC 錯誤訊息、**HPC 案例集** | 想登入 HPC、跑 job、傳檔案 |
| [`README.md`](README.md)（本檔）| 索引、整合工作流圖、三層自動化系統說明 | 第一次來、想看全貌 |

> 為何拆兩篇？git 跟 HPC 是兩套獨立技能 —— 你可能在本機只用 git、或在 HPC 只跑 slurm 不碰 git。獨立成兩篇方便針對性查找與成長。

---

## 整合工作流：本機 ↔ GitHub ↔ HPC

```
[本機編輯]                [GitHub]                [HPC]
git add <file>
git commit -m "..."
git push origin main  ──→  hpc_ibsafe_cbd_v1
                                │
                                ↓
                                          git pull origin main
                                          python slurm/schedule_*.py ...
                                          squeue -u $USER
                                          tail -f runs/.../train_*.out
```

### 第一次設定 HPC 端

```bash
ssh smyin@<HPC>
mkdir -p /home2020/home/icube/smyin/projects
cd /home2020/home/icube/smyin/projects
git clone git@github.com:dreamhunteryin/hpc_ibsafe_cbd_v1.git ibsafe_cbd_v1
cd ibsafe_cbd_v1
ls
```

### 之後的循環

每次本機改完後：

```bash
# === 本機 ===
git add <changed_files>
git commit -m "feat: 改了什麼"
git push origin main

# === HPC ===
ssh smyin@<HPC>
cd /home2020/home/icube/smyin/projects/ibsafe_cbd_v1
git pull origin main
python slurm/schedule_train.py --config configs/icglceaes_lora.yaml --experiment ibsafe_lora_v1 --num-runs 1
```

---

## 三層自動化系統

這份教學會持續成長：每次外科醫師遇到新的 git/HPC 操作問題並解決後，**新案例會被自動附加到 `git_basic.md` 或 `hpc_basic.md` 的案例集**。

整套機制有三層備援，任一層都能獨立啟動，避免單點失效：

```
       ┌─────────────────────────────────────────────────────┐
       │  使用者問 git/HPC 問題 → Claude 解決                │
       └─────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────────┐
   │  層級 2 — Hook                                         │
   │  Stop event 觸發 → 偵測本回合是否有 git/HPC 關鍵字     │
   │  → 注入 system-reminder 提醒 Claude 考慮寫案例         │
   └────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────────┐
   │  層級 1 — Slash Command  /git-hpc-note                 │
   │  Claude 收到提醒判斷情境合適 → 自主呼叫此命令          │
   │  → 路由（git 還 HPC）→ 格式化案例 → append 到對應檔   │
   └────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────────┐
   │  層級 3 — Memory Feedback                              │
   │  跨 session 慣例：即使 hook/skill 失效也保留行為       │
   └────────────────────────────────────────────────────────┘
```

### 路由規則

| 案例主題 | 寫到哪 |
|---|---|
| 純 git 指令（add/commit/push/pull/remote/stash 等）| `git_basic.md` Part 5 |
| GitHub 認證、SSH key、token、ssh-keygen | `git_basic.md` Part 5 |
| 純 HPC 環境（conda/mamba、slurm、目錄、權限）| `hpc_basic.md` Part 11 |
| 檔案傳輸 scp/rsync | `hpc_basic.md` Part 11 |
| 跨 git + HPC（如在 HPC 上 clone 並設 SSH key）| 主要動作所在那篇，另一篇加 `→ 參見 [案例 N]()` cross-reference |

### 案例格式（兩篇共用）

````markdown
### 案例 N：<簡短主題>

**日期**：YYYY-MM-DD
**情境**：<為什麼會遇到這個問題、當時的目標>

**執行**：
```bash
<實際打的指令>
```

**輸出 / 錯誤訊息**（若有）：
```
<原樣貼上>
```

**處理 / 解法**：
<逐步說明>

**重點**：
- <可帶走的原則>
- <避免的陷阱>
````

### 排序原則

- 案例**永遠 append 在最後**，編號往上加
- 不重排既有案例（保留時間順序，便於追溯遇到問題的脈絡）
- 若新案例跟舊案例高度相關，在新案例開頭加一行：「→ 參見 [案例 N](#案例-n)」

### 何時手動觸發

如果某次解決過程沒被 hook 偵測到、或 Claude 判斷不主動寫，外科醫師可以直接打：

```
/git-hpc-note
```

或自然地說「把剛才這個 git 問題加進案例集」。

### 三層的實作位置

| 層級 | 機制 | 檔案位置 |
|---|---|---|
| 1 | Slash Command | `.claude/commands/git-hpc-note.md` |
| 2 | Stop Hook | `.claude/hooks/detect-git-hpc-activity.sh` + `.claude/settings.json` |
| 3 | Memory Feedback | `~/.claude/projects/<project-id>/memory/feedback_git_hpc_tutorial_autoupdate.md` |

> Layer 1+2 的檔案在 `.claude/` 目錄下，會跟著 git repo 走，HPC pull 後也能用。
> Layer 3 在使用者本機 `~/.claude/`，不入 git，但慣例描述夠完整，新環境讀進來就懂。

---

## 維護建議

- 案例編號永遠連續累加，不重用已刪案例的編號
- 案例「最後更新」日期（檔尾）由 slash command 自動更新
- 若同一個錯誤訊息出現第二次，**不要新增重複案例**，改去既有案例補充新的觀察
- 教學內容（Part 1-4 / Part 1-6）若需要修正或擴充，直接編輯對應檔（不需走自動化流程）

---

*最後更新：2026-05-07*
