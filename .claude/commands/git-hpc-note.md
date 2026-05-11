---
description: 把最近一次解決的 git/HPC 操作問答整理成案例，append 到 tutorial/basic_git_and_hpc_skills/git_basic.md 或 hpc_basic.md
---

你被呼叫來執行「Git/HPC 教學自動更新」流程。請依下列步驟操作：

## 步驟

### 1. 掃描最近對話脈絡

回顧最近 5-10 個 turn 的對話，找出一段「**使用者遇到 git/HPC 操作問題 → 你協助解決**」的完整片段。需要抓出：
- 使用者當時的目標 / 情境
- 使用者實際打的指令（從 Bash 工具呼叫紀錄裡看）
- 出現的輸出 / 錯誤訊息
- 解法 / 處理步驟
- 可以帶走的原則

若找不到合適片段（例如近期都在改 model code、沒有 git/HPC 操作），**禮貌回報「最近對話沒找到值得記錄的 git/HPC 操作案例」並停止**，不要硬寫。

### 2. 路由（決定寫到哪一篇）

| 主題 | 目標檔 |
|---|---|
| 純 git 指令（add/commit/push/pull/remote/stash/clone/branch/merge/rebase）| `tutorial/basic_git_and_hpc_skills/git_basic.md` |
| GitHub 認證、SSH key、PAT、ssh-keygen、`~/.ssh/config` | `tutorial/basic_git_and_hpc_skills/git_basic.md` |
| HPC SSH 登入、目錄、權限 | `tutorial/basic_git_and_hpc_skills/hpc_basic.md` |
| conda / mamba 環境 | `tutorial/basic_git_and_hpc_skills/hpc_basic.md` |
| Slurm（squeue/sbatch/scancel/scontrol/sinfo/srun）| `tutorial/basic_git_and_hpc_skills/hpc_basic.md` |
| scp / rsync 檔案傳輸 | `tutorial/basic_git_and_hpc_skills/hpc_basic.md` |
| 跨 git + HPC（如在 HPC 上做 git clone）| 主要動作那篇；另一篇加 cross-reference |

### 3. 找下一個案例編號

讀目標檔，grep `### 案例 N` 找最大的 N，新案例用 N+1。

### 4. 撰寫案例（用標準格式）

````markdown
### 案例 N：<簡短主題（10 字內，動詞起頭）>

**日期**：YYYY-MM-DD（用今天日期）
**情境**：<為什麼遇到這個問題、當時想做什麼>

**執行**：
```bash
<實際打的指令，去掉敏感資訊但保留結構>
```

**輸出 / 錯誤訊息**：
```
<原樣貼上，若沒有就省略此區塊>
```

**處理 / 解法**：
<逐步條列解法。若涉及多個情境，用表格>

**重點**：
- <可帶走的原則>
- <避免的陷阱>
- <若跟既有案例相關，加：→ 參見 [案例 X](#案例-x簡短主題)>
````

### 5. Append 到目標檔

用 Edit 工具，把新案例插在「最後更新：YYYY-MM-DD」**之前**、最後一個既有案例**之後**。同時更新檔尾日期為今天。

跨檔 cross-reference 時，記得另一篇也要加一行 `→ 參見 [其他檔的案例 N](其他檔.md#案例-n)`。

### 6. 簡短回報

```
✅ 已新增「案例 N：<主題>」到 <git_basic.md|hpc_basic.md>
```

不要重複貼整個案例內容（使用者打開檔案就能看）。

## 注意

- **不要**為了寫而寫：閒聊、微不足道的單一指令、純看 log，都不算案例
- **不要**重複既有案例：若主題跟某個既有案例高度相似，改去那個案例補充內容，不要新增
- **不要**把使用者的真實密碼、token、完整路徑（含敏感目錄）寫進案例
- 若使用者明確說「不用記」「太瑣碎別加」就跳過

## 觸發來源

此命令可由：
1. 使用者直接打 `/git-hpc-note`
2. 你看到 Stop hook 注入的提醒「考慮呼叫 /git-hpc-note」後自主呼叫
3. 使用者自然語言（「把剛才這個加進案例集」）

三種來源都用同一套邏輯處理。
