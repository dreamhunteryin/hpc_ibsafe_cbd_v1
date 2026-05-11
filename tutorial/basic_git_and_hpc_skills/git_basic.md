# Git 基本操作教學

> 純 Git 相關的概念、指令、認證設定、常見錯誤訊息、Git 案例集。
> HPC 操作見 [`hpc_basic.md`](hpc_basic.md)，整合工作流見 [`README.md`](README.md)。

---

## 目錄

- [Part 1：Git 三區概念與基本指令](#part-1git-三區概念與基本指令)
- [Part 2：Remote 管理](#part-2remote-管理)
- [Part 3：認證設定（SSH vs HTTPS）](#part-3認證設定ssh-vs-https)
- [Part 4：常見 Git 錯誤訊息與誤解](#part-4常見-git-錯誤訊息與誤解)
- [Part 5：Git 案例集](#part-5git-案例集)

---

## Part 1：Git 三區概念與基本指令

### 三個區域

```
[Working Directory]      ← 你正在編輯的檔案
       ↓ git add
[Staging Area / Index]   ← 已標記、準備 commit 的變動
       ↓ git commit
[Local Repository]       ← 本機 .git/ 內的版本歷史
       ↓ git push
[Remote Repository]      ← GitHub 上的 repo
```

### 看狀態

```bash
git status              # 看哪些檔案改了、哪些已加入 staging、哪些是 untracked
git log --oneline -10   # 看最近 10 筆 commit
git diff                # 看 working directory 跟 staging 的差別
git diff --staged       # 看 staging 跟 last commit 的差別
```

### 提交一次變動的標準三步

```bash
git add <檔案>           # 把指定檔案放進 staging
git add .                # 把所有變動都放進 staging（小心會把不該追蹤的也加進來）
git commit -m "訊息"     # 把 staging 的內容封成一個 commit
git push origin main     # 推到 GitHub 的 main branch
```

### 拉最新版本

```bash
git pull origin main     # 從 GitHub 抓 main branch 並 merge 到本機
git fetch origin         # 只抓不 merge（給想先看再決定的場景）
```

### Stash：暫存未提交的變動

```bash
git stash                # 把目前未 commit 的變動收起來（可切 branch / pull 等）
git stash list           # 看暫存清單
git stash pop            # 把最近一次 stash 還原並從清單移除
git stash drop           # 直接丟棄最近一次 stash
```

---

## Part 2：Remote 管理

Remote 就是「遠端 repo 的暱稱」。慣例上：
- `origin` = 你自己的 fork / 主要 push 目的地
- `upstream` = 上游官方 repo（通常只 fetch、不 push）

### 看目前 remote 設定

```bash
git remote -v
```

範例輸出：
```
origin    git@github.com:dreamhunteryin/hpc_ibsafe_cbd_v1.git (fetch)
origin    git@github.com:dreamhunteryin/hpc_ibsafe_cbd_v1.git (push)
```

### 新增 / 重命名 / 變更 / 刪除 remote

```bash
git remote add <name> <url>             # 新增
git remote rename <old> <new>           # 重命名（例如把 origin 改成 old_origin 保留紀錄）
git remote set-url <name> <new_url>     # 改 URL（例如 HTTPS 換 SSH）
git remote remove <name>                # 刪除
```

### HTTPS URL vs SSH URL

| 形式 | 範例 | 認證方式 |
|---|---|---|
| HTTPS | `https://github.com/user/repo.git` | 帳號 + Personal Access Token（PAT）|
| SSH | `git@github.com:user/repo.git` | SSH key |

→ **HPC 上強烈建議用 SSH**，避免每次都要輸入 token。

---

## Part 3：認證設定（SSH vs HTTPS）

### 選項 A：HTTPS + Personal Access Token

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
2. 勾選 `repo` 權限，產生 token（一次性顯示，立刻複製）
3. push 時：
   ```
   Username: <你的 GitHub 帳號>
   Password: <貼 token，不是 GitHub 密碼>
   ```
4. 存進 credential helper 避免每次輸入：
   ```bash
   git config --global credential.helper store
   ```

### 選項 B：SSH key（推薦）

#### B-1. 檢查是否已有 key

```bash
ls -la ~/.ssh/
# 看有沒有 id_ed25519 / id_ed25519.pub / id_rsa / id_rsa.pub
```

#### B-2. 產生新 key（如果沒有）

```bash
ssh-keygen -t ed25519 -C "<你的識別字串，如 smyin@hpc>" -f ~/.ssh/id_ed25519_github
# 一直按 Enter（passphrase 留空，HPC 自動 pull 才方便）
```

#### B-3. 把 public key 加到 GitHub

```bash
cat ~/.ssh/id_ed25519_github.pub
```

複製整串輸出（`ssh-ed25519 AAAA... 你的識別字串`），貼到：

GitHub → Settings → SSH and GPG keys → New SSH key → 取個有意義的名字（如 `HPC icube`）→ 貼上 → Add SSH key

#### B-4. 設定 SSH config 告訴 git 用哪把 key

編輯 `~/.ssh/config`：

```bash
cat >> ~/.ssh/config <<EOF

Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
EOF

chmod 600 ~/.ssh/config
```

#### B-5. 驗證

```bash
ssh -T git@github.com
```

**成功訊號**：
```
Hi <你的GitHub帳號>! You've successfully authenticated, but GitHub does not provide shell access.
```

⚠️ **「does not provide shell access」是正常的、預期的訊息**，不是錯誤。看到 `Hi <帳號>!` 就代表 SSH key 成功通過認證。詳見 [Part 5 案例 1](#案例-1ssh--t-githubcom-訊息誤解)。

---

## Part 4：常見 Git 錯誤訊息與誤解

### 「Hi xxx! You've successfully authenticated, but GitHub does not provide shell access.」

✅ **這是成功訊號**。你只是要驗證 SSH key 能不能跟 GitHub 認證，不是真的要登入 GitHub 的 shell。看到這行就代表 OK。

### 「Permission denied (publickey).」

❌ SSH key 沒有被 GitHub 接受。可能原因：
- public key 還沒加到 GitHub Settings → SSH keys
- `~/.ssh/config` 沒指向正確的 key
- key 的權限太開放（應該 `chmod 600 ~/.ssh/id_*`）

### 「fatal: not a git repository (or any of the parent directories): .git」

❌ 你不在 git repo 裡面。先 `pwd` 確認位置，再 `cd` 到正確的 repo 根目錄。

### 「remote: Repository not found.」

❌ 你 clone 或 push 的 URL 拼錯，或 repo 不存在 / 是 private 但你沒權限。先到瀏覽器確認 repo URL 對不對。

### 「Your branch is ahead of 'origin/main' by N commits.」

⚠️ 你本機有 N 筆 commit 還沒 push。執行 `git push origin main`。

### 「Your branch is behind 'origin/main' by N commits.」

⚠️ GitHub 上有你本機沒有的 commit。執行 `git pull origin main`。

### 「CONFLICT (content): Merge conflict in <file>」

❌ Pull 進來的內容跟本機改動衝突。打開 `<file>`，找 `<<<<<<<` / `=======` / `>>>>>>>` 標記，手動選擇要保留的內容，然後 `git add <file>` 再 `git commit`。

### 「! [rejected]  main -> main (non-fast-forward)」

❌ 想 push 但遠端有你本機沒有的 commit。先 `git pull --rebase origin main` 整合，再 `git push`。如果你確定要覆蓋遠端（很少這麼做），可用 `git push --force`，但這會丟掉別人的 commit，**慎用**。

---

## Part 5：Git 案例集

> 每個案例記錄一次實際操作中遇到的情境、執行的指令、結果。新案例由 Claude 自動附加在此區下方（見 [README.md 三層自動化說明](README.md#三層自動化系統)）。

---

### 案例 1：SSH -T github.com 訊息誤解

**日期**：2026-05-07
**情境**：在 HPC 上驗證 SSH key 是否能跟 GitHub 認證。

**執行**：
```bash
ssh -T git@github.com
```

**輸出**：
```
Hi dreamhunteryin! You've successfully authenticated, but GitHub does not provide shell access.
```

**誤解**：以為「does not provide shell access」是錯誤訊息。

**處理 / 解法**：這是 GitHub 故意設計的回應 —— GitHub 不開放 SSH shell 登入，但你只是要驗證認證能否通過。看到 `Hi <帳號>!` 就代表 **SSH key 已經成功被 GitHub 接受**，可以放心執行 git clone / push / pull。

**重點**：
- ✅ 出現 `Hi <你的帳號>!` → 認證成功
- ❌ 出現 `Permission denied (publickey).` → 認證失敗，需要重新設定 key

---

### 案例 2：把 git remote 從舊 repo 換到新 repo

**日期**：2026-05-07
**情境**：本機 repo 原本 clone 自 `iBSAFE_CBD_v1.git`，要改為與新 repo `hpc_ibsafe_cbd_v1.git` 做版本控制，HPC 也要從新 repo pull。

**執行**：

1. **GitHub 上**先建好新 repo（不要勾 Initialize with README，保持空的）。

2. **本機**保留舊 origin 紀錄、設定新 origin、push：
```bash
cd /home/shihminyin/CAMMA/iBSAFE_v1/iBSAFE_CBD_v1
git remote rename origin old_origin
git remote add origin git@github.com:dreamhunteryin/hpc_ibsafe_cbd_v1.git
git remote -v   # 確認
git push -u origin main
```

3. **HPC 上** clone 新 repo（→ 細節見 [`hpc_basic.md` 案例 1](hpc_basic.md#案例-1hpc-上已有同名資料夾的處理)）。

**重點**：
- `git remote rename` 比直接 `remove + add` 安全，舊 URL 仍可用 `old_origin` 取回。
- `git push -u origin main` 的 `-u` 會把 `origin/main` 設為預設追蹤分支，之後可直接 `git push`。
- 新 repo 必須是**空的**才能直接 push。如果不小心勾了 Initialize，會出現 non-fast-forward 錯誤，要先 `git pull --rebase origin main` 再 push。

---

*最後更新：2026-05-07*
