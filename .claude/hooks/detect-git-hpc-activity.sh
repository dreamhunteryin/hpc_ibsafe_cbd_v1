#!/usr/bin/env bash
# detect-git-hpc-activity.sh
#
# Stop hook：每回合 assistant 結束時觸發。
# 掃描 transcript 看本回合是否有 git/HPC 基本操作活動，
# 若有則注入 system-reminder，提示 Claude 考慮呼叫 /git-hpc-note。
#
# Hook payload 由 stdin 讀入（JSON），其中 transcript_path 指向當前對話的 JSONL。

set -euo pipefail

# 讀 hook payload
payload=$(cat)

# 抽出 transcript path
transcript_path=$(echo "$payload" | python3 -c "import sys, json; print(json.load(sys.stdin).get('transcript_path', ''))" 2>/dev/null || echo "")

if [[ -z "$transcript_path" || ! -f "$transcript_path" ]]; then
    # 沒拿到 transcript 就靜默退出（不阻塞）
    exit 0
fi

# 只看最近 30 行（本回合的活動），避免歷史回合誤觸發
recent=$(tail -n 30 "$transcript_path" 2>/dev/null || echo "")

if [[ -z "$recent" ]]; then
    exit 0
fi

# Git 關鍵字（指令層級，避免抓到 markdown 描述）
git_pattern='git (add|commit|push|pull|clone|remote|stash|fetch|reset|merge|rebase|checkout|branch|status|log|diff)\b|ssh -T git@|ssh-keygen'

# HPC 關鍵字
hpc_pattern='\b(squeue|sbatch|scancel|scontrol|sinfo|srun)\b|mamba activate|conda activate|module load|^scp |^rsync '

# 偵測
git_hit=false
hpc_hit=false

if echo "$recent" | grep -qE "$git_pattern"; then
    git_hit=true
fi

if echo "$recent" | grep -qE "$hpc_pattern"; then
    hpc_hit=true
fi

if [[ "$git_hit" == false && "$hpc_hit" == false ]]; then
    # 沒有 git/HPC 活動，不打擾
    exit 0
fi

# 構造提示訊息
topic=""
if [[ "$git_hit" == true && "$hpc_hit" == true ]]; then
    topic="git + HPC"
elif [[ "$git_hit" == true ]]; then
    topic="git"
else
    topic="HPC"
fi

# 用 additionalContext 注入提醒
# 注意 JSON 必須合法
cat <<EOF
{
  "decision": "approve",
  "additionalContext": "[git-hpc-note hook] 偵測到本回合涉及 ${topic} 基本操作。若剛協助使用者解決了一個值得記錄的問題（不是純執行、不是閒聊），請考慮主動呼叫 /git-hpc-note 將此案例追加到 tutorial/basic_git_and_hpc_skills/ 對應檔。若不適用，忽略此提醒即可。"
}
EOF

exit 0
