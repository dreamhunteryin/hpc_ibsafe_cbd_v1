# cbd_v1 — 學習筆記

> 此資料夾(`tutorial/`)專門存放外科醫師視角的學習筆記。所有 `.md` 都放這裡,不污染主程式目錄。

## 子專案的角色

**CBD(common bile duct,總膽管)影像辨識模型**的第一版(`v1`)實作。

推測任務:在膽囊切除術術中影像(腹腔鏡 video frame)中辨識或分割總膽管——這是 critical view of safety(CVS)安全辨識的關鍵解剖之一。**確切任務定義(分類 / 偵測 / 分割?輸入是單張 frame 還是 video clip?)待向工程師確認**。

臨床對應:模型扮演「術中第二雙眼睛」,在外科醫師夾閉膽囊管前提示總膽管的位置,降低 bile duct injury 的風險。

## 檔案結構總覽

```
cbd_v1/
├── compute_cbd_prediction_metrics.py  # 模型預測指標的計算腳本
├── configs/                           # YAML 設定檔(模型 / 訓練 / 資料)
├── data_utils/                        # 資料前處理 utilities
├── src/                               # 模型與核心邏輯
├── train/                             # 訓練流程
├── infer/                             # 推論流程
└── slurm/                             # Slurm 叢集排程腳本
```

> `slurm/` 的存在意味訓練是跑在 **HPC 叢集**上(很可能是 Strasbourg / Jean Zay 之類的高效能運算中心),不是在本機。Slurm 是 HPC 排隊系統——可以類比為「手術房排程系統」,把訓練 job 排進可用的 GPU 時段。

## 待理解的關鍵問題

1. **任務類型**:classification / detection / segmentation?輸出是 mask、bbox、還是 label?
2. **輸入資料**:單張影像?短 video clip?用了什麼 pre-processing?
3. **模型架構**:CNN / Transformer / 其他?有 pretrained backbone 嗎?
4. **訓練流程**:loss function、optimizer、epochs、metric?
5. **評估方式**:`compute_cbd_prediction_metrics.py` 用了哪些 metric?

## 筆記目錄

(待補。建議命名:`01_configs_walkthrough.md`、`02_data_pipeline.md`、`03_model_architecture.md`、`04_training_loop.md`、`05_inference.md`、`06_metrics.md`)

## Git Remote 設定(已完成)

- `origin` → `git@github.com:dreamhunteryin/iBSAFE_CBD_v1.git`(您的 GitHub,push 目的地)
- `upstream` → `git@forge.icube.unistra.fr:CAMMA/code/bsafe/cbd_v1.git`(工程師 Luc Vedrenne 的 CAMMA GitLab,只用 `git fetch upstream` 拿更新,**不可** push)
