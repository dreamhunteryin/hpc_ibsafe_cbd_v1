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

### Stage 1 — SAM3 + LoRA Fine-tuning(完成,2026-05-02;補充筆記 2026-05-05)

📂 `stage1_SAM3+LoRA_Fine-tuning/`

- [01 — SAM3 + LoRA 總覽](stage1_SAM3+LoRA_Fine-tuning/01_sam3_lora_overview.md):pipeline 角色、工程師原文逐句拆解、SAM3 鳥瞰圖
- [02 — SAM3 架構深解](stage1_SAM3+LoRA_Fine-tuning/02_sam3_architecture_deep.md):Vision/Text/Geometry encoder + Transformer encoder/decoder + Segmentation head 五大組件 + 完整 forward pass
- [02 補充 — Notes(架構觀念深化)](stage1_SAM3+LoRA_Fine-tuning/02_Notes_sam3_architecture_deep.md):ICG 流向、空間推理能力極限、stage 1 / stage 2 分工、多 bbox 訓練的限制
- [03 — LoRA 原理 + 凍結策略](stage1_SAM3+LoRA_Fine-tuning/03_lora_principles_and_freeze.md):低秩分解數學、本專案自寫 3 層類別、注入流程、為何凍結 mask decoder 仍能產出新 mask
- [04 — 資料管線](stage1_SAM3+LoRA_Fine-tuning/04_data_pipeline_class_prompted_box.md):COCO JSON 解析、`CammaSam3Dataset` 流程、文字 prompt 怎麼跟 box 對應、兩個 dataset 對照
- [05 — 訓練主迴圈 + Slurm](stage1_SAM3+LoRA_Fine-tuning/05_training_loop_and_slurm.md):Trainer 拆解、Loss 組合、AdamW+cosine、bf16、grad accumulation、DDP、HPC 提交
- [06 — 推論 + 評估 + 實操](stage1_SAM3+LoRA_Fine-tuning/06_inference_eval_handson_recipe.md):三種推論模式、metrics 計算、**從零跑通的 4 階段實操腳本**
- [06 補充 — Notes(推論流向)](stage1_SAM3+LoRA_Fine-tuning/06_Notes_inference_eval_handson_recipe.md):finetuned SAM3 input/output 解析、原生格式 vs 工程師包裝、class 名稱與座標的去向
- 📎 `docs/`:Lora.pdf、SAM3.pdf 原始論文 PDF

### Stage 2 — ConvNeXt + Spatiotemporal Transformer(進行中,2026-05-05)

📂 `stage2_ConvNeXt_Spatiotemporal/`

ConvNeXt 教學(完成,5 章):

- [00 — ConvNeXt 教學規劃大綱](stage2_ConvNeXt_Spatiotemporal/00_plan_convnext_teaching.md):5 章規劃、已掃描事實清單、決策點
- [01 — ConvNeXt 架構基本](stage2_ConvNeXt_Spatiotemporal/01_convnext_architecture_basics.md):2022 Meta 出品、4 個 stage 金字塔結構、為何選 Small
- [02 — 在 CBD v2 的角色](stage2_ConvNeXt_Spatiotemporal/02_convnext_role_in_cbdv2.md):輸入輸出形狀、v1(global pool) vs v2(spatial grid)、為何不沿用 SAM3 image encoder
- [03 — 凍結策略與差別學習率](stage2_ConvNeXt_Spatiotemporal/03_convnext_freeze_strategies.md):三種模式、`last_stage` 為何是預設、`backbone_lr` 對 `new_layers_lr` 的 10× 設計
- [04 — RGB + mask 融合](stage2_ConvNeXt_Spatiotemporal/04_rgb_mask_fusion.md):stage 1 mask 進場、v1 frame-level vs v2 spatial-level fusion、1×1 conv、3 種 position embedding
- [05 — 銜接 Temporal Transformer](stage2_ConvNeXt_Spatiotemporal/05_handoff_to_temporal_transformer.md):CLS token、box_query DETR 風格、五個 head 的角色、ConvNeXt 教學收尾

待撰寫(下一章節主題):

- Spatiotemporal Transformer 內部:joint attention、box_query 學「鎖定」、CLS token 彙整 clip 語意
- 訓練主迴圈與 multi-task loss(box_l1 / giou / center_ce / heatmap_bce / type_ce)
- 推論流程與評估指標(`compute_cbd_prediction_metrics.py`)
- 模型 deployment 技術

## Git Remote 設定(已完成)

- `origin` → `git@github.com:dreamhunteryin/iBSAFE_CBD_v1.git`(您的 GitHub,push 目的地)
- `upstream` → `git@forge.icube.unistra.fr:CAMMA/code/bsafe/cbd_v1.git`(工程師 Luc Vedrenne 的 CAMMA GitLab,只用 `git fetch upstream` 拿更新,**不可** push)
