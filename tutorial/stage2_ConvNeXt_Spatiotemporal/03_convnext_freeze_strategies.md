# 第 3 章 — ConvNeXt 凍結策略與差別學習率

> 撰寫日期:2026-05-05
> 風格:精簡淺白、外科視角優先,程式碼放尾巴
> 目的:解釋本專案為何選 `last_stage` 凍結模式,以及搭配的 `backbone_lr` / `new_layers_lr` 雙軌學習率設計

---

## 為什麼需要「凍結」這件事?

當你拿一個**已經在 ImageNet 上訓練好的** ConvNeXt 開始訓本專案時,backbone 已經會「看圖」了。它認識邊緣、紋理、形狀梯度,甚至認識「動物的眼睛」「車輪」「樹葉」這些常見物體。

問題是:你的目標任務不是分類動物或物件,而是**找出腹腔鏡影像中的 CBD 位置**。所以你必須讓模型「微調」——但**怎麼微調**,有講究。

兩個極端:

- **完全不動 backbone**(凍結全部):用既有特徵抽取能力,只訓後面的新層。優點:訓得快、不會過擬合;缺點:backbone 不適應內視鏡風格的影像
- **整個 backbone 一起重訓**:讓 backbone 也學內視鏡影像;缺點:訓練資料量不夠時容易**忘記** ImageNet 上學到的低層基本功(學術上叫 catastrophic forgetting)

中間路線:**只解凍最後一個 stage**。因為——

- 早期 stage(stage 1-3)學的是「邊緣、紋理」這類**通用低層特徵**——不論你看的是貓還是膽囊,這些東西都長得差不多,沒必要重訓
- 最後 stage(stage 4)學的是「**整體語意**」——這部分跟「貓 vs 狗 vs 車」高度相關,但跟「膽囊 vs 肝臟 vs 總膽管」沒什麼關係,**需要重訓**

這就是 transfer learning 的核心智慧。

**外科類比**:像是你聘了一位「美容外科」資深主治來做「肝膽外科」——他多年訓練出來的解剖認知、組織質感判斷、止血技巧(底層基本功)直接沿用,不需要重學;但「肝膽外科特有的決策邏輯」(高層語意)必須重新訓練。讓他全部從零學是浪費,讓他完全套用美容外科的決策邏輯也不行。

---

## 本專案的三種凍結模式

`src/cbd/model.py` 第 32-38 行的 `resolve_backbone_mode` 函式定義了三種模式:

| 模式 | 行為 | model.py 內的實作 |
|---|---|---|
| `freeze_all` | 整個 backbone 凍結,只訓新層 | 第 59-61 行(v1)、第 139-140 行(v2 預設先全凍) |
| `last_stage` | 只解凍 `features[6:]`(最後一個 stage) | 第 144-147 行(v2) |
| `full` | 整個 backbone 一起訓 | 第 141-143 行(v2 完整解凍) |

`features[6:]` 是什麼?ConvNeXt 在 torchvision 裡的 `features` 是個 8 個元素的 Sequential(0:stem,1-2:stage1,3-4:stage2,5-6:stage3,7:stage4)。**`features[6:]` 取最後兩個元素——也就是 stage 3 後半 + stage 4**。

`bsafe_cbd.yaml` 第 50 行:

```yaml
unfreeze_backbone_mode: last_stage
```

所以**主訓練設定就是 `last_stage`**——保留前面的低層特徵,只解凍最後兩個 block 讓它適應內視鏡影像。

---

## 預設模式怎麼決定?

`src/cbd/model.py` 第 32-38 行:

```python
def resolve_backbone_mode(model_config: dict, variant: str) -> str:
    if "unfreeze_backbone_mode" in model_config:
        return str(model_config["unfreeze_backbone_mode"]).strip().lower()
    if variant == "v2_spatiotemporal":
        return "last_stage"
    freeze_backbone = bool(model_config.get("freeze_backbone", True))
    return "freeze_all" if freeze_backbone else "full"
```

邏輯:

1. **如果 config 有 `unfreeze_backbone_mode` → 用 config 指定的**
2. 否則,如果是 v2 → 預設 `last_stage`
3. 否則(v1)→ 看 `freeze_backbone` 旗標(預設 `True`),`True` 走 `freeze_all`,`False` 走 `full`

**設計推測**:工程師發現 v2 的空間 grid 預測對 backbone 的「最後一層」很敏感——如果 stage 4 沒重訓,模型可能無法針對「膽囊在哪個 16×16 區塊」這種**精細語意定位**做出判斷。所以 v2 預設就是 `last_stage`,寫死當作 sane default。

v1 反正空間資訊已經 pool 掉,backbone 重不重訓影響沒那麼大,所以 v1 預設只是 `freeze_all`(safe default,訓練快又穩)。

---

## 配套的差別學習率(differential learning rate)

**只解凍最後一個 stage 還不夠**——你還必須讓「解凍的 backbone 部分」學得**比新層慢**。否則梯度一波衝下來,backbone 會被推離原本的 ImageNet 表達。

`bsafe_cbd.yaml` 第 61-63 行:

```yaml
learning_rate: 1.0e-4   # 預設(萬一 backbone_lr/new_layers_lr 沒設時的 fallback)
backbone_lr: 1.0e-5     # ★ backbone 解凍部分的學習率
new_layers_lr: 1.0e-4   # ★ 新層的學習率
```

差 **10 倍**——backbone 比新層慢 10 倍學。

`src/cbd/engine.py` 第 157-180 行就是把這兩種 learning rate 透過 PyTorch optimizer 的 `param_groups` 機制設定的:

```python
backbone_params = self.model.backbone_trainable_parameters()  # 拿到解凍的 backbone 參數
backbone_ids = {id(p) for p in backbone_params}
new_params = [p for p in self.model.parameters()
              if p.requires_grad and id(p) not in backbone_ids]
# 兩個 param group:
#   group 1: new_params  ←  lr = 1e-4
#   group 2: backbone_params  ←  lr = 1e-5
```

`AdamW` 接受 `param_groups` 後,**每個 group 用自己的 lr**——這就是 PyTorch 標準的差別學習率寫法。

**外科類比**:像是訓練主治+fellow 共同學新術式——主治(backbone)只在「自己原本不熟的最後一步」做小幅校正(慢慢學);fellow(新層)從零開始,放手讓他從頭學起(快快學)。兩個人速度不同步,但結果是整個團隊磨合最快。

---

## 為什麼差別學習率是 10 倍而不是其他倍數?

這個比例(10×)是深度學習圈子常見的經驗值。教科書級的解釋:

- 預訓練的 backbone 已經處在「**接近 ImageNet 任務的最佳解附近**」,只需要微調——所以學習率要小
- 新層是「從亂數初始化開始」,離最佳解很遠——學習率要大
- 太接近(差 1-3 倍):backbone 容易被新層的梯度帶偏,**忘記基本功**
- 太懸殊(差 100 倍以上):backbone 幾乎不動,等於白解凍

10 倍是「夠不同、又不會壓抑 backbone 微調」的常見甜蜜點。本專案沿用這個慣例。

---

## 如果你想自己改設定該怎麼動?

**不要改既有 yaml**——按 CLAUDE.md 規則,既有檔案不能改。但你可以:

1. **新建一個 `configs/my_experiment.yaml`**,複製 `bsafe_cbd.yaml`,只改你想試的部分
2. 試 `unfreeze_backbone_mode: full`(整個 backbone 解凍)看訓練是否更穩定,但小心過擬合
3. 試 `backbone_lr: 5e-6`(更慢)、`new_layers_lr: 5e-4`(更快),看 loss 曲線變化

如果你只是想「**看現有設定真實跑時 backbone 哪些參數 requires_grad**」,可以加個 debug 腳本:

```python
from cbd.model import CBDBoxModel
import yaml
config = yaml.safe_load(open("configs/bsafe_cbd.yaml"))
model = CBDBoxModel(config["model"])
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
backbone_trainable = sum(p.numel() for p in model.backbone_trainable_parameters())
print(f"total: {total:,}")
print(f"trainable: {trainable:,}")
print(f"backbone_trainable: {backbone_trainable:,}")
```

---

## 回到外科直覺:為何「不要動低層」這麼重要?

這個問題其實有對應的外科訓練哲學。

當住院醫師升上 fellow 學專科時,**沒有人會把他原本的「打結、止血、組織辨識、無菌操作」這些底層技能砍掉重練**——這些是跨專科通用的基本功,辛苦練起來的成本不能浪費。

fellow 的訓練重點放在「**那個專科特有的高階決策**」——肝膽 fellow 學的是 Calot's triangle 怎麼分離、CBD 怎麼確認、什麼時候該 convert open;**不是**重新學怎麼打結。

ConvNeXt 的 stage 1-3 就是「打結止血」這種跨任務基本功,沒必要在內視鏡資料上重訓——硬要重訓還可能因為**資料量不夠**而學壞。stage 4 才是「肝膽專科的高階語意」,這部分必須**針對內視鏡影像重新訓練**才能用。

---

## 這一章你需要帶走的重點

1. **凍結策略 = transfer learning 的精髓**——保留 backbone 的低層通用特徵,只重訓高層任務語意
2. 三種模式:`freeze_all`(全凍)、`last_stage`(只解凍 features[6:])、`full`(全解凍)
3. **本專案 v2 預設 `last_stage`**(`bsafe_cbd.yaml:50`),而且這是寫死在程式碼的 sane default(`model.py:35-36`)
4. v1 預設 `freeze_all`——因為 v1 把空間 pool 掉,backbone 對結果影響沒 v2 那麼敏感
5. 差別學習率:`backbone_lr=1e-5`、`new_layers_lr=1e-4`,**差 10 倍**——backbone 慢學、新層快學
6. 實作方式:PyTorch optimizer 的 `param_groups` 機制(`engine.py:157-180`)
7. 外科直覺:fellow 訓練不重學基本功,只重訓專科高階決策——同樣的智慧

---

## 進一步深挖的線索

- transfer learning 的經典論文:Yosinski et al. *How transferable are features in deep neural networks?*(NeurIPS 2014)——首次系統化分析「越深越專屬,越淺越通用」
- 醫療 vision 的 transfer learning 大型 benchmark:Raghu et al. *Transfusion: Understanding Transfer Learning for Medical Imaging*(NeurIPS 2019)——比較 ImageNet 預訓練 vs 從零訓 在醫療影像上的效果
- 差別學習率最早被推廣的場域:fast.ai 的 *discriminative fine-tuning* 概念(ULMFiT 論文,Howard & Ruder 2018)

---

## 對話脈絡記錄

- **2026-05-05**:第 3 章拆解凍結策略+差別學習率。`resolve_backbone_mode`(model.py:32-38)寫死了 v2 → `last_stage` 的 sane default,工程師沒在 config 註解,但從程式邏輯可以反推設計理由。差別學習率 10× 不是隨便挑的——是 transfer learning 圈的慣例。

---

## 程式碼速查總表

### 三種凍結模式的實作

| 模式 | model.py 位置 | 程式碼摘要 |
|---|---|---|
| `freeze_all`(v1) | 第 59-61 行 | 全 backbone `requires_grad = False` |
| `freeze_all`(v2) | 第 139-140 行 | v2 預設先把全 backbone 凍結 |
| `full`(v2) | 第 141-143 行 | 「全凍 → 全解凍」覆蓋上 |
| `last_stage`(v2) | 第 144-147 行 | 「全凍 → 只把 features[6:] 解凍」 |

### 預設模式的決定邏輯

| 檔案 | 行號 | 規則 |
|---|---|---|
| `model.py:32-38` | `resolve_backbone_mode` 函式 | 先看 config 有無 `unfreeze_backbone_mode`;沒有就看 variant;v2 預設 `last_stage`、v1 看 `freeze_backbone` 旗標 |
| `bsafe_cbd.yaml:50` | `unfreeze_backbone_mode: last_stage` | 主設定明文指定,不靠 fallback |

### 差別學習率的設定

| 檔案 | 行號 | 內容 |
|---|---|---|
| `bsafe_cbd.yaml:61` | `learning_rate: 1.0e-4` | fallback default |
| `bsafe_cbd.yaml:62` | `backbone_lr: 1.0e-5` | 解凍 backbone 用的 lr |
| `bsafe_cbd.yaml:63` | `new_layers_lr: 1.0e-4` | 新層用的 lr(差 10×) |
| `engine.py:157` | `backbone_params = self.model.backbone_trainable_parameters()` | 拿出 backbone 內仍需訓練的參數 |
| `engine.py:158` | `backbone_ids = {id(p) for p in backbone_params}` | 用記憶體 id 區分 |
| `engine.py:159-163` | 篩出 `new_params` | requires_grad 但不在 backbone 中的參數 |
| `engine.py:165-180` | 構造 `optimizer_params` | 兩個 param group 給 AdamW |

### 取出 backbone 可訓練參數的介面

| 檔案 | 行號 | 內容 |
|---|---|---|
| `model.py:104-105` | `CBDV1GlobalPool.backbone_trainable_parameters` | 對 v1 |
| `model.py:203-204` | `CBDV2Spatiotemporal.backbone_trainable_parameters` | 對 v2 |
| `model.py:265-266` | `CBDBoxModel.backbone_trainable_parameters` | 統一介面,內部委派給 impl |

### 快速驗證命令

```bash
# 確認 yaml 中差別學習率設定
grep -n "lr\|learning_rate" configs/bsafe_cbd.yaml

# 確認 model.py 三種模式的處理
grep -n "freeze_all\|last_stage\|features\[6:\]" src/cbd/model.py

# 用 Python 計算當前模式下實際凍結 / 解凍的參數量
python -c "
import yaml
from cbd.model import CBDBoxModel
config = yaml.safe_load(open('configs/bsafe_cbd.yaml'))
model = CBDBoxModel(config['model'])
total = sum(p.numel() for p in model.parameters())
backbone_total = sum(p.numel() for p in model.impl.rgb_backbone.parameters())
backbone_trainable = sum(p.numel() for p in model.backbone_trainable_parameters())
new_trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad and not any(p is bp for bp in model.backbone_trainable_parameters()))
print(f'backbone total:      {backbone_total:>12,}')
print(f'backbone trainable:  {backbone_trainable:>12,}  ({100*backbone_trainable/backbone_total:.1f}%)')
print(f'new layers trainable:{new_trainable:>12,}')
"
```

最後一個命令會印出**實際凍結比例**——預設設定下,backbone 應該只有 stage 4 那部分(大約佔 ConvNeXt 全部參數的 30-40%)是 trainable。
