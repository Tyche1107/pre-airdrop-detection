# 实验思维导图 + 结果总览

---

## 核心逻辑链

```
ARTEMIS 只能事后识别，钱已被薅走
        ↓
能提前识别吗？
        ↓
[01] 特征工程 + [02] LightGBM
        → AUC 0.905 > ARTEMIS 0.803 ✅
        ↓
能提前多久？
        ↓
[03][12] Temporal Ablation T0→T-180
        → T-90: 0.902 / T-180: 0.895，几乎不掉
        ↓
        ├─ 为什么这么早就能识别？
        │        ↓
        │   [04][10] 特征重要性 + SHAP
        │        → buy_collections 最稳定
        │        → 越早"互动广度"越重要于"交易量"
        │        ↓
        │   所有特征都必要吗？
        │        ↓
        │   [06] 特征组消融
        │        → Diversity单组就超ARTEMIS
        │        → DeFi组几乎无用
        │
        ├─ 复杂图模型更好吗？
        │        ↓
        │   [07] 图特征 + LightGBM
        │        → +0.001，边际贡献极小
        │        ↓
        │   [09][15] GNN 事后 vs 事前
        │        → 事后GNN: 0.976 / 事前GNN: 0.586（崩了）
        │        → GNN依赖空投后数据，事前无效
        │
        ├─ 模型是真泛化还是记忆噪声？
        │        ↓
        │   [08] 时间泛化 + 种群泛化
        │        → 时间: -0.006 / 种群: 0.744 ✅
        │        ↓
        │   [05] 阈值分析
        │        → 实际部署可调精度/召回权衡
        │        ↓
        │   [11] Sybil内部聚类
        │        → 3类行为模式（散户/中量/超级机器人）
        │        ↓
        │   [13] 对抗鲁棒性
        │        → 猎人降低diversity可逃检，但积分也跟着降
        │        → 无法两全，自我矛盾
        │
        └─ 能推广到其他协议吗？
                 ↓
        [14][16b] Blur→Hop 零样本 vs 微调
                 → 零样本: 0.50 / 1%微调: 0.981
                 ↓
        为什么零样本差？协议间规律不同？
                 ↓
        [16] 三协议LOPO（Blur / Hop / Gitcoin）
                 → 零样本 0.22~0.47
                 → Gitcoin域内只有0.634（捐赠行为 ≠ 交易行为）
                 ↓
        同类协议迁移会更好吗？
                 ↓
        [进行中] LayerZero（Hop同类：桥接）
                 → 假设 Hop→LZ > Blur→LZ
```

---

## 实验结果图

### [02][03] 基准模型 vs ARTEMIS + 时间消融

**结论：** 事前简单模型 AUC 0.905，全面超越事后复杂 GNN（ARTEMIS 0.803）

![model_comparison](data/model_comparison_plot.png)

---

**结论：** T0→T-90 AUC 仅降 0.006，提前3个月仍稳定

![temporal_ablation](data/temporal_ablation_plot.png)

---

### [12] 扩展时间消融 T-120 / T-150 / T-180

**结论：** 信号6个月前仍存在（AUC 0.895），衰减曲线平滑

![temporal_ablation_extended](data/temporal_ablation_extended_plot.png)

---

### [04] 特征重要性跨时间窗稳定性

**结论：** buy_collections 稳定第一；unique_interactions 越早权重越高

![feature_stability](data/feature_stability_plot.png)

---

### [10] SHAP 分析

**结论：** 高 buy_collections = 强正向信号；低 wallet_age = 新号批量特征

![shap](data/shap_beeswarm.png)

---

### [06] 特征组消融

**结论：** Activity 0.863 > Diversity 0.854 > Behavioral 0.848 > Volume 0.846 > DeFi 0.544

![ablation](data/ablation_plot.png)

---

### [05] Precision-Recall + 阈值分析

**结论：** threshold=0.5 时 F1 最优；实际部署可按需调整 precision/recall 权衡

![pr](data/pr_and_comparison_plot.png)

![threshold](data/precision_recall_threshold.png)

---

### [07] 图特征增强

**结论：** 行为特征 0.904 + 图特征 → 0.905，图信息边际贡献 +0.001

![graph](data/graph_augmentation_plot.png)

---

### [09][15] GNN 对比：事后 vs 事前

**结论：** 事后 GNN 0.976，事前 GNN 崩至 0.586；GNN 依赖空投后 transfer，事前无效

| 模型 | 数据 | AUC |
|------|------|-----|
| ARTEMIS（论文） | 事后 | 0.803 |
| LightGBM（我们，T-30） | 事前 | 0.905 |
| ArtemisNet GNN（我们复现） | 事后 | 0.976 |
| ArtemisNet GNN（我们，T-30） | 事前 | 0.586 |

---

### [08] 泛化实验

**结论：** 时间泛化仅降 0.006；种群泛化 0.744，模型非记忆噪声

![generalization](data/generalization_plot.png)

---

### [11] Sybil 内部聚类

**结论：** 3类猎人：散户（49K个）/ 中量（601个）/ 超级机器人（6个，9099次买入）

![clustering](data/sybil_clustering_plot.png)

---

### [13] 对抗鲁棒性

**结论：** 降低 diversity 可逃避检测，但同时减少积分——猎人无法两全

| 场景 | AUC |
|------|-----|
| 正常 Sybil | 0.904 |
| Diversity -50% | 0.883 |
| Diversity -80% | 0.869 |
| 完全模仿正常用户 | 0.862 |

![adversarial](data/adversarial_plot.png)

---

### [14][16b] 跨协议：Blur → Hop（零样本 vs 微调）

**结论：** 零样本 0.50，1% 微调即达 0.981，1% 和 20% 效果相同

![cross_protocol](data/cross_protocol_plot.png)

---

### [16] 三协议 LOPO（Blur / Hop / Gitcoin）

**结论：** 零样本 0.22~0.47；Gitcoin 域内仅 0.634（捐赠场景特征体系不同）

| 训练 | 测试 | AUC | 类型 |
|------|------|-----|------|
| Hop + Gitcoin | Blur | 0.220 | zero-shot |
| Blur + Gitcoin | Hop | 0.376 | zero-shot |
| Blur + Hop | Gitcoin | 0.466 | zero-shot |
| Blur 域内 | Blur | 0.862 | in-domain |
| Hop 域内 | Hop | 0.975 | in-domain |
| Gitcoin 域内 | Gitcoin | 0.634 | in-domain |

---

### [进行中] LayerZero 多链数据

ETH + Arbitrum + Polygon，29849 个地址，后台抓取中

**假设：** Hop → LZ > Blur → LZ（同类桥接协议间迁移更好）
