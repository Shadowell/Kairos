# 工业级多尺度量化时序模型架构设计说明书 (Multi-Scale Alpha Architecture)

> **文档版本**: 1.0.0  
> **状态**: RFC / 核心架构演进方案  
> **适用范围**: Kairos 模型核心架构、特征引擎（Feature Engine）、数据流流水线与在线推理服务（Runtime）

---

## 1. 架构演进背景与核心动因

Kairos 现行架构（v0.1.0）基于 Kronos 探索了“BSQ Tokenizer + 外生变量旁路通道（Exogenous Bypass Channel）+ 分位数回归头”的微调范式。然而，在 OKX 永续合约多通道实际实验中暴露了若干结构性缺陷（详见 [`docs/CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) 及 GitHub Issue #1 ~ #8）：

1. **建模目标冲突（精神分裂）**：强行将生成式自回归语言模型（预测下一个价格 Token）与判别式量化 Alpha（预测横截面收益率相对强弱）混合训练。高阶自回归 CE Loss 掩盖了微弱的金融 Alpha 信号。
2. **多尺度时序严重错配**：将 8 小时变动一次的宏观费率（Funding Rate）用 `ffill` 强行拉平成 1 分钟网格，在长仅 256 根 bar（~4.2 小时）的高频局部上下文中，根本无法覆盖衍生品的结算宏观周期，特征滚动 Z-Score 严重退化为 0。
3. **特征工程与模型容量倒挂**：人为将 24 个可由 OHLCV 简单非线性计算出的技术指标（RSI, MACD, Boll 等）硬编码进 32 维外生通道，占用了宝贵的通道容量，却缺乏真正的外部增量信息（如截面 Beta/残差、订单流深度、全网多空杠杆率）。
4. **训练与在线服务断层**：离线采用批量 pandas 向量化清洗，在线服务（`kairos-serve`）直接回退到原版自回归生成，未打通统一的特征算子运行时，极易产生特征穿越与分布漂移（Training-Serving Skew）。

本说明书系统性提出 **Kairos 2.0 工业级多尺度量化架构（Multi-Scale Alpha Architecture）**，明确量化判别式定位，重构特征与模型体系。

---

## 2. 核心设计原则

1. **Alpha 判别式主导，与生成式解耦（Discriminative Alpha First）**
   - 时序基础模型（Kronos Backbone）作为**通用时序特征抽取器**，通过冻结或弱学习率保留底层几何先验。
   - 彻底剥离昂贵且与量化排序无关的 Token 自回归生成任务，全面转向**截面排序优化（Pairwise / Listwise Ranking）**与**多步收益率方差归一化预测**。
2. **多尺度分层交互（Hierarchical Multi-Temporal Scales）**
   - 微观尺度（1min / 5min）：专注捕捉高频微观结构、短周期动量与波动率挤压。
   - 宏观尺度（1h / 8h / 1d）：专注捕捉衍生品期限套利空间、资金费率分位、持仓增减仓趋势与全市场宏观 Beta。
   - 采用 **Cross-Attention（交叉注意力）与 FiLM（Feature-wise Linear Modulation）** 进行多尺度融合，而非粗暴的行网格广播。
3. **真实外生信息增量（Genuine Exogenous Increment）**
   - 剔除冗余的 OHLCV 人工二次变换，释放特征槽位。
   - 聚焦三类核心外生信息：**跨币种截面相对强弱与残差收益**、**衍生品定价偏离（Basis, Funding Rate）**、**全网杠杆与流动性分布（OI, 多空人数比）**。
4. **离线-在线统一特征引擎（Unified Feature Store & Runtime）**
   - 统一抽象状态化算子（Stateful Operator），使得离线回测批量向量化与在线单步流式更新逻辑完全同构，消除 Training-Serving Skew。

---

## 3. 全局分层系统架构

系统划分为清晰的 5 层结构：

```
┌────────────────────────────────────────────────────────────────────────┐
│                        5. 策略与服务层 (Strategy / Serve)              │
│   • 实时 HTTP/gRPC 推理服务 (kairos-serve)  • 截面多空对冲组合构建器   │
└────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ 实时特征向量 / 模型分位数打分
┌────────────────────────────────────────────────────────────────────────┐
│                        4. 模型推理与评估层 (Model & Evaluation)         │
│   • Multi-Scale Kronos Alpha Predictor (微观-宏观多尺度 Transformer)    │
│   • 截面 Rank-IC / ICIR / 分组回测引擎 (统一时区与确定性步长)         │
└────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ 模型权重 / 梯度反向传播
┌────────────────────────────────────────────────────────────────────────┐
│                        3. 训练执行层 (Distributed Training)            │
│   • DDP 分布式确定性采样器 (Deterministic Distributed Sampler)         │
│   • 梯度累积与通信优化 (with model.no_sync())                           │
│   • 多 Horizon 收益率方差均衡损失 (Balanced Multi-Horizon Pinball Loss)│
└────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ 训练 / 验证 / 测试数据集 Bundle
┌────────────────────────────────────────────────────────────────────────┐
│                        2. 统一特征引擎 (Unified Feature Engine)         │
│   • 微观特征抽取器 (1min OHLCV Waveform)                               │
│   • 宏观衍生品状态抽取器 (8h Funding, Basis, OI Trend)                  │
│   • 截面残差引擎 (Cross-Sectional Market Neutralizer)                  │
│   • 离线连续块切分 (Gap-Aware Non-leaking Block Splitter)             │
└────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ 统一 UTC 毫秒对齐 Parquet
┌────────────────────────────────────────────────────────────────────────┐
│                        1. 多源数据采集层 (Data Ingestion Layer)        │
│   • 现货与永续行情采集器 (OKX / Binance Vision)                        │
│   • 衍生品 Sidecar 采集器 (Funding Rate, OI, Spot Mid)                 │
│   • 严格 UTC 时区守卫 (Timezone Guard, 消除机器本地时区偏移)           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 模型架构设计：Multi-Scale Kronos Alpha

### 4.1 核心网络拓扑

```
[微观 1min K-line: B, T_micro, 6]            [宏观衍生品/截面状态: B, T_macro, D_macro]
           │                                                    │
           ▼                                                    ▼
    Kronos Tokenizer                                     Macro Linear Projector
           │                                                    │
    Hierarchical Embedding                                      │
           │                                                    │
           ▼ (Micro Tokens)                                     ▼ (Macro Context Keys/Values)
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                        Multi-Scale Transformer Backbone                           │
 │                                                                                   │
 │   Layer 1..N:                                                                     │
 │   1. Self-Attention over Micro Window (捕捉高频时序几何)                           │
 │   2. Cross-Attention: Micro Queries attend to Macro Context (宏观条件调制)       │
 │   3. Feed-Forward Network                                                         │
 └───────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼ (Latent Hidden State)
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                       Unified Alpha Prediction Heads                              │
 │                                                                                   │
 │   [Head A] Multi-Horizon Quantile Return Head (1, 5, 15, 30 min)                  │
 │            • 目标: 真实对数收益率分位数 log(P_{t+h} / P_t)                         │
 │            • 损失: 方差归一化 Pinball Loss                                        │
 │                                                                                   │
 │   [Head B] Cross-Sectional Ranking Head (Pairwise Margin Loss)                    │
 │            • 目标: 同一时间戳截面相对强弱排序 (Rank-IC 导向)                       │
 └───────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 模块详细规格

#### 1. 微观时序流（Micro Sequence Stream）
- **输入**：$X_{\text{micro}} \in \mathbb{R}^{B \times T_{\text{micro}} \times 6}$，标准 OHLCV 窗口（默认 $T_{\text{micro}} = 256$ 根 1min bar）。
- **编码**：经过冻结的 `KronosTokenizer` 转换为离散 Token `(s1, s2)`，映射至 `d_model` 维度的 Embedding 空间。

#### 2. 宏观上下文流（Macro Context Stream）
- **输入**：$X_{\text{macro}} \in \mathbb{R}^{B \times T_{\text{macro}} \times D_{\text{macro}}}$，以 1 小时为步长，覆盖过去 72 小时（$T_{\text{macro}} = 72$）。
- **内容**：
  - 过去 9 次结算的资金费率分布与累计溢价；
  - 过去 72 小时的持仓量变化斜率与成交持仓比（Volume-to-OI Ratio）；
  - 过去 72 小时标的相对 BTC/ETH 的截面超额收益（Residual Return）；
  - 永续-现货基差（Basis）均值与偏离度。
- **投影**：通过轻量两层 MLP 映射为 `d_model` 维度的宏观上下文键值对（Keys/Values）。

#### 3. 跨尺度调制机制（Cross-Attention & FiLM）
在 Transformer 的关键隐藏层中插入 Cross-Attention 层：
$$\text{Query} = H_{\text{micro}}, \quad \text{Key, Value} = H_{\text{macro}}$$
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
微观高频的每一根 1min K 线在计算注意力时，能够感知宏观上当前处于“资金费率极度过热”还是“持仓量巨幅离场”的宏观状态，实现物理意义上的多尺度调制。

### 4.3 训练损失函数：解耦 CE，专注 Alpha

弃用原版语言模型预测下一个 Token 的 Cross-Entropy 损失，定义 **Alpha-Centric Joint Loss**：

$$\mathcal{L} = \mathcal{L}_{\text{multi-horizon}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}}$$

#### 1. 方差均衡的多步分位数损失（Variance-Normalized Pinball Loss）
对预测步长 $h \in \{1, 5, 15, 30\}$，直接对真实对数收益率 $r_{t, h} = \ln(P_{t+h} / P_t)$ 预测 $Q$ 个分位数：
$$\mathcal{L}_{\text{multi-horizon}} = \sum_{h \in H} \frac{1}{\sigma_h} \text{PinballLoss}(\hat{r}_{t, h}^{(q)}, r_{t, h})$$
- **关键设计**：除以历史样本步长标准差 $\sigma_h \approx \sigma_1 \sqrt{h}$ 进行损失归一化，彻底消除 $h=30$ 的绝对误差主导并淹没 $h=1, 5$ 梯度的问题。

#### 2. 截面排序对比损失（Pairwise Ranking Loss）
在同批次抽取的 $N$ 个不同标的中，若在时间 $t$ 标的 $A$ 的实际收益率大于标的 $B$（$r_A > r_B$），约束预测分位数均值：
$$\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{P}|} \sum_{(A, B) \in \mathcal{P}} \max(0, \gamma - (\hat{r}_A - \hat{r}_B))$$
直接将模型参数向最大化 **Rank-IC（Spearman 相关系数）** 的方向驱动。

---

## 5. 特征工程体系重构

彻底淘汰 24 维冗余 OHLCV 技术指标，构建全新的 **微观-宏观-截面三元特征表（MMX Feature Schema）**：

| 特征类别 | 字段名称 | 频率 | 物理含义与量化逻辑 | 计算公式 / 来源 |
| :--- | :--- | :--- | :--- | :--- |
| **微观原始** | `ohlcv_norm` | 1min | 基础价格与成交量几何形态 | 由 Kronos Tokenizer 直接处理 |
| **微观微观结构** | `vol_imbalance` | 1min | 主动买卖方向与成交量失衡 | 阳线/阴线成交量比率 |
| **微观微观结构** | `realized_vol` | 1min | 日内已实现高频波动率 | 过去 30 步对数收益率平方和 |
| **宏观衍生品** | `funding_rate_raw` | 8h / 1h | 当前资金费率水平 | 交易所结算接口 |
| **宏观衍生品** | `funding_percentile` | 1h | 资金费率在过去 30 天的历史分位数 | 720 小时滚动经验分位数，范围 $[0, 1]$ |
| **宏观衍生品** | `basis_z` | 1h | 永续价格与现货价格偏离度 Z-Score | $(P_{\text{perp}} / P_{\text{spot}} - 1)$ 过去 72h Z-Score |
| **宏观衍生品** | `oi_log_change` | 1h | 全网未平仓合约量变化趋势 | $\ln(\text{OI}_t / \text{OI}_{t-1})$ |
| **宏观衍生品** | `vol_oi_ratio` | 1h | 成交持仓比（识别纯投机 vs 结构换手） | 1h 累计成交量 / OI |
| **宏观截面** | `market_beta` | 1h | 标的对 BTC/ETH 市场大盘的敏感度 | 过去 72h 滚动回归斜率 $\beta$ |
| **宏观截面** | `residual_ret` | 1h | 剥离大盘后的纯特质Alpha动量 | $r_{\text{sym}} - \beta \cdot r_{\text{BTC}}$ |
| **宏观截面** | `cross_rank_ret` | 1h | 标的在全市场 Top100 中的相对动量排名 | 截面百分比排名 $[0, 1]$ |

---

## 6. 数据流与数据切分规范

为杜绝数据泄露并保证分布式训练正确性，严格重构数据切分与加载逻辑：

### 6.1 连续块时序切分（Gap-Aware Block Splitter）
1. **基于日期的确定性切块**：
   按自然日（UTC 日期）划分 block，严禁将数据行数（row index）误用为天数除法。
2. **断点保护与隔离缓冲区（Purging & Embargo）**：
   每个划分给验证集的 block 两端增加宽度为 $W_{\text{lookback}} + W_{\text{horizon}}$ 的保护缓冲区（Embargo），该区间样本既不进入训练集也不进入验证集，杜绝任何时间边缘信息泄露。
3. **断点感知滑动窗口（Contiguous Segment Indexing）**：
   在打包输出中，明确记录每个连续时序片段的起始和结束点。`dataset.py` 构建样本索引时，严格断言 `end <= segment_end`，**坚决禁止滑动窗口跨越被抽走的断点进行物理拼接**。

### 6.2 确定性分布式加载器（DDP Dataset Spec）
- 恢复 PyTorch 标准 `__getitem__(self, index: int)`，样本由传入的全局 `index` 直接定位。
- 采用确定性 `DistributedSampler(dataset, shuffle=True/False, seed=seed)`。
- 验证集强制使用 `DistributedSampler(dataset, shuffle=False)`，保证每个验证 epoch 对固定样本进行完全一致、确定性且覆盖完整的评估。

---

## 7. 在线低延迟推理服务体系 (kairos-serve)

重构 `kairos/deploy/serve.py`，打通微调模型上线的最后一公里：

```
客户端请求 (JSON)
  • symbol: "BTC/USDT:USDT"
  • lookback_bars: 256 根 1min K 线
  • macro_state: { funding_rate, oi, spot_close } (可选，缺省则自动由服务侧回填)
       │
       ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │               Unified Streaming Feature Pipeline                │
 │   • 验证并规范输入时间序列 (UTC 校验)                            │
 │   • 增量计算微观特征与宏观上下文张量                             │
 └─────────────────────────────────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │              Kronos Multi-Scale Alpha Model Engine              │
 │   • 载入 fine-tuned best_model                                  │
 │   • 单次前向传播 (Single Forward Pass, ~15ms on GPU)             │
 └─────────────────────────────────────────────────────────────────┘
       │
       ▼
 响应输出 (JSON)
  • multi_horizon_returns: {
        "h1":  { "q10": -0.0005, "q50": 0.0002, "q90": 0.0009 },
        "h5":  { "q10": -0.0012, "q50": 0.0006, "q90": 0.0021 },
        "h15": { "q10": -0.0025, "q50": 0.0014, "q90": 0.0045 },
        "h30": { "q10": -0.0040, "q50": 0.0025, "q90": 0.0078 }
    }
  • alpha_score: 0.0025 (用于多空截面排序的标准分)
  • confidence_spread: 0.0118 (q90 - q10 置信区间宽度，用于风控仓位调制)
```

---

## 8. 演进路线与实施里程碑 (Implementation Roadmap)

| 阶段 | 周期 | 核心交付物 | 验收标准 |
| :--- | :--- | :--- | :--- |
| **Phase 1: 缺陷修复与基础加固** | 1 周 | 修复 Issue #1 ~ #8（时区统一、interleave 行数 bug 修复、增量采集修复、DDP 采样与梯度累加修复） | 现有回归测试全部通过，本地 CPU smoke 测试验证确定性验证集与 DDP 行为 |
| **Phase 2: 目标与损失体系重构** | 1 周 | 剥离自回归 CE，实现方差归一化的 Multi-Horizon Pinball Loss 与 Pairwise Rank Loss；统一对数收益率目标量纲 | 在现有数据集上验证 h1, h5, h30 梯度均衡性，h1/h5 IC 摆脱负向与近零状态 |
| **Phase 3: 多尺度特征与 Cross-Attention** | 2 周 | 重构特征引擎，引入宏观 1h/8h 衍生品特征与截面残差；在模型中加入多尺度交叉注意力调制层 | 在 Top100 永续数据集上完成全量微调，h30 Rank-IC 达到稳定 `> +0.06`，ICIR `> +0.5` |
| **Phase 4: 统一推理 Runtime 与服务上线** | 1 周 | 重写 `kairos-serve`，支持微调多尺度模型与统一特征抽取，输出置信区间与 Alpha Score | 端到端单次推理延迟 $\le 30\text{ms}$，提供完备的 API 文档与自动化集成测试 |

---

## 9. 术语与参考

- **Rank-IC (Information Coefficient)**: 因子在不同标的间的预测值排名与未来实际收益率排名的 Spearman 相关系数。
- **Cross-Sectional Alpha**: 剥离全市场系统性涨跌（Beta）后，寻找个币相对强弱的超额收益能力。
- **FiLM (Feature-wise Linear Modulation)**: 基于宏观条件向量对微观隐藏层进行动态缩放与偏置（Affine Transform）的调制技术。
- **Training-Serving Skew**: 离线批量特征工程与在线实时流式特征计算不一致导致模型线上效果大幅退化的工程现象。
