# Kronos A股调优完整手册（数据 → 训练 → 部署 → HF → 推理）

> 面向完全没有 Transformer 调优经验的读者。每一步都给出"为什么这么做 / 具体命令 / 常见坑"。
>
> 配套脚本已放在：
> - `kairos-collect` —— akshare 多源 A 股 K 线采集
> - `kairos/data/features.py` —— 技术指标 / 因子工程
> - `kairos-prepare` —— 转 Kronos 训练 pkl
> - `kairos-push-hf` —— 上传到 Hugging Face
> - `kairos-serve` —— FastAPI 推理服务
> - `kairos/models/kronos_ext.py` —— 扩展外生变量通道的 Transformer 改造示例

---

## 0. 为什么你当前预测"不准"——先定位瓶颈

Kronos 是一个**自回归 K 线生成模型**（decoder-only，类似 GPT），它只"看"过 OHLCV+amount 这 6 维数据。在 A 股场景下它大概率会遇到 3 个问题：

1. **域偏移（domain shift）**：预训练数据以全球 45 个交易所（含加密、美股、港股）为主，A 股节奏、涨跌停、T+1、换手率结构都不一样。
2. **信息严重不足**：只有 OHLCV 没有资金流、基本面、板块、情绪、宏观，模型永远学不到"为什么涨"。
3. **目标函数错配**：它学的是 token 级的交叉熵（复刻 K 线形状），不是"下一个 Δclose 的方向/分位数"。形状像 ≠ 点位准。

调优的核心路径：

```
[A-share 专属大规模分钟/日 K]  →  [加入外生因子]  →  [Tokenizer 微调]
→  [Predictor 微调（+ 外生通道改造）]  →  [分位/方向校准头]  →  [回测择优]
```

---

## 1. 数据收集（A 股 K 线，粒度越小越好）

### 1.1 akshare 分钟级的**硬约束**（必须先知道）

| 接口 | 粒度 | 可回溯历史 | 备注 |
|---|---|---|---|
| `stock_zh_a_hist` | 日/周/月 | **全历史**（2005-至今） | 首选、前复权/后复权 |
| `stock_zh_a_hist_min_em` | 1min | 近 **5** 个交易日 | 数据源硬限制 |
| `stock_zh_a_hist_min_em` | 5/15/30/60min | **近 3~6 个月** | 会随时间变 |
| `stock_zh_a_minute` (新浪) | 1/5/15/30/60min | 1min 近 9 日；5min 近 1 年 | 新浪源，偶尔限频 |
| `stock_zh_a_hist_pre_min_em` | 1min 盘前盘中 | 当日 | 做实时用 |

**结论**：想要"尽可能细"且"历史尽可能长"，在**免费**开源栈只有两条路：

- **路线 A（推荐入门）**：日线全历史（~20 年 × 5000 股 ≈ 2500 万行）+ 5min 过去 6 个月（日度累积到 2~3 年）。训练 context=512 条 5min ≈ 1 周窗口，足够学盘口微结构。
- **路线 B（想做更精）**：日线全历史 + **从今天起**每日 cron 抓 1min，累积半年后就有一份私有 1min 库。
- **路线 C（预算充足）**：改用 **Tushare Pro（≥2000 积分）/聚宽/米筐/万得**，能直接拿全历史 1min。

本手册的采集脚本同时支持 A/B 两条路，并预留了 `--source tushare` 钩子。

### 1.2 使用采集脚本

```bash
# 安装依赖
pip install akshare==1.14.* pandas pyarrow tqdm schedule

# 1. 全市场日线，一次性全历史
kairos-collect \
    --universe csi800 \
    --freq daily \
    --start 2015-01-01 \
    --end   2026-04-17 \
    --adjust qfq \
    --out   ./raw/daily

# 2. 全市场 5min（尽可能长，通常 3~6 月）
kairos-collect \
    --universe csi300 \
    --freq 5min \
    --adjust qfq \
    --out   ./raw/5min

# 3. 每日定时累积（crontab 15:30 跑一次）
kairos-collect \
    --universe csi300 --freq 1min \
    --daily-append --out ./raw/1min
```

产出：`./raw/{freq}/{symbol}.parquet`，字段统一为：
`datetime, open, high, low, close, volume, amount, turnover, pct_chg, vwap`。

### 1.3 股票池（universe）怎么选

| 池子 | 数量 | 用途 |
|---|---|---|
| csi300 | 300 | 最干净的大盘股，调参首选 |
| csi500/800/1000 | 500/800/1000 | 中小盘，样本更多 |
| all_ashare | ~5200 | 练大模型、最终微调 |

建议：**先 csi300 跑通全链路，再扩到 csi800 或全市场**。训练数据量从几 GB 涨到几十 GB，相应成本和时长也会变化。

---

## 2. 特征扩展：K 线之外该加什么

Kronos 原生只吃 `[open, high, low, close, volume, amount]` 共 6 维。下面按照"增益/实现难度"优先级给出建议，**前 3 类几乎一定要加**：

### 2.1 技术面（廉价、立刻能加）

- **收益/对数收益**：`log(close_t / close_{t-1})`——数值稳定，建议作为第一序列特征
- **波动率**：ATR(14)、Parkinson、RV（realized volatility）
- **动量**：ROC(5/10/20)、RSI(14)、MACD、Stoch(9,3)
- **均线/偏离度**：MA5/20/60、close/MA - 1、BOLL (N=20, K=2)
- **量价**：OBV、MFI、AMF (amount / volume 近似 vwap)
- **市场微结构**：当日振幅 `(high-low)/pre_close`、上影/下影比、十字星标记
- **换手率**：turnover（最重要的 A 股特色因子之一）
- **资金流**：主力/超大单净流入（akshare `stock_individual_fund_flow`）

### 2.2 横截面/板块（A 股特色）

- **板块归一化收益**：成分股 - 所在行业指数
- **同行业/同概念相关股**的 k 日收益均值
- **指数收益**：CSI300、CSI500、创业板、ChiNext 作为外生条件
- **北向资金**（`stock_hsgt_*` 系列）、融资融券（margin）——对大盘股显著

### 2.3 基本面/事件（低频，但对长周期预测关键）

- PE/PB/PS（TTM） `stock_a_indicator_lg`
- 财报季窗口标记（公告前 5 日、财报后 5 日）
- 分红除权日标记、停复牌标记
- 大宗交易占比

### 2.4 日历/时间（已经部分有）

Kronos 已经用了 `[minute, hour, weekday, day, month]`。A 股再加：
- `day_of_month_in_quarter`、`is_quarter_end`
- `is_before_holiday`、`distance_to_holiday`（春节/国庆等长假前流动性异常）
- `intraday_bucket`（开盘/尾盘/午盘切分，用于分钟级）

### 2.5 宏观/情绪（可选增强）

- USD/CNY、10Y 国债收益率、LIBOR、Brent、VIX
- 新闻/研报情绪分（需要 NLP 管道，难度较高）
- Google Trends / 百度指数

配套的 `kairos/data/features.py` 已经实现了 2.1 和 2.4 的大部分、2.2 的指数/板块收益。2.3、2.5 留了接口。

---

## 3. 模型改造：如何让 Kronos 吃进这些新特征

这是最棘手的部分。Kronos 的 pipeline 是：

```
OHLCV(6d) ──Tokenizer──> 离散 token (s1, s2) ──Transformer──> 下一 token
```

Tokenizer 的 `d_in=6` 是**固定**的，你如果直接改它的 `d_in`，就要 **从头重训 Tokenizer 和 Predictor**，官方预训练权重全部作废——对没经验的用户不推荐。

下面给出 3 种方案，按**成本从低到高**排列：

### 3.1 方案 A：保留 Tokenizer，外生变量旁路融合（**强烈推荐入门**）

思路：K 线仍然走 Tokenizer → token 序列；**外生特征走一条并行 Embedding**，在 Transformer 内部用 `add / concat / cross-attention` 融合。

```
 OHLCV ──Tokenizer──> s1,s2 ─┐
                             ├─ + ─► TransformerBlocks ─► heads
 外生因子 X ──Linear──► x_ext ┘
                             ↑
 时间戳 ──TempEmb──► t_emb ───┘
```

**优点**：
- 可以继续加载 NeoQuasar 预训练权重（大部分参数复用）
- 新增参数仅 `Linear(F_ext, d_model)` + 可选一层 FFN，训练快
- 不破坏 Kronos 自回归生成能力

**实现改动（见 `kairos/models/kronos_ext.py`）**：

```python
class KronosWithExogenous(Kronos):
    def __init__(self, *args, n_exog=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.exog_proj = nn.Sequential(
            nn.Linear(n_exog, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.exog_norm = RMSNorm(self.d_model)

    def forward(self, s1_ids, s2_ids, stamp=None, exog=None,
                padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if exog is not None:
            x = x + self.exog_norm(self.exog_proj(exog))
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        ...
```

Dataset 侧同时返回 `(x_kline, x_stamp, x_exog)`；训练时冻结 `embedding + 前 N 层 transformer`，只训 `exog_proj + 最后几层 + head`，叫 **渐进解冻（progressive unfreeze）**。

### 3.2 方案 B：Tokenizer 重训，`d_in` 扩展到 12~20

思路：把 Tokenizer 的 `d_in` 从 6 改成 6 + k 个关键因子（如 turnover/ma20_norm/atr/rsi），**从头预训练**。

**优点**：外生信号直接进入离散 token，信息融合更深。
**缺点**：要跑几十 epoch 大规模预训练，个人预算基本不够（见第 5 节成本）。
**适用**：已经通过方案 A 跑通、想进一步压榨精度、并且有 8xA100 以上。

关键改动：
- `kairos/training/config.py`: `feature_list = ['open','high','low','close','vol','amt','turn','atr','rsi','ret']`
- `model/kronos.py`: 模型实例化时 `d_in=len(feature_list)`
- 重新训练 Tokenizer（从随机或从 `d_in=6` 的 warm-start 再+ linear adapter）
- 重新训练 Predictor

### 3.3 方案 C：把"回归头"挂到 Transformer 侧输出

Kronos 的 `DualHead` 输出的是离散 token 分布。你想要的是**方向/收益的数值**，可以在 transformer 的最后一层 hidden 上**新加一个回归头**：

```python
class ReturnHead(nn.Module):
    def __init__(self, d_model, n_quantiles=9):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_quantiles)   # 输出 9 个分位
        )
    def forward(self, h):
        return self.fc(h)
```

损失函数用 **pinball loss（分位回归）** 或 **Huber loss**，跟原 CE loss 加权合并。这样模型同时学会"像 K 线"和"预测未来 Δclose 的 5% / 50% / 95% 分位"，后者直接用于交易信号。

> 小贴士：回测你会发现 **分位回归 + 方向头** 的稳健性远好于"看形状"的自回归生成。

### 3.4 我给你的路线建议

```
Week 1:  akshare 日线采集 + 因子工程 → 跑通 finetune 原版 (6 维)
Week 2:  方案 A：外生通道 + 渐进解冻微调 Predictor
Week 3:  方案 C：加分位回归头，联合训练
Week 4:  回测调参 + 5min 数据接入 + 部署
Month 2: 若仍不满意，考虑方案 B（重训 Tokenizer）
```

---

## 4. 训练：单机 & 多卡 & 云

### 4.1 最小可运行 (本地/Mac M-series)

```bash
# 只训 Predictor，batch 小一点，只用一块 GPU/MPS
export CUDA_VISIBLE_DEVICES=0
cd finetune && torchrun --standalone --nproc_per_node=1 train_predictor.py
# 或 Mac:
python train_predictor.py  # 需要自己处理 mps device
```

在 M2 Max 64GB 上，Kronos-small（25M）跑 csi300 日线 30 epoch 约 12~18 小时，Kronos-mini（4M）约 2~3 小时。

### 4.2 多卡 (一机 N 卡)

```bash
torchrun --standalone --nproc_per_node=4 -m kairos.training.train_tokenizer
torchrun --standalone --nproc_per_node=4 -m kairos.training.train_predictor
```

显存经验值（bf16）：

| 模型 | context=512, bs=50 | bs=128 |
|---|---|---|
| Kronos-mini (4M) | ~3 GB | ~6 GB |
| Kronos-small (25M) | ~8 GB | ~16 GB |
| Kronos-base (102M) | ~18 GB | ~32 GB |
| Kronos-large (500M) | ~36 GB | OOM on 40G |

### 4.3 关键超参（项目 `config.py` 中可调）

| 参数 | 建议（A 股日线） | 说明 |
|---|---|---|
| lookback_window | 90~128 | 约 4~6 个月的过去 K |
| predict_window | 5~10 | 预测未来 1~2 周 |
| batch_size | 50→128 | 按显存拉满 |
| epochs | 30→50 | 有 early stop 可以 |
| tokenizer_lr | 2e-4 | 已经合理 |
| predictor_lr | 4e-5 → **1e-5** | 外生通道进来要更保守 |
| weight_decay | 0.1 | 防过拟合 |
| clip | 5.0 | 标准化后的截断 |

---

## 5. 云厂商训练成本对比（2026-Q1 公开价，仅供参考）

以"**完整微调 Kronos-small 30 epoch，CSI800 日线 + 5min**"为基准，按经验约 **40 GPU·小时 (A100-40G)**。

| 平台 | GPU | 按量价(¥/h, 税前) | 竞价/抢占价 | 40 小时单次成本 | 备注 |
|---|---|---|---|---|---|
| **AutoDL** | RTX 4090 24G | ¥1.8~2.5 | — | ¥70~100 | 性价比之王；小模型够用 |
| AutoDL | A800 80G | ¥6~8 | — | ¥240~320 | 大模型/长 context 首选 |
| **阿里云 PAI-DSW** | V100-32G | ¥9~12 | ¥3~4（抢占） | ¥360~480 / ¥120~160 | 企业稳定 |
| 阿里云 PAI | A100-80G | ¥28~32 | ¥10 | ¥1120 / ¥400 | 新一代旗舰 |
| **腾讯云** | V100 | ¥10 | — | ¥400 | — |
| 腾讯云 | A10 | ¥6 | — | ¥240 | 性价比不错 |
| **火山引擎** | A100-40G | ¥22 | ¥8 | ¥880 / ¥320 | 抢占便宜 |
| 华为云 ModelArts | Ascend 910 | ¥18 | — | ¥720 | 国产卡，torch 需 MindSpore 适配 |
| **百度千帆** | A800 | ¥20 | — | ¥800 | 赠券活动多 |
| **Google Colab Pro+** | A100 | $10/mo 订阅 | — | ≈¥72/月 | 时段有限，适合实验 |
| Colab Pro | T4/L4 | $10/mo | — | ≈¥72/月 | 很慢 |
| **Lambda Labs** | A100-80G | $1.29/h ≈ ¥9.3 | — | **¥370** | 按秒计费，海外稳 |
| Lambda | H100-80G | $2.49/h ≈ ¥18 | — | ¥720 | 最快 |
| **vast.ai** | RTX 4090 | $0.4~0.7/h | — | ¥120~200 | 海外众包，注意稳定性 |
| **RunPod** | A100-80G | $1.19/h | 抢占 $0.79 | ¥340 / ¥230 | Serverless 选项 |
| AWS p3.2xlarge | V100 | $3.06/h | spot $1 | ¥880 / ¥290 | 老牌，贵 |
| AWS p4d.24xlarge | 8×A100-40G | $32/h | spot $10 | — | 大规模训练 |

**实战建议**：

1. **调参阶段**：AutoDL 4090 / Colab Pro+，单次几十元跑通 pipeline。
2. **正式训练**：AutoDL A800 或 Lambda A100，单次预算 ¥300~500。
3. **大规模预训练**（方案 B，Tokenizer 重训）：8×A100 spot 实例 ~40 小时，¥2~5k。
4. **国内合规**：数据不出境就选阿里/火山/AutoDL，训练产物也留在境内再上传 HF。

> **省钱技巧**：bf16 + `torch.compile()` + `gradient_checkpointing` 可以把 batch 翻倍；训练前先用 Kronos-mini 跑通，再换 small/base；保存 checkpoint 用 safetensors。

---

## 6. 部署：把你的微调模型变成服务

### 6.1 上传到 Hugging Face Hub

```bash
huggingface-cli login        # 输入你的 write token
kairos-push-hf \
    --tokenizer-ckpt ./outputs/models/finetune_tokenizer_demo/checkpoints/best_model \
    --predictor-ckpt ./outputs/models/finetune_predictor_demo/checkpoints/best_model \
    --repo-tokenizer your-username/Kronos-Tokenizer-ashare \
    --repo-predictor your-username/Kronos-small-ashare \
    --private
```

关键点：Kronos 继承了 `PyTorchModelHubMixin`，`save_pretrained()` / `push_to_hub()` 直接可用。

### 6.2 本地/服务器推理服务（FastAPI）

```bash
pip install fastapi uvicorn[standard]
kairos-serve \
    --tokenizer your-username/Kronos-Tokenizer-ashare \
    --predictor your-username/Kronos-small-ashare \
    --device cuda:0 --port 8000
```

调用：

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "600977",
    "lookback": 400,
    "pred_len": 20,
    "T": 0.6, "top_p": 0.9, "sample_count": 5
  }'
```

返回 JSON 里包含未来 20 根 K 的 OHLCV + 分位区间（如果你训了分位头）+ 方向概率。

### 6.3 生产化建议

- **模型热加载**：用 `accelerate` + `safetensors`，冷启动 3s 内。
- **请求合批**：`predict_batch` 已经支持，`BatchScheduler` 30ms 聚一次。
- **指数预测缓存**：CSI300 等大盘指数每 5min 推一次，投机流量可以直接命中。
- **回测回归**：CI 里跑一个固定种子的小回测，RegNet-式监控 IC/ICIR 漂移。
- **风控旁路**：推理完成后做合理性检查（涨跌停内、量非负）。

---

## 7. 推理使用（完整例子）

```python
from model.kronos_ext import KronosWithExogenous
from model import KronosTokenizer, KronosPredictor
from data_pipeline.build_features import build_features
import akshare as ak
import pandas as pd

tokenizer = KronosTokenizer.from_pretrained("your-username/Kronos-Tokenizer-ashare")
model     = KronosWithExogenous.from_pretrained("your-username/Kronos-small-ashare")

predictor = KronosPredictor(model, tokenizer, max_context=512)

# 取 600977 最近 400 个交易日
df = ak.stock_zh_a_hist(symbol="600977", period="daily", adjust="qfq").tail(400)
df = df.rename(columns={
    "日期":"datetime","开盘":"open","收盘":"close","最高":"high",
    "最低":"low","成交量":"volume","成交额":"amount","换手率":"turnover"
})
df['datetime'] = pd.to_datetime(df['datetime'])
df = build_features(df)              # 加入 RSI/ATR/MA 等

x_df = df[["open","high","low","close","volume","amount"]]
x_ts = df["datetime"]
y_ts = pd.bdate_range(df["datetime"].iloc[-1] + pd.Timedelta(days=1), periods=10)

pred = predictor.predict(x_df, x_ts, y_ts, pred_len=10,
                         T=0.6, top_p=0.9, sample_count=5)
print(pred)
```

---

## 8. 常见坑

1. **涨跌停截断**：训练集里不要直接把涨停 `pct_chg=9.98%` 当正常点，模型会被严重带偏。建议打 `is_limit_up/down` 标记并用外生通道喂进去。
2. **复权方式**：后复权会让历史价格变成负数区间，**统一用前复权 `qfq`**。
3. **幸存者偏差**：只拿当前 CSI300 成分会严重偏乐观；应该用**历史成分股列表**（akshare `index_stock_cons_sina` 对应日期）。
4. **分钟数据跳空**：午休 11:30~13:00 没有 K，别用 `pd.date_range` 直接生成 y_timestamp。
5. **正则化**：Kronos 在单只股票窗口内做 z-score，**不要**全市场一起标准化。
6. **信号到交易**：预测的 `close` 不是 alpha。计算 `(pred_close_{t+h} / last_close - 1)` 做 cross-sectional 排序才是正常姿势。

---

## 9. 下一步你应该做什么（今天就能开始）

1. `pip install akshare pyarrow tqdm fastapi uvicorn huggingface_hub` 
2. `kairos-collect --universe csi300 --freq daily --start 2018-01-01 --end 2026-04-17 --out ./raw/daily`
3. `kairos-prepare --raw ./raw/daily --out ./finetune/data/processed_datasets`
4. 在 `kairos/training/config.py` 里把 `pretrained_tokenizer_path/predictor_path` 改成 `NeoQuasar/Kronos-Tokenizer-base` / `NeoQuasar/Kronos-small`
5. `torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor`  （1 卡也可以）
6. 跑 `finetune/qlib_test.py`（或你自己的回测脚本）看 IC
7. 如果 IC > 0.03，继续做方案 A（外生通道）和方案 C（分位头）
8. 上传到 HF，部署 FastAPI

祝炼丹顺利。
