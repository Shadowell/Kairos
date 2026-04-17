# Kairos 术语速查（新手向）

> 这份文档把 Kronos fine-tune 流程里会反复出现的名词**按"我们在这个项目里干了什么"的顺序**讲一遍。每个词都带具体例子和实际数字，看完你应该能理解 v1 / v2 训练报告里的每句话。

目录：

1. [数据层：K 线 / 因子 / 外生变量](#1-数据层)
2. [模型层：Kronos / Tokenizer / Transformer / 分位回归头](#2-模型层)
3. [训练层：监督目标 / 损失函数 / 优化器 / 学习率调度](#3-训练层)
4. [评估层：IC / Rank-IC / ICIR / 命中率](#4-评估层)
5. [常见"病症"：过拟合 / 分布偏移 / 泄漏](#5-常见病症)
6. [工程词：ssh / tmux / nohup / scp / HF mirror / torchrun](#6-工程词)
7. [我们两轮训练用到的超参数逐个解释](#7-我们两轮训练用到的超参数逐个解释)

---

## 1. 数据层

### 1.1 K 线（Candlestick / OHLCV）

A 股每个交易日一根 K 线，含 6 个字段：

| 列名 | 含义 |
|---|---|
| `open` | 开盘价 |
| `high` | 当日最高价 |
| `low`  | 当日最低价 |
| `close`| 收盘价 |
| `volume`（`vol`） | 成交股数 |
| `amount`（`amt`） | 成交金额（元） |

我们 `raw/daily/000001.parquet` 里就是平安银行过去 2009 个交易日（2018-01-02 ~ 2026-04-16）的日线。

> ⚠️ 某些数据源（如新浪、腾讯）返回的 K 线 **没有 amount 列**。我们在 `prepare_dataset.py` 里做了兜底：`amount ≈ close × volume`，否则下游 `dropna` 会把整表清空。

### 1.2 因子（Factor） / 外生变量（Exogenous Feature）

除了原始 K 线外，我们还人工计算了一堆"特征"喂给模型，例如：

- **5/10/20/60 日均线**（Moving Average）
- **RSI**（相对强弱指数，衡量超买超卖）
- **MACD**（快慢均线差）
- **波动率**（最近 N 日收益率标准差）
- **相对 CSI300 的超额收益**（个股 vs 沪深 300 的差值）
- …

这些**派生出来的特征**在模型里叫"外生变量"（Exogenous Feature，代码里是 `exog`），共 **32 列**，定义在 `kairos/data/features.py` 的 `EXOG_COLS`。

> "外生"这个词来自计量经济学，指的是"来自模型外部、模型不负责预测它本身、只把它当输入"的变量。跟它相对的是"内生变量"（模型自己要预测的那个 y）。

### 1.3 Universe（股票池）

训练用哪些股票的集合。我们用的是 **CSI 300**（沪深 300 成分股），300 只大盘股。代码中 `get_universe("csi300")` 返回这 300 个股票代码。

### 1.4 Parquet

一种列式存储的二进制表格格式，比 CSV 小很多、读写快。`raw/daily/000001.parquet` 就是一个。用 `pd.read_parquet()` 读取。

### 1.5 Pickle（.pkl）

Python 的序列化格式，可以把任意 Python 对象存成二进制文件。我们的 `train_data.pkl` 存的是 `Dict[symbol_str, pd.DataFrame]`：每只股票对应一段 K 线 DataFrame。

### 1.6 Train / Val / Test 集（三分）

机器学习标准切分：

- **Train（训练集）**：模型"看得到"，用来更新权重
- **Val（验证集 / Validation）**：模型"看不到、但我们能看到"，每个 epoch 结束后拿它测一次，用来决定什么时候停、选哪个 checkpoint
- **Test（测试集）**：**完全隔离**。只在一切都训完之后用它报告最终效果。**不能**根据 test 结果回头再调参，否则就是"用 test 调参"= 作弊。

我们两轮的切法：

| 版本 | Train | Val | Test |
|---|---|---|---|
| v1 | 2018-2023 | **2024 年（单年）** | 2025-至今 |
| v2 | 2018-2024 的 85% 块 | 2018-2024 的 15% 块（跨年随机抽） | 2025-至今 |

v1 的 val 集中在 2024 单一年，**风格偏差大**（见 [分布偏移](#52-分布偏移-distribution-shift)），这也是 v1 失败的主要原因之一。

### 1.7 Block-level Interleaved Split（块级交错切分）

v2 用的切法：把 2018-2024 的所有交易日先切成连续的 **20 日 block**（约一个月），再从所有 block 里随机抽 15% 当 val，其余当 train。

```
                ┌──────── 2018 ────────┐  ┌──── 2024 ────┐
 blocks: [B001][B002][B003][B004][B005]...[B079][B080][B081]
 assign:   T    V    T    T    V  ...  T    T    V       ← 随机
```

这样 val 涵盖各个年份、各种行情风格，不会像 v1 那样只看到 2024 一年。

为什么不直接"每日随机 15%"？因为日度随机抽会让 val 和 train 相邻日高度相关（昨日在 train、今日在 val），等于变相**数据泄漏**。按 block 抽至少保证 val 和 train 之间有 20 天的"隔离区"。

---

## 2. 模型层

### 2.1 Kronos（我们 fine-tune 的基座模型）

NeoQuasar 开源的一个金融时序预训练模型，结构上是：

```
   K 线 (T×6) ──► Tokenizer ──► token id 序列 ──► Transformer ──► 预测下一个 token
```

特点：它把连续的 K 线**离散化**成 token（类似把图片切成 patch），然后像语言模型一样预测下一个 token。我们用的是 **Kronos-small**（~25M 参数）。

HuggingFace 仓库：`NeoQuasar/Kronos-small` / `NeoQuasar/Kronos-Tokenizer-base`

### 2.2 Tokenizer（分词器 / 离散化器）

在 NLP 里，tokenizer 把"我爱你"切成 `[我, 爱, 你]` 三个 token id。
在 Kronos 里，tokenizer 把每根 K 线的 6 维连续向量 `[open, high, low, close, vol, amt]` 映射成**两个整数 token**（`s1_id`, `s2_id`）——所以叫 **Hierarchical**（两级）tokenization。

我们训练时 **tokenizer 权重是冻结的**，只用它预处理；真正在学习的是 Transformer 部分。

### 2.3 Transformer

现代大模型的核心结构。细节不展开，只要理解：

- 由多个 **Transformer block（层）** 堆叠组成
- 每一层主要是 **self-attention + feedforward**
- Kronos-small 是 **8 层**
- 越深的层越接近输出、学到的特征越"任务相关"
- 越浅的层越接近输入、学到的特征越"通用"

### 2.4 Embedding（嵌入）

把离散的 token id 转换成连续的向量。例如 token id `37` → `[0.12, -0.4, 0.8, ...]`（`d_model` 维）。Kronos 里 `d_model=256`。

### 2.5 冻结 / 解冻（Freeze / Unfreeze）

**冻结某一层** = 这一层的权重在训练时**不更新**（`requires_grad = False`）。

为什么要冻结？基座模型是拿几十亿条数据预训练的，通用知识都存在浅层。我们只有 300 股 × 7 年的数据，如果把整个模型全都放开训，很容易把这些通用知识"冲掉"（灾难性遗忘）。

所以做法是：**冻结浅层，只解冻最后几层 + 我们新加的头部**。这就是 `unfreeze_last_n` 的含义。

v1: `unfreeze_last_n=2` → 解冻最后 2 层 transformer + exog encoder + return head
v2: `unfreeze_last_n=1` → 解冻最后 1 层 + 新头部（更保守）

### 2.6 Exogenous Encoder（外生编码器）

Kronos 原本只吃 6 维 OHLCV。我们在 `kairos/models/kronos_ext.py` 里加了一条**旁路通道**：把前面 1.2 讲的 32 维 exog 特征，先过一个 `Linear → SiLU → Linear` 投影成和主通道一样的 256 维，再**加到** token embedding 上。

```
  OHLCV token ──► embedding (256-d) ──┐
                                      +──► Transformer
  Exog (32-d) ──► linear proj (256-d)─┘
```

这么做的好处是：保留了 Kronos 原始预训练权重的可迁移性，又把我们的额外特征塞了进去。

门控 `gate` 被零初始化，所以训练开始时 exog 的贡献为 0，等同于原版 Kronos；训练过程中模型自己决定要不要激活这个旁路。

### 2.7 Return Head（收益回归头 / 分位回归头）

我们在 Transformer 的最后一层 hidden state 上加的一个**额外输出头**。它做的事：

- 给定当前时刻 t 的隐状态 → 预测未来 **h=5 天** 的**收益分位数**（9 个分位：0.1, 0.2, ..., 0.9）

一般的回归只输出"中位数"一个点。分位回归多预测 9 个分位 → 直接给出**预测的置信区间**。训练用的损失是 **pinball loss**（分位损失）。

backtest 时我们只取中位数（第 5 个分位 = 0.5）当作"下一步收益预测值"，再跟真实收益算 IC。

---

## 3. 训练层

### 3.1 Epoch / Step / Batch

- **Batch**（批）：一次前向 + 反向传播用到的一组样本。`batch_size=50` 意味着一次喂 50 个窗口给模型。
- **Step**（步）：做完一次"前向 + 反向 + 参数更新"叫一步。
- **Epoch**（轮）：把训练集"过一遍"叫一个 epoch。

我们的 `n_train_iter=50000` 意思是：**每个 epoch 从训练池里随机抽 50000 个样本**（而不是跑完整个池子，训练池实际有 36 万个样本）。所以每个 epoch = 50000 / 50 = **1000 step**。

### 3.2 样本 / 窗口（Sample / Window）

对于时序预测，一个"样本"是一个**滑动窗口**：

```
某只股票的日线 ───────────────────────────►
        ↑                   ↑
    [lookback=90 天]    [predict=10 天]
        └──── 一个样本 ────┘
    这 100 天一起输入模型，模型要预测未来 token
```

所以 36 万样本 ≠ 36 万个日期，而是"300 股 × 每股 ~1200 个有效滑动起点"。

### 3.3 Teacher Forcing（教师强制）

训练语言模型时的标准做法。模型逐个 token 预测：

- 预测第 t+1 个 token 时，**输入给模型的上下文用"真实的" t 之前的 token**，而不是模型自己前一步预测出来的 token
- 这样训练更稳定、收敛更快
- 推理时用不了 teacher forcing（因为真值未知），只能用自己的预测接下去，所以存在训练-推理分布差

### 3.4 损失函数（Loss）

衡量"模型预测与真值差多少"的标量。训练目标 = 让 loss 越来越小。

我们的 loss 是**两部分加权和**：

```
total_loss = ce_weight * CE_loss  +  quantile_weight * Pinball_loss
```

#### 3.4.1 CE Loss（Cross-Entropy，交叉熵）

用于"模型吐出 token id 的预测正确率"：给 `vocab_size` 个选项，模型要说对下一个 token 是哪个。

v1 里看到的 `val_ce=4.57` / v2 的 `val_ce=2.95` 都是这个值。

直观理解：
- `ce = ln(vocab_size)` 表示完全瞎猜（均匀分布）
- Kronos 的 s1 vocab_size = 1024，瞎猜 = ln(1024) ≈ **6.93**
- 我们 v2 的 2.95 意味着模型"大约把可能选项缩小到了 $e^{2.95} ≈ 19$ 个"

#### 3.4.2 Pinball Loss（分位损失）

用于分位回归头。对每个分位 q ∈ [0.1, 0.9]，惩罚"预测比真值小"和"预测比真值大"的代价不同：

```
loss_q = max(q·(y - ŷ), (q-1)·(y - ŷ))
```

所有 9 个分位的 loss 求和就是总 pinball。

### 3.5 优化器（Optimizer）

怎么**拿 loss 的梯度去更新权重**的算法。我们用 **AdamW**——Adam 的带权重衰减版本，是现在最通用的选择。

关键参数：

- **learning rate（学习率 `lr`）**：每一步参数更新的"步长"。太大 → 不稳定震荡；太小 → 学得慢。
- **weight decay**：正则化项，每步把权重往 0 拉一点，防止过拟合。
- **betas (β1, β2)**：动量参数，一般用默认 (0.9, 0.95)。

我们 v1 用 `lr=4e-5`，v2 降到 `lr=5e-6`（降 8 倍），因为 v1 看到 val_ce 从 ep1 就爬升，说明新初始化的头部梯度太大被打飞了。

### 3.6 Learning Rate Schedule（学习率调度）

训练过程中不保持 lr 恒定，而是**按规则变化**。我们用 **OneCycleLR**：

```
lr ──┐   ←── warmup：从 lr/10 升到 max_lr
     │ ╱╲
     │╱  ╲______  ←── cosine：从 max_lr 慢慢降到接近 0
     0──────────► step
        ↑
    pct_start (warmup 占总训练的比例)
```

- **warmup（热身）**：一开始不直接用最大 lr，从 `lr/10` 慢慢升上去。原因是：新初始化的层（exog、return_head）权重随机，直接用大 lr 容易飞。
- **pct_start**：warmup 占比。v1 = 3%（太短，新头部还没稳就开始 cosine 下降），v2 = 10%（warmup 更温和）。

### 3.7 早停（Early Stopping）

训练过程中，如果 **val loss 连续 N 个 epoch 不降**，就提前停掉，不管还有多少 epoch 没跑。

- `patience`：容忍几个 epoch 不降才停。我们 v2 用 `patience=3`。
- 好处：省算力；避免继续跑只会越训越差（过拟合）。
- 训练完保留的是"历史上 val loss 最低那一个 epoch 的 checkpoint"，不是最后一个。

v2 实际发生：
```
ep1 val=2.97 [save]
ep2 val=2.95 [save best]  ←── 保存的就是这个
ep3 val=3.02 [patience 1/3]
ep4 val=3.31 [patience 2/3]
ep5 val=3.72 [patience 3/3 → 停]
```

### 3.8 Checkpoint（检查点 / ckpt）

模型权重的文件快照。我们保存的是 HuggingFace 格式：

```
best_model/
├─ config.json        ← 模型架构描述
├─ model.safetensors  ← 权重（97 MB）
└─ README.md
```

可以直接用 `KronosWithExogenous.from_pretrained("best_model/")` 加载。

### 3.9 DDP（DistributedDataParallel，分布式数据并行）

多卡训练时让每张 GPU 各跑一部分 batch，然后同步梯度的机制。单卡训练用不上，但代码里还是用了 `torchrun --nproc_per_node=1` 这个框架——只是 world_size=1 退化成单卡。

背后有个叫 `nccl`（NVIDIA Collective Communications Library）的通信库，CPU 训练时会回退到 `gloo`。

### 3.10 DataLoader / Sampler

`DataLoader` 负责从 Dataset 里批量取样 → 拼成 batch → 喂给 GPU。`DistributedSampler` 是 DDP 场景下让每张卡只看数据子集的 sampler。

### 3.11 Random Seed（随机种子）

让"随机过程可复现"的数字。我们在 `kairos-prepare --seed 42` 里用来固定 block 抽样，保证你重跑一次还能得到同一份切分。

---

## 4. 评估层

回测报告里所有指标只有这 4 个，都要读懂。

### 4.1 IC（Information Coefficient，信息系数）

**IC = Pearson 相关系数(模型预测分数, 真实未来收益率)**

```
IC = cov(score, ret) / (std(score) * std(ret))
```

范围 [-1, 1]：
- `IC = +1`：预测越高、未来收益越大（完美正相关）
- `IC = 0`：无关
- `IC = -1`：完全反向（反过来用也能赚钱）

量化行业经验值：
- `|IC| < 0.02`：几乎无效
- `|IC| = 0.03 ~ 0.05`：有信号但弱
- `|IC| > 0.05`：已经不错
- `|IC| > 0.1`：行业顶级（多数是回测过拟合）

我们 v2 的 `h1 IC = -0.020`，基本在"无效 + 略微反向"这个层次。

### 4.2 Rank-IC（秩 IC / Spearman Correlation）

**Rank-IC = Spearman 相关系数(score, ret)** = 把两列都换成"名次"后的 Pearson。

为什么要这个？因为实盘选股关心的是**"谁高谁低"的排序**（选 Top K 股票做多），而不是预测值绝对大小。Rank-IC 对异常值更鲁棒。

一般 `|Rank-IC| > |IC|` 说明模型能提供排序信号但绝对值预测不准。反之说明模型被异常值带偏了。

### 4.3 ICIR（IC 的信息比率）

把每天 cross-sectional IC 算一遍（300 只股票在这一天的预测 vs 真值），216 个交易日就得到 216 个日度 IC。然后：

```
ICIR = mean(daily_IC) / std(daily_IC)
```

含义："IC 的稳定性"。IC 为正但波动巨大 → ICIR 低，说明不可靠。实盘策略更看 ICIR 而不是 IC：

- `ICIR > 0.3`：相当稳定的因子
- `ICIR > 0.5`：很好
- 我们 v2: `ICIR = -0.13`：既负又不稳定

### 4.4 命中率（Hit Rate / Directional Accuracy）

**模型预测涨 / 跌的方向跟真实一致的比例**。随机瞎猜是 50%。

- `55%`：小有帮助
- `60%+`：很难做到（考虑到真实市场噪声）

我们两轮都在 49-50%，等同瞎猜。

### 4.5 Baseline（基线）

**"什么都不做"的对照参考**。我们做的 baseline 对比：

- `未 fine-tune 的 Kronos-small`（heads 随机初始化）直接拿去预测 test 集 → `h1 IC = -0.007`
- `v1 fine-tune` → `h1 IC = -0.021`（比 baseline 还差）
- `v2 fine-tune` → `h1 IC = -0.020`（也比 baseline 差）

baseline 是绝不能跳过的对照！**有了 baseline 才能知道你的训练是"有用"还是"反而变差"**。

---

## 5. 常见病症

### 5.1 过拟合（Overfitting）

模型把训练集记住了（包括噪声），但验证/测试集表现很差。典型症状：

```
train_loss  ↓  ↓  ↓  ↓  ↓
val_loss    ↑  ↑  ↑  ↑  ↑   ← 这就是过拟合
```

应对：
- 早停
- 降学习率
- 加正则（dropout / weight decay）
- 减容量（比如我们 `unfreeze_last_n` 从 2 降到 1）
- 加数据

### 5.2 分布偏移（Distribution Shift）

训练数据和 val/test 数据的**统计分布不一样**。

v1 里最典型的体现：train 覆盖 2018-2023（含股灾、贸易战、疫情、牛市切换），val 单独用 2024 年（AI + 高股息那波独立行情），**2024 相对前五年有结构性差异**，模型在 train 上学到的规律在 val 上不适用，val_loss 从 ep1 就开始爬。

v2 的 interleave 切分就是为了打破这个偏移。

### 5.3 数据泄漏（Data Leakage）

Train 集"偷看"到了本不应该看到的未来信息，导致训练时效果假好。常见形式：

- 用了未来数据计算的特征（如用整个历史 std 归一化，泄漏了 test 期的信息）
- Train/Val 切分不当（比如按样本随机切，相邻日落入不同集合）
- 指数数据（CSI300）更新滞后于个股等

v2 里我们的滑窗归一化是**只用窗口内 lookback 部分计算 mu/sd**，没用未来信息，所以这里没泄漏。

### 5.4 灾难性遗忘（Catastrophic Forgetting）

在预训练模型上 fine-tune 太激进（lr 太大、epoch 太多、放开太多层），会把**预训练学到的通用知识"冲掉"**。表现是 fine-tune 后的模型比 baseline 还差。

v1/v2 都有这个嫌疑——v1 fine-tune 后 IC 比 baseline 更差 0.014，就是典型的灾难性遗忘。

---

## 6. 工程词

### 6.1 SSH / SCP

- **ssh**：安全远程登录别人机器。`ssh -p 30083 root@connect.westd.seetacloud.com` 就是连 AutoDL。
- **scp**：基于 ssh 的文件传输。`scp local.py root@host:/remote/path/` 把本地文件推到远端。

### 6.2 tmux / nohup

两种在 ssh 断开后**让程序继续跑**的方式：

- **tmux**：虚拟终端，可以反复 attach/detach。推荐，但 AutoDL 镜像里没装（需要 `apt install tmux`）。
- **nohup … &**：把进程脱离当前 shell，输出重定向到文件。穷人版。我们本次用的就是 `nohup bash run_train.sh > train.out 2>&1 &`。

### 6.3 HuggingFace / HF mirror

**HuggingFace**：全球最大的开源模型托管平台（`huggingface.co`）。Kronos 模型就放在上面。

**HF mirror**：中国大陆访问 HF 慢，`hf-mirror.com` 是国内镜像。通过环境变量切换：

```
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache   # 模型缓存目录
```

### 6.4 代理冲突（Proxy Conflict）

AutoDL 的"学术加速器"通过 `http_proxy` 环境变量把所有流量走国际代理，结果把**走 hf-mirror 的请求也给加速（= 绕远路到国外服务器）**，导致访问国内镜像反而变慢到超时。

解决办法：`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY` 在跑训练前清掉。我们 `run_train.sh` 里就有这行。

### 6.5 torchrun

PyTorch 官方的训练启动器，用来拉起 DDP。写法：

```
torchrun --standalone --nproc_per_node=1 kairos/training/train_predictor.py
```

- `--standalone`：单机模式
- `--nproc_per_node=1`：启动 1 个 worker 进程（单卡）
- 多卡时改成 `--nproc_per_node=4`

### 6.6 venv / pip -e

- **venv**（virtual environment）：Python 隔离环境。`python -m venv .venv` 在当前目录建一个。所有 pip install 只影响这个环境，不污染系统 Python。
- **pip install -e .**：editable install。把当前项目以"可编辑"方式装到 venv 里，这样你改了源码立即生效，不用每次 `pip install` 重装。

---

## 7. 我们两轮训练用到的超参数逐个解释

对应 `kairos/training/config.py` 里的 `TrainConfig`。

```python
# ─── Data ───
dataset_path = "./finetune/data/processed_datasets"  # 数据集 pkl 目录
lookback_window = 90        # 用过去 90 天预测
predict_window = 10         # 预测未来 10 天
max_context = 512           # Transformer 的最大上下文长度
n_exog = 32                 # 外生特征维度

# ─── Training ───
seed = 100                  # 随机种子
clip = 5.0                  # 归一化后的数值裁剪范围 [-5, 5]
epochs = 15                 # 总 epoch 上限（被 patience 早停）
batch_size = 50             # 每步喂 50 个样本
n_train_iter = 50000        # 每个 epoch 从训练池抽 50000 样本 = 1000 step
n_val_iter = 10000          # 每个 epoch val 用 10000 样本 = 200 step
log_interval = 100          # 每 100 step 打印一次训练 loss

predictor_learning_rate = 5e-6   # 预测器 lr（v2 降到 v1 的 1/8）
warmup_pct = 0.10                # warmup 占总训练 10%
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_weight_decay = 0.05    # L2 正则
num_workers = 2             # DataLoader 预取线程
patience = 3                # val 连续 3 个 epoch 不降就停

# ─── Loss ───
ce_weight = 0.5             # CE loss 权重（v2 降到 0.5）
quantile_weight = 2.0       # pinball loss 权重（v2 升到 2.0，主导梯度）

# ─── Model ───
use_exog = True             # 启用外生通道
use_return_head = True      # 启用分位回归头
return_horizon = 5          # 回归头预测未来 5 天
n_quantiles = 9             # 预测 9 个分位
unfreeze_last_n = 1         # 只解冻最后 1 层 transformer

# ─── Pretrained ───
pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor_path = "NeoQuasar/Kronos-small"
```

---

## 附录：v1 vs v2 对比一览

### 先说清楚一件事：v1 和 v2 跑的是同一个模型结构

两轮训练**都启用了方案 A（外生通道）+ 方案 C（分位回归头）**——代码层面 `use_exog=True` 和 `use_return_head=True` 两个开关在 v1 和 v2 里都没动过，`KronosWithExogenous` 结构一模一样。

训练日志里都能看到同一行印记：

```
[KronosWithExogenous] 载入 136/147 层；新初始化 11 层（exog / return_head）
```

"136 层来自官方 Kronos-small 权重（方案 A 和 C 共用的基座），11 层是我们新加的（方案 A 的 `exog_encoder.*` + 方案 C 的 `return_head.*`），随机初始化等着被训练"。

> **v1 → v2 只改了"训练设置"（学习率、损失权重、切分、早停），没碰结构。**
> 所以 v1 失败 ≠ 方案 A/C 失败。v2 把训练治稳了但 IC 依然负的，问题出在监督信号本身（next-token CE + 差分收益 pinball），跟 2025 年的收益方向没有可迁移关系。

### 超参数对比表

| 维度 | v1（失败版） | v2（改进版） | 差别 |
|---|---|---|---|
| **结构** | | | |
| 方案 A（外生通道） | ✅ 开 | ✅ 开 | **未变** |
| 方案 C（分位回归头） | ✅ 开 | ✅ 开 | **未变** |
| **数据** | | | |
| val 切分 | 2024 单年 | interleave 块级抽 | 消除单年偏差 |
| **损失权重**（方案 C 贡献） | | | |
| `ce_weight` | 1.0 | 0.5 | 降一半 |
| `quantile_weight` | 0.5 | 2.0 | **升 4×**（让方案 C 梯度主导） |
| **优化器** | | | |
| 预测器 lr | 4e-5 | 5e-6 | **降 8×** |
| warmup_pct | 3% | 10% | 更温和 |
| weight_decay | 0.1 | 0.05 | 小 lr 配小 wd |
| **容量控制** | | | |
| `unfreeze_last_n` | 2 | 1 | 更保守 |
| **训练调度** | | | |
| epochs | 30 | 15 + `patience=3` | 加早停 |
| **产出** | | | |
| val_ce (best) | **4.57** @ ep1 | **2.95** @ ep2 | ⬇ 35% |
| test h1 IC | **-0.021** | **-0.020** | 几乎没变 |
| test h1 Rank-IC | -0.002 | -0.005 | 几乎没变 |
| **结论** | val 立即过拟合 | val 正常，但 test 无信号 | 根因是监督目标 |

### 要怎么读懂这张表

- **结构不变，改的全是训练设置** → v2 证明了"方案 A + C 的架构本身跑得稳"
- **val_ce 从 4.57 → 2.95** → 过拟合确实被治好了（这是工程面的胜利）
- **test IC 仍是负的** → 但对 2025 年实际收益没帮助，所以**不是调参能解决的，得从数据或监督目标层面换招**

### 下一轮该怎么改（从根上动）

1. **换监督目标**（最便宜）：砍掉 CE loss，让方案 C 的分位头**独占**梯度，目标直接改成 h=1 的对数收益率（而非"normalized close 的差分"）。这样训练信号与 backtest 指标直接对齐。
2. **换 baseline 对照**（最该先做）：先在本地跑一个 LightGBM + 32 维 exog → h1 收益率的回归，看 IC 能不能转正。**如果 LightGBM 都跑不出信号，说明是数据问题，就该去搞数据；如果 LightGBM 跑得出，说明是 Kronos fine-tune 方式的问题，才值得继续调模型。**
3. **扩数据**（最贵）：300 股 → 全 A 股 5000+；7 年 → 15 年（2010-）；加分钟线。

优先级：**先做 2 定位病因**，再决定走 1 还是 3。

---

## 多市场架构术语（Phase 2 新增）

| 名词 | 一句话解释 | 代码入口 |
|---|---|---|
| `MarketAdapter` | 每个市场（A 股、加密、外汇...）都有一个 adapter，负责 `list_symbols` / `fetch_ohlcv` / `trading_calendar` / `market_features` | `kairos/data/markets/base.py` |
| `CryptoExchange` | 在 crypto adapter 下面再分一层，具体的交易所（OKX、Binance...）实现，通过 ccxt 统一接入 | `kairos/data/markets/crypto_exchanges/base.py` |
| `COMMON_EXOG_COLS` | 跨市场通用的 24 维因子（收益、RSI、MACD、波动、均线、布林、量价、微结构、pad） | `kairos/data/common_features.py` |
| `MARKET_EXOG_COLS` | 各 adapter 贡献的 8 维市场特征；A 股是换手/日历/超额收益，crypto 是 funding/OI/basis/dominance/小时三角 + pad | 各 adapter 的类属性 |
| `FeatureContext` | 特征计算时的 side channel——A 股用它传指数 K 线，crypto 用它传 funding / OI / spot 价；adapter 自取所需 | `kairos/data/markets/base.py` |
| `meta.json` | `kairos-prepare` 在输出目录里落盘的数据集清单（market / freq / exog_cols / ranges），训练和回测侧自动读取 | `kairos/data/prepare_dataset.py` |
| `preset_for("crypto-1min")` | 返回一组针对 1min crypto 调好的 `TrainConfig` 超参（`lookback=256`, `horizon=30`, `ce_weight=0.7`...）的字典 | `kairos/training/config.py` |
| `aggregation`（回测） | cross-sectional IC 按什么粒度聚合：`date` / `hour` / `minute` / `none` | `kairos/training/backtest_ic.py` |

**关键不变式**：
- `len(COMMON_EXOG_COLS) + len(adapter.MARKET_EXOG_COLS) == 32`
  任何新 adapter 都必须保证这一点，否则 `build_features` 直接抛错。
- 模型的 `n_exog=32` 不变，所以老的 A 股 checkpoint 能被 crypto 数据直接加载做 ablation（输入 shape 对得上，只是语义不同）。

---

## 延伸阅读（项目内）

- [AUTODL_GUIDE.md](AUTODL_GUIDE.md) — AutoDL 上整套部署流程
- [TUNING_PLAYBOOK.md](TUNING_PLAYBOOK.md) — 调参实战手册
- [CRYPTO_GUIDE.md](CRYPTO_GUIDE.md) — 加密市场接入与训练指引
- [../artifacts/autodl_run_v2_20260417/train_summary.md](../artifacts/autodl_run_v2_20260417/train_summary.md) — v2 本次报告
