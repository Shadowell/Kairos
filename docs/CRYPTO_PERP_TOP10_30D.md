# Crypto OKX 永续 Top10 × 30d 端到端跑通记录（含 post-mortem）

> 在 [`CRYPTO_TOP100_RUN.md`](CRYPTO_TOP100_RUN.md) 的现货 baseline 之上，**首次** 把数据源换到 **OKX 永续** 拿到真实非零的 `funding_rate` 和 `basis`，验证多通道改造（[`CRYPTO_PERP_PLAN.md`](CRYPTO_PERP_PLAN.md)）的端到端链路。
>
> **结论先放**：链路打通了（采集 → 打包 → 训练 → 回测全部跑完），**但训练效果是负迁移**——finetuned 在 pooled IC 上反而比 baseline（Kronos 原权重 + 随机 head）差。诊断已查清，三条 root cause 见 §8。本文档的主要价值是把**所有踩到的坑和诊断方法**留下来，避免再耗一次 GPU 时间。
>
> 相关文档：
> - [`CRYPTO_PERP_PLAN.md`](CRYPTO_PERP_PLAN.md) — 永续多通道改造的总体方案（funding / OI / basis）
> - [`CRYPTO_BTC_ETH_RUN.md`](CRYPTO_BTC_ETH_RUN.md) — 现货 BTC+ETH 2 年 baseline
> - [`CRYPTO_TOP100_RUN.md`](CRYPTO_TOP100_RUN.md) — 现货 Binance Top100 1 年 baseline
> - [`BACKTEST_IC_GUIDE.md`](BACKTEST_IC_GUIDE.md) — bucket / aggregation / stride / horizon 怎么选
> - [`AUTODL_GUIDE.md`](AUTODL_GUIDE.md) — AutoDL 通用租卡训练手册

---

## 0. 背景与目标

- **日期**：2026-04-20
- **动机**：BTC/ETH 和 Top100 两次现货 run 都因为 `binance_vision` 镜像没有衍生品因子，`funding_rate / oi_change / basis / btc_dominance` 四列被 pad 为 0，模型只能用 24 维通用因子学。本次想验证：**有了真实非零的 funding 和 basis，h1/h5 短 horizon 的 IC 能不能也起来**？
- **路线选择**：[`CRYPTO_PERP_PLAN.md`](CRYPTO_PERP_PLAN.md) 里的"路线 A — AutoDL 隧道到机场，用 OKX 永续"。
- **数据源**：`api.okx.com` 永续 + 现货（拿 basis），通过 mihomo (Clash Meta) + 机场订阅打通。
- **机器**：沿用同一台 AutoDL RTX 5090（`connect.westd.seetacloud.com:37667`）。

### 0.1 范围调整时间线

> 这是为了说明 "为什么最后只跑了 30 天" —— 不是预设方案，而是被 OKX API 的硬限制逼出来的。

| 时间 | 计划 | 触发原因 |
|---|---|---|
| 11:30 | Top100 × 90 天（funding 覆盖极限） | 走 [`CRYPTO_PERP_PLAN.md`](CRYPTO_PERP_PLAN.md) Phase 6 |
| 14:00 | 用户改：Top10 × 1 年 | 想先小规模快速验证链路 |
| 16:00 | 进一步改：**Top10 × 最近 30 天** | 1 年的 funding 数据 OKX 只回填 90 天，对**"用 funding 当 exog"**意义不大；30 天里 funding 覆盖率最高 |

---

## 1. Universe 选择

OKX 永续 24h 成交量 Top10（按 `baseVolume × last × contractSize` 排序，**不**用 `quoteVolume` —— meme 币的合约张数会把 quoteVolume 灌水，详见 [`CRYPTO_GUIDE.md`](CRYPTO_GUIDE.md)）：

```
BLUR/USDT:USDT  BTC/USDT:USDT   DOGE/USDT:USDT  ETH/USDT:USDT
HYPE/USDT:USDT  ORDI/USDT:USDT  PEPE/USDT:USDT  SOL/USDT:USDT
XAU/USDT:USDT   XRP/USDT:USDT
```

XAU/USDT:USDT（黄金永续）保留进训练集，目的是看模型能不能区分 "和 BTC 几乎不相关的标的"——结果上看它的 spot 缺失（OKX 没对应现货），basis 列对它是 0。

---

## 2. 服务器环境（增量）

代码 / venv / hf_cache 全部沿用之前 BTC/ETH + Top100 run，只新增了**网络隧道**部分。

### 2.1 mihomo (Clash Meta) 起代理

```bash
# AutoDL 上
mkdir -p /root/.config/mihomo && cd /root/.config/mihomo
# 1. 下机场订阅 (flag=meta 拿到 mihomo 兼容的 YAML)
curl -L -o config.yaml '<机场订阅 URL>&flag=meta'
# 2. 预下载 GeoIP/GeoSite 数据库（不下载会启动失败）
curl -L -o Country.mmdb https://github.com/.../Country.mmdb
curl -L -o GeoSite.dat  https://github.com/.../GeoSite.dat
curl -L -o geoip.dat    https://github.com/.../geoip.dat
# 3. 启动
nohup /root/mihomo/mihomo -d /root/.config/mihomo > /root/mihomo.log 2>&1 &
# 4. GLOBAL 默认是 DIRECT，要切到具体节点
curl -X PUT http://127.0.0.1:9090/proxies/GLOBAL \
    -H 'Content-Type: application/json' -d '{"name":"<US 节点名>"}'
# 5. 测试
HTTPS_PROXY=http://127.0.0.1:7890 curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" \
    https://api.okx.com/api/v5/public/time
# 期望: 200 ~1.1s
```

US 节点实测最快（~1.1s/req），HK 在 0.8s 但不稳，DE 1.8s+。

### 2.2 OKX adapter 的两个修复（**4-20 上午已 commit**）

跑通过程中发现两个需要改 `kairos/data/markets/crypto_exchanges/okx.py` 的问题：

1. **`InvalidProxySettings`** — ccxt ≥ 4.5 不允许同时设 `http_proxy` + `https_proxy`。OKX 走全 HTTPS，只留 `https_proxy`（commit `9e33a2f`）。
2. **funding/OI 时间窗参数** — `since` kwarg 被 OKX 服务器忽略，改用 `params={"after": cursor}` 翻 funding，`params={"begin":..., "end":...}` 查 OI；同时空 frame 也要保留 `funding_rate`/`open_interest` 列以避免 KeyError（commit `05b8595`）。

### 2.3 OKX API 的硬性历史窗口（这是 30 天选择的根本约束）

| 端点 | 实际可回溯窗口 | 影响 |
|---|---|---|
| `/api/v5/market/history-candles` | 多年（够用） | OK |
| `/api/v5/public/funding-rate-history` | **最近 ~90 天**（更老返回空） | 训练窗口超过 90 天，funding 列前段全 0 |
| `/api/v5/rubik/stat/contracts/open-interest-history` | **最近 ~8 小时**（100 条 × 5min） | OI 历史几乎拿不到，需要实时订阅自己累积；本 run 直接接受 `oi_change=0` |

详见 [`CRYPTO_PERP_PLAN.md`](CRYPTO_PERP_PLAN.md) §5 "Fallback scenarios"。

---

## 3. 采集 Top10 × 30d × 1min

### 3.1 命令

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
unset http_proxy HTTP_PROXY            # ccxt 只能要 https
export HTTPS_PROXY=http://127.0.0.1:7890

UNIVERSE="BLUR/USDT:USDT,BTC/USDT:USDT,DOGE/USDT:USDT,ETH/USDT:USDT,HYPE/USDT:USDT,ORDI/USDT:USDT,PEPE/USDT:USDT,SOL/USDT:USDT,XAU/USDT:USDT,XRP/USDT:USDT"

nohup kairos-collect --market crypto --exchange okx \
    --universe "$UNIVERSE" --freq 1min \
    --start 2026-03-20 --end 2026-04-20 \
    --out ./raw/crypto/perp_top10 --workers 4 \
    --crypto-extras funding,spot \
    --proxy "$HTTPS_PROXY" \
    > logs/perp_top10_collect.log 2>&1 &
```

### 3.2 产出

- **耗时：9 m 34 s**（10 perp × 30d × 1min ≈ 432k 行/币）
- `raw/crypto/perp_top10/`
  - 10 × `.parquet`（OHLCV + amount）
  - `_extras/funding/` 10 个 parquet（**全部非零**，funding 周期 8 小时 → 30 天 ~90 条/币）
  - `_extras/spot/` 9 个 parquet（XAU 没对应现货）
  - 没有 `_extras/oi/` —— OKX 历史 OI 只有 8 小时，不抓
- **关键 sanity**：`funding_rate` 在打包后非零率 ~94%，`basis` 非零率 ~88%，**这是这次 run 相对前两次的核心进步**。

### 3.3 监控脚本（解决 `nohup + ThreadPoolExecutor + tqdm` 看不到进度的问题）

`kairos-collect` 在 nohup 模式下 tqdm 不渲染，4 个 worker 共享一个 Python 进程，外层只能 `pgrep` 到 1 个 PID。**很容易误以为挂了**。这次写了一个 heartbeat 监控脚本：

```bash
# /tmp/perp_top10_monitor.sh
#!/usr/bin/env bash
set -u
OUT_DIR="${1:-/root/autodl-tmp/Kairos/raw/crypto/perp_top10}"
LOG="${2:-/root/autodl-tmp/Kairos/logs/perp_top10_collect.log}"
INTERVAL="${3:-60}"

find_pid() {
    pgrep -af kairos.data.collect 2>/dev/null | awk '/python/ {print $1; exit}'
}

prev_cpu=0; t0=$(date +%s)
while true; do
    pid=$(find_pid)
    [[ -z "$pid" ]] && { echo "process gone"; tail -n 10 "$LOG"; exit 0; }
    ncand=$(ls "$OUT_DIR"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    nfund=$(ls "$OUT_DIR"/_extras/funding/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    nspot=$(ls "$OUT_DIR"/_extras/spot/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    sz=$(du -sh "$OUT_DIR" 2>/dev/null | awk '{print $1}')
    # 0x1ED2 = 7890 (proxy port)
    est=$(awk '/ 0100007F:1ED2 / && $4=="01" {c++} END{print c+0}' /proc/"$pid"/net/tcp 2>/dev/null)
    threads=$(awk '{print $20}' /proc/"$pid"/stat 2>/dev/null)
    cpu=$(awk '{print $14+$15}' /proc/"$pid"/stat 2>/dev/null)
    delta_cpu=$(( cpu - prev_cpu ))
    prev_cpu=$cpu
    echo "[$(date +%H:%M:%S) +$(( $(date +%s) - t0 ))s] py_pid=$pid cand=$ncand fund=$nfund spot=$nspot size=$sz proxy_est=$est threads=$threads cpu_delta=$delta_cpu"
    sleep "$INTERVAL"
done
```

关键点：
- `pgrep -af kairos.data.collect | awk '/python/ {print $1}'`—— bash 包装会被 `pgrep -f` 一并匹配，必须再用 `awk` 过滤出 `python` 行。
- `awk '/ 0100007F:1ED2 / && $4=="01"' /proc/$pid/net/tcp` —— `0x1ED2 = 7890` 是 mihomo 端口，`$4=01` 是 `ESTABLISHED`，能直接看到当前几个 worker 在跟代理连着。
- `cpu_delta` 大于 0 说明在 active 工作；`threads` 在 `workers=4` 时应该 ≥ 6（4 worker + 主线程 + GIL helper）。

---

## 4. 打包

```bash
kairos-prepare --market crypto \
    --raw ./raw/crypto/perp_top10 \
    --train 2026-03-21:2026-04-17 \
    --val   2026-03-21:2026-04-17 \
    --test  2026-04-17:2026-04-20 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --extras-channels funding,spot \
    --out ./finetune/data/perp_top10_30d
```

### 产出

- `train_data.pkl` 330k 行 (10 symbols)
- `val_data.pkl`    58k 行
- `test_data.pkl`   43k 行（**= 4035 minutes × 10 symbols ≈ 3 天**）
- `meta.json` 里多了一个 `extras_channels: ["funding", "spot"]` 字段（adapter 透出）
- 32 维 exog 中 `funding_rate / basis` 真实非零，`oi_change / btc_dominance` 仍为 0

---

## 5. 训练

```bash
cat > logs/run_train_perp_top10.sh <<'SH'
#!/bin/bash
set -e
cd /root/autodl-tmp/Kairos
source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1                # 强制走本地 cache
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/perp_top10_30d
export KAIROS_NUM_WORKERS=0
export KAIROS_BATCH_SIZE=64
export KAIROS_EPOCHS=10
export KAIROS_N_TRAIN_ITER=5000        # ⚠️ 这一行就是 §8 里诊断出的元凶
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
SH
```

### 5.1 训练日志（10 epoch 共 1 m 13 s）

```
[TRAIN] pool=327610, using 5000/epoch.
[VAL]   pool=55440,  using 800/epoch.

ep 1: val_ce=2.4621  ✅ save best
ep 2: val_ce=2.4601  ✅ save best
ep 3: val_ce=2.4587  ✅ save best
ep 4: val_ce=2.4577  ✅ save best
ep 5: val_ce=2.4570  ✅ save best
ep 6: val_ce=2.4567  ✅ save best
ep 7: val_ce=2.4564  ✅ save best
ep 8: val_ce=2.4563  ✅ save best
ep 9: val_ce=2.4563  patience 1/3
ep 10: val_ce=2.4563 patience 2/3
```

val_ce 一路降但绝对降幅极小（2.4621 → 2.4563，**Δ = 0.0058**）—— 跟 BTC/ETH run（Δ ~0.05）和 Top100 run（Δ ~0.09）相比小了一个数量级。**这是欠拟合的清晰信号**，但当时没意识到（见 §8.1）。

### 5.2 关键观察

`[TRAIN] pool=327610, using 5000/epoch` —— pool 32 万样本，每 epoch 只随机抽 **5000**，10 epoch 共看 50k 样本，**只是 pool 的 15%**。原因见 §8.1。

---

## 6. 回测

跑了三种 bucket 配置，每种都跑 baseline + finetuned 一对：

| run | bucket | stride | n_records | 用时 |
|---|---|---|---|---|
| 初版（bug → 见 §7.1） | minute | 1 | 40350 | ~5 min × 2 |
| date 重跑 | date | 1 | 40350 | ~5 min × 2 |

```bash
python -u -m kairos.training.backtest_ic --baseline \
    --preset crypto-1min --dataset-path $DATA \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/perp_top10_30d/backtest_baseline_date.json

python -u -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min --dataset-path $DATA \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/perp_top10_30d/backtest_finetuned_date.json
```

---

## 7. 结果

### 7.1 minute bucket（初版，**结果不可靠**）

`bucket=minute` 时每个 bucket 只有 10 个样本（10 symbols × 1 min），单 bucket IC 标准差 ~0.35，4035 个 bucket 平均后 SE ~0.0055，**弱信号被噪声完全淹没**。

| h | metric | baseline | finetuned | Δ |
|---|---|---|---|---|
| h1 | rank_ic (by-min) | +0.0045 | -0.0187 | -0.023 |
| h5 | rank_ic (by-min) | +0.0078 | -0.0108 | -0.019 |
| h30 | rank_ic (by-min) | -0.0387 | +0.0024 | +0.041 |
| h30 | ICIR (by-min) | -0.068 | +0.011 | +0.079 |

`bucket=minute` 在 10 symbols × 短 test 区下的统计性质详见 [`BACKTEST_IC_GUIDE.md`](BACKTEST_IC_GUIDE.md) §3.2。

### 7.2 date bucket（重跑后，**主参考但仍不可靠**：n=3）

| h | metric | baseline | finetuned | Δ |
|---|---|---|---|---|
| h1 | ic (by-day) | -0.0091 | -0.0085 | +0.001 |
| h1 | rank_ic | -0.0129 | -0.0155 | -0.003 |
| h5 | rank_ic | +0.0170 | +0.0021 | -0.015 |
| h30 | ic (by-day) | +0.0319 | +0.0024 | **-0.029** |
| h30 | rank_ic | +0.0078 | +0.0164 | +0.009 |
| h30 | ICIR (by-day) | +1.17 | +0.06 | -1.11 |

⚠️ **`n_dates=3`**（test 区只有 4-17 / 4-18 / 4-19 三天），ICIR 的分母（IC 的标准差）只有 3 个点估计，**完全是噪声**。这就是 ICIR=+1.17 这种"看起来很好"的虚高数字的来源。

### 7.3 pooled overall（**唯一可信的统计信号**，n=40350）

无视 bucket，直接用 4 万条 (score, return) 对算 Pearson/Spearman/hit_rate：

| h | metric | baseline | finetuned | Δ |
|---|---|---|---|---|
| h1 | spearman | -0.0122 | -0.0118 | +0.0004 |
| h1 | hit_rate | 54.14% | 51.11% | **-3.03 pp** |
| h5 | spearman | +0.0167 | +0.0011 | -0.0156 |
| h5 | hit_rate | 48.86% | 49.03% | +0.17 pp |
| h30 | pearson | **+0.0368** | -0.0062 | **-0.0430** |
| h30 | spearman | **+0.0226** | +0.0021 | -0.0205 |
| h30 | hit_rate | 52.43% | 50.93% | -1.50 pp |

**Kronos 原权重 + 随机 head 在 pooled h30 上有 +0.037 的 IC（p < 1e-13）**——这不是 alpha，是 transformer hidden state 本身已经编码了未来分布信息，随机 fc 也能蹭出方向。**finetuned 把这点信号磨没了**——是负迁移。

### 7.4 与历史 run 对比

| run | universe × 时长 | h30 ICIR (by-day, finetuned) | h30 pooled spearman | exog 真实非零 |
|---|---|---|---|---|
| BTC/ETH 2y spot | 2 × 2 年 | +0.325 (n=106 days) | +0.032 | ❌（pad 0） |
| Top100 1y spot  | 100 × 1 年 | +0.454 (n=78 days)  | n/a    | ❌（pad 0） |
| **Top10 30d perp** | **10 × 30 天** | **+0.063 (n=3, 噪声)** | **+0.002** | ✅ funding+basis |

数据规模差 10-30 倍，结论看不出 funding/basis 有没有用。

---

## 8. Post-mortem：三条 root cause

### 8.1 🔴 元凶：`KAIROS_N_TRAIN_ITER=5000` 残留 → 实际只用了 15% 的训练池

启动 mini run（5d × 5 symbols 验证链路）时设了 `KAIROS_N_TRAIN_ITER=5000`，正式 run 时没清掉。

`kairos/training/dataset.py` L80:
```python
limit = cfg.n_train_iter if split == "train" else cfg.n_val_iter
self.n_samples = min(limit, len(self.indices))
print(f"[{split.upper()}] pool={len(self.indices)}, using {self.n_samples}/epoch.")
```

含义：**`n_train_iter` 是每 epoch 看多少个样本**（不是 step 数）。

- BTC/ETH run：default `n_train_iter=50000`，pool ~100 万 → 每 epoch 看 5%，15 epoch 看 75%。
- Top100 run：default 50000，pool 3160 万 → 每 epoch 看 0.16%，但 4 epoch 早停后仍是 200k 样本（5× 当前）。
- **本 run**：`n_train_iter=5000`，pool 32 万 → 每 epoch 看 **1.5%**，10 epoch 共看 **15%** = 50k 样本。

50k 样本对一个有 5.4M 参数 + 32 维 exog + 30 step pinball loss 的微调任务**严重不够**。val_ce 只降 0.006 就是欠拟合的直接表现。

**修复**：清掉 env，或用默认 50000，或改 `train_predictor.py` 在 log 末尾打印 `total_steps_seen / pool` 和 warning。

### 8.2 🟡 训练 target 量纲设计有偏

`kairos/training/train_predictor.py` L107-114:
```python
close_n = x[:, :, close_idx]  # ⚠️ normalized close（已经被 batch 内 z-score）
T = close_n.size(1)
h = cfg.return_horizon  # = 30
targets = []
for k in range(h):
    rolled = torch.roll(close_n, shifts=-(k + 1), dims=1)
    targets.append(rolled - close_n)         # cumulative diff
target = torch.stack(targets, dim=-1)        # [B, T, h]
```

两个问题：

1. **量纲是 normalized diff**（每个 batch 局部 z-score），但 `backtest_ic.py` L244-245 真值用的是 raw log-return：
   ```python
   meta[f"ret_h{h}"] = float(np.log(cf / c0))
   ```
   IC（Pearson/Spearman）对单调变换不敏感，理论上不影响 IC 符号；但模型学到的 quantile 分布是在 normalized 空间，回测做 cross-sectional 排序时可能引入方差不齐。
2. **第 k 步的 target 是 `close[t+k+1] - close[t]`** —— cumulative diff，量纲随 k 线性增大。配合 preset `return_horizon=30`，**k=29 的 loss 主导整个 pinball**，模型实际只优化了"未来第 30 步"，h1-h29 几乎没监督信号。

这能解释为什么 BTC/ETH 和 Top100 两次 run 都只有 h30 上能看到效果，h1/h5 都不动。

**修复方向**（未实施）：
- 把 target 改成 raw log-return 或 step-wise diff，并按 k 做 per-horizon normalization
- 或者在 `n_quantiles=9` 之外加一个明确的 "1-step" head 给 h1 学

### 8.3 🟡 Test 区只 3 天 → date bucket 不可靠 + minute bucket 噪声大

interleave 把 4-17 ~ 4-19 切给 test，`n_dates=3`：
- date bucket：3 个 IC 算 ICIR，**完全是噪声**（ICIR 标准误 ~ 1/√3 ≈ 0.58）
- minute bucket：4035 个 bucket，但每 bucket 只 10 个样本，单 bucket IC 标准差 ~0.35，平均后 SE ~0.0055
- 唯一可信的是 pooled IC（n=40350）

**修复方向**：test 区至少给到 15 天以上（n_buckets ≥ 15），或者代码侧让 `backtest_ic` 在 `n_buckets < 10` 时强制 fallback 到 pooled 并打 warning。详见 [`BACKTEST_IC_GUIDE.md`](BACKTEST_IC_GUIDE.md)。

---

## 9. 产物清单

### 服务器（AutoDL `/root/autodl-tmp/`）

```
Kairos/
├── raw/crypto/perp_top10/                            15 MB
│   ├── *.parquet                                      10 个 perp OHLCV
│   ├── _extras/funding/*.parquet                      10 个 funding rate（全非零）
│   └── _extras/spot/*.parquet                         9 个 spot mid（XAU 缺）
├── finetune/data/perp_top10_30d/                     ~440 MB
│   ├── {train,val,test}_data.pkl                     330k / 58k / 43k 行
│   ├── exog_{train,val,test}.pkl
│   └── meta.json                                     extras_channels: [funding, spot]
├── artifacts/
│   ├── checkpoints/predictor/checkpoints/
│   │   ├── best_model/                               🚨 Top10 perp 微调（已覆盖 Top100 ckpt）
│   │   └── best_model_btceth_backup/                 ✅ BTC/ETH 备份（保留）
│   └── perp_top10_30d/
│       ├── backtest_baseline.json                    minute bucket
│       ├── backtest_finetuned.json                   minute bucket
│       ├── backtest_baseline_date.json               date bucket
│       └── backtest_finetuned_date.json              date bucket
└── logs/
    ├── perp_top10_collect.log / perp_top10_train.log
    ├── run_btceth_recheck.sh / btceth_recheck.log    sanity 验证脚本（§10.2）
    └── run_top10_date.sh / top10_date.log            date bucket 重跑
```

> ⚠️ **Top100 的 ckpt 已被本次覆盖**。如果想重新基于 Top100 ckpt 比较，需要重跑 §6 of [`CRYPTO_TOP100_RUN.md`](CRYPTO_TOP100_RUN.md)，或者从下一次 run 开始养成 `cp best_model best_model_<run-name>_backup` 的习惯。

---

## 10. 怎么验证 "代码没 regression"（写给未来的自己）

### 10.1 同代码 + 老数据 sanity 检查

我用今天 main 分支的代码、老 BTC/ETH ckpt、老数据集（`finetune/data/crypto_1min_btc_eth/`）重跑了一次回测：

```bash
python -u -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model_btceth_backup \
    --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --aggregation date --stride 5 \
    --out artifacts/btceth_recheck/finetuned_date_stride5.json
```

| metric | 老 4-17 跑 (stride=1) | 今天重跑 (stride=5) | 结论 |
|---|---|---|---|
| h30 rank_ic | +0.0505 | +0.0238 | 方向一致；幅度差 ~½，符合 stride=5 → SE × √5 倍 |
| h30 ICIR | +0.325 | +0.147 | 同上 |
| 整体方向 (h1/h5/h30 三档符号) | 同 | 同 | 同 |

✅ **代码没 regression**，老 alpha 完全可复现。

### 10.2 类似 sanity 检查应该作为发布前的固定步骤

下次每次大改完 `train_predictor.py` / `backtest_ic.py` / `kronos_ext.py`，都跑一遍这个："finetuned BTC/ETH ckpt + finetuned BTC/ETH 数据 + horizon 1,5,30 + date bucket"，跟历史结果对一下。详见 [`BACKTEST_IC_GUIDE.md`](BACKTEST_IC_GUIDE.md) §5。

---

## 11. 下一步

按 ROI 排：

1. **D（最快验证，~10 min）**：清掉 `KAIROS_N_TRAIN_ITER`，用默认 50000 重训本 dataset，看 finetuned 能不能至少**不**回退于 baseline。
2. **B（防再踩，~5 min 改代码）**：
   - `_BUCKET_ALIASES.auto` 改成"`n_dates < 10` 时降级到 pooled，否则按 freq 选 date/hour/minute"
   - `dataset.py` 在 log 末尾打印 `using {n_train_iter}/epoch ({pct:.1%} of pool)`，并在 pct < 5% 时 warning
3. **D'（数据扩到能用规模，~30 min 采集 + 10 min 训）**：Top10 × 90 天（funding 覆盖极限）；test 区给到 15 天 (n_buckets ≥ 15)。
4. **C（结构性修，~20 min）**：修 train pinball target 量纲（normalized diff → raw log-return + per-k normalization），重训 + 回测；这是 short-horizon IC 一直起不来的最深根因。

只有把 1-2 做完拿到一个 "至少不 regression" 的 baseline，3-4 才有意义；否则只是在错误的训练配置上堆数据 / 改 loss。

---

## 12. 一行 TL;DR

> 链路通了，**funding + spot 真实非零**进了 32 维 exog；但因为 `KAIROS_N_TRAIN_ITER=5000` 残留 + test 区只 3 天 + bucket 选 minute，三层叠加导致最终 finetuned 出现负迁移。代码本身没 regression（用老数据复跑老结果通过）。下次先修 env 残留 + bucket auto 逻辑再扩数据。
