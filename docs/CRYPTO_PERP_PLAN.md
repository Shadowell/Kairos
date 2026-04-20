# Crypto 永续多通道数据计划书（Phase 1）

> **状态**：草稿 / 待 review
> **目的**：让 `--market crypto` 真正利用 funding / OI / basis，彻底解锁 `MARKET_EXOG_COLS` 的 5 个"数据驱动"位。
> **不破坏**：A 股主路径、binance_vision 现货采集、已有的现货 Top100 run。

---

## 0. 动机 — 为什么必须改

CRYPTO_TOP100_RUN.md §11 的 h30 结果很亮（ICIR 0.454），但 h1/h5 翻负。
排查后发现根因 = **EXOG 的 5 个数据驱动位 (`funding_rate`, `funding_rate_z`, `oi_change`, `basis`, `btc_dominance`) 在任何现有 run 里都永远是 0**。

原因：

1. `CryptoAdapter.market_features` 设计上会吃 `context.extras["funding_rate"]` 等，但**没有任何代码往 `extras` 里塞东西**。
2. `kairos-collect` 的 crypto 路径只落盘 OHLCV parquet，不采集 funding / OI / spot。
3. `prepare_dataset.build_features(...)` 不传 `extras`，`extras={}` → adapter `_align_series` 返回全 NaN → fillna(0) → 就是 0。

因此：哪怕我们切到 OKX 永续，只要不改代码，EXOG 的"永续专属"那 5 列还是 0，模型看到的"永续"跟"现货"几乎没差。

---

## 1. 数据落盘规范

### 1.1 目录布局

```
raw/crypto/perp_1min_top100/          # 主目录，存 perp OHLCV parquet
  BTC_USDT-USDT.parquet               # 与现有 sanitize_symbol 一致
  ETH_USDT-USDT.parquet
  ...
  _extras/                            # 辅助因子目录（双下划线前缀避免和 symbol 名冲突）
    funding/
      BTC_USDT-USDT.parquet           # 单列 funding_rate，8h 一条
      ...
    open_interest/
      BTC_USDT-USDT.parquet           # 单列 open_interest，5min 一条
      ...
    spot/
      BTC_USDT-USDT.parquet           # spot close 1min，用来算 basis
      ...
    btc_dominance.parquet             # 全市场共享一条，1min 或 1h 粒度
```

### 1.2 parquet schema（所有辅助 parquet 共用约定）

| 列名 | 类型 | 说明 |
|---|---|---|
| `datetime` | `datetime64[ns]` (naive, UTC) | 与主 parquet 同步，便于 `_align_series` reindex |
| `<payload>` | float64 | 单列 payload：`funding_rate` / `open_interest` / `spot_close` / `btc_dominance` |

- 一个辅助 parquet 只存一列 payload + datetime，不掺别的，方便直接 `df.set_index("datetime")["<col>"]`。
- `datetime` 必须已去重、升序排列。

### 1.3 为什么不直接合并到主 parquet

- 频率不同：K-line 1min、funding 8h、OI 5min —— 合并进主 parquet 需要 ffill/reindex，会丢失原始频率信息。
- 再采集时，重跑某一通道不会牵连主 K-line。
- 跟现有的 `fetch_one → to_parquet` 保持解耦，不改旧的 dataframe schema。

---

## 2. 采集层改动（Phase 2）

### 2.1 API 扩展

在 `kairos-collect` 增加 CLI flag（默认关闭，不影响现货 run）：

```
--crypto-extras {none,funding,oi,spot,basis,all}[,...]
    default: none (backwards-compatible)
    值:
      funding   → 采 OKX funding-rate-history (8h 粒度)
      oi        → 采 OKX open-interest-history (5m 粒度)
      spot      → 采对应 spot symbol 的 1min OHLCV（用于算 basis）
      basis     → 只采 spot，打包时再算 basis（= spot 的别名，保留两个名字）
      all       → funding + oi + spot
```

### 2.2 分工

| 模块 | 职责 |
|---|---|
| `kairos/data/collect.py::main` | 解析 `--crypto-extras`，下发给 adapter |
| `kairos/data/markets/crypto.py::CryptoAdapter.fetch_extras(symbol, …)` | 新方法：按请求集合调 OKX 对应端点，返回 `{"funding": df, "oi": df, "spot": df}` |
| `kairos/data/crypto_extras.py`（新文件） | 落盘 util：`save_extras(out_dir, symbol, extras_dict)` + `load_extras(out_dir, symbol, kinds)` |
| `kairos/data/collect.py::fetch_one` | 采完 OHLCV 后，若 `--crypto-extras != none`，再串行调 `fetch_extras` 写 `_extras/<kind>/` |

### 2.3 为什么串行、不并发

- OKX 的限流是**端点级 IP 限流**（`public/*` 20 req/2s，`rubik/stat/*` 5 req/2s），同一 symbol 内 extras 4 类各跑一次，总请求数小（funding 10-50 次 + OI ~30 次 + spot 和主 K-line 对等），4 worker 并发基本不会触发。
- 极端情况下加一个全局 `ccxt enableRateLimit=True` 就够了，不需要自己写限流。

### 2.4 断点续传

复用现有 `daily_append` 逻辑：

- 每种 extras parquet 单独 `_load_existing`，若已存在则只增量抓新时间段。
- 失败的 extras 不影响主 OHLCV 保存（采主成功 + extras 失败 = 记 warn，任务仍算 ok）。

---

## 3. 打包层改动（Phase 3）

### 3.1 改动点

`kairos/data/prepare_dataset.py::per_symbol_split`（L120）当前：

```python
df = build_features(df, index_df, market=market, symbol=path.stem)
```

改为：

```python
extras = None
if market == "crypto":
    extras = crypto_extras.load_for_symbol(
        raw_dir=path.parent,        # .../perp_1min_top100/
        symbol_stem=path.stem,      # "BTC_USDT-USDT"
        freq=freq,
    )
df = build_features(df, index_df, market=market, symbol=path.stem, extras=extras)
```

### 3.2 `load_for_symbol` 行为

- 查 `path.parent / "_extras" / <kind> / <stem>.parquet` 是否存在，存在就读出来，以 `datetime` 为 index 返回 Series。
- 缺失的 kind 直接不进 `extras` dict，adapter 侧 `_align_series` 会返回 NaN → fillna(0)。**向后兼容**：如果用户没采 extras，打包行为跟今天完全一致。
- 返回的 dict key 跟 `CryptoAdapter.market_features` 读取的 key 保持一致：
  - `"funding_rate"` → funding parquet 的 `funding_rate` 列
  - `"open_interest"` → OI parquet 的 `open_interest` 列
  - `"spot_close"` → spot parquet 的 `close` 列
  - `"btc_dominance"` → `_extras/btc_dominance.parquet`（全市场共享文件）

### 3.3 meta.json 扩展

打包结束后写的 `meta.json` 增加一个字段：

```json
{
  "market": "crypto",
  "freq": "1min",
  "exog_channels_available": ["funding_rate", "oi_change", "basis"],
  ...
}
```

下游 backtest 或文档生成可以直接读这个，无需重新探测目录。

---

## 4. 不做的事

1. **不动 KronosWithExogenous 的 `n_exog=32`**。架构不变式。
2. **不做 spot VWAP / spot volume**。basis 只用 spot close。
3. **不做 binance_futures adapter**。留在后续做。
4. **不删除、不重命名** 已有的 parquet 目录（`binance_vision/*` 现货继续可用）。

---

## 5. 时间预算

| Phase | 工作 | 估时 |
|---|---|---|
| 2 | 改 collect.py + crypto adapter + crypto_extras.py 写入 | 2-3 h |
| 3 | 改 prepare_dataset.py + crypto_extras.py 读取 | 1 h |
| 4 | 本地 Top3 × 1 day smoke（验证 extras 值有异、build_features 32 列非零） | 30 min |
| 5 | AutoDL Top5 × 1 day mini end-to-end | 2 h |
| 6 | Top100 × 1 年全量采集 + 训练 + 回测 | 10-12 h（大头在采集和回测 stride=10） |
| 7 | 写 docs/CRYPTO_PERP_RUN.md + 对比表 | 1 h |
| **总计** | | **~18 h** |

---

## 6. 失败场景 & fallback

| 场景 | fallback |
|---|---|
| OKX API 某个 symbol funding/OI 404（非永续或已下架） | 该 kind 不写 parquet，打包侧 `_align_series → NaN → 0`，继续训练 |
| mihomo 断连 / 节点被封 | `fetch_extras` 的 ccxt retry 会报错，`fetch_one` 记 `fail` 但不中断整批 |
| spot symbol 不存在（某些 alt coin 只有 perp，没对应 spot） | 跳过 spot，basis=0 |
| OKX funding 历史只有 3 个月（老数据）| 当前 run 只要最新 1 年，落在 OKX 有覆盖的区间里，没问题；若超出，新老时段都存进去，打包时 `_align_series` ffill 兜底 |

---

## 7. 验证清单（Phase 4/5 用）

- [ ] 采完 BTC-USDT-SWAP 一天：主 parquet 1440 行、funding parquet 3 行（8h）、OI parquet 288 行（5m）、spot parquet 1440 行
- [ ] `build_features(perp_df, market="crypto", extras={funding, oi, spot})` 输出 32 列，其中 `funding_rate` 非零比例 > 0
- [ ] `prepare_dataset` 产出 `exog_train.pkl` 的均值/方差对 5 个永续专属列**非零**（z-score 后应该是有分布的）
- [ ] `train_predictor` smoke（KAIROS_SMOKE=1）跑通
- [ ] `backtest_ic` smoke 跑通，结果文件落盘
