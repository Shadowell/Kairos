# kairos-serve HTTP API

`kairos-serve` 由 `kairos.deploy.serve` 实现。它加载 `KronosTokenizer` 和基础 `Kronos` predictor checkpoint，然后调用 `KronosPredictor.predict` 采样未来 K 线。

服务端**不负责抓交易所数据**。调用方必须在请求体里提供最近的 OHLCV bars。这样可以让服务部署不依赖 OKX 网络可用性，也避免 `/predict` 内部隐藏数据源行为。

## 启动服务

```bash
kairos-serve \
  --tokenizer NeoQuasar/Kronos-Tokenizer-base \
  --predictor NeoQuasar/Kronos-small \
  --host 0.0.0.0 \
  --port 8000
```

主要参数：

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--tokenizer` | 是 | Hugging Face repo id 或本地 tokenizer checkpoint 路径 |
| `--predictor` | 是 | Hugging Face repo id 或本地 Kronos predictor checkpoint 路径 |
| `--device` | 否 | 默认优先 CUDA，其次 MPS，最后 CPU |
| `--max-context` | 否 | 传给 `KronosPredictor` 的最大上下文长度 |
| `--host` / `--port` | 否 | Uvicorn 监听地址 |

## `GET /health`

响应示例：

```json
{
  "status": "ok",
  "device": "cpu",
  "max_context": 512
}
```

## `POST /predict`

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `symbol` | string | 是 | - | 交易所原生 symbol，例如 `BTC/USDT` |
| `market_type` | string | 否 | `spot` | `spot` 或 `swap` |
| `freq` | string | 否 | `1min` | `1min`、`3min`、`5min`、`15min`、`30min`、`60min`、`1h`、`2h`、`4h`、`1d`、`daily` |
| `bars` | array | 是 | - | 至少 32 根 OHLCV bars |
| `lookback` | int | 否 | `400` | 使用最近多少根 bars 作为上下文 |
| `pred_len` | int | 否 | `30` | 向未来采样多少根 bars |
| `T` | float | 否 | `0.6` | 采样温度 |
| `top_p` | float | 否 | `0.9` | nucleus sampling 阈值 |
| `top_k` | int | 否 | `0` | top-k 采样，`0` 表示关闭 |
| `sample_count` | int | 否 | `5` | 采样轨迹数量 |

`bars` 每一项字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `datetime` | 是 | K 线时间戳，推荐 ISO-8601 |
| `open` / `high` / `low` / `close` | 是 | 价格字段 |
| `volume` | 是 | 调用方提供的基础币成交量或合约张数 |
| `amount` | 否 | 计价币成交额；缺失时服务端用 `close * volume` 近似 |

请求示例：

```json
{
  "symbol": "BTC/USDT",
  "market_type": "spot",
  "freq": "1min",
  "pred_len": 3,
  "bars": [
    {
      "datetime": "2026-04-30T00:00:00Z",
      "open": 70000.0,
      "high": 70020.0,
      "low": 69980.0,
      "close": 70010.0,
      "volume": 12.5,
      "amount": 875125.0
    }
  ]
}
```

真实请求必须提供至少 32 根 bars。

响应字段：

| 字段 | 说明 |
| --- | --- |
| `symbol`、`market_type`、`freq` | 回传请求元数据 |
| `last_close` | 输入序列最后一根 close |
| `pred_close` | 预测 close 序列 |
| `pred_mean_return` | 预测 close 相对 `last_close` 的平均收益 |
| `pred_direction_prob_up` | 预测 close 高于 `last_close` 的比例 |
| `forecast` | 未来 bar 对象，包含 time、open、high、low、close、volume |

## Curl 示例

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  --data @request.json
```

## 注意事项

- `kairos-serve` 当前走原版 Kronos predictor 路径，不是 32 维外生通道 predictor 路径。
- 该接口适合快速服务演示。生产部署应在调用服务前校验上游 K 线归一化和时间戳单调性。
