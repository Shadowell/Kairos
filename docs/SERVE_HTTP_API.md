# kairos-serve HTTP API（`/predict` JSON 约定）

`kairos-serve` 由 `kairos.deploy.serve` 实现：`KronosTokenizer` + 原版 **`Kronos`** 预测器 + `KronosPredictor.predict` 采样未来 K 线；**不涉及** `KronosWithExogenous` 或 32 维外生通道。行情来自 **akshare**（A 股）。

启动示例：

```bash
kairos-serve --tokenizer NeoQuasar/Kronos-Tokenizer-base \
  --predictor NeoQuasar/Kronos-small
```

下文中的「类型」均为 JSON 语义；数值在传输中为 JSON number/string。

---

## `GET /health`

**响应 200**（示例）：

```json
{
  "status": "ok",
  "device": "cpu",
  "max_context": 512
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定为 `"ok"` |
| `device` | string | 推理设备字符串 |
| `max_context` | integer | `KronosPredictor.max_context` |

---

## `POST /predict`

- **Content-Type**：`application/json`
- **实现**：[`kairos/deploy/serve.py`](../kairos/deploy/serve.py) 中的 `PredictRequest` / `PredictResponse`

### 请求体（JSON）

| 字段 | 类型 | 必填 | 默认值 | 约束 / 枚举 | 说明 |
|------|------|------|--------|-------------|------|
| `symbol` | string | 是 | — | — | 六位 A 股代码，如 `"600977"` |
| `freq` | string | 否 | `"daily"` | `daily`、`5min`、`15min`、`30min`、`60min` | `daily` 用日线；其余用分钟 EM 接口 |
| `lookback` | integer | 否 | `400` | `[32, 2000]` | 使用的历史 bar 数量（尾部截取） |
| `pred_len` | integer | 否 | `20` | `[1, 240]` | 预测未来多少个时间点（与响应中列表长度一致） |
| `T` | float | 否 | `0.6` | `(0, 2]` | 采样温度 |
| `top_p` | float | 否 | `0.9` | `(0, 1]` | nucleus 采样 |
| `top_k` | integer | 否 | `0` | `≥ 0` | top-k（0 表示不按 top-k 截断，依实现而定） |
| `sample_count` | integer | 否 | `5` | `[1, 32]` | 每条预测路径采样条数 |
| `adjust` | string | 否 | `"qfq"` | — | akshare `adjust`，如复权选项 |

**请求示例**：

```json
{
  "symbol": "600977",
  "freq": "daily",
  "lookback": 400,
  "pred_len": 20,
  "T": 0.6,
  "top_p": 0.9,
  "top_k": 0,
  "sample_count": 5,
  "adjust": "qfq"
}
```

### 响应体（200，JSON）

| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | string | 与请求一致 |
| `freq` | string | 与请求一致 |
| `last_close` | float | 历史窗口最后一根的收盘价 |
| `pred_close` | array of float | 每条预测采样得到的收盘价序列，`length === pred_len` |
| `pred_mean_return` | float | `pred_close` 相对 `last_close` 的简单平均收益 |
| `pred_direction_prob_up` | float | 预测收盘价中大于 `last_close` 的比例（0–1） |
| `forecast` | array of object | 与 `pred_df` 每行对齐，见下表 |

**`forecast` 每项**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `time` | string | 该步时间，`"%Y-%m-%d %H:%M:%S"` |
| `open` | float | 预测开盘价 |
| `high` | float | 预测最高价 |
| `low` | float | 预测最低价 |
| `close` | float | 预测收盘价 |
| `volume` | float | 预测成交量 |

**响应示例（片段）**：

```json
{
  "symbol": "600977",
  "freq": "daily",
  "last_close": 12.34,
  "pred_close": [12.4, 12.41, 12.42],
  "pred_mean_return": 0.003,
  "pred_direction_prob_up": 0.65,
  "forecast": [
    {
      "time": "2026-04-22 00:00:00",
      "open": 12.35,
      "high": 12.5,
      "low": 12.3,
      "close": 12.45,
      "volume": 1000000.0
    }
  ]
}
```

### 常见错误响应

服务端以 FastAPI `HTTPException` 返回，`body` 中通常包含 `detail`（字符串）。

| HTTP 状态码 | 说明（来自现有分支） |
|-------------|----------------------|
| 400 | 历史 bar 不足（`len(df) < 32`） |
| 404 | 数据源无数据 |
| 502 | akshare 拉取异常（`data source error: ...`） |

`422`：请求 JSON 不符合 Pydantic 校验（缺必填字段、类型错误、越界等）。

---

## `curl` 示例

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"600977","freq":"daily","pred_len":5}'
```

---

## 与 CLI 的对应关系

入口：`python -m kairos.deploy.serve` 或 `kairos-serve`（见 `pyproject.toml` `[project.scripts]`）。
