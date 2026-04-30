# kairos-serve HTTP API

`kairos-serve` is implemented by `kairos.deploy.serve`. It loads a
`KronosTokenizer`, a base `Kronos` predictor checkpoint, and calls
`KronosPredictor.predict` to sample future bars.

The service does **not** fetch exchange data. The caller must provide recent
OHLCV bars in the request body. This keeps serving independent from OKX network
availability and avoids hidden data-source behavior inside `/predict`.

## Start The Server

```bash
kairos-serve \
  --tokenizer NeoQuasar/Kronos-Tokenizer-base \
  --predictor NeoQuasar/Kronos-small \
  --host 0.0.0.0 \
  --port 8000
```

Important arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--tokenizer` | Yes | HF repo id or local tokenizer checkpoint path |
| `--predictor` | Yes | HF repo id or local Kronos predictor checkpoint path |
| `--device` | No | Defaults to CUDA, then MPS, then CPU |
| `--max-context` | No | Max context length passed to `KronosPredictor` |
| `--host` / `--port` | No | Uvicorn bind address |

## `GET /health`

Response:

```json
{
  "status": "ok",
  "device": "cpu",
  "max_context": 512
}
```

## `POST /predict`

Request fields:

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `symbol` | string | Yes | - | Exchange-native symbol, e.g. `BTC/USDT` |
| `market_type` | string | No | `spot` | `spot` or `swap` |
| `freq` | string | No | `1min` | `1min`, `3min`, `5min`, `15min`, `30min`, `60min`, `1h`, `2h`, `4h`, `1d`, `daily` |
| `bars` | array | Yes | - | At least 32 OHLCV bars |
| `lookback` | int | No | `400` | Tail bars used for context |
| `pred_len` | int | No | `30` | Number of future bars to sample |
| `T` | float | No | `0.6` | Sampling temperature |
| `top_p` | float | No | `0.9` | Nucleus sampling threshold |
| `top_k` | int | No | `0` | Top-k sampling; `0` disables |
| `sample_count` | int | No | `5` | Number of sampled trajectories |

Each item in `bars`:

| Field | Required | Description |
| --- | --- | --- |
| `datetime` | Yes | Bar timestamp, ISO-8601 preferred |
| `open` / `high` / `low` / `close` | Yes | Price fields |
| `volume` | Yes | Base or contract volume as provided by the caller |
| `amount` | No | Quote amount; defaults to `close * volume` |

Example:

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

The real request must include at least 32 bars.

Response fields:

| Field | Description |
| --- | --- |
| `symbol`, `market_type`, `freq` | Echoed request metadata |
| `last_close` | Last input close |
| `pred_close` | Forecast close sequence |
| `pred_mean_return` | Mean forecast close return vs `last_close` |
| `pred_direction_prob_up` | Fraction of sampled forecast closes above `last_close` |
| `forecast` | Future bar objects with time, open, high, low, close, volume |

## Curl Skeleton

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  --data @request.json
```

## Notes

- `kairos-serve` currently runs the original Kronos predictor path, not the
  exogenous-channel predictor path.
- The endpoint is suitable for quick serving demos. Production deployment should
  validate upstream bar normalization and timestamp monotonicity before calling
  the service.
