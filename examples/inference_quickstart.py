"""Kairos 推理 quickstart: 用 NeoQuasar 官方 Kronos 权重直接预测 A 股。

跑之前: pip install -e .
"""

from __future__ import annotations

import pandas as pd

from kairos import Kronos, KronosPredictor, KronosTokenizer


def main(symbol: str = "600977", lookback: int = 400, pred_len: int = 20):
    import akshare as ak

    print(f"[1/4] 拉取 {symbol} 日线")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq").tail(lookback)
    df = df.rename(columns={
        "日期": "datetime", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount",
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print("[2/4] 加载 Kronos")
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tok, max_context=512)

    print("[3/4] 推理")
    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df["datetime"]
    y_ts = pd.Series(pd.bdate_range(
        start=x_ts.iloc[-1] + pd.Timedelta(days=1),
        periods=pred_len,
    ))
    pred = predictor.predict(x_df, x_ts, y_ts, pred_len=pred_len,
                             T=0.6, top_p=0.9, sample_count=5, verbose=False)

    print("[4/4] 预测结果:")
    print(pred.head(10))


if __name__ == "__main__":
    main()
