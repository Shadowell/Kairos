# kairos-serve HTTP API

API 0.3 使用 `kairos.inference.KairosPredictor`，与回测共享历史归一化、外生通道和收益头。
只接受带 `kairos_manifest.json` 的 contract v2 模型包，目标是 `log(close[t+h]/close[t])`。
旧版归一化价格差 checkpoint 必须重训，不能通过补写 manifest 转换。

## 启动

```bash
kairos-serve --predictor artifacts/checkpoints/predictor/<run_id>/checkpoints/best_model \
  --device cpu --host 127.0.0.1 --port 8000
```

模型目录必须整体保留：模型配置/权重、manifest、`tokenizer/` 快照。下载 Hugging Face 模型时先下载完整 snapshot。
`--tokenizer` 不再用于替换模型绑定的 tokenizer；可省略，传入时必须是包内快照路径。
默认使用 CUDA（可用时）或 CPU。上下文长度从模型包读取，不再提供 `--max-context` 覆盖。
服务不采集行情或补抓历史。

## GET /health

返回 status、device、max_context（实际 lookback）、target、freq、return_horizon。
这仅验证服务已加载模型，不证明数据源新鲜度或预测有效性。

## POST /predict

| 字段 | 约束 |
| --- | --- |
| symbol | 必填，非空，例如 BTC/USDT |
| market_type | spot 或 swap；模型包标注时必须一致 |
| freq | 默认 1min，必须与模型包的 bar 时长一致 |
| bars | 不少于模型 lookback，至多 10000 根；升序、无重复，使用的末尾历史须连续 |
| exog_cols | 启用外生通道时必填，等于 manifest 的有序 32 列 schema |
| exog | 启用外生通道时必填，二维数值数组，逐行与 bars 对齐；所有值有限 |
| lookback | 可省略，填写时必须等于模型契约 |
| pred_len | 可省略，默认全部已训练期限；不得超出 return_horizon |

每根 bar 包含 datetime、open、high、low、close、volume、可选 amount。
时间按 UTC 解释；OHLC 必须为有限正数，volume/amount 非负；缺失 amount 使用 close*volume。
外生数值必须由同一版本的 `build_features` 及相同历史/sidecar 口径产生，不能把缺失通道当成任意零值。
建议直接使用打包数据中的 exog 行构造离线验收请求。服务不对乱序或无效输入进行静默修复。

## 响应

- target 固定为 log_return，anchor_time 为最后一个可见 bar 的 UTC 时间。
- quantile_levels 对应收益头的分位水平，例如 0.1、0.2、…、0.9。
- forecast 每项包含 horizon、UTC time、log_return_quantiles、median_return、median_close、quantile_crossing。
- median_return = exp(预测对数收益中位数)-1；median_close = last_close*exp(预测对数收益中位数)。
- pred_close 是各期限 median_close 的便捷列表。
- quantile_crossing 为 true 表示原始分位数发生交叉；服务不会排序并掩盖模型问题。

API 0.3 不再返回 `pred_direction_prob_up`、`pred_mean_return` 或虚构的 OHLC/volume 预测。
分位数并不自动构成经过校准的上涨概率。旧参数 T/top_p/top_k/sample_count 不适用于收益头并会被拒绝。

## 错误与迁移

请求结构错误返回 422；频率/schema/期限/时序不匹配返回 400；非有限模型结果返回 500。
启动时旧模型包、tokenizer 哈希不符或模型配置冲突会明确失败。
迁移时重新打包数据、训练 v2 模型、更新调用方请求/响应处理；保留旧实验和指标用于追溯。
