# 下一组缺陷：UTC sidecar、horizon 损失平衡、精确增量采集

## User Scenarios & Testing
### US1 / Issue 3 (P1)
同一绝对时间的 UTC、带偏移和无时区外生数据产生相同因子；缺失通道可为空，但已提供的无效/冲突数据不得静默变成零信号。
### US2 / Issue 5 (P1)
所有 horizon 继续输出原始 log-return；训练与验证使用一致、来自训练集的逆波动率权重，防止长 horizon 单靠尺度主导监督。
### US3 / Issue 8 (P1)
1min/5min/1h/1d 等频率从最后一根 bar 的下一步精确续采；不会漏掉当天剩余数据，时间比较不依赖本机时区。

## Requirements
- FR-001 所有 sidecar/主时间转 UTC 统一比较；对齐只允许过去向前填充；保留原输入不变。
- FR-002 None/空通道仍按缺失处理；无效时间、NaT、重复时间冲突、错误shape必须可见地拒绝，不把RangeIndex当Unix纳秒。稳定排序合法的乱序sidecar。
- FR-003 权重为训练目标标准差的倒数并归一为均值1；配置正数floor防零方差，固定seed限制样本数；不得读取val/test或每批重新估计。
- FR-004 `_train` 预先确定全rank相同权重，训练/验证共同使用；模型包保存方法、样本数/实际权重及floor，旧包输出和推理契约不变；支持显式uniform对照。
- FR-005 daily-append按bar_delta和UTC推进；精确判断是否已到排除结束边界，裸日期end包括整日；保留既有行情并去重排序。
- FR-006 追加模式遇到损坏既有文件/无效时间先失败并保留原文件；无新数据时保持历史不变。extras使用与新增OHLCV一致的请求窗口。

## Success Criteria
新增回归先红后绿；全量pytest -x -W error；真实临时parquet和小型CPU predictor smoke通过；不启动生产采集、付费训练或部署。

## Scope and Clarifications
复用既有UTC解析、32维schema、v2 bundle和索引采样。#1/#2/#4/#6/#7核心已落地主线；本轮不重做，也不实施#9多尺度RFC。
权重只影响训练损失，不改变预测值的金融量纲。min/max horizon不扩展。
