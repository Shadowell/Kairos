# Implementation Plan

## Technical Context
复用pandas/PyTorch、Spec Kit1.0.6、当前206test基线，Python3.12本地CPU验证。

## Constitution Check
固定32维、因果特征与既有数据保护；不覆盖已提交v2契约；按文件归属并行，root统一审查提交推送。

## Modules
- Issue3：markets/crypto.py的_align_series与crypto_extras.py的时间规范化/读取诊断；仅缺失允许neutral fallback；针对UTC、偏移、NaT、重复/乱序加回归。
- Issue5：training/horizon_weights.py估计训练集raw目标的逆标准差权重，config统一参数，train_predictor首次训练前确定；pinball_loss支持正horizon_weights且均值归一；artifacts显式记录方法/有效权重并验证。
- Issue8：collect.py以UTC和bar_delta计算续采起点/结束条件；坏历史显式失败保护，真实parquet+fake adapter验证实际请求和落盘。

## Validation
各agent先复现后改；不得修改彼此文件。root核对#3/#8 sidecar交接、#5loss/manifest兼容，做真实打包、CPU smoke、全部测试、diff/sensitive检查；更新已有相关用户文档。
