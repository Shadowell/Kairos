# Implementation Plan

## Technical Context
Python 3.12 隔离环境（项目最低 3.10）、pandas、numpy、pyarrow、pytest；离线合成行情。
范围：kairos/data/prepare_dataset.py、kairos/data/markets/crypto.py、tests/test_data_windows.py。

## Constitution Check
固定 32 维、不改因子、不用未来数据；不触及历史产物；仅提交本轮文件；不运行训练或部署。

## Design
1. 分离范围分隔符与 ISO 时间中的冒号；使用 UTC 归一化端点验证先后关系。
2. 日期结束端转换到下一日零点前，显式时刻保持包含；采集器以 UTC 解析，裸日期结束端为次日零点（交易所接口使用半开区间）。
3. amount 在构造因子之前逐行补齐。
4. CLI 在读数据前解析并检查范围重叠；time 模式三段互斥，interleave 的合并 fit 段不得与 test 重叠。
5. 无输入、任一 split 无数据时抛出明确 CLI 错误，延迟创建输出目录及写文件。
6. 更新 README 和 AGENTS 中被修复行为取代的旧说明。

## Validation
回归先红后绿；pytest -x；真实 parquet 的 CLI 成功与失败路径；git diff --check。
依赖下载不参与验收；不将本轮测试结果推断为模型收益或线上服务可靠性。
