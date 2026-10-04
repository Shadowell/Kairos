# Implementation Plan

## Technical Context
Python/PyTorch/pandas；复用现有 Spec Kit 1.0.6 和 .venv；不修改 vendored Kronos。

## Constitution Check
32 维与数据保护不变；无未来数据；先回归后修复；全部外部发布须遵守项目授权。

## Design and Ownership
- 数据模块：kairos/data/contracts.py 提供 bar_delta(freq)、validate_main_frame(df)、validate_exog_frame(main,exog,columns)、contiguous_starts(index,window,freq)、normalize_window(values,lookback,clip)、log_return_targets(close,anchor,horizon)。prepare_dataset 按日历分块并存 schema/freq；Dataset 兼容三元组，include_targets=True 额外返回 [H] 标签。
- 训练模块：修复 loss，train_predictor 使用锚点 log-return 标签和联合验证；config 运行配置；artifacts.py 保存/加载版本化 kairos_manifest.json 及 tokenizer。
- 推理模块：kairos/inference.py 的 KairosPredictor.from_checkpoint 与 predict_batch(list[main],list[exog]) 返回 [B,H,Q]，权威配置来自 manifest。
- 回测模块：共享推理器、同锚点真值、真实原版 Kronos baseline、分离横截面/时序/pooled 口径和种子报告。
- 服务模块：版本化 bundle 加载、外生输入、分位数输出、严格请求校验；HF 上传保留完整包。

## Validation
各模块先失败用例后实现；独立小模型集成；模型权重加载；CPU smoke；全量 pytest -x、静态检查、敏感内容检查；最后对 spec/tasks converge。根 agent 统一 review、commit 和 push。
