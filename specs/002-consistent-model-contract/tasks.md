# Tasks

## Phase 1: Contracts
- [x] T001 在 specs/002-consistent-model-contract 定义时间、标签、schema、模型包和接口契约。
## Phase 2: US1
- [x] T002 [US1] 修复 kairos/data/prepare_dataset.py 日历切分及 manifest，增加数据契约与回归。
- [x] T003 [US1] 修复 kairos/training/dataset.py 连续窗口、严格外生校验及确定性采样。
## Phase 3: US2
- [x] T004 [US2] 修复 kairos/models/kronos_ext.py loss，训练标签、联合验证、DDP 早停和累积梯度。
- [x] T005 [US2] 在 kairos/training/artifacts.py 实现独立运行和模型包，训练绑定 tokenizer/schema/数据版本。
## Phase 4: US3
- [x] T006 [US3] 新增 kairos/inference.py，共享真实推理和模型契约校验。
- [x] T007 [US3] 修复 kairos/training/backtest_ic.py baseline、标签和统计口径。
- [x] T008 [US3] 接通 kairos/deploy/serve.py 外生模型/分位数，修复 HF 上传契约。
## Phase 5: Verification
- [x] T009 更新现有 README/运维/评测文档并标注旧指标适用边界。
- [x] T010 执行完整回归、CPU smoke、权重加载和端到端契约验收，修复 review/converge 遗漏。

## Dependencies
T001 后数据、训练、回测可按已确定接口并行；T006/T008 依赖共享契约；最后统一 T009/T010。

## Phase 6: Convergence — Tokenizer integration
- [x] T011 根据 FR-002 在 train_tokenizer.py 使用跨 rank 相同的 epoch 采样映射。
- [x] T012 在 train_tokenizer.py 集成独立 Issue #7 的完整批内分块、按样本加权和 no_sync（FR-004），保留 v2 predictor 实现。
- [x] T013 在 tests/test_tokenizer_accumulation.py 验证不整除尾块、跨 rank epoch 与实际双进程同步，核对全量回归。
- [x] T014 修复 Tokenizer 非零 rank 的 best 状态（FR-004）与无需 exog 的主通道兼容，跑真实小模型 CPU smoke；记录 BSQ 分块语义。
- [x] T015 审查发现 Tokenizer 空验证集可被误报为最佳值 0：增加空 loader 和非有限验证损失拒绝回归，满足 FR-004。
