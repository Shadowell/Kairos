# Tasks

## Phase 1: Setup
- [x] T001 在 .specify/ 建立项目约束和规格，并配置隔离测试环境 .venv/。

## Phase 2: US1 时间窗口
- [x] T002 [US1] 在 tests/test_data_windows.py 添加日期、ISO 时间、倒序、UTC 时区与零点边界失败回归。
- [x] T003 [US1] 修复 kairos/data/prepare_dataset.py 范围解析/切片及 kairos/data/markets/crypto.py UTC 转换。

## Phase 3: US2 打包可靠性
- [x] T004 [US2] 在 tests/test_data_windows.py 添加 amount 缺失、空输入、无效/重叠范围及输出保护回归。
- [x] T005 [US2] 修复 kairos/data/prepare_dataset.py 补齐逻辑及 CLI 写入前检查。

## Phase 4: Verification
- [x] T006 更新 README.md、AGENTS.md 的日期契约；运行全部测试与离线 CLI 打包，检查 diff 并逐条核对 spec.md。

## Dependencies
T001 → T002 → T003 → T004 → T005 → T006。单会话按顺序执行，先失败用例后实现。
