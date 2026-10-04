# Kairos Constitution

## Core Principles

### I. 数据时间一致性
采集和打包必须使用明确的 UTC 边界；因子不得使用未来信息。

### II. 架构兼容
必须维持 24+8=32 个外生因子、现有 pickle 布局和 meta.json 契约。

### III. 可复现修复
Bug 修复必须先有失败用例，再实现并运行针对性回归及 pytest -x。

### IV. 数据与凭据保护
不得提交原始数据、模型、缓存或秘密；不得覆盖用户既有修改及实验产物。

### V. 有限范围
本机只执行离线测试和小规模 CPU 验证；远程训练、部署与付费任务需要相应授权。

## 技术约束
Python 3.10+；保持现有代码风格和 CLI 入口。修改数据链路后运行最小打包验证。

## 开发工作流
使用项目 Spec Kit：specify → clarify → plan → tasks → analyze → implement → converge。
小修复仅保留 spec.md、plan.md、tasks.md，排查过程留在对话中。遵守 AGENTS.md 的 Git 规则。

## Governance
用户当前指令和 AGENTS.md 优先。原则调整必须记录适用范围、日期及版本；完成前核验约束。

**Version**: 1.0.0 | **Ratified**: 2026-10-04 | **Last Amended**: 2026-10-04
