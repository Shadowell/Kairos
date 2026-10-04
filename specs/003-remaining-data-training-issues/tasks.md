# Tasks

## Phase 1: Specification
- [x] T001 核实#3/#5/#8剩余缺陷并完成spec/plan一致性检查，无关键待澄清项。
## Phase 2: US1
- [x] T002 [US1] 为sidecar混合时区、缺失/无效/重复/乱序时间补失败回归，修复crypto.py/crypto_extras.py。
## Phase 3: US2
- [x] T003 [US2] 为训练集horizon权重估计、损失平衡及validation隔离补回归，修复模型/训练配置与bundle记录。
## Phase 4: US3
- [x] T004 [US3] 为精确续采、日期结束边界和历史保护补回归，修复collect.py。
## Phase 5: Integration
- [x] T005 更新已有相关文档，核对32维/v2契约、运行完整回归和真实离线数据/CPU验证，converge后提交推送。

## Phase 6: Convergence
- [x] T006 [US1] 真实adapter不得吞掉ValueError/TypeError/KeyError，覆盖实际adapter→collect错误传播。
- [x] T007 [US2] 权重估计/模型包校验提前拒绝float32下溢，提示提高floor。
- [x] T008 [US3] 严格sidecar失败不推进主行情，支持主行情已齐时单独补齐sidecar；暂存文件不进入parquet发现列表。
- [x] T009 [US3] 将batch失败传递为CLI非零退出并验证成功状态。
