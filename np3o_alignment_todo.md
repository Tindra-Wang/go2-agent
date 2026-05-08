# NP3O alignment TODO for `agent_diy`

## P0: current status

- [x] Fix `RewardProcess` import startup failure.
  - `agent_diy/feature/__init__.py` now lazy-loads `RewardProcess` so agent startup is no longer blocked when only `ActData` is needed.

- [x] Fix evaluation action batching bug.
  - `agent_diy/agent.py` exploit path is now aligned with `agent_ppo` and returns batched actions correctly in eval.

- [x] Add rollout/eval diagnostics used during bring-up.
  - Rollout logs now print `infos` keys/shapes and cost source for quick environment inspection.

## P1: highest-impact NP3O gaps still remaining

- [ ] Replace heuristic cost derivation with authoritative environment costs.
  - Current local path still depends on `_derive_costs()` in `agent_diy/workflow/train_workflow.py:221`.
  - Target NP3O behavior is env-native per-step `costs` from `env.step(...)`.
  - `require_explicit_costs` now exists in `agent_diy/conf/conf.py`; enable it once env infos reliably provides per-env costs.
  - Current investigation result: the visible env only exposes `infos = {'all_done': bool}` in rollout logs, so this item is blocked outside this repo and is deferred for now.

- [x] Reconcile cost semantics and threshold scale.
  - Added `cost_scale` (default `0.02` ≈ NP3O `cost*dt`) in `agent_diy/conf/conf.py`, applied in `_derive_costs` for derived (termination/episode-summary) sources only; explicit env-provided costs are passed through.
  - `cost_d_values` 现与 `(1-gamma) * cost_return` 同尺度，文档已更新。Default `[0.0]` matches NP3O `Go2ConstraintHimRoughCfg.costs.d_values`.
  - `termination_as_cost` 仍为兜底路径，使用前会乘 `cost_scale` 后写入 `costs[:, 0]`。

- [x] Decide whether to keep adaptive penalty updates or restore NP3O-style scheduled growth.
  - Default mode now `penalty_mode = "scheduled"`：`_update_penalty_scheduled` 复刻 `LocomotionWithNP3O/algorithm/np3o.py::update_k_value`：每个 learn() 调用一次，`k *= growth_rate ** iter`，由 `penalty_max` 上限截断。
  - 若需回到反馈式行为，在 stage config 设置 `penalty_mode = "adaptive"` 即可，原有 `penalty_lr/decay` 仍生效。
  - Decision recorded：默认 NP3O 调度，`adaptive` 仅作为可切换的兜底实验项。

- [x] Review timeout cost bootstrapping against NP3O.
  - 增加 `timeout_cost_bootstrap` 配置：`"value"`(默认) 用 `cost_values * timeout_mask` 自举（与 reward 路径对称、更符合 GAE 截断处理）；`"self"` 复刻 NP3O 原版 `costs += gamma * costs * timeout` 行为以便对照。
  - 若环境给出与 NP3O 一致的“原始 cost+dt”，把该项设为 `"self"` 即可严格对齐参考实现。

## P2: architecture and observation parity work

- [ ] Decide whether to stay NP3O-lite or move toward full NP3O/HIM architecture.
  - Current local model is still simple actor/critic/cost-critic MLP in `agent_diy/model/model.py:35`.
  - NP3O reference includes history encoder, teacher/student structure, privileged latent, imitation-related modules.
  - Status: 暂保持 NP3O-lite（无 HIM 编码器/无 teacher-student/无 imi loss），改造工作量大且依赖多步 obs/历史采集、特权 critic 输入与 imitation 优化器，超出当前对齐迭代的 P0~P1 范围。等 P1 验证稳定后再启动。

- [ ] Review observation parity with NP3O.
  - Current policy obs path: `agent_diy/feature/policy_observation_process.py:9`.
  - Current critic obs path: `agent_diy/feature/critic_observation_process.py:9`.
  - Decide whether to add history features, privileged inputs, or scan/latent encoders.
  - Status: 与 P2 模型项绑定，当前 obs 已对齐 Isaac Lab 默认 (proprio45+scan256+critic privileged)，缺历史拼接与 nav latent；建议随 HIM 改造一起规划。

- [ ] Decide whether to extend from single-cost to multi-cost constraints.
  - Current local config still uses `num_costs = 1` in `agent_diy/conf/conf.py:67`.
  - NP3O reference commonly uses multiple named costs.

## P3: rollout / training loop parity

- [ ] Review rollout storage details against NP3O.
  - Compare minibatch permutation behavior, recurrent hooks, and numerical sanitization in `agent_diy/feature/definition.py:157` and `agent_diy/feature/definition.py:213`.
  - Keep only the differences that are intentional for Kaiwu integration.

- [ ] Review training workflow differences versus NP3O runner.
  - Local workflow lives in `agent_diy/workflow/train_workflow.py:115`.
  - NP3O uses a different iteration-driven runner design.
  - Decide whether any scheduler/logging/save behavior should be aligned further.

## P4: verification and diagnostics

- [ ] Run a clean startup verification for both train and eval.
  - Confirm agent import works.
  - Confirm model load works.
  - Confirm first env step succeeds in eval.

- [ ] Run a short training verification.
  - Confirm `violation_loss` is no longer pinned at zero.
  - Confirm `k_value` changes as intended.
  - Confirm `cost_value_loss` stays finite.

- [ ] Validate cost-path observability.
  - Confirm logs show `cost_source`, `raw_mean_violation`, `positive_mean_violation`, `k_value_mean`, and `k_value_max`.

## Suggested execution order

1. Fix `RewardProcess` import failure.
2. Fix eval action batching bug.
3. Replace heuristic costs with env-native costs if available.
4. Reconcile penalty schedule choice.
5. Revisit timeout cost bootstrap.
6. Decide how far to go on architecture/observation parity.
7. Consider multi-cost support.
8. Re-run train/eval verification.
