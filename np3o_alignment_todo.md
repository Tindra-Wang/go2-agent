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

- [x] HIM-lite history encoder（actor 端）已落地。
  - `agent_diy/model/model.py` 新增 `HistoryEncoder`：history_len*proprio → MLP → latent；actor 输入由 `[base, latent]` 拼接，critic / cost_critic 仍用未增广的 critic_obs（NP3O 对称：critic 是特权观测，不需要历史）。
  - `agent_diy/agent.py` 通过 `stage.use_history_encoder/history_len/history_latent_dim/history_encoder_dims` 注入；评估端在 `Agent` 内自维护 `_eval_history` 并在 `exploit/reset` 中同步推进/清空，与训练端行为一致。
  - `agent_diy/workflow/train_workflow.py` 维护 `history_buf`：每步用「时刻 t 的 raw proprio」推进，并对 dones 行清零；存入 `RolloutStorage` 的 obs 即为增广后的 actor 输入。
  - 默认 `history_len=10` 对齐 NP3O `Go2ConstraintHimRoughCfg.env.history_len`；如需关闭设 `use_history_encoder=False` 即可回退。

- [ ] HIM contrastive / estimator 损失（teacher-student、imi-loss、Barlow-Twins）。
  - 仍未实现：NP3O `actor_student_backbone` 中的对比学习头与 imitation 优化器；当前为「end-to-end，无 contrastive loss」的 HIM-lite。
  - 后续可按 NP3O `algorithm/np3o.py::imi_flag/imitation_learning_loss` 接入。

- [x] Observation parity：actor 端补齐历史；critic 端保持特权（lin_vel + effort）。
  - 详见 P2.1。当前缺 nav latent / scan encoder，仍记为「NP3O-lite」。

- [x] 多代价：3 具名 cost 已对齐 NP3O。
  - `StageConfig.num_costs = 3`，`cost_names = ["dof_pos_limits", "torque_limit", "dof_vel_limits"]`，`cost_d_values = [0.0, 0.0, 0.0]`。
  - `agent_diy/workflow/train_workflow.py::_compute_native_costs` 直接从 `env.scene["robot"].data` 计算三具名 cost：
    - `dof_pos_limits = Σ clamp(超出 soft 极限的部分)`
    - `torque_limit  = Σ clamp(|τ| - τ_lim*soft_τ, min=0)`
    - `dof_vel_limits = Σ clamp(|q̇| - q̇_lim*soft_qd, min=0, max=1)`
  - 由 `cost_scale ≈ 0.02` 缩放到与 NP3O `cost*dt` 同量级；显式 `infos['costs']` 优先级仍最高，env-native 次之，最后才走 termination/episode 兜底。
  - `cost_source_id = 4.0` 对应 `env_native`，可在监控里直接观察是否走到了 NP3O 路径。

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
