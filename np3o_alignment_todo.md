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

- [ ] Reconcile cost semantics and threshold scale.
  - Make sure `cost_d_values` matches the same scale as the environment-provided cost returns.
  - Revalidate `termination_as_cost` usage in `agent_diy/conf/conf.py:67`.

- [ ] Decide whether to keep adaptive penalty updates or restore NP3O-style scheduled growth.
  - Current local behavior: `agent_diy/algorithm/algorithm.py:318`.
  - NP3O reference behavior: fixed growth schedule in `LocomotionWithNP3O/algorithm/np3o.py:166`.
  - Choose one intentionally rather than drifting between both designs.

- [ ] Review timeout cost bootstrapping against NP3O.
  - Current local timeout handling in `agent_diy/workflow/train_workflow.py:330` uses `cost_values`.
  - Compare and, if desired, align with NP3O cost target construction.

## P2: architecture and observation parity work

- [ ] Decide whether to stay NP3O-lite or move toward full NP3O/HIM architecture.
  - Current local model is still simple actor/critic/cost-critic MLP in `agent_diy/model/model.py:35`.
  - NP3O reference includes history encoder, teacher/student structure, privileged latent, imitation-related modules.

- [ ] Review observation parity with NP3O.
  - Current policy obs path: `agent_diy/feature/policy_observation_process.py:9`.
  - Current critic obs path: `agent_diy/feature/critic_observation_process.py:9`.
  - Decide whether to add history features, privileged inputs, or scan/latent encoders.

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
