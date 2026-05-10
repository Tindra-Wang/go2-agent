# NP3O Alignment TODO

## Change history

### Round 1 (initial stair-stability shaping)
- Enabled `feet_stumble` weight=-0.05
- Enabled `feet_slide` weight=-0.03
- Strengthened `feet_height_body` weight=-0.08, target_height=-0.26
- Strengthened `correct_base_height` weight=-0.35

### Round 2 (anti-stuck-leg, terrain exposure, robustness) — PARTIALLY REVERTED
- Strengthened `no_fly` weight: -0.5 → -1.0 (fix single-leg-stuck-in-air) ✅ kept
- Strengthened `foot_mirror_up` weight: -0.05 → -0.08 (diagonal-leg coordination) ✅ kept
- ~~Raised `max_init_terrain_level`: 2 → 3~~ ❌ reverted to 2 (caused collapse)
- ~~Strengthened `termination` weight: -1.0 → -2.0~~ ❌ reverted to -1.0 (caused collapse)
- ~~Raised `max_push_vel_xy`: 0.5 → 0.8~~ ❌ reverted to 0.5 (caused collapse)

Lesson learned: terrain difficulty, push strength, and termination penalty must NOT be raised simultaneously. Each of these amplifies the others — harder terrain means more falls, stronger pushes mean more falls, higher fall penalty means the value function collapses when falls are frequent. These must be introduced one at a time, after the policy is already stable on the current difficulty.

### Round 3 (fix circling-at-spawn and edge-stuck — overnight training)
Root cause analysis:
- "出生点转圈": `forward_velocity` (body-frame, weight=1.2) rewards circling; net reward from circling > net reward from climbing stairs with penalties
- "下楼卡住": no penalty for zero progress; staying stuck has zero cost while stepping down triggers multiple penalties

Changes:
- `forward_velocity` weight: 1.2 → 0.3 (eliminate circling attractor — body-frame speed no longer profitable enough to beat progress)
- `progress` weight: 18 → 25 (make world-frame displacement the absolute dominant signal)
- NEW `stall_penalty` weight: -0.5 (penalize "body moving but no world-frame progress" — directly targets both circling and edge-stuck)
- `_upright_gate` descent floor: 0.7 → 0.6 (reduce descent penalty so "attempt to step down" is cheaper than "stay stuck")

## Current status

Eval scores after round 1:

| Terrain | Total | Forward | Time | Pose | Energy |
|---------|-------|---------|------|------|--------|
| pyramid_slope | 77.53 | 99.99 | 78.41 | 58.99 | 50.27 |
| pyramid_slope_inv | 72.69 | 99.93 | 75.35 | 45.72 | 42.55 |
| pyramid_stairs | 67.92 | 95.33 | 66.33 | 34.83 | 47.75 |
| pyramid_stairs_inv | 59.87 | 90.23 | 59.48 | 28.91 | 30.50 |

Remaining issues observed in video:
- One leg gets stuck in the air (frozen), robot stalls
- Uphill stairs: sometimes fails and walks backward, circles, retries from different angle
- Stair ascent still has direct failures/falls

## Priority 1 (next eval): verify round-2 changes

Goal: confirm whether the anti-stuck-leg, curriculum, and robustness changes reduce stair failures.

Check after retraining:
- `pyramid_stairs_inv` fall/abnormal count — should decrease
- `pyramid_stairs_inv` timeout count — should decrease
- `pyramid_stairs` fall count — should decrease
- Whether "one leg stuck in air" still appears in replay
- Whether "fail → circle → retry" still appears in replay
- Whether stair ascent success rate improves

## Priority 2: further strengthen clearance and base-height if stairs still lag

If stair behavior is still unstable after round 2:

- `feet_height_body`: consider raising weight to -0.10 and/or target_height to -0.22
- `correct_base_height`: consider raising weight to -0.5
- `feet_stumble`: consider raising weight to -0.08

Only do this if round-2 changes are not enough by themselves.

## Priority 3: reassess idle-stabilization rewards if timeout/idling remains

These are currently disabled:
- `has_contact`
- `stand_nice`
- `upward`

If, after round-2 changes, the policy still shows:
- stopping before stairs/slopes
- timeout without progress
- local standstill behavior near difficult transitions

Then consider a constrained reintroduction:
- Only enable `has_contact` with a very small weight (e.g. 0.1)
- Gate it more aggressively (require both lin and ang cmd near zero)
- Do NOT re-enable `upward` (free standing reward)

This is lower priority because it can easily reintroduce the old idling attractor.

## Priority 4: domain randomization alignment with NP3O

NP3O uses significantly more randomization:
- `max_push_vel_xy = 1.0` (we are now at 0.8)
- Motor strength/Kp/Kd randomization
- Base mass/CoM randomization
- Lag randomization
- Explicit external disturbances

If stair contact recovery is still fragile after the above priorities:
- Raise `max_push_vel_xy` to 1.0
- Add motor strength randomization if the framework supports it
- Add base mass randomization if the framework supports it

## Priority 5: compare remaining NP3O mismatches

If performance is still capped after the above, review:
- Whether the descent-side posture gating (`_upright_gate` floor=0.7) is too strict or too weak
- Whether the forward-only command regime hurts stair recovery (NP3O uses heading commands)
- Whether additional NP3O-style `collision_up` shaping is needed
- Whether the `feet_slide` implementation matches NP3O's `foot_slide_up` semantics exactly

## Suggested evaluation order

1. Retrain with round-2 changes (no_fly, foot_mirror, curriculum, termination, push).
2. Check `pyramid_stairs` and `pyramid_stairs_inv` curves and replay.
3. If falls remain common, strengthen clearance/base-height (priority 2).
4. If timeout/idling persists, cautiously reassess idle rewards (priority 3).
5. If contact recovery is fragile, increase domain randomization (priority 4).
6. If still capped, deep-dive remaining NP3O mismatches (priority 5).
