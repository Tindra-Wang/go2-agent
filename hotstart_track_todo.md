# Hot-start Track Finetune TODO

## Stage 1 — config-only 稳定化（本次直接落地）
- [ ] 调整 `agent_diy/conf/train_env_conf_track_nav.toml` 的命令采样分布，降低 stop/急转/侧移带来的分布突变
- [ ] 提高 locomotion 锚点：`track_lin_vel_xy`、`track_ang_vel_z`
- [ ] 提高动作平滑与步态约束：`action_rate`、`action_smoothness`、`feet_slide`、`foot_mirror_up`
- [ ] 降低过强导航 shaping：`heuristic_navigation`、`deadend_escape`
- [ ] 保留 `reach_goal`，避免一次性改 reward 逻辑代码
- [ ] 对比修改前后的训练曲线与视频表现

## Stage 1.5 — 窄路防挂墙 / 脱困后纠偏（本次直接落地）
- [x] 提高 `feet_stumble`，减少脚挂墙边/台阶边试探
- [x] 提高 `obstacle_evasion` 惩罚，减少前方受阻时仍然硬顶
- [x] 降低 `heuristic_navigation` 与 `deadend_escape`，减弱脱困时的大幅硬扭
- [x] 提高 `wall_proximity_brake`，让窄路入口更早减速

## Stage 1.5b — 保留 anti-stuck，放松 anti-turn（本次直接落地）
- [x] 保留 `feet_stumble = -0.2` 与 `obstacle_evasion = -1.5`
- [x] 回调 `heuristic_navigation`，恢复部分重新找路能力
- [x] 回调 `deadend_escape`，避免脱困驱动力被压得过低
- [x] 回调 `wall_proximity_brake`，避免窄路入口减速过强

## Stage 2 — 增加 true goal progress reward
- [ ] 在 `agent_diy/feature/reward_process.py` 中新增或强化基于 `dist_to_goal(t-1) - dist_to_goal(t)` 的密集奖励
- [ ] 将 heuristic 型奖励降为辅助项
- [ ] 重新平衡 nav / locomotion reward 比例

## Stage 3 — curriculum / finetune 节奏优化
- [ ] 根据 Stage 1 曲线决定是否暂时关闭 `curriculum`
- [ ] 必要时降低 `NavConfig.lr`
- [ ] 制定 best checkpoint 回滚继续训策略

## Stage 4 — 反退化专项修正
- [ ] 根据录像排查 circling / wall-hugging / flailing / stair-edge stuck
- [ ] 仅保留对完成率和姿态稳定性有正贡献的 shaping
