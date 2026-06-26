# RL Wheel Balance Learning Notes

这份笔记用于补齐当前 HopperTrex 两轮平衡项目最需要的基础知识。重点不是把所有理论一次性学完，而是建立一个能指导实验判断的知识框架：为什么策略会失败、指标怎么看、viewer 看到的现象如何回到 reward / termination / geometry / control design 上解释。

当前阶段的核心目标是：

- 固定腿姿态，只用两个主轮完成原地动态平衡。
- 保持纯主轮接地，避免 thigh / calf / chassis 等非轮结构接地。
- 从 clean balance 进入 robust stationary balance，即从小初始扰动中恢复。
- 暂时不加入前进速度命令、不开放腿控制、不上复杂地形。

---

## 1. RL 与 PPO 基础

### 1.1 MDP 是什么

强化学习通常被建模为马尔可夫决策过程（Markov Decision Process, MDP）：

```text
state_t -> action_t -> reward_t, state_{t+1}, done_t
```

在机器人仿真里，可以这样对应：

- `state`: 仿真器里的完整物理状态，例如 root pose、关节位置、速度、接触状态。
- `observation`: 策略真正能看到的输入，通常是 state 的一部分或经过处理后的特征。
- `action`: 策略输出给控制器或 actuator 的命令。
- `reward`: 当前动作是否接近目标的数值反馈。
- `done`: episode 是否终止，例如摔倒、触地、超时、NaN。

重要区别：

- `state` 不一定直接给 policy。
- `observation` 设计不好，policy 可能根本看不到恢复平衡所需的信息。
- `reward` 只是训练信号，不等于真实目标本身。
- `done` 会强烈影响数据分布，过早终止会让策略学不到恢复动作。

### 1.2 Observation / Action / Reward / Done

本项目里最关键的是这四件事是否和真实目标一致。

`observation` 要回答：策略是否知道自己快倒了？

常见内容包括：

- base orientation 或 gravity vector。
- base angular velocity。
- wheel velocity。
- joint position / velocity。
- previous action。
- command velocity。

`action` 要回答：策略到底能控制什么？

当前 clean balance 阶段使用 1 维耦合轮控：

```text
action = shared wheel command
left wheel = action
right wheel = action
legs = fixed
```

这样做的目的不是最终形态，而是先把问题压缩为类似平衡车的基础任务，避免四条腿关节引入太多探索自由度。

`reward` 要回答：策略做对了吗？

本项目最关键的 reward 设计是区分：

```text
真正两轮动态平衡
```

和：

```text
靠 thigh / calf / chassis 低姿态触地苟活
```

如果 reward 只奖励 alive、upright 或 orientation，策略可能找到“看起来没倒，但其实靠非轮接触支撑”的局部解。

`done` 要回答：什么行为必须判失败？

对于当前阶段，以下终止项很重要：

- `bad_orientation`: 姿态偏差过大。
- `root_too_low`: root 高度过低。
- `non_wheel_ground_contact`: 非轮结构接地。
- `nan_detection`: 数值异常。
- `time_out`: 成功撑到 episode 结束。

### 1.3 PPO 在做什么

PPO（Proximal Policy Optimization）是一种 on-policy actor-critic 算法。它同时训练两个网络：

- actor / policy: 根据 observation 输出 action 分布。
- critic / value function: 估计当前 observation 未来能拿到多少 reward。

PPO 的核心思想是：每次更新 policy 时，不要让新策略离旧策略太远。这样可以减少一次梯度更新把策略“训崩”的概率。

常见 loss：

- `surrogate loss`: 让 actor 更倾向于产生高 advantage 的 action。
- `value loss`: 让 critic 更准确预测未来回报。
- `entropy loss`: 鼓励探索，避免 action 分布过早塌缩。

可以粗略理解为：

```text
total loss = policy improvement + value prediction + exploration regularization
```

### 1.4 Advantage 和 episode length 怎么看

Advantage 表示某个 action 相比 critic 预期表现更好还是更差。PPO 根据 advantage 调整 action 概率。

`Mean episode length` 是本项目非常重要的指标：

- 如果接近最大 episode 长度，说明策略至少能活到 timeout。
- 如果一直很短，例如 20-30 steps，说明策略几乎刚开始就失败。
- 如果 reward 提高但 episode length 仍短，要警惕 reward 被某个局部项误导。

当前 clean balance 的成功标准不是只看 mean reward，而是同时看：

```text
Mean episode length 接近上限
time_out 高
bad_orientation 低
root_too_low = 0
non_wheel_ground_contact = 0
viewer 中只有主轮接地
```

### 1.5 Entropy、learning rate 与 seed 敏感性

RL 训练对随机性很敏感，尤其是这种早期必须先发现“能站住一点”的平衡任务。

seed 影响：

- 网络初始化。
- 环境 reset 随机数。
- action 采样。
- batch 采样顺序。
- GPU 并行数值细节。

所以同样配置可能出现：

```text
seed A 学会站立
seed B 一开始失败，然后一直在失败数据里打转
```

`entropy_coef` 太低时，策略可能过早收敛到坏动作；太高时，又可能一直乱动。`learning_rate` 太高时，成功策略也可能被更新破坏；太低时，训练很慢或卡住。当前更推荐用 clean checkpoint resume，再用较低学习率做鲁棒微调。

---

## 2. 两轮倒立摆与轮腿机器人基础

### 2.1 为什么它像倒立摆，但又不只是倒立摆

两轮平衡车可以近似看成倒立摆（inverted pendulum）：

```text
轮子在地面提供水平加速度
机体重心在轮轴上方
控制轮速让接地点移动到重心下方
```

但 HopperTrex 不是理想倒立摆，原因包括：

- 它有腿，腿的几何会改变重心和碰撞体位置。
- 它有多个 collision geom，不是只有两个轮子会接触地面。
- 轮轴高度、root 高度、腿姿态决定谁先接地。
- 轮腿机器人有 joint limits、actuator limits、摩擦、接触刚度等现实约束。

所以“腿固定不动”不等于“腿不影响任务”。腿即使不控制，也会通过质量分布和碰撞几何影响平衡。

### 2.2 腿姿态为什么会影响平衡

腿姿态至少影响三件事：

1. 主轮是否是最低接地点。
2. 机体重心相对轮轴的位置。
3. 非轮结构距离地面的安全间隙。

如果 calf / thigh 比主轮更低，或者稍微倾斜后更容易先接地，策略会发现一种错误捷径：

```text
降低 root height -> 非轮接地 -> 用机体/腿支撑 -> reward 还不一定立刻变差
```

这不是动态平衡，而是低姿态支撑。它在 TensorBoard 里可能看起来“不算太差”，但 viewer 里能直接看出来。

### 2.3 主轮最低接地点

干净的两轮平衡需要满足：

```text
主轮 collision geom 是最低有效接触点
非轮 collision geom 与地面有足够 clearance
root height 不进入低姿态危险区
```

这就是为什么 `fixed_wheel_sweep.py` 很有价值。它不是训练脚本，而是几何诊断工具，用来回答：

- 哪些 geom 在不同 root height / pitch / roll 下先接地？
- 主轮接触是否稳定？
- thigh / calf / chassis 有没有潜在触地风险？

### 2.4 重心、轮轴与主动恢复

动态平衡不是“完全不动”，而是不断主动修正：

```text
机体向前倒 -> 轮子向前加速 -> 接地点追到重心下方
机体向后倒 -> 轮子向后加速 -> 接地点追到重心下方
```

如果策略只会维持某个低姿态，它不一定具备恢复能力。下一阶段 robust balance 的意义就是验证：

- reset 后有轻微 roll / pitch 扰动时能否回正。
- 有轻微 root velocity 时能否消除漂移。
- 只有主轮接地时能否主动恢复，而不是用非轮结构“刹住”。

---

## 3. MuJoCo / MjLab 实践知识

### 3.1 MJCF 与 collision geom

MuJoCo 使用 MJCF 描述机器人模型。一个 body 可以有多个 geom：

- visual geom: 主要用于显示。
- collision geom: 参与接触和碰撞。

训练时真正影响物理的是 collision geom。viewer 里看起来“腿没有碰地”，不代表 collision geom 没接触；要结合 contact sensor 或诊断脚本确认。

本项目里要重点关注：

- wheel geom 是否正确参与接触。
- calf / thigh / chassis 是否会接地。
- geom 的 group / name / collision 设置是否能被 sensor 正确识别。

### 3.2 Contact sensor 的作用

contact sensor 用来把物理接触转换成 reward / termination 能用的信息。

当前任务需要区分：

```text
wheel-ground contact
non-wheel-ground contact
```

原因是两者语义完全相反：

- 主轮接地是必要条件。
- 非轮接地是失败模式。

如果不区分，策略就可能把所有接触都当作可用支撑。

### 3.3 ManagerBasedRlEnv 的模块

MjLab 的 `ManagerBasedRlEnv` 通常把任务拆成多个 manager：

- action: 策略输出如何映射到仿真控制。
- observation: 给 policy 的输入。
- reward: 每步奖励。
- termination: episode 终止条件。
- event: reset 或阶段性事件。
- command: 速度命令、目标姿态等任务命令。

这套结构的好处是清晰，坏处是调试时要知道“问题属于哪个 manager”。

例如：

- 机器人刚 reset 就姿态不对，多半查 event / initial state / asset。
- policy action 维度不对，查 action manager。
- reward 高但 viewer 不对，查 reward 是否漏掉失败模式。
- episode 很短，查 termination 是否过严或几何真的有问题。

### 3.4 Event 与 reset 随机扰动

clean balance 阶段 reset 应该尽量干净，让策略先学会基本站立。

robust balance 阶段则加入小扰动：

```text
roll/pitch: ±2 deg
root linear velocity x: ±0.05 m/s
root angular velocity roll/pitch: ±0.10 rad/s
```

这类扰动通常通过 event manager 加入，例如：

- `reset_scene_to_default`
- `reset_root_state_uniform`

注意：扰动不应该一开始太大。否则策略还没掌握基本站立，就会被大量失败 episode 淹没。

---

## 4. 本项目经验总结

### 4.1 旧失败模式

之前的主要失败不是 PPO 代码错，也不是 `uv` 环境错，而是任务定义允许了错误解：

```text
非轮结构接地支撑
低姿态苟活
reward 指标看似改善
viewer 中不是真正两轮平衡
```

这说明机器人 RL 里不能只信 mean reward。reward 是训练信号，不是最终真相。

### 4.2 几何诊断比盲目调参更重要

之前试过 pulse、teacher gain、contact penalty、hard termination 等小修小补，但没有根治。原因是底层几何和控制结构仍允许失败模式。

更有效的路径是：

```text
先确认主轮是最低接地点
再固定腿姿态
再限制 action space
再设计 clean wheel support reward / termination
最后多 seed 验证
```

这个顺序比盲目加大训练时长更可靠。

### 4.3 固定腿姿态修正

当前 clean balance 的关键工程判断是：腿虽然不动，但必须放在一个不会抢占接触的姿态。

修正后的思想是：

- 让主轮成为稳定最低接触点。
- 提高非轮结构离地间隙。
- 保持 root height 进入合理平衡区。
- 避免 calf / thigh 在倾斜时先碰地。

### 4.4 1 维耦合轮控

当前阶段把 action 简化为 1 维耦合轮控，是为了先解决最小闭环：

```text
只训练前后动态平衡
不训练转向
不训练侧向移动
不训练腿部动作
```

这让问题更接近经典平衡车，也更容易诊断：如果这个版本都站不住，就不应该急着开放腿或上速度跟踪。

### 4.5 Viewer 验证优先于 reward

当前项目最重要的验收原则：

```text
TensorBoard 指标只能筛选候选模型
viewer 里的接触和姿态才是最终判断
```

必须看：

- 是否只有两个主轮接地。
- thigh / calf / chassis 是否离地。
- root height 是否维持合理。
- 倾斜后是否主动回正。
- 是否只是靠卡住或低姿态支撑。

---

## 5. 训练工作流

### 5.1 Smoke test

每次换机器、换代码、换任务，都先跑 smoke test：

```powershell
uv run python src\hoppertrex_mjlab\assets\spawn_robot.py --steps 5
uv run python src\hoppertrex_mjlab\scripts\zero_agent.py --device cuda:0 --num_envs 1 --max_steps 100
uv run python src\hoppertrex_mjlab\scripts\random_agent.py --device cuda:0 --num_envs 1 --max_steps 100
```

目标不是性能，而是确认：

- asset 能加载。
- env 能创建。
- action / observation 维度正确。
- 没有 NaN。
- GPU / MuJoCo / MjLab 能跑通。

### 5.2 Fixed-wheel sweep

在改腿姿态或碰撞体后，先跑几何诊断：

```powershell
uv run python src\hoppertrex_mjlab\scripts\fixed_wheel_sweep.py --steps 150
```

重点看：

- wheel contact 是否稳定。
- non-wheel contact 是否为 0。
- 哪些高度/角度进入危险区。

### 5.3 Clean training

clean balance 先不加扰动：

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py --env.scene.num-envs 256 --agent.max-iterations 500 --agent.save-interval 50 --agent.seed 1 --agent.run-name clean_wheel_seed1
```

验收标准：

- 多个 seed 都能站。
- `non_wheel_ground_contact = 0`。
- `root_too_low = 0`。
- viewer 中只有主轮接地。

### 5.4 Multi-seed 验证

单个 seed 成功不能说明任务稳定。至少要跑：

```text
seed1
seed2
seed3
```

如果有些 seed 成功、有些失败，说明 reward / termination / reset / exploration 还不稳定。当前 clean wheel 阶段已经从 seed 敏感走到多个 seed 都能平衡，这是一个重要里程碑。

### 5.5 Checkpoint replay

训练完成后必须回放：

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\play.py --agent trained --checkpoint-file <path-to-model.pt> --num-envs 1 --device cuda:0
```

不要手猜路径，优先用真实 run 目录里的 checkpoint。

### 5.6 Robust task resume

下一阶段从 clean checkpoint 继续训练 robust balance：

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Robust-v0 --env.scene.num-envs 256 --agent.max-iterations 500 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*clean_wheel.*seed1.*" --agent.load-checkpoint "model_499.pt" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.002 --agent.run-name robust_init_seed1
```

为什么从 checkpoint resume：

- action 维度没变。
- observation 维度没变。
- clean policy 已经会站。
- robust 只是要求它从小扰动中恢复。

这比从随机初始化直接训扰动环境更稳。

---

## 6. 推荐阅读清单

### 6.1 强化学习基础

- Sutton & Barto, *Reinforcement Learning: An Introduction*。建议先读第 3 章 MDP、第 9-13 章函数逼近和 policy gradient。
- Schulman et al., 2017, *Proximal Policy Optimization Algorithms*。PPO 原论文，重点看 clipped objective 的动机。
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。比原论文更适合作为入门解释。

### 6.2 物理仿真与 MuJoCo

- Todorov et al., 2012, *MuJoCo: A physics engine for model-based control*。理解 MuJoCo 的定位：接触丰富、适合机器人控制。
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)。重点查 MJCF、geom、contact、actuator。
- `mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning`。用于理解本项目框架的 manager-based 任务组织方式。

### 6.3 腿式机器人 RL

- Rudin et al., 2022, *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*。理解大规模并行仿真为什么能让 legged RL 快速训练。
- Hwangbo et al., 2019, *Learning agile and dynamic motor skills for legged robots*。经典 sim-to-real 腿式机器人 RL 工作。
- Tan et al., 2018, *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*。理解 domain randomization 和 sim-to-real 基础。

### 6.4 轮腿和平衡控制

- *First-principles-driven wheel-legged robot control via deep reinforcement learning*。重点看如何结合一阶原理、平衡控制和深度强化学习。
- 检索关键词：`wheel-legged robot reinforcement learning balance control`。
- 检索关键词：`two wheeled inverted pendulum LQR control`。
- 检索关键词：`wheeled biped robot dynamic balance reinforcement learning`。

### 6.5 控制基础补充

- Inverted pendulum（倒立摆）建模。
- Linear Quadratic Regulator（LQR）。
- PID 与状态反馈控制。
- Center of Mass（CoM）、Zero Moment Point（ZMP）、support polygon。

这些不需要一次性深入，但要知道它们在回答什么问题：

```text
系统为什么会倒？
控制输入如何改变重心和接地点关系？
什么状态可控，什么状态不可控？
```

---

## 7. 当前阶段的判断准则

短期不要急着加入速度命令或腿控制。更合理的顺序是：

1. clean fixed-leg two-wheel balance。
2. robust stationary balance with small reset perturbations。
3. larger perturbations。
4. simple forward velocity command。
5. turn / yaw command。
6. leg action 或 terrain。

每一阶段都要同时通过：

- TensorBoard 指标。
- 多 seed 复现。
- checkpoint replay。
- viewer 接触验证。

如果 viewer 和 reward 冲突，以 viewer 为准。

