# C1 强基线上的残差恢复 Pilot v1

协议确认：2026-10-07；复审日：2026-10-20（Asia/Shanghai）。
本研究只使用仿真，不涉及实机和楼梯。由本机 agent 实现、测试、自查、发布和分析；用户在机房执行命令、检查 viewer、回传结果。审查方式为 **self review，不是独立审计**。

## 1. 研究边界

问题：在固定 C1 candidate24 经典控制栈之上，从零残差输出初始化的 PPO 是否能改善平地扰动恢复而不损害原有控制能力？C1 是现有较成熟基线，不声称最优。旧 E6 仅作为历史背景，不能与本次结果混算。

旧 E6 的三份精确哈希标定及 Stage2 载体未找到。本轮不复原、不改写旧 checkpoint，也不声称在线重现 12.42%。原 evidence bundle、Stage0–5 和楼梯协议保持不变。

机器可读协议和**完整解析配置**分别位于同目录下 artifacts/recovery_c1_pilot_v1/protocol.json、resolved_config.json。代码发布 SHA、协议哈希、配置哈希、五工件哈希共同绑定；改变上述内容需要新版本、记录原因，不能继续沿用旧授权。

## 2. 固定设计

- Task：HopperTrex-Recovery-C1-Pilot-v1；不加入旧阶段晋级链。
- C：完整冻结 C1 五工件，残差为零。
- H：同一 C1，加六通道有界残差。动作顺序与尺度继承 Stage5，尺度为 (0.5, 0.3, 0.035, 0.035, 0.035, 0.035)。
- Actor/critic：新建 128–128 ELU MLP，34 维实际观测；不增加特权观测。新 optimizer；actor 输出层全零。
- 初始 Gaussian std 为 (0.15, 0.10, 0.05, 0.05, 0.05, 0.05)。初始**确定性均值**等于经典路径，不声称随机采样动作为零。
- 继承 Stage5 全部奖励和已实现事件。训练事件实际为两项 reset 加 push_robot；play 无定时 push。randomization_level=2 不等于实现了质量／摩擦随机化。
- PPO 使用解析配置固定值：学习率 0.001、adaptive schedule、clip 0.2、entropy 0.005、5 epochs、4 minibatches、gamma 0.99、lambda 0.95、desired KL 0.01。
- 正式 seeds 11、12、13；256 envs、24 steps/update、每 seed 1000 updates，分别为 6,144,000 transitions。未计为全流水线独立复现，因为本轮共享冻结的经典标定。
- 只评最终 model_999.pt；每完成 100 次更新保存一次，因此保存编号为 99、199、…、999。RSL-RL 默认零基编号的其他保存调用被过滤。没有 K-best 或自动延长。
- 通用 train.py 不得启动本任务。只允许带基线及人工放行记录的专用入口。中断不自动 resume。

## 3. 恢复评估

每个 reset 仅一个扰动，300 步稳定、300 步观察。50 Hz 控制、5 ms 物理步长。
中心姿态为冻结 C1 范围的算术中心；静止命令。扰动向世界坐标 vx 和 pitch-rate 分量添加同号速度增量；单位分别为 0.04 m/s、0.06 rad/s。

| 条件 | 用途 |
|---|---|
| 4× | 较弱扰动；训练幅度内敏感性，不声称 OOD |
| 8× | 预先指定主终点 |
| 12× | 额外压力测试；不能事后删掉，不作为训练前放行条件 |

每个条件 4 批 × 32 envs = 128 trials，每批前半正向、后半负向。开发 seeds 211–214，留出 seeds 911–914。训练、调试和选择不查看留出集。每批在 C/H 两臂重新设置 RNG 和控制步计数，记录实际初始相对位置、姿态、根速度、关节状态和哈希，逐 trial 校验 pairing ID。重复初始状态数量如实报告，不把重复状态当新的随机性。

健康范围：|vx error|≤0.06 m/s、|yaw-rate error|≤0.08 rad/s、|height error|≤0.015 m、|pitch error|≤0.04 rad。连续满足 25 步，记录该区段的起始时刻。

任何稳定／观察段内的 termination、非轮接触或自动 reset 均锁存失败；观察期未恢复也失败。失败恢复分数为 6 秒，同时单独报告失败率；不得只平均成功 trial。自动 reset 前读取安全信号，重置后的健康姿态不能覆盖失败。

不对本平地实验额外施加 R0 的“任何子步双轮无支撑即失败”语义；不把这里的资格解释成 R0 资格或安全证明。

轨迹包含误差、健康标记、动作、实际残差、轮目标、执行器力、根状态及累计失败标记。峰值执行器力和控制用量不等同于电气能耗。分母为零时收益为 null，不制造百分比。

## 4. 基础能力与判据

C1 十五格复用已有 evaluation_cells、REGISTERED_CAPS 和 aggregate_candidate：九个姿态节点静止，中心／两角点 ±0.05 m/s；100 步稳定加 200 步测量；16 envs。C/H 都用同一采集器，无重新辨识或调参。

长时保留复用原 integrated suite：正反直行、原地 yaw、组合命令、姿态和综合控制，32 envs、3000 steps、300 warmup、800 window。原数值判据及出处不改变。重复使用相同的原定义不等于声称复现了历史实验。

训练放行需要：
1. M0 CUDA 环境、依赖、注册和五工件校验通过；
2. 本轮 C1 十五格与 integrated 保留基线通过；
3. 开发 4×/8× 全部恢复，且没有硬失败；
4. 完成固定版本的 viewer 会话，用户明确给出 PASS 和观察；
5. 本机自查记录绑定上述两个结果 manifest 后，才能创建训练授权。

三 seed 完成后，续投筛选要求：8× 至少两 seed 正改善、平均相对改善≥10%；4× 平均不退化；4×/8× 没有硬失败；保留能力通过；12× 数据完整；viewer 无反证。三个 seed 只是 pilot，不是统计充分性保证。

## 5. 操作接口与状态流

入口：scripts/run_recovery_c1_pilot.ps1。必须提供 Phase、ExpectedGitSha、CampaignRoot；Train/Evaluate 必须明确单个 Seed。内部不执行 git pull。

| Phase | 输出目录 | 前置 |
|---|---|---|
| Validate | validate | 新环境，固定代码／依赖／工件；不正式训练 |
| Baseline | baseline | M0 完整 CUDA 结果 |
| Baseline + Viewer | baseline_viewer | baseline 已完整，使用零策略；不代表用户已 PASS |
| Train | train_seed11 等 | baseline 放行；seed12/13 还需前一 seed 的人工复审 |
| Evaluate | evaluate_seed11 等 | 对应最终 checkpoint；重新收集配对 C/H 留出评估 |
| Evaluate + Viewer | evaluate_seed11_viewer 等 | 该 seed 评估完成；只看最终候选 |
| Package | 指定运行目录名.zip + .sha256 | 来源、文件集合和哈希完整 |

Viewer 使用 Viser，最多 3000 个模拟步，受同一 supervisor/GPU 预算限制。开启命令 GUI 后再测试滑条；用户看持续摇摆、漂移、迟停及方向异常。viewer 程序完成不是人工 PASS，结果中的 viewer_verdict 保持 null。

GPU 分配：Validate/Baseline/基线 viewer 总计≤7200 秒；训练≤21600 秒且每 seed≤7200 秒；Evaluate/候选 viewer 总计≤7200 秒；总计≤36000 秒。GPU 子进程从启动到结束的墙钟时间计入，包括导入、编译和失败。纯安装、CPU 分析不计入 GPU 小时。

账本跨命令持久保存、互斥运行。supervisor 崩溃留下 active reservation/锁时停止后续运行，必须人工核对，不可删账本归零。超时停止属于预算不足，不是被试失败。失败目录保留，不自动重试；重试须先诊断并记录独立授权和已消耗预算。

只有完整文件夹可打包；结果通过 staging 目录生成，不覆盖旧输出。所有阶段均停止在交接点，不自动进入下一阶段。

### 人工授权（本机生成，不要求用户编造字段）

campaign/authorizations/baseline_review.json 包含完整身份字段、baseline_manifest_sha256、viewer_manifest_sha256、viewer_verdict="PASS"、用户原始反馈 user_feedback 和 review_kind="self_review_with_user_viewer"。两个 manifest 必须对应本次完整结果。

seed11_review.json / seed12_review.json 对应上一 seed 的 evaluation_manifest_sha256、viewer_manifest_sha256、viewer_verdict、user_feedback 和 decision="CONTINUE"。没有文件或任何绑定不匹配都拒绝下一 seed。授权只表示按预算继续收集数据，不代表该 seed 的收益成立。

### 本机诊断

Validate 的 Device=cpu、LocalCheck、CpuSmoke 仅用于实现验证：两环境、两次 PPO 更新，以及缩小的真实采集器检查。明确 evidence_eligible=false；CPU 结果、脏树结果不能授权正式运行。

CPU 诊断不会生成可参与训练的 checkpoint。任何诊断 trial 都不进入正式统计。最终 CUDA 控制资格、训练、结果及人工 viewer 必须在机房另行完成。

## 6. 离线分析和交付

analyze_recovery_c1_pilot 模块读取三个完整 Evaluate 目录，重新核对哈希、配对和汇总，生成 seed_results.csv 与 decision.json。缺 seed 报证据不完整，不自动给出负结果。数值通过仍返回 REQUIRES_REVIEW_AND_USER_VIEWER。

首份交付是资产审计、本机软件／CPU 验收和不训练的 M0 命令卡。M1/M2 在收到前一步结果后单独下发。没有用户批准，不改变控制器、奖励、动作权限、扰动协议、预算或研究路线。
