# Hybrid P1 Leg Mainline

## P1.1 Expanded Posture Envelope

Status: **PASS** (seed 1, RTX machine-room rollout, code `6e1f03c` plus
the height-floor fitter at `5c20e8e`). This is a development qualification,
not a three-seed promotion result.

The expanded static sweep contained 121 cells. After the measured dynamic
height floor (`height >= 0.28 m`) and height-priority inscription, the
qualified command rectangle is:

```text
height = [0.2907321708, 0.3276857266] m
pitch  = [0.0000000000, 0.0320000000] rad
legacy coefficient map_hash = 8849ce39ff24b3342376dbae9c62d658c01288ad8c2b71dcd2ec20741b19a2f1
```

The uncompensated 25-cell balance probe had zero terminations and zero
non-wheel contacts, but measured worst steady drift `0.0393466 m/s`.
The station calibration reduced the compensated worst drift to
`0.0066133 m/s` (about 83%, below the `0.015 m/s` qualification limit).
The compensated grid also recorded:

```text
worst height RMSE     = 0.0008473 m
worst pitch RMSE      = 0.0061633 rad
worst pitch-rate p99  = 0.22465 rad/s
terminations          = 0
non-wheel contact     = 0
vx +/-0.05 error      = about 0.0019-0.0048 m/s
```

Canonical machine-room evidence (stored outside Git):

| Artifact | SHA-256 |
| --- | --- |
| `posture_map_seed1_floor028.json` | `eefe72151b1ada42d2e95ffe582dafbe11f6616e46703f71d243831d911a69a6` |
| `balance_uncompensated_floor028_seed1.json` | `d90fc874811fb171d5fb8d76f0b12416e771ba58cb6a42b49db5218099d78623` |
| `station_calibration_floor028_seed1.json` | `6a325c209bac17904276c5f37b030ecdfc327a6ffa9fc7248a3e3cab535dacdf` |
| `balance_compensated_floor028_seed1.json` | `ec566af60aa17968c3a82eba5a17045242a2853567496d1f39c60f6b9b85eccb` |

The legacy `map_hash` covers fitted coefficients only. For P1.2, the same
measured data was migrated offline into a full-envelope identity without a
new simulation run:

```text
posture_artifact_hash = 0d54fca78b38a880678d0ee69964ac86cb18e1a1f62a0ee716a4715071687ad3
station_calibration_hash = a4d805ce87fff2ef786c740ff366d24833e4c1162a9f70740cc1941dbeaf004a
full-hash posture file SHA-256 = 7408ff66f97619d0c89324b653d1fea1c41a8ff8d4bb73b4539bfe9cb24cf2af
full-hash station file SHA-256 = 551974d1d8f785c299d0a7b15e3caf61af45634f92aea65790eb166055582cb9
repository posture JSON SHA-256 = 593f3e9770bbb4b7afd74b0c6fa935603146c968c487aafd28e56ac500dbc5e0
repository station JSON SHA-256 = 7b7e84bbbd660680ca0f7a0799bfc83199a132ea2a1770a5f74478b32dc17c3a
```

Legacy Stage5 artifacts remain valid under coefficient-only binding. New P1
artifacts bind both `map_hash` and `posture_artifact_hash`; a station artifact
from another command envelope is rejected even when coefficients are equal.

## P1.2 Leg Residual Authority

Status: **RUN, STOP_FOR_ANALYSIS** (seed 1, RTX machine-room rollout,
code `136dc00`).

Pre-registered matrix:

```text
leg residual scale = {0.035, 0.070, 0.100} rad
seed = 1
training = 256 envs x 100 iterations
save interval = 25; screen newest K=3 checkpoints
recovery = center posture, 8x kick, 32 envs x 4 kicks = 128 events
```

Before training, the zero-residual recovery is repeated twice to measure the
run noise. Every scale must report retention and robust safety screens,
matched full-policy recovery, evaluation-time leg ablation, residual
magnitude and saturation, terminations, and non-wheel contact.

The matrix stopped for analysis after all three rows. Five checkpoints passed
the retention screen, but all five failed the robust screen on the same four
correlated zero-speed standing metrics. No failure involved a termination or
non-wheel contact. Because no checkpoint passed both screens, the protocol
correctly did not enter the 128-event recovery or evaluation-time leg-ablation
probe. It did not select a scale, launch more seeds, or add iterations.

The five P1.2 candidate checkpoint files are no longer available. The local
`D:\mjlab_workspace\model_99.pt` file, when present, has SHA-256
`48a053f0d1acaa799b105c20bce29d5f4a738da36fd9117d4bdf5f03f2c63937`.
It is the frozen formal Stage5 candidate, not one of the P1.2 candidates, and
must not be relabelled or used to reconstruct the P1.2 matrix.

Machine-room entry point:

```powershell
& .\scripts\run_hybrid_leg_authority_seed1.ps1
```

This command is historical only. Do not rerun the matrix to replace the lost
checkpoint files.

## P2 k.0 Classical Stair Height Probe

Status: **FORMAL GPU RUN, CLASSICAL_DEATH_HEIGHT_BRACKETED** on branch
`codex/p2-stair-probe` at `9edb8b7` (seed 1, RTX machine room,
MjLab `43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6`).

The zero-residual classical stack is evaluated on MjLab `pyramid_stairs` at
heights 0.00-0.10 m in 0.01 m increments with a fixed 0.30 m step width. Each
height uses 16 seed-controlled reset trials and three repeats for each of two posture cards:
the envelope center `(0.3092089487 m, 0.016 rad)` and the high posture
`(0.3276857266 m, 0 rad)`. Runs start on flat ground outside the first riser,
settle for 2 s, command `+0.07 m/s` for at most 10 s, and require 0.5 s beyond
the first riser with no termination or non-wheel contact. Reset x/y and small
forward/pitch-rate perturbations are generated reproducibly from seed 1 and
saved in every trial row; their registered bounds are 0.02 m, 0.03 m,
0.01 m/s, and 0.02 rad/s respectively.

The probe is a non-promotable allocation measurement. It loads no checkpoint
or yaw artifact, applies six zero policy actions, reports both posture cards
separately, and never starts P3. If only one posture card fails within the
registered sweep, the result is `EXTEND_SWEEP_BEFORE_P3` until both cards have
a common failure height; it is not variance evidence. The machine-room entry
point is:

```powershell
& .\scripts\run_hybrid_p2_stair_height_probe.ps1
```

The corrected formal run measured the same monotonic hard cliff on both
pre-registered posture cards: every flat-control trial passed, every trial at
0.01-0.10 m failed to cross the first riser, and no trial terminated or made
non-wheel contact. The first common failure height and recorded P3 candidate
is therefore 0.01 m. The first retained GPU run at `f4a4d4b` had already
completed all 1056 trials, but its wrapper rejected every trial because a
float64-exact `1.0e-9` root-height assertion was tighter than float32 GPU
state read-back. Commit `9edb8b7` widened only that assertion to
`5.0e-8` m; it did not change the terrain, reset, command, success, or
classification protocol.

## P2 k.0 Stall-Mechanism Diagnostic

Status: **FORMAL GPU RUN, REGISTERED WHEEL_SPIN_FRICTION_LIMITED;
CAUSAL ISOLATION INSUFFICIENT** on `codex/p2-stair-probe@46acfc5`
(seed 1, RTX 2080 SUPER). This remains observational, not a gate and not a
P3 promotion.

The diagnostic compares eight paired command cells on flat ground and the
0.01 m stair. It keeps the envelope-center height, sweeps qualified and
diagnostic-only lean-in pitches, and adds two 0.10 m/s cells that remain inside
the velocity-calibration domain but outside the Stage5 task range. Each cell
uses the same 16 seed-controlled resets, settles for 4 s, drives for 10 s, and
records the final 3 s stall window. The recorded channels are the signed
forward wheel target and wheel speed using `0.5 * (right - left)`, modeled
actuator torque and saturation, wheel slip, body velocity, pitch error, and
progress toward the riser.

The fixed classifications distinguish a viable classical card, friction-
limited wheel spin, torque-saturated stationary stall, balance-loop drive-
target collapse, a mixed mechanism, and an invalid flat baseline. Candidate
cells must also pass their paired flat control. The run loads no checkpoint or
yaw artifact and cannot authorize training or promotion. Machine-room entry
point after the branch is committed and pulled:

```powershell
& .\scripts\run_hybrid_p2_stall_diagnostic.ps1
```

The formal result contains 256/256 trials, 16/16 registered cells, complete
GPU/runtime and artifact provenance, and a matching SHA256
`d97b3b3d79242173c53fa3957d4720cfd82dfa234218e637c9679acd8b0b2b3e`.
Every flat control passed with zero terminations and zero non-wheel contacts.
No registered cell reached the 50% candidate-success bar. The strongest cell,
`vx=0.10 m/s, pitch=-0.032 rad`, crossed in 2/16 trials (12.5%) and moved
the median progress to within 0.4 mm of the registered face reference.

The frozen rule emitted `WHEEL_SPIN_FRICTION_LIMITED` because the baseline
1 cm cell had torque saturation 0.9846 and wheel-slip metric 0.0602 m/s. That
label is protocol-valid but not causally isolated: the paired flat baseline
already had saturation 0.9904 and slip 0.0471 m/s, so saturation decreased and
slip increased by only 0.0131 m/s at the stair. Across cells the 1 cm wheel
target fell to 44-59% of its paired flat value, while the actual pitch was
pushed to +0.110 to +0.156 rad despite zero or negative pitch commands. The
evidence therefore supports a coupled contact/traction, posture-deflection,
and drive-authority mechanism; it does not justify a friction-only claim or
automatic P3 launch. A follow-up must calibrate against paired flat deltas and
capture contact-force/friction-cone channels before causal promotion.

## P2 k.0 Paired First-Impact Causal Capture v2

Status: **FORMAL GPU RUN COMPLETE; ANALYSIS_READY; STOP_FOR_VARIANCE_ANALYSIS**
on `codex/p2-stair-probe@364e053` (seed 1, RTX 2080 SUPER, MjLab
`43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6`). This follow-up does not revise the
registered v1 output. It exists because the v1 absolute slip and saturation
thresholds were also crossed by the paired flat controls and therefore did
not isolate stair-specific causality.

The v2 capture freezes two representative command cards only:

```text
pitch_zero       : vx=0.07 m/s, pitch= 0.000 rad
fast_lean_0p032  : vx=0.10 m/s, pitch=-0.032 rad
```

Each card runs 16 paired reset slots on flat ground and the 0.01 m stair,
for 64 total trials and 32 flat/stair pairs. The envelope-center height is
fixed at 0.3092089487 m. Yaw is zero, all six policy actions are zero, and
no checkpoint or yaw artifact is loaded. The diagnostic adds an independent
wheel-terrain sensor with eight retained contact slots per wheel; it does not
modify the task's reward or termination contact sensors.

The first-riser selector is only a time anchor. It requires contact position
near the registered first face, significant horizontal normal, and nonzero
normal force. A local CPU interface check on 2026-07-24 observed maximum
`|normal_x|` about 0.106 on flat and 0.446 at the 1 cm riser, and confirmed
that the 1 cm env stalls near the face without termination or automatic
reset. These observations justify capture visibility, not a physical gate.

For every stair trial, v2 stores the raw contact slots at first impact and a
columnar 101-sample series from 25 control steps before through 75 steps after
impact. Its flat partner is sampled at the same absolute drive steps. The
output includes flat, stair, and `stair - flat` values for progress, pitch,
body velocity, wheel target/speed, modeled actuator torque and saturation,
slip, contact counts, contact-normal geometry, and normal/tangential forces.

The only formal classifications are `ANALYSIS_READY` and `INVALID_CAPTURE`.
They describe capture completeness, not friction-only, torque-only, or
drive-collapse causality. The wrapper records GPU model/driver/runtime and
all artifact/Git provenance, writes `stall_causal_v2.json`,
`protocol_note.json`, and `SHA256SUMS.txt`, and refuses training,
checkpoint, migration, yaw calibration, promotion, or automatic P3 actions.

Machine-room entry point after pulling the published branch head:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hybrid_p2_stall_causal_v2.ps1
```

The returned capture is complete: 64/64 trials, 32/32 valid paired captures,
101 aligned samples per capture, no invalid pairs, and both flat controls at
16/16 success with zero termination and zero non-wheel contact. The archived
file SHA-256 values are
`aabe90d88bbc7bfdfe89036d0e5c10f42be2cc082d38295acee1103d3c86b301`
for `stall_causal_v2.json` and
`366254fa2045f1bdfa210ae2262192e597d24ebff22ae85eda1d5e1d82354734`
for `protocol_note.json`.

The first-impact alignment supports the ordered interpretation
`riser contact -> pitch deflection -> balance-loop drive suppression`. At the
impact tick the selected riser normal force is about 120-126 N while pitch and
wheel-target deltas remain near zero. By 0.1 s after impact, pitch has moved by
about +0.02 rad and the signed forward wheel target has fallen by roughly
1.5-1.8 rad/s relative to paired flat. Torque and saturation do not increase
relative to flat, so the result rejects friction-only, torque-only, and pure
pre-impact target-collapse explanations without assigning a single cause.

The strongest static card remains non-repeatable: `fast_lean_0p032` changed
from 2/16 successes in the v1 eight-cell session to 7/16 in the v2 two-cell
session. The 31.25 percentage-point difference is consistent with a
session/order or near-boundary numerical sensitivity that has not been
isolated. Therefore the formal decision is `STOP_FOR_VARIANCE_ANALYSIS`: no
old P3 launch and no training. This result is a
`C0_FIXED_GAIN_LQR_FAILURE_CHARACTERIZATION`, not a global upper bound on
classical control. The next branch must first build and freeze a stronger
classical controller before residual PPO is considered.

## Classical Upper Bound C1-C3

Status: **C1 NINE-NODE GPU DATA FROZEN; 27-Q/R FLAT GATE PREREGISTERED,
NOT YET RUN** on `codex/p2-classical-upper-bound`. C0 remains frozen evidence.
The new path adds
a hash-bound 3x3 gain-scheduled LQR artifact, equilibrium-pitch state
construction, bounded bilinear interpolation, deploy-stack compatibility,
proprioceptive contact qualification, a deployable stair state machine,
offline CEM utilities, and the six-dimensional residual-PPO observation and
promotion contract. Legacy single-gain artifacts and default Stage0-5 configs
remain compatible.

The schedule grid uses heights 0.2907321708, 0.3092089487, and 0.3276857266 m.
Pitch must be the widest formally qualified symmetric range among +/-0.032,
+/-0.024, and +/-0.016 rad. Each of the nine nodes requires 32 envs, 2500
steps, 250 warmup steps, hold 5, 20% held-out data, rank-four controllability,
NRMSE <=0.15, and no PD fallback. A schedule artifact is rejected unless its
selection record proves that all 27 registered Q/R candidates were evaluated
by the flat safety gate.

The contact detector is intentionally restricted to pitch-rate change,
wheel-speed error, and body deceleration. The fitter rejects any detector that
does not achieve zero flat false-positive sequences and at least 95% detection
within three ticks. Applying that gate to the archived C0 capture currently
finds no qualified detector; this does not authorize contact truth or a weaker
threshold. Detector fitting must be repeated on the eventual C1 paired data.
Until C1 qualification succeeds, C2/C3 maneuver freezing and stair PPO remain
ineligible.

### Codex runtime-loader correction (2026-07-25)

Codex：更正 13——`1c57e53` 新增的
`registered_fixed_symmetric_hull_rectangle` 已被 fitter 与 C1 preflight
接受，但 Stage3 task runtime 仍使用旧的两项 verification-method 白名单，
因此机房首次 3x3 dynamic balance qualification 在创建环境前失败，并未产生
正式 qualification 数据。提交
`19b1c2630145ec040cf32893d4758e13f963b91b` 将 verification methods 收口到
共享常量，同时覆盖 task loader 与 portable classical-stack loader，并加入两条
回归测试。真实 `p032` artifact 已在本地通过两个 loader 和同一 Stage3 probe
入口的最短 CPU 环境 smoke；完整 513 项 unittest 通过。原 artifact 的 fitter
provenance 仍固定为 `1c57e53`，runtime 最低版本则固定为 `19b1c26`，二者不得
混为同一个 SHA。下一步仍只允许重跑 3x3 dynamic balance qualification；不得
启动 identification、C2、C3、C* 或 PPO。

### Codex C1 posture/station closure (2026-07-25)

Codex：更正 14——机房在 `c2d1ffe` 上完成 registered `3x3` 未补偿与补偿
qualification。未补偿 JSON SHA256 为
`4ae258eaf73121fd1cffc1186c5611b20a3c95b1ef684060fafa39383b55ca06`；
由其拟合的 station artifact 文件 SHA256 为
`f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a`，
内部 hash 为
`c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a`。
补偿复验 JSON SHA256 为
`c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55`，
记录 `station_calibration_qualified=true`、零 termination、零 non-wheel
contact、最坏 drift `0.0069368 m/s`、height RMSE `0.0008191 m`、pitch
RMSE `0.0059484 rad`，全部通过注册阈值。C1 posture/station requalification
因此正式 PASS。

四份小型正式 JSON 已冻结到
`docs/experiments/artifacts/c1_posture_requalification_seed1/` 并设为
`-text -diff`。新增 `run_hybrid_c1_identification_collection.ps1` 将下一次机房
操作固定为一条命令：校验远端/MjLab/GPU/全部 artifact，依次采集 3x3 九节点的
32 env、2500 step、250 warmup、hold 5、20% held-out、seed 1 数据，拒绝覆盖，
生成逐节点日志、protocol note、SHA256SUMS 和下载 ZIP。该 wrapper 只采集
identification 数据，不训练、不生成 checkpoint，也不启动 C2/C3/C*/PPO；
27 组 Q/R flat-gate 必须等九节点数据拉回后离线拟合并另行裁决。

### Codex C1 identification closure and flat-gate preregistration (2026-07-25)

Claude：发布 `e54bd1a604b08b634821d88ce3a53a0f2fe66724`，修复 Windows
PowerShell 5.1 的 GPU/runtime provenance 查询后，机房完成九节点正式采集。
冻结 ZIP SHA256 为
`364590b8d9f2f5c66fdaac2b3fa124ee914236e33f6fc47e31e75f64d53c72e2`；
九个节点均为 80,000 有效样本、0 丢弃，采集协议为 CUDA 0、seed 1、
32 env、2500 steps、250 warmup、hold 5、amplitude 0.35、held-out 0.20。

Codex：接续 Claude 未提交的 flat-gate evaluator，并更正 PowerShell 5.1
`protocol_note.json` BOM 读取失败、全候选失败时错误写 selection 后崩溃、
输出目录可被静默复用，以及 ZIP/SHA256SUMS/log/sidecar/array/provenance
校验不完整等问题。正式数据离线复算固定得到 27 candidates、243 node fits、
最低 controllability rank 4、最大 held-out NRMSE `0.0996135568032975`、
0 fallback。evaluator 与单元测试提交为
`ffbb01850787ceead53ba407a0a7bf9c6f6a9b11`。

flat-gate 是 Q/R 选择筛选，不是 C* 正序/反序/独立会话正式验证。单一新建
Stage3 play env 按候选索引 0-26 运行；每个 cell reset。固定 15-cell 顺序为：
九个 `vx=0` 网格（height 外层、pitch 内层），然后 center、最低高度/负 pitch、
最高高度/正 pitch 三点各按 `+0.05`、`-0.05 m/s`。硬安全门为零 termination
和零 non-wheel contact。冻结 floor/cap（cap=`1.5x floor`）为：

| Metric | Floor | Cap |
|---|---:|---:|
| worst velocity error | 0.0069367592 m/s | 0.0104051388 m/s |
| p95 pitch error | 0.0155811692 rad | 0.0233717537 rad |
| p99 pitch rate | 0.2190857083 rad/s | 0.3286285624 rad/s |

wheel-target rate 仅作第四级字典序排序，不作硬门。通过候选按
`worst velocity error -> p95 pitch -> p99 pitch rate -> wheel-target rate`
选择。存在通过候选时裁决 `C1_FLAT_GATE_SELECTED` 并写 selection；27 个候选
完整运行但全失败时裁决 `NO_QR_CANDIDATE_PASSED_FLAT_GATE`，不写 selection，
正常返回并停止。两条路径都写 detail/adjudication，且均为
`promotion_eligible=false`、`training_eligible=false`、`checkpoint=null`。
artifact、协议、拟合、CUDA 或运行错误属于无效运行并返回非零。

机房入口固定为无参数 `scripts/run_hybrid_c1_flat_gate.ps1`。wrapper 钉住远端
HEAD、上述 evaluator 提交、MjLab、九节点 ZIP、五份 Git artifact、Windows
PowerShell 5.1 与 CUDA 0；显式清除 yaw artifact；使用
`.incomplete.<guid>` 工作目录，完整成功后发布结果目录与 ZIP。该阶段不构建
最终 schedule，不授权 C2/C3/C*/PPO；只有选中结果下载并离线审查后，才允许
单独构建 selected-Q/R 九节点 controller 与 schedule。

Codex：最终完整性复核补充更正：固定 ZIP 的 SHA 校验现在继续逐 entry
字节绑定到 evaluator 实际读取的解压目录，目录与其 manifest 不能再同步改写后自证；
wrapper 校验 `pyproject.toml` 声明的 `../mjlab-main`，并在 `uv sync` 后反查
Python 实际导入的 MjLab Git 根与冻结 SHA；最终 ZIP 先发布，结果目录移动失败时回滚
ZIP，避免留下半发布的正式目录。真实冻结输入重跑仍复现 27 candidates / 243 fits、
minimum rank 4、maximum NRMSE `0.0996135568032975`、fallback 0；完整 537 项
unittest 与 18 项 focused tests 通过。

Codex：机房首次运行 `9de9a2f` 在环境配置前停止，因为 wrapper 漏搜了当前仓库
`experiments/` 下的冻结九节点目录与同名 ZIP；数据未受影响，GPU evaluator 未启动。
路径发现现同时覆盖仓库 `experiments/`、仓库根和既有上级 workspace 布局，并继续要求
目录与 ZIP 同处且 ZIP SHA 完全匹配。

### Codex C1 affine-equilibrium recovery correction (2026-07-25)

机房在 d9a31382ca71a01d1eaca54d6445998609ff7558 完成 27 组正式
flat-gate，ZIP SHA256 为
01d753b240b5f6d8f010c91fb8ea895948e4098f6f16df0bce4aae93fa244ab1。
裁决为 NO_QR_CANDIDATE_PASSED_FLAT_GATE、passed=0/27、next_step=STOP；
结果已冻结到 docs/experiments/artifacts/c1_flat_gate_failure_seed1/。27 组均非
safety clean，最少仍有 164 次 termination，但 non-wheel contact 始终为零；
最佳速度、p95 pitch 和 p99 pitch-rate 分别仍为 cap 的约 38.8、18.6 和 6.2 倍。

Codex：更正原 schedule 只从 pitch 中减去 equilibrium，却没有对应的 equilibrium
control input；这不是完整的非零工作点线性控制律。恢复协议改为在零 residual
warmup 的最后 100 steps 测量完整四维 x_eq 与单输入 u_eq，数据使用 delta_x/delta_u，
runtime 固定执行 u = u_eq - K(x-x_eq)。同时加入旧合格 K 的闭环谱半径非回归锚定；
Q/R 候选只有在全部九节点都不比 incumbent 差超过 0.01 时才能以最大全局 blend
alpha 进入 GPU smoke。旧 schema1 schedule 保持兼容，但不能作为新的 C1 证据。

当前仍不授权 build 最终 schedule、C2/C3/C*/PPO 或训练。新的 affine-v3 九节点
采集 wrapper 会在打包前先验证同会话 incumbent，再让 27 个谱半径锚定候选仅运行
中心姿态的 vx=0/±0.05 三个 cell；incumbent 失败属于无效运行，候选全失败则正常
归档并 STOP，任何路径都不会自动重跑完整 27x15 flat-gate。

### Codex C1 affine center-smoke correction (2026-07-26)

机房在 `0c7bd78893998f0a1c6d58615fb3ea7fd97f0bdd` 完成 affine-v3 九节点
采集和中心 smoke；ZIP SHA256 为
`10e0f8f498107406e969e9f7d8390f8ac8c22f5838b60d5254e65196453eb4f9`。
九节点均为 80,000 有效样本、0 丢弃，243 fits 的 minimum rank=4、maximum
NRMSE=`0.09578188586978251`、fallback=0。裁决虽然是
`AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP`、0/27，但 27 个候选全部 safety clean，
全部通过 pitch 与 pitch-rate cap；唯一失败项是 velocity cap。最佳 candidate 6 的
worst velocity error=`0.011755385249853131 m/s`，仅比冻结 cap
`0.01040513883344829 m/s` 高约 `0.00135 m/s`；其 p99 pitch-rate 从 legacy
incumbent 的 `0.2252661884` 降到 `0.1544178873 rad/s`。

Codex：进一步 review 确认旧 smoke 有两个协议漏洞。第一，所谓 incumbent gate 只运行
了 zero-equilibrium legacy incumbent，没有在写入测得的 `x_eq/u_eq` 后以 alpha=0
验证 affine-incumbent 等价性，因此坐标/前馈错误与 gain 错误会被混淆。第二，闭环
谱半径约束只保护稳定性，不保护速度通道；所有候选均选择 alpha=0.5，中心节点的
deployed `vx/wheel-speed` gain norm 只剩 incumbent 的约 50.6%，造成 ±0.05 m/s
系统性超速。修复要求先通过 affine-incumbent alpha=0 gate，并要求速度两维 gain
沿实际 command-error 方向 `[0,0,1,1/r]` 的投影增益至少保留 incumbent 的 70%；
真实九节点离线重算后 27 个候选统一选择 alpha=0.25，最小 command-gain
ratio=`0.7374369770`。新 retry wrapper 复用冻结
九节点 ZIP，不重新采集、不训练、不运行旧 27x15 flat-gate；入口固定为
`scripts/run_hybrid_c1_affine_center_smoke_retry.ps1`，wrapper SHA256 为
`e9e2b45187daca6f97221a98264a0450f4bbf86423f35523570435824e419375`。
正式 GPU 结果仍待机房。

### Codex C1 candidate-24 final-gate preregistration (2026-07-26)

机房 retry 结果已下载并完成字节复核：`affine_center_smoke_retry.json`
SHA256=`18cea95353b227b47370af25265f16c2450ba25e224069e08c52f92d6d472f07`，
`protocol_note.json`
SHA256=`c336eb937a12252412bb2a8837504eaf13635fb5a0f3a4da2d33a1a1443b5c98`，
正式 ZIP
SHA256=`86521c7e5762b669a2c179c590f5c08fbd6454d165087ee8a02b86ae293f14dd`。
裁决为 `AFFINE_CENTER_SMOKE_HAS_CANDIDATES`，27/27 通过 legacy incumbent、
affine incumbent alpha=0、termination/contact、velocity、pitch 和 pitch-rate 门。

Codex：按冻结字典序
`worst velocity error -> p95 pitch -> p99 pitch rate -> wheel-target rate`
选择 candidate 24：Q=`[40,4,8,1]`、R=`[0.5]`、alpha=`0.25`。其中心
smoke 聚合为 worst velocity error=`0.00502255260944366 m/s`、p95 pitch=
`0.0111373793333769 rad`、p99 pitch-rate=`0.176143005490303 rad/s`、
wheel-target rate=`0.365479528903961`，且 safety clean。该选择不等于 C1
最终资格，也不授权训练或 PPO。

Codex：为避免继续运行低信息量的 27-candidate screen，最终 C1 平地资格只运行
candidate 24 一次，共 15 cells：九个注册 posture 节点的 `vx=0`，以及 center、
最低高度/负 pitch、最高高度/正 pitch 各 `+0.05/-0.05 m/s`。继续使用冻结 caps
`0.01040513883344829 m/s`、`0.023371753748506308 rad`、
`0.3286285623908043 rad/s` 和零 termination/contact。通过分类固定为
`C1_AFFINE_FULL_GATE_SELECTED`；失败分类固定为
`C1_AFFINE_FULL_GATE_FAILED_STOP`，失败后关闭 C1，不再调平地 gate。
两类均正常归档、返回 0，并保持 `promotion_eligible=false`、
`training_eligible=false`、`checkpoint=null`、`yaw_calibration_hash=null`。

新入口 `scripts/run_hybrid_c1_affine_full_gate.ps1` 绑定冻结九节点 ZIP、retry
JSON/protocol/ZIP、两份 SHA256SUMS 及其全部目录文件；只创建一个 Stage3 CUDA
env、只运行 candidate 24，不重新采集、不重新拟合 27 组、不 build schedule、
不训练、不进入 C2/C3/CEM/PPO。通过时额外生成
`c1_affine_full_gate_selection.json`；schedule validator 会同时校验 27-candidate
retry 排序、candidate 24 的 15-cell 最终结果及二者的 SHA256 绑定，失败时禁止生成
selection。通过并下载审查后，下一步是在开发机离线 build candidate-24 schedule、
冻结 C1，随后立即转入 C2 台阶接触检测/FSM。

Codex 机房更正：首次更新到 `fb3b1b8` 后，wrapper 在 GPU evaluator 启动前因
Windows checkout 将 LF 转为 CRLF 而触发 raw-byte self-hash mismatch；输入、环境和
正式证据均未被修改。自检现先把 CRLF/CR 规范化为 LF，再以 UTF-8 no-BOM 计算
SHA256；同一脚本的 LF 与 CRLF 形式都必须得到注册哈希（见下方 Claude 更正后的
当前值）。该修复只消除跨平台换行误报，不改变 candidate、15-cell 协议、阈值或
输出语义。

（Claude: 审计更正 2026-07-26——fb3b1b8 的 full-gate wrapper 运行时探针 payload
使用了内层双引号 `getattr(mujoco,"__version__",None)`。PowerShell 5.1 传参给
native 命令时不转义内层双引号，实测（本机 5.1 + 项目 .venv Python）该行必然以
`NameError: name '__version__' is not defined` 失败并使 wrapper 在 GPU 评估开始前
抛出 'Unable to query Python runtime provenance.'——这是 e54bd1a 教训的回归；其余
五个已在机房成功运行的 wrapper 均用双单引号 `''__version__''` 形式。已将该行改为
与 retry wrapper 逐字一致的已验证形式，并重新生成自哈希。修正后 wrapper 的注册
canonical SHA256 为
`eb9d27bea2e5b2ea163708daca9e756d9eeec6221aed5bfca09373d721283dd9`。
除该行外未改动任何 pin、协议、阈值或输出语义；本地 575 unittest 全过，
PowerShell AST 0 错误，修正后的 payload 已在真实 PowerShell 5.1 下实证返回
合法 JSON。机房请先 `git pull --ff-only` 再运行。）

### Codex integrity correction (2026-07-24)

Codex：更正此前 `76d1fcf` 的 C1 本地实现完整性边界。该提交不能作为
机房 C1 preflight 的最终运行版本；最低合格实现提交为
`1c57e5317618e40dada38aad1728b91d4d5e1d1d`。本次 review 修复了
schedule/posture 依赖环、27 组 Q/R 选择记录可伪造、collector 与 runtime
状态定义不一致、node controller 未绑定拟合时 NPZ/sidecar 哈希、stair
maneuver 姿态瞬时跳变、deployment 丢失 `stair_mode`，以及普通 clone 下
MjLab 路径错误。
此外 parser 强制使用注册的三个精确高度和三档允许 pitch bound，并拒绝 JSON
boolean 冒充 selected index 或 Q/R 数值。C1 wrapper 现在拒绝早于该完整性提交
的 checkout。

Codex：2026-07-24 机房返回的 11x11 sweep（NPZ SHA256
`7f6bc1b3955fc41d8c58101dffbfd90decf17e6d76f184b0bbc25f777d883f1c`）
证明 22 个 invalid 与 22 个 non-wheel-contact 完全同掩码，剩余 99 点无额外
异常。进一步 review 发现旧 `--pitch-half-span` 会优化偏置 pitch center 并再次
shrink，不能实现 C1 注册的零中心对称节点。提交 `1c57e53` 新增显式 fixed-height
+ symmetric-pitch hull verification，保留 legacy inscription 不变，并把 fitter Git SHA
绑定进 posture artifact hash。`±0.032` 注册矩形已在该 sweep hull 内通过静态验证；
它仍需机房动态 balance qualification，不能直接视为 C1 合格。

这只是代码与 provenance 修复，不代表 C1 已产生正式 GPU 数据。当前仍然
禁止启动 C2/C3、C* 冻结或 residual PPO；下一步只能先完成姿态重新资格、
九节点采集和 27 组 flat-gate 选择证据。

### C1 closure and C2 unlock (2026-07-26)

（Claude: C1 affine full gate 正式通过。机房在 `1f54968` 完成 candidate-24
15-cell 最终资格，classification=`C1_AFFINE_FULL_GATE_SELECTED`；15 cells
全部通过 caps（worst velocity=0.00406 m/s、p95 pitch=0.01770 rad、
p99 pitch-rate=0.24428 rad/s），零 termination，零 non-wheel contact。
九节点离线拟合完成（Q=[40,4,8,1], R=[0.5], alpha=0.25, incumbent_gain
九节点一致）；gain-scheduled LQR artifact 已冻结于
`docs/experiments/artifacts/c1_schedule_candidate24_1f54968_seed1/`，
schedule_hash=`8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203`。
按 line 294 umbrella rule，**C2 阶段正式解锁**。）

## C2 Proprioceptive Contact Detector on C1 Stack

Status: **MEASUREMENT-IMPLEMENTATION REPAIR, INDEPENDENT AUDIT REQUIRED** on
`codex/p2-classical-upper-bound`. C1 schedule artifact (candidate-24
gain-scheduled LQR, schedule_hash `8fe8548b...`) is frozen and unlocks this
phase. C3 remains blocked until a deployment-equivalent detector qualifies.

### Preregistered Protocol (2026-07-26)

（Claude: C2 paired-capture 协议预注册——在 candidate-24 schedule 栈上重演
first-impact paired capture，输出兼容 `stall_causal_v2.json` schema，供
detector fitter 消费。）

**Prerequisite**: C1 schedule artifact (schedule_hash
`8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203`) must be
frozen and deployed via `HOPPERTREX_HYBRID_CONTROLLER_PATH`.

**Probe identity**: `hybrid_c2_paired_capture_v1`

**Protocol parameters**:
- Heights: `[0.0, 0.01]` m (flat control + first-failure stair height from C0)
- Command cells: same as C0 v2 (`pitch_zero` pitch=0.0 vx=0.07 m/s,
  `fast_lean_0p032` pitch=-0.032 vx=0.10 m/s)
- Envs per height: 16 (official) / 1 (smoke)
- Settle steps: 200 (official) / 2 (smoke)
- Drive steps: 500 (official) / 8 (smoke)
- Pre-impact steps: 25 (official) / 1 (smoke)
- Post-impact steps: 75 (official) / 1 (smoke)
- Stable steps: 25 (official) / 2 (smoke)
- Device: `cuda:0` (official only)
- Seed: 1
- Task: `HopperTrex-Hybrid-v2-Stage5` play env (same registry id the C0 v2
  capture emitted; payload `task` field must equal it)

**Output schema**: `stall_causal_v2.json` compatible
- `probe`: `"hybrid_c2_paired_capture_v1"`
- `paired_captures[]`: array of valid paired captures
  - `aligned_series.flat`: columnar 101-sample series (`pitch_rate_radps`,
    `wheel_speed_error_radps`, `body_vx_mps`), recorded directly from the
    deployment detector inputs
  - `aligned_series.stair`: same fields, aligned to first riser impact
- `protocol.pre_impact_steps`: 25 (used by fitter as impact index)
- `protocol.detector_signal_schema`: `deployment_direct_v1`
- `protocol.control_dt_s`: 0.02
- `classification`: `"ANALYSIS_READY"` or `"INVALID_CAPTURE"`
- Provenance bindings: `controller_schedule_hash` (must equal
  `8fe8548b...`), `calibration_hash`, `posture_artifact_hash`,
  `station_calibration_hash`, `git_sha`, `mjlab_git_sha`

（Codex: C2-h 逐 tick 等价更正 2026-07-30——`deployment_direct_v1` 名称
并不成立：第三路 `body_vx_mps` 来自仿真 privileged root velocity，部署端却只有
轮速里程计，二者差分得到的 deceleration 不等价；wheel reference 还漏了已部署的
station drift；101 点 impact 对齐窗也没有 FSM 激活 mask，无法重放在线尝试。
新 schema=`deployment_attempt_v2`：probe 在 chassis IMU site 直接读取 MuJoCo
accelerometer，扣除姿态投影重力后写 `body_deceleration_mps2`；硬件 HAL 的
`ImuSample.forward_deceleration` 进入同一个 detector API；wheel error 包含 station
drift；每个 flat/stair attempt 保存完整 500 tick 三路信号及由 wheel odometry
`progress < 0.35 m` 生成的 `detector_active` mask。FSM 在 stair attempt 开始时复位并
激活 detector，APPROACH/PRELOAD 的首次触发锁存到 CONTACT_WAIT；首样本只初始化
pitch-rate baseline，不再与硬编码 0 比较。fitter 对同一 mask 逐 tick 重放，旧
`deployment_direct_v1` 和无 mask capture fail-fast。阈值、125 格网、2-of-3、连续
2 tick、零 flat 误报和 95%/3-tick 门均未改变。C2-g ZIP 因缺直接加速度、完整尝试和
激活 mask，不能由新 fitter 消费；该历史证据保留但不能授权 detector/C3。）

**Flat control success gate**: ≥90% success rate with zero termination and
zero non-wheel contact (same as C0 v2).

**Implementation entry point**: `scripts/run_hybrid_c2_paired_capture.ps1`
(wrapper canonical SHA256:
`06ee9df95c912bfb69eeb93800b645a4149cf93bc184f5e8dcde0499c7f6af81`).

（Codex: 2026-07-30 独立审计修复补充：`ANALYSIS_READY` 现在由 probe、
wrapper 与 fitter 三层强制绑定为 2 个 command cells、32/32 valid、零 invalid、
零 stair termination、零 missing impact、每 trial 500 drive steps、每侧三字段各
101 samples；`INVALID_CAPTURE` 仍可封存为异常证据，但 fitter 必须拒绝。此补强仅约束
证据资格，detector 的 125 候选格网、阈值、2 tick、95%/3 tick 门与选优顺序未改。）

（Codex: C2-i 非证据 flat-floor calibration 结果 2026-07-30——在独立
`FLOOR_AUDIT_PASS` 后，RTX 2080 SUPER/cuda:0/seed 1 完成 2 cells × 16 flat
attempts，均为 200 settle + 500 drive、零 termination/non-wheel contact。总体 active
tick=6300；pitch-rate adjacent delta max/p99.9=`0.6647082865/0.4477291450` rad/s，
absolute wheel error=`4.5132474899/4.0106787953` rad/s，direct nonnegative body
deceleration=`3.7896840572/3.5313107421` m/s²。固定格网的 pitch/wheel 最大阈值
0.10/1.00 均不严格高于 overall max；故按看数据前冻结分支裁决为
`FLOOR_GRID_UNCOVERED_STOP`。artifact SHA256=
`3d831dfe4182af9132f4a3fc351d78ed5f4fcdfa67fba1ef221f5136965796ae`，
top-level/nested 均 `evidence_eligible=false`、`detector_fit_eligible=false`。不得修改
125 格网、正式重采 C2、拟合 detector 或进入 C3；下一步只允许 [D] 设计诊断。）

（Claude: 审计更正 2026-07-26——4242ae8 提交的 wrapper 第 8 行 `$RequiredBase`
为无效占位哈希 `716a9b39469c…3e3e3e3e`（716a9b3 的真实全长哈希是
`716a9b30eeb234e171f1606495581e7744e34a7c`）。机房照旧运行时
`git merge-base --is-ancestor` 会因对象不存在立即抛错，GPU 评估开始前
wrapper 即中止——铁律 8"零占位符"违规，且首轮审计漏检。已改为真实哈希并
重新生成规范化自哈希（中间注册值 `afff29ae…`，取代最初的 `bda6f987…`）。）

（Claude: 语义复审更正 2 2026-07-26——全面语义复审再发现两个机房阻断缺陷并修复：
(1) probe `_capture_provenance` 缺失——原代码把 `hybrid_provenance_lines(env)`
（返回 `list[str]`）用 `**` 展开进 payload，正式采集将在全部 GPU 工作完成后的
写盘前 TypeError 且 wrapper 校验的 git_sha/mjlab_git_sha/calibration_hash/
posture_artifact_hash/station_calibration_hash 五键根本不存在;已按机房实证过的
C0 stall diagnostic 模式重建 provenance dict，并新增合约测试
`tests/test_probe_hybrid_c2_paired_capture.py` 锁定 wrapper 消费键集与
classification 集合。(2) wrapper 缺 `$env:PYTHONPATH`——本仓 pyproject 无
[build-system]，包永不安装进 venv，全部 12 个机房实证 wrapper 均设 PYTHONPATH
而本 wrapper 未设，机房必 ModuleNotFoundError;已补（PS 5.1 实测：无 PYTHONPATH
exit 1，有则 exit 0）。修正后注册 canonical SHA256 =
`266363f88aec3e9b3acebc9c46eb265482432a7cdb84e84dce63056f6e957df9`
（取代 `afff29ae…`）。协议、阈值、cells、分类语义零改动。）

（Claude: 三路复审更正 2026-07-27——9e1441c 复审发现 wrapper 第 180 行要求
payload `task == "HopperTrex-Hybrid-v2-Stage3"`，但 probe 的 env cfg 经
`stair.make_stair_env_cfg` → `load_env_cfg(TASK)` 构造，`TASK =
"HopperTrex-Hybrid-v2-Stage5"`；机房真实交付工件
`hybrid_p2_stall_causal_v2_364e053_seed1/stall_causal_v2.json`（同一构造路径）
实证 `task = "HopperTrex-Hybrid-v2-Stage5"`。按原文执行，GPU 采集全部完成后
wrapper 必抛 "identity does not match"——机房必白跑。Stage3 为转录错误
（C1 full gate 用的是 Stage3 play env，串写到了 C2）。已改 wrapper 为 Stage5、
在协议参数中补注册 Task 行、并在合约测试新增 `stair.TASK` 身份锁定
（`test_task_identity_matches_wrapper_expectation`）。修正后注册 canonical
SHA256 = `cdbcff7fea92325442c6718b160d5b4ef0942115390b4ca016b2cec9425907a3`
（取代 `266363f8…`）。协议、阈值、cells、分类语义零改动。）

（Codex: C2 detector 裁决实现更正 2026-07-30——@8496f7c 的正式 capture
虽然通过运行完整性检查，但其 detector series 与部署输入不等价：(1) fitter 用
`np.gradient(pitch_rad, 0.02)` 重建 pitch rate，该实现为中心差分并读取未来样本，
而部署直接消费 `root_link_ang_vel_b[:,1]`；(2) fitter 用
`wheel_speed_radps-wheel_target_radps`，其中 wheel target 是闭环 LQR 输出，部署
detector 则使用 `signed_wheel_speed-(scale*command_vx+bias)/wheel_radius`。旧 schema
没有保存原生 pitch rate，无法从冻结 ZIP 严格重放部署 detector。因此旧 fitter 的
`0 qualified` 只证明错误测量实现下无候选，**不构成本体感受 detector 的有效
falsification**。修复将三路部署输入直接写入 `deployment_direct_v1` schema，fitter
明确拒绝旧 schema；stair 资格按首次检测裁决，impact 前检测不能被后续检测掩盖。
125 候选格网、2-of-3、连续 2 tick、零 flat false positive、95%/3-tick 门均未改变。
原 ZIP SHA256 `ee31c4576cb93841e858a9b1426b27cc819417cfbbcf7461e93071901ddd2187`
永久保留且不得覆盖。修复须经另一 agent 独立审计后，方可授权一次新的正式 capture；
审计/重采前 C3/CEM/C*/PPO 全部继续禁止。）

Machine-room command after pulling the published branch head:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_hybrid_c2_paired_capture.ps1
```

**Detector fitting**: Once capture is complete, run
`fit_hybrid_stair_contact_detector.py --input <capture json> --output
<detector json>`. Fitter searches 125-candidate grid (pitch-rate 0.02-0.10 ×
wheel-error 0.10-1.00 × decel 0.5-5.0, consecutive_ticks=2). Qualification
requires **zero flat false-positive sequences** and **≥95% detection within 3
ticks**. All-fail raises `RuntimeError` = valid failure; no weaker threshold
authorized. Qualified detector enables C3; all-fail = STOP, user decision
point.

**Decision branches**:
- Qualified detector found → freeze detector artifact, proceed to C3
- All-fail on C1 data → valid negative result; decision options: (1) narrow
  working domain and recapture, (2) declare proprioceptive detection
  infeasible and escalate to user decision point; **no on-the-fly threshold
  relaxation**

**Prohibited actions**: This phase does not train, does not generate
checkpoint, does not authorize C3/CEM/PPO until detector is qualified.

## Yaw Calibration GPU Requalification (Parallel Task)

Status: **PROTOCOL PREREGISTERED, NOT YET IMPLEMENTED** on
`codex/p2-classical-upper-bound`. This is a parallel task (does not wait for
C2 completion) to rebuild the yaw calibration artifact, prerequisite for
sim-to-real R0-R3 and Stage1-B/Stage2 training.

### Context and Rationale (2026-07-26)

（Claude: Historical formal yaw artifact 已随旧机房物理清空不可恢复（见
`docs/experiments/artifacts/hybrid_runtime_seed1/manifest.json` line 24:
"The historical formal yaw artifact is unavailable and is not replaced by an
unqualified artifact"）。重建 GPU 版本是实机路线 R0-R3 的前置依赖（yaw 是
classical baseline 必需组件），也是 Stage1-B/Stage2 训练的前置（baseline
需要 yaw feedforward map）。按 8.15 硬截止倒排，若 8.5 决策走实机路线，现在不
启动则来不及（yaw 2-3 天 + Stage1-B 3-4 天 + Stage2 3-4 天 = 8-11 天）。）

### Preregistered Protocol (2026-07-26)

**Prerequisite**: Stage0 LQR gain artifact (controller_gain_hash
`8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98`).

**Probe identity**: `hybrid_yaw_transfer_gpu_v1`

**Protocol parameters**:
- Task: `HopperTrex-Hybrid-v2-Stage2`
- Device: `cuda:0`
- Num envs: 16
- Settle steps: 50
- Measure steps: 150
- Yaw actions: densified sweep for piecewise-linear fit quality
  `[-1.0, -0.85, -0.7, -0.55, -0.4, -0.25, -0.15, 0.15, 0.25, 0.4, 0.55, 0.7,
  0.85, 1.0]` (14 points, symmetric around zero to pin (0, 0))
- Probe yaw scale: 1.0 (full nominal differential reach)
- Seed: 1

**Output artifact**: `yaw_calibration.json`
- Schema version: 2
- Breakpoints: fitted monotone (body_yaw_rate, wheel_differential) pairs
  pinned at (0.0, 0.0)
- `controller_gain_hash`: `8fee25a0...` (bound to Stage0 LQR)
- `yaw_calibration_hash`: self-hash via canonical JSON SHA-256
- Provenance: `git_sha`, `mjlab_git_sha`, GPU model/driver

**Qualification caps** (inherited from historical yaw diagnostic, to be
verified on GPU):
- All 14 yaw actions must reach settle (zero termination, zero non-wheel
  contact)
- Measured transfer must be monotone after pinning (0, 0)
- Breakpoint body-yaw axis (element 0) must span the Stage2 yaw command
  domain [-0.10, 0.10] rad/s; breakpoint differential axis (element 1) must
  reach at least [-0.8, 0.8] rad/s sweep authority

**Implementation entry point**: `scripts/run_hybrid_yaw_gpu_requalification.ps1`
(wrapper canonical SHA256:
`35e841dd76cfab52e2981a7a4e05720dc253233aba0ad47928018f66a1edbcd2`).

（Claude: 语义复审更正 2026-07-26——ec71d1b 首版 wrapper 存在四个缺陷，全面
语义复审发现并修复：(1) [阻断] 第 60 行读 `$ControllerContent.controller_gain_hash`，
但 Stage0 工件 `controller_seed1.json` 的真实字段名是 `gain_hash`——StrictMode
下属性不存在直接抛错，机房必死（PS 5.1 用真工件实测 `.gain_hash` 读取通过）;
(2) [阻断] 缺 `$env:PYTHONPATH`，包永不安装进 venv，机房必 ModuleNotFoundError
（实测无 PYTHONPATH exit 1）;(3) [重大] probe 调用用 `Invoke-Expression … 2>&1`
在 `$ErrorActionPreference='Stop'` 下运行——python 任何 stderr 输出（torch/warp
启动告警属常态）都会变成 NativeCommandError 中途击杀 probe;已换成机房实证的
受控 EAP 辅助函数 `Invoke-NativeLogged`（splatted 参数 + `$LASTEXITCODE` 检查）;
(4) [重大] 注册 qualification caps 中"breakpoints 覆盖 ≥ [-0.8, 0.8] rad/s"
无人机械强制而 wrapper 无条件盖 `YAW_GPU_QUALIFIED` 章;已在 wrapper 内加
覆盖检查（≥3 breakpoints 且 min ≤ −0.8、max ≥ +0.8，不满足即 throw）。
"零 termination"cap 由 probe 的 SystemExit 门强制;"单调性"由
`validate_yaw_breakpoints` 在写盘时强制。修正后注册 canonical SHA256 =
`8d2bc8a61f1a8936334810ce3eaf2c1728b8ddd6de0b99ea5480b1b8a9ae7819`
（取代 `4de60f00…`）。协议参数（14 yaw actions/seed/steps/task/device）零改动。）

（Claude: 三路复审更正 2026-07-27——9e1441c 复审发现注册 cap 本身抄错了轴：
"breakpoints 覆盖 ≥ [-0.8, 0.8] rad/s body yaw rate"物理不可满足。实证：历史
CPU yaw 诊断工件（`yaw_calibration_preliminary_cpu.json`）在满幅差速 ±1.0 下
实测机体偏航仅覆盖 [-0.2747, +0.2709] rad/s——0.8 这个数属于**差速轴**
（breakpoint 元素 1，历史值 ±1.0、次外点 ±0.85），注册时误写到机体偏航轴
（元素 0）。9e1441c 的 wrapper 忠实执行错误注册文本，PS 5.1 实测其 cap 检查在
真实历史工件上必抛——GPU 全程跑完后 wrapper 中止，机房必白跑。这不是放宽
门槛，而是修正注册转录错误（"继承自历史诊断"的 cap 若按原文执行，历史诊断
自身都无法通过——归谬）。修正后 cap 为双轴：机体偏航轴必须覆盖 Stage2 yaw
命令域 [-0.10, 0.10] rad/s（必要条件，非新作阈值：FF 的输入轴必须覆盖其服务
的命令域；历史工件 ±0.27 满足），差速轴必须达到 [-0.8, 0.8]（继承值回归其
本轴；历史工件 ±1.0 满足）。PS 5.1 四用例实测：真实历史工件 PASS、2 断点
THROW、偏航轴不足 THROW、差速轴不足 THROW。修正后注册 canonical SHA256 =
`35e841dd76cfab52e2981a7a4e05720dc253233aba0ad47928018f66a1edbcd2`
（取代 `8d2bc8a6…`）。协议参数零改动。）

**Artifact freezing**: Once qualified, freeze to
`docs/experiments/artifacts/yaw_gpu_<git_sha>_seed1/yaw_calibration.json` +
SHA256SUMS. Register `yaw_calibration_hash` in this document under a new
"Yaw Calibration" subsection (or append to this section if no dedicated
subsection exists).

**Binding requirement**: Sim-to-real R3 runtime and Stage1-B/Stage2 training
must set `HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH` to point to this artifact;
loader will validate `controller_gain_hash` binding at load time.

**Decision branches**:
- Qualification passed → freeze artifact, unblock Stage1-B training
- Qualification failed (non-monotone or insufficient range) → valid negative
  result; decision options: (1) densify sweep around stiction knee, (2)
  escalate to user decision point; **no on-the-fly relaxation of monotonicity
  or range requirements**

**Prohibited actions**: This phase does not train, does not generate
checkpoint, does not authorize Stage1-B/Stage2 until yaw artifact is frozen.
