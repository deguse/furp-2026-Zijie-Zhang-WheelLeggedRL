# HopperTrex A1 Poster v3

日期：2026-10-01。独立第三版，未覆盖第二版、学校模板或原始实验材料。

## 使用与编辑
- PPTX / PDF 为一页标准 A1 竖版，594 × 841 mm；文字和校徽位于 25 mm 安全区内，深蓝横幅出血至页面边缘。
- 高清 PNG 为 3564 × 5046 px；印刷优先使用 PDF，按实际大小 100% 输出。
- 主标题 72 pt；一级标题约 32–34 pt；正文约 20–21 pt；主图标注 16 pt / 1 pt 线宽；图表轴标签 17 pt。
- 四个图表均为原生可编辑图表，分别嵌入完整数值工作簿；框图、直接标注、10% 目标线、40 格热力图均可编辑。
- PowerPoint 的显示数值与旧版一致。工作簿采用 Excel 可保存的 15 位有效数字，最大绝对舍入差 4.6×10⁻¹⁶；原始完整精度 JSON 未变，位于 ../source/poster_data.json。

## 机构、图片与视频
- Faculty of Science and Engineering；Zijie Zhang ([20721407])；已移除 Supervisor 和身份占位。
- 使用用户指定的 Annotated Robotic Rover Chassis.png，清理原图烘焙标注和透明杂边，再叠加可编辑标签与扰动箭头。遮挡位置采用确定性局部插值；示意 CoM 沿用用户图，不代表计算所得的质量中心。
- 二维码沿用第二版用户提供的真实二维码，目标 https://qrco.de/bh2NyL；此前解析为 https://www.youtube.com/watch?v=dLVX1cyY2kQ。未更换视频、上传或发布新内容。

## 与请求中绝对化文案的差异
保留指定模块、主旨、指标和强调方式，但以下文字按已有证据表述，未逐字加入未经验证的结论：
1. “strictly during high-impulse transients” 改为改善 high-impulse transient recovery：现有动作接口未证明存在仅瞬态开启的门控。
2. “eliminating actuator shock” 改为 reducing actuator shock：94.44% 是峰值降低，并非峰值归零。
3. “without leg actuation” 改为 with leg residuals masked：消融屏蔽学习残差，而非禁用腿执行器。~70% 为相对改善幅度减少约 69.22%，且两结果使用各自匹配基线，不是严格因果分解。
4. ≤40 ms 写为仿真结果支持的 provisional control latency budget，并保留硬件验证要求；零终止事件不等同于实机安全保证。
5. 未加入未验证的“100% reproducible”，采用 Reproducible simulation workflow。

## 数值口径与来源
- 全残差恢复：1.01296875 → 0.8871875 s，12.41709% ≈ 12.42%。
- 腿残差消融：0.99765625 → 0.95953125 s，3.82146% ≈ 3.82%。同 checkpoint，评估时屏蔽；每臂 128 次扰动，单训练 seed。
- 25 个匹配姿态工况最坏漂移：0.0710283592 → 0.0115892617 m/s，降低 83.68%。图中保留全部 10 条序列 / 50 个 y 值。
- 四档参考整形最坏峰值：0.9955744743 → 0.0553341247 m/s，降低 94.44%；最坏 settling time 0.78375 → 1.7425 s。
- 延迟矩阵为 2 策略 × 5 延迟 × 4 噪声档，格值是终止事件数，非成功率。两图共享同一非线性色阶；0 统一 #F4F5F7，正值由橙黄到深红，所有整数保持原值（包括 Classical 的 1 和 2）。
- 原始恢复记录：closeout_2026-09-29/evidence/stage5/seed1_stage5_robust_formal.json 及 seed1_stage5_robust_formal_legs_ablated.json。
- 经典层记录：posture_balance_qualification_seed1_m012_uncompensated.json、posture_balance_qualification_seed1_m012_compensated.json、posture_transition_qualification_seed1.json 及 shaped_t05/t10/t20 三档。
- 延迟记录：hybrid_latency_noise_6e1f03c/tolerance_seed1.json。完整继承来源说明见第二版/第一版工作目录。

## 验收记录
已检查 PowerPoint 实际导出、PDF 栅格化预览和主图/经典图/架构/边界局部；检查无文字溢出、身份占位或缺失图表。测试副本修改图表标题、数据值和尺寸后保存、重开成功。最终结构验收与数据核对位于 ../build，测试文件位于 ../review；它们不是额外交付物。
