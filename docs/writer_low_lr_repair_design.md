# Writer N1800 恒定低学习率修复续训合同

2026-09-21 Owner要求按专家最新修正执行。本合同替代已结束的Writer稳定性诊断作为当前active design；
不改变已封存的覆盖重训、MT-BC与旧Test结论。

## 问题、父状态与唯一变量

本阶段检验：覆盖Writer在N1800完整训练状态上，把后续更新的applied learning rate固定为
`2.959936e-5`，能否恢复并稳定超过原覆盖训练最佳N1000的117/400。

唯一父节点为
`/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/writer/checkpoints/macro_00001800`。
恢复完整Writer、Text/VL/Action Meta、AdamW moments与step、sampler cursor及四个rank RNG；第一条新更新是
global step 1801。诊断N-L@36不含完整正式恢复状态，不作为父节点。

唯一科学变化是未来applied LR固定为`2.959936e-5`。AdamW `betas=(0.9,0.95)`、`eps=1e-8`、
`weight_decay=1e-4`、global clip1，以及36任务、每更新4任务、每任务21条跨episode主query＋7条同视频
教学query、教学权重1/3、单相机K1、完整H、38-target rank16 A/B全部保持。Source、normalization和MT-BC300冻结。
无warmup、LR sweep、新seed、任务降权、loss或架构变化。

首次启动属于保持完整状态但改变后续优化schedule的phase continuation，不称原高学习率轨迹的exact resume；
新phase内的后续中断从完整checkpoint exact resume。

## 实现、恢复与数值合同

在现有Writer训练器中加入显式、窄范围`phase_continuation`，不建立第二套trainer。先恢复并核对父状态：

- 模型参数和optimizer参数所有权完全一致，每个trainable参数都有AdamW状态；
- optimizer、scheduler、sampler均位于global1800，四rank RNG按原topology恢复；
- 父节点实际LR为`1.627965011517147e-4`，随后只把未来schedule切换为固定LR；
- 第1801次optimizer update实际使用`2.959936e-5`，scheduler不会在更新后改回旧曲线；
- checkpoint保存global与phase cursor，重载后仍保持固定LR和事件连续性。

日志同时记录`global_step`、`phase_step`、`lr_applied`、`lr_next`、主/教学loss、四模块grad norm、clip前总norm、
task occurrence、query identity、墙钟与显存。避免逐步复制全部参数到CPU。

保持N1800的gpu02四rank、frame chunk8、policy microbatch16×4、原query sharding与dtype/kernel合同。
正式运行来自clean pushed detached worktree。启动前按AGENTS实时检查两节点GPU、data0/data1 quota和study峰值；
资源不满足则记录工程阻塞，不静默改单卡、frame chunk或物理拓扑。

## 正式节点、控制器与停止

新运行根为`/data0/user/ymdai/ember_runs/writer_low_lr_repair_20260921`。每100次新更新保存并完成一次
correct Validation400：global1900、2000、2100、2200……。1900/2000只是最初观察点。每段必须在完整400行、
worker exit和readout验收后裁决，控制器不得提前运行下一段；不读取部分成功流。

早停历史只含1900起的新phase节点，N1800=92、N1000=117、MT-BC300=155只作研究参照：

1. 至少4个新节点，最近3个均低于它们之前的phase最高，平均差至少8/400，最近3点斜率不大于0：
   `sustained_decline`。
2. 至少6个新节点，最近5个未超过此前phase最高、范围不大于8/400且斜率不大于0：`plateau`。
3. 前两项未触发时，自最后一次严格phase最高刷新后已有6个连续新节点，且最近3点斜率不大于0：
   `no_progress_review`，报告“无进展，停止复核”，不宣称收敛。

一次下降或训练loss不能停止；刷新严格新高或有真实恢复时继续。没有2000、2250或3000硬终点。

## 选点、对照与停止边界

只用完整correct400选点。若新phase最高不严格超过117，保留原N1000为selected，整理全曲线后停止；
不做controls或其它补救。若严格超过117，在新phase节点中取correct最高；并列节点统一补same-task-other400，
再按other最高、仍同分取最早。冻结唯一候选后完成same-task-other400和cross-suite-wrong400；wrong不影响选点。
不自动做shuffle/reverse。

本阶段不启动Test、FT、RL、外部比较，也不追加其它LR、N1000起点、旧Writer暖启动、seed、任务权重、loss、
架构或诊断矩阵。工程错误可在不变科学合同内修复；科学负结果按上述规则停止。

## 交付

仓库报告根为`docs/review_materials/20260921/writer_low_lr_repair/`，保留协议与父谱系、训练metrics、每个
Validation400原始行、phase早停历史、选点记录、实际执行的other/wrong、完整曲线PNG/SVG及绘图源码、
completion/工程中断记录。图中同时展示旧覆盖Writer曲线，并在1800标出LR intervention；不截去下降段。
