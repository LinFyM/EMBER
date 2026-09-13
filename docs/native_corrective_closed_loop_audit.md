# 固定真实纠正标签的跨视频／初始化闭环前提

2026-09-13登记。Owner授权依完整正负证据继续推导与验证；整体有益视频特异性goal未完成，暂不强制145/400。
本项只评估既有state-free真实纠正oracle的训练侧闭环作用，不训练Writer、不改变现有标签或选择checkpoint。

## 1. 为什么先检验这个接口

[原生纠正传递§7](native_corrective_transfer_audit.md#7-完整结果与关闭裁决)证明给定真实动作纠正后，
既有rank16更新能降低独立episode的t1/full10动作预测误差；当时明确没有环境步或完整闭环。
[获取分析§6–8](native_correction_acquisition_audit.md#8-幅度方向分析完整结果与关闭裁决)又表明实际Writer拟合较弱、
整体gain解释有限，乐观固定A投影有较高系数代价。这些事实仍未证明“精确学会这些标签”值得作为闭环修复重点。

当前源码已使用signed tanh的R及自由B，不能再把它们当新修正。旧账本`fcdb6e43:docs/research_history.md`
§143–150保留逐帧读出局部改善、跨task共享漂移与单位化／gain失败；旧G1的容量与投影行为损失也不等于当前G的闭环证据。
因此先检验真实目标G本身，而不是追加梯度冲突、坐标校准或完整Writer训练来解释参数误差。

竞争解释是：当前标签确有可重复的闭环作用而合法共享获取不足；或动作误差下降尚未变成稳定闭环作用，
因此单纯提高这些参数标签的拟合不能被视为已确立的行为修复方向。混合结果也必须保留，不强求唯一根因。

## 2. 冻结输入与完整有限面板

- 唯一oracle为原`native_corrective_transfer_20260913/formal/`中的state-free LoRA，固定train24×demo16–19，96套。
  从`native_correction_writer_20260913/correction_labels/registration.json`引用原件；构造operator为f39d594f。
  不读新teacher动作／state，不构造新label，不重新定幅、投影、截断或合并LoRA，也不引入true-state分支。
- 每套LoRA使用该task的init32、33、34、35各一次，构成24task×4teacher×4init=384条oracle闭环记录。
  分成demo16／17／18／19四个固定面板，各96条；一个面板每task固定一套完整38-target rank16 LoRA。
  这明确是有限教学池的训练诊断：每条teacher在四个init复用，不声称正式50-video无放回或validation400。
- 冻结source在同24task×4init新运行96条参照，仅运行一次；每条source结果按相同task/init供四teacher配对使用。
  统计不能把这96条source当384个独立样本。旧source15/96只作历史参照，不代替本次配对参照或改变采样。
- 全部五面板共480个实际episodes，在读取部分结果后不停止、扩展、筛视频或更换初态。
  不读取validation/Test，不进行correct/wrong、same-task-other、无视频、shuffled/reversed等Writer资格或最终controls。

这是一项**使用真实动作构造标签的privileged训练诊断**，不是action-hidden部署方法。
task索引仅在外部评测器中选择已封存oracle，不进入任何Writer；闭环只安装一套LoRA，source参数始终冻结。
不能把该oracle的任何成功计入合法Writer分数、checkpoint资格、最终视频必要性或完整goal完成。

## 3. 执行与配对

复用canonical `scripts/evaluate_pi05.py`的static task-LoRA adapter、cost-balanced dynamic queue、long-first与persistent workers。
官方render256/model224、双相机rotate180、8维执行者state／7维action、10flow、执行前5action后replan、
dummy settling10、成功即终止及220/280/300/520 horizons保持。无exploration、reward学习或额外rollout控制器。

使用`development_train/screen/state_count4/init32–35`已有诊断接口；这里的screen是数据规模模式，不承担checkpoint选择。
source与四oracle严格复用task、init、环境seed、policy noise按task/init/replan的schedule、source／normalization／tokenizer和全部执行合同。
每条oracle记录保留teacher demo、构造位置原metadata引用与原adapter路径，不能伪造Writer或G1 checkpoint身份。

## 4. 预登记读出与停止

令`s_tdi`为固定task t、teacher d、init i的oracle成功，`s0_ti`为source成功。
每task先对四teacher、四init等权，得到`Δ_t=mean_di(s_tdi−s0_ti)`；主结果为24task等权均值。
固定seed20260926、20,000次task-cluster paired bootstrap，抽样单位是task，报告主净率的95%区间。
同时完整报告每task／suite／teacher／init的成功率、breadth、retained/gained/lost、churn与success-set重合。
每teacher对source的96条配对单独展示；聚合384中的source重复引用和权重明确标注，不冒充400-row正式分数。

仅在以下条件全部满足时，称当前标签取得本有限面板上的可重复闭环前提：主净率区间下界严格>0，
至少两个suite净正，四个teacher面板各自净增严格>0，四个init分组各自净增也严格>0。
这些规则检验固定标签的作用，不形成一个新Writer方法或其它qualification gate。

- 通过：保留G确有本面板跨teacher／init闭环作用的事实；下一项才优先区分合法共享纠正获取与学习信用。
  不从中自动启动完整Writer、调坐标或续训，仍需有实质机制差异和可失败预测。
- 不通过或混合：停止把“更精确拟合这套G便能带来可重复闭环收益”当已确立的修复依据；
  保留实际局部收益及原独立动作预测正事实。不把有限non-pass扩大成所有真实纠正、视频或梯度方法无效。

无论分支均关闭该项，不追加teacher、init、adapter幅度、rank、probe、flow、seed或轮数扫描。
不从该诊断选择四个旧Writer中的任何一个；整体goal继续。

## 5. 实施、资源与生命周期

在现有static task-LoRA来源检查中只补一种明确的训练侧oracle身份，复用全部实际执行逻辑；
原文件已612行，本次采用局部来源分支而不复制评测器或拆出没有复用价值的小模块。
这属于有关闭触发的临时诊断支持：本项全部结果核验后退役该分支，clean pushed运行commit及原件保留复现代码。
保持所有旧adapter的来源检查；实际96个标签的task/demo/state-free/完整shape与来源核验，加已有static evaluator回归即可。

正式执行来自clean pushed commit的detached frozen worktree。复用source、tokenizer、数据、assets及96套原LoRA，不复制大资产。
结果放原data0研究根的`oracle_rollout/`，五个eval目录、raw rows、aggregate、completion、配对统计与日志新增峰值预算1GiB；
data1文档与frozen worktree预算512MiB。创建前查strg01对应独立quota、相关个人目录用量和共享容量。
launch前同时live检查gpu01／gpu02，单节点至多使用6张有实际吞吐收益的A40，明确UUID／进程／共驻余量，不等待凑卡或dummy占卡。

保留registration、launch、四份oracle引用manifest、原始480rows、各面板完整退出／aggregate、配对与统计原件。
公式、源配置／RNG配对、task/video/init全覆盖、单完整adapter与worker退出核验后关闭；不新增hash、全树扫描或部署旁路。
