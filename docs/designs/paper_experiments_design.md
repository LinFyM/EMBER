# 论文后继实验合同与旧冻结24任务实验记录

> 文档角色：封存设计合同。原文中的“当前／active／下一步”仅指当时阶段；当前授权与暂停状态以[progress](../../progress.md)为准。保留原科学定义和结果，不据此恢复执行。

2026-09-20，Owner明确设置goal并授权。实际进度只看[progress](../../progress.md)。

适用范围：第1–2节保留旧冻结1500/425与旧split的封存合同；新方法身份与split由[覆盖重训合同](coverage_retraining_design.md)替代。
第3–5节的FT/RL/外部比较、信息墙与停止条件继续适用，输入替换为新协议各自唯一选定模型。下述旧checkpoint编号不用于新训练选点。

## 1. 固定方法与用途

EMBER固定单相机agentview、K1、同视频教学主组1500；Source-71为aligned raw1000，MT-BC为既有aligned rank128共享425。
复用现有资产，不重训、不把EMBER叠加到MT-BC、不根据Test更换checkpoint。模型路径及frozen runtime在launch登记。
MT-BC425依据既有validation85/89/86选定，与旧450曝光合同分开报告；MT-BC50演示、Writer46演示差异公开。
本轮明确允许development24模型冻结进入Test8，不要求32任务fresh final训练；原训练合同与信息墙不改写。

## 2. Test两阶段决策

A：Source、MT-BC、EMBER correct各400，task等权，固定每task init0..49和全池teacher每轮各一次。
复用canonical seed20260911 video mapping及官方评测合同，source/MT-BC不虚构视频输入；共同env/policy RNG严格配对。
三面板完整后统一读结果：EMBER成功总数减MT-BC必须≥40（10pp）才进入B，否则暂停并询问Owner，不启动下游。
报告per-task/suite、breadth、配对R/G/L、churn与task-cluster bootstrap95%CI（20000次，seed20260915）。
不以Validation的165作为Test绝对门槛，不据运行中部分rows裁决或改分配。
B：EMBER other/wrong/shuffled/reversed各400。同task换视频、跨suite donor及真实帧变换沿现有配对接口。
整批收齐后报告correct/other相对wrong/order的配对差额和区间。内容或时序优势消失/证据不明确则暂停询问；
不新增惩罚loss、不换checkpoint、不自动增加seed。source是零LoRA参照，不称learned language-only。
Test已有2026-09-15共享L/R出口terminal900的sealed source400/correct400，历史86/49（research_history对应节）。
该历史底座/方法与当前不同，不混用分数、不宣称首次接触，不反馈本轮模型设计。

## 3. 单演示FT（仅在前项通过后执行）

默认validation/test全部16tasks。每task演示池预先分support候选与独立选点集，无重叠；eval50只作闭环评估。
每次一条support完整episode，EMBER只读视频/语言一次编译；FT用同episode观察/state/action更新合并MT-BC上的fresh rank16。
只梯度训练support；以独立选点集固定噪声/flow-time查询FM均值最低选预定checkpoint，平局取更早。
FT额外选点演示动作标签如实记入信息成本；不能声称全部目标阶段只接触一条演示。不使用eval结果重选。
训练配方、固定episode与query列表、预算/节点在Train资格后、held适应前追加。仍保持offset1、冻结normalization。
首轮若EMBER整体落后≥5pp，或较小差值已有明确负证据，则相关task补两条预定support独立重复，共三条。
不融合、不挑support；若持续劣势或重复结论混杂，暂停报告，不无限追加。单支持比较不冒充canonical无放回400。

## 4. Task-local RL（仅在前项通过后执行）

两臂：Source+生成完整A/B直接训练，对照merged MT-BC+fresh随机A/零B。均38targets/rank16，同缩放。
Writer只调用一次且冻结，共享基础权重冻结；optimizer/scheduler/critic fresh，不混FM。禁止重分解/归一化生成A/B。
先Train资格：zero-step功能、horizon50/replan5/flow10、正确概率重算/梯度/终止mask、真实学习、完整恢复。
预算和节点由真实profile及Train资格固定，正式目标task不调参。每区段与对应评测完整结束后读结果，不频繁轮询。
真实控制步为主横轴，optimizer steps辅助；训练/评测初态隔离；目标为所有预定节点EMBER任务等权曲线领先。
不理想或持续退化、不学习时暂停下一区段询问；AUC不替代Owner的全程领先目标。
目标任务动作/reward仅更新本次task-local参数，不能回流任何共享模型；零交互与适应结果独立。

## 5. 外部比较、交付与资产

WIZARD优先核查专家标签、输出覆盖、尺度表示及实现可得性；ViVLA随后，保留原方法功能，不拿缩水复现代替官方方法。
资格与实际预算待前项通过后登记，不提前提交昂贵外部训练。共享split与合法数据，成本包括专家及前置训练。
最终图文报告包括主表/逐task、视频controls、FT/RL曲线、编译/适应/推理成本、失败案例与原始行。
如触发暂停，先交付该阶段事实与待Owner裁决事项，不虚称整套goal完成。
Git按main集成推送、clean detached runtime；数据/source/tokenizer复用。大资产前查strg01独立quota及峰值。
本阶段预计三能力面板仅一套400-LoRA bank约2GiB，加日志和余量预留5GiB；controls另四bank约8GiB，后续launch前重估。
保留正式manifest/raw rows/aggregate/checkpoint来源，bank为可重建临时资产，用完确认无依赖后删除载荷；不删除唯一科研证据。

## 6. 后继接口只读核查（尚未运行适应实验）

FT/RL的共同缺口是合并共享MT-BC后重新安装fresh rank16。现有`source_sft/inference.py::FrozenSourceSFTAdapter`只安装rank128，
`lora.py::inject_task_lora`明确拒绝已有LoRA；不能直接叠加第二套adapter，也不能假设返回的是有`merge_and_unload`方法的PeftModel。
已安装PEFT的`inject_adapter_in_model`返回被就地修改的原policy；官方tuner合并逻辑先merge各层、恢复base层，再删除peft_config。
后续实现须对38个合法target完成相同步骤，冻结合并后的底座，再用现有identity初始化安装rank16；
数值资格比较相同observations/state/noise的动作输出，允许正常BF16舍入，不要求逐bit一致。优先进程内合并，避免每task复制完整9GB底座。

既有`expert_manifold/expert_training.py`是训练任务专家入口，绑定train任务清单及多演示dataset；
不能靠关闭角色检查把它变成held单support FT。可复用FM batch优化、offset1数据处理和完整checkpoint，
但support、独立选点episode、eval50及目标局部梯度权限需由独立且明确的适应合同绑定，共享训练的信息墙保持。
