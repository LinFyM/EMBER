# EMBER progress

## 当前工作（2026-09-17，统一Writer整套实验执行goal）

Owner授权仓库/data1整理、新架构实现与高效训练、与已有v5.2结果比较，以及有具体依据时修正重训，
全部结束后详细汇报。此前仅分析限制已被替代；goal仍active，无关旧实验不恢复。

**最新纠正：v5.2已经训练过，不得重训或续训。** 代理误把比较扩成fresh基线，于22:29 CST停止该重复run，
最后完整更新73，三rank全部退出、gpu01 p0/1/4释放；没有100步checkpoint或闭环结果。
误启动日志及处置保留在`runs/analysis/unified_writer_20260917/baseline/owner_scope_correction.json`。
对应launcher、待执行请求和active重训要求已撤销；该中止run不作为科学比较结果。

Owner最新明确要求：**新架构必须用六卡，并保持等效计算来提速**。此前代理建议四卡已被此要求替代。
每step仍是4task×21 queries＝global84，source/采样/FM/权重和一次optimizer更新不变；
已实现视频native帧分片与query分片，六卡参与实际计算。两组三卡分担四个条件，
组内在j9/j18统一表示处进行可微汇集，保留完整跨帧联合处理；梯度等效、实际吞吐和恢复验证通过。

当前阶段：**新Writer六卡正式训练首段0→600**。2026-09-17 23:23 CST已从clean pushed detached
`184947cb` fresh启动，gpu02 p0/1/2/3/4/6，tmux `ember-unified-native-train`。
正式run contract已核实六rank、两组三卡、global84与source trainable=0；每100保存完整状态，600后执行correct400/train96。
Owner已撤销补300步评测的要求；300请求未执行即撤销，保留取消记录，首轮600及后续每300步的节奏不变。
原生三进程Gloo检查覆盖2/2/1和1/1/0帧分片、已打开Meta/写回与checkpoint重算，输出和汇总梯度符合串行目标。
完整FM的query切片保持原始随机batch/offset与汇总余切；1–4及6rank的任务分配验证保持4条件/84queries。
六卡已完成3→6完整恢复，24逻辑条件/504queries无重复计数；集成训练/物化/native检查66项通过。
Owner进一步要求提高显存利用后，已实测8/12/16/20帧chunk并选择20；配置已登记为formal可执行，正式权重仍须fresh。

## 已完成的整理与实现

源码整理在clean pushed `bf8aea37`交付：旧ECP捕获合同、SFT在线验证、重复functional wrapper和结束A配置已退役；
相关110项检查及训练入口通过。已删除23,224个可重建LoRA payload，准确释放96.322GiB，
另移除13个干净、已集成且无运行依赖的临时工作树。保留400个外部硬链接payload、一个评测不完整的bank、
9个有未合入/未提交历史工作的worktree，以及全部formal checkpoint/raw rows/数据/source/专家材料。
清理逐项原件：`runs/analysis/workspace_cleanup_20260917.json`。

统一架构已集成：`11e96245`为temporal/model，`14432969`为native encoder，`199eade0`为runtime/schema。
实构造13,451,008参数；旧Core/P/AdaLN运行路径已替换，仅一个canonical Writer。两个实现worktree均已集成并移除。
集成native/temporal/LoRA检查35项、训练/恢复/物化/评测相关196项通过。结构检查无hard，接口校验保持集中所有权。
`5800c2dc`将冻结视觉embedding移出native checkpoint反向重算，14项相关检查通过；正式配置guard两项通过。
`dac5ee70`增加原生紧凑网格的可微汇集，无参数/schema变化。task_execution中已无调用的旧cache复制/mmap规划退役；
同一cost-balanced调度器负责条件分组，不新增并行runner或模型实现。原有长调度函数保持单一、确定性的分配职责。

## 实测执行配置与运行入口

六卡gpu02 p0/1/2/3/4/6，两组rank(0,1,2)/(3,4,5)。相同前6个注册更新、8帧chunk下，
热身后六卡平均8.6068秒，四卡12.5173秒，吞吐提升1.454倍；六卡完整恢复至6，峰值reserved19.87GiB。
最长105帧所在的实际4task/global84面板为task19/16/35/38、帧数52/31/48/105：
8帧chunk热身15.52秒；12帧15.28–15.66；16帧15.15–15.48；20帧三次15.16–15.19，选择20。
20帧实际峰值allocated34.04GiB，allocator reserved最高43.77GiB；共驻卡按可用容量回收缓存，实际分配有余量。
采样中本任务SM：p2约71%，其余五卡85.5–92.7%；p2原进程约29.4%，与此前约26%接近。
这是活动采样，不冒充他人作业的独立吞吐证明。三Meta、双写回及两处联合块皆有信用，source始终冻结。
正式初始化不复用profile权重。统一原件入口：`runs/analysis/unified_writer_20260917/unified/six_gpu_profile.json`；
旧单卡/四卡证据保留为profile历史，不构成当前资源配置。

Study根为`/data0/user/ymdai/ember_runs/unified_writer_20260917`，仓库入口
`runs/analysis/unified_writer_20260917`为symlink；复用canonical source/data，不复制大资产。
最新strg01 data0用108,502,252KiB/soft1,073,741,824KiB，shared余1,358,981,960KiB；
个人ember_runs实际54,656,882,052字节。预留新增峰值上限200GiB覆盖新方法、必要诊断及至多一轮集中修正。
这些是launch前快照，资源变化时刷新；data1独立预算不混用。

新架构配置为`configs/pi05_writer.json`，外部sealed copy为study的`unified/config.json`。
冻结runtime在`.codex/tmp/unified-native-runtime`；`unified/launch_600.sh`为六卡首段入口，正式配置为20帧chunk。
仅新方法的14份物化请求已校验：600/900/1200/1500/1800/2100/2400，每节点correct400和train96。
映射seed20260911；validation每task50条teacher各一次，train为states32–35和held视频46–49。
`paired_panel_registration.json`登记范围，`materialize.sh`/`evaluate.sh`仅接受unified。

## Active design与裁决

唯一active design：[统一Writer设计](docs/v52_evidence_based_writer_design.md)。新架构fresh共同纯FM学习；
v5.2复用既有曲线/固定step900复核，同source比较复用对齐A的已完成面板，不新增v5.2训练。
比较需明确source、camera/H、recipe及曝光差异，只有合同相容的已有rows才做strict paired计算。
不以不同设置下的分差孤立归因于某个模块，不用最终controls返工架构；Test关闭、无RL或held梯度。
正式性能仍须correct严格>145/400及相邻稳定、breadth、四suite/GoalLong和同task视频鲁棒性；尚无新模型闭环分数。

旧A训练完成3000、最新完整评测2700为108/400与train64/96；3000评测不自动恢复。
完整历史与设计证据见findings§107–116、[证据审计](docs/v52_evidence_audit_20260917.md)及
[研究历史](docs/research_history.md)。旧长篇状态由Git `176759a7:progress.md`保留。
