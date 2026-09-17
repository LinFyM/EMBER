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
现在实现视频native帧分片与query分片，六卡参与实际计算。计划以两组三卡分担四个条件，
组内在j9/j18统一表示处进行可微汇集，保留完整跨帧联合处理；以梯度等效、实际吞吐和恢复验证裁决实现。

当前阶段：**六卡等效执行实现**。新架构本体及四卡profile已完成，formal尚未启动。
配置已回到pending_six_gpu_profile，六卡代码、验证和profile完成并封存后再启动新方法。

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

## 实测执行配置与运行入口

最长train视频105帧、完整21-query FM：优化后8/8物理分块热身24.71–24.89秒，reserved20.62GiB。
frame12仅约2%更快且额外占9.8GiB，query12更慢，故选择8/8。第3次更新确认中层联合块有梯度；
三Meta、双写回、decoder与完整76输出正常，source始终冻结。四卡gpu02 p0/1/3/6完成1–3并完整恢复至6，
热身/恢复更新9.42–14.16秒，峰值21.025GiB，24条件/504queries。正式初始化不复用任何profile权重。
原件在`runs/analysis/unified_writer_20260917/unified/native_profile.json`及`ddp_profile_*`。

Study根为`/data0/user/ymdai/ember_runs/unified_writer_20260917`，仓库入口
`runs/analysis/unified_writer_20260917`为symlink；复用canonical source/data，不复制大资产。
最新strg01 data0用108,497,784KiB/soft1,073,741,824KiB，shared余1,359,764,568KiB；
个人ember_runs实际54,692,421,632字节。预留新增峰值上限200GiB覆盖新方法、必要诊断及至多一轮集中修正。
这些是launch前快照，资源变化时刷新；data1独立预算不混用。

新架构配置为`configs/pi05_writer.json`，外部sealed copy为study的`unified/config.json`。
冻结runtime在`.codex/tmp/unified-native-runtime`；`unified/launch_600.sh`尚为未执行的四卡模板，六卡profile后替换；当前配置拒绝formal启动。
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
