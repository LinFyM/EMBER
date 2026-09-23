# 原生动作读出：视角与t=1端点诊断（2026-09-13）

> 文档角色：历史研究分析。原文中的“当前／active／下一步”仅指当时阶段；当前授权与暂停状态以[progress](../../progress.md)为准。保留原科学定义和结果，不据此恢复执行。

本项已完成并关闭；执行状态由progress登记。不重开已关闭的state补全实验。

## 1. 为什么需要这个比较

上一诊断证明冻结source在state-free双相机RGB＋语言条件下完整10步采样的动作MSE为.13672735，低于动作均值.25059744。
但现行执行对齐Writer只读agentview、采用t=1完整H，存在视角和读取阶段两个区别；不能把差异单因归给采样步数。
原run_contract的observer.camera_view=agentview、probe_seed=1729；此处直接复用这两个已封存字段。

当前FM以`x_t=(1−t)a+tε`为输入、`ε−a`为监督速度。因此在t=1，x_1=ε，
平方风险最优的速度为`v*(ε,1,O)=ε−E[a|O]`，所以`ε−v*`就是条件动作均值。
这是独立Gaussian噪声、该插值路径和理想条件期望下的代数，不声称有限source在端点达到最优。
完整生成样本的有限均值还含采样误差，不能由步数更多推导其MSE必然更低。

原生速度由`action_out_proj(H)`产生，当前观察器正是读取此投影前的完整50×1024 H。
若当前t=1 H经已存在的原生输出头即可形成有用估计，便缺少把瓶颈归因于“没有做完denoise”的依据。
这不保证后继随机投影、Meta、E或Compiler保留知识，更不证明预测动作等于演示中实际完成的过程。

旧v4已经10步积分（`f2163eea:src/ember/writer/action_forecast.py:302–323`）。
其完整复审`3a6f801d:docs/action_forecast_writer_v4_root_cause.md`§3–10既记录Meta校准恶化，也通过
目标选择／抓取回放定位了绝对时间绑定下的错误控制偏置；仅换成frame-local Intent＋Transition的提案随后被撤回。
因此本项不重建旧forecast编译图、不假设latest更准或响应差是纠错，也不借历史controls运行新controls。

## 2. 冻结合同

复用[source输入诊断](source_state_input_audit.md)的全部384个train24位置、state-free exact language、
source、normalization、15×7后续动作和8个配对噪声。真实state不进入本项forward，无Writer／Meta／新梯度或rollout。

固定两因素四单元；每个视角内比较读取阶段，每个阶段内比较相机范围，不用同时改变两项的对角差额命名根因。

| 视角 | t=1原生端点估计 | 完整10步生成 |
| --- | --- | --- |
| agentview | 新增，当前观察器的RGB范围 | 新增 |
| dual | 新增 | 复用上一state_free全部原件，不重复生成 |

只新增一个计算端点：t=1，一次原生denoise，用`ε−vθ(ε,1,O)`估计动作。通过原生采样API的单次Euler更新
实现该公式，不将其当作拟采用的一步rollout sampler。完整50H仍由source计算，只评分前15×7。

两个t=1单元固定报告两种噪声口径，不在它们之间选优；两个完整生成单元均保持原8噪声均值：

- `paired_mean8`：与上一诊断相同的8噪声和均值，用来比较端点估计与完整生成均值。
- `public_probe`：现行执行对齐Writer配置的observer.probe_seed所生成的唯一50×32 Gaussian，所有位置共用，
  对应未适配source下现行观察器的固定probe口径。种子从该封存配置读取并记录，不重采样或择优。

仍只在输出后读取动作评分；同一批9个噪声可并行，以免重复图像／prefix计算。已有完整采样结果直接复用。
两种端点与动作均值、同视角完整采样的差额都完整报告；相同读出下的视角差额亦报告。
固定seed20260915、20,000次task-cluster paired bootstrap，每task16位置等权、24task等权。

## 3. 预登记解释与停止

- 若agentview两种端点估计相对任务动作均值的差额95%CI均严格为正且至少两suite为正，说明当前RGB范围的t=1 H
  有可直接读出的动作预测价值；停止“缺少完整denoise是当前首要接口缺口”的投入依据，不扩大flow计算。
- 若agentview端点未通过该项，而同视角完整生成相对动作均值的差额区间严格为正，支持端点校准／固定probe消费受限；
  仍须处理预测与真实过程、跨初态绑定及可学习性问题，不能直接启动新Writer。
- 若agentview两种阶段均不能胜过动作均值而dual可以，则先保留视觉范围限制这一解释，不将对角差额归因flow深度。
  任何视角修正仍须对照既有双视角历史，不因本局部结果直接启动新Writer。
- paired_mean8与public_probe不一致时，记录差异，不改seed／样本数，不从单个较好分数选择新的probe。

本项没有full10→其它步数序列或flow-time搜索。它比较的是正在消费的t=1接口与已测的完整source输出，
不以更好MSE选择部署sampler、checkpoint或新的时序架构。一次完成后关闭，临时入口退役。

## 4. 资源与复核

复用原样本／资产与6fa6177f冻结诊断loader，新入口由clean pushed detached树执行；新输出放既有
`runs/analysis/source_state_input_20260913/`的endpoint前缀，不覆盖三臂原件。
同节点最多三张合适A40独立执行三个新增单元，无NCCL；现场核对两节点GPU与data1独立quota，
预估额外峰值1GiB，含约220MiB代码树与小于8MiB预测。
保存原始逐位置预测、固定probe种子、配对差额、所有task／suite、退出状态和精确运行命令，不训练状态或动作头。

## 5. 完整结果与裁决

三个新增单元各384位置全部exit0，新增1,152位置；既有dual/full10完整复用。逐行身份、noise、动作区间和
任务均值配对通过；保存的8／9个原始预测能重建所有评分。没有梯度、Writer／Meta、LoRA或rollout。

| 视角／读出 | 动作MSE | 相对任务动作均值的改善及task-cluster95%CI |
| --- | ---: | --- |
| agentview / full10 mean8 | .36548145 | −.11488401 [−.15451953, −.08161489] |
| agentview / t1 mean8 | .34788516 | −.09728772 [−.13581532, −.06455726] |
| agentview / t1 public | .34731886 | −.09672142 [−.13454582, −.06442142] |
| dual / full10 mean8 | .13672735 | +.11387009 [+.08509872, +.13946848] |
| dual / t1 mean8 | .13217241 | +.11842502 [+.08891629, +.14417254] |
| dual / t1 public | .13176813 | +.11882931 [+.08965598, +.14429032] |
| task action mean | .25059744 | 参照 |

agentview三项均在四suite落后任务均值；full10为23/24task落后，两种t1均20/24落后。
dual三项均在四suite胜过任务均值，均22/24task为正。
同读出下agentview−dual差额：full10 +.22875410、CI[+.20100961,+.25719621]；t1 mean8 +.21571275、
CI[+.18717154,+.24534044]；t1 public +.21555073、CI[+.18742479,+.24470501]，三项均24/24task为正。

同视角下，agentview的full10反而比端点更差：相对mean8/public为+.01759629/+.01816259，区间均严格为正。
dual的full10−t1 mean8为+.00455493、CI[−.00002963,+.00888925]；public为+.00495922、
CI[+.00025245,+.00956361]。这些差额完整保留，不在mean8/public之间选优，也不选择一步部署sampler。

**触发预登记第三分支：视觉范围限制优先保留，停止以增加denoise深度作为当前修复依据。**
当前agentview的失败与双相机t1的正结果同时成立；不能把后者写成现行agentview观察器已充分。
本项识别的是输入相机范围对冻结source动作预测的影响，尚未区分新增腕部内容、训练分布匹配及其交互；
没有识别视频动态、Meta适配、跨初态编译或闭环收益，也不能由预测差额推出新Writer会通过。

代码冻结e1a06b46，loader仍6fa6177f。三个新单元循环约302.89/248.95/249.84秒，allocated峰值9.88/10.04/10.04GiB；
单节点三卡，已退出。全部原件在`runs/analysis/source_state_input_20260913/endpoint_*`，包含raw/samples、
逐task／suite及所有配对差额、独立summary脚本、READOUT和launch contract。一次诊断已关闭，临时active入口退役。

## 6. 后续机制判断的边界

既有双视角能力由99ee2d03实现；[Horizon设计§3.1](../designs/horizon_relation_video_writer_design.md#31-同episode同步双视角输入2026-09-10读取能力扩展)
明确其交付只是输入实现与验证，不能当作已经完成双视角学习的正负证据。当前V-JEPA先验合同仍限定agentview，
直接把observer配置改为dual会被拒绝；这属于登记的输入合同，不是本轮发现的软件bug。
后续先审计已有run contracts与双视角历史范围，再推导保持单一LoRA及同一视频先验的最小输入修正与停止条件；
本项不自动重启训练，不把旧双视频K比较误认为双相机比较。整体goal未完成。

只读库存审计覆盖`runs/outputs`下一／两级的1,669份`run_contract.json`：38份内嵌observer，
其中4份显式agentview、34份沿用缺省agentview；其余1,631份没有内嵌observer，不能据此判定相机范围。
在这38份可判定的合同内未发现dual学习记录，不把有限范围库存结论外推为全部历史绝无双视角实验。
逐份路径与状态保存在`endpoint_camera_contract_audit.json`及其脚本。下一步是据此完成最小输入修正的设计判断。
