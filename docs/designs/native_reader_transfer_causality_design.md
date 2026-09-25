# 数据干预造成的控制变化：原生读取 Meta 与其余 Writer 的冻结交叉

2026-09-25。是否 active 只看 `progress.md`。机器合同为
`configs/native_reader_transfer_causality_v1/experiment_spec.json`。
主讨论 `01a0cd94-65da-7b22-8ca9-7ba35f454632` 负责判断，现有 Sol
`01a0cd90-ebb7-77a1-a20b-a858825d2f66` 负责实现、执行、原件及完成 Queue。

## 1. 由什么证据触发，解决什么选择

关系支持 stage1 的1500原始行已由主讨论独立复核。预期的“补齐两条关系同时带来正交互与绝对改善”未兑现：
Goal correct 的 C00/C01/C10/C11 为36/37/26/35，other为40/37/29/33；
correct交互+16pp但CI[-8,40]，C11−C00为−2pp；other相应+14pp和−14pp。
新增四项支持操作大多17–20/20，不能再把全部问题解释为这些操作完全没学会。
但Source本来就有较高支持能力，旧共同边黑碗→盘子也未单独验收，不能声称整条组合路径已经获得可迁移表示。

更可区分的新事实是：固定 j=0，i 的任务替换77→76使Goal correct下降20pp、other下降22pp。
同原件的后验连续几何显示C00→C10：25步EEF到目标body的水平距离均值4.62→7.05cm，
目标持续位移45→35，干扰物先持续位移5→16（2cm、连续5控制步）；1/3cm口径方向亦同。
这些是描述，不是抓持、语义标签或完整中介证明。Object仍38/50，且50条均曾移动目标，故不是全局运动能力消失。
完整裁决与原件入口见 findings §143。

本批问题是：**这项已观察到的数据敏感性，主要经原生读取适配改变了输入证据，还是经其余Writer改变了证据到控制的映射？**
结果决定是否值得优先研究原生读取校准，或继续研究条件映射的学习约束。不是先假定某模块坏，再寻找支持。
不扩大训练池、换loss或重训。C00与C10是本次数据干预端点，不是择优checkpoint；两者均固定1260。
端点选择是stage1结果后的诊断选择，不伪装为stage1事前主比较或独立确认性结果。

历史约束必须保留：

- §44–46旧P/C/D有限更新和端点交叉已见读取、解码及交互的异质作用，反对从单个混合失败判定模块损坏。
  本批不重复旧三模块扫描；新比较固定训练时长/seed/模型，只改变一个明确训练任务，并在固定原生坐标边界做两组交叉。
- §118–119旧正确P在错误Core下仍能贡献能力，§128静态末P修复与匹配续训没有额外净收益；不据当前结果恢复P-only路线。
- §129当前private Writer不弱于freeAB，缩head更新未修复；不把外部Writer作用解释成容量不足或建议再缩步幅。
- §135–139早期xy有局部因果作用，但最终action_out开关未修复Goal且损失Object；不继续扫描动作轴或执行侧层。
- 旧v5.2普通FM和同视频教学存在有限正证据；当前阴性不证明视频、联合Meta训练或整个架构不可能有效。

## 2. 计算边界与数学预测

把部署前生成计算写为

\[
 z=R_N(L,V;\theta_0),\quad \Delta=G_W(z,L),\quad
 F_{ij}(q,V)=\operatorname{Flow}_{10}(q;\theta_0+G_{W_j}(R_{N_i}(L,V;\theta_0),L)).
\]

此式按实际代码的数据流解释，不引入新的模型模块。N **仅**是
`semantic_encoder.{text_meta_lora,vl_meta_lora,action_meta_lora}` 的全部A/B参数；
W为其余所有可学习参数，包括language/interaction projections、patch grounding、Core、Procedure、Compiler及八FactorHeads。
N不是整个视频encoder；W也不是仅最终输出头。冻结source、template buffers和fixed_suffix_noise不属干预变量。
现有checkpoint原件共622 tensors：N为432 tensors/2,469,888参数；共同buffer为76 templates和1 fixed_suffix_noise；其余113 tensors为W。
实施时核对实际名字/shape/角色，不因未来键名漂移静默按“剩下的都算W”继续。

0表示C_S00@1260，1表示C_S10@1260。四格：

| 条件 | N来源 | W来源 | 含义 |
| --- | --- | --- | --- |
| N0_W0 | C_S00 | C_S00 | 新批内部self参照 |
| N1_W0 | C_S10 | C_S00 | 只转移原生读取适配 |
| N0_W1 | C_S00 | C_S10 | 固定其余Writer，只撤回原生读取变化 |
| N1_W1 | C_S10 | C_S10 | 新批内部self参照 |

三组Meta在冻结source原生层坐标中安装；不交换两个独立学习的256维外部latent，不平均最终LoRA，也不产生第二执行adapter。
混合权重没有共同训练，仍可能有分布不匹配或协同适应；固定坐标不消除这种解释。
混合仅是冻结因果诊断，不作为部署候选、checkpoint融合方案或训练课程。

令p_ij为同一任务的闭环成功率。报告以下全部预定对比：
N在W0/W1下的效应`p10−p00`、`p11−p01`；W在N0/N1下的效应`p01−p00`、`p11−p10`；
交互`p11−p10−p01+p00`；完整端点差`p11−p00`。同时明确报告恢复方向`p01−p11`和`p10−p11`。

- **H_N：原生读取变化传递主要损害。** Goal的`p10<p00`且`p11<p01`；撤回N应恢复正确闭环，
  不能只有native activation差异或动作向旧端点偏移。Object用于检查恢复是否牺牲另一种能力。
- **H_W：作用主要在其余条件映射。** `p01<p00`且`p11<p10`，撤回W比撤回N更符合恢复方向。
  这不进一步区分projection、Procedure、Compiler、heads，也不证明容量限制。
- **H_joint：共同适应/非线性交互。** 交叉后两边都更差、两个上下文效应反向或交互明显。
  此时不归罪某一边，不据此冻结模块；只登记当前两分法不能给出可迁移的单边修复。
- **原端点差未在新批复现。** 新p00/p11若方向或成功集合大变，先报告重放差异与区间；
  不把旧36/26代入新主比较，不换成另一组端点追逐结果，也不自动增加重复次数。

功能读出只用各实际rollout的**第一轮**真实10-flow输出：此时四格q/初态/语言/噪声相同，后续各自到达状态不当作同query。
记录full50×7和实际前5环境动作，主读数为前5的xy，另经相同canonical OSC裁剪/缩放作执行命令参照。
逐state定义

\[
 D=F_{11}-F_{00},\quad
 E_N=\tfrac12[(F_{10}-F_{00})+(F_{11}-F_{01})],\quad
 E_W=\tfrac12[(F_{01}-F_{00})+(F_{11}-F_{10})].
\]

核对`D=E_N+E_W`，报告两者norm、signed xy、交互向量及
`sum <E_N,D> / sum ||D||²`（分母实际为零则null），W同理。
投影可负或大于1，不能叫失败归因百分比，也不用它选模型或决定闭环是否算通过。
这是有限冻结作用的精确分解，不把一阶Jacobian或参数norm当真实功能贡献；闭环结果优先。

## 3. 数据、评测与规模

只用关系支持C_S00/C_S10完整1260checkpoint，训练commit仍`7dc95edbba00cf61439700d77fb321eb8df95c07`。
Source与normalization复用原研究1000节点，不改变实际内容；不要恢复旧policy已读目标40动作的权重。
原两endpoint的metadata/配置模型拓扑相同，训练pool不同需原样登记。

- 两任务固定global14/Object local4、global21/Goal local1，均属已有Train诊断任务；每格state0..49。
- 复用原stage1 exact language、correct50-video与state映射、env/policy RNG及全部canonical评测规则。
  每任务每格50条teacher各一次；不按失败state筛选，不从其它任务借视频。
- 四格共400新闭环。不是原stage1扩容或重写；四格在一个新clean pushed detached evaluator实现上完成。
  新批内部self参照用于保留已知重放波动，旧200条只作独立历史差异附表。
- endpoint银行可只读复用原E3合法bank；两个混合格各100完整LoRA需重新读取真实视频并生成。
  从不平均/拼接最终adapter。逐bank记录N/W来源及实际物化commit，不能把混合权重冒称某个训练checkpoint。
- 16固定双相机full病例：四格×两任务×state0/25；全部400条保留既有compact action_chunks/执行动作、
  T+1 body/EEF/夹爪与BDDL谓词。复用已修复的passive采集，不引入新感知标签或新物理干预。
- 不新增teacher expert action/state读取、gradient、optimizer、训练、官方Validation/Test、wrong/shuffle/reverse、
  其它节点/任务、3300预测或RL。没有按视频或混合格选择部署模型。

所有对比按task逐state共同bootstrap20,000次、seed20260925；一个训练seed的条件性描述区间，不覆盖重训方差，未作多重比较校正。
报告per-task、R/G/L、churn、success-set Jaccard，完整保留两端点相对stage1每state差异。
连续几何复用stage1已说明的1/2/3cm且连续5步位移指标、25步EEF水平距离/高度，并保存真实数值；
它们是描述性读出，不自动命名抓取/语义错误。仅两个任务，不称held400、四suite保持或有益视频修复。

## 4. 工程验收、资源与停止

Sol从最新main隔离开发，复用唯一Writer runtime/materialization及canonical evaluator，不复制trainer或另造policy执行面。
推荐在加载完整父Writer后按显式N/W名单替换对应参数，source/template/noise只读；也可用等价、经过验收的现有机制。
不落盘伪造带optimizer的混合checkpoint。项目源码的结构性增加按code-architecture-gate；不为本批新增通用监控框架。
正常运行等待退出事件，若当前等待被中断，不以结束任务替代交接；完成后主动Queue主讨论。

正式前验收：

1. CPU检查两父配置、622键的角色划分、shape、有限值、模板/固定probe身份、信息墙和teacher/state调度。
   参数来源逐键可追溯；仅为实际干预检查N/W载入，不做无关全树hash或全模型逐bit核查。
2. 同源组合的真实Writer输出应恢复对应parent行为。固定task14/21、state0/25四条件，
   新构造self与直接加载parent走同次dtype/batch的动作前缀比较；接受正常数值差异，若超过0.03 normalized action maxabs先查具体实现，不扩大容差掩盖错误。
   混合生成全部76A/B，真实source始终冻结，不安装两套执行adapter。
3. 最多4条非正式工程闭环（common non-held task58、state0；完整真实采集/适配接口），不作科学分数。
   正式pilot8=四格×两目标×state0，计入400，不重跑。检查worker正常退出、RNG/身份、T+1及full/capture；
   pilot只作工程准入，不按成功分数改模型、规模或对比。通过即执行剩余392。
4. 所有新正式rollout统一一份clean pushed detached实现；银行来源例外须如实登记（原endpoint E3和新混合实现）。
   代码开发/集成及时推main；旧所有formal树、bank与原件只读。实际命令/env/来源/资源写入一份launch合同。

新根`/data0/user/ymdai/ember_runs/native_reader_transfer_causality_20260925`；创建大产物前由Sol查询strg01独立quota、共享容量，
测现有占用并估计峰值。新增data0上限8GiB（含混合bank、轨迹和临时文件），data1新增代码上限768MiB；不复制source、数据或父checkpoint。
每次launch实时核对gpu01/gpu02，项目合计至多6物理卡；利用能增加吞吐的空闲/可共驻卡，不等凑卡，不跨节点拼训练。
仅物化/独立评测，禁止dummy占卡与例行进度轮询。profile从真实操作计时估算ETA，不先承诺全流程时长。
正式计算预算上限12 GPU-hours，正常预期显著低于该上限；逼近上限或工程缺项时停止受影响工作回报，不能自行扩容。

完成输出 `analysis/completion.json`：400唯一有效行、16full、400完整连续trace、来源/配对/worker退出、
全部两任务八个预定对比及初态函数分解、几何原件、历史self差异、实际资源与缺项。
完成即主动Queue主讨论并停止新增实验。结果可能只定位这一次数据敏感性的传递路径；
不把任意混合格涨分直接宣布为统一根因或合法部署修复，不自动接fresh冻结Meta、扫层、续训或完整评测矩阵。
