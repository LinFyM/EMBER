# 冻结策略的前段轨迹与视频条件因果诊断

> 2026-09-24完成并关闭：1600分支已独立复核，结果和解释边界见findings§136。原冻结合同保留，不据此重复执行；当前状态见[progress](../../progress.md)。
> 本批不训练、不选checkpoint；分支执行是定位用的干预，不是EMBER部署候选或新的正式400成绩。

机器规格：`configs/frozen_prefix_causality_v1/experiment_spec.json`。
研究根：`/data0/user/ymdai/ember_runs/frozen_prefix_causality_20260924`。

## 1. 依据与要区分的解释

上一批54面板13200条原始行经主讨论独立复核，见findings§135。语言B后四节点均高于直接A，
不支持“生成链普遍无法学到直接LoRA能力”的强解释。C@420 correct/other/wrong为110/98/126；
C相对Source仍获得76个成功，但丢失24个，B@630丢失10个。Goal21（碗→炉子）Source40/50、
B@630为42、C@420 correct25/other25/wrong36；Object14（ketchup→basket）Source0、B45、C46/45/46。
这两个任务分别提供保持受损和能力获得的对照，不能只在一个受损任务上把抑制适配器当修复。

固定full case Goal21/state0在630节点：B成功，C可见把盘子拿到炉子上，D成功；D1260的该case也失败。
这只是提出对象选择假设的个案依据。9月14日Semantic-Path完整128回放已有错实例、抓取/运输、阶段保持等异质失败；
9月13日空间信用Q/K监督曾提高一般能力，Goal仍低于Source。因此本批不从图片直接推出“加grounding模块”或统一occupancy根因。
历史完整解释见`docs/analyses/semantic_path_behavior_replay.md`和v52 evidence audit的E32–33；
9月22日保存动作回放的工程依据见`docs/review_materials/20260922/writer_causal_diagnostics/behavior_pairs/README.md`。

竞争解释及预测：

- **H_state：早期动作引起的状态历史主导失败。** 给C同一个B前段产生的状态，会显著改变后续成功；
  若C在该共同状态接手仍持续劣于B，尤其不同视频仍改变接手后的对象行为，则纯粹“已经走偏”不足。
- **H_condition：冻结C的条件→执行函数在相同状态下仍受不合适教学条件影响。** 固定前段轨迹、当前观测与后续RNG，
  correct/other/wrong仍产生方向一致的行为及成功差异。若差异只在C自己的历史上出现、B前段后消失，则此强版本被削弱。
  wrong跨suite同时改变场景、对象和内容；即使成立，也只定位有害条件作用，不能称已经识别动态语义或某个Writer模块。
- **H_control：目标对象选择基本相同，差异主要来自接触、运输或恢复控制。** 若目标对象确实先被移动，随后才失败，
  则“错误对象选择足够解释差距”不成立。几何移动、接触、抓持和最终目标分别记录，不能混为同一个标签。

这些解释可并存。两个有意选定的诊断任务、单seed不能推出全40任务的统一原因。

## 2. 数学干预

固定LoRA下的实际动作前缀写作`a_t=F_theta(o(s_t), l, xi_t)`，`F`包含真实10步flow及反归一化；
环境连续执行5动作后为`s_(t+5)=P_5(s_t,a_t)`。一阶扰动可写成
`delta s_(t+5) ≈ (D_s P_5 + D_a P_5 D_s F) delta s_t + D_a P_5 D_theta F delta theta`。
此局部关系不是跨接触/对象切换的全局线性模型；它说明不同自然rollout混合了状态历史与策略函数两个变化源。

定义`Y(q→p,k)`：从原相同init完整重放锚策略q已保存的前k个实际环境动作，然后由冻结p接手，
总horizon和原RNG时钟不变，按原成功即终止规则得到结果。每个分支都重建相同历史，不只覆盖8维proprio。

- 固定q/k，`Y(q→C_correct,k)-Y(q→C_wrong,k)`隔离**同一已实现状态历史上的视频条件效应**；other为另一合法教学实例。
- 固定接手策略p，`Y(B→p,k)-Y(C_correct→p,k)`测量干预前段历史的总效应。
- `Y(B→C_correct,k)`与`Y(B→B,k)`比较同一B历史下接手函数；`Y(C_correct→B,k)`检查B能否从C历史恢复。

两锚策略的状态并不彼此相同，不能把跨锚差值说成同状态动作效应；也不做有跨世界假设的“自然直接/间接效应”分解。
B@630只是能力参照和前段轨迹提供者，B/C训练节点不同，不由这一比较估计架构或目标的纯因果效应。
演示分布MSE只约束有限`d_expert`上的误差；本次操纵的是实际环境状态历史，不新增held动作误差读数。

## 3. 冻结数据和模型

- 全部复用四臂研究`conditional_compilation_diagnostics_20260923`及其原始run contracts和banks。
- 任务global IDs固定`[14,21]`，各50个init states `0..49`，不按单行成功筛选。它们是已看过结果的diagnostic-held Train任务，
  明确为后验定位任务；不是新的盲测，也不接触官方Validation/Test。
- 锚策略：`B_language@630 correct`、`C_video_fm@420 correct`。
- 接手策略：上述B，以及同一C@420的correct、same-task-other、cross-suite-wrong。每个条件只加载一套完整LoRA。
- B bank：原`materialization/B_language_630_held/manifest.json`；C三者为原`C_video_fm_420_{held,other,wrong}/manifest.json`。
  复用原teacher/state映射及已物化权重；不重新选视频、不平均/缩放/融合LoRA、不生成新模型。
- 原k=0完整结果仅复用上述4面板中两任务的400条原件。新增k固定为`25,50`个实际环境控制步（不含settling），
  总计`2 tasks × 50 states × 2 anchors × 2 cuts × 4 followers = 1600`个分支条件。
- 不读取teacher actions/state/reward来监督或选点。原policy执行动作可用于重放；simulator对象位姿只用于诊断记录，
  不进入Writer或policy新增输入。Source、normalization、tokenizer、exact language和官方执行合同不变。

## 4. 实现与真实性检查

从最新main隔离开发；复用既有pi05_eval的环境初始化、动作变换、policy加载、stateless RNG、动态队列和capture。
新增能力限定为登记的保存动作前段重放与接手，不另复制一套完整evaluator。非平凡源码变更使用code-architecture-gate。
原`43d801b1`冻结树和四臂原件只读；新正式运行必须使用验证后集成main并推送的clean detached commit。

1. 每分支按原seed/init和dummy settling10初始化，然后重放对应原轨迹`executed_action_prefixes`。
   该字段已经是实际环境动作，不得再次反归一化；`action_chunks`是另一个表示，不能直接替代。
   每段按`replan_steps`取到k，不能填零、跳帧、只改robot state或重复settling。
2. 在原replan节点与保存的8维state比较；位置/夹爪绝对误差≤1e-4，axis-angle≤1e-3。
   这是动作重放合同检查，不要求模型逐bit一致。超出则停止受影响的分支，保留实际偏差，不放松阈值凑通过。
3. 同一锚/截断的所有followers必须得到相同历史和状态；记录simulator full state及可用controller/goal状态以供核对，
   但正式分支通过重放得到这些状态，不依赖不完整snapshot恢复。动作切换只发生在5步replan边界。
4. 接手后采用原policy噪声序列的`k/5`位置继续，不重置RNG。噪声由原root seed、suite、task、init state和
   全局replan index经canonical `policy_noise_seed`生成；原轨迹若较早终止，仍按同一函数继续索引，不能重复最后一个
   已保存seed或因保存列表用尽而截短新分支。已保存区间核对相同seed；总执行时间仍是suite horizon，成功立即终止。
   若前段已成功则不强行继续，四个followers统一记为prefix-terminal；总分与排除此类条件的接手分析分别报告。
5. 先完成两任务states`0,25`、两锚、两cut的16个同策略接手分支，作为正式矩阵的一部分验证重放、时钟和记录。
   与原行的正常微小推理差异如实保留，不为复现成功强改dtype/batch或反复重跑；系统性偏差需定位工程原因。
   若模型/adapter身份、初态、执行动作、RNG或重放合同不符，暂停相关执行并回报。
6. 机制smoke只使用这两个任务中的一个既定case，不扩成新的成功率筛选。正式矩阵保持一个实现commit，禁止运行中热修。

## 5. 记录和裁决

- 每分支保留原row/轨迹引用、task/state、两种策略身份、teacher metadata、k、重放state差值、RNG起点、
  prefix-terminal、success/steps、实际动作前缀、实际BDDL谓词时间线及worker退出码。
- 在环境端每控制步记录相关可移动对象的body position、夹爪及EEF位姿；对象名称和目标/干扰物对应依据实际BDDL/asset审计。
  用预注册几何读数：相对episode初始位置，水平位移≥3cm或向上位移≥2cm，持续至少3控制步，记首次越界时点和对象。
  同时保留连续数值、cut时已移动状态。该读数只叫持续位移，不自动叫抓持/语义选择；遮挡和不确定接触不得编造标签。
- 全部保留compact记录；仅固定states`0,25`所有分支保存双相机full capture（64个分支），不因成败换案例。
  若prefix-terminal则保存实际可用画面并标记，不补造tail。
- 输出完整`2 anchors × 2 cuts × 4 followers`的每task成功数、paired retained/gained/lost及上述预定义差值。
  两任务单独报告，不用两task bootstrap伪造跨任务推断；可按每task的50个state做配对bootstrap20000/seed20260924，
  表示固定模型、固定任务和这批初态的描述性不确定性。不得把多followers或多cuts当独立样本扩N。
- k=0原件与新同策略接手分支的差异单列；主要因果比较用同一新执行合同下的配对分支，不能隐藏重放偏差。
- 不能仅以动作/对象位移距离或一张图裁决。联合成功、实际谓词、目标/干扰物位移和完整固定case检查H_state/H_condition/H_control；
  “两种解释仍分不开”是有效结论，不能自行追加干预来追逐某种结果。

主讨论收到完整原件后决定：是否需要定位条件表示到LoRA的功能作用、是否需要检验早期状态支持，或撤销对象选择的单一解释。
本批不做训练修复、不读新增held expert actions、不拆tau/horizon续训、不新增官方400评测或Test；下一批另行登记。

## 6. 资源、交付和停止

新增data0峰值预算16GiB（64个双相机full branches约6GiB，加compact、sim记录和余量），data1新增≤1GiB。
创建study/运行树及GPU任务前，由Sol检查strg01对应独立quota、共享容量并细化实际峰值估计；大资产与banks只读复用。
最多4张物理GPU用于独立persistent workers，无DDP；两节点live admission及项目总上限继续适用，按实际吞吐和余量选择卡及共驻。
不为凑卡等待，不引入额外监控agent。正常运行等待退出/完成事件，异常或即时资源调度才轻量检查。

Sol负责run registration、exact commands、代码commit、实现验证、launch/exit、1600分支原始行、固定case索引和机械汇总。
已有根或同任务登记时先查完成状态，禁止重复启动/覆盖。若无法按合同取得原动作或有效重放，应回报具体缺项，不能替换为新policy采样前段。
完成本批主动Queue回主讨论`01a0cd94-65da-7b22-8ca9-7ba35f454632`，然后停止新增实验；不要求例行收到回执。
