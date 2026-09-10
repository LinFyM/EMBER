# EMBER：当前方法改进的专家复核材料（2026-09-11）

**请求：依据实际代码与现存证据，判断现有Horizon方法的主要瓶颈，提出有机制依据的修正，并给出下一阶段的推进顺序。** 当前最强Horizon候选validation126/400，另一个初始化119/400，均未达到目标；最强候选的完整视频对照也没有形成正确动态过程的稳定整体优势。我们已经做了大量分析，但没有得到覆盖主要问题、经验证有效的解决方案。

本包为只能访问远程仓库的专家准备。请先读本页和[咨询prompt](EXPERT_PROMPT.md)，再按索引读取必要记录，不必一次阅读全部数据。当前科学执行暂停，本次没有新实验或方法采纳；专家建议须经Owner讨论后才进入执行。

## 1. 远程可用的证据与模型身份

- [index.json](index.json)：8个学习run的精确commit/原配置、每份本地原件对应的已提交文件、导出范围。
- [panel_summary.json](panel_summary.json)：直接从13,901条导出outcome rows复算的119个面板成功数、per-task、per-suite和breadth。
- [verify_evidence.py](verify_evidence.py)：仅需Python标准库，复算全部面板，并核验两初始化的all/off和最新九臂跨模型实际配对。运行`python docs/review_materials/20260911/verify_evidence.py`即可。
- [analysis/](analysis/)：完整学习矩阵、几何/信用分析、有限更新与视频控制；含640个真实视频条件和32个reader信用条件的逐条件JSONL。
- [panels/](panels/)：全部119面板的逐条结果、元数据和学习面板执行合同；包含policy-noise seeds、teacher选择和已有stage predicates。
- [methods/](methods/)：临时诊断脚本的路径规范化、非执行文本快照，可检查干预是否真的作用于原始RGB、查询或参数组。
- [figures/](figures/)：既有BBQ四例、橙汁一例的所有干预臂轨迹contact sheets；没有新增或改绘轨迹。

**main不等于126候选。** 此包科学源码参照为`043b58ca`（整理前main）；整理不采纳候选科学行为。请使用下面的精确版本：

| 标识 | 源码与原配置 | 科学区别 |
| --- | --- | --- |
| main canonical | [main horizon.py](../../../src/ember/writer/horizon.py)、[默认配置](../../../configs/pi05_horizon_writer_v1.json) | Compiler额外语言query启用；已实现dual输入，但默认agentview；默认节点是旧首段配置快照，不是下一轮launch |
| all7 | [9abc9b95](https://github.com/LinFyM/EMBER/tree/9abc9b956d76b6a9918bcd9836eee25806a9448b)、[配置](model_records/all7/run_contract.json) | 当前逐帧上下文条件版本的原始对照；单视角、独立D |
| off7/off11 | [45e16633 horizon.py](https://github.com/LinFyM/EMBER/blob/45e16633394e1baa9af8cefba698661b41aab539/src/ember/writer/horizon.py#L186)、[off7配置](model_records/off7/run_contract.json)、[off11配置](model_records/off11/run_contract.json) | `backend_conditioning=local_h_read`使额外Compiler query route归零；保留exact language、local和H-read；独立D与main相同 |
| shared7 | [c63f55dc](https://github.com/LinFyM/EMBER/tree/c63f55dc7e36ed51b426063f68caa49245b5af76)、[配置](model_records/shared7/run_contract.json) | 在all背景把同target不同rank的A/B侧D绑定；该实验未被采纳 |

`独立D`指每target/rank/side独立参数，**仍跨task共享**，不是为每个task建立字典。all11使用45e16633的all模式；准确模型映射见index。仅本页称“最强”的是本轮Horizon候选，不是全部历史最高单点，也不是已通过资格的selected checkpoint。

## 2. 目标、数据与实际流水线

目标：generic `lerobot/pi05_base`建立的冻结π0.5-LIBERO source，加exact task language与一条或多条action-hidden正确教学视频，rollout前一次生成唯一完整38-target LoRA，随后从未见初始化闭环完成任务。当前实测为K=1、rank16；具体rank/decoder/memory token不是科学目标。

固定LIBERO40 train24/validation8/test8；source训练语料为LIBERO-90经specification去重后71 tasks×50成功episodes。source未读取目标40动作；normalization由source冻结。当前Writer训练只用train24。允许未来经严格语义去重审计、排除validation/test的non-held meta-task扩展；没有因此获准直接扩展数据或新训。不得制作人工process数据或新仿真任务绕过当前问题。

当前数据流：

```text
exact language + 有序agentview RGB（stride5）
  → frozen vision/Gemma真实prefix，最终Z/KV与真实task-token mask
  → 冻结Action Expert + 可学习观察侧Meta，固定public probe、flow_time=1
  → action_out_proj前完整50×1024响应（不是采样出的正确未来动作轨迹）
  → 四组：过去4帧软对应 → 完整H query → 两端Z视觉核实
           → 历史u顺序GRU → H-read → 过去单向长程
           → 前三组逐H回写，第四组输出P4
  → 两层集合Compiler（608个target/rank槽；off仅取消额外语言query route）
  → native decoder D生成唯一38-target、76个A/B张量
  → source依据机器人自身当前双相机观测和8维state闭环执行
```

teacher输入实际为单视角；policy执行侧双相机不能混同teacher双视角训练。main最近接通同步双视角teacher输入，只做过真实smoke和梯度检查，未训练，证据为[dual smoke](analysis/dual_view_input_smoke.json)。原生prefix本身已含目标language；“Value来自视频分支”并不保证动态像素信息必不可少。

Writer及观察侧Meta fresh联合端到端纯FM，source基础权重冻结。每update随机每suite一task、每task64 queries，全局256；teacher与动作query同task跨episode。teacher训练池0–15，动作池16–41；独立诊断动作42–45，训练task诊断视频46–49。400步约102,400 FM queries；每run实际曝光、loss和配置已导出，不能把步数本身当作平台证明。

部署Writer不得读teacher动作/state/reward/terminal/task ID/filename/object pose，也不允许task-local交互或优化。source执行policy读取自身state合法。validation/test无梯度，Test未触碰；本轮视频乱序/倒序诊断是Owner明确授权的例外，已看过的结果不能冒充未触碰的最终因果资格。

## 3. 学习、保持与历史反证

| 固定实验 | validation200→400 /400 | train200→400 /96 | validation自身保留/新增/丢失 |
| --- | --- | --- | --- |
| all init7 | 103→90 | 41→49 | 56/34/47 |
| Compiler-off init7 | 108→126 | 40→56 | 82/44/26 |
| all init11 | 100→108 | 41→56 | 62/46/38 |
| Compiler-off init11 | 104→119 | 44→61 | 67/52/37 |
| D绑定 init7 | 115→82 | 32→60 | 59/23/56 |

[Compiler确认矩阵](analysis/compiler_confirmation_matrix.json)、[共享矩阵](analysis/sharing_matrix.json)、[完整条件2×2](analysis/retrieval_matrix.json)、[早期条件对照](analysis/language_matrix.json)保留所有方向，包括非首选臂。`local_compiler`仅关H-read，`local_only`关H-read及Compiler额外条件，`none`关这些learned条件路径；**none不是去掉原生prefix全部language，也不是learned language-only baseline。**

Compiler-off的验证净收益在两个初始化为+36/+11；init7换另一正确video关联也保留收益。但init11候选BBQ35→19，丢20个旧成功；整体丢37，breadth7→6，稳定保持未解决。两个候选200→400仍增长，不能认定平台或续训必无效。旧all原样延长及其它上下文干预未解决能力/保持，不能只靠loss继续同样训练。

历史反证必须保留：source47/400，train24 rank128 SFT相邻109/107；旧四task配方v5.2/v6已有132/121，v6 task-complete曾143；GOMQ151单点未保持，后续按rank16合同重物化为136，不能混成当前合规稳定强模型。原始强结果、各自controls、参数化与训练差异见[9月7日证据包](../20260907/README.md)及[research_history](../../research_history.md)。这些历史说明纯FM和视频→LoRA并非完全没有能力，也不能把不同配方优点拼成一个模型。

## 4. 最强候选完整视频检查

固定off7 macro400（validation126），train24×states32–35，每task完整demo46、另一正确demo47；错误视频按预定同suite/跨suite循环映射，保留目标language。乱序/倒序重排真实RGB后重新完整forward；静态首/中/末帧分别重复到原长度，全部报告。

| 条件 | all400 /96 | off400 /96 |
| --- | ---: | ---: |
| 正确完整视频 | 51 | 53 |
| 同task另一正确视频 | 53 | 60 |
| 同suite错误task视频 | 48 | 59 |
| 跨suite错误task视频 | 46 | 59 |
| 乱序 | 47 | 55 |
| 倒序 | 50 | 52 |
| 重复首帧 | 52 | 58 |
| 重复中帧 | 51 | 53 |
| 重复末帧 | 48 | 53 |

来源：[最强候选对照](analysis/best_video_controls.json)、[all200/400对照](analysis/all_video_controls.json)、对应[逐条面板](panels/diagnostics/best_model_video_controls/step400/)。off完整相对all仅+2（保留43/新增10/丢8），错误与静态条件也改善。该固定面板未建立正确动态过程的稳定必要增量；不能把validation收益解释为该问题已解决。

输入仍改变行为：off首帧丢6增11，跨suite错误丢4增10，倒序丢9增8。两条正确视频都成功的47个state中，三种静态全失败为0，两种错误全失败为0，乱序/倒序都失败为5。这些是描述性逐state集合，不是可部署union、新gate或普遍时序理解证明。冻结输入变化、每task两条正确视频/四初态及单视角限制结论；没有单独训练的language/static baseline，也不能据此证明纯task记忆或完全忽略视频。此前只测all后概括当前方法是分析选择错误；本包明确给出补测后的最强候选结果。

## 5. 机制分析能支持什么，不能支持什么

| 观察/干预 | 已支持的结论 | 尚未支持的推论 |
| --- | --- | --- |
| 共用额外language shift经LayerNorm压缩槽间Q差异；off首层真实读取差异约提高20–27倍 | 有具体查询组织与真实读取变化，fresh删除有局部净收益 | 查询相似就是容量崩溃；它唯一解释保持问题；增加rank分工loss必能修复 |
| 四个all/off×200/400面板，双层Q差异L/H实际改变读取；同时H均未超过native normal | 强制差异幅度没有一致即时闭环修复 | 几何指标就是能力刻度；冻结缩放完整否定持久身份/内容分离的新学习方法 |
| D绑定train32→60、val115→82；共享D在当前代码下近共同方向学习 | 当前绑定改变优化几何且损害迁移/保持 | 所有共享有害；高rank相似必然无能力；应直接冻结D |
| 四次其它task真实FM更新使橙汁成功变为操作BBQ，零当前梯度Adam对照仍成功 | 真实跨任务学习可损害另一条件的目标选择；P变化可单独复现该例 | 已找到全部历史坏更新；所有训练task均退化；可以直接外推最强off模型 |
| all200/400的BBQ四例，D400单独致3例转绿瓶，P400单独2例，C400单独0例；有交互 | 输出映射与读取端都可影响目标选择；只改Compiler不足以覆盖已见故障机制 | 四例是全局错误率；P/C/D混合等于分别训练；回退某块即完整修复 |
| train8端点旧P/C+新D为22/32，高于全400的18，但也丢5 | D更新里存在有用能力，同时有保持代价与共同适应 | D普遍没学会；该混合是可选的新模型 |

源数据：[实际视频统计](analysis/actual_video_summary.json)、[reader信用](analysis/reader_credit_summary.json)、[D学习核](analysis/decoder_kernel_summary.json)、[Q中介](analysis/query_mediator.json)、[有限更新](analysis/bounded_updates/)、[BBQ](analysis/bbq_endpoint/)、[橙汁](analysis/orange_replay/)、[端点](analysis/endpoint_swaps/)。全部详细论证见[findings§39–49](../../../findings.md)。

有限更新从all7完整400的临时副本做401–404四macro，共1024 queries；受害的四个预选训练task不在本次更新中。M是四次零当前梯度AdamW，保留m/v、decay与scheduler；FULL−M含真实梯度及其后续moment影响。单卡串行是诊断拓扑，不是formal exact-resume。混合P/C/D的状态未共同训练。橙汁CD失败仍围绕正确盒子操作，P/PC/PD/PCD则转选BBQ；目标选择失败与操作失败分开标注。

既有图片：[橙汁九臂](figures/orange_replay_state32_all_arms.jpg)，BBQ [state0](figures/bbq_endpoint_state00_all_arms.jpg)、[state12](figures/bbq_endpoint_state12_all_arms.jpg)、[state25](figures/bbq_endpoint_state25_all_arms.jpg)、[state37](figures/bbq_endpoint_state37_all_arms.jpg)。目标类别是人工观察解释，成功与配对来自原始rows；不能当作内部错误patch测量。

## 6. 我们当前的看法与尚未识别的问题

以下是请专家检验的工作判断，不能视为新方案已被证实：

1. 当前最需要解释的是**正确教学证据未转成可靠条件增量、跨任务更新的获取与保持失配**。Compiler删除只有局部实证支持，不能把主要问题收束成query几何。
2. 固定task的语言与静态场景都预测正确动作，跨episode FM阻断逐帧复制但不强制使用动态过程。这是可辨识性风险，与最新对照相容；尚未证明模型纯记task，也没有证明必须换RL。旧FM强结果是重要反证。
3. P/C/D都是跨task函数的共同组成，训练task误差下降并不保证另一条件的动作函数仍正确。我们已有有限更新和端点行为证据，但未确定能同时改善未见task获取与保持的学习干预。
4. 可学习language/static prior加视频残差、训练task动作函数保持约束、更多合法non-held mappings、可信失败邻域动作监督、观察侧prefix适配等，都仍是候选方向。没有证据支持现在同时叠加，也不应把这些想法包装为已解决。请独立审查其机制、信息墙、历史近等价失败和新增代价，允许否定。
5. 双视角真实学习、Compiler-off后续增长与每100步strict400已经登记为未来待办；它们不能充当对“这轮分析支持哪些修正”的全部回答。

希望专家给出少量有优先级的具体修正：改哪个接口或学习目标、保留什么、预期改善哪类真实行为、为何区别于旧尝试，以及怎样用最小有信息量的学习/冻结对照决定继续或放弃。欢迎指出这些分析本身哪里过度归因或不具区分力。

## 7. 导出合同与未提供内容

本包约79.27 MiB科学记录，含119 outcome面板/13,901 rows；其中当前原因深化与最强补测共3565 rows，其余为最新学习/确认面板。普通JSON保留科学字段；JSONL按完整record切片。逐行相同的`horizon_writer_lora.method/writer_checkpoint/source_checkpoint`仅保存在该面板metadata的`horizon_writer_lora_common`，合回各row即可还原导出前的规范化内容；已逐行验证。result metadata中重复adapter对象由同目录contract提供。主机、进程、认证、启动和资源字段删除，机器路径变成占位符；没有新增校验hash。

index中的`source`是本地原件身份，不是声称该路径已在远程。应读取对应`exports`，并按`rows_files`顺序连接。未提供模型/optimizer张量、数据集、完整视频/轨迹张量、640条件的NPZ激活或本地环境；不能从缺失材料编造推断。新增逐条件数值和方法快照支持审查分析，但不是让专家在没有资产时重跑GPU。

历史包的原始强模型数据仍在9月7日目录，本次不重复拷贝。旧设计、失效执行状态与未采纳分支保留为历史；当前权威入口为[progress](../../../progress.md)和[task_plan](../../../task_plan.md)。
