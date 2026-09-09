# K1逐帧上下文过程条件：首段配对结果

日期：2026-09-09。按[active design §8.2.8](horizon_relation_video_writer_design.md)预登记完成fresh 100/200、两个correct strict400和200 held-video train96。训练、物化、评测全部正常结束；原件位于`runs/analysis/horizon_relation_writer_20260908/k1_frame_contextual/`，完整模型与raw rows位于`runs/outputs/horizon_k1_frame_contextual_v1_seed7_20260909/`。

## 1. 裁决及解释边界

**本次访问改动尚无整体收益。** 100/200 correct为52/103，前轮first-query-only为75/110；200 train96为41，前轮46。不能将相近的held FM、接通的上下文梯度或200的breadth7当作修复证据。200的Spatial1与Goal23各只有1/50，原先困难区域仍未可靠获取。

本轮自身100→200新增65、丢14，净增51；相对前轮的缺口由23缩至7。200仍处于接近已有早期水平且继续获取的状态，尚未覆盖前轮200以后能力回落的区间。因此按§8.2.8的这一分支登记同配方300/400保持段，不能因为100较弱就提前声称整个访问机制已被否定，也不能用200的恢复声称长期修复。

这一判断有投入取舍：两个节点和train96均低于前轮，没有已证实的优势；续段的用途仅是区分较慢获取与后续保持，不能把一次继续训练解释为正结果。300/400完成后若仍无实际获取或保持优势，不以loss继续下降、微小breadth或单个高点追加训练、扫超参。整轮目标仍是严格>145及全部稳定、广度和视频因果资格。

## 2. 主要变量与执行证据

四组过程条件从静态exact token embeddings的单query位置化attention，改为同一次冻结Gemma prefix中的逐帧上下文exact task-token状态，经原reader读取。保留原reader参数、当前帧/过去依赖、完整H=50、四组过程和视觉核实、first-query-only compiler、原native D及K1/4suite×64query采样。静态语言仍只用于compiler首次检索；不是增加一套backbone或新外部语义信息。

本改动同时改变输入表示、数值分布及梯度路径。reader复用两种输入域是已登记的选择；结果不单独分离上下文语义、视觉条件化、尺度或共享reader优化的作用。原图的Z重读已经包含上下文tokens，不能说旧图完全没有语义；本轮也未测最终视频因果controls。

正式运行面为clean pushed detached `9abc9b95`，Writer 368,675,520参数、Meta 626,688参数保持。fresh 200更新为800条件/51,200 queries，实际341种task-video、各suite200条件；100/200实际task、video、action episodes/frames、noise seeds、权重和帧数均与前轮相同节点逐条匹配。没有从profile或旧模型继承学习状态。

200更新均值15.566秒，正式完整墙钟3413.33秒；allocated/reserved峰值31.900/34.645GiB。两个完整checkpoint通过public inspector。100的400条bank及200的400+96条bank全部新生成、无跨checkpoint复用；每个面板实际teacher schedule、无放回映射、checkpoint身份及信息墙检查通过。

定向工程验证覆盖102项：101项一次通过，另1项旧错误消息断言更新后单独通过。真实8update profile及93采样帧最长视频完整反传通过，context与Meta梯度非零、source冻结。这些只说明机制可执行，不证明行为有效。

## 3. validation8完整配对结果

每列均为单checkpoint、8task×50个固定state；跨轮相同task/state/video/environment/policy RNG和source normalization。前轮为first-query-only，原始Horizon只作附加参照。

| global ID与任务简述 | 前轮100 | 本轮100 | 前轮200 | 本轮200 |
|---|---:|---:|---:|---:|
| 1：ramekin旁黑碗→盘 | 0 | 0 | 0 | 1 |
| 3：cookie box上黑碗→盘 | 1 | 0 | 3 | 0 |
| 11：cream cheese→篮 | 33 | 24 | 31 | 29 |
| 13：BBQ sauce→篮 | 3 | 2 | 28 | 25 |
| 23：开上抽屉并放碗 | 0 | 1 | 0 | 1 |
| 26：cream cheese→碗 | 31 | 21 | 38 | 36 |
| 31：cream cheese与butter→篮 | 3 | 1 | 9 | 6 |
| 32：开炉并放moka pot | 4 | 3 | 1 | 5 |
| **合计 /400** | **75** | **52** | **110** | **103** |
| 有成功任务 /8 | 6 | 6 | 6 | 7 |
| Spatial/Object/Goal/Long | 1/36/31/7 | 0/26/22/4 | 3/59/38/10 | 1/54/37/11 |

100四suite均比前轮低。200 Object仍少5、Goal少1、Spatial少2，Long多1；Long32增加4但31减少3。不能只挑Long32或breadth作整体优势。100与200的Goal23虽然各1次成功，其state不重合；Long从4增至11也只保留1次，局部低分仍不稳定。

| reference→本轮candidate | 保留 | 新增 | 丢失 | churn | Jaccard |
|---|---:|---:|---:|---:|---:|
| 前轮100→本轮100 | 38 | 14 | 37 | 51/400 | .4270 |
| 前轮200→本轮200 | 76 | 27 | 34 | 61/400 | .5547 |
| 本轮100→本轮200 | 38 | 65 | 14 | 79/400 | .3248 |
| 原始100=55→本轮100 | 36 | 16 | 19 | 35/400 | .5070 |
| 原始200=110→本轮200 | 77 | 26 | 33 | 59/400 | .5662 |
| source47→本轮200 | 37 | 66 | 10 | 76/400 | .3274 |

前轮200→本轮200的Long仅保留2、新增9、丢8；原始200→本轮200的Long为0/11/7。11次成功不是已经稳定的Long能力。相邻本轮100→200的净增长主要来自Object28与Goal15，另Long7、Spatial1；BBQ2→25仍处早期获取阶段，保持尚未测到。

100的6个worker、200的8个worker均exit0；wrapper分别1440.21/1112.77秒。所有比较通过canonical完整配对检查。原件`step100/completed_summary.json`、`step200/completed_summary.json`及对应`vs_*.json`保存逐task/suite和全部成功集合。

## 4. 训练分布上的独立诊断

固定train24×states32–35、held teacher46–49各一次，96行与前轮及原始同节点实际配对。该面板用于训练能力诊断，不能选validation checkpoint，也不能用每task仅4个state外推训练总体。

| 指标 | source | 原始200 | 前轮200 | 本轮200 |
|---|---:|---:|---:|---:|
| Spatial /24 | 6 | 12 | 11 | 11 |
| Object /24 | 0 | 15 | 12 | 11 |
| Goal /24 | 7 | 16 | 14 | 13 |
| Long /24 | 2 | 9 | 9 | 6 |
| 合计 /96 | 15 | 52 | 46 | 41 |
| 有成功任务 /24 | 7 | 20 | 18 | 18 |

前轮→本轮R/G/L=35/6/11，churn17、J=.6731；原始→本轮33/8/19，source→本轮11/30/4。book35从3/4降至0，双moka38从0增至2；其它局部变化完整保留在`train96_step200/vs_first_query200.json`。本轮相对source仍有广泛训练任务学习，但没有超越前轮。三个worker均exit0、wrapper1016.96秒，raw rows和全部配对检查保留。

冻结held-action诊断为0/200各24task×128queries，actual video/action/time-noise与前轮配对，无梯度。200本轮均值.111031413、前轮.111183766，仅11/24任务本轮更低；均值接近未对应行为等价。step0为.151469383/.151466233，属正常执行低位差异，不宣称逐元素一致。此FM不参与checkpoint选择或成为续训理由。

## 5. 接下来的受控问题

从本轮完整200状态exact-resume至400，预登记300/400两个single checkpoints和各自correct400；400补同口径train96及原固定held FM。Writer、Meta、optimizer/scheduler、sampler、seed、数据和rank topology保持；physical microbatch只按现场余量配置。

比较先看本轮相邻获取/保持，再与前轮300/400=106/103、train400=59，以及原始86/87对照。不能把本轮100低起点带来的大增量代替绝对性能，不能把不同task或checkpoint的成功拼接成一个结果。若没有实际能力扩展和保持改善，结束这一访问干预的原样续训，回到有区分力的机制分析；不将负结果扩大成上下文语义或完整过程图被整体否定。

本首段未启动same-task-other、wrong/no-video/static/shuffled/reversed、额外meta tasks、RL或Test。资格与最终controls仍严格遵守§8.3。
