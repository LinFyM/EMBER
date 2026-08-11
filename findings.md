# EMBER Findings

更新时间：2026-08-11。本文只保留跨架构仍成立的第一性原理结论。逐方法结果、精确旧 commit/root 与禁止重复项
见`docs/research_history.md`；逐日原始记录可由 Git commit`3a6f801`读取。

## 1. Current empirical boundary

长期目标是同一 shared method、同一 single checkpoint 的 strict paired correct 严格`>150/400`。目前未达到：

| method | correct | same | wrong | shuffled | reversed | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v5.2 old | 132 | 138 | 74 | 82 | 83 | 视频特异性强，absolute不足 |
| v5.2 task-complete | 120 | 109 | 107 | 111 | 124 | recipe削弱Procedure和absolute |
| v6 old | 121 | 122 | 111 | 84 | 47 | 顺序影响强，但absolute低、旋转大 |
| v6-fast task-complete | 143 | 135 | 125 | 128 | 129 | 历史最好single checkpoint，视频margin弱 |

最新 uniform pivot-rank14 裁决：

| arm | correct | breadth | relative to old134 retained/gained/lost |
| --- | ---: | ---: | ---: |
| immutable old full-rank | 134 | 6 | reference |
| compiler-only old-cache transform | 138 | 7 | `119/19/15` |
| online-regenerated rank14 | 128 | 7 | `113/15/21` |

compiler-only净增4，却违反预注册 lost`<=10`；增益由 Long1 净`+11`掩盖 Spatial/Object 净`-3/-4`。从
compiler 到 online 又有`13 gained/23 lost`。因此 uniform compression 与 online regeneration 都会造成
独立的 target-heterogeneous capability rotation。rank14 路线、Gate C、cycle1、controls和训练已经退役。

## 2. The actual learning problem

EMBER不是从视频复制一段动作。部署时策略面对未见初始化，Writer必须从语言与 action-hidden video 提取跨
轨迹仍成立的对象、关系、目标和有向过程，再把这些信息编译成能改变 frozen policy 行为的一套完整 LoRA。

监督训练故意让 video 与 action query 同 task、跨 episode。这样阻断逐帧动作复制捷径，并要求一套 LoRA
覆盖不同初态；但它也造成核心不可识别性：同一 task 的 action target 对不同 teacher video 在统计上相同，
所以普通 positive functional loss 可以只学 task controller，而无需解释哪段视频证据、哪个时间顺序使该
controller成立。架构必须让视频信息成为生成 LoRA 的必要动态值，最终仍要由严格 controls 证明它被正确使用。

## 3. Video use versus correct video understanding

历史 v4 已证明 shuffled/reversed 会改变 latent、LoRA 和 action；但其 shuffled 成功甚至高于 correct。原因不是
“模型没看视频”，而是它把 absolute frame phase、translation trajectory 等低层相关性编译成 controller。

因此有四个不同命题：

1. 视频改变 hidden state；
2. 视频改变 effective LoRA；
3. 视频改变 policy action；
4. 正确视频的语义与顺序产生比 wrong/shuffled/reversed 更有用的闭环行为。

只有第4项支持“从教学视频学到任务知识”。normal order 是训练分布并不自动赋予它因果语义；shuffle/reverse
必须在真实输入帧上完整重排，保持 task/state/RNG/video identity 配对。correct 需要沿有用 policy direction
胜过 negative，而不是仅靠把 negative 推坏形成 margin。

## 4. Why task drift is the central optimization symptom

EMBER多次出现某些 task 在一个 checkpoint 新增、另一些同时丢失。SFB 的八 checkpoint success union 是193，
single best却只有127，证明能力存在于训练轨迹各处，却没有稳定共存于同一参数状态。

已被分别观测到的换手来源包括：

- 不同 task、query、flow noise 的 functional gradients 近正交或相互冲突；
- 同 task 不同 teacher video 的 correction 在 effective LoRA 空间近正交；
- shared factor、Reader、condition map 或 Program→factor compiler 把异质 cotangent 压成 common update；
- 离线 feature-row/RLS 保留不等于 held on-policy state occupancy 保留；
- 强制压缩、refactorization与 online regeneration 即便误差很小，也会跨 closed-loop decision boundary；
- optimizer在轮流获取能力，而非形成可重复的共同累积。

所以“训练更久”“loss继续下降”或“checkpoint union很高”都不能解决 drift。每个新设计必须解释它为什么能让
不同 task/video 的有用更新在同一 checkpoint 共存，并用 paired gained/lost/churn验证。

## 5. LoRA health is a diagnostic, not the objective

正常 SFT LoRA 提供有价值的 policy geometry 参照：能量不能小到近 identity，target间结构不能完全同质，
有效 rank 与 coordinate participation 也不应无故坍缩。但历史结果同时否定两种极端：

- 过低能量、近 rank1、target cosine接近1常解释写出端容量不足；
- 人工追求均匀谱、高 stable rank、正交、更多 atom/lane/expert 或 SFT量级 norm，也反复得到更差 closed-loop。

policy真正需要的几何可能天然 q-dominant、低 effective rank、跨层 coherent。raw A/B 还有 gauge ambiguity，
因此正式比较优先用 effective `BA`、fixed-action response与closed-loop，而不是因子符号或漂亮谱形。

Reward-Credit进一步发现：continuous q/v tangent约`1e-8 RMS`，低于非零 native BF16 factor约`1e-4`的局部
ULP；但随后 uniform rank14 为 tangent 预留 physical slots 又损伤 old support。这说明数值瓶颈真实存在，
却不能靠牺牲已有 policy manifold 来修；更不能为了低位重现降低 batch、dtype效率或GPU利用率。

## 6. Functional objectives are weak surrogates

source-action SmoothL1、reconstruction、expert cosine、gradient consistency、MC variance、tube radius与row-wise
retention都曾显著改善，同时 strict closed-loop不升反降。原因是离线 action queries只覆盖局部功能切片，而
rollout会改变状态分布并跨越离散成功阈值；tiny parameter error也可能落在高敏感 policy方向。

证据层级固定为：

1. shape/freeze/resume/finite/closure：工程合同；
2. representation→compiler→effective BA→fixed-action：机制证据；
3. real strict paired400：方法裁决；
4. five/six-arm controls：视频因果解释。

前两层只能决定是否值得跑下一层，不能选择最终checkpoint。80-row screen也曾与full400对保留性给出相反
结论；正式选择不得依赖小panel。

## 7. What task experts solve and do not solve

正式24-task expert bank统一续到step2000。direct-expert train closed-loop从step250到2000为
`432/557/624/638/658` of 1200，step2000有23/24 tasks非零。它证明 task-local SFT LoRA 是
policy-effective target，并提供能量、rank coordinate与跨target结构参考。

它不解决：

- held task 泛化；
- 同 task 不同视频的特异性；
- 正确、乱序、倒序视频的时间因果；
- shared Writer在24 tasks上的稳定共同积累。

同一 task 的 expert reconstruction target 对所有视频恒定，所以只用 reconstruction 训练最多学到
task-level parameter manifold。soft/hard 24-expert bank在held只得`15/80`与`3/80`，又说明一个可近精确重建
的训练task dictionary不能直接成为deployment route。expert只能作为train24 privileged teacher或几何依据，
不能成为held输入、nearest-neighbor oracle或第二套LoRA。

## 8. What few-shot may solve and cannot be assumed to solve

K4历史实验显示，多视频集合可以削弱单条demo的偶然速度、视角和轨迹细节，并改善permutation、same-video与
leave-one-out内部稳定性。这支持用户的直觉：跨demo不变量可能更接近高层任务知识。

但K4 best只有108，full24 gradient retention约`.043`，并未解决正确顺序的policy-effective credit或
single-checkpoint drift。简单平均video features、平均LoRA或让数量越多越好，都可能抹掉有向过程或只增加
计算。未来few-shot必须作为独立matched变量：固定`k`或显式定义variable-cardinality set encoder，保留每条
视频内部时序，禁止挑video，和相同训练/eval预算的one-shot对照。它不是当前已证明的捷径。

## 9. Coherent historical causal chain

1. v4暴露错误的absolute-time/action-phase shortcut。
2. v5/v5.2分离Semantic Core与Procedure，增强wrong-video辨识，但时序在compiler端衰减。
3. v6-fast把absolute推到143，也暴露架构与task-complete recipe强耦合及晚期换手。
4. v7/v8/v10与Loom/Core/Prior说明更漂亮内部时序、fusion或去DC不保证policy-effective方向。
5. Target-Spectral、atom/lane/expert路线说明强制“健康几何”不能替代闭环。
6. Target-Bound证明动态路径能工作；SFB的union193证明核心是共享共存与稳定积累。
7. variance reduction证明functional estimator不是首因。
8. few-shot与video-grounded experts证明视频敏感、参数隔离和集合稳定性仍非充分条件。
9. task experts给出有效task target，但Expert-Manifold routing/reconstruction未带来held support。
10. Balanced residual、RLS、Reward-Credit依次定位跨video正交、offline/on-policy错位和native-factor ULP。
11. 最终rank14去混杂证明compression与regeneration都会独立破坏能力，不能再用微小数值补丁救路线。

这条链要求下一设计延续已有认知，只改变最早失效接口。负结果只淘汰实际受检验的假设：rank14失败不等于
所有rank reservation失败，expert bank失败也不等于所有task-level manifold监督失败。

## 10. Current falsifiable hypothesis

PICK已经给出一次清晰的接口定位：frozen-policy innovation的raw same/order结构、full48 correct/negative motion、
Program→BA→action和吞吐均工作，但48-key regularized Gram condition=`483.61515>200`。因此不能把它写成方法
成功，也不能把失败归因于video encoder、compiler或动作增益；PICK未进入formal训练。

当前唯一active假设PICK-GC只把PICK的static block换成terminal goal residual，保留centered causal prefix。
train24 cache的same/cross/reverse与condition门，以及exact target-language wrong full48均已通过：raw rank48、
condition=`152.45803`；world6 mechanism condition=`152.61008`、retained/null=`24/24`、Program→BA→action闭合。
B32 deployment又证明zero-memory native LoRA/action bit-exact并完成canonical cache-to-rollout。因此PICK原先的
common-mode conditioning接口已经被这一单变量修复，不能再把后续失败归因于key数值不可解或compiler无增益。

`5200bee`封存后live资源没有单节点world6。full48实现会在collective后按task ordinal固定排序完整24 task，
没有memory all-reduce，所以world4/local6只改变执行分区，不改变video/query/RNG、full48输入或FP32 solve。
当前formal资格先关闭，必须在world4 discarded profile复现机制门；这项拓扑门不能替代后续closed-loop裁决。

最早未决接口现为blind offline full24 credit是否覆盖held on-policy occupancy并让多task support在同一Program
memory共存。formal fresh`0→10`后的strict paired400若未过`144/breadth6/lost8/gained>lost`，只淘汰当前
PICK-GC+blind-credit组合；heterogeneous topology、few-shot和新的occupancy credit仍是未授权开放问题。
