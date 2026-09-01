# 1. 核心结论

我核验了固定提交 **`main@a185fe223d1ef77635d83696c3e164a48520edbf`**，该提交的结论性消息为“conclude bank-conditioned primal qualification”。我沿其可达历史一直追到无父提交的初始 commit `1226236beb342d56c7741164e8da1c4eb8d70725`；GitHub 提交分页在下一页为空，因而覆盖了该提交的完整可达历史。审查中按 `AGENTS.md` 的 authority 顺序读完了当前文档、96 节 `research_history`、七份专家原文、README，并针对关键归因检查了实现、配置、测试和对应提交。

**我的主要裁决是 A：保留 ECP Native-Factor 大框架，但停止当前 G3 函数类，替换一个已经相当明确地被定位的结构接口。**

这个最早接口不是泛泛的“Program 不好”，也不是 rank4、真实 native X/Y、signed pooling、Stage 0 或 zero-interaction 合同已经失败。它是：

> **Program 与当前 native bank 如何共同形成一个会随 bank 内容旋转的高维功能方向。**

当前实现先把 bank 压缩成高度相似的 summary，再用 family-level 标量 gate 调制 event-additive anchor。充分校准后的结果表明：

* bank-dependent candidate delta 的 correct/wrong cosine 仍只有约 `.718–.772`，说明 bank 中确实还有可用差异；
* 但 summary、scalar gate 和占主导的 shared free delta 分别约为 `.991`、`.965–.971`、`.993–.995`；
* F 路径能够降低 wrong，却同时降低 correct，margin 最终只有约 `.185`；
* free anchor 已移动到与 candidate anchor 同量级，因而不能继续归因于步长不足。

因此，**bank 差异在到达最终功能方向之前被“标量化、加法化、公共方向化”了**。这是当前停止条件，而不是整个 ECP 的停止条件。

我建议的新接口是 **Program-Conditioned Native-Bank Tangent Transport，PNBTT**：

* 保留 G2 的 owner-specific、ordered Natural Program；
* 保留 38 targets 的真实 native X/Y、每视频单位质量、IEEE bank statistics、exact signed replay、rank4 和 carrier12；
* 删除 R5-style absolute base primal、family scalar gate、shared event-additive free anchor和“base + bounded correction”语义；
* 让 Program 产生 query，真实 bank candidate 产生 key，并始终把真实 X/Y 作为 value；
* 用当前 bank 的 key covariance whiten query，再在同一 bank 上 exact antithetic signed pooling；
* 这样产生的方向一阶近似为

  $$
  \operatorname{Cov}_{B}(v,k)\,C_B^{-1/2}q(P),
  $$

  即一个真正随 bank 改变的**矩阵值切线传输**，而不是一个标量 gate 乘公共 anchor。

对于“是否直接整体训练”，我的判断是：

* **应当立即停止继续堆叠当前 task-local/internal G3 变体。**
* 但不应把当前已被证伪的 scalar-gated graph 原样拿去长程 joint training，因为那只会把结构瓶颈与优化瓶颈重新混在一起。
* PNBTT 只需要一个 topology-matched task-local capacity Gate 和一个无 task-token 的真实 Program Gate；若通过，就应立即进入 matched whole-Writer joint adjudication，不再设置漫长的组件课程。

现有证据还不足以支持 C，即“ECP/Native-Factor 已接近根本停止”。G1、G2、P1、S0、R10 以及历史 GOMQ 分别证明了 native rank4 容量、视频动态、current-bank transport 容量、bank specificity 上界、真实 Program 可获得中等功能效用，以及 action-hidden video 确实能够改变闭环策略。它们没有证明最终共享规则一定存在，但显著反驳了现在就做根本否定。

**次选分支是 B，但只在一个干净的 PNBTT real-Program Gate 失败之后启动。**那时应放弃固定 Natural Program 瓶颈，另开“language + ordered native-bank tokens 直接产生 signed measure”的新 Writer，而不是继续修补 ECP Program→bank 接口。

Action Meta 继续关闭。owner 设定的边界合理：只有 base Writer 已有稳定、跨相邻 checkpoint 的闭环增量，且残余错误明确集中在 action-in/out 控制细节时，才做一次严格 matched on/off。

---

# 2. 从最早 EMBER 到当前提交的逐架构审计

以下把只改变 width、LR、seed、版本号而没有改变科学接口的连续版本合并为一个架构族；否则会把“逐架构审计”退化为没有信息增益的 commit 罗列。早期事实和提交索引以 `research_history` 为主，关键 Gate 再由提交、配置和代码交叉核对。

## 2.1 固定能力、早期 Writer 与 GOMQ

| 架构/提交                                            | 相对上一版的真实科学变化；数据与监督；训练/冻结对象                                                                                  | Gate 或闭环结果                                                                          | 真正淘汰了什么                               | 没有淘汰什么 / 混杂                                       |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------- |
| 初始研究命题，`1226236` 起                               | 从 generic frozen PI0.5 出发，目标是 language + action-hidden video → 一次性 LoRA；初期先建立 source、task expert、LoRA 和严格评测 | 形成 EMBER 的固定信息墙与 zero-interaction 目标                                                | 无                                     | 只是问题定义                                            |
| 固定 source / task-local rank16 oracle             | task-local Action Expert rank16 直接训练；Writer 不参与                                                             | validation8 source `48/400`；task-local oracle `250/400`，四 suite `73/78/58/41`       | 淘汰“Action Expert LoRA 本身没有闭环容量”       | 不证明共享 Writer 能推断 LoRA                             |
| held5 source / carrier / members                 | held5 source `21/250`；共享 carrier `43/250`；多个独立 successful member                                            | independent successful members `113/250`，五个 task 包含 Goal/Long                       | 淘汰“held5 没有成功策略或 policy-effect 信号”    | member 是 task-local privileged 资产，不是部署 Writer     |
| mobile-rank4 解析投影                                | 把 known-success updates 投影到 carrier12 之外的 mobile rank4                                                      | held5 五个 task 均有非零可实现容量                                                             | 显著削弱“rank4 必然无法覆盖 Goal/Long”          | 不是共享可学习性证明；12+4 也未被证明全局最优                         |
| action-memory / belief Writer                    | 视频表征直接写入共享 memory/LoRA；以行为或表征损失训练                                                                           | 闭环接近 source；controls 弱                                                              | 当时的直接 memory decoder                  | 不淘汰所有视频参数生成                                       |
| LOOM / CVADR                                     | 引入更结构化的时序、视觉与 action-dynamics 表征                                                                            | 仍缺稳定闭环增量                                                                            | 这些特定时序 decoder                        | 内部表示可能含信息，但未可靠进入 LoRA                             |
| LMMPC/LPCP / layer-matched memory                | 按 PI0.5 层与 Action Expert 结构做 layer-matched 编译                                                               | 代表性自然视频裁决约 correct/language-only/video-only/endpoints=`41/39/40/39`，Goal/Long=`0/0` | “更贴层”本身不足                             | 不证明 native activation bank 无价值                    |
| reward/outer credit / gradient/open-memory query | 用 reward、functional gradient 或 query 形式给 Writer credit；deployment 仍受信息墙限制                                   | 单向 outer credit 还从 41 降至 39；高 churn                                                 | 相应 credit assignment 实现               | 没有隔离清楚表征、坐标与 realizer                             |
| **GOMQ rank32**，关键链 `f2f9290/ac233fa`            | 独立于 ECP；更大 rank 的 gradient/open-memory query Writer                                                         | correct `151/400`；same-task `139`、wrong `131`、shuffled `127`、reversed `115`         | 淘汰“action-hidden video 绝对不可能产生有用参数更新” | correct 与 controls 间隔太小；视频因果弱；rank32 不匹配当前 rank16 |
| GOMQ 统一 rank16，`3075b3c`                         | 只把输出能力统一到当前 rank16 合同                                                                                       | correct `136/400`                                                                   | 证明其 151 部分依赖容量                        | 不能当作 ECP 成功；不能说明 Native-Factor 无价值                |

这里 GOMQ 留下的最重要证据不是“应该恢复 GOMQ”，而是：

1. action-hidden video 在 zero-interaction 前确实有可能改变闭环能力；
2. 闭环分数可以接近目标线；
3. 但若条件机制不具备足够 specificity，正确视频、错误视频和时序 controls 会一起得到高分；
4. adapter rank 是实质变量。

因此它应保留为 Final baseline 和“高能力、低因果特异性”的反例，而不是恢复为当前路线。

## 2.2 旧 ECP、Native-Factor G1 与 Natural Program G2

| 架构/提交                                                        | 真实科学变化；数据与训练边界                                                                                                                                                 | Gate / 结果                                                                    | 真正淘汰了什么                                                    | 未淘汰 / 混杂                                |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------- |
| ECP Stage 0 native observer，`e675b87/bb06676`                | 用 frozen PI0.5 owner/layer/horizon native signals 形成 ordered events；Stage 0/observer 可训练，policy 冻结                                                             | non-degeneracy、task separation、event occupancy 通过；Action Meta 中性             | 淘汰“observer 完全常量或 event slots 不工作”                         | 未证明 Program→LoRA                        |
| ECP Stage1 v1–v24                                            | fixed/free Program、deterministic/mean privileged code、A/B hyperdecoder、owner-local support、outcome binding、mapping-diverse compiler 等                          | 多次内部 loss 下降，但 held/shared 闭环无广泛提升                                           | 这一整族直接 code→factor decoder                                 | 版本间存在多变量变化；不能外推到任何 bank-native compiler |
| MDCO / PECS                                                  | 用 denoising、particle、local policy-effect 轨迹强化 latent→effect                                                                                                    | 内部预测可改善，闭环仍差；PECS 终止                                                         | 当时的 latent/effect supervision                              | privileged policy evidence 本身仍可能有用      |
| structured privileged fixed-A                                | 把 privileged effect 映射到固定 A 坐标                                                                                                                                 | `78/250`，breadth 3/5，Goal/Long 0                                             | fixed-A realization 是实质瓶颈                                  | 不能否定 mobile coordinate                  |
| known-success fixed-A 投影                                     | 三个成功 member 投影到 fixed-A                                                                                                                                        | `49/41/35`，Goal/Long 0                                                       | 更直接证明 fixed-A 丢失必要方向                                       | 不说明 rank4 总容量不足                         |
| raw mobile-rank4 solver                                      | 允许 A/B 都移动，但只有短程 raw-factor 优化                                                                                                                                 | 约 `49/250`                                                                   | 短 solver 未找到成功 basin                                       | 优化失败与坐标容量未完全分离                          |
| fixed-effect / balanced-SVD realizer                         | 先产生 effect code，再固定解码为因子                                                                                                                                       | 闭环约 `33/37`，低于 carrier                                                       | fixed effect-code realizer                                 | 不否定从真实 bank 直接取方向                       |
| centered two-sided / fit-span                                | 两侧坐标、SVD、fit-span，对 known updates 的 cosine 可达 `.877–.960`                                                                                                      | strict `80/250`，breadth3/5，Goal/Long0                                        | “高 update cosine 即足够功能”                                    | cosine 与闭环混杂；只淘汰对应坐标                    |
| privileged q_\pi / fingerprint / phase                       | 尝试从 privileged policy evidence 得出神经 latent 或 phase                                                                                                             | 部分 held 分离、phase decoder 有小增量，但无法保留成功成员能力                                    | 神经 q_\pi 作为 canonical deployment bridge                    | set-valued functional critic 仍可保留       |
| 人工 process / recovery                                        | 人造 opposite-order、primitive、recovery experts 和蒸馏数据                                                                                                             | Gate A `14/100`，相对 A3 gained0/lost23；运行合同正常                                  | 当前人工 recovery teacher 与 process 路线                         | 不否定自然视频；也不是 ECP 本体失败                    |
| **2026-08-24 Native-Factor 重构**                              | 删除 neural q_\pi 与 fixed realizer；Pass A 产 Natural Program；Pass B 读真实 38-target X/Y；privileged policies 只作 set-valued critic；signed pooling 产 rank4，拼 carrier12 | 架构重置                                                                         | 旧 latent→fixed effect 路线                                   | 新路线尚需逐接口 Gate                           |
| G1 初版 `9a6f434`                                              | task-local free Program/code，真实 native bank 与唯一 rank16                                                                                                         | `88/250`，`33/18/37/0/0`                                                      | 首版 pooling 不足                                              | 仍不能责怪共享 Program，因为这里是 task-local        |
| scalar q-output span 审计 `822147b`                            | 分析无 bias q output 与 action-in output 的可达子空间                                                                                                                    | q-output 只在 W 列空间；action-in 后来修正为 `span(W,b)`、最多 33D；被排除方向对 Goal/Long 必要     | scalar output pooling 的结构上限                                | 初始“32D”表述对带 bias action-in 不精确          |
| q-head / stable bank / exact init / member selection         | head-wise pooling、稳定子空间、精确零初始化、set-valued member 选择                                                                                                            | 若干轮仍未过，但归因逐步从优化移向 action-in native block                                     | 淘汰单纯 q-head 或初始化补丁                                         | 不能否定 native-factor 总体                   |
| **G1 action-in native block pass**，实现 `31f0053`、记录 `5617a4e` | action-in 按真实 native input width 构建 block；其余任务本质不变                                                                                                             | strict `114/250`；五 task `35/31/45/2/1`；breadth5/5，Goal2，Long1；retention35/43 | 证明真实 bank + free-code signed pooling 可形成强 task-local rank4 | **不证明共享 Program→selection**             |
| G2 首版 full vs endpoints                                      | 训练 Natural Program 与 owner readout；Program schema固定                                                                                                            | full 相对 endpoints 仅 `+0.0226%`                                               | 初始 Program 基本静态                                            | 不说明视频无动态，只说明接口没利用                       |
| 去静态旁路 / frozen observer / owner readout                      | 依次删除静态 bypass、只用 frozen observer、检查 owner-specific readout                                                                                                     | `-0.0570%`、约 `+0.005%`；readout 近常数                                           | 对应静态捷径和 observer 假设                                        | 多数变化仍未让动态成为必要                           |
| temporal residual / cadence                                  | 引入 temporal residual，修正 optimizer cadence                                                                                                                      | 约 `+0.0381%`，再到 `+0.3080%`                                                   | 证明优化 cadence 是一个真实混杂                                       | 仍不足以过动态 Gate                            |
| macro20 未锚定 alignment                                        | 延长到 macro20                                                                                                                                                    | `8.6878%`，但 K>1 alignment 坍缩成单事件                                             | “宏训练更久即可”                                                  | Program 有能力但 alignment 退化               |
| **boundary-anchored monotonic G2 pass**，`c1493a1/86e434e`    | 只固定 monotonic alignment 两端；数据、loss、K、readout不变                                                                                                                 | `22.2047%`；probe38/40、median events4；K1/K4、same-task通过                       | 证明 Natural Program 中有可重复的视频动态                              | 不证明这些动态足以产生正确 policy；没有共享闭环结果           |

## 2.3 G3 第 21–96 节完整主线

| G3 架构族                                               | 真实科学变化                                                              | 主要结果                                                                                                               | 淘汰了什么                                                              | 没有淘汰什么 / 关键混杂                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| frozen-Program shared compiler v1 macro5/macro10     | 冻结 G2 Program，训练共享 Program→native selection                         | 早期 functional mapping 很弱                                                                                           | 首版共享 scorer                                                        | 不足以归罪 Program 或 bank                                                           |
| native-teacher v2                                    | 用 native-feasible teacher/factor supervision代替纯 latent监督            | teacher reconstruction 与 functional credit 冲突                                                                      | “更像 teacher factor 即更好”                                            | teacher 本身可能坐标不唯一                                                              |
| four-family LOTO dual-basis                          | q/v/action-in/action-out 分 family，并做 leave-one-task-out             | 揭示 family 差异，但 shared held 仍弱                                                                                      | 单一共同坐标                                                             | 未检验 current-bank global operator                                               |
| F1 current-bank operator，`435cb4a`                   | task-local primal 经当前 bank covariance/dual，再 exact replay           | 四 family operator recovery 近 `.9995+`                                                                              | 排除 full inverse/回放的基本实现错误                                          | 只是 task-local fixed direction                                                  |
| F2 `C=I`                                             | 移除 current-bank covariance                                          | full451 约 `.022`                                                                                                   | 证明 bank-global dual 是必要结构                                          | 不说明 shared Program 能产生 primal                                                  |
| F3 shared current-bank mapping，`78b7e58`             | frozen Program→shared primal→current-bank dual                      | fit/held/p10/task-holdout约 `.0865/.0831/.0726/.0962`                                                               | 当前 shared map                                                      | 不说明是否是 Program、坐标还是 loss                                                       |
| equal-subspace / family / fixed-owner                | 改共享空间与 owner/family factorization                                   | 均未形成稳定 shared held                                                                                                 | 对应共享分解                                                             | 大量结果仍是内部 recovery                                                              |
| direct-native                                        | 直接让 Program 输出 native direction                                     | 出现近常量/退化                                                                                                           | 简单 Program→native vector                                           | 当前 bank 未真正参与方向                                                                |
| feature-common-code                                  | 先学 shared low-dimensional code                                      | train-only nullspace 很大                                                                                            | feature code 与功能不一致                                                | 不等于视频没有可用信息                                                                    |
| task-stable anchor                                   | 学跨视频稳定 anchor                                                       | held 约 `.142`                                                                                                      | 一个公共 anchor 不够                                                     | 预示后来的 shared-direction问题                                                       |
| owner-query FiLM                                     | owner query 调制 bank read                                            | 约 `.163`，主要靠 action-in                                                                                             | 轻量 FiLM                                                            | 未给 candidate-level 高维旋转                                                        |
| candidate compatibility / additive joint             | Program、candidate compatibility与 additive base                      | 约 `.129` 或更低                                                                                                       | pointwise/additive交互                                               | 仍可能有真实 whole-bank interaction                                                  |
| functional consensus                                 | fit-only 功能 consensus                                               | held 约 `.946`                                                                                                      | 证明 task-local functional teacher方向跨视频稳定                            | consensus 是 task-local privileged，不是 shared Program                            |
| functional anchor / IEEE                             | frozen Program 拟合 functional anchor；修正 TF32/IEEE                    | fresh 仍约 `.083`                                                                                                    | 数值误差不是根因                                                           | anchor 坐标本身可能不适合                                                               |
| functional-polar                                     | 直接对完整功能梯度做 task-local polar                                         | task-local witness 很强，但约 `58s/K1`                                                                                  | 证明 full functional geometry中有解                                     | 不是可部署共享算法，且太慢                                                                  |
| low-dimensional native-Q sketch                      | 用低维 query 逼近 full polar                                             | S1约 `.156`；free/analytic 可达性差距巨大                                                                                   | 当前低维 chart                                                         | 不否定更合适的 bank-relative chart                                                    |
| **Program-primal P0/P1**                             | Program 产 primal，当前 bank 做 global dual/exact replay                 | P0数值合同通过；P1六 task fit/held `.9717/.9545`，held/optimistic `.9922`，四 family `.9398–.9954`                            | 排除 primal→current-bank dual/replay 容量不足                            | task-local primals；不证明 shared Program 或 wrong-bank specificity                 |
| G2-B behavior sufficiency                            | pointwise、role-local、joint-role、global-calibrated behavior geometry | old Program 对 exact-rank4 behavior约 `.2695`；language-only接近；后续仍弱                                                   | 这些独立 behavior-Gram objectives                                      | 部分 run 更新不足，不能证明 Program schema不可能                                             |
| J2 positive control                                  | task-local joint Program-primal                                     | held retention约 `1.014`，factor `.808`                                                                              | 证明图与 direct functional credit 可工作                                  | task-local                                                                     |
| J2 shared                                            | 联合 Natural Program + shared primal                                  | train/held `.171/.165`；task-held `.123/-.109`；跨 task functional gradients多为负                                       | 当前 shared Euclidean/joint映射                                        | 不等于所有共享规则不存在                                                                   |
| J3 counterfactual routing                            | wrong Program/role routing                                          | 主要把 wrong 变坏，未提升 correct                                                                                           | 单纯反事实 routing loss                                                 | 可能需要 bank-relative interaction                                                 |
| R1–R3 routing/set critic/grouped output              | routing tokens、set-valued critic、分组 output                          | 某些内部 family 改善，primary held无实质提升                                                                                   | binary/token routing、小型 critic、只修 output                           | q/v 与 shared direction仍未解决                                                     |
| R4 functional-code init                              | 用功能 code 初始化                                                        | train/held约 `.819/.839`，wrong-token margin高，但 action-in弱，chart漂移                                                   | 当前可训练 chart                                                        | 说明 fixed chart可能更好                                                             |
| **R5 fixed chart**，`9e6b6a7`                         | 冻结强 functional chart                                                | train/held `.940/.963`，四 family约 `.816/.839/.821/.837`                                                             | 证明 fixed route/head/function有强 capacity                            | Program 仍未接回；correct/wrong bank未测试                                             |
| R6 Natural Program reconnect                         | 把真实 Program 接回 R5 chart                                             | 约 `.165`                                                                                                           | 直接 Program→固定 chart                                                | 不说明 Program无信息，只说明坐标不匹配                                                        |
| R7/R9 functional chart/outer direction               | outer cosine较高或稳定 chart                                             | 功能效用仍可为负                                                                                                           | cosine/surrogate不能代替功能                                             | 强化了必须 direct functional 裁决                                                     |
| **R10 true functional refinement**，`731a769`         | 用真实跨 episode functional credit优化 Natural Program→chart              | train/held `.560/.544`，wrong-Program `.279`，task-held `.151`                                                       | 淘汰“真实 Program 完全不能接受功能 credit”                                     | 效用仍不足且泛化弱                                                                      |
| R11 raw Stage0 probe                                 | 直接用 raw Stage0代替Program                                             | 更差                                                                                                                 | 没有支持“压缩 Program 是唯一根因”                                             | chart/shape同时改变，实验混杂                                                           |
| R12/R13 binary support                               | 估计 support/AUC并做 full/half routing                                  | AUC `.831`，但 held/task-held routing不泛化；阈值跳变                                                                        | binary operator selection                                          | 不否定连续方向形成                                                                      |
| candidate interaction v1–v4                          | normalized/raw/anchor/pointwise vector interaction                  | v3保 correct但wrong同样高；v4 correct/wrong梯度约 `-.966`；task-local vector witness可同时 correct≈`.72/.60`、wrong≈`-.52/-.38`  | set-independent pointwise函数类                                       | task-local witness证明 bank 可读且分离解存在                                             |
| exact-effective-rank4 qualification                  | 检查功能更新是否能落入当前 exact rank4                                           | task-local资格成立                                                                                                     | 排除“功能方向全在rank4外”作为首因                                               | shared映射仍未解决                                                                   |
| EBSRI S0 free-summary                                | whole-bank set-relative summary，task-local自由summary                 | task1 correct约 `.93–.95`、wrong约 `-.49–-.54`；task93 correct约 `.89–.91`、wrong约 `-.16`                                | 证明 whole-bank summary+B1 有task-local解                              | free summary可绕过真实 B0                                                           |
| EBSRI S1 real-summary                                | 用真实 set summary代替free summary                                       | task1/93 task-local仍较强                                                                                             | 证明 real bank summary 可支持task-local interaction                     | 尚非 shared                                                                      |
| shared S2 effective                                  | 训练 shared mapping拟合effective target                                 | 失败；effective与direct functional gradient median cosine仅`.0219`，6/16为负                                               | 该 surrogate                                                        | 不能把失败归因于架构                                                                     |
| S2 direct-functional / polish                        | 直接功能 loss；再以 unit gradient polish                                   | correct常保持高但wrong也高；训练 specificity不能迁移到task1/93                                                                    | 当前 shared EBSRI 与 polish                                           | whole-bank task-local能力仍在                                                      |
| rank/event/owner/relational quotient                 | 切断 absolute code，压缩为quotient                                        | task93均未过                                                                                                          | 当时具体 global-token quotient                                         | 后来发现 global free token、B0作用域和 raw Program旁路同时改变，原归因过强                          |
| **Program-through-bank S0**，`b11dc3e/bc5c34a`        | raw Program旁路切断；scope-matched free summary；summary-only B1          | task1 `.989/.974/.989`、wrong约`-.565`；task93 `.947/.940/.917`、wrong约`-.342/-.394`；正式 pass                           | 证明 fixed-base+summary-only B1有强 capacity/specificity               | 只是 free-summary upper bound                                                    |
| **S1 “real summary”**，`9047230/1cdfbfa`              | free tree换成真实 B0 set read；其余 Gate相同                                 | task1 `.827/.855/.798`，task93 `.777/.793/.720`；wrong与margin通过，correct/held失败                                       | correction-only real-B0 response                                   | 正式 config 实际仍用 fixed orthogonal task token，而非部署 G2 Program；不能称为真实 Program充分性测试 |
| bank-conditioned primal，`eb9f295`                    | whole-bank response前移到 primal                                       | task1 correct `.951/.931/.925`，wrong `.428/.477`；task93 `.917/.923/.888`，wrong `.627/.654`                         | 恢复correct说明前移有效                                                    | wrong specificity不足                                                            |
| Q_free                                               | 放开 task-local query；修正1/38 under-travel后固定32×坐标校准                   | task93 correct约 `.808/.826/.795`，wrong `.526/.534`                                                                 | query步长不足不是完整解释                                                    | 出现capacity–specificity折衷                                                       |
| base-LR A_free                                       | 加 shared full-native free anchor                                    | correct `.815/.833/.797`，wrong `.512/.524`；anchor RMS仅`.0094`                                                      | 只淘汰小幅A_free                                                        | 不能淘汰充分移动的full-native span                                                      |
| **calibrated A_free + F=0**，`e02f4ca/144d59b/a185fe` | 只把 free-anchor 步长固定放大32×；同checkpoint做精确F=0                          | final correct `.853296/.858892/.818467`，wrong `.611592/.668511`；free RMS`.17664`；F=0使correct和wrong同时上升；正式 non-pass | **停止 summary→family-scalar gate→shared event-additive anchor 参数化** | 不淘汰 Program schema、真实X/Y、signed pooling、rank4或ECP整体                            |

**总的历史结论：**

* 早期 Writer 主要失败在“视频表征或 generic latent 直接解码到参数”。
* 旧 ECP 主要失败在 privileged latent、固定坐标和 realizer。
* Native-Factor G1 解决了 task-local realization。
* G2 解决了“Program 是否含动态”的最低门。
* G3 反复证明：**task-local功能方向存在，但共享、bank-specific、可迁移的方向形成规则一直没有建立。**
* 最近一周的价值不是又积累了一批 non-pass，而是把失败从 generic “shared compiler”进一步压缩到了一个具体的参数化接口。

---

# 3. 当前根因的证据排序

## 3.1 最早接口

用第一性原理表示，Writer 必须实现：

$$
F:\; (P,\;B_X,\;B_Y)\longrightarrow
\{a_{j,r},b_{j,r}\}_{j=1}^{38,r=1}^{4},
$$

其中：

* \(P\) 是 language/video 形成的 Program；
* \(B_X,B_Y\) 是由同一教学视频在 frozen PI0.5 中形成的真实 native candidate banks；
* \(a,b\) 必须共同决定一个 rank4 policy tangent；
* 换成 wrong video/bank 后，方向必须发生**功能相关的旋转**，而不仅是幅度变化。

当前实现近似把这个函数分解为：

$$
d_B=d_0(P)+\sum_e g_e(S(P,B))\,A_e(P,B)+A_e^{\rm free}.
$$

其中 Program/bank 的高维交互先被压成 summary，后续只剩 family scalar gate 和 additive anchor。代码中 `input_primal_gate`、`output_primal_gate` 对每个 rank/event只输出一个标量，`_add_native_anchor` 再执行 gate 与 anchor 的乘积求和；可选 A_free 是跨 arms、banks、videos 共享的 full-native anchor。

因此我把最早接口表述为：

> **Program-conditioned current-bank tangent formation，而不是 summary-conditioned scalar correction。**

## 3.2 候选根因排序

|     排名 | 候选解释                                       | 状态                       | 证据判断                                                                                                                                          |
| -----: | ------------------------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
|  **1** | 当前 B0/B1、base、anchor、scalar gate 的因子化错误    | **已证明，范围明确**             | S0证明下游 B1 有解；S1 correction-only损失correct；primal恢复correct却丢specificity；Q/A/F=0充分控制仍呈capacity–specificity折衷；最终几何直接显示summary/gate/free delta高度平行 |
|  **2** | Program 与真实 bank 的联合坐标/方向不可识别              | **强推断，是当前最早未知接口**        | bank-dependent candidate delta尚有`.718–.772`差异，但在后续公共方向中消失；P1证明给定 primal 后 current-bank dual有容量；缺的是从 Program 与当前 bank共同得到 primal               |
|  **3** | shared Program→bank selection 欠识别          | **强推断**                  | task-local upper bounds反复通过，shared S2/J2/R6等失败；但尚未测试矩阵值 bank transport，因此不能说所有共享规则都不存在                                                        |
|  **4** | functional Panel 或 staged supervision 制造问题 | **已证明是重要混杂，但不是当前首因**     | effective surrogate与direct functional梯度近正交；早期TF32也制造过假问题；但 direct-functional、充分校准和同checkpoint F=0 后当前结构仍失败                                    |
|  **5** | Program 本身没有功能信息                           | **未知，但现有证据不支持把它排第一**     | G2只证明动态，不证明行为；G2-B较弱。但R10能把真实Natural Program推到`.56/.54`的中等功能效用。最新所谓“real S1”又实际使用固定task token，不能作为G2 Program失败证据                              |
|  **6** | correct/wrong 功能梯度在数据上本质冲突                 | **只在当前参数化中成立；全局解释被反驳**   | candidate v4中梯度约`-.966`，但free-summary S0、vector task-local witness和Program-through-bank S0均能同时恢复correct并强压wrong，所以冲突不是环境本质，而是当前坐标下的梯度冲突       |
|  **7** | task diversity 不足                          | **未知、可能是后续问题，但不是当前最早接口** | shared模型只在有限task上训练可能过拟合；然而当前失败在task1/93 task-local fixed-token资格中就出现，不需要先诉诸更多task                                                            |
|  **8** | native bank有task-local解但没有可共享规则            | **真正未知**                 | 所有 task-local pass都不能证明shared rule存在；但现在尚未测试 bank-dependent矩阵值 operator，因此也没有理由判定不存在                                                          |
|  **9** | fixed carrier12 + residual4 限制             | **当前证据显著削弱**             | G1 `114/250`、mobile-rank4解析容量、P1和多种task-local upper bound均成立；只有在新的transport已经充分而full-rank16明显更强时才应重开                                          |
| **10** | zero-interaction/action-hidden合同本身信息不足     | **远未证明**                 | GOMQ rank32 `151/400`、G2动态、R10及多种bank-specific upper bound均说明输入有用；但最终共享、因果、稳定地超过145仍是开放问题                                                     |

## 3.3 已证明、强推断与未知的边界

**已证明：**

* Action Expert rank16 LoRA有闭环容量。
* mobile rank4 在 held5 五个 task 上有解析容量。
* 真实 native banks + signed pooling 可形成强 task-local rank4。
* Natural Program含有可用的有序视频动态。
* 给定合适 primal，current-bank global dual/exact replay 能跨视频保留方向。
* summary-only B1存在强 task-local capacity–specificity 解。
* 当前 scalar-gated additive-anchor 参数化无法同时满足 task93 correct、wrong 和 margin。
* 这个 non-pass 不是 anchor under-travel、普通 query step、OOM、TF32、Action Meta或held梯度泄漏造成的。

**强推断：**

* 当前最早缺口是 bank-relative高维方向形成。
* 现有 shared coordinate 把可用 bank 差异压成了近公共方向。
* 继续改 scalar gate、anchor scale、普通 width/LR不会产生新机制证据。

**未知：**

* 冻结 G2 Program在一个合适的 bank-relative operator中是否足够。
* 使用更多 non-held tasks 后是否存在可泛化的共享规则。
* rank4 在最终 validation8 的完整目标线上是否足够。
* Final 端到端训练是否会自动形成比 staged组件更好的内部表示。
* zero-interaction合同最终能否稳定达到 `>145/400`。

---

# 4. 对继续 ECP、另开架构、直接 joint 或根本停止的裁决

## 主要选择：A

**保留 ECP Native-Factor，但替换 Program–bank 联合方向接口。**

应继承的资产：

* frozen generic PI0.5 与38个目标层；
* Stage 0 native observer与真实 native X/Y hooks；
* G2 owner-specific Natural Program和boundary-anchored ordered alignment；
* 每视频单位质量、K视频置换不变聚合；
* q/v/action-in/action-out及output abs/adj/init/goal的真实分组；
* IEEE current-bank statistics；
* exact antithetic signed pooling；
* small-core balanced canonicalization；
* carrier12 + residual4 的唯一完整rank16物化；
* set-valued functional critic与cross-episode action/flow infrastructure；
* Action Meta关闭。

应停止的历史包袱：

* neural q_\pi、fingerprint、phase latent；
* fixed effect code和fixed realizer；
* Program-only absolute base primal；
* family scalar gate；
* shared free anchor；
* binary full/half routing；
* global free token；
* base + bounded correction；
* surrogate/effective-rank4 target作为正式Gate；
* “先预测一个与bank无关的方向，再让bank只做轻微修正”。

## 为什么不是 C

C要求现有证据已经接近“Native-Factor或zero-interaction合同本身不可行”。目前没有：

* G1和多个task-local Gate都成立；
* G2 Program不是静态常量；
* R10证明真实 Program能接受functional credit；
* P1证明 native-bank transport有高容量；
* S0证明 bank specificity与correct capacity可同时存在；
* GOMQ说明 video-conditioned参数生成确实能达到接近目标的闭环分数。

真正未解决的是共享规则，而不是输入、输出和所有中间资产都失败。

## 为什么不直接用当前实现做 D

把当前 scalar-gated architecture直接长程 joint training，会产生无法解释的结果：

* 若失败，不能区分参数化错误与优化失败；
* 若小幅提高，可能只是 Program-only base或shared anchor吸收了更多训练task；
* 若训练task高、held低，又会回到“是否task diversity不足”的循环。

因此 D 应当成为 **A 的紧接后续**，而不是用来回避已定位的接口。

## 次选：B，但有明确触发条件

只有当下述 matched Gate 成立时转 B：

1. topology-matched free query + real bank transport能通过；
2. 去掉task-local query，换成冻结 G2 Program、无task ID、无Program-only value path后，仍在task1和task93系统性无法保留correct/held；
3. full Program也不优于language/endpoints，且不是优化或bank chart问题。

这会把首因推进到 **Natural Program瓶颈本身**。届时新分支应直接让 exact language与ordered native-bank tokens共同产生signed measure，而不是恢复旧Writer、GOMQ、q_\pi或fixed realizer。

---

# 5. 唯一推荐架构：Program-Conditioned Native-Bank Tangent Transport

记：

* target \(j\in\{1,\ldots,38\}\)；
* mobile rank \(r\in\{1,\ldots,4\}\)；
* event \(e\in\{1,\ldots,E\}, E\le8\)；
* video \(k\in\{1,\ldots,K\}\)；
* bank side/type \(s\in\{X,Y_{\rm abs},Y_{\rm adj},Y_{\rm init},Y_{\rm goal}\}\)。

## 5.1 Pass A：保留什么

保留当前 G2 schema：

$$
\begin{aligned}
P_{\rm lang}&\in\mathbb R^{38\times128},\\
P_{\rm scene}&\in\mathbb R^{38\times128},\\
P_{\rm process}&\in\mathbb R^{E\times38\times128},\\
\rho&\in\mathbb R^E,\\
\tau&\in\mathbb R^{E\times2},\\
\sigma&\in\mathbb R^{E\times38\times128}.
\end{aligned}
$$

每条视频继续独立保序编码；跨视频在 boundary-anchored monotonic event alignment 后，以固定 \(1/K\) 质量聚合。`NaturalProgram` 当前实现已经把静态旁路移除，并使 temporal output 读取 event-bearing `P_process`；这一部分不应重新设计。

对每个 target/rank/event/side 形成 query context：

$$
z^{s}_{jre}
=
\operatorname{LN}\Big[
P_{\rm lang,j},
P_{\rm scene,j},
P_{\rm process,e,j},
\rho_e,
\tau_e,
\sigma_{e,j},
o_j,
u_r,
v_e,
t_s
\Big].
$$

其中 \(o_j,u_r,v_e,t_s\) 是**LoRA target、rank、event和native type的固定结构身份**，不是环境task ID。

共享 query trunk 输出：

$$
q^{s}_{jre}
=
\operatorname{normalize}
\left(Q_s z^s_{jre}\right)
\in\mathbb R^m.
$$

不允许存在 \(z\to\mathbb R^{d_{\rm in/out}}\) 的直接 additive native vector。

## 5.2 真实 bank candidate 如何成为 key 和 value

对 target \(j\)、side \(s\) 的每个真实 candidate：

$$
v^s_{j,n}\in\mathbb R^{d_{j,s}}.
$$

这些就是现有 Pass B 捕获的真实 X/Y、adjacent/init/goal differences；不生成 synthetic basis，不读取teacher action。现有 native hooks、candidate multiplicity和output type partition应原样保留。

先做当前bank内固定归一化：

$$
\hat v^s_{j,n}
=
\frac{v^s_{j,n}-\bar v^s_j}
{\operatorname{rms}_B(v^s_j)+\epsilon}.
$$

key：

$$
k^s_{j,n}
=
\operatorname{LN}
\left(
K^s_j\hat v^s_{j,n}
+
M_s\,m_{j,n}
\right)
\in\mathbb R^m,
$$

其中：

* \(K^s_j\) 是按 LoRA target区分、跨环境task共享的 key projection；
* \(m_{j,n}\) 只含允许的结构信息：frame/horizon、antithetic probe、output type、target/family；
* 不含task ID、filename、pose、reward、outcome；
* **value始终是原始真实 \(v^s_{j,n}\)**，不是神经value decoder。

这与旧 fixed realizer 的关键差异是：`K`只决定如何寻找candidate，不能独立产生policy方向；最终方向必须属于当前bank真实candidate的signed span。

## 5.3 跨视频与 event measure

对视频 \(k\) 中 candidate \(n\)：

$$
\mu^{(k)}_{e,n}
\propto
M^{(k)}_{\operatorname{frame}(n),e}
\cdot q_{\rm frame}(n)
\cdot q_{\rm multiplicity}(n),
$$

并强制每条视频在每个有效scope总质量相同。然后：

$$
\mu_{e,n}
=
\frac1K\sum_{k=1}^{K}\mu^{(k)}_{e,n}.
$$

这样：

* 不是平均raw frames；
* 不是先为每条视频产生LoRA再平均；
* 不是学习video reliability；
* 最终只有一个联合candidate measure和一个rank4 residual。

## 5.4 B0：只做 current-bank 几何，不再产生“base”

B0流式计算：

$$
\bar k^s_{je}
=
\sum_n\mu_{e,n}k^s_{j,n},
$$

$$
C^s_{je}
=
\sum_n\mu_{e,n}
(k^s_{j,n}-\bar k^s_{je})
(k^s_{j,n}-\bar k^s_{je})^\top
+\lambda I.
$$

在IEEE FP32或必要时FP64中做 Cholesky/eigendecomposition：

$$
u^s_{jre}
=
(C^s_{je})^{-1/2}q^s_{jre}.
$$

这里仍然需要显式 B0/B1，但含义改变：

* **B0不是产生 base primal。**
* **B1不是给base加bounded correction。**
* B0只建立当前bank的坐标与whitening；
* B1在同一bank上执行唯一exact signed selection。

调试时还可只读计算 cross-covariance：

$$
T^s_{je}
=
\sum_n\mu_{e,n}
(v^s_{j,n}-\bar v^s_{je})
(k^s_{j,n}-\bar k^s_{je})^\top,
$$

但 \(T\) 不必在deployment显式物化。

## 5.5 B1：Program 与 bank 在 candidate 权重处发生高维交互

重放同一bank：

$$
\ell^s_{jre,n}
=
\frac{
(u^s_{jre})^\top
(k^s_{j,n}-\bar k^s_{je})
}{
\operatorname{rms}_{\mu}(\ell)+\epsilon
}.
$$

形成 antithetic measure：

$$
\alpha^{s,+}_{jre,n}
=
\operatorname{softmax}_{\mu}
\left(\frac{\ell^s_{jre,n}}{\theta_s}\right),
$$

$$
\alpha^{s,-}_{jre,n}
=
\operatorname{softmax}_{\mu}
\left(-\frac{\ell^s_{jre,n}}{\theta_s}\right).
$$

方向：

$$
d^s_{jre}
=
\sum_n
\left(
\alpha^{s,+}_{jre,n}
-
\alpha^{s,-}_{jre,n}
\right)
v^s_{j,n}.
$$

在小score极限，一阶展开为：

$$
d^s_{jre}
\approx
\frac{2}{\theta_s}
\operatorname{Cov}_{\mu}(v^s,k^s)
(C^s_{je})^{-1/2}
q^s_{jre}.
$$

这是本架构的核心：

* 当前bank通过 \(\operatorname{Cov}(v,k)\) 决定“query的哪个方向映射到哪个policy方向”；
* wrong bank不仅改变一个标量，也会改变整个线性算子；
* Program不能绕开bank；
* bank也不能在没有Program query时自行输出固定task adapter。

## 5.6 唯一 rank4 residual

对每个 target/rank：

$$
a_{j,r}
=
\sum_e \tilde\rho_e\,d^X_{jre},
$$

$$
b_{j,r}
=
\sum_e\tilde\rho_e
\sum_{s\in Y}
c_{j,s}\,d^s_{jre}.
$$

\(c_{j,s}\) 只做task-independent、预注册的family/type RMS平衡，不能依condition输出family scalar gate。

随后：

$$
\Delta W_j
=
\sum_{r=1}^{4}
b_{j,r}a_{j,r}^{\top}.
$$

再沿用现有 small-core balanced SVD/canonicalization，严格物化：

$$
\text{LoRA}_{j}
=
\text{carrier12}_{j}
\oplus
\text{residual4}_{j}.
$$

现有 `native_materialization.py` 已检查38 targets、rank4 residual和carrier12拼接成一套完整rank16；这一实现应直接复用。

## 5.7 梯度路径

训练期直接功能 loss 的梯度为：

$$
L_{\rm func}
\rightarrow
\pi_{\theta+\Delta\theta(P,B)}
\rightarrow
\Delta W_j
\rightarrow
d^X,d^Y
\rightarrow
\alpha^\pm
\rightarrow
q,K,C^{-1/2}
\rightarrow
P.
$$

* frozen source/backbone仍可对生成LoRA做VJP，但自身不更新；
* native X/Y可缓存为常数；
* key/query/whitening和最终signed measure保留梯度；
* task-local资格时冻结Pass A；
* whole-Writer joint时解冻 Stage0、Pass A和PNBTT；
* 不再把factor cosine、reconstruction或effective teacher作为正式优化目标。

## 5.8 防止 absolute task-code 旁路的硬合同

实现必须加入以下不可协商测试：

1. **zero-value test**：把真实 candidate values 全置零，residual必须精确为零。
2. **no-Program-value-path**：Program输出不能直接进入 \(d_{\rm in/out}\) native vector。
3. **bank permutation invariance**：candidate顺序改变，结果只允许浮点容差差异。
4. **video permutation invariance**：K视频在各自保序编码后换序，最终LoRA不变。
5. **bank swap sensitivity**：固定Program、替换真实bank，\(C\)、signed weights和最终方向必须改变。
6. **same-query arm identity**：task-local free query资格中，correct/wrong bank必须共享完全相同query，不能按arm查表。
7. **single-adapter**：任何condition只物化一套38-target rank16。
8. **no learned video reliability**：K条视频固定等质量。
9. **no task lookup**：deployment state dict中不存在环境task/task-pair参数。
10. **Action Meta absent**。

## 5.9 当前模块的保留、删除与替换

| 模块                                              | 决定                            |
| ----------------------------------------------- | ----------------------------- |
| Stage0 native observer                          | 保留；Final随机候选中随机初始化            |
| G2 Natural Program / boundary alignment         | 保留并作为component-init；joint时可训练 |
| 真实X/Y与四类output bank                             | 保留                            |
| candidate multiplicity、frame quadrature         | 保留                            |
| IEEE current-bank covariance                    | 保留，改为key-space whitening      |
| exact signed replay                             | 保留                            |
| small-core canonicalization                     | 保留                            |
| carrier12 + residual4                           | 首版保留                          |
| R5 absolute Program primal                      | 删除deployment主路径               |
| fixed base + bounded correction                 | 删除                            |
| family scalar gate                              | 删除                            |
| shared A_free anchor                            | 删除                            |
| global/free summary token                       | 仅可用于诊断，不进入候选                  |
| binary full/half operator                       | 删除                            |
| effective-rank4 teacher loss                    | 只可作只读诊断，不作Gate                |
| Program-conditioned candidate key/value decoder | 用PNBTT query/key/value替换      |

## 5.10 如何扩容

不应直接扫 width、head或rank。扩容顺序必须由机制证据触发：

1. 先看训练task上 \(T=\operatorname{Cov}(v,k)\) 的功能梯度投影谱；
2. 若 key dimension \(m\) 截断了跨family的有效谱，再提高 \(m\)；
3. 若单一key chart无法覆盖q/v/action-in/out，先改为family-shared trunk + target-specific低秩 key projections；
4. 只有 PNBTT task-local/full-rank16 oracle明显优于rank4 residual，才重开carrier/task rank分配；
5. task数扩大时优先共享query trunk和family trunk，target-specific参数只解决维度不一致，不能变成task表。

---

# 6. 最小决定性实验序列及每个 Gate

## 阶段 E0：证据与实现合同

**使用数据：** task1、task93各一个已缓存correct/wrong condition；不产生训练梯度。

**训练/冻结：** 全部冻结。

**输出：** PNBTT B0/B1统计、signed weights、38-target residual和完整rank16。

**验证问题：** 新图是否真的满足“Program只能query bank、bank value决定方向”。

**Gate：**

* 上述十项hard tests全部通过；
* chunked/non-chunked、candidate permutation、K permutation一致；
* IEEE solve有限；
* step0 residual为零；
* Action Meta参数数为0；
* 单一完整rank16被真实policy消费。

**失败解释：** 工程或拓扑错误，不进入任何科学训练。

**主要瓶颈：** native bank replay；可复用现有冻结X/Y cache，B0只保存 \(O(m^2)\) sufficient statistics。

---

## 阶段 E1：topology-matched free-query transport capacity

**tasks与数据：**

* task1，wrong task8；
* task93，wrong task94；
* 每task correct fit0、fit1、held各一条视频；
* wrong fit0、fit1；
* same correct exact language；
* K=1开始。

**产生梯度的数据：**

* correct fit0、fit1；
* wrong fit0。

**零梯度：**

* correct held；
* wrong fit1；
* Panel-B；
* validation/test；
* shuffled/reversed。

**训练：**

* 每个task一个training-only free query tensor，但同一task的correct/wrong、fit/held共享同一query；
* task-shared target/family key projections；
* PNBTT query whitening与固定type normalization。

**冻结：**

* source/backbone；
* Stage0/native bank；
* Natural Program；
* carrier；
* scale；
* native values和exact replay；
* 不存在free native direction、free anchor或free summary tree。

**输出：** 真实bank signed span中的唯一rank4 residual。

**因果问题：**

> 当唯一可学习的condition自由度只是同一个query，而policy方向必须来自当前真实bank时，当前bank operator能否同时恢复correct并压低wrong？

**Gate：** 直接复用现有双task绝对合同：

* correct fit各自 `≥.85`；
* correct held `≥.80`；
* wrong各自 `≤.25`；
* minimum correct − maximum wrong `≥.50`；
* all-correct > all-wrong；
* near-bound fraction不超过现有上限；
* 相邻两个checkpoint结论一致。

**失败解释：** PNBTT transport函数类本身不足；不能责怪Natural Program或task diversity。此时应修改bank tangent parameterization，而不是进入joint。

**通过后：** fresh进入E2，不加载task-local query或optimizer；E1只提供准入证据。

**吞吐：** 预计与现有S0/S1同级，主要是两次bank replay和Panel VJP。task1/93可以各占一张GPU，shared key梯度按task-equal权重聚合。

---

## 阶段 E2：真正的 frozen Natural Program → bank tangent Gate

这是当前历史中真正缺失的资格。正式 S1 虽然命名为“real Program-through-bank”，配置的 `program_source` 实际为：

`fixed_nontrainable_128d_orthogonal_task_token`

同时 `natural_program` 被列入 frozen，而不是作为每条真实视频的 query source。因而它测试的是“fixed token query真实bank”，不是部署态 G2 Program。

**tasks与数据：** 与E1完全相同；增加K=1/2/4 matched views。

**产生梯度的数据：** correct fit0/fit1、wrong fit0。

**训练：**

* shared `Program→query` trunk；
* target/family key projections；
* PNBTT bank transport。

**冻结：**

* frozen G2 Natural Program及boundary alignment；
* source、Stage0/native X/Y、carrier、scale；
* 无task token、无task ID。

**输出：** 每个language/video condition的一套rank16。

**因果问题：**

> 真实 G2 Program 是否含有足够信息，通过一个会随bank旋转的operator形成correct-specific方向？

**Gate：**

* E1的全部绝对 correct/wrong/held Gate；
* same-task held必须保持；
* K1/K2/K4方向与功能均保持；
* full Program对language-only和first+final/endpoints必须在task1和task93均形成正的paired功能差，并在相邻checkpoint保持同号；
* 不使用shuffled/reversed。

**失败解释：**

* E1通过而E2失败：最早接口推进到Natural Program→query或Program信息本身；
* 若full不优于endpoints：G2的内部动态未转化为功能充分性；
* 这时启动次选B，而不是再做ECP G3局部修补。

**通过后：** 立即进入共享/整体训练，不再增加新的task-local数学变体。

---

## 阶段 E3：共享 PNBTT 功能资格，嵌入整体训练的早期段

E3不应成为另一个长达多日的组件项目，而应是 whole-Writer run的预注册早期检查点。

**gradient tasks：**

* meta：`1, 8, 9, 32, 52`；
* target：`72, 73, 75, 93, 94`。

**true task-held：**

* meta task2；
* target task74。

这与现有G3 task split保持一致，避免因换task重新解释结果。正式S1配置也记录了该划分。

**数据：**

* 每个gradient task两条fit视频、一条held视频；
* video/action跨episode；
* correct exact language固定；
* wrong-video/bank按预注册family-balanced pairing；
* K在1/2/4间按固定分布采样。

**产生梯度：**

* gradient tasks的correct fit与wrong fit；
* cross-episode action/flow Panel-A。

**零梯度：**

* same-task held；
* task2/74；
* Panel-B；
* validation/test；
* shuffled/reversed。

**训练：**

* shared Program→query；
* shared/family key trunks与target-specific必要projection；
* PNBTT；
* Natural Program暂时冻结。

**冻结：**

* source/backbone；
* Stage0/native bank；
* carrier；
* scale先冻结。

**Gate：** 复用此前P2/shared mapping合同，而不是发明新指标：

* held median recovery `≥.75`；
* held p10 `≥.50`；
* held/fit `≥.80`；
* correct-vs-wrong median margin `≥.10`；
* q/v/action-in/action-out四family均有正贡献；
* task2/74均为正；
* K1/K4与same-task保持；
* 两个相邻checkpoint结论一致；
* direct functional为正式Gate，factor/effective指标只读。

此前P1明确把这些门留给shared Program mapping，而没有把P1高分当作shared成功。

**失败解释：**

* E2通过、E3失败：task-local信息足够，但共享规则/任务覆盖不足；
* 这时不再改PNBTT局部模块，而是在相同架构下扩大到全部授权non-held tasks并直接joint；
* 若扩大数据仍只记忆gradient tasks，再裁决是否另开B。

**通过后：** 同一fresh run解冻Pass A并进入E4；不重新初始化。

**主要瓶颈：** functional PI0.5 VJP，而不是PNBTT的小矩阵solve。

---

## 阶段 E4：matched whole-Writer joint adjudication

**候选A：component-init**

* Stage0、G2 Program、PNBTT由通过的组件初始化；
* fresh optimizer/scheduler；
* joint时全部Writer可训练。

**候选B：fully-random**

* 只有source/backbone和carrier冻结；
* Stage0、Natural Program、PNBTT及所有Writer heads全部随机初始化；
* 同一计算图、参数量、数据、loss、task sampling、update数和checkpoint cadence。

**开发数据：**

* train24；
* 经精确语义审计并排除validation/test重合的LIBERO-90 non-held tasks；
* validation8只做正式选择，不产生梯度；
* held5只作低成本机制/趋势监控，不能替代validation8。

**训练：**

* 全部Writer；
* scale只有在PNBTT方向已经成立、残余表现明确受幅度限制时才解冻；
* source/backbone、carrier始终冻结。

**输出：** 每个condition唯一一套38-target rank16。

**Gate：**

* validation8 strict paired correct严格 `>145/400`；
* 不接受单点峰值；
* 至少相邻single checkpoints稳定；
* 低churn、高breadth；
* Spatial/Object/Goal/Long均非零；
* Goal/Long有实质贡献；
* same-task不同视频高retention；
* full优于language/no-video、endpoints和wrong-video；
* checkpoint选定并冻结后才测试shuffled/reversed。

这些是owner唯一正式性能合同。

**失败解释：**

* component-init成功、random失败：分阶段资产有实质价值；
* random成功、component-init失败：G1/G2课程形成了有害先验；
* 两者functional指标通过但闭环都失败：Panel/flow objective与closed-loop出现根本鸿沟；
* 两者都未产生稳定闭环增量：才开始接近ECP或zero-interaction根本停止讨论。

---

# 7. G4/Final、fully-random Writer、最小 loss 与 Action Meta

## 7.1 现在是否已经有足够组件证据进入整体裁决

**有。**

足够的含义不是“当前shared compiler已解决”，而是：

* downstream native rank4 realization有容量；
* Program有动态；
* direct functional credit能传到Program；
* bank transport给定primal时有容量；
* correct/wrong specificity存在task-local解；
* 当前失败的结构位置已经足够清楚。

因此没有理由继续做更多 summary、quotient、scalar gate、free anchor 或binary route变体。

唯一前置Gate是：

> **PNBTT 的真实bank transport capacity，以及去掉task token后真实Natural Program能否驱动它。**

这两个Gate通过后，继续分解组件的边际信息量低于whole-Writer joint。

## 7.2 最小充分 loss

我不建议再叠加 behavior-Gram、factor reconstruction、cosine、hidden separation、chart alignment、effective teacher和多个polish目标。最小loss只需要三项。

### 1. Correct cross-episode functional loss

condition来自task \(t\) 的language与video episode \(A\)，监督来自独立episode \(B\)：

$$
L_{\rm task}
=
\operatorname{softmin}_{m\in\mathcal M_t}
\sum_{\xi\in B}
w_\xi\,
\ell_{\rm flow/action}
\left(
\pi_{\theta+\Delta\theta(c)}(\xi),
a^{(m)}_\xi
\right).
$$

* \(\mathcal M_t\) 是多个独立successful expert members；
* critic是set-valued，不要求Writer复现某个特定LoRA；
* video/action严格跨episode；
* deployment仍看不到action。

### 2. Wrong-video necessity loss

同一correct exact language，分别配correct video与wrong video：

$$
L_{\rm necessity}
=
\left[
m+
L_{\rm task}(c_{\rm correct})
-
L_{\rm task}(c_{\rm wrong})
\right]_+ .
$$

这使video成为功能必要条件，而不是匹配开关。wrong条件不应被训练成任意破坏policy，因此必须与preservation一起使用。

### 3. Preservation loss

$$
L_{\rm preserve}
=
D_{\rm policy}
\left(
\pi_{\theta+\Delta\theta(c)},
\pi_{\rm carrier}
\right)
\quad\text{on source/unrelated states}
$$

并对wrong-video adapter加入“不能比carrier显著更坏”的单侧约束。

总loss：

$$
L
=
L_{\rm task}
+
\lambda_{\rm nec}L_{\rm necessity}
+
\lambda_{\rm pres}L_{\rm preserve}.
$$

权重只允许用train tasks上预注册的梯度量级校准，不能用validation8调节。所有loss按task、family和carrier可改善空间归一化，避免action-out或长episode支配。

**不加入：**

* shuffled/reversed训练；
* factor MSE；
* LoRA cosine；
* hidden separation；
* task-ID classification；
* teacher-LoRA reconstruction；
* endpoint判别辅助loss；
* learned video reliability。

## 7.3 on-policy loss

只有在离线cross-episode功能loss已经产生稳定闭环增量后，才加入条件式on-policy：

* 只对当前Writer尚未完成但有部分进展的状态采样；
* reward/terminal只在训练期给共享Writer梯度；
* 不进入deployment输入；
* 保持correct/wrong-video与preservation三项；
* 不能把rollout后task-local更新混入最终分数。

否则，过早on-policy会把“不会生成正确LoRA”与“policy探索差”重新混在一起。

## 7.4 fully-random Writer如何公平比较

公平性必须包括：

* 同一PNBTT拓扑；
* 相同参数量；
* 相同task与video/action rows；
* 相同global task/family权重；
* 相同optimizer、scheduler、更新数和gradient calls；
* 相同checkpoint cadence；
* 相同validation读取次数；
* 相同world topology；
* 相同信息墙。

区别只能是初始化：

* component-init加载Stage0/G2/PNBTT通过的组件；
* random除source/backbone/carrier外全部随机。

结果同时报告：

1. 从joint起点开始的matched compute；
2. component-init此前组件训练的总附加compute。

这样既公平比较最终优化，也不隐藏课程预训练成本。

## 7.5 Action Meta

继续关闭的边界合理。

只在以下条件同时成立时做一次 matched on/off：

* base Writer相对carrier有稳定closed-loop增量；
* 至少两个相邻checkpoint success set高度重合；
* Goal/Long已有非零贡献；
* 误差审计明确集中在action-in/out控制细节，而不是task识别、对象选择或过程顺序；
* Action Meta仍只能并入同一最终rank16，不成为第二adapter。

若增量只体现在内部action loss、但闭环breadth或retention下降，则永久关闭。

---

# 8. 吞吐、GPU与数据规模扩展

## 8.1 仓库中已有的实测边界

* Program-through-bank S0：task1/task93峰值约 `41.30/29.29GB`，median step约 `5.42/11.24s`。
* S1：峰值约 `41.24/39.84GB`，median step约 `26.21/22.53s`。
* P1六task在六卡上的总墙钟约3分47秒，说明task-parallel资格本身可以很快；真正昂贵的是反复functional replay与复杂set interaction，而不是所有G3都天然耗时。

owner合同允许1–6张真实提高吞吐的A40弹性分片，world size不能写死为2；科学权重、optimizer cadence和exact-resume topology必须保持。

## 8.2 PNBTT的主要计算瓶颈

按重要性排序：

1. frozen PI0.5 action/flow functional forward与VJP；
2. 38 targets的native candidate key projection；
3. B1 exact signed replay；
4. K视频、多output type的candidate重放；
5. 生成rank16后的真实policy forward；
6. 最后的simulator closed-loop。

\(m\times m\) covariance solve本身不是主要瓶颈，只要 \(m\ll d_{\rm native}\)。

## 8.3 缓存边界

可缓存：

* frozen Stage0/native X/Y；
* candidate metadata与frame/event索引；
* frozen action/flow Panel rows；
* quadrature和multiplicity；
* carrier activations；
* Program在E1/E2冻结时可缓存。

不能缓存：

* joint阶段可训练Natural Program的输出；
* trainable key projection；
* PNBTT signed logits；
* 任何由当前Writer checkpoint决定的LoRA或policy outcome。

## 8.4 两遍流式执行

### B0

每张卡处理candidate shard，累计：

* mass；
* key mean；
* key second moment；
* 必要时cross-cov diagnostic。

然后all-reduce sufficient statistics，统一求 \(C^{-1/2}\)。

### B1

各卡重放自己的candidate shard，先得到global log-partition所需的max/sum-exp统计，再all-reduce，最后累计真实value signed sums。

这样：

* 不需在单卡物化全部candidate；
* 不改变candidate科学权重；
* 不平均每卡LoRA；
* 最终只在全局统计上形成一个rank4 residual。

## 8.5 task与condition并行

* 不同task/condition天然data parallel；
* 同一task多个video可并行编码，但event-aligned sufficient statistics必须在形成LoRA前合并；
* task1与task93耗时差异较大，应按profile做动态worker分配；
* 动态调度只能改变墙钟，不能改变task/family在loss中的权重；
* 每个optimizer step先按task内video等权，再按family等权，再按task等权。

## 8.6 数值精度

* frozen PI0.5大矩阵forward可用已验证的BF16；
* covariance、whitening、log-partition和small-core canonicalization使用IEEE FP32；
* 条件数过高时只把小矩阵solve升到FP64；
* 不恢复TF32 covariance；
* candidate chunk边界必须固定或使用可重现归并树，避免P0曾出现的近 \(10^6\) 条件数下chunk drift。

## 8.7 数据规模扩大

顺序应是：

1. task1/93 topology Gate；
2. 10 gradient +2 task-held shared Gate；
3. train24；
4. 审计后的71 non-overlap LIBERO tasks；
5. 方法冻结后按owner合同fresh使用32 source tasks并评测test8。

增大数据时保持：

* 每task相同采样质量；
* 每video相同质量；
* successful lineage独立，不把同一训练轨迹的多个checkpoint当独立知识；
* wrong-video pairing跨family平衡；
* validation/test不产生梯度。

不应通过给“容易任务更多rows”来人为提高整体loss下降，否则shared rule会再次退化为Spatial/Object优先、Goal/Long缺失。

---

# 9. 文档—代码—实验—专家原文冲突

## 9.1 README的“当前状态”已经过期

固定提交的 README 仍声称“当前唯一活动资格是 J2 joint Program–primal functional qualification”。实际 history §91–96 已经经过R1–R13、candidate interaction、EBSRI、Program-through-bank和bank-conditioned primal，并在 calibrated A_free non-pass处停止。README的阅读顺序也只列到第四份后续专家意见，遗漏了8月30日、8月31日和9月1日三份关键原文。

## 9.2 owner requirements §4的架构段落落后于动态进度

`current_owner_requirements.md` 仍把“full base query + candidate bounded correction”写成当前方向，但 pinned commit已经停止这一函数类。稳定目标、信息墙和Final合同仍有authority；其中“当前方法实现”应由 `task_plan/findings/progress/research_history` 的后续结论覆盖。

## 9.3 S1的“real Program”命名不准确

S1文档反复称“real Program-through-bank”，但正式config写的是：

* `program_source = fixed_nontrainable_128d_orthogonal_task_token`
* `natural_program` frozen；
* deployment candidate=false。

所以这轮准确名称应是：

> **fixed-token, real-bank set-read S1**

而不是“真实 G2 Natural Program S1”。它能定位real B0/set read问题，却不能直接说明G2 Program信息不足。

## 9.4 config状态字段滞后

S0/S1和若干primal配置仍标记 `active_*_qualification`，但当前authority已经正式停止该参数化。代码和配置可以保留用于复现，但应将status改为`retired_after_formal_non_pass`或增加superseded authority，避免后续自动化恢复旧路线。

## 9.5 quotient的早期归因后来被推翻

三类quotient最初被读作“summary-only不可行”。后续代码审计发现：

* raw Program旁路是否存在同时改变；
* inducing-dependent字段被覆盖；
* 一个global free token被广播到原本应逐target/group/type的scope。

因此它们只证明“过窄global condition + 删除raw code”失败。最新history已经正确收回了更强归因。

## 9.6 S2 effective failure不能直接当架构失败

effective-rank4 surrogate与direct functional gradient median cosine只有约`.0219`，且6/16为负。这说明至少一部分“shared S2失败”来自监督目标错位。好在后续direct-functional和polish也未解决迁移，因此当前结构仍可判non-pass，但淘汰范围必须分别记录。

## 9.7 P1“operator已解决”的表述容易过强

P1确实证明：

* 给定task-local稳定primal；
* 当前bank global dual；
* exact signed replay；
* 跨video有高容量。

它没有测试wrong-bank specificity。后来R5/primal结果显示同一机制可以在correct和wrong bank上都高，因此“operator已解决”只能指**capacity transport**，不能指conditional selection。

## 9.8 R11 raw Stage0不能支持“Program压缩无害”或“有害”

R11同时改变了输入形态、chart接口和参数分布，因此只能说明那一个raw probe更差。不能把它作为继续冻结G2 Program的决定性依据。

## 9.9 GOMQ的151与当前目标不是同口径结构比较

151来自rank32；当前合同是rank16。统一rank16后为136。它仍是重要证据，但不能用“151超过145”直接宣称当前部署合同已被历史方法满足。

## 9.10 functional Panel不是closed-loop

多个文档段落把“功能恢复”“policy functional evaluation”和“闭环效果”在叙述上靠得很近。G3的Panel-B是frozen-policy action/flow functional probe，不是LIBERO simulator strict rollout。到当前提交为止，**没有一个shared ECP G3 checkpoint被证明达到新的正式closed-loop增量**。

## 9.11 final训练task authority仍存在轻微不统一

历史专家原文曾写“71 meta + train24”；owner稳定文档又写方法确定后fresh使用32 source tasks评测test8。合理解释应是：

* 71+train24可用于development/meta-training；
* 最终方法冻结后按固定benchmark协议fresh重训32 source tasks并测test8。

这应在Final config中明确写成两个不同stage，避免把validation选择与test训练混在一起。

## 9.12 action-in span的32D/33D表述

早期分析把action-in结构上限简写为32D；后续考虑bias后修正为 `span(W,b)`、最多33D。主结论——scalar q-output span丢失Goal/Long必要方向——不受影响，但历史表格应统一使用33D上界。

## 9.13 没有发现冲突的关键合同

代码、config和定向测试在以下方面一致：

* correct held、wrong fit1、Panel-B零反传；
* wrong bank使用correct task exact language；
* Action Meta不存在；
* source/Stage0按资格配置冻结；
* 每个condition只物化一套38-target rank12+4 rank16；
* `compose_rank12_plus_rank4`验证所有targets完整；
* Program不能直接作为`bank_conditioned_primals`的显式参数进入最终base之外的B1 path。

因此当前non-pass不是明显的信息墙违规或多adapter作弊。

---

# 10. 远程仓库仍无法裁决的最小原始 artifact

远程仓库提交了结果摘要、配置、代码和artifact路径，但 `runs/`、`data/`、`models/` 被忽略。因而我能严格审查“提交的证据链和实现合同”，但不能独立重算每个formal结论。README也明确说明这些资产不在远端。

真正需要补充的最小artifact如下。

| artifact                                       | 最小内容                                                                        | 为什么必须                                                               |
| ---------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 当前S0/S1/primal/Q_free/A_free formal aggregates | 原始`aggregate.json`，每个arm的最终和相邻checkpoint指标                                  | 重新计算所有Gate，确认没有摘要抄写或聚合口径错误                                          |
| Panel-B逐行结果                                    | task、video、episode、row、member、base/generated loss、normalization denominator | 检查paired统计、task/family权重、correct/wrong margin和是否有少数row支配            |
| calibrated A_free checkpoint                   | model state、optimizer state、step70/110、free/candidate anchors、family gates  | 独立重算RMS、cosine、F=0和“under-travel已排除”                                |
| F=0因果审计原始张量                                    | 同checkpoint correct/wrong各层base、candidate delta、free delta、summary、gate     | 验证`.718–.772`、`.991`、`.965–.971`、`.993–.995`并定位哪些targets主导          |
| S1实际Program/query trace                        | 每condition的query来源标记、fixed token、B0 summary、native anchor                   | 解决“real Program”命名与配置的歧义，确认没有实际加载G2 tensors                         |
| G1 strict250逐rollout rows                      | source/carrier/generated/member success、suite/task/seed、adapter hash        | 验证114/250、breadth5/5、Goal2、Long1以及single-adapter对应关系                |
| G2 `c1493a1` Program tensors                   | `P_lang/P_scene/P_process/rho/tau/sigma`、alignment、K1/K4和same-task traces   | 判断PNBTT E2所需Program是否真的跨video稳定，而不仅是汇总probe分数                       |
| R5/R10/J2/R13逐condition结果                      | correct/wrong Program、correct/wrong bank、task-held、梯度或VJP摘要                 | 重新判断Program functional sufficiency、routing和跨task梯度冲突                |
| source/oracle/GOMQ原始strict rows                | validation8 48/250/151/136对应的task、suite、seed与checkpoint                     | 统一所有闭环baseline和rank32/rank16比较                                      |
| cache与split manifest                           | video ID、action episode ID、task role、member lineage、cache authority、commit  | 验证cross-episode、no validation/test gradient、lineage独立和wrong pairing |
| formal运行环境                                     | PyTorch/CUDA、TF32开关、GPU UUID、world topology、exit log、peak memory            | 排除旧TF32、设备映射、partial run和resume差异                                   |
| 生成的76个LoRA tensors                             | 每condition完整state dict和policy加载记录                                           | 独立确认没有缺target、混合adapter或评测加载错位                                      |

其中对**当前根因裁决**最必要的只有前三组：

1. calibrated A_free逐行Panel-B；
2. step110 checkpoint；
3. F=0逐层张量审计。

如果这三组与提交摘要一致，我对“当前scalar-gated additive-anchor参数化已经结束”的置信度很高。

对“PNBTT是否值得继续”的额外必要资产只有：

* G2真实Program tensors；
* 对应task1/93 native-bank cache；
* cross-episode Panel-A/B rows。

其余历史大规模raw rollout不应成为实现新接口的前置条件。

---

**最终建议可以压缩成一句执行指令：**

> 冻结并归档当前 G3；实现一个没有 Program-only value path、没有family scalar gate、没有shared native anchor的 Program-query/current-bank-key/real-X/Y-value exact signed transport；用task1/93做free-query与真实G2 Program两级因果Gate；通过后立即启动component-init与fully-random两个matched whole-Writer joint runs，并只用稳定closed-loop结果裁决EMBER。
