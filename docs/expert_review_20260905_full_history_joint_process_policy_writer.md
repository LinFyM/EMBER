# EMBER 历史复盘与下一步整体方案

**我的主判断是：早期 EMBER 已经取得了不能被近期弱结果抹掉的真实能力，但“当时主要只剩任务能力漂移”只得到部分支持。** v6-fast task-complete 的143/400确实非常接近当前数值门槛，但它同时存在能力集中、部分任务未解锁、视频特异性不足，以及训练后成功集合持续换手的问题。把143稳定住很有价值，却仍不足以完成当前合同。

**后续研究解决了不少真实的局部问题，但没有证明这些改进足以补偿强 Writer 的闭环能力损失。** 特别是，原生坐标可达性、动态表示质量、task-local functional recovery、attention 分配和共享 Writer 的闭环能力，是不同层次的证据。项目反复在前几层取得进展，然后把这些进展当成继续替换整图的理由；最后一层却始终没有稳定积累。

我的首选方向是：**保留早期强图已经证明有价值的职责——任务落地、Action Expert 驱动的过程读取、整套策略的联合编译——用符合当前完整 horizon 合同的主干重新实现；保留 native bank 作为策略更新的物理坐标，但不再让视频理解从第一层就等同于38个独立 target 的因子读取。**

不过，**第一项实验不应是立即实现新整图并长跑，而应先用没有专属 task query、训练关系完全一致的部署图，比较单任务学习与少任务共同学习。** 当前最重要的证据缺口，正位于这里。

---

## 一、核验范围与证据边界

我在开始和结束审查时核验的远端 main 都是：

```text
00b2e77798c3af47d4efa5bab9d5e75041c9ed31
docs: seal v4 results and finalize repository handoff
```

没有发现比指定基准更新的 main 提交。当前 v4 已封存，没有 active design；以下建议不构成启动实验或恢复旧实现的授权。

还有一个影响历史解释的事实：可核验的无父初始提交 `1226236…` 日期是 **2026年7月17日**，与当前提交之间的 compare 返回1756个后续提交。初始 README 的目标包括“Writer 提供最低可用能力，再通过 task-local RL 精修”，而非要求 Writer 一次直接生成可靠专家。当前要求更强。这个变化不能解释所有后续停滞，但复盘时不能把今天的完成标准原封不动投射到最初立项。7月初可能存在的仓库外讨论，本次没有原始材料可以核验。

本报告的证据分为四类：

| 类别              | 本次实际覆盖                                                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **直接核验**        | 指定提交、关键历史原设计、v6 的实际语义/时序/compiler 代码、当前 capture/tokenizer/composer/materializer、shared/task-local 训练与评测代码、tracked GOMQ 物化结果 |
| **仓库记载，未复跑**    | 历史正式闭环、梯度审计、各阶段 functional 与容量结果                                                                                            |
| **附件提供的本地事实**   | v4 原始日志的参数量、exposure 汇总、接手者对 raw rows 的重新配对等；部分数字也被当前 tracked 文档记载                                                          |
| **本报告提出的推断或设计** | 双侧零因子的优化问题、首选主干、训练与判别实验；均不是已验证结论                                                                                            |

我没有逐条审计1756个提交，也没有读取服务器上 ignored 的全部 checkpoint 和逐行运行资产。因此，下文是**主要研究链的重建与关键接口审查**，不能冒充所有历史实验的完整复算。尤其是 Semantic Factor-Basis、variance-reduced estimator 及若干 guard 的全部原始配方，本次仍主要依据较完整历史账本；这些限制会约束我对其根因的判断，而不会被省略。附件本身也明确要求区分这些证据来源。

---

## 二、历史到底推进了什么，又在哪里失去了能力

以下缩写仅用于压缩表格：

```text
E = ac233fa0e94b40c525d75746ef2d8fdfb4dc0046
V = 3a6f801d08facb3e855ab24f84e0b53cb8802e88
L = 8553b613de7791df50e0f3ef85678fcaca1cac0c
C = 00b2e77798c3af47d4efa5bab9d5e75041c9ed31
```

### 2.1 主线复盘

| 主线与原问题                                                       | 实际干预和获得的证据                                                                                                                             | 裁决边界与转向是否充分                                                                              | 原始定位                                                                                                                                       |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **立项、source、专家上界**：能否由任务描述产生可用参数起点                           | 初始设想包含后续 RL；后来 source 达48/400，隔离 task-local rank16 oracle 达250/400                                                                     | 证明执行接口与训练资产存在明显能力空间；不证明视频到参数的共享映射                                                        | `1226236:README.md`；`C:docs/research_history.md` 开头。                                                                                       |
| **Action-Forecast v4→v5**：将逐帧动作预测组织成视频过程                     | v4 的 absolute-time Plan/Revision 出现 shuffled 更强；原始交叉移植把主效应定位到 forecast 之后的错误时间对齐                                                       | **这是有较强因果依据的结构替换**：应删除没有被识别的共享绝对时钟，不是泛泛增加 temporal 容量                                    | `e8920cc7:docs/action_forecast_writer_v5_decision.md`。                                                                                     |
| **v5.2/v6 强 Writer**：语义瓶颈、过程读取与 compiler 表达力                 | task-queried patch grounding；Semantic Core、视觉变化与 Action probe 结合；完整动态因子和全 slot 协调。出现132、143等真实闭环                                       | 已经证明这组职责能够共同产生有效行为。由于多处结构及训练配方同时变化，不能识别每个组件的独立贡献                                         | `V:docs/action_forecast_writer_v5_2_design.md`、`...v6_design.md`；`V:src/ember/writer/{video_program,temporal}.py`。                         |
| **几何、共享表示与估计方差**：能力为何不能共存                                    | Semantic Factor-Basis single127、union193；variance-reduced best126，held functional 改善但闭环下降；多个固定方向/字典方案更弱                                | 排除了若干几何约束和“主要是 flow Monte Carlo 噪声”的简单解释；没有排除所有共享训练方法                                    | `E:docs/research_history.md §3.1–3.2`。                                                                                                     |
| **prior、expert projection、tangent、guards**：保持旧成功并增加新成功       | v6-prior、Expert-Component、Tangent Tube 未稳定增加能力；若干成功点约束、投影保住 train 局部性质，held 仍换手                                                        | 说明局部约束与 held 闭环分布不等价。NCCL、OOM、未进入机制阶段的运行不能算科学否决                                          | `E:docs/research_history.md §3.3–3.4`。                                                                                                     |
| **Dynamic-K / Shared-Core**：降低同任务示范差异                        | K4 达130/139，同任务 BA 方差降低约9倍；更稳定的视频集合表示没有自动改变 task mean                                                                                  | 真正解决了部分输入条件方差，未解决训练后任务能力交换                                                               | `E:docs/research_history.md §3.5`。                                                                                                         |
| **LPCP**：让分层 Action Expert 证据进入强图                            | 在冻结 AS139 上增加层/rank 条件化读取；143/400、breadth7；120 retained、23 gained、19 lost                                                              | 证明分层原生读取有正价值；否定的是“只修冻结 compiler 的 query，并继续旧 functional credit，就能稳定积累”                   | `L:docs/action_forecast_writer_v6_layerwise_probe_conditioned_procedure_design.md §10.1`。                                                  |
| **MCTC、SEOD、GOMQ**：稀疏 reward、成功 occupancy 和真实 query 梯度       | MCTC142→142→136；SEOD129→135→143→136；GOMQ151→135→131，相邻 churn42、34                                                                      | 成功 expert occupancy 与跨视频一致性均不足。GOMQ 修复了 memory query 梯度被缓存切断这一真实实现问题，但没有解决相邻稳定性          | `L:...successful_expert_occupancy_distillation_design.md`、`...gradient_open_memory_query_design.md`；`E:docs/research_history.md §3.6–3.7`。 |
| **LMMPC、functional bridge、旧 ECP Stage1 v1–v24**：更结构化地连接表示与策略 | layer/token mixing、K-set、M2P、latent/effect/code→factor 等多种映射；局部预测或表示改善，整图闭环长期较弱                                                        | 这些路线帮助定位接口，却没有证明整体替换强图的代价值得。LMMPC 也构成“增加跨层/slot attention 就会解决”的反证                       | `E:docs/research_history.md §3.8–3.9`；`C:docs/research_history.md` 早期 ECP 段。                                                               |
| **Native-Factor G1**：native 输出坐标是否有容量                        | 发现 scalar q-output、action-in span 限制；修正 native grouping 后，privileged 解析初始化 step0 达114/250、breadth5                                     | **解决了具体的可达性问题**。这是强接口证据，但不是 shared attention 训练成功                                        | `C:docs/research_history.md` G1 段，关键提交 `31f0053`。                                                                                          |
| **G2**：原生响应能否形成动态表示                                          | 修复静态旁路、owner、梯度与训练 cadence，完整表示相对 endpoints 的 held action+progress loss 改善22.2047%                                                     | 证明原生响应中有可学习动态信息；没有证明该表示与某个 compiler 组合必然有闭环收益                                            | `C:docs/research_history.md` G2 段，关键提交 `c1493a1`。                                                                                          |
| **G3 / Program–Bank / PNBTT**：如何把过程变成功能方向                    | 真实 bank、dual/primal、candidate interaction、summary/gate、whitening/transport 等连续干预。PNBTT 提高 specificity，却未同时恢复两个 task 的 correct capacity | 有些坐标诊断成立；但越来越长的局部接口链并未得到整图共享学习资格。PNBTT 的 E1 free-query 失败不能被写成 frozen G2 Program 的 E2 失败 | `C:docs/research_history.md` G3、§74–77、§98–103。                                                                                            |
| **Process–Composer→Unified v4**：减少专用坐标转换，直接读证据               | 逐步修复完整 horizon、relation consumer、frame ownership、X/Y side、softmax 分组；近期 held5 相邻结果仍未稳定超过 carrier43                                     | 局部修复有意义；但 v4 只有短资格剂量，既不能证明函数类不可学习，也不能凭局部修复继续投入长跑                                         | `C:findings.md`；`C:docs/unified_policy_native_factor_writer_design.md`。                                                                    |

这条历史最重要的结构是：

> **早期先获得了一条真正能工作的端到端映射；随后为理解和稳定它，逐步把问题分解成更清楚的局部接口。局部接口的科学质量提高了，但成功行为没有被可靠地带入每次重构。**

这不等于应该停止机制研究。它意味着机制研究需要承担一个更明确的责任：**新接口应证明它保留了什么行为能力，而不仅说明它解决了什么内部缺陷。**

### 2.2 两项必须更正的历史解释

**第一，GOMQ 的151→136不是已经证明的有效 rank 容量损失。**

原始物化合同是：

$$
A_{32}=\begin{bmatrix}A_0\\A_0\end{bmatrix},
\qquad
B_{32}=\begin{bmatrix}B_0&\Delta B\end{bmatrix}.
$$

所以在实数代数中：

$$
B_{32}A_{32}=(B_0+\Delta B)A_0,
$$

有效 rank 本来就不超过16。tracked 结果确实记录了重新物化后136/400，以及123 retained、13 gained、28 lost，但这不能证明“151依赖更高有效 rank”。原始 archival card 与结果 JSON 的证据，比后来的专家摘要更有权威。也没有理由据此重跑 dtype、rank 或 seed 去挽救151。

**第二，PNBTT 没有完成其真实 Program 进入后的共享裁决。**

最后的 gate-aligned E1 已经把 necessity hinge 从0.10对齐到0.50，仍在 correct/held capacity 上稳定 non-pass；其余分支没有进入 E2。因此，它排除的是实际测过的 free-query real-bank transport 参数化，不是“G2 已通过的动态表示接入后仍然无用”。把这个区别省略，会错误地为下一次整体换架构提供依据。

---

## 三、是否当时真的只差“漂移”

### 3.1 不同版本的长处不能拼成一个虚构的强基线

五臂依次是 correct / same-task-other / wrong / shuffled / reversed，每臂400：

| 版本                    |                        五臂结果 | 当前能够成立的判断                                                |
| --------------------- | --------------------------: | -------------------------------------------------------- |
| v5.2 old              |    132 / 138 / 74 / 82 / 83 | 内容与顺序的行为差异较强；absolute 距146仍有14个成功的缺口                     |
| v5.2 task-complete    | 120 / 109 / 107 / 111 / 124 | 更改训练配方后，absolute 与视频优势同时退化                               |
| v6 old                |   121 / 122 / 111 / 84 / 47 | 顺序确实影响闭环，但 correct 对 wrong 的差距有限                         |
| v6-fast task-complete | 143 / 135 / 125 / 128 / 129 | absolute 最接近目标，视频条件差距比 v5.2 old 小；后续131/130/132/126未稳定保持 |

这些数字来自较完整历史账本，与附件一致。不能把 v5.2 old 的视频差距、v6-fast 的143，以及 LPCP 的 breadth7，组合成一个实际从未存在的 checkpoint。 

LPCP 原报告还明确记录：历史 v6-fast 的 breadth 是6，LPCP 才提高到7；两者虽同为143，teacher schedule 不同，只能做 count-only 对比。LPCP 自身四 suite 为5/83/38/17，Goal3仍为0，新增解锁的 Long2 只有1次成功。**早期不是没有 Goal/Long 能力，而是能力贡献不均、任务覆盖仍有明显缺口。**

因此，对 owner 当时判断的准确回答是：

> **“能力漂移已经是主要障碍之一”有很强支持；“除漂移之外，方法已经满足完成条件”没有得到支持。**

### 3.2 必须拆开三类变化

**同一 checkpoint 的条件敏感性。**
换同任务视频、K、policy RNG 或初始化，成功行为会变化。Dynamic Slot-Set、Shared-Core 对其中的视频条件方差确实有改善证据。

**训练更新后的成功集合交换。**
即便同任务不同视频产生的更新高度一致，下一次共享参数更新仍可能丢失旧成功。SEOD、GOMQ 的结果正说明两者可以同时存在。

**重构、训练配方和评测合同变化。**
v5.2 old→task-complete、不同 K、不同 teacher schedule、rank 物化方式，以及 validation8 与 held5 panel，都属于这一类。它们不能画成同一条“模型训练越久越差”的曲线。

这三类问题需要不同干预：多视频集合读取主要针对第一类；共享目标、行为回放与占据分布针对第二类；严格 matched 重构针对第三类。

### 3.3 union193说明什么

它说明不同 checkpoint 的成功集合具有互补性，单点能力并未达到“所有历史成功的并集”。这对“只是完全学不会这些任务”的解释构成反证。

但它不证明存在一个同容量、同输入限制的共享 Writer，能无损实现这个并集。不同成功策略可能在参数空间不相容，成功也可能依赖不同的视频或状态敏感性；oracle 选择 checkpoint 不能部署。**union 是诊断能力交换的证据，不是可直接提取的额外66个成功。**

### 3.4 为什么高 cosine 没有保住行为

LPCP 相对 AS139 的 effective-BA cosine 均值约0.99999479，仍出现23 gained、19 lost；Goal3 的同任务修正一致性较高，也仍为0。由此已经可以否定“参数改动很小、方向很一致，所以原有成功大体会保留”的替代判据。

其原因在逻辑上并不矛盾：参数距离描述的是整个更新矩阵；闭环成功取决于策略在一串实际访问状态上的动作影响。少量关键状态上的小变化，可能改变后续访问的状态和最终结果。应测量后者，而不是继续提高前者的 cosine。

---

## 四、当前最可能的瓶颈，以及哪些解释还不成立

| 判断                                                | 可信度       | 支持证据                                                                    | 必须保留的反证或限制                                                              |
| ------------------------------------------------- | --------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **共享训练产生的有用方向，不能稳定覆盖并保持多任务闭环行为**                  | 高         | v6 后续回落、LPCP/SEOD/GOMQ 的 paired churn；多个 train guard 无法外推               | 尚不能全部归因于“梯度冲突”；实际 Adam、状态分布、读出参数化也参与                                    |
| **离线 functional 改善不是闭环改善的充分代理**                   | 高         | variance-reduced、多个 shared 阶段出现 proxy 改善而闭环不升                           | task72 的 task-local functional 改善确实带来34→40闭环，故该信号并非无效                   |
| **近期共享图的学习问题尚未被同条件正控定位**                          | 高         | 当前 task-local 增加专属 query、冻结 evidence，shared 则相反；v4每 task只有8–9 exposures | 不能把“训练很短”当成盲目续训理由                                                       |
| **在给定 language 与执行观测后，现有数据未必充分识别视频过程的必要性**        | 中高        | cross-episode 阻断轨迹复制，却保留 task/language shortcut；元数据审计未证明功能歧义            | 历史正确视频与 controls 的闭环差异说明视频确实曾有用                                         |
| **过早 target 分解、缺少共同过程工作表示限制了主干**                  | 中         | v4 每 target 只更新自己的 latent，原生响应读取局限于本 target                             | 共享权重、共同 prefix/velocity、整体 functional 梯度已提供间接协调；v6本来就有全 slot attention  |
| **双侧动态零因子使近零区域功能学习迟缓**                            | 待验证       | 由当前 signed readout 可推导局部二阶性质                                            | 真实视频不一定处在近零区域；cap、bank 分布、初始化都可能更重要                                     |
| **主要问题是缺少 Action Meta、rank4太小、attention mass 太低** | 当前不支持作为主因 | 局部存在这些可能性                                                               | Action Meta 只有特定中性结果；G1有rank4容量；成功 task-local 也可近rank1；v4拆softmax没有稳定净增 |

相关训练、局部闭环和数据审计依据见当前 findings、代码及附件。  

### 4.1 当前代码核验的关键结果

当前 `capture.py` 的确使用 state-free prefix、两个固定 antithetic probes、`flow_time=1`，捕获19个 boundary、完整50 horizon、velocity 与 native X/Y。prefix 上的 `mean(1)` 只去掉两个 probe 下重复的 prefix 数值副本，不是在平均视频帧。

当前 `process.py` 实际是 tokenizer，没有独立 process/event 主干；even/odd 组合保留两个 probe 的信息。它们只是响应的重参数化，不能自动解释成已校准的动作敏感性或视频运动。

`composer.py` 确实按 target 循环，target 内进行 teacher-time 与 rank/side attention；不存在直接跨 target latent attention。但这只说明**少了一种直接信息交换路径**，不能推出整个系统不能协调。旧 v6 的 `PostFusionSlotBlock` 已有全 slot attention，其不稳定结果也证明“加跨 target attention”不是充分答案。

shared 训练没有重演 SEOD 的 memory-input 断梯度问题：它先得到真实冻结 policy 对 LoRA leaves 的梯度，再重新执行 Writer，通过 chain-rule surrogate 回传 whole Writer。task-local 则缓存冻结 evidence，并使用额外 task query。两者不是同条件学习能力比较。

另一个应记录的口径是：

$$
L_i^{\mathrm{train}}=\frac{L_i}{c_i},
\qquad
c_i=\sqrt{\operatorname{mean}_{v\in \mathrm{PanelA}}
L_{\mathrm{carrier},i,v}^{\,2}}.
$$

因此，近似等频 task sampling 不等于各任务具有相同的有效功能梯度权重。这不是新发现的既定性能根因，但后续分析梯度支配时，不能只看9 meta + 3 target 的次数。

### 4.2 一个值得测、但不能先宣布为根因的问题

当前两侧都采用：

$$
q^\pm=b(C)+\delta^\pm(D),
$$

并由两个 softmax 的差分别形成 \(A\) 和 \(B\)。在固定 bank/context、接近 \(D=0\) 的区域：

$$
A(D)=O(D),\qquad B(D)=O(D),
$$

所以：

$$
\Delta W(D)=B(D)A(D)=O(D^2).
$$

这意味着“静态零 mobile”不仅约束了最终行为，也同时关闭了低秩两侧；在零点，过程表示通往功能更新的一阶通道消失。该推导来自实际 readout 结构，但**尚未证明 v4 的训练主要停在这个区域**。

它与历史 G3 的双侧子空间学习迟缓相呼应，也与早期 v6 非零 A、零 B 的起步方式形成有价值对照。因此它值得一个 matched 实验，而不值得直接成为新的总体解释。

---

## 五、首选整体方案

### 5.1 方向选择

我选择的是：

> **以强 Writer 已验证的职责为基准，构建“共同过程表示—整策略 queries”反复交互的原生响应 Writer；首轮保留12+4输出接口来隔离主干与学习问题。**

这不等于恢复旧 v6。旧实现确实有：

```python
interaction_projection(suffix_hidden.mean(dim=1))
```

它违反当前完整 horizon 读取要求，不能恢复为 active 路径。应继承的是 task-queried patch grounding、过程与语义的分工、全策略联合编译职责，而不是该平均操作或整个旧 checkpoint。

也不应声称“新图已经保留旧能力”。**职责继承只是有证据的设计选择；行为继承仍必须用同合同闭环来证明。**

### 5.2 完整数据流

```text
部署输入
exact language + K条正确、action-hidden、内部有序视频
                         │
                         ▼
冻结 source：逐帧原生响应捕获
prefix / H[probe, layer, horizon] / layer residual / velocity / raw X,Y
                         │
              每条视频独立处理
                         ▼
任务落地 + 原生过程工作状态 P
                         ⇅
整策略结构 queries Q：38 targets × rank4 × X/Y side
  重复标准 attention/MLP blocks：
  读完整原生响应 → 沿teacher time交互
  → P与Q双向交互 → Q跨target协调 → 再读当前native bank
                         │
                         ▼
跨视频置换不变的 learned set read
不拼接为物理时间，不平均raw features或LoRA
                         │
                         ▼
当前视频raw X/Y上的signed factor readout
                         │
                         ▼
完整target更新cap + 明确rank/scale物化
frozen carrier12 + mobile4 → 唯一38-target rank16 LoRA
                         │
                         ▼
执行source + 当前机器人观测/state
rollout期间不再调用Writer
```

训练只在这张部署图之外增加授权的 action query、成功行为监督和 loss。所有 learned 主干共同训练，不要求先冻结一个“完美 process”再训练 compiler。

### 5.3 原生证据：读取什么，不赋予它什么含义

| 证据                       | shape/轴                                   | 用途                        | 不能当成什么        |
| ------------------------ | ----------------------------------------- | ------------------------- | ------------- |
| 图像 patch、language prefix | \([T,256,2048]\)、\([T,L,2048]\)           | 任务对象、关系与目标落地；形成读取 query   | 已经理解整段过程      |
| layer states             | \([T,2,19,50,1024]\)                      | 当前视觉条件下，各层对未来动作位置的原生响应结构  | 教师真实未来动作      |
| layer residual           | \([T,2,18,50,1024]\)                      | 同一次前向中的层间计算变化             | 视频帧间运动        |
| flow velocity、probe      | \([T,2,50,32]\)、\([2,50,32]\)             | 明确响应是在什么噪声条件下产生；提供动作生成端信息 | 完整去噪动作、正确动作真值 |
| native \(X_j,Y_j\)       | 每 target 保留 \(T,p,h,\text{native width}\) | 最终因子的物理 value 来源          | 已经具备正确更新方向    |

π0.5 的 action expert 建模的是条件动作生成的 flow field；一次噪声端点前向不是完成采样后的动作轨迹。这里的机制解释同时得到原论文和当前 capture 代码支持。([arXiv][1])

必须保持五条轴的区别：

$$
t_{\text{teacher}},\quad h_{\text{action}},\quad s_{\text{flow}},
\quad \ell_{\text{layer}},\quad p_{\text{probe}}.
$$

新主干可以学习“某段视觉变化对应哪些 horizon、哪些 layer 的响应变化”，但没有 \(t+h\) 这样的预设机器人时钟，也不把网络深度解释成任务阶段。

### 5.4 两类 learned 工作状态，而非固定 Program schema

**过程状态 \(P\)。**
每条视频保留：

$$
P^{(k)}\in\mathbb R^{T_k\times M\times d}.
$$

首个 matched 实现可沿用 \(d=128\)，每帧放少量工作 tokens；例如 \(M=8\) 只是每帧工作容量，**不是规定一个任务必须有8个事件或阶段**。

这些 tokens 先通过 exact-language 条件化的 patch attention 形成任务落地，再读取同帧完整的 layer/probe/horizon 响应。第一次 horizon 压缩发生在这里，且已经是 task-conditioned learned read。随后沿真实 teacher time 交互，保留每帧工作状态，而非立即变成一个全视频向量。

**策略 queries \(Q\)。**

$$
Q^{(k)}\in\mathbb R^{38\times4\times2\times d}.
$$

它们的初始参数只由 owner、rank、X/Y side 等共享结构身份构成；没有 task ID，也没有每任务可训练 query。不同视频产生不同 \(Q\) 状态，不等于不同视频拥有专属参数。

重复 block 中，\(Q\) 读取 \(P\)，在整套 target/rank/side 之间交流；\(P\) 再读取这些当前策略问题，决定下一层需要从完整原生响应中读什么。native bank 的 ownership 仍严格保留，但**语义与过程推理不从第一步就被切成38个独立子问题**。

这与旧 Process–Composer 的区别不是“多加 attention”，而是：

* 没有冻结、一次性输出后再解释的 process 坐标边界；
* policy queries 可以反过来影响后续过程读取；
* 原生响应和 native bank 可在后续 block 重新访问；
* 所有模块接受同一个真实功能目标的联合梯度。

这仍然是待验证的架构假设；旧 v6/LMMPC 的失败要求我们对“跨 target 交互本身足够”保持明确怀疑。

### 5.5 跨视频规则

每条视频先独立完成上述过程与策略状态计算。集合阶段用共同 queries 对所有视频的 **learned states** 做 cross-attention，最终对当前所有视频的 raw native candidates 做一次联合 signed readout。

候选基础质量按视频归一，避免长视频仅凭 token 数量占优势。视频序号不作为语义 embedding；视频内部位置重新起算，不能把视频1末尾与视频2开头建立物理相邻关系。

首个诊断只声称 K1。后续要声称 K1/K2/K4，就必须从现有授权 demonstrations 中提供至少4条不同 fit 视频，并实际训练这些 cardinalities；不能用两条 fit 视频重复凑出“K4”。

### 5.6 因子读出：保留 native support，修改零更新的实现方式

首选采用**非对称动态因子化**：

$$
A_j=A^{\mathrm{context}}_j+\Delta A^{\mathrm{process}}_j,
\qquad
B_j=\Delta B^{\mathrm{process}}_j.
$$

所有向量仍来自当前视频的 raw X/Y signed pooling；\(A^{\mathrm{context}}\) 不是自由参数字典，也不是固定 global-A。

具体实现上，A 侧允许由共同 context 产生不相同的正负基础 query；B 侧仍保留共同基础 query，并由过程 innovation 产生两支差异。因此静态 innovation 为零时：

$$
B_j=0,\qquad \Delta W_j=B_jA_j=0,
$$

但一般不要求 \(A_j=0\)。在非退化 \(A^{\mathrm{context}}\) 附近：

$$
\Delta W_j
\approx \Delta B^{\mathrm{process}}_jA^{\mathrm{context}}_j,
$$

存在一阶功能学习路径。

它保留了“静态本身不能打开 mobile”的约束，却不同时关闭低秩两侧。必须测量的是它是否改善真实 functional 学习及闭环，而不只是梯度变大。

q 的 native head grouping、action-in 的 native-width output grouping 首轮保留。G1 对这两项有明确容量依据；没有理由在测试新主干时同时丢弃。

### 5.7 唯一 LoRA 与梯度路径

首轮保持：

$$
A_j^{16}=
\begin{bmatrix}A_j^{12}\\A_j^4\end{bmatrix},
\qquad
B_j^{16}=
\begin{bmatrix}B_j^{12}&B_j^4\end{bmatrix}.
$$

所有因子按最终 rank16 contract 的 scale 解释，不能在拼接后再分别叠加 rank4/rank12 的另一套 \(\alpha/r\)。执行器只挂载这76个 tensor。

cap 作用于完整的 \(B_jA_j\)，不以两个因子各自范数代替行为约束。沿用现有一次 small-core canonicalization 时，需要明确记录训练 raw factor 路径与正式物化路径的功能差异；当前训练使用 `canonicalize=False`，Panel-B 使用 `True`，这是直接可核验的区别，**不是本次已经证明的性能根因**。不应重演把物化变化解释成有效 rank 容量变化的错误。

梯度路径为：

```text
授权action/成功行为目标
→ 装载唯一LoRA的真实policy functional loss
→ 全部38个target的LoRA cotangent
→ signed pooling
→ whole-policy queries与过程状态
→ evidence projections
```

source 权重、carrier、原始 capture 不更新。可以继续使用当前分离 policy VJP 与 Writer 重算的链式法则实现，避免两张大图同时驻留；但不能把 learned observer 或 learned process 的状态缓存成永久 detached 输入。

### 5.8 唯一条件备选：释放完整rank16，而非继续堆解码器

只有当**同一新主干、同一数据和功能目标下**，12+4 的无专属 query 学习明确受到输出可达性限制，而完整 rank16 对应控制有稳定功能及行为改善，才考虑让全部16个 rank 成为 task-conditioned。

此时应直接输出一套完整 rank16 A/B，把共享 prior 放在这16个 rank 的初始化或共享参数内，不能部署 `carrier12 + mobile16`。

历史 PNBTT 的 full-rank16 oracle 没有给出跨 task 一致优势，所以它不构成现在扩 rank 的授权；同样，它也没有否定不同主干与读出下的完整16-rank 候选。

---

## 六、共享训练应如何给出“正确且可保留”的方向

### 6.1 第一阶段：先证明正确视频能教会真实共享图

第一阶段保留当前最直接的监督：**正确视频 → 唯一 LoRA → 同任务独立 episode 的真实 action flow loss**。

方向的正确性来自授权任务的真实 actions，而不是来自 source response。source response 只提供读取与表达更新的坐标和归纳偏置；source 即使做错任务，也不因此成为监督真值。

采样应明确为 task、episode、action query 的分层采样。每个 task 的视频与 action episode 必须隔离，video/visit 由 per-task occurrence 驱动。当前代码已经修复过 global-step 周期别名，不能再把这项旧问题重新当成未尝试的新优化。

首轮与现有参考保持同一 allowlist、同一 loss reduction、同一 normalizer，先避免把架构改变与任务重新加权混在一起。完整训练计划则应明确区分：

$$
\text{采样概率}\times
\text{显式task权重}\times
\text{loss normalizer}^{-1},
$$

并报告实际作用于共享参数的权重，而不是只写“task balanced”。

### 6.2 第二阶段：在行为空间保留成功，不统一压小 LoRA

获得真实共享学习信号后，才加入成功行为保持。

我建议保持对象为：

> **授权非 held 任务中，source 或先前共享 Writer 已经实际闭环成功的行为片段。**

source 未成功的行为不能获得“因为来自 source 所以应保持”的特权；旧 Writer 的失败动作也不能被当成正目标。保留监督应作用于成功轨迹覆盖的真实执行状态与动作函数，不作用于 LoRA cosine、权重重建或全局 update norm。

训练上，成功回放与改进样本共同进入 task-balanced 目标。回放数据属于训练资产，不属于部署输入；可以包含早期、中间和后期任务阶段，避免只守住某几个初始成功点。这里不要求所有旧成功零损失，也不使用“所有 task/view 同时下降才能提交”的硬门——历史已经展示这种要求可能把共享更新逼近 no-op。

**这是一项有条件重试，不应包装为全新思想。** success guards、distillation、SEOD 已经尝试过相关方向。新的设计必须用同一部署图检验：改善来自更有用的行为覆盖与共享表示，还是仅仅把更新压小、重新保住弱 baseline。

### 6.3 何时才引入 learner occupancy 或 reward

仅在出现如下分离时引入：

```text
正确视频下，训练与新视频的离线功能学习成立
但相同模型在真实闭环中持续出现明显状态分布外失败
```

这时有价值的变量是**监督状态分布**，而不是换一个 loss 名字。

SEOD 已经使用成功 expert 的 occupancy，且用 matched noise 查询 expert/student 动作；不能再建议“加入成功 expert occupancy”并把它当成尚未尝试的关键突破。它依然主要覆盖 expert 能访问的状态，未必覆盖当前 Writer 犯错后进入的状态。

后续允许的做法是：在现有授权任务和场景内，从当前 Writer 的训练 rollout 收集实际访问状态，仅在 privileged expert 或成功 continuation 提供可信目标时给予纠正信号。失败且没有可信目标的状态不能自动变成正监督。训练期 reward 用于判断行为质量，不能成为 deployment Writer 输入。

这与模仿学习中的占据分布问题一致，但相关理论并不能直接保证 EMBER 的非凸共享 Writer 稳定；它只说明为什么 expert/offline 状态分布上的拟合不足以保证 learner 闭环性能。([Proceedings of Machine Learning Research][2])

我没有完成所有历史 online-credit 运行的 state manifest 复核，因此**不声称 EMBER 从未做过等价 learner-occupancy 尝试**。若旧运行已覆盖相同变量，新方案必须承认是在重试，并给出新的结构或数据条件，否则不值得重复。

### 6.4 正监督能否保证模型使用视频过程

不能保证。

设授权数据中，在给定 exact language 和执行观测后，所有正确视频都对应近似相同的最优策略，那么只最小化：

$$
\mathbb E_{V,a\mid L}\,
\ell\bigl(\pi_{\theta+\Delta(V,L)},a\bigr)
$$

允许模型弱化视频过程，主要使用 language 或静态场景。cross-episode 只排除了复制同一条轨迹，不能排除这个解。

把 raw bank 设为唯一 value、把静态输入设为零 mobile，也不能从逻辑上消除 shortcut：模型仍可能用 task/language 控制一个几乎固定的动态读取规则。

当前授权数据内最有信息量的分析，是找到并核验：

> 在控制语言、执行条件及容易利用的静态差异后，不同正确过程是否真的要求不同功能行为。

元数据里存在“同语言不同场景”还不够；执行 policy 自己的观测也可能消除歧义。当前73-task audit 尚未完成这种功能层面的证明，且 held5 Goal push procedure 的梯度覆盖仍有缺口。

因此，本方案能提供更合理的过程学习路径，但不能预先承诺强视频因果优势。最终优势必须由冻结后的完整 controls 裁决，不能用 wrong/shuffle/reverse 训练制造。

---

## 七、与历史逐项对照：这次究竟改变什么

| 本次改变                                | 最接近的旧尝试                                                | 旧证据实际排除了什么                                            | 本次真实变量与可证伪预测                                                                       |
| ----------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 任务落地后再压缩原生响应                        | v5.2 patch grounding；v6 Semantic/Procedure             | task-token-only 有语义瓶颈；旧 horizon mean 不能恢复             | 保留 task-queried patch 职责，但对完整 horizon 做条件读取。若新图在无专属 query 的小任务拟合就弱于参考，不能用“表示更正确”辩护 |
| 过程状态与整策略 queries 反复交互               | v6 全 slot compiler；LMMPC axial mixing；Process–Composer | 多加全局 attention 不足；冻结 learned-coordinate handoff 也可能失败 | 改变的是后续读取受策略问题反馈影响、raw evidence 可重访。若只有 cross-target attention 增益而无学习/行为改善，应删除该复杂度  |
| native bank 作为物理 value              | G1、G3、PNBTT                                            | G1证明部分可达性；PNBTT特定 transport 丢失 capacity               | 保留 grouping、raw X/Y，删除 covariance/whitening/transport 链；仍需证明受限 readout 的共享可学性      |
| A-context + B-dynamic               | 当前双侧 signed innovation；早期非零A/零B                        | 当前结构有近零二阶效应的可能；旧固定A又确有可达性问题                           | A是当前视频可学习的 bank-conditioned A，不是固定全局A。预测应是同预算下真实 functional 学习改善，不只是梯度放大           |
| 无专属 query 的局部/共享对照                  | 当前 task-local 正控                                       | 专属 query + 冻结 evidence 的局部恢复不能证明 shared graph         | 只改变是否共享参数。局部成功、共同失败才支持共享共存障碍；两者皆失败则应回到输入/读出/优化                                     |
| 按 exposure 比较 component-init/random | G2 cadence；v4短资格                                       | 少量 update 不足以否定函数类；投影初始化不等于继承G2动态机制                   | 同拓扑、同数据、同曝光量联合训练。若random最终追平，init是加速；若只有component成立，需定位真正继承了什么                     |
| 成功行为回放、条件化 occupancy                | guards、SEOD、GOMQ                                       | 局部成功点、expert occupancy、跨视频一致性都不保证 held 保留             | 以真实共享图的成功行为和访问状态裁决。若只降低更新量、没有新增行为覆盖，则没有解决原问题                                       |
| 首轮保留12+4                            | G1 rank4、PNBTT full16                                  | rank4非零容量成立；12+4未被证明全局最优，full16也没有自动解决PNBTT           | 暂时固定输出接口以定位主干。只有 matched 完整16-rank 控制给出广泛功能/行为优势，才释放全部rank                         |

这些对照的原件依据分别是 v5.2/v6 实现、LPCP/SEOD/GOMQ 原设计，以及当前 G1–PNBTT 和 Unified 记录；其中原始方差缩减及部分 guard 的细节仍属于未完全复核项。

---

## 八、最小实验：先让竞争解释产生不同预测

### 实验一：同图、无 task query 的“分别学”与“共同学”

**这是我建议首先授权的实验。**

选择少量授权 gradient tasks，包含已有局部参考的 task1、task72、task93，再加入一项有现成成功专家资产的 Goal 任务。task93 的闭环上界较弱，不能单独承担 functional→behavior 裁决；task72有更明确的34→35→40局部参照。这个小 panel 用于定位机制，不冒充任务泛化证据。

两组模型：

| 对照   | 参数关系               | 其他条件                                                |
| ---- | ------------------ | --------------------------------------------------- |
| 分别学习 | 每任务一个独立 clone，仅作诊断 | 与 shared 完全相同拓扑、初始化、可训练模块；**没有 task query**         |
| 共同学习 | 所有任务使用一个共享 Writer  | 相同视频、action rows、per-task exposure、optimizer规则与输出合同 |

两组都训练 whole Writer，不能继续让局部组冻结 evidence、共享组训练 evidence。共享组每次逻辑更新覆盖全部小集合任务；不同 GPU 数只改变物理执行，不改变任务均值。

建议以**每任务8、32、64次 exposure**读取曲线。这里64是首轮预算，不是“64次未通过便否定函数类”的科学阈值。四任务、两组、每任务64次，总计512次 task exposures、按8 rows约4096次 action-row采样；与附件所述 v4 的600 exposures/4800 rows 同一量级，而且更集中地回答共享学习问题。缓存、评测和不同视频长度仍需单独计费。

**结果分支必须预先写清：**

| 观察                      | 最有支持的解释                              | 下一步                                |
| ----------------------- | ------------------------------------ | ---------------------------------- |
| 无 query 的单任务 clone 也学不动 | 输入/读出/优化或监督本身不足；旧 task-local 正控掩盖了困难 | 先测非对称因子读出，不能直接怪共享泛化                |
| clone 学会，共享图明显学不会       | 真实共享训练与共存问题                          | 优先测试共同过程—整策略主干；检查实际共享更新对各 task 的影响 |
| fit 视频学会，新同任务视频失败       | 条件过拟合、示范 nuisance、视频覆盖不足             | 改正确视频采样与过程读取，不急着增加 task 数          |
| 同任务新视频成立，任务留出失败         | 共享规则迁移或数据覆盖问题                        | 检查组合/程序覆盖，不能靠更多同任务步数解释             |
| functional 成立，局部闭环不成立   | 状态分布或监督影响与行为不匹配                      | 才进入 occupancy/行为目标分支               |
| 原 v4 同图共享已形成可保留闭环增量     | 训练剂量/正控错配比架构缺陷更重要                    | 不应因为已提出新架构就强行替换它                   |

这项实验比“再做一个带专属 query 的容量正控”更有价值，也比直接上73任务长跑更容易改变路线判断。

### 实验二：按结果测试主案，而非同时改变所有因素

若实验一显示输出端学习迟缓，先只改变 A/B 的动态零结构，保持其他输入、主干、rank、cap、数据和 loss 不变。比较有限步 functional 改善、新视频迁移与局部闭环，而非以 Jacobian 或梯度范数通过作结。

若局部学习成立而共同学习失败，再比较现有 target-local 主干与提出的共同过程—整策略主干。两者使用相同读出、同一小任务集合和 exposure。增加的参数、训练时间必须单列；不能把更大预算产生的提升归因于跨 target 机制。

随后在同拓扑下比较 component-init 与 fully-random。允许端到端联合训练，不要求随机图先通过冻结中间模块的全部 Gate。

### 实验三：扩任务前先分开三种泛化

按现有 allowlist 扩展时，分别记录：

```text
训练task + 新action episode
训练task + 新teacher video
留出task + 新teacher video/action episode
```

重复使用过的 task6/79 可以继续作为诊断，但不能称为 fresh selector。若加入更多现有非 held tasks，要把“覆盖了新的程序/组合”与“同类任务数增加”分开报告。

K1 共同学习有行为证据后，再引入真实 K1/K2/K4 训练；same-task 新视频鲁棒性应作为正视频泛化指标持续记录。shuffled/reversed 不进入这一阶段。

### 实验四：正式能力积累与最终 controls

正式比较只能使用同一合同的单 checkpoint，锁定 task/state、teacher schedule、K、env seed、policy noise 共同前缀、rank和物化方式。

相邻点必须报告：

$$
\text{net}=gained-lost,
$$

$$
\text{churn}=\frac{gained+lost}{N},
$$

$$
\text{Jaccard}
=\frac{retained}{retained+gained+lost}.
$$

同时列出每 task、每 suite 和 breadth，不能只报告总分。不同 teacher schedule 的旧143只能做 count-only 参照，不能伪造 paired retained/gained/lost。

能力保留判据应包含：

* 相对有效参考具有可解释的净增，而非仅出现不同成功；
* 相邻点没有重演明显的 gained/lost 反转；
* 不能靠牺牲 Goal/Long 或把成功压回少数任务维持总分；
* 允许合理旧成功损失，但损失必须被真实新增能力和稳定性证明值得。

当前正式数值要求意味着 selected checkpoint 至少146/400。还要满足相邻稳定、低 churn、breadth 和 suite 要求。**现有材料没有提供一个已经冻结、可直接套用的“低 churn”统一数值阈值，我不会编造一个，也不恢复 LPCP 当年的 `lost≤10` 旧门。** 该阈值应在正式 selector 前登记，不能看过结果再决定。

选定并冻结 checkpoint 后，才运行完整 same-task、wrong、no-video/language-only、first+final、shuffled、reversed controls。controls 失败应如实限定最终结论，不能反过来进入新一轮训练或架构修改。当前8个 Test tasks仍保留到方法冻结。

---

## 九、资源与实现路径：1–6张A40

### 9.1 复用与替换

| 保留资产/实现                                        | 用法                                            |
| ---------------------------------------------- | --------------------------------------------- |
| 冻结 source、source normalization、allowlists      | 不改变数据权限和执行起点                                  |
| `capture.py`                                   | 继续完整19 boundary、50 horizon、双 probe、raw X/Y 捕获 |
| native grouping、signed pooling、物化合同            | 保留物理可达性和唯一 LoRA 接口                            |
| `functional.py` 与分离 VJP/Writer 重算              | 继续用真实 policy 功能梯度，控制显存                        |
| per-task occurrence schedule、shared mmap cache | 已修好的工程能力不重写                                   |
| 原始 paired evaluator 与 formal manifests         | 保持行为比较合同                                      |

主要替换的是 `process.py` 的纯 tokenizer 后续职责，以及 `composer.py` 的 target-local learned 主干；native bank 读取可以物理分块执行，但不应因此把 learned 状态再次切成38个独立计算问题。

### 9.2 真正的资源瓶颈

当前最重的部分不只是约300万 Writer 参数，而是 full native evidence、bank attention 和通过大 policy 的 functional VJP。现有 v4 最长视频 smoke 记录约36.91GiB reserved，说明单张A40已经接近需要认真管理的范围；这不是对新主干显存的保证。

按已经核验的 shape 做粗略计算，单帧仅 layer states 就约为：

$$
2\times19\times50\times1024\times2
\approx 3.7\ \text{MiB},
$$

尚未计 raw X/Y、prefix、投影、梯度与工作缓存。这只是 BF16 tensor 字节量估计，不能当成实测峰值。

新主干应让较小的 \(P,Q\) 常驻，原生 evidence/bank 分块重放。分块必须保持完整 horizon 和全候选归约，不能靠 horizon subsampling 或截断长视频降低成本。

### 9.3 单节点执行方式

**1张A40：** 小任务实验串行执行每 task 的 forward/VJP，累积完整逻辑任务均值后更新；clone 诊断也串行运行，共享只读 CPU/mmap evidence。

**2–4张A40：** 以 task 分配 policy 副本和 functional VJP，all-reduce 只聚合 Writer 梯度；按真实 frame 长度做负载均衡，但不改变 task 权重。

**5–6张A40：** 优先把额外卡用于冻结 checkpoint 的并行评测或独立初始化对照，不为了用满卡而改变科学 batch 语义。

profile 至少分开：capture、数据传输、Writer forward、policy VJP、Writer backward、物化和 rollout。先覆盖选定小集合中的最长视频，再在扩任务/K前覆盖完整授权集合的最长条件。当前没有新主干的实测吞吐，所以不应给出“可以在若干小时完成”或默认长跑的承诺。

---

## 十、Action Meta：有限的可选项，不是主案前提

**我的判断：首轮关闭，但不把“必须先有 base Writer 闭环收益”当成不可讨论的科学公理。**

旧 v6 的实现确实在 teacher 路径安装 text、VL 和 Action Meta-LoRA；所以旧强分数不能用于证明“Action Meta 无价值”。另一方面，Stage0 某一次 matched 中性结果，也只能约束当时的 observer、冻结方式和训练目标。当前 `stage0_meta_training.py` 显示它是对冻结 native observer 的独立校准路线，不能外推为所有 observer-side 适配都中性。

值得触发 observer-side Action Meta 的证据应是：

> 在已能学习的同一 Writer/读出下，state-free 教学输入产生的原生响应存在可定位的输入域不足，而有限共享 observer 适配能改善正确视频的功能学习或迁移。

它不要求先证明所有闭环问题已解决，但也不能只凭“没有 teacher state”就默认必须加入。

若启用，职责应严格限定为读取侧：

```text
冻结基础权重 + 共享observer Meta
→ 用于过程理解的H/response

未适配的执行source
→ 最终raw X/Y bank与执行坐标
```

两条坐标来源必须明确，不能把 observer 适配后的 bank 默认为执行 source 的 native bank。Meta 训练只使用授权 non-held 数据，冻结后重新生成对应 evidence cache；若保持联合训练，就不能继续使用失效的固定 response cache。

observer Meta 不进入执行 policy，因此不增加第二套执行 adapter。若将 Meta 放到执行端，它就必须计入唯一 rank16 的总预算，并重新做归因比较；首轮没有充分理由这样做。

LIBERO 结果也不能据此声称跨视角、跨人类/机器人示范泛化。那是尚无当前数据证据支持的范围。

---

## 十一、真正影响决策的待补证据

不是再补一批泛泛的诊断，而是以下几类材料：

| 缺口                     | 最小材料                                                                                | 会怎样改变判断                        |
| ---------------------- | ----------------------------------------------------------------------------------- | ------------------------------ |
| **v5.2/v6强结果的完整配方可比性** | 对应 run contract、checkpoint selection、每 task exposure、K/teacher schedule、逐 task 正式结果 | 决定能否把配方变化与结构变化分开，哪些历史行为可作为严格参照 |
| **原始稳定化干预的等价性**        | SFB、variance-reduced、关键 success guard/online-credit 的配置、真实更新对象、状态采样 manifest        | 判断新训练建议究竟改变了因果变量，还是重复已失败的方法    |
| **当前无专属 query 的学习曲线**  | 同图 clone/shared、相同冻结关系和 exposure 的 fit/新视频/闭环结果                                     | 直接决定先修读出、共享表示、数据覆盖还是 occupancy |
| **功能层面的视频可辨识性**        | 授权任务 allowlist、语义/程序覆盖及条件化功能歧义审计                                                    | 决定当前数据能支持多强的视频过程必要性主张          |
| **物化与正式稳定性合同**         | raw→canonical 的同条件功能差异；正式低-churn定义                                                  | 避免把数值执行变化当容量变化，或事后定义稳定性        |

其中前两项尚未全部由本次原件核验完成。因此，我不声称已经排除所有旧训练路线，也不声称新方案与所有旧 online-credit 尝试都已证明不同。

---

## 最终建议

EMBER 不应再以“最新图存在一个解释得通的局部缺陷”为充分理由，启动下一次长程整体重构。也不应因为旧模型接近目标，就认为冻结旧 v6、加一点保留约束能够完成任务；LPCP、SEOD、GOMQ 已经提供了直接反证。

**最值得保留的是旧强图的工作职责，以及真实闭环能力作为研究主线的地位。最值得改变的是：让局部容量、共享学习和能力积累重新在同一部署图、同一数据合同下接受检验。**

因此，我建议 owner 首先只批准：

> **无专属 task query、whole-Writer 同条件训练下的少任务“分别学—共同学”对照，并配套新视频与局部闭环曲线。**

它若证明现有图能共同学习，应优先解决训练与行为积累，而不是为了新架构而换图；它若证明局部能学、共享不能学，才为上述“共同过程—整策略联合 Writer”提供真正的投入依据。

这比再获得一个局部 Gate pass，更有可能结束 EMBER 目前反复“解释得更清楚，却离完整行为证明更远”的循环。

[1]: https://arxiv.org/html/2504.16054v1 "https://arxiv.org/html/2504.16054v1"
[2]: https://proceedings.mlr.press/v15/ross11a.html "https://proceedings.mlr.press/v15/ross11a.html"
