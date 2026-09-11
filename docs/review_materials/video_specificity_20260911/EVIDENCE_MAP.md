# 从历史实验提取机制约束

本页是证据导航，不是一套已选定的理论。整理沿用原始记录与现存历史复核，没有重新运行全部实验或独立审计每个历史张量。
请结合[最新证据](LATEST_EVIDENCE.md)，主动检查下面的解释边界，并允许提出与历史作者不同的理论。

## 1. 普通监督的视频依赖：v5.2与v6

v5.2从逐帧图文中保留task-token轴并读取真实patch，形成Semantic Core；Action Expert probe进入有向Procedure，
Core条件化读取时间中心化Procedure后融合，由FactorHeads生成LoRA。Text/VL/Action三组观察Meta参与学习。
旧前端存在horizon均值压缩，不能仅因有正结果就在当前合同下恢复这一计算；其设计与训练仍是重要分析对象。

训练采用正常顺序正样本动作监督，没有contrast/order辅助目标。旧teacher与动作在同一50-episode池独立调度，
未实施当前互斥分池，不能把两代跨episode合同写成完全相同。

| 配方 | correct / other / wrong / shuffled / reversed，每臂400 |
| --- | --- |
| v5.2 old | 132 / 138 / 74 / 82 / 83 |
| v5.2 task-complete | 120 / 109 / 107 / 111 / 124 |
| v6 old | 121 / 122 / 111 / 84 / 47 |
| v6-fast task-complete | 143 / 135 / 125 / 128 / 129 |

v5.2两组具有相同Writer配置/模型源码，但训练同时改变task/update、queries/condition、更新数、LR时标和曝光：
4→24 task/update，21→20 queries/condition，900→400 updates，75600→192000 queries，3600→9600条件调用。
旧4task/update按轮转覆盖完整train24，不是只训练四个task。同架构差异不是某个训练因素的单变量证明。

v6又耦合改变语义读取、视觉transition与heads；v6-fast后续450/500/550/600为131/130/132/126。
这些事实支持普通监督可以学出正确视频偏好，也说明架构形式不保证这种偏好。不存在不依赖视频的解才可研究视频，并非项目要求；
反过来，“FM允许视频无关解”也无法单独解释正负两类结果。旧controls欠缺公平训练的static参照，不能直接推出普遍操作理解。

原件：[v5.2设计](https://github.com/LinFyM/EMBER/blob/529da6bbe290f7393422937aa7cc278cee732107/docs/action_forecast_writer_v5_2_design.md)、
[旧temporal实现](https://github.com/LinFyM/EMBER/blob/529da6bbe290f7393422937aa7cc278cee732107/src/ember/writer/temporal.py)、
[v6设计](https://github.com/LinFyM/EMBER/blob/3a6f801d/docs/action_forecast_writer_v6_design.md)。
完整rows/config/metrics用[9月7日索引](../20260907/index.json)中的`historical_v52`、`historical_v52_taskcomplete`、`historical_v6_fast`定位。
v6 old本页沿用历史记录，没有声称它包含于这三个导出组或本次重新复算。

## 2. 多视频、共享参数与稳定化

DynamicSlotSet/SharedCore将同task有效更新变化降低约9.26/9.69倍，分数约130/139；DirectFamilyB的K1/K4为102/98，
更多视频及更小参数变化没有保证更强行为。LPCP K4曾143/400、breadth7，但与另一个143的teacher日程不同。
一次LPCP配对比较的BA cosine接近.99999479，仍有23 gained、19 lost；参数稳定与成功集合稳定不同。

GOMQ使用强carrier及成功expert occupancy信用，cycle2/3/4为151/135/131，相邻churn42/34；cycle2五臂为
151/139/131/127/115，**它也有历史视频差额，不能说v5.2以后完全没有特异性**。但这些性质没有形成整体稳定资格。
原rank32拼接的实数代数有效rank本来≤16；按rank16重物化136不能解释为已证明的32→16有效容量损失。

原件：[分层历史](../../research_history.md)§2；[rank解释修正](https://github.com/LinFyM/EMBER/blob/ac233fa0/docs/gomq_rank16_archival_card_20260824.md)、
[重物化结果](https://github.com/LinFyM/EMBER/blob/3075b3c7/docs/evidence/gomq_20260824/gomq_cycle2_effective_rank16_strict400.json)。
9月7日索引的`historical_dynamic_slot/shared_core/lpcp/gomq/gomq_rank16`包含相应已导出记录。

## 3. 输出可达性和动态可读性：G1/G2

G1把成功更新投影到scalar signed-output span后，strict250由120降109，Goal/Long由11/8降0/0。
这是丢失方向有行为作用的直接干预。按原生q heads/action-in blocks分组、结合privileged初始化后114/250、breadth5，G1通过。
这证明特定native-factor局部可达性，不要求后继继续signed pooling，也不等于自然视频共享编译已成立。

G2的monotonic DP曾发生单event坍缩；首尾边界锚定后full/endpoints action+progress loss为.28167/.36207，
改善22.2%，probe38/40，机制Gate通过。有序原生response含可读动态的正证据必须保留，但该Gate不能替代完整Writer闭环。

原件：[历史§3](../../research_history.md)；G1 authority `31f0053`、G2 `c1493a1`；
[冻结完整账本](https://github.com/LinFyM/EMBER/blob/fcdb6e43706c5fcedf10eaa5d2d459602b263016/docs/research_history.md)旧§9–20。
本包提供这些记录的Git导航与现存摘要，没有新上传全部G1/G2原始rows或张量。

## 4. 局部正控与真实共享路径之间：G3/primal/EBSRI/PNBTT

current-bank operator保留analytic update，primal P1 fit/held recovery为.9717/.9545；task-local、fixed chart和free-summary
多次通过，接回真实共享Program或bank/routing之后仍失败。给定或单独优化一个有效code，与从自然视频共享预测该code不同。
PNBTT最后对齐necessity hinge后wrong/margin通过、correct/held仍失败；它停在free-query E1，真实Program E2没有执行，不能据此否定G2。

这组证据约束“局部有容量所以共享路径应当工作”的推论；负结果限于实际函数类、接口和预算，不能归纳成所有共享学习不可行。
反复叠加summary、transport、whitening、anchor、gate等没有形成稳定完整方法，但也不能把这些实验抹成没有任何正证据。

原件：[历史§4](../../research_history.md)；冻结账本旧§21–103；P1 `c9e8198`，PNBTT `e65c6388`、launch `2050de9e`。
需要某个具体推导时沿账本读取对应冻结设计；本包未把所有局部recovery重新计算为闭环分数。

## 5. 完整输出与共享获取：P/Q和clone

P/Q联合生成完整38-target A/B、移除独立carrier与span限制后，同四task短学习64步fit/held由旧41/39变64/62，各150。
主要改善Goal，Spatial有损失。完整rank、head、carrier和span等变化耦合，不能把组合收益归给唯一因素。

完整输出mixed73与target18终点训练诊断为42/180、55/180。同两弱task的whole-Writer独立clone为14/20，共享3或4/20；
共享历程为2/3/4/3，没有先达到强能力再遗忘。“只是未见task迁移难”不足以解释；容量、表示、优化和任务支持仍未单独定位。
width256只有训练结果，没有闭环成绩，不能补写成功或失败。

原件：[历史§6](../../research_history.md)；完整输出`b2bb03ce`、meta73 `041aff55`、target18 `351feb48`、clone `6efdd2e0`、
random `f3717836`；9月7日索引中`pq_shared/pq_single_task/pq_random128/pq_random256`。

## 6. 更完整的Horizon与实际学习干预

9月7日完整H/层轴/关系图有真实梯度和局部读出收益，但train24 correct192/384为69→67，other72→64。
9月8日后改为末层完整50-H、过去四帧对应、H-query、两端视觉核实、短GRU、四组局部/单向长程交替、前三组逐H回写，
最终编译完整A/B。数学上保留输入和计算依赖，并未保证训练选择有用的过程信息。

Horizon检索条件2×2中，仅关闭Compiler额外语言仿射query的候选108→126，另一初始化104→119；净收益方向保留，稳定保持未复现。
D跨rank绑定提高训练获取，却validation115→82。真实读取支持首层query差异受压缩，但attention没有饱和；冻结Q中介没有统一修复，
因此query几何不能被当成全部失败根因。少量其它task真实更新造成目标转移，P/D及交互有行为作用；不能从局部实例外推整个模型。

最强off400固定训练九臂correct/other/同suite错/跨suite错/乱序/倒序/首中末静态为
53/60/59/59/55/52/58/53/53，各96。输入改变成功集合，正确动态却没有稳定净收益；这不证明逐例等效、完全不看视频或纯task记忆。

原件：[findings](../../../findings.md)§39–49、[旧包索引](../20260911/index.json)、[已导出方法](../20260911/methods/)。
2×2/off科学commit `45e16633`，D绑定`c63f55dc`；旧包中的all/off/几何/真实更新模型身份不能混用。

## 7. 最新消费接口、条件分配和局部参照

R把每task一条K1×64queries改为两条独立K1×32queries，不是K2，也不平均LoRA；C另改为语义先行的过程消费；
S用全部真实帧和原生响应形成无序集合，不是单帧static。R400为85/400，对同query预算旧off126；C/S200为50/45，
训练任务有获取却未改善迁移。C/S只观察100/200，不能宣称收敛。off原样续训126→73→54，同时训练56→64/96，获取与迁移继续分离。

C完整冻结诊断显示correct42、两错46/46、乱序45、静态49、S51、source15，各96；固定语义后correct/static过程同为15/32。
重复首帧仍产生正常视频.692–.935倍的中心化P4。这是表示性质，不是“约82%的行为来自伪动态”的归因。
局部无变化参照扣除了同gap/窗口下静态响应，未引入错误标签或惩罚；新200六臂45/42/41/37/41/40，差额区间都包含零，
validation34/400。它尚未证明可信视频收益，也没有观察300/400。

本轮原件全部从[LATEST_EVIDENCE](LATEST_EVIDENCE.md)和本包index进入。当前最早可明确描述的表示缺陷不必就是整个失败的根因，
保留这种可能性；不能因为已实现某个修正就将理论限定在其附近。

## 8. 综合推导需要面对的证据边界

- Task-local专家250/400、G1容量、G2动态可读、早期Writer闭环与后继局部获取回答不同问题，不能互相替代。
- 对比K、视频池、task数、queries和optimizer updates时分别计量，不把更多同task演示当更多独立控制规律。
- 老方案有coarse/均值前端、观察侧适配、decoder共享方式及采样差异；任何一项都不是未经控制就能定责的原因。
- 不同checkpoint或配方的高分、视频差距、稳定性不能合并；小面板、路径替换和冻结破坏输入的分布边界必须保留。
- 历史设计中的“根因/排除”有时是当时作者的解释。例如B缩放无收益不足以普遍排除decoder学习问题；专家须核对推理本身。
- v5.2原设计的320 routing identities与当前608 target/rank槽不能按数量直接推断输出覆盖，需读旧family routing与完整LoRA合同。
- 本地图提供理论要解释的约束，不要求专家逐条重训历史，也不预先决定回到旧结构、必须新loss或保留当前模块。
