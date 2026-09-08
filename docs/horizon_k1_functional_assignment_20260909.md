# K1冻结训练任务：完整adapter功能对应

日期：2026-09-09。按[active design §8.2.7](horizon_relation_video_writer_design.md)预登记完成，无训练更新、无Writer重新生成、无新rollout及validation/test动作访问。原件为`runs/analysis/horizon_relation_writer_20260908/k1_first_query_only/functional_assignment/`，逐行`task*.json`、`summary.json`及`runtime.json`完整保留。

## 1. 结论与作用边界

新first-query-only在训练分布内已经形成有用的条件功能区别，且200→400更强：24个任务的两条视频在400均比source低FM，也均比其余23个任务adapter的平均FM低；两个16-query半面板方向都一致。结合train96从46增至59、validation从110降至103，这进一步降低“训练内条件编译普遍未形成”的优先级，支持继续聚焦已获取的任务特化如何迁移和保持。

它没有证明语义表示充分，也没有证明FM已足够支持闭环。特别是同suite细分优势在Object/部分Long很小，不能用微小差值或排名确定错误路由。执行policy仍读正确language，多个任务合理共享技能修正；不存在“每task必须有不同LoRA”的额外要求。本矩阵不单独定责前端、decoder、优化器或状态覆盖。

## 2. 固定面板与实际执行

- 行：全部固定train24；每task从episodes42–45按seed=`20260908+task`固定32个真实观测/state/action queries。policy保持该行的正确language。
- 列：新200/400现成train96 banks中全部24task×teacher46/47的完整38-target rank16 LoRA。每套分别执行，未平均或择优；另有identity/source列。
- 合计24×97×32=74,496次query级FM预测。每行的实际actions/frames/time/noise跨97列配对；sampler未推进、无参数梯度、无optimizer。两个checkpoint和两teacher均完整覆盖。
- 比较的是完整task-language-and-video条件生成的adapter迁移，不是单独改变video的因果control；不替代最终wrong/shuffled/reversed等实验，也不用此矩阵选checkpoint或构建任务adapter字典。

执行来自clean pushed frozen `fea45593`的source、数据与原生functional API，分析脚本随原件封存。首个全forward版本因97列重复计算同一冻结query prefix，在首行完成前终止；无完整行被用于分析。替代执行每行只建立四个8-query原生prefix KV，再通过native `denoise_step`计算每列配对FM，完整449.10秒、峰值11.750GiB、exit0。每行约16秒，所有24行均完成。

首行source及首个同task adapter各8query与完整native forward核对，共额外16次验证预测；逐点7维loss平均绝对差为.001956/.001835，相对.892%/1.122%，在预登记.002/2%容差内。这是有数值差异的缓存执行，不能把小于该尺度的细微排名当作稳健机制证据；此抽查也不是全矩阵误差上界。不要将本32-query面板数值与既有128-query或16-query FM直接拼接。

## 3. 整体与分suite结果

下表“其它−自身”为其它条件adapter的FM减去同task条件adapter的FM，正值表示同task平均更好；任务等权，两个teacher分别计算后才汇总指标。

| 指标 | 200 | 400 |
|---|---:|---:|
| source FM | .154875 | .154875 |
| 同task FM | .111621 | .106557 |
| 其余23task−同task | .009396 | .015491 |
| 同suite其余5task−同task | .003599 | .005578 |
| 跨suite18task−同task | .011007 | .018244 |
| 两teacher均优于source的tasks | 24/24 | 24/24 |
| 两teacher、两半面板均优于其余23task均值 | 22/24 | 24/24 |
| 两teacher均优于同suite均值 | 19/24 | 22/24 |
| 两teacher、两半面板均优于同suite均值 | 15/24 | 18/24 |
| 自身在全部24列排名第一的conditions | 6/48 | 20/48 |
| 自身在同suite6列排名第一的conditions | 16/48 | 26/48 |

排名仅作描述，尤其近似并列不能机械解释。400相比200，21/24任务的两teacher平均FM降低；这与128-query面板的23/24并不矛盾，两者采样数量及数值执行不同。

| suite | source | 自身200→400 | 同suite其它−自身200→400 |
|---|---:|---:|---:|
| Spatial | .184800 | .142445→.136662 | .003276→.010385 |
| Object | .117215 | .084987→.079961 | .000911→.002015 |
| Goal | .172169 | .105471→.102127 | .008694→.006875 |
| Long | .145318 | .113582→.107480 | .001513→.003039 |

Spatial训练任务的功能特化明显增强，但同期验证Spatial仍只有3/100、训练96该suite仅11→12。Object训练闭环12→21，其同suite细分FM优势仍较小。这些差异说明不能把某一个FM margin当作闭环能力或语义理解的统一尺度。

400同suite两teacher均为正的条件未覆盖task14、36：14两teacher方向相反且量级很小，36平均为负、两半面板亦有差异。task15/25/28/35的同suite两半面板有异号；其中35虽然全部24列排名第一，平均margin仍很小。完整逐task原件包含所有差值、半面板与列映射，不能只挑这些弱例子扩大为全局缺陷。

## 4. 对下一项干预的约束

1. 不再以“视频路径断开”“训练内没有任务区分”或“参数量大就是过拟合”为已确认根因。图文task tokens确在现有Z读取中，global语言编码也含位置；此前完整功能/图像与本矩阵证据均保留。
2. 当前global语言条件仍由静态token embeddings经位置化单query attention生成。其各head的token score在该读出前不依赖其它token的上下文；它可以学习任务条件，但本身没有直接继承冻结Gemma的上下文task-token语义。其它路径仍可提供语义，不能据此声称整个Writer只有词袋。
3. 若下一项检验上下文语义访问，应明确新增的是现有原生图文task-token状态到过程条件的可学习访问，而非新外部信息或恢复一整套前置S。保留完整H、过去局部关系、过程形成后的两端视觉核实、四组时序及完整LoRA输出，以受控学习和strict400裁决迁移是否改善。
4. rank共享已有Target-Owned近等价负结果；Task-Grounded Visual-Value/Full-Factor也曾为88/86/86/96及91，不能把其结构名称当作修复保证。额外meta任务、专家监督与状态覆盖仍是竞争解释，本矩阵未将它们排除，也不自动授权在同一轮混入。
