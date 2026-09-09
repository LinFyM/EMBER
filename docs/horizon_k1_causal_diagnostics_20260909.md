# K1能力缺口：冻结接口与监督诊断

日期：2026-09-09。本报告逐步记录诊断协议、实测结果及边界。Owner允许深入分析、修正及非正式验证，明确禁止正式架构/训练方式修改和正式训练启动。已有上下文400评测收尾继续；canonical Writer、Meta、source与全部checkpoint保留。

## 1. 已知问题与竞争解释

- 原始K1 correct55/110/86/87，移除Compiler直接语言内容残差后75/110/106/103；本轮逐帧上下文条件52/103/79/90。前一干预有局部收益，未修复获取/保持；本轮至300无整体优势。
- 上一轮train96由46升59，训练内adapter功能对应增强；本轮41升49，400仍低于前轮59。不支持普遍未学会条件编译，也不证明训练总体已解决。
- 本轮400 held FM .105074533，比前轮.105737594低，但train96反而少10；平均误差不能作为闭环收益代理。
- 既有回放包含错误对象/实例、正确子目标后未完成、抓取/时限与干扰物等多种缺口；不能统一归因motor或单个过程模块。

本轮检验四类解释：额外语言条件影响信息使用；有用过程关系未形成或在读出中未保留；Compiler/native读出未把表示转成正确功能修正；监督分布/共享更新与闭环关键决策存在不一致。它们可能共同存在，不预先要求唯一根因。

## 2. 证据等级与边界

1. 完整已有闭环、逐任务配对及实际运行合同是行为事实。
2. 冻结接口干预确认特定参数状态下的功能依赖；大幅干预可能偏离训练时激活分布，不能替代fresh学习或证明删除后有益。
3. 局部可达性/oracle若实施，只能说明固定接口及预算内能否改善，不自动说明共享可学习、跨任务可迁移或deployment合法。须在执行前另记数据、变量与预算。
4. 梯度、几何、attention图或微小FM差异只作定位。最终方案以证据支持程度和可区分下一步表达，不把结构名字当机制证据。

不访问validation/test actions或产生其梯度；不使用最终wrong/static/no-video/shuffled/reversed controls反哺设计。模型诊断只用train24。已有正式validation只按登记口径完成分析。

## 3. B1冻结分支干预（结果前登记）

对象固定为同架构上下文checkpoint200/400，冻结运行面`9abc9b95`。全部train24、正确teacher46/47；每task固定seed=`20260908+task`的32条独立动作queries，来自原diagnostic episodes42–45。两个teacher、两个checkpoint和全部arms共享实际query/frame/time/noise。训练sampler不推进，source/Writer/Meta无参数更新或梯度。

九臂每条件均完整执行：

| 臂 | 精确定义 | 保留 |
|---|---|---|
| normal | 原生完整运行 | 全图 |
| local_language_zero | 四组local入口的language张量置零 | 原生语言、上下文Z、H-read、Compiler语言 |
| hread_language_zero | 四组read_language线性层输入置零，保留其已学bias | H-read attention和数据依赖K/V；不是重新学习的Query |
| compiler_language_zero | query_language输出置零 | target/rank首次query、P4和全部残差/解码 |
| all_backend_language_zero | 同时上述三项 | 原生语言/Z task tokens/R任务条件化 |
| visual_read_zero | 四组两端visual_read的最终output置零 | 原生Z语言条件/R、对应与局部组织 |
| local_neighbor_zero | 四组neighbor_output输出置零 | 当前H、local后FFN、H-read/长程 |
| temporal_attention_zero | 四组temporal.attention.output置零 | temporal残差与FFN、局部关系 |
| writeback_zero | 前三组writeback返回其输入states | 所有组局部/H-read/长程及第四组P4 |

临时hook只存在于独立分析进程，不改正式源码/配置、checkpoint或输入视频；视频不重排、不删帧，H=50与原生prefix保持。干预支持的是特定分支的边际依赖，不能把visual_read零当全部视觉消除，也不能把all language零当无语言模型。

每条件记录原生50×7 loss的逐query、horizon与动作维度统计，前5/后45对照，实际速度预测扰动，hook调用覆盖与输入配对。全量统计到24task才解释；不由单task正负选checkpoint。两teacher分别报告再汇总，防止一条视频偶然性。附同query source参照。

可复用此前已核验的冻结query-prefix缓存与原生denoise_step，避免每arm重复冻结prefix。首task每checkpoint的normal与source各做一次8query完整native forward核对；沿用逐点loss平均绝对误差≤.002且相对≤2%的既有界限。该核对不是全体误差上界，接近数值差异尺度的结果保持不确定；必要后续仅对有决策价值的差异使用完整native forward复核。

## 4. 并行分析与后续选择

只读CPU量化实际监督的episode进度、50horizon尾部padding、前5动作权重与夹爪事件覆盖；结合真实数据轴解释，不把episode进度当精确语义阶段。历史复核聚焦专家/clone、G1/G2与接口oracle，防止把不同监督/初始化的正结果当当前模块保证。

B1及上述量化结束后，按结果登记有信息量的下一项：实际10-step采样与FM的差异、冻结接口局部可达性、或train任务诊断闭环。避免重复已有functional assignment和几何敏感性，不默认全面重训或盲扫参数。后续具体执行和原件由本报告及progress记录。

## 5. 执行与结果

B1已完整执行并配对核验通过。400 train96已完整49/96、breadth20，全部配对/六worker退出通过；原件`runs/analysis/horizon_relation_writer_20260908/k1_frame_contextual/segment200_400/train96_step400/completed_summary.json`。400 validation完整90/400；完整本轮证据位于`k1_frame_contextual/segment200_400/round_evidence.json`。


### 5.1 B1结果与下一步含义

两个checkpoint各48条件、13824 arm query预测，模型更新0、两个wrapper exit0（303.87/303.71秒）、peak11.779GiB。逐条件真实action/video/time/noise配对及hook覆盖检查通过。原件`runs/analysis/horizon_relation_writer_20260908/causal_diagnostics_20260909/branches/summary.json`，保留每个条件的逐坐标loss、velocity、raw actions/padding与实际noise。

| 干预：相对normal的FM变化 | 200 | 400 | 400两条video均变差tasks |
|---|---:|---:|---:|
| local language零 | +.006976 | +.007630 | 21/24 |
| H-read language输入零 | -.000001 | +.000003 | 5/24 |
| Compiler language零 | +.000778 | +.000576 | 15/24 |
| 三处同时零 | +.006363 | +.007570 | 21/24 |
| visual-read零 | +.000429 | +.000367 | 17/24 |
| local-neighbor零 | +.035638 | +.039144 | 24/24 |
| temporal attention零 | +.001784 | +.001157 | 19/24 |
| writeback零 | +.002043 | +.000849 | 17/24 |

normal .111548→.105841，source同面板.154875。局部消息分支有显著功能影响，但它承载当前H、语言与关系等混合信息，不能单独证明跨帧动态已经正确理解。H-read额外language的边际影响很小；visual-read、Compiler语言与其它较小差异须结合完整native执行及动作变化，不能简单称无用。400各干预速度RMS仍非零（H-read约.00396、visual-read .01363、local-neighbor .20686、temporal .04199）。

### 5.2 监督分布与坐标误差

CPU逐条重建102400实际queries/512万目标位置：padding16.4778%，末动作及其重复17.1496%；前5真实动作占总loss坐标9.866%。起点进度均值.5001、各十分位约均匀，展开目标最前/末10%占1.94%/26.76%。624 episodes共有1418次夹爪符号切换，全部进入过监督target，1382次进入过前5位置；正负夹爪指令48.41%/51.59%，并非事件整体缺席或静止动作主导。Long padding最低8.95%却仍弱，不能由该分布直接解释主问题。原脚本与统计在`causal_diagnostics_20260909/supervision_distribution/`。

B1 normal的前5 loss .122280→.115106，前5 valid .007133的误差下降，整体改善约92%来自真实目标位置；因此不能用“只是padding被拟合得更好”解释FM/闭环不一致。夹爪切换±2位置的task等权条件均值1.17034→1.19754，前5邻域1.00746→1.03722；但这些只来自固定32query小面板，事件数、条件均值与总loss贡献口径不同，不能直接认定夹爪是唯一根因。

### 5.3 历史能力对照的有效范围

train24独立rank16专家为658/1200，预算32000 queries/task且动作池0–49；说明当前source/LoRA空间存在较强局部能力，不证明当前短学习易达或所有任务均强。旧whole-Writer两task clone 14/20对共享3/20说明局部可学与共享学习代价可分离，但没有定位单个模块。

G1通过点114/250的launch明确`--stop-after-step 0`、optimizer_updates_before_checkpoint=0，是从已知成功参考出发的解析signed构造；旧500步优化为88/84。它不能被当作FM学会视频到参数的证据。G2用直接phase action/progress/predicate等监督，未生成LoRA闭环。旧局部primal fit/held恢复.9717/.9545，而冻结G2共享readout held约.25–.27，仅说明那套表示/reader与功能方向之间仍有缺口，不单独证明当前P4丢失信息。

本轮后续必须比较当前checkpoint的相邻接口实际功能和行为，不重做一份单一Program几何probe就宣称定位。

## 6. B2完整原生FM与部署采样诊断（结果前登记）

固定train24，每task原episodes42–45、seed20260908+task、128queries；微批8，各列共享实际FM time/noise与另行固定的128×50×32采样noise。比较12列：source；context200/400各teacher46/47；first-query400各teacher46/47；train专家2000；context400teacher46的H-read language零、Compiler language零、visual-read零及三处language同时零。干预定义沿B1，不改架构或参数学习状态。

FM使用完整native联合forward、训练BF16计算；实际动作使用官方10-step `predict_action_chunk`且无outer autocast，保留双相机、状态prompt、source normalization和动作维度。可缓存同task/microbatch/执行模式下真正native生成的冻结prefix；不跨FM/采样混用，也不把B1近似cachedFM当原生复核。首个非source列首8query做cache/uncached真实动作核对并记录误差；不扩大dtype。

保存逐坐标FM、10-step动作/目标/实际noise与pad；比较前5/后45、valid/pad、xyz/rotation/gripper和事件邻域。采样误差相对单条示范仍受多解和时机差异影响，不代替闭环。专家训练读过42–45，只作为privileged能力参照，不当独立held泛化证据；本诊断不使用其validation专家。

## 7. C冻结接口局部oracle（结果前登记）

只用当前400，固定train tasks **0/7/14/16/20/25/34/35**（四suite各二，包含当前强弱任务及专家本身较弱的反例），正确teacher46。正常Writer生成P4、最终C和完整A/B作为共同起点。固定128个fit queries来自16–41（seed20260910+task），128个独立query来自42–45（seed20260908+task）；fit固定真实time/noise，所有接口使用同一组。无validation/test梯度。

分别优化三个临时变量：P4（Compiler/decoder冻结）、C[38,16,256]（decoder全部冻结）、完整76张量A/B（source冻结）。原Writer/Meta参数一律冻结，采样器不推进，无正式optimizer/scheduler/checkpoint写入。变量写为`base + .1*rms(base)*delta`，delta从零开始，独立Adam lr=.1、无weight decay、固定64步；每tensor原始dtype在实际消费前保持。仅为统一相对局部步幅，不声称三种接口的函数空间距离相同。

固定记录0/16/32/64的fit与独立FM，保留64步最终adapter；不用独立query或闭环选择中间最优点。预算是每臂8192重复query预测，unique fit128；一次未收敛不能证明空间不可达。必要的诊断闭环只在上述train tasks、固定states32–35，用同teacher46生成的normal与各64步oracle严格配对，并复用官方执行逻辑；独立标记action-supervised task-local oracle，不能纳入zero-interaction成绩或部署。

若P4/C都能改善实际功能和行为而normal弱，提高对上游获取/共享学习的关注；若直接A/B有效而C无效，只能定位冻结读出或该局部求解；若所有接口仅fit改善而独立/闭环不改善，进一步区分样本覆盖、监督与局部求解，不据此直接改目标函数。所有结果允许混合，不预设必须支持某个方案。

### 7.1 C执行修正（科学结果前）

首轮临时脚本的source identity未转GPU，执行前即device mismatch，已修正；随后首task完整原生数值核对显示沿用B1的近似KV路径超出登记界限。因此C统一使用完整native联合FM forward，只复用同fit/独立面板、同microbatch的冻结image/text embedding；全部action-expert与联合transformer计算保持原生执行。旧尝试只保留失败日志，未产生可解释的oracle结果，不改变任务/数据/64步优化协议。该失败限制近似缓存的复用范围，也强化B1微小差异必须由B2完整原生复核的边界。

完整native反传micro8在部分共驻设备OOM，未作科学解读。保持逻辑128/time-noise/64步，改物理micro2（GPU02p0）及micro4（p2/4/6）；每task内部三个接口及其fit/独立基准一致。四进程分别tasks0/14、7/16、20/34、25/35，同节点连同B2共六张有用卡。实际显存与吞吐由日志记录，不为低位一致扩大dtype或固定batch1。

### 7.2 固定train诊断闭环

八个task各states32–35、teacher46，normal/P4-final64/C-final64/AB-final64严格配对，另补同source及既有expert2000作为执行能力参照，共6×32=192行。source/专家不消费teacher；专家训练池0–49已包含本次动作池，只是privileged参照。使用现有cost-balanced动态queue、long-first、persistent policy/environment workers和完整官方rollout_shard；保留BDDL goal predicate变化作为部分进度信号，不录额外图像、不用于梯度或选点。与正式train96逐state不同teacher46–49的面板分开，重新运行normal，不能借用原96行冒充严格配对。
