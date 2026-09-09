# K1能力缺口：冻结接口与监督诊断

日期：2026-09-09。本报告逐步记录诊断协议、实测结果及边界。Owner允许深入分析、修正及非正式验证，明确禁止正式架构/训练方式修改和正式训练启动。上下文400全部正式证据已收齐；canonical Writer、Meta、source与全部checkpoint保留。

## 1. 已知问题与竞争解释

- 原始K1 correct55/110/86/87，移除Compiler直接语言内容残差后75/110/106/103；本轮逐帧上下文条件52/103/79/90。前一干预有局部收益，未修复获取/保持；本轮至400无整体优势。
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

### 7.3 原生梯度执行的实证修正

进一步核对原生联合attention：有效prefix不能读取suffix，但拼接计算图仍会让冻结Pali流参与反传。只在其18个o_proj输入处临时detach，保留完整native联合前向、全部prefix K/V数值和AE反传；前提为source冻结、LoRA仅AE、loss只取suffix。

固定current400 task0/teacher46的8真实queries、micro2，原路径与该切点对照：原生action head逐点差0，完整76张量LoRA梯度max abs/relative L2差均0；18hooks各执行4次。allocated峰值16.445→12.579GiB，单次计时3.232→2.035秒（同卡共驻且非隔离benchmark，仅证实本次成本降低）。原件`local_oracle/prefix_gradient_check.json`与脚本。

据此统一重启同64步oracle，物理micro4/8/8/8，数据/噪声/目标/临时参数初始化和预算保持；此前中途结果不选点、不混入最终曲线。该调整仅为删除对AE LoRA无贡献的反向计算，不改正式源码、模型架构或训练方法。全局可推广性仍由冻结prefix和信息流合同约束，不用于任何带prefix可训练参数的场景。

## 8. B2完整原生执行结果

两组各12task全部exit0，native FM与10-step采样各36,864 query/chunk预测（12列×24task×128queries），模型更新0、sampler不推进。所有实际time/noise/target配对；每task非source首8query的cached与uncached官方采样最大动作差均0。两wrapper2148.45/2105.15秒，task峰值约15.82GiB。原件`causal_diagnostics_20260909/openloop/summary.json`及两组逐task JSON/NPZ。下表checkpoint列先分别计算teacher46/47，再平均指标；没有平均LoRA或选择视频。

| 模型 | 全50×7 FM | 前5 valid FM | 原生采样前5 valid动作MSE | 前5夹爪符号错误率 |
|---|---:|---:|---:|---:|
| source | .151461 | .182649 | .147954 | 7.562% |
| context200 | .111063 | .125180 | .105742 | 5.927% |
| context400 | .105060 | .118425 | .102172 | 6.182% |
| first-query400 | .105684 | .120072 | .104265 | 6.330% |
| expert2000 privileged | .104119 | .117922 | .108964 | 6.216% |

200→400全FM 24/24任务改善，前5 valid FM 23/24改善；真正10步采样前5 valid动作MSE 15/24改善、9/24变差，均值下降.003570。xyz/rotation分别平均下降.008147/.003272，gripper MSE增加.009269；符号错误率增加0.255个百分点。因此夹爪存在分量性退化，但并非所有动作或阶段都变差。

更关键的是，**相对first-query400，本轮context400原生采样前5 valid误差在19/24任务更低、均值低.002093，train96却49对59，validation90对103。** 单纯把一阶FM替换成10步采样后的示范动作距离，仍没有解决指标与行为排序不一致。它支持进一步关注决定成败的决策/状态覆盖与可迁移功能，而不能证明某一个loss权重就是原因；离线动作还存在多解和时机差异。

### 8.1 对夹爪事件小面板解释的修正

完整128query面板中有22task具备前5夹爪切换±2位置。该邻域的FM gripper平均下降.058989，实际采样gripper MSE从约1.7880降至1.7200（下降.068040，12task改善/10变差）。先前B1 32query小面板中事件邻域变差的方向没有在此复现，不能把它列作已确认主因。

同时，专家的前5采样MSE .108964高于context400，并且事件邻域约1.83698也更高；专家训练读过query episodes且其既有658/1200不是当前32state配对面板，所以这只是进一步提醒不能用离线示范距离排序闭环能力，不能直接声称本面板专家行为更强。后续以固定诊断闭环实际结果为准。

### 8.2 额外语言与视觉分支的实际动作影响

均相对context400/teacher46，保持原生语言、Z/R和真实视频顺序。RMS为同采样noise下预测动作的实际变化，和平均误差变化是不同量。

| 冻结干预 | Δ全FM | Δ原生采样前5 valid MSE | 前5动作变化RMS | 前5MSE变差tasks |
|---|---:|---:|---:|---:|
| H-read language输入零 | -.00000048 | +.0000183 | .00224 | 10/24 |
| Compiler language输出零 | +.000430 | +.002779 | .05073 | 15/24 |
| visual-read输出零 | +.000153 | +.000133 | .02871 | 13/24 |
| 三处后端language同时零 | +.008403 | +.008470 | .12740 | 20/24 |

H-read额外language的当前边际作用很小，B1结论得到完整native复核。Compiler language与整体后端language影响明显，不能据“原生表示已有语言”推断当前参数自动学会不依赖这些路径；直接置零仍不是fresh删除性能的对照。

visual-read的平均误差变化虽小，实际动作RMS .02871并不为零，说明可能存在方向相反的收益/损失；不能称其未被使用。正确teacher46→47的前5动作RMS在context400约.01765、first-query400约.02910；两条同task正确视频的接近输出可能是稳健性，也可能是共同静态信息主导，不能由此单独裁定动态证据必要性。最终视频因果controls继续封存。

## 9. B3冻结分支的train24配对闭环（结果前登记）

B2显示动作扰动和平均MSE可以明显不同，尤其visual-read；因此直接补当前400的训练任务行为依赖，不以平均误差替代闭环。固定全部train24、正确teacher46、states32–35；normal及B2四个冻结干预（H-read language零、Compiler language零、visual-read零、三处language同时零）五臂，共480行。每task五臂从同一个真实R/Z条件完整生成各一套LoRA；无视频重排、无训练更新、无held任务或最终视频controls。

每臂新跑4个state，normal也重新执行，不借正式train96中teacher46–49逐state不同的旧面板。使用与C相同的官方source/preprocessing/flow10/replan5/RNG/asset/终止条件、dynamic cost-balanced long-first queue、persistent policy/env workers，保留部分BDDL goal谓词，不录额外图像。预先保留全部五臂结果、逐task/suite/breadth及R/G/L/churn，不由行为挑分支或选checkpoint。这里检验的是冻结模型当前依赖，不能替代fresh删除后的模型比较。新增adapter/rows预计小于.6GiB，纳入原B2+C诊断总2GiB预算；C等待期间可使用已完成B2释放的两卡。


### 9.1 B3执行记录

120套完整adapter物化及480行queue准备已全部exit0，进程内87.61秒、allocated峰值10.507GiB、adapter共456005760 bytes。实际24task各五臂teacher/frame序列相同，frame stride5并按既有loader保留最后一帧，全部38target/76张量；只读取action-hidden teacher46，不构建action query数据。物化与评测source相同，normal也新生成。

在双节点实查和strg01/data1 quota确认后，GPU02p1/3各两个persistent worker开始cost-balanced动态queue；连同C四卡共六张有用设备。原件`causal_diagnostics_20260909/branch_rollout/manifest.json`、`launch_contract.json`与`closedloop/contract.json`；所有最终行仍待收齐后统一解释。


B3首次worker启动停在LIBERO初始化提示：临时prepare未链接既有asset config；当时尚无queue claim或评测行。已停止本任务四个进程，保留startup日志，链接reference contract同目录的canonical `libero_config`后重启；C prepare同步补齐该链接。首批真实rows正常落盘，配置修正不改变policy、asset内容或任务协议。

两worker/GPU实际占约22–24GiB、余量21–23GiB；结合此前相同官方执行路径三worker验证，再次实查两节点后每卡增加一个persistent worker，保持相同4-env物理批次与实际RNG、同队列，合计两卡六worker。后处理同时区分初始已满足、执行中新达到及末尾又丢失的BDDL goal slots；它只是部分目标进度证据，不是完整行为分类。

## 10. C固定fit采样点的冻结重采样检查（结果前登记）

C的128条fit queries在64步中使用固定time/noise；因此fit改善和独立episode表现不同，既可能来自状态/动作覆盖，也可能仅适应该批FM采样点。补一个冻结面板：全部八task、同128个fit图像/state/动作，保持episodes16–41和query选择seed；只将原`policy_rng_seed`固定加100000003，重新独立生成FM time/noise。比较normal及P4/C/AB最终64步保存的完整LoRA，共4096次query预测；不生成梯度、不更新任何局部或正式变量、不重新选择节点、不读新视频或held tasks。

完整native联合FM、物理micro8、BF16及原source/normalization保持。记录真实raw actions/pad与query metadata，验证和原fit面板相同；保存新noise/time，四臂严格共享。若原fit和新噪声fit均改善，而独立episode无改善，证据更偏向状态/episode层面的泛化限制；若只在原fit采样点改善，需降低对局部oracle可达性的解释力度。一次128query重采样不构成FM总体积分上界，也不能单独推翻FM学习方法。该面板预计小于30MiB，可与最终C闭环并行，仍纳入原2GiB诊断预算。
