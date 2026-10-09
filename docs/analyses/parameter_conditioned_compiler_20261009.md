# 实际参数条件编译：完整学习、180科学停止与已发生的360证据

2026-10-09；实验执行/消费 session `01a11a0a-fd39-74b1-83ba-001ede5330df`；主讨论
`01a11b05-3460-7eb0-aca8-177d6d86ef48`。科学合同是
[parameter_conditioned_compiler_20261009](../designs/parameter_conditioned_compiler_20261009.md)，
最近似完整方案及启动推导见[首批分析§7–8](experience_conditioned_compiler_20261009.md)与 findings§411。
本报告直接读取原始行、实际事件、训练 metrics、代码及退出/费用；没有新增训练、held离线标签读取或Test使用。

## 1. 完成范围与正式裁决

χ180 correct strict paired400 为 **127/400**；同一固定 train48 的 MT/end/null 为 **27/25/25**。
这同时满足预注册的 `formal180<=135 AND train end<=MT AND train end<=null`，本批科学停止。
最终 attempt `topology_scaleup_001` 退出0、status=`scientific_stopped`；不是需要修补的工程失败。

由于180 formal与后段工作合法并行，池0全部144条件、180完整checkpoint、刷新72条件、360完整训练checkpoint、
两个train48面板均已先完成。360 formal已经完成的32个条件全部保存，368 pending取消；没有360完整400、
没有selected checkpoint。不能改选180，也不能在看到局部好结果后静默取消停止规则、补完队列或声称正式相邻稳定性。

关键原件根为 `/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009`（下称ROOT）：

- [机器可读科学统计与完整成功集合](/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009/scientific_analysis.json)实际位于
  `/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009/scientific_analysis.json`；生成器为 `ROOT/analysis/consume_batch.py`。
- `ROOT/evaluation/{formal180,train180,train360}/results.json`，以及各自contract、queue、primary shards、条件record、原始经验与完整参数。
- `ROOT/evaluation/formal360/queue.sqlite3`及32个已完成primary shards；没有完整results文件。
- `ROOT/pools/{pool0,refresh180}/manifest.json`、实际经验/参数版本；`ROOT/training/metrics.jsonl`及180/360完整checkpoint。
- `ROOT/batch_attempts/topology_scaleup_001/{batch_execution.json,batch_exit_code,wakeup_receipt.json}`、
  `ROOT/early_stop_requested.json`；所有旧attempt及失败保留。
- `ROOT/costs.jsonl`、`ROOT/analysis/batch_resource_audit_20261009/{resource_exit_audit.json,report.md,evidence_snapshots.json}`。

消费时重新验证完整400的每task各50个state、50条teacher各一次，两个train面板各16个condition×3个state，
以及原始language/scene/video/RNG和policy-noise共同前缀。MT、T2340和旧135的正式参照沿用contract所指原件；
train强MT只实际执行一次，360复用同一原始48行。没有重复fresh采集或重跑已完成条件。

## 2. 能力保持、获得和损失

下表的task为suite内local ID；每task50行。R/G/L均以强MT→χ180计算。

| suite/task | MT | T2340 | 旧experience | χ180 | R/G/L |
|---|---:|---:|---:|---:|---|
| Spatial3 | 41 | 45 | 38 | 39 | 35/4/6 |
| Spatial6 | 7 | 6 | 2 | 0 | 0/0/7 |
| Object1 | 36 | 45 | 29 | 29 | 26/3/10 |
| Object6 | 10 | 5 | 14 | 10 | 3/7/7 |
| Goal3 | 0 | 0 | 1 | 0 | 0/0/0 |
| Goal6 | 35 | 36 | 32 | 34 | 32/2/3 |
| Long1 | 24 | 24 | 19 | 15 | 10/5/14 |
| Long9 | 0 | 0 | 0 | 0 | 0/0/0 |
| 合计 | 153 | 161 | 135 | 127 | 106/21/47 |

四suite按 Spatial/Object/Goal/Long 为 **39/39/34/15**，强MT为48/46/35/24，T为51/50/36/24，旧135为40/43/33/19。
相对MT四suite均下降；Object6总分相同但交换14行，不能把净分不变写成能力不变。
χ180 breadth=5/8，MT/T为6/8，旧135为7/8；没有可靠覆盖扩大。

| 正式参照→χ180 | R | G | L | churn/400 | success-set Jaccard |
|---|---:|---:|---:|---:|---:|
| MT153 | 106 | 21 | 47 | 68/400=17% | 0.6092 |
| T2340 161 | 98 | 29 | 63 | 92/400=23% | 0.5158 |
| experience135 | 97 | 30 | 38 | 68/400=17% | 0.5879 |

不是所有任务都恶化：相对旧135，Spatial3增1、Goal6增2；相对T，Object6增5。
但这些局部变化没有抵消Object1、Spatial6及Long1的损失，不能只选正例解释新方法成立。

固定train面板每task6行（两个teacher×三个相同新初态），不是48个独立任务。

| train suite/task | MT | end180 | null180 | end360 | null360 |
|---|---:|---:|---:|---:|---:|
| Spatial0 | 6 | 2 | 2 | 2 | 2 |
| Spatial1 | 0 | 0 | 0 | 0 | 0 |
| Object2 | 2 | 2 | 2 | 2 | 2 |
| Object3 | 4 | 5 | 5 | 6 | 5 |
| Goal0 | 1 | 2 | 2 | 0 | 2 |
| Goal1 | 6 | 6 | 6 | 6 | 6 |
| Long2 | 2 | 2 | 4 | 3 | 3 |
| Long4 | 6 | 6 | 4 | 6 | 5 |
| 合计/breadth | 27/7 | 25/7 | 25/7 | 25/6 | 25/7 |

MT→end180 R20/G5/L7，churn12/48、Jaccard0.6250；MT→end360 R21/G4/L6，churn10/48、Jaccard0.6774。
null→end180 R21/G4/L4，churn8/48、Jaccard0.7241；null→end360 R23/G2/L2，churn4/48、Jaccard0.8519。
end180→end360 R23/G2/L2，churn4/48、Jaccard0.8519；null相邻R21/G4/L4，churn8/48、Jaccard0.7241。
因此没有训练域净修订或优于null的证据，不能把正式缺口只解释为held迁移。总分相同也不证明经验完全不影响结果。

一个具体不利条件是Spatial0/teacher43：180实际实践793steps、4次编辑、自身success，之后三个新初态MT全成功、end/null全失败；
360同teacher实践1024steps预算停止，end/null仍全失败。teacher29下MT也全成功，但两个节点end/null均只成功2/3。
Object3则180为5/6、360为6/6；Goal0从180的2/6变为360的0/6，null仍2/6。收益和相邻损失都保存，
不能把自身一次成功、累计经历或某个task改善当成后续固定LoRA在新初态保留能力的保证。

## 3. 已发生的360正式局部改善

32个已完成条件全部属于Long1，由long-first队列及实际启动/停止产生；不是随机的八任务代表样本。
χ360为 **20/32**，同32行χ180为6、MT16、T17、旧experience14。

| 同32行参照→χ360 | R/G/L | churn/32 | Jaccard |
|---|---|---:|---:|
| χ180 | 5/15/1 | 16/32 | 0.2381 |
| MT | 13/7/3 | 10/32 | 0.5652 |
| T2340 | 10/10/7 | 17/32 | 0.3704 |
| experience135 | 12/8/2 | 10/32 | 0.5455 |

这是确实存在、幅度不小的局部正证据；不能把整批阴性写成360所有能力都无改善或理论上不可学。
同时，它只证明已完成Long1子集的变化，不证明χ360完整400胜MT、相邻稳定或真实经验/视频的独立贡献。
180严重非通过和train无净收益的停止合同仍执行；所有已完成32行、22,362实践steps、13,279 finalsteps及停止启动开销保留。
若main认为此局部变化值得后续投入，须形成新的明确判断与合同，不能将此次未执行368行写成已完成或事后选点。

## 4. 实际学习曝光与连续机制解释

本批要解决的是“来自教学和实际经历的参数修订，为什么没有变成更好的固定闭环能力”。
实际联合干预为强MT起点、真实参数条件、零编辑保持、成功经验进入编辑和共享事件复用，
同时取消旧持久Q及本批PG。联合结果不能单独归因于其中任一变化。

**实际输入怎样抵达参数。** Λ为38个target的完整rank128 A/B，参数编码器按A行/B列、固定MT矩阵RMS单位、
target/rank地址与共享角色投影形成`38×128×256`的q；它不是上一轮持久隐Q。真实经验包含每个决策的前后双相机Φ、
真实执行flow t=1/.1的完整`50×1024` H、前后8维proprio、实际执行的至多5×7动作及mask、reward/done/预算反馈和episode/step。
完整patch/hidden与numeric事实汇入每决策token，再经有序两层temporal及16-slot池化；失败不被过滤。

合法teacher只有exact language和stride5有序RGB，source Φ为`512×2048`、固定MT probe H为`50×1024`。
Reader先用language、经验槽和实际q调制8个空间查询，再读取每帧完整patch/H，加入实际帧位置、camera/horizon/modality及端点标记，
经四层temporal得到language+8T tokens。Editor读取教学/经验，并沿rank/target轴交互形成δq。
这给出了信息可影响读取与编辑的真实算子；代码和attention结构本身没有证明这些token学到了抓取、放置或纠错关系。

**零点的数学范围。** 每个64维实际块x、固定RMS单位s、可训练地址e进入

\[
b=W_x(x/s)+W_e e+b_0,\qquad
\delta x=sW_o[\operatorname{SiLU}(b+W_q\delta q)-\operatorname{SiLU}(b)],\qquad x'=x+\delta x.
\]

Wq/Wo非零、δq末投影零初始化，故δq=0时实际incoming保持；条件Jacobian为
`s Wo diag(SiLU'(b)) Wq`，不要求第一步所有上游梯度非零。
实测update1四个配对FM差均0，只有update_out梯度非零；其余六个参数组从update2到360共359次非零。
因此“始终identity/断图”与当前证据不符。但学习后δq不再为零，identity并不保证保持功能。
物理LoRA更新实际满足
`Δ(BA)=BδA+δB A+δBδA`；这些全层扰动会改变原生执行hidden和动作，不能用因子保持形式替代闭环保持验证。

**真实曝光限制了学到什么。** 池0144 condition来自36映射各4条，行为固定MT、不神经读取教学，保存242个实际episode endpoint；
所有incoming都是MT。180更新的720事件因此全部在同一MT参数起点学习，虽然E与视频不同。
刷新72 condition来自每task另外2条教学，在180算法下保存120个实际endpoint；其中72个MT incoming、48个编辑后incoming，
57个自身成功条件在停止前完成了最后编辑，实际success经验不再漏掉。

360更新共1440事件/40,320 query，两个180半程均36任务等权各20事件，总各40；180次E屏蔽正好1/8。
后半720事件池0/刷新各360；整个训练1080池0、360刷新。335个独特condition-endpoint被使用，重复使用不是更多独立任务。
仅 **81/1440** 事件使用非MT incoming（后半81/720、刷新81/360）；其中未屏蔽74，覆盖16个映射。
因此“有实际参数接口”不等于已经充分学习递推参数变化。此覆盖使递推分布偏移成为有依据的竞争解释，
但不是唯一根因：formal180的J=1子集161行根本没有前次编辑incoming，仍对MT为R83/G12/L18，净丢6。
不能将全部损失推给递推曝光不足，更不能仅据此机械增加编辑轮数或无限重读。

**实际信用教了什么。** 每事件从同task的7条非teacher episode各四区间取28个native FM query，监督首7动作维、
50-step latent；incoming/outgoing使用同query/noise/time及原生prefix。实际目标为

\[
\frac1{4}\frac1{28}\sum_q\left[L_{FM}(\Lambda_{out};q)
+0.2\max\{0,L_{FM}(\Lambda_{out};q)-\operatorname{stopgrad}L_{FM}(\Lambda_{in};q)\}\right].
\]

真实完整A/B余切通过整链replay回传经验/参数编码、读取、editor和decoder；source、原始E/H、incoming和环境停止梯度。
全四事件归一化由SUM梯度保持，world1→4未改变有效batch或任务权重。新condition没有optimizer、没有拟合自身动作轨迹。
本批没有SDE PG，自己的reward/success在E中是输入，不是一个直接比较这次编辑闭环收益的训练目标。
Query抽样独立于该E的实际失败位置；共享G可以从视频/语言学到一个一般编辑，同时不给不同失败经历足够可区分的修正信用。

配对训练FM确实逐渐改善，不能说完全没学到。八个45-update窗口的平均out-in为
`−0.0000367, −0.0004275, −0.0005479, −0.0004631, −0.0006158, −0.0005892, −0.0007414, −0.0007909`；
最后窗口平均相对差约−0.77%，138/180事件改善、42变差；全程1039改善、397变差、首步4个相同。
但每窗口query/noise/行为混合不同，这不是固定验证集的收敛曲线；keep最后窗口均值0.001084，只约束抽到的expert query。

这条损失在expert分布平均50-step速度误差，部署却在自身到达状态用10-step flow、执行前5动作后重规划，
再经环境转移和新初态判断success。即使expert平均误差下降，也没有数学保证实际visited状态或最早控制步骤保持；
重复编辑又会改变之后的visited分布，而本批没有穿过这些环境转移或完整递推的回报梯度。
由这个明确的目标差距，到train无净收益、J1也有损失、formal丢47条MT能力，存在一条一致的竞争解释；
它不等于已经测出唯一根因，也不允许用一般“FM不等于闭环”句子替代后续实际定位。

**经验和视频的证据范围。** end与null总分相同但配对集合不同，说明真实E条件能影响固定行为，尚未证明净有益贡献。
null从MT按真实链相同次数/停止点递推empty E，不输入真实中间Λ；因链长/停止仍条件于真实链，
它不是独立充分学习的无经验算法。视频没有held同task-other或充分language-only对照，本批不能宣称视频增量成立或不可能。
两个train teacher不同也同时伴随不同实践seed，不是独立的视频因果对照。

本批降低“修复初始identity、漏成功、真实参数入口和事件复用，再靠此跨episode FM便足以学成有用编辑”的支持。
它不否定所有FM、真实反馈或视频编译。旧T161/other150/public103的有限视频正证据、旧shared-SDE161→158及experience135阴性都继续约束取舍。
目前值得main继续推导的是：合法non-held学习如何让不同真实E/Λ对应可区分且能保持已有能力的信用，
以及这条信用如何用上教学的操作知识；直接加RL、继续加轮数或rank/LR/seed小扫都不由当前证据自动推出。
若仅改递推曝光，应预测多次编辑条件的损失优先下降而J1缺口仍存在；若改实际控制分布的信用，须同时解释J1保持和train净收益。
若只改善Long1，要保留其它task退步并判断是否形成总体能力，不能把局部正例拼成完整方法资格。
具体后继设计和投入由main消费原件后自主决定，本执行批不擅自追加科学矩阵。

## 5. 读取、实践、失败和计算的完整费用

下表仅完整条件实践，steps包含settling；尾部是预算终止最后episode的子集，不可再加到失败steps中。

| 阶段 | 条件 | own success/预算停 | steps | resets | 失败steps | 预算尾部steps | 初读+重读 |
|---|---:|---|---:|---:|---:|---:|---|
| MT池0 | 144 | 113/31 | 58,588 | 242 | 40,844 | 8,904 | 0+0 |
| χ180刷新 | 72 | 57/15 | 29,472 | 120 | 20,050 | 4,070 | 72+48 |
| train180 | 16 | 12/4 | 7,969 | 33 | 5,966 | 1,196 | 16+17 |
| train360 | 16 | 12/4 | 8,214 | 34 | 6,276 | 1,196 | 16+18 |
| formal180 | 400 | 220/180 | 248,461 | 916 | 208,590 | 46,030 | 400+516 |
| formal360已发生 | 32 | 20/12 | 22,362 | 49 | 14,938 | 5,928 | 32+17 |
| 合计 | 680 | 434/246 | 375,066 | 1,394 | 296,664 | 67,324 | 536+616 |

池0的242是实际数据endpoint，不是教学轮数。formal180实际J分布1/2/3/4/5为161/110/16/78/35；
这是不同suite horizon与success/1024预算产生的事后结果，没有预设J菜单或“必须重读两轮”。
220个自身success条件final118/220；180个预算停条件final9/180，后者花184,320steps、619全片读取。
formal失败episode占83.95%实践steps；它们与46,030尾部全保留。实践与最终新初态不配对，不能用这些条件分组宣称J或success因果效应。
原首批formal读取1114、本批180为916，但能力从135降到127，不能用少读宣称学习效率更高。

| formal180 task | own success/50 | 实践steps | 失败steps | 全片读取 | final/50 |
|---|---:|---:|---:|---:|---:|
| Spatial3 | 50 | 8,388 | 2,070 | 59 | 39 |
| Spatial6 | 15 | 40,664 | 38,370 | 201 | 0 |
| Object1 | 49 | 13,607 | 5,084 | 67 | 29 |
| Object6 | 25 | 36,131 | 30,820 | 143 | 10 |
| Goal3 | 0 | 51,200 | 51,200 | 200 | 0 |
| Goal6 | 49 | 10,759 | 5,054 | 66 | 34 |
| Long1 | 32 | 36,512 | 24,792 | 80 | 15 |
| Long9 | 0 | 51,200 | 51,200 | 100 | 0 |

Spatial6/Goal3/Long9用掉143,064实践steps和501次全片读取，三个task最终150行全失败；
另三个高own-success任务也并未全保留到新初态。失败成本和既有能力损失须一起解释，不能只减少实践或只看成功条件。

完整条件实践375,066 +实际final189,780 +原失败/部分经验3,450 = worker真实568,296steps；
再加profile4064，整批实际 **572,360环境steps**。failed原件没有因后续成功恢复而被扣除。
共享学习两次完整神经passes/事件共2880全片等价；采集/面板/formal实际1152，train null额外67，合计 **4099**。
缓存命中的教学读取、真实replay也计入；该神经passes口径不包括内部激活重算或可重建native特征计算，后者实际费用仍在GPUh。
训练/推理、保存与失败不只按成功条件分母计费；cumulative condition wall重叠，不能加成批次wall。

63个费用窗口和18份worker回执全部闭合，按`(node,physicalGPU)`并集合并allocation/rank重叠。
实际退出回执修正为 **13.053185 GPUh**；原账本保守为 **13.480595 GPUh**，均包括profile、失败、刷新/360和停止开销。
唯一修正扣除远端collect已经退出、Supervisor SIGSTOP仍保留的0.427410逻辑尾段；原账本不修改。
占卡wall并集 **3.951856h**；首次科学GPU07:47:20UTC至最后退出12:39:08UTC的elapsed **4.863415h**；
最大同时6卡、各节点自身峰GPU01 4/GPU02 3。profile0.465401、collect2.065363、shared train6.216128、eval4.306293GPUh。
这些是生命周期费用，不是CUDA kernel时间积分或高UTL证明。

| 实际共享训练段 | 平均update秒 | queries/s（update wall） |
|---|---:|---:|
| world1更新1–135 | 24.17 | 4.634 |
| world4更新136–180 | 14.31 | 7.825 |
| world4更新181–360 | 21.23 | 5.276 |

不同段经验长度、缓存及共驻负载不同，不宣称受控2.11倍或world4最优。原135完整checkpoint迁移保留optimizer/cursor/逻辑流，
rank0恢复RNG、新rank1/2/3按canonical seed初始化，不称bitwise exact。physical microbatch28选择来自与56/112近等速、
更低峰的真实FM profile；native64/teacher128/经验64/decoder131072及persistent B16均有当前实测，未为填卡扩大科学样本。
formal180实际native平均batch14.43，整轮实践+final367,796steps，2个worker的3.100063GPUh含加载/结束；
CPU env_operation是多个环境并行求和，不能据其10,339.77秒直接计算墙钟占比。
原缓存跨node阻塞已通过host shard/bulk/asynchronous bounded写入修复，保留原卡零UTL失败；不是把整批低UTL自动归因于decoder。
后段实际共有4061秒train360/formal180重叠，刷新/train面板/formal180三者重叠约1080秒；没有让ready后段等待全部180退出。

## 6. 存储、退出和移交边界

当前ROOT allocated **275.60GiB**、logical275.54GiB，超过224GiB登记计划 **51.60GiB**；历史连续峰未知，至少不低于此快照。
data1独立strg01 quota使用1094.54/2048GiB，余量953.46GiB，配额未越界，但不能把配额有余量写成存储计划达标。
原估计145–200GiB没有兑现；实际闭合时才完整分解出evaluation154.384GiB、pools53.955GiB、派生cache63.984GiB。
存储峰估计和边界监控不足保留为执行问题，应由main据真实事实自主调整后继投入与保存安排，不请求Owner逐项许可。
本次没有删除原始经验、失败、实际参数或180/360有效checkpoint来掩盖越计划；
约64GiB可重建host缓存及无恢复用途的临时/冗余状态有明确候选范围，后续按实际生命周期与需求自主处理。
之前11410个旧直接缓存实释放45.3518GiB保留于`analysis/utl_runtime_20261009/retired_cache_manifest.json`，不与当前du相加冒称观测峰。

最终双节点现场未发现本ROOT或ymdai剩余compute GPU进程，科学退出资源已释放；其他用户没有被kill/pause/reset。
本次唯一wakeup在退出后约9.01秒核实真实`item/agentMessage/delta`处理，route=start、无设置覆盖；无需打开session页面。
最终消息只发当前main一次，active Steer或idle/notLoaded保留设置resume/direct start；以`ROOT/main_delivery_receipt.json`实际处理为移交证据。
读取/修复源码ec81→6553→0ec5→db534的有效产物和失败均保留，frozen未原地修改；最终科学/docs由clean pushed main交付。
本批停止不关闭EMBER：main负责消费原件、比较上述竞争解释和局部正负证据，连续推导合理后继并自主接续，
不能把批次阴性、预算取舍或当前未selected变成等待Owner指定下一步的停点。
