# rank8多起点实例收口：教师供给与G优化原件

当前已完成16个教师事件、每事件160/480两完整checkpoint及selection4/audit16，32个完整checkpoint和960原始行。
全部7,680实际更新、860,160 demo queries已从完成的metrics逐条核对无缺失/重复；rec/keep实际训练query分别3,840/7,680。
主讨论已修订科学取舍：已运行G480自然保存后收口，π与全部规定报告/controls不再启动。
逐H重训Q的必要性没有得到比较证据；本bank保留为旧实验，不重标为新强教师合同。
教师端点有own-parent控制增益，G240有真实优化记录；两者尚未建立生成LoRA的闭环能力或视频增量。当前G480仍在原冻结版本运行，最终退出/费用在实际完成后补齐。

## 教师控制增益与不利个例

所有事件相对自己的真实parent配对，audit共256条件：parent84→T160116→T480139。
T480相对parent retained70/gained69/lost14/churn83；相邻160→480 retained96/gained43/lost20/churn63。
parent包含MT和seed160/480，不将合计84误写成静态MT分数。

|task|own-parent/64|T160/64|T480/64|T480 gained/lost|
|---|---:|---:|---:|---:|
|12|32|43|50|18/0|
|29|12|15|20|13/5|
|32|30|46|54|26/2|
|38|10|12|15|12/7|

按seed/mid/late/return四类，各64配对的parent→480分别11→31、30→38、34→39、9→31。
seed38为3→2，late32为14→14且gained1/lost1；task38总体改善仍小、相邻与parent交换较多。
不把这些阴性或持平事件重采、不按audit筛除端点。所有32端点均合法，selection规则给各事件两节点各0.5；
已登记“全bank合法节点均无own-parent audit净增”停止线未触发，按原完整合同进入G。

## 真实训练与函数/参数作用

8个真实恢复标签、64个成功parent功能标签来自实际捕获/恢复，keep仅采同parent执行成功的记录。
return仍为MT parent与seed160实际产生的H，未把seed160成功计作MT keep。
三类实际parent为MT、seed160、seed480；两条H分别来自独立完整episode，不是超过1024步后拼成还能继续的H_mix。
这覆盖有限教师分布的单episode历史，未覆盖任意G生成parent或多producer深H。

各事件完整480更新的均值9.849–19.701秒；λ的A/B实际改变及S_prox幅度见原件逐事件数字。
T480对own-parent的首个native前5动作RMSE为0.012654–0.214048，比较用相同已登记初态/环境/policy及实际首noise。
这里只读取原生query已经产生的函数原件，没有新增forward；首个动作片段差异不能替代完整闭环或视频因果证据。
S_G从完整16事件、task¼/event¼/selection q_T形成一次固定：76尺度、32加权端点，floor_active=0。
审计仅消费停止线，没有进入S_G或q_T数值。

## 身份、费用、恢复与科学收口

教师来源b73d15c8/fdcee632按各launch/manifest保留；首次32-renderer工程失败、16-renderer恢复及所有费用均保留。
教师整批结束累计53.698015811747304GPUh，含carry-in18.640412103864882；本ROOT35.05760370788242、53回执含1失败。
当前新增ROOT约1.4GiB、原ROOT约4.1GiB；独立data1 quota足以容纳当前64GiB合同，planned total peak约32GiB。
Owner已取消累计GPU小时硬门槛，96仅保留为历史计划；成本继续准确计账/预测。

G240已从clean pushed detached02781720正式fresh启动：gpu01四worker物理1/6/0/2，逻辑B4/task等权/event¼、chunk16。
当前0/2新空闲，较长32/38任务映射这两卡；3/4显存只剩2341MiB，未沿用此前比较卡组。四卡现场余量均覆盖此前84帧meta-reader chunk16实测峰值。完整G240的allocator高水位未保存，不能把reader峰值称为完整更新峰值。
完整B4两/四卡实测47.432790/34.562157秒仍是选择依据；当前更大chunk与新卡组的收益须由正式G240完整节点核验。
后继既有metrics及local/report完成字段补process_GPU_peak，用allocator现有高水位计数，不新增forward/同步/profile；
G240冻结源码保持，后续G/π消费者承接后再报告实际峰值。
4worker只承担4个独立task-loss分片，额外rank在现有消费者无额外task计算可并行，未为卡数本身铺新计算。
G240后同拓扑/完整optimizer+sampler+rank RNG续480。当前wrapper只派发G480，不含π/报告自动晋级；
480完整checkpoint后按main新取舍结束，旧完整§13范围中的π/报告未执行。

## 可复核原件

ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261011/rank8_multistart`。

- `teacher_summary.json`：全部16事件、q_T、参数/函数差异、960原始行及完整账目。
- `analysis/teacher_paired_rows.jsonl`：640显式selection/audit parent—节点配对。
- `analysis/teacher_bank_aggregate.json`、`teacher_learning_coverage.json`：每task/parent类型与实际更新/曝光。
- `events/*/teacher/checkpoints/update_{160,480}`、`events/*/readout`：完整恢复与原始读出；manifest保留各真实来源。
- `bank_scale.json`、`coordinates.json`、`initial.safetensors`：固定坐标、真实Λ0与32端点权重。
- `analysis/G240_plan.json`、`jobs/G240/attempt_001/launch.json`：精确命令、每卡live/quota、取消硬限与物理配置。

## G240学习覆盖与实际吞吐

240次逻辑B4更新、每task240个机会，四类事件实际抽样分别：12为66/63/54/57，29为53/62/75/50，
32为70/73/50/47，38为62/56/71/51；登记为等机会随机流，未改为按结果重采。没有缺标签机会。
完整checkpoint有486 optimizer states及4rank RNG，全部source gradient记录为false。两组meta各960记录非零梯度；
G各Reader/static/field与meta参数相对fresh初值实际改变，逐模块数字在`analysis/G240_consumer.json`。
这些证据支持真实优化发生，不能由此认定视频/经验知识已学会或G输出能控制；尚未生成G240实践探针。

该阶段实际wall3750.377903秒/4.167086559GPUh，完整B4 warmed15.458848秒、末50为14.792761秒。
与此前34.562157秒比较同时涉及实际event/坐标/H分布、空闲卡和chunk16，不能把差额全归因某个优化。
G240结算累计57.86510237114892GPUh；G480启动时预测1–1.25h/4–5GPUh。原π/报告剩余5–10h预测已被科学收口覆盖，未成为实际费用。
G480从新clean pushed detachedc12c66d8同四卡接续240的完整optimizer/sampler/rank RNG，统计修订不改计算。

## 主讨论撤回的数据组织主张与未执行范围

32个节点来自每task四事件各fresh480；G实际parent仅MT/seed160/seed480，分支教师只是监督端点。
全部q_T为160/480各0.5，独立audit的退步节点仍在合法分布内。当前正例只说明部分Q优于自身parent，
没有证明逐H新拟合Q优于同task一条训练轨迹保存节点、经能力实测选择共同强Q。固定Q也可以承接各parent真实实践的恢复监督。
main已撤回“单轨迹起点比本bank更集中”与“固定目标无法保留实践监督”的概括，并负责重新登记后继。

本实例从未启动π-local的数据路径/128更新、RL16、两个48行报告、完整U或same-task-other/wrong视频干预。
不存在G候选的实际执行/首5函数读出，也没有新的Val/Test/400或视频因果结论。G直接动作辅助loss仍是未定候选，没有实现/运行。
本次收口是科学组织修订，不是G能力阴性、GPUh额度停机或工程失败；不删旧bank、不重跑教师、不自行增加新方案。
