# rank8多起点完整Writer首批：教师供给已完成，G/π继续执行

当前已完成16个教师事件、每事件160/480两完整checkpoint及selection4/audit16，32个完整checkpoint和960原始行。
全部7,680实际更新、860,160 demo queries已从完成的metrics逐条核对无缺失/重复；rec/keep实际训练query分别3,840/7,680。
这些是共享Writer的训练端点供给，尚不构成G、视频理解或held任务能力证据。本报告将在整批G/π与规定读出完成后补齐。

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

## 身份、费用、恢复与后继执行

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
G240后同拓扑/完整optimizer+sampler+rank RNG续480，随后固定ψ开展π及全部规定报告。尚未完成范围不写为成功。

## 可复核原件

ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261011/rank8_multistart`。

- `teacher_summary.json`：全部16事件、q_T、参数/函数差异、960原始行及完整账目。
- `analysis/teacher_paired_rows.jsonl`：640显式selection/audit parent—节点配对。
- `analysis/teacher_bank_aggregate.json`、`teacher_learning_coverage.json`：每task/parent类型与实际更新/曝光。
- `events/*/teacher/checkpoints/update_{160,480}`、`events/*/readout`：完整恢复与原始读出；manifest保留各真实来源。
- `bank_scale.json`、`coordinates.json`、`initial.safetensors`：固定坐标、真实Λ0与32端点权重。
- `analysis/G240_plan.json`、`jobs/G240/attempt_001/launch.json`：精确命令、每卡live/quota、取消硬限与物理配置。
