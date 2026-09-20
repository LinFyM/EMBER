# EMBER progress

## 接手续行快照（2026-09-20 18:14 CST）

`main`工作区干净，本地与远端`main`均为`07467977`；两个detached runtime仍由正式任务使用。当前Codex任务已建立继续推进的goal。Source新Validation评测与MT-BC首段50更新在gpu01仍有实际进程；`source_validation.exit=1`是17:07首次准入失败留下的旧文件，当前运行结束后才可据新exit及完整产物裁决。MT-BC控制器正在等待首段完整评测。

Writer 200节点已完成正式correct400，110/400；原200 checkpoint和物化bank用于恢复，没有重训。其控制器续接400时因gpu02:2被其他用户短时占用，在训练前GPU预检退出1；旧exit保留为`*.gpu_contention_initial`。18:13双节点实时预检后，原定gpu02:0,1,2,4满足准入；从`macro_00000200` exact-resume启动400段，正式训练进程已确认。恢复入口为`launch/resume_writer_controller_400.sh`，完成后由同一控制器自动执行完整评测、早停与后续区段。没有启动第二条Writer轨迹；不得将此次工程准入失败或旧Source exit解释为科学结果。

下一次只在完整结束信号或明确工程退出后读取对应exit、正式400行和早停历史；不读取部分分数。新Test及FT/RL仍受既定资格门槛约束。

## 当前状态（2026-09-20，新goal：单次覆盖重训）

Owner已授权采用专家最后的24/8/8＋12辅助方案重训EMBER与MT-BC，Source复用；性能及特异性正常后继续原后继实验。
协议与训练入口接入完成。17:03启动一次性配置profile：gpu02:0,1,2,4 Writer9updates；gpu01:0 MT-BC3updates。两节点tmux与真实训练PID已核对，Writer9步覆盖全部36task，随后完整resume至10通过；均值9.44s/update、peak21.38GiB。配置冻结于1a32a0cf，已提交正式Writer首段200→完整correct400（gpu02:0,1,2,4）；Source新Validation400在gpu01:1共驻重启。Active design：[覆盖重训合同](docs/coverage_retraining_design.md)。任务规格与36个训练HDF5 metadata覆盖审计通过，未读held动作。
coverage_audit已完成36任务规格/metadata审计；训练组件分别完成并集成。CPU集成94项中原fixture缺data字段造成5项失败，修复fixture后45项视频测试通过；另8项协议/早停测试通过。
配置profile与日志：`/data0/user/ymdai/ember_runs/coverage_retraining_20260920`。合计5卡；Writer含低负载共驻卡。完整结束后读结果，等待期间接入评测编排。
新训练无固定2000/600硬终点，仅固定观察间隔；不追加fresh seed或k折；不达预期停下询问。
低负载GPU可共驻，双节点live选卡；不监控代理、不读取部分评测成绩。

### 当前执行与资产

Source新Validation初次提交因旧入口固定32GiB要求在worker启动前拒绝；已按source单worker与materialized policy共用12GiB+2GiB预算修复，在1a32a0cf冻结runtime重启，无已完成rollout被丢弃。

MT-BC profile完整3updates通过，每步36task各16queries，平均149.76s/update、预留33.64GiB。已从clean pushed detached3ebb979b启动fresh首段50→完整400（gpu01:0），正式训练PID已核验；不复用profile权重。Writer已按profile冻结正式首段，Source共驻评测在worker启动前的限制已修复。

两个训练控制器已在各自节点tmux独立运行：`ember-coverage-writer-controller` / `ember-coverage-mtbc-controller`。
首段完成后由`study/launch/continue_training.sh`读取完整400裁决，按原间隔自动exact-resume直到登记早停；不再由主agent逐段读成绩或手工重复launch。
最终完成信号为`ember-coverage-writer-training-complete`（gpu02）和`ember-coverage-mtbc-training-complete`（gpu01）；Source仍为`ember-coverage-source-validation-complete`。
任一失败立即结束对应控制器，exit/log留在study/launch。每段live选卡合同不变；已评测且继续训练的Writer bank仅退役可再生载荷，manifest/rows/完整checkpoint保留。

本轮已清理400个已结束旧Test物化载荷及18个完成profile参数文件，释放实际占块2,489,434,112字节；完整清单在新study/asset_retirement.json。全部旧formal checkpoint、raw rows及profile合同/指标/恢复证据保留。


Test other/wrong/shuffle/reverse接口已通过canonical编译/评测路径开放，需同method_freeze及paired_correct_manifest。两处旧fixture补data字段后，horizon/video-controls合计148项CPU检查通过；尚未授权越过性能门槛运行Test controls。

Writer首段200及correct400物化完成后，最初shell launcher退出2，未启动评测；其运行期间曾原地改写launcher，疑似解析位置受影响，原stderr未保留，不能确证。
已保存`writer_first_stage_launch_failure.json`，使用独立不可变`recover_writer_evaluation.sh`仅启动该bank的完整评测，不重训或重新物化。后续脚本变更采用原子替换，现有进程继续读原inode；阶段恢复通过后再接回控制器。

## 前阶段已封存：旧Test未达门槛

冻结Source1000 / MT-BC425 / EMBER1500的Test8各400已完成：78 / 74 / 82。
EMBER只领先MT-BC8条（2pp），低于Owner至少40条的推进门槛；任务簇bootstrap95%CI为[-8.51,11.25]pp。
1200行、18个worker退出、共同source/normalization、逐行state/RNG与50条teacher无放回映射全部核对通过。
Owner已选择先看报告并与专家讨论；controls、FT、RL及外部比较全部停止，等待新指示，不重选模型。
[图文报告](docs/review_materials/20260920/test_capacity/report.md)附逐任务、suite、breadth、配对成功集合与CSV原始行。
Active design登记[paper_experiments_design](docs/paper_experiments_design.md)，当前仅为已冻结待裁决合同，不授权自动进入下一阶段。
Study：runs/analysis/paper_experiments_20260920；正式结果、manifest及readout.py保留；400个旧物化LoRA载荷已在新训练启动后退役，可从保留的formal1500 checkpoint重新生成。
首次nohup提交未存活；tmux重新提交后正常完成。EMBER原定gpu02:3忙，物化前拒绝后改gpu02:0,1，科学合同不变。
证据见study/initial_launch_failure.json、allocation_retry.json及每组launcher_completion.json。
临时detached runtime 45a39ba1已在完成后删除；无本轮分支。原checkpoint与数据不动，未为本轮创建新训练checkpoint。
后续等待采用完成信号，不读中间成绩，不使用监控代理。24任务Test入口的13项Source-SFT与45项Writer tests已通过。

## 上轮已封存结果（2026-09-20）

Owner授权的专家最终修订、主候选、唯一匹配消融、后续趋势、消融视频特异性和条件性双相机全部完成。
该轮没有剩余作业；[合同](docs/video_teaching_writer_design.md)已封存。
先读[专家讨论总报告](docs/review_materials/20260919/final_report.md)与[可复制提示词](docs/review_materials/20260919/expert_discussion_prompt.md)。

| 更新 | 同视频 correct /400 | 消融 correct /400 | 双相机 correct /400 |
| ---: | ---: | ---: | ---: |
| 900 | 149 | 125 | 117 |
| 1200 | 174 | 136 | 108 |
| 1500 | 165 | 147 | 108 |
| 1800 | 160 | 136 | 未运行 |
| 2100 | 158 | 159 | 未运行 |

原1500相邻qualification选定主checkpoint后保持冻结；1200的174不作selected结果，追加窗口不重选。
主组1500→2100 correct165→158、other165→156；消融147→159、other155→161。
主组原窗口的正确能力增量未保持到末点，不证明消融更好或两者等价；[续训报告](docs/review_materials/20260919/continuation_report.md)。
固定1500的主组correct/other/wrong/shuffle/reverse/source为165/165/124/113/113/50，消融147/155/122/104/86/50。
两组都有顺序敏感性，但六个匹配DID区间均跨零，尚未证明教学项增强视频特异性；[视频检查](docs/review_materials/20260919/ablation_controls_report.md)。
双相机fixed1500 other105，correct/other相对单相机均下降；train52/64/67及FM下降未转为验证能力；[相机对照](docs/review_materials/20260919/dual_camera_report.md)。
2100和双相机没有额外controls，不能继承旧因果结论。无Test、RL、融合、挑视频或旧A/v5.2重训。

全部40个正式面板、12,048条新闭环、5700实际更新及638,400主＋辅助queries核对通过，共同source与续训父历史只计一次。
保留57个唯一arm/step完整checkpoint、run contract、manifest、raw rows和worker日志；[完成清单](docs/review_materials/20260919/completion.json)。
双相机按真实四卡profile采用world4/frame8/microbatch16，训练与物化／闭环流水并行，末轮三面板在两节点六卡并行。
全部训练/driver/正式workers正常退出；08:50评测收齐，08:51两节点核验无本轮GPU或训练／物化／评测作业。
source始终冻结，完整resume、6000/8400事件、全池teacher映射、逐行state/RNG和真实RGB变换核对通过。
实现阶段CPU检查主教学图累计246项、相机相关222项通过（覆盖有重叠）；后续分析只读取已有证据，不新增模型forward。

正式主/消融/续训/双相机的冻结commit为39c3919c/bd497edc/d1474ce0/35124aa9；临时runtime树已移除，可按commit重建。
本地原件统一在`runs/analysis/video_teaching_20260919/`，远程保留方法、配对CSV/JSON、曲线、输入示例和报告。
findings§121–122与[研究历史](docs/research_history.md)保存完整解释及边界；后继选择供Owner与专家讨论，需新的明确授权。

## 历史证据入口

[A的learned frame-set报告](docs/review_materials/20260918/frameset_report.md)保留上轮匹配诊断；
[46组证据审计](docs/v52_evidence_audit_20260917.md)、[findings](findings.md)§117–122及
[研究历史](docs/research_history.md)索引统一Writer、A900机制、Core/Procedure交叉与更早实验。
历史结果、旧源码与配置均按各自封存口径解释，不恢复已结束路线。
