# 真实flow监督的历史边界复核

2026-09-26，主讨论在§150后形成新诊断前的只读复核；不是新训练许可。
审计覆盖完整相关设计、实现和原始结果，辅助审计任务为`/root/historical_functional_supervision`。
不能把“此前没有直接检验某个联合接口”写成“该接口就是根因”。

## 既有正负证据

| 历史机制 | 实际监督位置与信用 | 结果与边界 |
| --- | --- | --- |
| SEOD | 成功expert环境轨迹；相同观测/noise下各自真实十步，学生端点动作对expert端点动作MSE；八进度区间选最大动作差 | strict400 129→135→143→136；有收益但相邻未保持。不是在学生中间latent上查询expert velocity |
| GOMQ | 同一SEOD监督分布，从同一LPCP fresh重新打开memory-query梯度 | 151→135→131；151对匹配SEOD135的R/G/L=122/29/13。不能称endpoint监督完全无用 |
| phase decoder | expert成功观测和expert chunk的Gaussian/Beta-time插值；candidate对expert完整50×32 velocity蒸馏 | held5 source21/250、projected44/44、direct74/108；有获取但direct保持不足 |
| phase learner-state aggregation | 30条projected-policy环境rollout、37个expert members、每member八learner states，与成功面板1:1；旧student chunk再与Gaussian插值 | 44/44→54/47，direct保持.311/.278。确实查询过learner环境状态上的expert velocity，但不是学生实际十步访问的中间latent |
| Video Functional | 执行query读视频的可学辅助头，普通离线action/noise插值；不是privileged task expert | reader弱于学生；关闭蒸馏后train45→55/96、validation74→72/400，不是泛化保持修复 |
| Gtrue原生纠正 | 裸Source在固定t1/noise上的真动作cotangent，rank16投影，无student rollout或积分中间态 | state-free full10 MSE .164930→.155748；固定oracle闭环90/384对复用Source68/384，+5.73pp。七task净正，task35贡献11/22净增，Object无获取 |
| 合法Native Correction / Local Field | RGB预测上述t1纠正/局部cotangent，另有跨episode普通FM | validation分别50/51→50/49、48/47→47/50，没有继承oracle的广泛能力 |
| Process Pullback free_q/free_AB | 固定离线支持query上的普通FM局部求解，full10只在训练后读出 | source/原输出/free_q/free_AB=17/20/18/46（/96）；不是十步端点学习。后续L/R出口仍未建立广泛保持 |

不可重命名重试的项目包括：真实十步endpoint loss、成功expert occupancy、learner环境状态＋expert velocity、
执行query读视频的辅助信用、真cotangent直接形成参数、自由完整A/B的局部可达性比较。
SEOD让学生自己的latent参与端点反传；phase aggregation让expert在student环境观测与插值latent上提供velocity。
两者不能合并冒充“当前Writer的实际积分latent上逐步expert指导”已经检验，也不能据此认定该缺口重要。

## 原件入口与完整论证

- SEOD：`8553b613:docs/action_forecast_writer_v6_lpcp_cfmg_successful_expert_occupancy_distillation_design.md:50–94`；
  同commit `src/ember/reward/loss.py:106–134`、`src/ember/writer/reward_preference.py:177–257`。
- GOMQ：`8553b613:docs/action_forecast_writer_v6_lpcp_cfmg_gradient_open_memory_query_design.md:3–6,78–91`；
  `ac233fa0:docs/evidence/gomq_20260823/gomq_cycle2_causal_adjudication.json`。
- phase：`966353e:src/ember/functional_adaptation/functional_response.py:41–86,255–277`；
  `phase_decoder_panels.py:264–337`、`phase_decoder_training.py:510–568`；
  `ac233fa0:docs/evidence/functional_adaptation_20260819/train24_phase_decoder{,_state_aggregation}_held5_20260821.json`。
- [Video Functional设计](../designs/video_functional_writer_design.md)，原件`runs/analysis/video_functional_20260911/paired_summary.json`。
- [Gtrue构造](native_corrective_transfer_audit.md)、[oracle闭环](native_corrective_closed_loop_audit.md)，findings§88–94；
  原件`/data0/user/ymdai/ember_runs/native_correction_writer_20260913/oracle_rollout/decision.json`。
- [Native Correction](../designs/native_correction_writer_design.md)、[Local Field](../designs/local_correction_field_design.md)。
- Pullback：`adc31a15:docs/process_pullback_writer_design.md:201–214`及`scripts/diagnose_pullback_reachability.py:194–227`；
  原件`runs/analysis/process_pullback_writer_20260914/causal_diagnostics/diagnostic_metrics.json`。
- 近期时间轴检查已经测过插值time曲线及真实采样路径；未发现简单A/B相消或仅t1拟合改善，见
  [夜间分析](overnight_root_cause_analysis_20260922.md)。本次不是把“多步与单步不同”重新当发现。

## 当前可复用expert的具体限制

旧train24 step2000的24套adapter均存在，原严格train1200行独立复算为658成功。
当前fit28与旧train24交集13项：2/4/5/7/12/19/22/25/28/29/34/35/37；task17没有对应旧expert。
旧bank根`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`虽名为step1000，
本次查的是各task内`checkpoints/step_00002000/adapter.safetensors`。
原闭环文件`runs/outputs/pi05_task_expert_bank_devtrain24x50_step2000_formal_r3_1362d15_20260808/results.json`。

拟诊断global2/global12的旧成绩为41/50、46/50；不是它们在新面板或student状态上的保证。
旧expert必须与`pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`一起查询；
当前student绑定`pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000`。
两base配置相同、权重不同；共同normalization/tokenizer及评测接口相同。旧训练offset0，当前训练offset1，
不得把旧adapter挂到新base后沿用expert身份。两份基础权重都存在，支持独立加载，无需新训练或复制大资产。
本次只用两个合法train任务，不读held专家/官方Val/Test，不把旧expert作为部署第二adapter。

## 新问题的来源与可否证边界

[Flow Matching](https://arxiv.org/abs/2210.02747)以规定的条件概率路径回归速度场；
[DAgger](https://proceedings.mlr.press/v15/ross11a.html)解释了策略自身诱导的环境状态分布为何有别于离线示范。
这里另外区分同一环境观测下的**去噪latent路径**；下述应用是本项目的待测推论，不是两篇论文替EMBER验证的结论。
普通FM拟合与实际十步控制之间仍有函数、分布、有限积分和闭环四层，不能把其中任一差异自动叫根因。

下一项[冻结flow指导位置干预](../designs/flow_path_intervention_design.md)只检验expert纠正在实际中间latent上是否有用，
以首步纠正、完整expert与原student为对照，同时测两种插值锚及实际路径的场差。
若有益也只支持后续有针对性的学习分布对照，不直接采纳新loss或宣布视频已有效。
