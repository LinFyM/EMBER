# EMBER progress

## 当前状态（2026-09-15）

Owner已要求原因诊断结束后自主决定方法修改、实施、正式训练与测试，无需再次审查；持续goal仍active。
原因诊断已全部完成。当前唯一active design为[Process Pullback共享可学习出口](docs/process_pullback_writer_design.md)。
依据训练侧q／完整A/B的真实闭环差距，给原G P增加共享identity起步的L/R乘法变换；视频过程、7维q、纯FM和数据保持。
旧900视频controls只描述旧模型，不反馈新设计。固定Test只在本轮方法冻结后使用，无梯度／选点／设计反馈。
结果直接在对话交付，不新增用户报告，不开展RL、不自动回退v5.2。

## 新窗口执行

- 共享出口与精确信用已集成：模型`e1ca29b9`、v2 bank fixture`6dffa85c`、冻结Test准入`6e3f9953`。
  同次裸A0/B0先反传共享出口，再用原q伴随及R/Z重放，不重做第二次source编译。
- 核心直接autograd等30项检查通过；主树旧fixture已同步，相关bank/controls/workers102项通过。
  Test分支125项检查通过，集成后相关bank和controls的121项检查通过。
- 真实最长task38/demo36、105帧、完整50-horizon两次联合更新正常结束；第二步24.283s，
  allocated峰值37.049GiB、reserved37.770GiB；Writer／两Meta／共享出口均获得有效finite信用，source无可训练参数。
  单次部署编译完整38-target耗时7.850s、无loss/optimizer；profile初始化已丢弃。原件在新研究`profile/actual/`。
- Fresh900预登记3,600条件／230,400queries，300/600/900各validation400和train96。
  终点900预先固定，方法冻结后完整video controls及source/correct Test400；资格与终点描述分别裁决。
- 正式训练已从clean pushed detached `85ecfd18` 在gpu01:0–3启动，首两次全局更新18.850／21.974s；
  第二步Writer及两Meta均有有效信用。首个300节点已正常结束，墙钟5,611.741s，完整保存100／200／300。
  实际allocated峰值37.615GiB、reserved43.740GiB。该节点validation400／train96及完整配对核验均已结束。
  Train为21/96、breadth9/24、相对source R/G/L=14/7/3；validation为50/400、breadth4/8、R/G/L=31/19/16，
  S/O/G/L=0/16/32/2。Validation净增的task-cluster95%区间[-7,+8.5]pp，当前未证明共享出口改善泛化。
  独立训练动作FM从.153279524变为.148711546；不把loss或本次局部结果当作能力通过。
  两套评测均正常结束，496套条件、原始rows及所有workers完成证据保留。
- 600段训练正常结束，累计2,400条件／153,600queries，段墙钟5,123.126s；400／500／600完整checkpoint保留。
  新生成496套条件后，validation400与train96及全部18 workers均正常结束，完整配对核验通过。
  Train20/96、breadth8/24、source R/G/L=13/7/4；validation75/400、breadth3/8、source R/G/L=37/38/10，
  S/O/G/L=0/38/34/3。75次成功只来自Object1、Goal6及Long2；相对source净增区间[-4.5,+25.25]pp，仍未建立广泛迁移。
  Validation300→600 R/G/L=39/36/11、churn47、Jaccard .4535；train相邻13/7/8、churn15、Jaccard .4643。
  独立动作FM为.144942572；599次identity后更新的Writer／两Meta信用均finite非零。实际闭环能力与覆盖仍优先于loss。
  已从600按原四rank topology、优化器／scheduler／sampler／rank RNG exact-resume启动900段，不按中间分数选峰值。
- 首次启动在配置状态标签检查处退出，未创建训练runtime或更新参数；登记字段修正后重新启动。
  原失败日志与commit保留于新研究`failed_attempts/initial_registration/`，不混入正式曝光。

新研究根`runs/analysis/process_pullback_learned_outlet_20260915/`，额外峰值预算32GiB。
续训900前strg01 data1用量878,010,872KiB、soft1,073,741,824KiB；新研究5,810,292,265bytes，
额外32GiB峰值仍在独立quota内，共享data1约83TiB可用。全部source／数据／环境复用，不复制大资产。
实际拓扑、命令与完成状态以新研究launch contract为准。

## 已关闭的固定出口900窗口

原训练d6defa69、物化／评测f5d9db78，fresh900完整结束；九份checkpoint、六面板1,488条rows保留。

| 节点 | Train /96 | Train breadth /24 | Validation /400 | Validation breadth /8 |
| --- | ---: | ---: | ---: | ---: |
| Source | 17 | 7 | 47 | 3 |
| 300 | 24 | 8 | 64 | 3 |
| 600 | 22 | 8 | 72 | 4 |
| 900 | 26 | 9 | 64 | 4 |

Validation相邻R/G/L为56/16/8、52/12/20，Jaccard .7000/.6190；最终Long归零，三个节点均未覆盖四suite。
实际623/624教学条件曝光；梯度及运行合同有效，属于有局部获取但未获广泛迁移保持的科学non-pass。
不原样追加训练或小扫。完整原件在`runs/analysis/process_pullback_writer_20260914/`，历史与限定结论见findings§100。

## 已完成的原因诊断

完整原件为上述根的`causal_diagnostics/`，预登记473cec18、冻结实现adc31a15，新完成2,272条闭环。

- 全train24共同teacher16／900起点：source17、原输出20、free_q18、free_AB46（均/96）；free_AB breadth19、
  S/O/G/L11/17/13/5，对free_q差额CI[+13.54,+44.79]pp。独立full10误差为.171922/.150169/.140512。
  有界拟合全部有效，不证明数学上界或共享RGB已可学，不复用为新模型初始化。
- 训练池teacher16–19的300/600/900为18/20/22，独立teacher46–49为24/22/26；没有一致训练视频优势。
- 固定900 correct/other/wrong/shuffled/reversed/no-video为64/65/48/35/67/47（均/400）。
  Correct未优于倒序；乱序差额的跨task区间仍跨零。不能宣称有益方向特异性成立。
- 32组／96条correct回放全部实际查看，95/96复现原成败；唯一差异Long2/state0/600原成功、本次失败，不改原分数。
  对象／实例选择、获取、放置和组合子目标推进均有真实失败，不能统一归为后半段遗忘。
- Other官方400及12 workers完整正常，但外层脚本运行中被改写造成收尾127及console log覆盖；原始结果不受影响。
  事故记录保留，不重跑挑结果。其余诊断正常结束，当前没有本轮旧诊断GPU任务。

原正式证据、唯一checkpoint及冻结运行树保留；诊断专用运行代码由Git adc31a15保存并从main退役。
当前任务计划见[task_plan](task_plan.md)，稳定要求见[owner requirements](docs/current_owner_requirements.md)，历史见[research_history](docs/research_history.md)。
