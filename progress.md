# EMBER progress

## 当前状态（2026-09-15）

Owner已要求原因诊断结束后自主决定方法修改、实施、正式训练与测试，无需再次审查；持续goal仍active。
原因诊断已全部完成。当前唯一active design为[Process Pullback共享可学习出口](docs/process_pullback_writer_design.md)。
依据训练侧q／完整A/B的真实闭环差距，给原G P增加共享identity起步的L/R乘法变换；视频过程、7维q、纯FM和数据保持。
旧900视频controls只描述旧模型，不反馈新设计。固定Test只在本轮方法冻结后使用，无梯度／选点／设计反馈。
结果直接在对话交付，不新增用户报告，不开展RL、不自动回退v5.2。

## 正在完成的修改

- 模型及精确信用已在隔离分支实现：`9b131ea0`，30项针对性CPU检查通过，尚待主树集成和完整验证。
  新接口保留同次裸A0/B0，先反传共享出口再用原q伴随与R/Z重放，不重做第二次source编译。
- 主树同步v9/v2配置、训练／物化metadata及本设计；诊断专用拟合入口和static准入已退役，历史冻结树仍保留。
- 正在补最终冻结后的Test correct400准入；只有固定test8、50初态／视频及匹配method freeze record才能执行。
- 新窗口预登记fresh900、3,600条件／230,400queries，300/600/900各validation400和train96。
  终点900预先固定，方法冻结后完成video controls与source/correct Test400；终点读出与能力资格分别裁决。
- 尚未启动新正式训练；先完成集成、最长真实视频两次联合更新与部署profile，再按实时GPU／quota合同启动。

新研究根`runs/analysis/process_pullback_learned_outlet_20260915/`，额外峰值预算32GiB。
本次strg01 data1 quota用量872,330,316KiB、soft1,073,741,824KiB，独立余量约192.08GiB；
旧研究实际26,145,380,221bytes，共享data1约83TiB可用。新研究复用全部source／数据／环境，不复制大资产。

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
