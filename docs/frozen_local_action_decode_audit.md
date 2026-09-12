# 冻结局部动作读出的生成能力诊断

2026-09-13，新增outcome前登记；本只读诊断已完整结束并关闭，结果见末节，旧局部训练不恢复。
依据[视频信息复核](video_information_identifiability.md)，本项检验一个具体提案的前提：能否把旧局部读出已经学到的
动作知识直接作为参数生成的中间量。没有提出或启动这种新Writer；先判断其现成知识是否足够支持继续推导。

## 问题与竞争预测

旧local FM在step200为ordered .810114、frame_set .811286，小幅差额在24/24任务为正。
但预测器输入包含带噪真实动作；这些FM分数不等于只看视频就能输出准确动作。

- 如果冻结局部读出已经有可直接利用的动作知识，从纯噪声生成的动作均值应在新动作episode上胜过
  训练池逐task动作均值，而且ordered应优于匹配frame_set。
- 如果小幅有序FM优势不足以形成这一知识，生成误差可能不优于上述参照。此时停止把这个现成局部头接入
  Compiler的提案依据；不继续训练／调步数／换seed来追逐局部分数。
- 若前提成立，仍不等于Compiler已解决、完整过程已学会或新Writer会有效，只支持进一步推导显式中间量的绑定方式。
  不自动开完整训练、不选checkpoint、不完成goal。

这是对固定模型的有界生成能力检查，不穷尽所有局部反演函数类，也不通过破坏视频制造差额。
独立训练的ordered/frame_set差额仍不等于同一模型的顺序因果贡献。

## 固定对象与数据

- 模型固定为`local_action_grounded_20260912/{ordered,frame_set}/checkpoints/macro_00000200`，
  原运行面为clean pushed `5f4f440c992e470aaffcc873c847373ce6df43e5`。不比较其它checkpoint或恢复optimizer。
- 完全复用原train24、diagnostic42–45、每task16个四帧片段、seed20260913及每片段的既有noise seed。
  每臂384片段。真实RGB四帧为p/p+5/p+10/p+15，评估标签为actions[p+1:p+16]。
- 生成器只收到真实RGB、exact language和独立Gaussian噪声。所有标签只在输出后用于评分；
  不向observer/reader输入teacher state、真实动作、任务编号或目标谓词。无validation/test动作、梯度或环境交互。
- 正式step200的完整LoRA和闭环分数均不改变。本项不输出部署LoRA，没有模型选择或新的最终controls。

## 两项互补检查

1. **CPU解析参照。** 只用每task的action16–41，以episode等权、episode内全部合法15步片段等权计算
   105维动作均值μ与协方差Σ。按原8个噪声／time样本，计算Gaussian动作边缘分布的解析FM预测。
   若x=tε+(1−t)a，v=ε−a，其预测为
   `−μ + [tI−(1−t)Σ][t²I+(1−t)²Σ]⁻¹[x−(1−t)μ]`。
   只将浮点舍入导致的负特征值截至零，不拟合／扫描正则项。这是带task身份的离线诊断参照，不能作为部署task字典。
   在原384片段上与已经存在的两臂FM逐行配对；重建step0零输出loss核对标签、归一化与随机采样的口径。
   该参照可以暴露FM的解释范围，但其胜负不单独决定生成能力，仍执行下一项。
2. **冻结生成。** 复用原observer、encoder、local reader及其独立时间路由。从每片段8个Gaussian样本出发，
   使用固定10步Euler，t=1,.9,…,.1，`x←x−.1v`；无真实动作加噪输入，无迭代拟合。
   8次生成取均值作为条件动作估计，同时保留每次完整输出。这里平均的是噪声采样的诊断动作估计，
   不平均视频、frames或LoRA，不将此头接入机器人控制。

主指标为冻结source归一化单位下15×7动作均方误差；先按16clip平均，再24task等权。
预登记两项差额：task-mean误差减ordered误差、frame_set误差减ordered误差，正数表示ordered更好。
使用task-cluster配对bootstrap20,000次、seed20260913、双侧percentile95%区间；两个下界均>0且至少两个suite
有ordered对frame_set正差额，才认为本固定读出通过继续推导前提。否则停止该现成动作中间量提案。
报告全部task/suite、单样本误差、xyz与gripper误差；不由次要指标或有利子集推翻主判断。

## 执行与停止

只运行上述两臂step200，没有后续训练、步数／dtype／seed扫描或动作解码器再拟合。
直接复用原资产；新入口来自clean pushed detached worktree，旧模型运行面保持原commit不变。
CPU脚本及GPU两臂共用一个有限诊断入口`diagnose_local_action_reader.py`，不接入canonical Writer runtime。
完成后从active scripts退役，代码由冻结Git与本项证据保留。
原始数据、预测、统计与launch记录保存在既有`runs/analysis/local_action_grounded_20260912/inverse_action_*`。

2026-09-13启动前存储快照：strg01的/data1独立quota为966.2/1024 GiB，旧运行面218MiB、现有analysis2.5MiB；
共享可用83TiB。新增冻结入口运行面约218MiB、结果预计<20MiB，预留总峰值1GiB，不复制模型或dataset。
GPU启动前实时查两节点；最多两张真正执行任务的A40，每臂一张、无NCCL，CPU与GPU选择不改变task权重。
模型载入与4帧native读取预计每臂数分钟；出现非finite、输入合同或运行失败先定位具体工程问题，不扩展科学面板。
数值精度沿用正常BF16/FP32，不要求逐bit复现。首轮已有零输出loss对照足以检查评分口径，不新增防御性hash或全树扫描。

## 完整结果与裁决

入口冻结提交`017430b3`，历史模型运行面仍为`5f4f440c`；两臂各384片段完成、exit0，
每臂诊断循环约98.3秒，峰值allocated10.155GiB。CPU矩估计使用98,465个合法训练窗口、624个episode，
按原定义episode等权；384个step0零输出loss重建通过，固定片段、标签和噪声映射核对通过。

| 指标 | ordered | frame_set | task mean |
| --- | ---: | ---: | ---: |
| 8次生成均值的动作MSE | .31625053 | .31676502 | .25059744 |
| 单次生成动作MSE（8次平均） | .65159739 | .65250543 | — |

task mean−ordered为−.06565309，task-cluster95%CI[−.08381484,−.05098097]，24/24任务均不如task mean；
frame_set−ordered为+.00051449，95%CI[+.00032845,+.00070563]，21task正／3负，四suite总体同向。
原有序小幅优势延续到实际生成误差，但未达到两个登记参照同时通过的前提，**本诊断non-pass**。
不以这个相对正差额重开旧方法，也不追加noise数／积分步数／训练来改变裁决。

CPU的无视频task Gaussian FM=.26389586，两臂原FM=.81011446/.81128639；24/24任务解析参照均更低。
该参照使用noisy action与训练task身份，不能作为部署字典，也不是video-free闭环策略；
其作用是说明绝对局部FM仍明显落后一个简单动作边缘去噪器，不能仅凭有序差额把局部读出称为充分功能教师。

停止的是“把此现成局部动作估计直接作为参数生成中间量”的前提。固定8样本均值的MSE包含采样误差，
这项失败不证明所有逆动力学不可能、不排除表示含有部分有用知识，也不能把失败唯一归到E或Compiler。
两个解析参照使用全部授权action训练池，标签曝光不与仅800个局部训练片段匹配；因此它们检验现成输出的可用前提，
不构成同训练预算的算法比较，也不能唯一归因于模型结构或视频信息不足。
不创建新的完整Writer、selected checkpoint或最终controls；整体goal未完成。

原始预测和标签、逐task/suite、两项配对区间、完整原始行与日志保留在既有analysis根的
`inverse_action_READOUT.md`、`inverse_action_summary.json`、`inverse_action_marginal.json`、
`inverse_action_{ordered,frame_set}.json`、对应samples.npz及`inverse_action_launch_contract.json`。
有限入口从active scripts退役，源码保留于`017430b3:scripts/diagnose_local_action_reader.py`及其冻结运行面，
不恢复局部头的canonical训练实现。
