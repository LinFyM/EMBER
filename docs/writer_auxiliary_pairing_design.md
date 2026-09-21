# Active design：36任务 Writer 辅助 episode 配对 fresh 对照（2026-09-22）

状态：formal preflight已登记、尚未启动训练、物化或闭环。study根为
`/data0/user/ymdai/ember_runs/coverage_retraining_cross_episode_aux_20260922`，run contract为
`launch/launch_contract.json`；runtime固定为clean pushed detached `64947492`。2026-09-22 03:10 CST的双节点快照中，
gpu01无安全训练卡；gpu02:0--3均为低util、已知owner `gqma`的小显存共驻，最小空闲45858 MiB，按旧Writer
22.803 GiB峰值登记四卡frame8。data0/data1独立quota和data0共享余量均通过16 GiB峰值预算。当前窗口的训练上限
预注册为correct400完整节点1200；首次原早停记录独立冻结，之后的post-stop extension节点允许参与单独的extended-run
最终选点，均不得根据中间分数补改。原same-video、旧24任务消融与低LR修复的配置和结果保持原样。当前实现仅开放本文件
登记的一个fresh cross-episode候选。

## 两个损失的精确定义

在本候选中，两项动作监督都来自teacher以外的同任务episode，标签都会经过同一套生成LoRA。两项监督的不同在于：

| 项目 | 主FM | 跨episode端点前缀辅助 |
|---|---|---|
| 每个task条件的查询数 | 21 | 7 |
| 查询episode | 按既有轮转算法，逐query排除teacher | 先均匀抽一条非teacher episode，7条query都来自它 |
| 查询位置 | 原合法观察位置采样 | stride5合法位置，后续5个真实动作 |
| flow time | 原Beta时间采样 | 固定tau=1 |
| 带噪动作 | x_tau=tau*epsilon+(1-tau)*actions | x_1=epsilon，独立纯噪声 |
| 监督 | 全50个horizon位置的真实7维动作速度误差 | 仅前5个horizon位置的真实7维动作速度误差 |
| task内系数 | 1 | 1/3 |

记速度目标为epsilon-actions，两项都计算速度MSE。tau=1时，前缀误差也可写成epsilon-v_theta与动作标签的MSE。
该表达式来自单个端点forward；部署仍通过10步flow求解输出动作，不能把epsilon-v_theta当成已完成10步求解的动作。

主FM已经含有前5步和高噪声区间的训练。辅助项对纯噪声端点和实际执行前缀追加权重，双方监督存在重叠。
它不引入新的标签种类，也不保证得到更强的视频理解。本次保留它，是为了单独检验同视频/跨episode配对差异。
同时删除端点前缀项会改变监督数量、时间分布和horizon权重，属于另一个未获本计划授权的实验。

## 已提供的代码

- `configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json`：可供现有trainer读取的候选配置。
- `src/ember/writer/training.py`：显式声明新变体后才允许36任务cross_episode；旧默认路径保持。
- `src/ember/ecp/checkpoint.py`：默认同拓扑full-state恢复；仅ordinary dynamic resume的显式opt-in可以转换物理world size。
- `src/ember/writer/auxiliary_pairing.py`：严格配置匹配、生产采样器的真实元数据重放，以及原same-video曝光核验。
- `scripts/writer_aux_pairing.py`：唯一新增CPU入口，只执行上述preflight。
- `tests/test_writer_training.py`和`tests/test_writer_data.py`：分别拥有opt-in配置与原采样器/辅助查询的回归验证。

没有改动模型、原生forward、损失实现、数据角色或采样算法。日志字段`teaching_loss`为向后兼容继续保留；其episode语义由`data.teaching_episode`和真实事件记录确定。

## 已登记的执行顺序

1. PR #3已收敛、合入并推送；所有平行status/selection入口已删除。当前canonical代码仍须在实际launch前以clean pushed detached worktree固定。
2. 已在`progress.md`和`task_plan.md`登记本次单路fresh候选、状态与本设计；保留所有旧实验，且不把CPU验证写成GPU资格。
3. 使用canonical资产根和原coverage study，运行CPU preflight。原Source/数据不复制。
4. 沿用现有coverage分段controller、NUMA/torchrun包装器、materializer和官方evaluator。controller继续拥有完整面板读取、`validation_decision`、同分other选点、冻结声明和测量分支；本设计不保留第二套status/selection/finalize代码。
5. 每200更新完成correct400后，controller只在训练、物化、评测都完整退出且400行原件齐全时读取结果，再按既有选点合同进入下一段、同分other或预登记测量。严格保留原资源上限与每次launch检查。

```bash
# 使用仓库既有Python环境；新断言归入现有所有者。
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_writer_training.py tests/test_writer_data.py

# ASSET_ROOT、REFERENCE_STUDY、STUDY由canonical资产解析；不要填到Git配置里。
PYTHONPATH=src .venv/bin/python scripts/writer_aux_pairing.py \
  --asset-root "$ASSET_ROOT" \
  --reference-exposures "$REFERENCE_STUDY/training/writer/exposures.jsonl" \
  --output "$STUDY/launch/preflight.json"
```

训练仍调用`scripts/train_writer.py`，由原四卡/NUMA包装器传入：

```text
--config <CODE_ROOT>/configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json
--asset-root <ASSET_ROOT>
--output <STUDY>/training/writer
--mode formal
--stop-after-step 200
```

后续区段保持配置，增加`--resume <STUDY>/training/writer/checkpoints/macro_00000200`并将stop推进至400，依此类推。
首次fresh禁止`--extend-from`或`--phase-from`。默认恢复要求同物理拓扑；仅ordinary dynamic resume可显式附加
`--allow-topology-change`，从完整候选checkpoint恢复模型、optimizer、scheduler、sampler与immutable events，同时登记
`topology_transitions.jsonl`。它不适用于continuation或low-LR phase，且不承诺随机轨迹或低位浮点bitwise相同。
配置中的profile沿用已登记的同架构四卡观测；首次formal launch仍须使用world4并检查实际资源与首次更新。

物化/评测输出目录沿用：

```text
<STUDY>/materialized/writer_00000200_correct/manifest.json
<STUDY>/evaluation/writer_00000200/{results.json,run_contract.json,launcher_completion.json}
# other分别使用 _other 后缀。
```

controller保留原有的完整性检查：400个唯一task/state行、worker全0退出、materialization manifest、固定视频映射与checkpoint/run-contract一致。资源边界沿用后文六节点无新高规则；停止或平台时不把117/155参照分数混入候选自己的早停历史。停止且同分other齐全后，controller写入本study自己的selection、`method_freeze.json`和诊断声明；旧study的选定Writer/MT-BC绝不改写。最终测量分支不给Test/FT/RL授权，也不根据wrong/order成绩重选模型。

## 验证范围

当前分支尚未把重构后的测试或preflight标记为通过。`test_writer_training.py`覆盖显式opt-in，`test_writer_data.py`覆盖真实采样器的主/辅助事件关系；实际metadata/历史exposures核验只能由上述preflight完成，未运行前不能报告通过。
这些测试证明实现与协议检查的局部行为，不证明候选性能，也不替代正式GPU资格。

---

# 实验合同：辅助监督对应关系的单次匹配fresh训练

## 0. 决定、证据状态与授权范围

实现基线为合入PR #3后的canonical `main`；实际clean pushed detached runtime commit写入launch contract。same-video
参照配置及其原始覆盖运行仍锚定于`7f62c7b70e608e0dc97d06cb7004a698fd5b5fea`。

本计划建议执行一条新的 fresh Writer 轨迹，保留共同 Source-71、当前 36 个训练任务、模型结构和训练配方，仅将 7 个辅助动作查询改成同任务另一 episode。原覆盖训练的 same-video 轨迹作为已经完成的比较参照。

目前辅助对应关系尚未被确认为当前退步的根因。本次训练同时承担候选修复与匹配干预实验。任何执行端报告不得预写“已解决根因”。

本轮不执行 N1000 辅助切换续训；省去先续训、后 fresh 的重复成本。只允许一条 fresh 跨 episode 候选；无 MT-BC 重训、无第二 seed、无其它 Writer 候选。架构、数据划分、Source、LR、辅助权重、任务 batch 均保持。禁止采用 merged MT-BC 底座、常量输出支路、拆 head 或新增几何正则。

向 Codex 下达本协议意味着建议授权其中的一次 fresh 训练；本文件本身仅为执行规格，尚无实验被启动或完成。

## 1. 唯一科学变量

原目标：

`L = mean_21(cross_episode_main_FM) + (1/3) mean_7(same_video_endpoint_prefix_loss)`。

候选目标：

`L = mean_21(cross_episode_main_FM) + (1/3) mean_7(cross_episode_endpoint_prefix_loss)`。

两组中的主项均为官方 50-horizon、原 Beta flow-time 采样；辅助项均为 tau=1、50×32 独立高斯噪声、只监督前 5 步的真实 7 维动作。每个任务主项与辅助项各自求均值后组合，每次 4 任务等权。每个 condition 只生成同一套完整 LoRA，两个余切相加，再对全部 Writer / Text / VL / Action Meta 回传。

辅助 episode 从对应任务的 demo0..45 中均匀抽取并排除当前 teacher；七个辅助观察仍来自同一条被抽取的 episode。严格复用现有 `_teaching_event(..., teaching_episode='cross_episode')` 的随机算法和合法 stride5 位置。允许与主项的某个 action episode 偶然重合；主项本身始终排除 teacher。

teacher、21 主查询、任务序列、主 RNG 必须与原 fresh 覆盖训练逐事件一致。辅助噪声流保持原算法；辅助 episode、其观察、动作及受长度影响的 frame indices 按新规则改变。不得声称所有 112 个查询都完全相同。

固定均匀 teacher 与均匀排除自身的抽样下，辅助 episode 的边缘分布仍为均匀；两组主要区别是它与 teacher 的联合对应关系。有限样本的实际访问次数需保存，不保证两条已实现序列逐 episode 计数相等。

## 2. 固定资产与信息墙

- Source：当前 coverage 正式训练使用的 Source-71 aligned raw1000，冻结权重与原 normalization。
- 原比较轨迹：coverage same-video Writer，已完成 step200..1800 九个 correct400 节点。
- 原 Writer 最高：N1000，117/400；原 MT-BC：M300，155/400。均只作冻结比较，不作初始化。
- 新 Writer、三组读取 Meta、optimizer、scheduler、sampler 均 fresh，沿用原 initialization seed 与 seed7。
- 目标协议保持 `configs/libero_24_8_8_coverage_v1/protocol.json`；当前 36 Train、8 Validation、8 Test 身份不变。
- Writer 梯度使用每 task demo0..45；46..49 继续保留给冻结诊断。
- teacher action、state、reward、task ID、filename 等不得进入 Writer；执行 policy 自身的 observation/state 按原协议使用。
- 新 Test 不打开。目标任务 FT/RL、外部复现、专家蒸馏和 validation/test 梯度均不启动。
- 所有旧结果与谱系保留，不覆盖任何旧数据、选点或训练日志。

## 3. 代码改动与一次性资格检查

### 3.1 改动应留在既有入口

已核对：

- `learning_data.py::_teaching_event()` 已实现 cross_episode。
- `training.py::_validate_dynamic_schedule()` 对 36-task dynamic 配方硬编码了 `teaching_episode == 'same_video'`；必须为本实验显式开放 cross_episode。
- 原正式 continuation / low-LR 接口不应用于本次 fresh。`phase_continuation` 与 `continuation` 均不设置，不能继承旧 fixed low LR。

增加命名清楚的实验配置/变体，保留旧默认行为与所有数据角色检查。严禁通过删除整体配置校验或角色校验实现运行。模型文件不需要修改；任务采样与功能损失核心算法不需要复制一套。

新配置与原覆盖 fresh 配置的科学 diff 只允许 auxiliary episode relation；其余差异仅限设计标识、输出目录、来源说明与材料路径。后文的报告分支改变材料生成范围，不能改变训练目标。

### 3.2 CPU / 元数据测试，不运行额外 GPU 诊断矩阵

1. 当前 36-task 角色、source、normalization、38 targets、rank16、单视角和两种 episode 池校验。
2. 在原1800更新范围逐事件比较任务顺序、teacher、occurrence、主21的 demo/frame/offset/noise、辅助噪声。辅助 action episode 必须与 teacher 不同且属于合法池。
3. 辅助合法位置 p 满足 p 为 stride5 位置且后面有 5 个真实动作；保留不足7位置时登记有放回的既有行为。
4. 正确 mean_21 / mean_7 / 1/3 / task_weight；禁止额外除 world_size。
5. 相同构造和 initialization seed 应产生相同初始 Writer/Meta；若原始 step0 资产缺失，记录“由相同构造重建检查”，不能写成比对了不存在的原资产。
6. fresh optimizer 状态确实为空；同拓扑恢复候选自己的checkpoint后cursor、LR与RNG连续。显式拓扑转换只允许在完整
   ordinary dynamic checkpoint后发生，必须记录新旧world size、RNG来源和保持不变的逻辑更新/optimizer/scheduler/sampler。
7. 原 same-video 动态配置、旧24任务消融配置和既有评测不被新接口破坏。

没有具体异常就不扩大 E0/E1/E2，也不增加额外闭环 smoke。第一条正式更新检查 finite、实际 LR、任务与辅助事件；工程通过后继续同一条轨迹。

### 3.3 原轨迹参照可比性

原训练曲线来自另一个已执行 runtime。核对相关模型、梯度权重和原生执行算法的 diff，不把实现修复悄悄算成辅助对应关系效应。若发现影响数学目标或实际函数的旧错误，暂停因果归因并报告，不伪装成严格单变量结果。

## 4. 固定 fresh 训练配置

| 参数 | 设置 |
|---|---|
| optimizer | AdamW |
| peak LR | 3e-4 |
| betas | (0.9, 0.95) |
| eps / weight decay | 1e-8 / 1e-4 |
| clip | 原全局 1.0 |
| warmup_updates | 150 |
| original cosine clock | 18000 |
| original decay_lr | 1e-5 |
| additional tail | 1350..2250 |
| tail final ratio | 0.1；之后沿原函数保持非零 floor |
| sampling | 36任务均衡随机轮转，每9更新一轮，每更新4不同任务 |
| query counts | 主21 + 辅助7 / task |
| auxiliary lambda | 1/3 |
| teacher camera | agentview |
| output | 38-target、rank16、完整A/B |
| frame chunk / topology | 优先原 coverage frame8 / world4 |
| initialization / RNG | 原配置的所有 seed 不变 |

本轮保留原 LR 日程，避免同时检验新优化计划。日程的已知局限仍记录。不能继承最近低LR修复阶段的固定 2.959936e-5。

首次formal launch优先且只使用已登记的world4/frame8。后续完整checkpoint若原拓扑不再安全可用，ordinary dynamic
resume可显式登记物理拓扑转换；不得静默改拓扑/精度后宣称逐步等价。物理microbatch的合法调整依现有数值/逻辑合同
登记，无需为填满显存而改变数学batch。

T 个实际更新对应 4T 次视频条件、84T 主查询、28T 辅助查询。日志与完成统计使用真实执行量，不将样本重复计为新演示。

## 5. 训练过程与动态停止

- 每100更新保存可恢复 checkpoint。
- 每200更新完成一个 strict correct Validation400：200、400、600……。
- 每个完整训练段和完整评测结束后再裁决；不依据半个面板或动态图的局部结果改方法。
- 不规定1800/2000/2250为科学终点，也不为达到MT-BC分数无限续训。
- 继续复用覆盖实验的两条已登记停止纯函数及测试，参数不因候选分数临时改变：
  1. 最近3节点均低于此前历史最高，均值低至少8条/400，最近三点线性斜率<=0：持续回落。
  2. 最近5节点均未刷新历史最高，范围<=8条/400且线性斜率<=0：平台。
- 单次下降不停止。训练loss下降不能否决已触发的闭环规则。
- 在与原same-video轨迹相同的停止规则下比较整个算法结果；若候选提前停止，报告其实际有限窗口，不能自动延长或换LR救回。
- 若自本次候选最后一次严格刷新最高后连续6个完整节点无新高，且原两条规则均未触发，将此标为 `RESOURCE_REVIEW_NO_PROGRESS`，保存并停止自主工作。该条为无人值守的资源边界；它不定义模型收敛，后续可由Owner明确决定是否继续。
- 非finite、资源失败、流程异常或Owner stop标为对应中断；保留最近完整状态。

统计历史仅使用此fresh轨迹。N1000/117和M300/155只作参照线，不写入候选自己的早停历史。

### 5.1 已登记的无人值守预算与post-stop extension

2026-09-22 03:10 CST formal preflight以gpu02:0--3四卡frame8登记训练节点`200..1200`，而非读取任何中间成功率决定。
成本上限使用旧四卡mean update 9.8246 s、correct400评测873.25 s、每面板377 s物化/控制余量：1200训练和六个
correct400预计19289 s，另预留四个选中节点测量面板5000 s；data0额外峰值预算16 GiB。gpu01没有安全训练卡，且剩余
GPU没有不干扰主候选、能完成一条训练加完整400的余量，因此本窗口不启动纯主FM基准。

首次触发原两条早停规则时，controller立即保存触发节点、理由、截至该点的完整历史、最佳checkpoint集合和
`original_early_stop_selection.json`。若同分最高需要other400，先完成这些选点前提再继续；随后同一轨迹可仅到节点1200，
所有新节点标为post-stop extension。原合同选点保持冻结；extension节点预先允许参与独立`extended_run_final_selection.json`，
并按相同correct/other/earliest规则决定其冻结后测量。不得删除下降节点、重写原早停事实或依据extension成绩改变上限。

## 6. 模型选择与自动后继工作

### 6.1 训练结束时选择

选完整 correct Validation400最高的候选。多个同分最高时沿原Writer规则，仅为这些候选补other400，按other最高、再按较早步数选择。wrong/order不参与选择。

保存明确的选点清单，正式结论同时列出相邻节点、breadth、逐task/suite及R/G/L。若候选低于原N1000，不覆盖原选定模型。

### 6.2 停止后按预登记分支自动执行

A. 候选最高 <=117：
- 归档全部曲线、逐行比较、成本与已完成同分other。
- 不再增加wrong/order/Test；不新建任何训练分支。
- 结论范围：本次fresh跨episode配方未恢复到既有最佳；这项修正未获功能收益支持。

B. 候选最高为118..155：
- 冻结唯一候选后完成other400、wrong400；已合法完成的other直接复用。
- 不追加order/Test/FT/RL。
- 记录有限恢复；对MT-BC的优势尚未建立。训练停止前已执行的上升段按统一规则处理。

C. 候选最高 >=156：
- 冻结唯一候选后完成other、wrong、shuffle、reverse各400。
- 所有该分支登记面板收齐，无论某个control结果正负；不要据wrong结果换候选或调整方法。
- 主表、视频内容/时间证据、相邻能力和小样本任务簇不确定性分别报告。
- 新Test仍保持封闭；不自动进入RL/FT/外部比较。

这些分支只分配后继测量预算，分数118或156不代表统计显著、根因确诊或全面论文通过。

## 7. 分析与报告必须区分的问题

### 7.1 辅助对应关系是否改变了学习？

在候选与原参照都存在的step200..1800节点上逐行配对，报告候选减原same-video的差值、R/G/L、每task/suite结果。主图显示全部预定节点，不能只选择差距最大的节点。超过原参照实际终点1800的候选节点只能报告延长后的结果，不虚构原same-video对应分数。

不能直接用不同episode上的辅助loss均值判断谁更“会学视频”。两组辅助查询定义有意不同。

### 7.2 有无实际恢复、能否超过MT-BC？

分别报告：
- 超过原轨迹同step弱点；
- 超过原最佳117；
- 超过冻结M300的155；
- 相邻节点的维持与覆盖。

只提高到110不能描述成恢复；只高于某一个原轨迹谷值不等于超过旧最佳；一个峰值不证明稳定优势。Spatial3为事先登记的缺口定位，同时呈现全部8任务，禁止隐藏其它能力损失。

### 7.3 视频信息是否提供收益？

正确能力上升和正确视频的增量分别判断。所有视频臂沿原固定配对协议；wrong/order不参与训练或选点。same-task-other用于正确演示更换的稳定性。两个方法基于共同Source，当前模型无需构造常量输出能力。

### 7.4 根因措辞

即使本次成功，也最多首先支持“在当前任务池、初始化和日程下，改变辅助配对改善了学习”。不能直接推导某个内部模块是唯一原因、所有任务上same-video都有害、或其它实验差异全部由此解释。

失败时不得追加main-only、额外seed、lambda扫描、P-only梯度、MT-BC底座或head改造。完成负结果本身即为执行完成。

## 8. 长程运行组织

沿用已有分段controller和任务结束信号。允许Codex在完整阶段结束时执行预定义的机械比较、选点和材料分支；无需频繁读取部分结果，也无需监控子代理。

每次launch/resume前检查两节点GPU、他人进程及独立quota，遵守当时AGENTS总卡数边界。默认优先保障原4-rank Writer；段间释放设备给动态队列评测。训练/物化/评测/安全共驻统一计数，不以提高利用率为由占用他人忙卡或同时启动未登记候选。

独立CPU工作可以与正式训练并行：
- 验证原始CSV覆盖和对照映射；
- 整理两方法完整曲线、逐task面板及训练量；
- 为组会制作可由CSV复建的图与事实摘要；
- 完成后继图表脚本、数据字典与run索引。

禁止用无科学收益的GPU工作填满空闲时间。流程成功或失败都执行材料封存与推送。

工程故障：从最近完整候选checkpoint恢复相同条件；默认同拓扑exact resume，若必须转换物理world size，只走上述显式
ordinary dynamic恢复并保留其RNG边界。同类故障最多两次修复重试，仍失败则记录阻塞并退出。不得把资源中断写成持续下降
或收敛。若修复改变数学目标/数值算法，停止并登记，不继续作为原因果对照。

所有正式运行来自clean、pushed、detached runtime，不原地改写正在运行的代码。

## 9. 材料保存

建议新目录：`docs/review_materials/20260922/auxiliary_pairing_fresh/`，正式大型资产在现有quota允许的study位置，路径由canonical资产管理确定。

最少交付：

- `registration.json`：唯一变量、初始资产、代码commit、随机算法和所有停止/材料分支。
- `config_diff.json`、`preflight.json`、`event_alignment.json`：科学差异与资格检查。
- `training_steps.csv`、`task_exposures.csv`：全部真实更新，含 LR_applied/LR_next、主/辅助loss及已带权余切范数、总梯度范数、裁剪、模块梯度、时间和显存。
- `validation_history.json`、`validation_rows.csv`、`matched_step_comparisons.csv`、`per_task_nodes.csv`。
- `selection.json`：候选最高与旧117/M300155的不同口径，禁止controls驱动换点。
- `control_rows.csv`：实际被分支授权并完整完成的臂，未执行项单列。
- `completion.json`：真正完成的updates、C/Q、400行面板、worker退出、科学停止/资源中断/Owner停止/故障。单个exit0不能替代产物完整性。
- `report.md`及绘图源码：完整曲线、相邻R/G/L、逐任务热图、全部测量与边界。

轻量表推Git，大checkpoint与轨迹只保留索引。保留最佳、最新及邻近正式点的完整状态；无依赖载荷才清理。不新增大范围hash流水线，不复制Source或数据集，不退休新实验仍需引用的唯一资产。

## 10. 执行结束条件

本计划只授权一个fresh跨episode Writer候选及上述分支测量。执行代理产出事实、按固定规则推进并交付；任何分支都不要求“EMBER必须赢”才能结束。

一个候选成功，可以形成主线修复证据；一个候选失败，也应当结束本次自动工作。剩余的不确定性由Owner和后续审阅决定，不能变成无人值守的无限架构搜索。

## 证据入口（固定快照）

所有路径相对于 `LinFyM/EMBER@7f62c7b70e608e0dc97d06cb7004a698fd5b5fea`：

- `docs/video_teaching_writer_design.md`
- `docs/review_materials/20260919/continuation_report.md`
- `docs/review_materials/20260919/ablation_controls_report.md`
- `docs/coverage_retraining_design.md`
- `docs/review_materials/20260921/coverage_retraining/report.md`
- `docs/review_materials/20260921/coverage_retraining/diagnostics/writer_training_config_summary.csv`
- `docs/review_materials/20260921/writer_output_space_diagnostics/report.md`
- `src/ember/writer/{learning_data,function_credit,supervised,training,continuation}.py`
- `AGENTS.md`、`docs/current_owner_requirements.md`、`task_plan.md`、`progress.md`
