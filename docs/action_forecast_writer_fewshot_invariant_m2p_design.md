# Few-Shot Invariant-Program M2P Writer

状态：2026-08-06 implementation authority。本文覆盖上一轮 Factorized
Condition-Kernel Program Memory 的活动地位；后者的正式负结果继续保留，不能加载其
checkpoint、Program Memory 或 FactorHeads。

## 1. 决策

下一轮 canonical Writer 改为：

> exact task language + four independently sampled action-hidden same-task videos
> -> one joint invariant program -> one complete rank-16 PI05 LoRA.

四条视频必须在一次 Writer forward 内联合编码并只生成一套 LoRA。禁止逐视频生成
LoRA后平均、挑选视频、投票、ensemble或多checkpoint融合。部署和训练使用同一 K=4
范式；这不是训练期augmentation后退回one-shot。

该修改得到owner明确授权。长期裁决仍只认同一single checkpoint的strict paired
correct400，最低目标严格大于150。

## 2. 为什么此时打开 few-shot

历史结果已经分别排除了几个表面解释：

- v6把视频顺序变化传到Procedure、effective BA和action，但same-task video能量约
  `0.13%`，single best为143且checkpoint能力持续换手；
- Writer-v2的generic-language/correct-wrong训练把视频因果差做强，却把absolute降到
  99，故不再重复language dropout或contrastive margin；
- Direction Store、Target-Owned、PWAD和Policy-Lane分别增加condition、target、atom和
  lane ownership，均没有把条件差异稳定写入闭环有效方向；
- Condition-Kernel显式隔离condition credit并保留视频/顺序差异，但fresh decoder的
  LoRA norm约0.18、correct49，证明存储隔离不能替代高增益policy写出。

one-shot中任务程序、初态、具体轨迹、速度与偶然视觉细节来自同一条视频，模型没有观测
轴去区分“多演示共同的任务变化”和“单轨迹nuisance”。K4提供这一统计轴。新架构必须
利用它形成跨视频共同program，而不是把四条视频当作更多frames。

## 3. 不重复历史方法

本方法不做：

- generic/zero language、language dropout或wrong-video margin；
- SFT LoRA reconstruction、SFT basis复制或强制rank/正交/能量loss；
- task ID、teacher action、proprio、reward、terminal或object pose输入；
- fixed task store、RFF condition kernel、policy atom dictionary或独立A/B mixing；
- macro50 decoder freeze、B-only residual、global scale/gate或static bypass。

language仍是正确完整task language，但只参与video grounding和attention address；LoRA
动态value不能从纯language residual直达输出。frozen source policy在action query中仍按
官方接口收到正确language和observation。

## 4. 输入与采样合同

- 每个task visit从该task 50条teacher demos中确定性无放回抽K=4；同一set内不得重复。
- 四条video都只读取`obs/agentview_rgb`，stride5并包含final frame。
- action query仍为同task跨episode B20；先由预封存action sampler确定本visit的精确
  query episodes，再从其补集中确定性无放回抽四条teacher demos，因此两者不可能重合，
  且不会因video选择反向改变B20 action-query序列。
- 一个K4 set生成一次LoRA并复用于该task完整B20 query。
- full24先task内mean，再24-task等权；单卡逻辑数据量、global B20和macro定义不因A40
  或K4改变。
- evaluation每个paired row以原video ordinal为anchor，用固定无重复offset形成K4 set；
  correct/same/wrong/shuffled/reversed共享task/state/env/policy RNG和set ordinals。

## 5. Video-value-only condition memory

foundation descriptor保持source policy冻结。每条视频逐帧提取：

1. task-grounded visual-language innovation；
2. fixed-noise t=1 Action Expert interaction；
3. 四个有符号时序基`1, tau, cos(pi*tau), sin(pi*tau)`。

每条视频产生4个128维temporal value tokens，K4共16个tokens。task text descriptor只形成
attention query/key address；它不得以residual或value被加到condition memory。

`InvariantProgramEncoder`用32个latent queries读取16个video value tokens：第一次
cross-attention只返回video values，不保留latent/text query residual；之后才允许在已经
video-owned的content上做self-attention与FFN。由此在结构上保证：若video values被置零，
动态LoRA输出严格回到identity，而不是退化成language-only task adapter。

K4维度不使用shot ordinal embedding，set permutation必须严格等价。temporal basis保留
每条视频内部顺序；跨视频只做set aggregation。

## 6. 跨policy M2P decoder

公开LoRA有38个policy targets：18层q、18层v、action-in和action-out；每target有16个
rank lanes，共608个target/layer/rank queries。

`PolicyM2PDecoder`使用三层交替结构：

1. 608 queries cross-attend 32 invariant program slots；
2. 608个policy tokens全局self-attend，让layer/module/rank共同组织；
3. 再读program并做target-token FFN。

首个cross-attention同样不保留纯routing query residual，所以输出content必须由video
value产生。module/layer/rank embedding只决定读取位置。

每个实际policy target拥有自己的A/B row heads；同target的16 lanes共享head但不共享
rank token。A输出为identity template上的完整residual，B从物理零开始；两个因子始终都
是完整Writer输出。decoder、program encoder与全部target heads从fresh identity训练到
formal horizon结束，不提前冻结。

## 7. 训练与RL兼容性

第一轮仍用action-supervised functional PI05 flow loss建立可用cold start，因为它是当前
唯一覆盖24 tasks全部状态的dense policy signal。这里没有新增监督专用auxiliary loss；
functional cotangent直接穿过完整LoRA、M2P、invariant program和video descriptor。
不再在checkpoint边界读取validation actions计算held functional loss；历史已经证明该值
不能选择closed-loop checkpoint，这项监控只增加开销且不会进入本方法裁决。

后续reward训练若打开，复用同一个K4 Writer和608-token M2P：rollout reward只替换credit
来源，不替换condition representation、output head或deployment接口。因此本架构不是为
统一梯度下降临时拼接的SFT trick。

## 8. A40执行计划

1. CPU合同：K4 schedule、set permutation equality、step0 identity、video-zero identity、
   608-token shape、target-owned heads、freeze/gradient ownership、checkpoint拒载旧family。
2. 六卡A40 profile：16-frame encoder chunk、policy microbatch2、logical B20、full24；
   fresh0->1和独立exact-resume1->3。profile scheduler总轴固定为正式200步、只在step3
   early-stop，不能用三步压缩scheduler冒充正式早期优化语义。K4视频顺序串行编码以控制
   峰值，不能缩减B20。
3. fresh formal0->200，每25保存；strict correct400预注册50/100/150/200。
4. 若200点absolute/breadth/趋势仍有可信上升且没有机制硬失败，exact-resume到400；否则
   根据最早失败接口设计下一方法，不能用loss挑点。

## 9. 预注册裁决

每个正式候选报告：

- strict correct400、breadth、per-task/suite、相邻gained/lost、union/intersection；
- correct/same-task-other/cross-suite-wrong/shuffled/reversed严格配对五臂；
- K4 set置换等价、leave-one-video-out、另一个同task K4 set、全视频reverse/shuffle；
- invariant slots、608 M2P tokens、effective BA、fixed-action的condition传递；
- LoRA norm、stable rank、top singular energy、q/v/action能量和跨layer组织；
- task梯度/更新cosine、full24 energy retention与checkpoint能力换手。

mechanism最低门：

- set permutation输出数值等价；
- zero-video动态输出为identity；
- 同task另一K4 set与leave-one-out产生非零但小于cross-suite wrong的有界变化；
- reverse/shuffle到BA/action路径非零；
- LoRA幅度和fixed-action effect不能重演Condition-Kernel低增益平台。

行为门仍是single checkpoint strict correct400 `>150`。漂亮rank、视频差异、functional
loss下降或多checkpoint union都不能替代该门。

## 10. 失败后的唯一归因顺序

若失败，按以下最早接口裁决：

1. frozen video descriptor是否缺少跨demo共同语义；
2. invariant slots是否形成video-common而非task-common/zero content；
3. M2P是否把多维program压回共享方向；
4. effective BA是否有足够policy leverage；
5. 闭环credit是否仍与有效policy流形错位。

只有证据定位到相应接口才允许修改。不得再以增加K、head、rank、scale、训练时长或store
数量作为默认修补。

## 11. 当前执行证据（2026-08-06）

- fresh formal0→200已自然完成：200 finite macros、96,000 action queries、19,200 K4
  teacher videos、8个every25 checkpoints、0 clip/OOM/nonfinite，source policy保持冻结，
  validation/test action reads均为0。
- macro50/100 strict correct400=`70/94`、breadth=`6/6`；50→100 gained/lost=`42/18`、
  union/intersection=`112/52`。两个checkpoint的state、四条teacher demo indices、env seed与
  policy RNG严格相同，每task 50个sets覆盖50条unique teacher videos。
- 当前只是四点曲线的前半段，既没有过`>150`行为门，也已有18个success丢失；不得写成
  漂移已解或据此选择macro100。下一步固定评macro150/200后再选single winner做第9节分析。
