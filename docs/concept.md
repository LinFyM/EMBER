# EMBER Concept

状态：2026-08-12稳定问题定义。本文不指定活动架构或启动命令；active successor只取当前authority。

## One-sentence definition

```text
Writer(exact task language, action-hidden teaching video[s])
    -> one complete task-conditioned LoRA
    -> frozen source policy executes the task from unseen initializations
```

当前canonical benchmark保持one-shot：恰好一条video。few-shot是未来可研究变量，不是已经启用的合同。

## Why this problem matters

机器人教学视频包含对象、目标关系、动作阶段和时序因果，但不一定附带与目标机器人逐帧对齐的action。
EMBER不让policy在每个控制步反复观看示范，而是让shared Writer在rollout前阅读任务语言和视频一次，把其中的
任务知识编译成策略参数。生成的LoRA随后应对不同初始状态有效，而不是复现某条示范的低层轨迹。

这把学习问题分成三层：

1. 从视频识别跨初始化仍成立的高层任务内容和有向过程；
2. 将这些证据编译成policy-effective的完整LoRA；
3. 让多task知识在同一Writer checkpoint稳定共存，并泛化到held tasks/videos。

## Shared foundation

generic`lerobot/pi05_base`在目标LIBERO tasks上缺少足够embodiment能力，因此所有方法共享一个只在过滤后
LIBERO-90 source corpus上训练的π0.5-LIBERO base。source corpus与目标40做specification-only去重；
normalization只来自source并冻结。目标40固定为24 train / 8 validation / 8 test。

source base提供通用视觉、语言、机器人控制与action-space能力，但不读取目标40 actions。禁止使用已经读过
目标40 actions的`pi05_libero`，否则无法区分foundation competence与视频学习。

## Information wall

部署时Writer可读：

- exact task language；
- 当前合同下恰好一条原始action-hidden teacher video。

Writer不可读teacher action、proprio/state、reward、terminal、task ID、filename、episode identity、object
pose、hidden normalization或held outcome。language可提供query/context/address，但video必须是唯一dynamic
value；不能有language-only LoRA bypass、expert-bank route、第二套LoRA或checkpoint融合。

每个rollout只生成一套完整38-target public rank-16 LoRA。frozen source policy没有trainable parameter，
no-video/step0必须是functional identity。

## Learning without low-level pairing

AS训练中，video与action query来自同一task但不同episode。一条video生成一套LoRA，再由多条独立action
queries共同约束，使该LoRA不能只适配示范的单一初始化或逐帧动作。

这一错开监督是必要的防捷径手段，却不是视频理解的充分条件：同task所有videos共享task-level action target，
模型仍可能只从language或高频低层视觉相关性推断一个controller。架构需要让视频证据成为必要动态通路；
correct/same/wrong/shuffled/reversed/no-video闭环controls负责最终裁决。

## One-shot and few-shot

one-shot最严格，也最接近“一次示范即可学习”的目标；其风险是单条demo中的速度、路径、视角与偶然运动会
盖过高层不变量。

few-shot的合理动机是比较多条同task示范，保留各自内部时序，同时提取跨demo共同语义。有效设计应支持集合
置换不变而不破坏单视频内部顺序，并显式处理1/3/5等cardinality。简单平均frames、features或LoRAs不能自动
满足这些要求，也可能牺牲因果顺序和吞吐。

任何few-shot研究都必须另立matched authority：固定或明确定义`k`，不挑video，仍action-hidden，报告额外
计算，并与同训练/eval口径one-shot对照。历史K4只证明集合稳定性部分改善，不证明性能或drift已经解决。

## Evaluation meaning

最终方法由同一single checkpoint的official strict paired closed-loop评价：

- correct：正确task的自然顺序video；
- same-task-other：同task另一条video，测demo鲁棒性；
- wrong：另一suite视频，测内容特异性；
- shuffled/reversed：真实输入帧重排，测时序因果；
- no-video：测动态视频通路是否必要。

配对必须保持task、initial state、env/policy RNG与video ordinal。视频影响hidden、LoRA或action不等于视频被
正确理解；只有correct沿有用方向稳定胜过controls，才支持视频因果claim。

## Baselines and claim boundary

- frozen source base：无目标视频/动作的共同起点；
- Source-SFT：privileged target-action shared LoRA参照，held不读视频；
- AS-Writer：source action只进入functional loss，部署读视频；
- 可选RL-Writer：action入口关闭后用source environment reward更新Writer；
- matched video-conditioned policy：若实现，比较每步读视频与一次编译的取舍；
- final direct-action oracle：使用test actions的privileged ceiling，不是同信息墙baseline。

核心claim是：在共同source policy上，正确held教学视频生成的zero-interaction LoRA能带来高closed-loop性能，
且其优势来自视频内容与顺序而非language或通用adapter。reward adaptation若后来改善sample efficiency或终点，
是附加claim，不是EMBER定义本身。

## Success criterion

当前底线是同一shared method、同一single checkpoint strict paired correct严格`>150/400`，并继续提高absolute、
task breadth、稳定共同积累与视频因果性。LoRA能量、rank、cosine、重建/functional loss和内部margin只是解释
工具，不能替代真实闭环。
