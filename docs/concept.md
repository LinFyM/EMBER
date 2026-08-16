# EMBER Concept

状态：2026-08-13稳定问题定义。本文不绑定当前架构、LoRA rank或启动命令。

## One-sentence definition

```text
Writer(exact task language, one or more action-hidden correct teaching videos)
    -> one complete task-conditioned policy adaptation
    -> frozen source policy executes from unseen initializations
```

## Why this problem matters

人看正确示范后，会结合任务名称理解对象、目标关系、动作阶段和先后因果，再把这种理解迁移到自己的身体、视角
与初始状态；他不会逐像素、逐关节复制示范者的原轨迹。EMBER把这个过程转化为一次性参数编译：shared Writer
在rollout前读语言与视频，生成一套task adaptation；policy执行期间不再反复观看示范。

学习问题有三层：

1. 从视频识别跨初始化仍成立的高层任务内容与有向过程；
2. 将这些证据编译成policy-effective的完整adaptation；
3. 让多task、多video能力在同一Writer checkpoint稳定共存并泛化。

## Shared foundation

所有方法共享一个从generic`lerobot/pi05_base`出发、只在与目标40 specification-only去重后的LIBERO-90 source
tasks上训练并冻结的π0.5-LIBERO policy。normalization只来自source并冻结；目标40固定24 train / 8 validation /
8 test。不得使用已读过目标40 actions的`pi05_libero`。

source base提供通用视觉、语言、机器人控制和action-space能力，但不提供目标任务的专属完成方式。EMBER检验的是
教学视频能否在这一共同起点上写出新的task能力。

## Information wall

部署时Writer可读：

- exact task language；
- 一条或多条同task、自然顺序、action-hidden teaching videos。

Writer不可读teacher action、proprio/state、reward、terminal、task ID、filename、episode identity、object pose、
hidden normalization或held outcome。language可提供query/context/address，但video必须提供唯一dynamic value；
不能有language-only LoRA bypass、held expert route、第二套adaptation或checkpoint融合。

每个condition只生成一套完整38-target task adaptation。当前受控机制实验保留LPCP rank16 carrier加rank16
native-zero residual bank；rank、memory数量和decoder是
方法变量，不是问题定义。无论具体rank如何，都不能靠挑video、平均分别生成的LoRAs或裁剪关键policy targets
冒充完整任务学习。

## Learning without low-level pairing

AS训练让video与action query来自同一task但不同episode。一组video生成一套LoRA，再由多条独立action queries
共同约束，使它覆盖不同初态而非拟合teacher的单条路径。

错开监督能阻断低层复制，却不能自动保证视频理解：同task不同videos共享task-level supervision，模型仍可能
只学task identity或静态高频相关性。因此架构要保留视频动态过程，closed-loop controls要验证correct的内容和
顺序确实有用。

## One-shot, few-shot and dynamic cardinality

one-shot最直接检验“一次正确示范即可开始做对”，也最容易受单条demo的速度、路径、视角和偶然扰动影响。
few-shot可以从多条同task示范中提取跨demo共同程序，但必须保持每条video内部顺序，并在video之间使用置换不变
集合处理；简单平均frames、features或LoRAs并不满足要求。

方法可在训练中覆盖动态cardinality，并分别报告实际K下的能力。如果K1最好，就形成one-shot claim；如果K>1
最好，就形成few-shot claim；若性能随视频数量稳定提升，可形成scaling claim。无需为了形式公平故意削减更强
方法，但必须诚实报告视频数、总帧数、训练量、FLOPs、延迟和选择规则，且不能挑“最好视频”。

## Correct order is causal evidence

correct展示从初态到目标态的有效有向过程；shuffled破坏阶段连续性，reversed破坏正常物理与任务因果。正常顺序
作为训练分布并不自动产生理解。只有correct在严格配对闭环中沿有用方向胜过wrong、shuffled、reversed和
no-video，且same-task-other保持接近，才能支持视频因果claim。

## Baselines and claim boundary

- frozen source base：共同起点；
- Source-SFT：读target actions的privileged shared-LoRA reference；
- AS-Writer：核心zero-interaction方法，部署只读language+video；
- 可选RL-Writer：后续关闭action入口、用train24 reward调整Writer；
- 生成LoRA后的task-local RL：检验视频初始化是否提高交互学习效率的独立后半段；
- direct target-action oracle：最终privileged ceiling，不是同信息墙baseline。

核心claim首先是：正确教学视频生成的single-checkpoint zero-interaction adaptation带来高closed-loop性能，而且
优势来自视频内容和顺序。reward adaptation与task-local RL不能掩盖初始Writer不足。

## Success criterion

性能继续追求同一shared method、同一single checkpoint strict paired correct严格`>150/400`，并继续提高
absolute、breadth、共同积累和视频因果性。约145若由相邻checkpoints共同保持、成功集合低换手、same-task不同
视频鲁棒且correct相对wrong/shuffled/reversed/no-video有明确优势，也构成有价值的成立结果；单点151若高波动或
无视频因果性仍不合格。LoRA能量、rank、cosine、reconstruction、functional loss和内部margin只用于解释，不能
替代真实闭环。
