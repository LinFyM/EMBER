# Novelty and Baseline Landscape

本文件只讨论在当前 70/10/10 信息墙下，哪些近邻方法可以公平比较。论文原始协议不同，不能直接把原表数字并排当结论。

## EMBER 的增量

现有工作分别覆盖了参数生成、示范视频理解和 reward adaptation，但没有统一覆盖：

1. language + action-hidden video；
2. 生成完整 task-specific LoRA；
3. zero-interaction closed-loop utility；
4. Writer-only reward training；
5. 在同一 LoRA 上继续 ordinary task-local RL；
6. source-only outer learning；
7. shared-frozen held evaluation。

EMBER 的 novelty 必须由这条组合链和因果对照支持，而不是由一个新名词或 bank 假设支持。

## 可在相同 information wall 下比较的方法

### HyPoGen-style parameter generator

原理：用 source task specification 和 source action demonstrations 训练 hypernetwork，在未见 specification 上直接生成策略参数。

映射到 EMBER：

- train：70 tasks 的 language 和 teacher actions；
- val/test：只给 language，不给 action；
- 输出：与 EMBER 完全相同的 LoRA tensor contract，而非任意不同容量的 full policy；
- base：共享同一个 frozen source embodiment base；
- steps/data：与 Writer cold start 的 source training budget 匹配。

它是强 language-to-parameters baseline。虽然 owner 移除了“Language-only Writer”内部消融，最终仍可保留这个有外部方法语义的成熟 baseline；二者不要重复运行。

### DISC-style language-to-policy/LoRA generator

DISC 的主结果更多是在训练任务内换初态，不能原样证明未见 task 泛化。公平版本应：

- 用 70 source tasks 训练；
- 在 10 val/10 test unseen tasks 只读 language；
- 输出 matched LoRA；
- 不读取目标 action；
- 报告 unseen-task 成功率。

它与 HyPoGen 机制接近，可根据实现质量选择一个主 baseline、另一个作补充，避免两个几乎相同的巨大复现拖慢机制开发。

### ViVLA / See Once, Then Act-style video-conditioned policy

原理：在 source 上用 expert-video/robot-trajectory 配对训练，held task 只看一段或多段 expert video，直接执行而不做 action-supervised target fine-tune。

映射：

- train：同样 70×50 language/video/action teacher data；
- val/test：相同 language + action-hidden videos；
- 输出：直接条件动作策略，不要求生成参数；
- shared source base、数据量、steps 和 eval 匹配。

它是最重要的“视频有用，但不一定生成 LoRA”baseline，可检验 EMBER 的参数编译是否优于 direct conditioning。

### DAML-style learned adaptation

原理：source tasks 上从示范视频学一个 adaptation procedure，再在 held task 根据未标动作视频进行内层适应。

公平版本：

- source action labels 可用于 meta objective；
- val/test adaptation 只能消耗 language/video；
- 不读取目标 teacher actions；
- 若产生 target optimizer steps，要报告并匹配其 compute，而不是假装 zero-step；
- 最终在同一 held rollout 合同评估。

它与 EMBER 都利用 held video，但参数化和 adaptation procedure 不同。

### Direct language+video conditioning

在 shared source base 上增加/训练一个 source-only multimodal conditioning path，val/test 输入与 EMBER 完全相同，直接输出 actions。它不对应单篇论文，但解释力很强：

- 若 direct conditioning 强而 Writer 弱，问题更可能在 parameter generation/acquisition；
- 若两者都弱，可能是 video representation、source diversity 或 task information 不足；
- 若 Writer 强，才支持“编译为可适应 LoRA”有额外价值。

### Retrieval / average adapter

从 source task LoRA 库按 language/video embedding 检索最近 adapter，或平均若干 source adapters。它们：

- val/test 不用 actions；
- 使用相同 LoRA contract；
- 训练成本低；
- 可判断 Writer 是否只是在做粗粒度 nearest-task selection。

## Reward adaptation baseline

### Watch-Try-Learn / RIPT-style

这类方法在 held task 看 demonstration 后进行 trial/reward adaptation。EMBER 也允许 val/test task-local RL，因此它们并非“不公平”：

- 都可消费预声明 target reward interactions；
- 必须匹配或明确报告 interaction budget；
- trial/adaptation rollout 与最终 fresh evaluation 分开；
- shared source modules在 test 不更新，除非原方法不可避免且单独标注信息/参数差异。

核心 Base+ordinary LoRA RL 和 best Writer-init+LoRA RL 已在 EMBER task-local RL 阶段运行，不应在 baseline 阶段换名重复。WTL/RIPT-style 只在其 adaptation mechanism 本身提供独立科学比较时补。

## 不属于信息匹配 baseline

### Direct target LoRA SFT

它在 val/test 读取 teacher actions，违反 EMBER 的 hidden-action 主信息墙，但 owner 明确要求作为 oracle/reference：

- 说明目标 task 在同一 LoRA 空间是否可学；
- 给出 action-supervised 能力参考；
- 帮助区分 Writer 泛化失败与 LoRA/source-base 能力不足。

它不能用于声称 EMBER 打败/落后于一个公平 held baseline。

### Full/action-expert target fine-tuning

参数量和信息都更大，当前快速阶段不做。最终如需要，只作非 matched capability upper bound。

## Shared source base 的公平处理

不同近邻方法在原论文中可能训练整个 policy、conditioning module 或 hypernetwork。当前统一做法：

1. 所有方法从同一 frozen source embodiment base 开始；
2. 方法特有的 shared module 只在 70 source tasks 上训练；
3. source data、可见字段和 update budget 尽量匹配；
4. val/test 不更新 shared module；
5. 若某方法必须 source-finetune base 才忠实，单独允许相同 train-only budget，并明确它与“frozen shared base”主对照的差异。

这样既不禁止成熟 baseline 学 source tasks，也不允许它在 val/test 偷看 actions。

## SmolVLA 是否足够

SmolVLA 不是最终 SOTA 保证，但当前适合作机制开发：

- 规模允许 8 卡快速重训、多 arm 和大量 rollout；
- LeRobot/LIBERO 路径成熟；
- 有完整 action expert 和 PEFT 接口；
- 同一个 backbone 可实现 Writer、direct LoRA、conditioning 和 parameter-generator baselines。

绝大多数 baseline 的核心机制可以在 SmolVLA 上重实现，因此基础模型公平性更好。若在 SmolVLA 上机制成立，再在 OpenVLA-OFT 做 scale confirmation；现在换大模型会把基础 competence、工程吞吐和研究机制混在一起。

## 最终 baseline 分层

快速开发：

- frozen base；
- EMBER；
- direct LoRA oracle；
- ordinary LoRA RL；
- Writer-init LoRA RL。

机制成立后：

- HyPoGen/DISC-style language generator；
- ViVLA/DAML-style video adaptation；
- direct language+video conditioning；
- retrieval/average；
- 必要的 WTL/RIPT-style reward adaptation；
- 多 seeds、完整 tasks、统一 test。

主指标始终是闭环 success。flow loss、action error、LoRA distance、progress 和 time-to-success 只作诊断。
