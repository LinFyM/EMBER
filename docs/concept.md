# EMBER Concept

## 一句话定义

EMBER 是一个 multimodal task-conditioned hypernetwork / amortized LoRA initializer：

```text
Writer(language, action-hidden videos) -> task-specific LoRA
```

其中 frozen base policy 加载 task-specific LoRA 后应立即提高目标任务成功率；随后标准 task-local RL 可继续原位更新同一套 LoRA。

## 为什么不是多任务通用 LoRA

Writer 参数在 70 个 source tasks 间共享，学习“如何根据任务证据生成参数”。它每次接到一个 task 的 language/video 时输出不同 LoRA。若改成一套多任务通用 LoRA，就不再测试 task-conditioned parameter generation，也无法回答未见任务视频是否足以编译新技能。

## Source base、direct LoRA 与 EMBER

- source embodiment base：一个在 70 个 source tasks 的 3500 条 teacher episodes 上联合训练的共享 SmolVLA。
- direct task-local LoRA：针对一个 task，直接用该 task teacher actions 优化一套 LoRA。
- EMBER：Writer 看 language + action-hidden videos，输出该 task 的 LoRA；Writer 的 source 训练仍可通过 action loss 得到梯度。

source base 和 LoRA 都能改变动作策略，但训练范围与使用方式不同。当前项目不再设“LoRA 必须进一步超过 source base”这一人为 Gate。真正的目标是：在统一 source base 上，EMBER 对未见 task 只凭 language/video 也能比 frozen base 好。

## Functional supervision

对 source task，Writer 先产生一套 LoRA。把它 functional 地装入 frozen base，在 source teacher observation/action batch 上计算 SmolVLA 标准 flow/action behavior loss；梯度通过 policy 和 functional LoRA 回到 Writer。

direct LoRA 使用同类 behavior loss，但 optimizer 直接更新该 task 的 LoRA tensors。匹配两者的数据、LoRA 空间和 loss 后，差异就是“每任务直接拟合”与“从语言/视频摊销生成”。

## 视频表示

输入不是三帧摘要。冻结 VLM 对完整视频帧产生 features；Writer：

1. 在时间 chunk 内做 attention；
2. 把所有 chunks 聚合成 episode memories；
3. 对任意数量的 episode memories 做集合注意力；
4. 与完整语言 token memory 融合；
5. 以 layer/module/rank-aware decoder 生成 LoRA tensors。

chunking 是计算分块，不是丢帧或固定视频长度。训练使用 50 条 episode，架构接口不设 50 上限。

## 四个训练阶段

### Cold start

base 冻结，action监督只来自 source teacher episodes，更新 Writer。目标是 zero-interaction LoRA。

### Writer-only RL

base 冻结，Writer 生成 LoRA 后直接 rollout；不更新 LoRA，reward 只更新 Writer。目标是改善未来生成的初始化。

### Task-local RL

base/Writer 冻结，针对当前 task 原位更新完整 LoRA。比较 zero-init 与 Writer-init 的 matched adaptation。

### Source-only outer learning

inner loop 更新 source task LoRA，outer reward/meta objective 更新 Writer；base 仍冻结。目标是让 Writer initialization 更利于后续 adaptation。

## Information wall

| 阶段 | Language/video | Action labels | Reward | 更新对象 |
| --- | --- | --- | --- | --- |
| source base | train | train | 可选诊断 | shared action expert/projections |
| direct source LoRA | train | train | 否 | task LoRA |
| Writer cold start | train | train，仅作 loss | 否 | Writer |
| Writer-only RL | train | 否 | source | Writer |
| val/test zero-step | target | 否 | 否 | 无 |
| val/test task-local RL | target | 否 | target，计入预算 | task LoRA |
| direct val/test LoRA oracle | target | target | 否 | task LoRA |

direct val/test LoRA 故意越过主 information wall，因而只作能力参考。

## Claim boundary

最小正向 claim：

> 在同一 embodiment 和 simulator family 内，Writer 从未见任务的语言与 action-hidden robot videos 生成完整 task-specific LoRA；该 LoRA 在不约束后续 RL 搜索方向的情况下，提高 frozen base 的 zero-interaction 成功率，并可改善 matched ordinary task-local LoRA RL 的适应。

不能声称 Writer 学到了 optimizer、geometry、subspace 或 update direction。
