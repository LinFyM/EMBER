# EMBER Concept

## 一句话定义

```text
Writer(task language, one action-hidden teaching video)
    -> complete task-specific LoRA for a frozen π0.5 policy
```

EMBER 的核心是把无法直接用于 action-SFT 的视频任务知识编译成策略参数，不是规定必须经过某种 RL。

## 学习对象

`Action-Supervised Writer (AS-Writer)` 在 source tasks 上学习 video/language 到 LoRA 的映射。每个 source update：

1. 采一个 source task；
2. 随机采一条 action-hidden teacher video；
3. 从同 task 独立随机采另一条或同一条 agent episode 的 observation/action chunk；
4. Writer 生成完整 LoRA；
5. functional 地装到 frozen generic π0.5；
6. action loss 的梯度回到 Writer。

视频与 action episode 不配对，因此 Writer 不能靠逐帧对齐复制 action；它必须从 task-level visual procedure 和 language 提取可迁移信息。action 只进入 loss target。

## 为什么一条视频

当前故事是 one-shot teaching：执行前看一条教学视频，然后尝试任务。训练和测试都恰好一条，避免训练时多视频聚合、测试时单视频造成合同变化。

held evaluation 的每个 rollout 独立随机抽一条 teacher video，因此报告的是对该 task teacher-video distribution 的期望性能，不是挑选最好视频。

## LoRA 与 reward adaptation

Writer zero-interaction LoRA 是第一主结果。若它本身很强，无需为了叙事强制 RL。

可选 reward learning 有两类：

- `Reward-Trained Writer (RL-Writer)`：从随机 Writer 初始化，或仅做预声明的极短 AS warm-up，然后跨多个 source tasks 只用环境 reward 联合更新 Writer；默认不从完整 AS-Writer 继续，从而直接检验没有 teacher actions 能否学出 Writer；
- task-local LoRA RL：Writer/base 冻结，单 task 原位更新该 LoRA，与 identity/zero-init 做 matched comparison。

后者每个 adaptation run 只在开始时抽一次 teacher video，随后 LoRA 持续存在。两臂用相同 official BDDL random resets、seeds、interactions、updates 和 selection rule。固定 `.pruned_init` states 只用于 fresh evaluation。source-only outer learning 只能在 Phase F 之后。

## Base policy

当前先测试 generic pretrained π0.5 的 LIBERO zero-shot feasibility，不默认 action-SFT source base。generic π0.5 没有 LIBERO action normalization；必要的 action/state interface stats 只能从 24 development-train tasks 计算，且不更新模型权重。

如果 base feasibility 很低，是否增加 24/32-task source-base action-SFT 由 owner 根据结果另行决定，不是本轮自动分支。无论是否把它作为共同起点，最终都会报告一个在 32 source tasks 上、与 AS-Writer 匹配 optimizer-step budget 的 `Source-SFT π0.5` baseline；它在 test 不看 held video。

## Information wall

| 阶段 | target video | target action | target reward | 更新对象 |
| --- | --- | --- | --- | --- |
| 当前 generic π0.5 test | 否 | 否 | 只读最终 success | 无 |
| Writer source training | source one-video | source，仅 loss | 可选 | Writer |
| held zero-interaction | one-video | 否 | 否 | 无 |
| held task-local RL | run-start one-video | 否 | budgeted | task LoRA |
| direct LoRA oracle | 可选 | 是 | 否 | task LoRA |

## Claim boundary

最小 claim 是：在同 embodiment/simulator family 中，只有 language + action-hidden teaching video 时，EMBER 产生的 LoRA 比无视频/无信息初始化表现更好；若 matched reward adaptation 进一步更快或终点更高，再增加适应效率 claim。

不声称 Writer 必须是 task-local optimizer，也不声称 bank、geometry、subspace 或固定的多任务通用 LoRA。
