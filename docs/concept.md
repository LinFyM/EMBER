# EMBER Concept

## 一句话定义

```text
Writer(task language, exactly one action-hidden teaching video)
    -> complete task-specific LoRA for a frozen π0.5-LIBERO source policy
```

EMBER 把不能直接用于目标 action-SFT 的视频任务知识编译成策略参数。Writer直接生成的zero-interaction LoRA是第一主结果；reward practice可以增强，但不是方法定义。

## 共同地基

generic π0.5 已在目标8 test tasks上得到 `0/400`，说明它没有足够LIBERO embodiment competence。LIBERO-90与目标40 tasks的specification-only overlap过滤、剩余71 source tasks×50 action episodes联合训练和共享π0.5-LIBERO source base冻结均已完成。所有主方法从该base开始；`pi05_libero`因读过目标40 actions而禁止使用。

source base只负责基本camera/controller/robot/action与通用技能，不追求先解决目标任务。它要在全部40目标tasks的快速screen上开始产生部分成功，从而给Writer和reward exploration一个真实可用的地基。

## AS-Writer

每个source update：

1. 每rank从24 development train tasks的全局均衡schedule取得一个task；
2. 为该task visit抽1条action-hidden teacher video；
3. 该video经过one-shot Writer，只生成1套完整LoRA；
4. 采`B_a`条独立同task agent observation/action chunks；
5. 全部action queries都在这套LoRA下做一次frozen source-base functional
   forward；
6. `B_a`个action losses直接求均值并只更新Writer；下一step更换task。

video与action sample不要求配对，Writer不能靠逐帧复制目标action。同一套
LoRA同时接收`B_a`条不同初态、episode和action chunk的共同梯度；task被后续
访问时轮换teacher video，跨step SGD平均不同视频的偶然低层细节。推理和held
evaluation仍严格每次一条video。
held evaluation每rollout随机采一条正确task视频，报告对teacher-video分布的
性能，不挑最好video。

当前v5把视频理解分成两类互补表示：

- Semantic Core从被语言条件化的PaliGemma image-position hidden中读取对象、
  关系、场景和整体操作，并在结构上对同一组帧的shuffle不变；
- Causal Procedure从固定native suffix下的Action Expert interaction hidden
  读取任务如何按阶段推进，不再预测7D action trajectory。

Core先形成稳定LoRA content，Procedure再以zero-init refinement加入正确的有序
过程。完整实现合同见
[`action_forecast_writer_v5_design.md`](action_forecast_writer_v5_design.md)。

## RL-Writer

RL-Writer不是完整AS-Writer的默认后训练阶段。当前路线从新架构规定初态做短、task-balanced AS cold start，直到24个development-train tasks各在official random-reset rollout中至少成功一次；随后关闭action数据入口，跨source tasks只用官方环境reward训练video-to-LoRA映射。cold-start必须报告teacher-action consumption、每task first-success step和wall，不能加载完整AS best冒充独立RL路线。

## Source-SFT

Source-SFT在同一frozen source base上，以24/32 source task actions联合训练一套shared multi-task LoRA。它在held task只看language/current observation，不看teacher video。它和AS-Writer各自按validation选最佳，不强制相同步数；对比回答“一般source action adaptation”与“额外读取held video”之间的差异。

## Seen 与 wrong-video

seen-task比较检验方法是否先在训练过的目标task distribution上学到能力；它不能替代held泛化。

wrong-video是必要视频因果对照：保持正确language、执行task、init state和policy RNG，只把Writer输入换成另一suite的视频。若wrong-video LoRA与correct-video同样有效，说明增益可能主要来自language或通用adapter；只有correct明显优于wrong，才能支持视频内容被使用。

## Test-task reward adaptation

task-local RL只在最终test阶段开始。每个test task本身是adaptation training domain，因此可直接在该task上调优并训练到曲线接近最佳，不需要在validation上预先冻结算法。

三臂从同一source base开始：identity LoRA、AS-Writer LoRA、RL-Writer LoRA。每个task/adaptation seed随机选一条teacher video，两个Writer臂共用该视频并固定生成的初始化LoRA；随后只更新task LoRA。训练与checkpoint选择使用官方random BDDL reset reward rollouts；固定50 states只用于独立fresh evaluation。

## Privileged oracle

最终direct-action oracle不是per-task LoRA。它使用8个test tasks×每task50条teacher action episodes，联合训练一套shared multi-task LoRA。它证明同一LoRA空间在目标actions可见时能达到什么水平，但不属于与EMBER相同的信息条件。

## Information surfaces

| 方法/阶段 | 目标视频 | 目标action | 目标reward | 更新对象 |
| --- | --- | --- | --- | --- |
| frozen source base test | 否 | 否 | 只读评估success | 无 |
| Source-SFT source training | 否 | source task actions | 否 | shared LoRA |
| AS-Writer source training | one source video | source task actions仅进loss | 否 | Writer |
| RL-Writer source training | one source video | 明确报告的短AS cold start；转型后否 | source reward | Writer |
| held zero-interaction | one held video | 否 | 否 | 无 |
| test task-local RL | run-start one held video | 否 | test task reward | task LoRA |
| joint direct oracle | 可忽略 | 8 test tasks actions | 否 | one shared LoRA |

## Claim boundary

核心claim是：对于一个已有基本LIBERO能力、但没有目标task action supervision的共享source policy，正确held teaching video生成的LoRA比无视频source-policy adaptation更有用，并且明显优于cross-suite wrong-video生成的LoRA。若Writer初始化进一步让test-task reward training更快或终点更高，再增加adaptation claim。

不声称RL是EMBER必需组成，不声称target-action oracle是信息匹配baseline，也不使用bank、geometry、shared subspace或额外shared adapter。
