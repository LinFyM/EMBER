# EMBER

EMBER 研究能否把廉价但没有机器人 action 标注的教学视频，编译成可直接执行、也可继续利用环境反馈适应的 VLA 参数：

```text
task language + one action-hidden teaching video
                    -> shared Writer
                    -> complete task-specific LoRA
```

## 当前协议

- 基座改为通用预训练 π0.5，不默认先训练 source embodiment base。
- benchmark 使用 LIBERO-Spatial、Object、Goal、Long（官方 suite 名为 `libero_10`）共 40 tasks。
- 开发期每 suite 固定 6 train / 2 validation / 2 test，总计 24/8/8；validation 选完方法后，未来最终重训才合成 32 source / 8 test。
- EMBER 训练和测试都只看一条 teacher video。source 训练时，同 task 的视频与 action-supervised episode/chunk 独立随机抽样，不做同 episode 配对。
- held rollout 每次随机抽一条 teacher video，再由 Writer 生成 LoRA。
- 原“Writer cold start”正式改名为 `Action-Supervised Writer (AS-Writer)`。
- `Reward-Trained Writer (RL-Writer)` 是从随机初始化（或仅极短、预声明的 AS warm-up）开始的独立 source-reward 路线，不默认接在 AS-Writer 后面；它专门检验没有 teacher actions 能否训练 Writer。
- Writer 直接生成的 LoRA 若已足够强，不强制继续 RL；matched task-local LoRA RL 保留为后续第二实验。
- 最终增加 `Source-SFT π0.5` baseline：同样使用合并后的 32 source tasks，按 AS-Writer 的 optimizer-step budget 微调 π0.5，但 test 时不看 held video。
- 不使用 bank、geometry、shared subspace、residual escape、额外 shared adapter 或 MemLLM。

## 当前执行边界

通用 `pi05_base` 的预封存 8-task zero-shot feasibility 已完成：8 tasks × 50 fixed states 均为 `0/50`，总计 `0/400`。模型未训练，test teacher actions 未读取，也未使用已经在 LIBERO-40 上 fine-tune 的 `pi05_libero`。当前按 owner 要求停止，等待讨论是否建立 source-side action adaptation/base。

split、评测合同与结果 seal 封存在 `configs/libero_24_8_8_v1/`。旧 `configs/libero90_70_10_10/`、SmolVLA 训练结果和旧 Phase A–F 只作历史证据。

## 阅读顺序

1. `AGENTS.md`
2. `docs/execution_brief.md`
3. `task_plan.md`
4. `findings.md`
5. `progress.md`
6. `docs/concept.md`
7. `docs/decisions_and_open_questions.md`
8. `docs/novelty_and_landscape.md`

`docs/expert_plan.md` 是历史原文，不是活动 authority。
