# Prompt for the Next EMBER Session

将下面整段复制给新的 Codex session。

---

你现在在 bciA100 上接手独立研究项目 EMBER。

工作区：

```text
~/EMBER
/data/ymdai/projects/EMBER
```

远程：

```text
git@github.com:LinFyM/EMBER.git
https://github.com/LinFyM/EMBER
```

不要修改、依赖或混入 MemLLM。不要从 Git 历史恢复已退役的 60/15/15、Gate recovery、bank/geometry 或旧 runner。

## 先建立完整 Goal

先调用 `get_goal`。若没有 active Goal，调用 `create_goal`，不要设置 `token_budget`，objective 原文使用：

“在最多使用 8 张 NVIDIA A100 80GB、每卡训练平均预留约 10GB、matched fairness、exact-resume、train/validation/test 隔离和 shared-frozen held 约束下，把 EMBER 从当前 Gate -1 通过但带残差、Gate 0 通过但覆盖有限的状态推进到完整可复现实证：仅使用 LIBERO-90，以 specification-only、outcome-blind 方式封存同分布 70/10/10 task split；在全部 70×50 条成功 teacher episodes 上从通用 SmolVLA 训练一个共享 source embodiment base；训练 language + 任意数量和长度的 action-hidden teaching videos → Writer → 完整 task-specific LoRA 的 cold start，使其在多个不同类别未见 validation tasks 上明显超过 frozen base；完成只更新 Writer 的 source-only Writer-only RL；完成从最佳 Writer initialization 与 matched zero/identity initialization 出发的 ordinary task-local LoRA RL；完成 source-only reward/meta outer learning；在 Writer、base、encoders 和所有共享状态冻结的 test/held tasks 上只允许预声明 task-local LoRA reward adaptation；机制成立后从规定初态用全部训练数据重训所有方法，完成完整任务、seeds、强 baselines、统一 test 和 OpenVLA-OFT 规模确认。全过程不使用 bank、geometry、shared update subspace、residual escape 或额外 shared trainable adapter；任何前置环境、代码、split、source base、单一阶段或局部正结果都不能单独触发 Goal complete。”

创建后再次 `get_goal` 核验 objective 完整且没有 token budget。

## 必读

先完整阅读根目录 `AGENTS.md`，然后按其顺序阅读：

1. `README.md`
2. `docs/execution_brief.md`
3. `task_plan.md`
4. `findings.md`
5. `progress.md`
6. `docs/concept.md`
7. `docs/decisions_and_open_questions.md`
8. `docs/novelty_and_landscape.md`

`docs/expert_plan.md` 是历史原文，不是活动 authority。不要重新引入其 bank/geometry/60-15-15。

## 当前事实

- 当前活动树已经清理；旧可执行路径只在父提交 `999df28` 中作 provenance。
- 当前没有新 70/10/10 split、活动训练 config、正式 source base 或可续用 Writer checkpoint。
- 保留的实现内核是 LIBERO HDF5/task factor audit、full-video variable-episode Writer model/data/topology、runtime/gallery。
- 旧 Writer 曾在旧 source protocol 上得到真实 utility，但 validation 泛化稀疏；所有旧 checkpoint 与新协议不兼容。

## 第一批动作

1. `cd ~/EMBER && git pull --ff-only origin main && git status --short --branch`。
2. 只读检查 `nvidia-smi`、GPU 进程所有者、CUDA/driver、Python/PyTorch、磁盘和已有 cache。不得干扰无关进程。
3. 使用 `src/ember/libero_task_factors.py`，只读 90 条 language/scene，建立 deterministic role-aware factor table。
4. 在任何新协议 policy outcome 前生成同分布 70 train / 10 validation / 10 test：
   - val/test 每个 task-relevant role 在 train 有多个近邻；
   - scene/category/composition difficulty 大致同分布；
   - exact task/composition 互斥；
   - 只用 specification，不用 action/reward/proprio/terminal/normalization/policy result。
5. 永久封存 task IDs、factor rules、algorithm/seed、manifest 和 hashes。
6. 基于锁定的 `lerobot/smolvla_base` 建立一条最小 canonical source-base runner：
   - 全部 70×50 success episodes；
   - 一个共享 multi-task source embodiment base；
   - 成熟 SmolVLA action expert + necessary projections；
   - task-mixed deterministic no-replacement cycles；
   - 8-GPU DDP，一卡一 rank；
   - 先短测吞吐，再运行约 30 分钟；
   - checkpoint 在估算总 steps 的 1/3、2/3、3/3，并覆盖全部 70 tasks；
   - 只用 source/train evidence 选 base。
7. 用 LIBERO 官方 task suite、固定 init states、max horizon 400、SmolVLA execution horizon 50 测 source base；在 val/test 同时生成 frozen-base 与 target-action-supervised direct-LoRA 固定参考，但 test reference 不得驱动任何选择；不要恢复旧 h16 主协议。
8. 适配当前 Writer core 到新 source base、新 split 和每 task 全部 50 条完整 action-hidden videos：
   - frozen VLM feature cache；
   - Writer encoder/fusion/decoder 可训练；
   - source actions 只进 functional loss；
   - 37 targets、rank32、alpha16、dropout0；
   - Writer/direct/RL 全部同 LoRA 空间。
9. 冷启动跨 70 tasks 混合训练，总反馈预算通常不超过约 90 分钟，按 thirds exact-resume。
10. 在 10 个跨类别 validation tasks 上比较 frozen base / Writer / direct LoRA oracle，报告每任务原始成功数；每 task 先覆盖 50 个标准 init states，必要时加第二 policy RNG。
11. 在按类别预先固定的少量 train tasks 上额外比较 base / EMBER / direct LoRA，判断 Writer 是 source acquisition 不足还是主要不能泛化；这个诊断不替代 validation，也不能用于改 split。

## 后续顺序

Writer cold start → Writer-only RL（只更新 Writer）→ validation 选最佳 Writer → matched zero-init vs Writer-init task-local LoRA RL → source-only outer learning → 方案冻结 → 全数据多 seed 重训 → 完整 baselines/test → OpenVLA-OFT。

task-local RL 是每个目标 task 单独更新 LoRA。test 上允许使用预声明 reward budget，并可按预算内 adaptation reward 选择每 task checkpoint；规则、interaction/update budget 和 fresh evaluation 必须先在 validation 固定。不得用最终 test evaluation rows 反选。

快速核心 baselines：

- frozen source base；
- EMBER zero-interaction；
- direct task-local LoRA SFT oracle；
- source base + ordinary zero-init LoRA RL；
- best Writer init + identical LoRA RL。

最终再补 HyPoGen/DISC-style language generator、ViVLA/DAML-style video adaptation、direct language+video conditioning、retrieval/average 和必要 WTL/RIPT-style reward adaptation。不要跑 standalone Language-only Writer/Video-only Writer。

## 执行风格

- 真实实验优先，最小化脚手架。
- smoke 只看 mechanics，不看小分母性能。
- training 尽量用满 8 张合法空闲卡和真实显存；evaluation 优化有效 rollout/秒，不用 dummy tensors。
- 一次科学反馈通常 1–2 小时内；task-local RL 是所有 tasks 的总 wall-clock，不是每 task 90 分钟。
- checkpoint 必须保存完整训练和 sampler/RNG/interaction 状态。
- meaningful 结果后更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push。
- 如果不能启动，只报告一个具体可复现失败和最小修复；不要用“还能更严谨”当阻塞。

完成 Goal 核验和只读状态检查后，直接推进 Phase A，不要停在复述，也不要等待形式确认。

---
