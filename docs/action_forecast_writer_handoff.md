# Action-Forecast Writer execution handoff

状态：2026-07-24 owner 最终对齐，供新的独立 session 直接接手。

本文件是当前 Writer 子任务的活动设计与执行 authority。它覆盖此前
Action-Memory / temporal-RoPE Writer 的活动实现口径，但不改写那些实验的
历史结果。若旧配置、旧测试、`task_plan.md` 的已完成历史条目或早期设计文档
与本文件冲突，以本文件和 owner 最新指令为准。

## 1. 当前目标和停止条件

当前先完成一个闭合的 Action-Forecast Writer 子任务：

1. 用唯一 canonical 新架构原位替换旧 Action-Memory Writer；旧活动代码、
   schema、配置和只服务旧架构的测试退役，历史由 Git 和已有结果保存。
2. 在 GPU 0–3 上实测并封存训练的每 rank action-query batch、
   `frame_microbatch_size`、frame stride，以及评测的最佳并发参数。
3. 四卡训练按约 30 分钟一段推进；每段保存四个等间隔 checkpoint，优先对
   第二和第四个做完整 validation rollout，趋势不清时再补第一/第三个。
4. 找到 AS-Writer 的 validation 最佳 checkpoint。主要门槛是不要明显落后于
   owner 指定的 rank-128 Source-SFT 参考 `122/400`，最好超过。
5. 只对 observed-best checkpoint 做 correct-video、cross-suite wrong-video
   和 shuffled/reversed-video 机制诊断。Writer 要真正依赖视频任务内容，并比
   旧 temporal-RoPE Writer 更能感知帧顺序。
6. 若 AS 同时通过性能与视频/顺序特异性门槛，才推进 cold-start RL-Writer；
   若经过最小、证据驱动的架构/训练修正仍过不了，保留完整证据并停止汇报。
7. RL-Writer 先做一个独立、短 AS cold start，直到 24 个 source/train tasks
   每个在 official random-reset reward rollout 中至少成功一次，再切换为纯
   reward 更新。它不是从完整 AS-Writer best checkpoint 继续训练。
8. RL 阶段先在 train tasks 上训练到曲线平台，再对少量合适 checkpoint 做
   validation，尽可能定位最佳点。

本子任务完成前不要自动继续 final-32、test task-local RL、joint target-action
oracle 或 ViVLA。长期 EMBER Goal 仍存在，但本轮交接的边界是上述 AS/RL
Writer 子任务。

## 2. 不可改变的科学合同

- 输入始终是正确 task language 加恰好一条 action-hidden teacher video。
- Writer 输出 sealed π0.5 LoRA 空间中的完整 rank-16 task-specific LoRA。
- frozen source base、24 development-train tasks、独立 video/action episode、
  source normalization 和 functional policy contract不变。
- Writer 不得接收 teacher action、proprio、reward、terminal、task ID、
  filename 或隐藏 normalization。AS action 只进入 frozen policy 的 functional
  behavior loss，不进入 Writer。
- policy 在 AS 训练、correct/wrong/shuffled 评测的所有主分支中都接收正确的
  evaluation/action-query language；替换的只有 Writer 视频。
- 初始 AS 只用 normal positive functional action loss，不先加 contrast loss。
- 不使用 `pi05_libero`、MemLLM、bank、geometry、shared update subspace、
  residual escape、额外 shared trainable adapter 或未 merge 的 source LoRA。
- RL rollout 与 checkpoint selection 只用 LIBERO official BDDL/random reset；
  fixed `.pruned_init` 只用于与 RL 数据隔离的 fresh evaluation。
- 四卡就是四个 native DDP ranks；不使用 gradient accumulation 模拟八卡。
- 只有一个实际 `action_query_batch_size_per_rank`。不得再暴露一个与它不同的
  `functional_policy_microbatch_size`。`frame_microbatch_size` 是同一视频内
  T 帧穿过图文/action-forecast 路径时的内存切片，不改变 optimizer batch。

## 3. 唯一 canonical Action-Forecast Writer

### 3.1 变长视频与图文理解

- 以固定物理/控制时间间隔采样视频，而不是把所有视频均匀压成固定帧数。
- 视频长度 `T` 保持可变；batch 内只为张量化做局部 padding，并始终携带
  padding mask 和真实原始 frame index。
- 首轮必须比较 stride 5 与 stride 10。LIBERO 约 20 Hz 时分别对应约
  0.25 s 与 0.5 s。stride 5 的 24-task 数据统计为：平均 35.6 帧、
  中位 30、p95 70、最大 105。
- 每个采样帧和完整 task language 进入 π0.5 的视觉语言 backbone。语言必须
  使用 frozen PaliGemma 经过上下文处理后的表示，不能只取 embedding table。
- frozen SigLIP/full projected image tokens 可以缓存；task token IDs、固定语言
  embedding、固定 flow noise、对齐索引也可缓存。因为 VL Meta-LoRA 和
  imagined-state path 会更新，contextual PaliGemma prefix 不能跨 optimizer
  step 缓存。

### 3.2 Visual State Head 与 VL Meta-LoRA

- 用 8 个 coordinate queries 单向读取该帧的完整 image tokens，得到一个
  8 维连续 agent-centric imagined state。它是 action-hidden teacher video
  中模型“想象自己处在教师位置时”的连续 pseudo-state，不宣称是有监督的
  真实机器人坐标。
- imagined state 经过连续/Fourier/MLP state embedder 变成 8 个可微的
  PaliGemma-width virtual state tokens，插入原生 state-prefix 对应的位置。
  不要把连续状态离散化后再走 tokenizer，否则梯度被截断。
- PaliGemma 18 层 q/k/v/o 使用 Writer 内部 identity-init VL Meta-LoRA，
  首选 rank 4。作用是把 teacher-view 图像、正确语言和 virtual state 融合成
  source policy Action Expert 能理解的 agent-centric prefix KV。
- VL Meta-LoRA 只存在于 Writer 的教师视频理解路径；生成的公开 task LoRA
  执行时不携带它。

### 3.3 Action Expert forecast

- Action Expert 18 层 q/k/v/o 使用 Writer 内部 identity-init Action
  Meta-LoRA，首选 rank 8。它将上述 prefix 转换为 agent-centric action plan。
- 对每个采样帧执行完整 π0.5 flow inference：固定 `num_flow_steps=10`，
  action horizon 50。十次 flow 是同一个 18-layer Action Expert 的十次
  迭代，不是十层，也不是只预测十个动作。
- 同一 teacher video 的所有帧从完全相同的固定 `[50,32]` flow noise 开始，
  避免随机噪声被误当成 frame-to-frame revision。
- 每个 frame 的 PaliGemma prefix 只算一次；prefix KV 在十次 flow 迭代中
  原地复用。不要每个 flow step 重算视觉语言 prefix。
- 只保留最终 normalized `[T,50,7]` action plans。不要保存 10×18 层 hidden
  states，也不要把 imagined state 另行喂给 temporal encoder。

### 3.4 同绝对时刻的 Plan / Revision tokens

令采样帧原始控制步为 `t_i`，该帧预测为 `P_i[k]`，`k=0..49`。一个预测对应
的绝对动作时刻是：

```text
u = t_i + k
```

不能只比较一对相邻 chunk，也不能把同一未来时刻的四十几个预测全部平均掉。

- `Plan_u` 使用在时刻 `u` 前最近的一帧产生的 receding-horizon 决策：
  `i*(u)=max{i | t_i <= u}`，`Plan_u=P_i*(u-t_i*)`。这等价于把每次最新计划
  的可执行前缀拼起来，表示“模型在这个时刻最终决定做什么”。
- 对所有覆盖同一 `u` 的连续 forecasts，构造有序 revision：
  `Delta_i(u)=P_{i+1}[u-t_{i+1}]-P_i[u-t_i]`。
- 一个共享 Revision encoder 读取 old action、new action、delta、两个 lead
  time、真实 `Delta t`、revision count 和稳定性统计，汇总成一个
  `Revision_u`。它表示随着新 teacher frame 到来，同一绝对未来动作被怎样
  修正。没有 revision 的边界使用 learned `no_revision` token 加 count/mask，
  不能用全零冒充稳定。
- 每个绝对时间点输出 `[Plan_u, Revision_u]` 两个 width-256 tokens，得到
  变长序列 `[U,2,256]`。两类 token 使用相同 absolute-time RoPE，再加
  token-type embedding。
- 对未来人类视频，核心仍是按物理时间对齐：视频帧间隔对应机器人控制周期
  中能执行多少 actions，而不是把 “5” 写死。当前 LIBERO 可以严格用原始
  frame index 对齐。

### 3.5 Temporal encoder 与单向 LoRA query decoder

- Temporal encoder：width 256、8 heads、2 blocks、真实 absolute-time RoPE；
  接受可变 `U` 和 padding mask，不固定视频长度。
- 使用 320 个 learned LoRA queries：
  - `18 layers × 16 rank slots = 288` 个 expert layer/rank queries；
  - 16 个 `action_in_proj` rank queries；
  - 16 个 `action_out_proj` rank queries。
- 一个 expert layer/rank query 同时服务 q-A/q-B/v-A/v-B factor heads；不要
  为四个 factor 各复制一套 query。
- query decoder 用两个 block，包含 query self-attention 和显式
  cross-attention `Q(query) -> K/V(procedural memory)`。这是单向读取：
  temporal/video memory 不反向读取 320 个 output queries。不要用无 mask 的
  拼接 self-attention 制造不必要的双向计算。
- factor heads 生成当前 sealed rank-16 PEFT tensors：18 层 q/v，加
  `action_in_proj`、`action_out_proj` 的 A/B factors。普通 conditional bias
  允许存在；final factor projections 的 weight 和 bias 从零初始化，使 fresh
  public task LoRA 严格 functional identity。
- 不增加独立公共 LoRA 支路。这里不是全局 `bias=False`；conditional modules
  的 bias 可以学习，但所有输出必须由当前 language/video procedural memory
  经 query decoder 产生。

### 3.6 参数预算

目标是让 Writer trainable parameter count 与 rank-128 Source-SFT 的
`10,297,344` 接近，不能再用 10× 参数量解释优势。当前预算草案：

| 模块 | 目标参数量 |
|---|---:|
| Visual State Head + continuous state embedder | ~1.09M |
| VL Meta-LoRA，PaliGemma q/k/v/o，rank 4 | ~0.922M |
| Action Meta-LoRA，Action Expert q/k/v/o，rank 8 | ~1.253M |
| Revision encoder | ~1.05M |
| Temporal encoder | ~1.57M |
| 2-block one-way LoRA query decoder | ~2.10M |
| Factor heads | ~2.19M |
| embeddings/norm/bias | ~0.10M |
| 合计 | ~10.27M |

正式实现必须从真实 model/config 计算参数量；允许在不改变上述信息流的前提下
微调 hidden widths，使总量接近 `10.297M`，并记录每个模块的真实 count。
Writer 生成的 public rank-16 task LoRA 本身仍为 `1,287,168` scalars。

## 4. 代码所有权与退役边界

保留一条 runner，不创建 `v4`/`new`/`experimental` 平行执行路径：

- `scripts/train_as_writer.py` 仍是唯一 AS 入口。
- `scripts/evaluate_pi05.py` 仍是唯一 π0.5 rollout 入口。
- `src/ember/writer/model.py` 仍拥有完整 LoRA schema/decoder。
- 将 `src/ember/writer/action_memory.py` 退役，由一个职责清楚的
  `action_forecast.py`（或同等单一 owner）替换。
- `src/ember/writer/temporal.py` 原位改为 Plan/Revision variable-time owner。
- `as_contract.py`、checkpoint schema、training/inference/evaluator 调用点和
  targeted tests 同步更新为一个 `action_forecast` schema。
- 旧 `configs/pi05_as_writer_action_memory_v1.json` 由新的 canonical
  `configs/pi05_as_writer_action_forecast_v1.json` 替换；旧活动配置和只验证
  Action-Memory internals 的测试删除。历史结果由 Git、`findings.md` 和
  `progress.md` 保存，不创建 in-tree archive。
- 先用 `rg` 建立 callers/import/checkpoint ownership map。完成后要求活动
  source/config/test 中不再有 Action-Memory schema/import；provenance 文档
  可以保留历史名称。

结构变化受 `code-architecture-gate` 约束：优先 replacement/deletion，入口
保持薄，新增文件按职责拆分，避免让现有 700–800 行 legacy 文件继续膨胀。

接手时先完整阅读这些实际 owner，而不是从历史 runner 猜接口：

- `src/ember/writer/action_memory.py`：待退役的frame-prefix/Meta-LoRA owner；
- `src/ember/writer/model.py`：sealed LoRA tensor specs、factor ownership和
  `CompleteLoRAWriter`；
- `src/ember/writer/temporal.py`：待替换的旧variable-length temporal owner；
- `src/ember/writer/as_contract.py`、`as_step.py`、`training.py`、
  `checkpoint.py`、`inference.py`：config、functional gradient、DDP、
  exact-resume与评测装载合同；
- `tests/test_writer_model.py`、`test_writer_training.py`、
  `test_writer_checkpoint.py`、`test_writer_functional.py`：需要保留的机械不变量
  与需要随旧架构退役的内部断言；
- `.venv/lib/python3.12/site-packages/lerobot/policies/pi05/modeling_pi05.py`：
  当前 pinned LeRobot π0.5真实实现。重点核对`PI05Pytorch.embed_prefix`、
  `sample_actions`、`denoise_step`、`paligemma_with_expert`和
  `gemma_expert`，不要另写一个近似flow sampler。

## 5. 训练实现与显存/吞吐优化

- action-query policy batch 与 frame microbatch 是两个正交维度：
  `action_query_batch_size_per_rank` 是一个 task/video adapter 下的独立
  functional action queries；`frame_microbatch_size` 只是同一视频内部逐帧
  forecast 的切片。二者名字要直接表达这个区别。
- 不保留第二个 policy microbatch 参数；若显存只容纳每 rank 8 queries，
  logical/physical batch 都是 8。
- 先使用 BF16、fused SDPA/FlashAttention、静态安装的 Meta-LoRA、prefix KV
  reuse、固定十步 flow loop、`output_hidden_states=False`。
- 对 frame forecast 大块使用 activation checkpoint/rematerialization；只保存
  最终 action plans 和构建 Plan/Revision 所需的最小张量。
- 当前 functional LoRA leaf-gradient 机制可保留。若 Writer graph 与 frozen
  policy functional forward 同时驻留仍 OOM，可使用 exact two-pass replay：
  第一遍生成 detached adapter leaf 并求 policy 对 adapter 的梯度，释放 policy
  graph；第二遍重放可微 Writer，用 VJP 回传。它增加计算但不能改变 loss、
  sample/RNG 或 optimizer step 定义。
- 固定 checkpoint 的 evaluation 应按 checkpoint/task/language/video hash
  缓存生成的 public LoRA；复用已实现的 `per_sample_lora_batched_replan`，
  不退回逐 rollout materialize + sequential replan。

## 6. Profile 与封存顺序

先做最小 CPU/单卡 contract smoke，再立即进入四卡真实 profile：

1. shape、padding、绝对时间对齐、no-revision boundary、identity-init、
   frozen base、Meta-LoRA gradients、checkpoint exact-resume。
2. 典型视频和 p95 长视频分别实测 `frame_microbatch_size` 候选 1/2/4（能安全
   再试 8），记录 step wall、峰值 allocated/reserved 和 OOM。
3. 对稳定的 frame microbatch 实测每 rank action-query batch 候选，例如
   4/8/16；不做 gradient accumulation。
4. stride 5 与 10 都测真实完整 forward+backward。选择依据是有效
   action queries/s、长视频稳定性和最小 order/action-specificity smoke，
   不能只按显存或单步速度。
5. 用户允许极长序列把原先平均预留约 10GB 的缓冲吃到只剩几 GB，但正式配置
   仍必须在 p95/最大视频上稳定且不依赖 dummy allocations。
6. 评测基于当前 4 replicas/GPU、8 envs/replica 的稳定点，实测邻近组合和
   adapter 预生成/cache。旧 6 replicas/GPU 在 Writer 视频编码阶段 OOM；
   只有新路径通过真实 profile 才能采用，不能凭空宣称更快。
7. 以真实 rollouts/s 和完整 400-rollout wall 选评测配置；显存利用率不是
   主要指标。

profile 后把唯一选中值写进 canonical config、manifest 和 docs，删除临时
profile-only开关。估算不能替代实测。

## 7. AS 分段训练与 checkpoint 选择

- 只用 GPU 0,1,2,3，一卡一个同角色 DDP rank。GPU0 不承担额外 model
  server/controller。
- 每个正式 segment 目标 wall 约 30 分钟；根据实测 steady step time换算该段
  steps，并在 25%、50%、75%、100% 四个位置保存完整 checkpoint。
- 每段完成后优先完整评测第 2 和第 4 个 checkpoint。若第 4 个明显更好或没有
  下降，exact-resume 再加一段；若趋势需要更细粒度，再补第 1/3 个。
- val functional loss只作很弱的辅助线索。最终 best 由相同 paired
  `8 tasks × 50 fixed states` closed-loop success决定；报告 per-task counts、
  paired flips 和重复评测的不确定性，不能把一个 noisy 400-rollout点写成
  精确上限。
- 不因为 GPU 数变化机械缩放 step；steps、global queries、task/video
  conditions、wall/GPU-hours都同时记录。
- 每次先启动可运行训练/评测，再在不修改其 import/config/output contract 的
  前提下并行更新文档、parser、下一阶段代码。
- 只有 observed-best checkpoint 做 correct/wrong/shuffled/reversed 诊断。
  correct/wrong rollout保持 task/init/policy/video seeds配对；shuffled/reversed
  保留完全相同帧集合，只变顺序。

当前旧架构参考不是新模型的初始化：

- Action-Memory temporal-RoPE best：step 400，correct `108/400`。
- 同 checkpoint 的 cross-suite wrong video 会明显改变 adapter，但倒序/乱序
  的 effective-LoRA relative L2 中位数仅 `0.00937/0.00699`，说明近似
  bag-of-states。
- owner 指定 SFT 比较门槛：`122/400`。四卡 rank-128 SFT 已观测的当前轨迹
  best 为 step 700 的 `108/400`；`122/400` 来自此前 incumbent。新 Writer
  的目标仍按 owner 最新口径尽量达到/超过 122，而不是只超过 108。

## 8. RL-Writer 接续合同

只有 AS 性能和视频/顺序特异性都通过后执行：

1. 新建独立 RL-Writer run，从新架构的规定 identity initialization开始。
2. 做短、task-balanced AS cold start；不是加载完整 AS best。持续做官方
   random-reset reward screen，直到 24 个 train tasks 每个至少有一次真实
   success。记录每 task first-success step、teacher action queries 和 wall。
3. 达到全 task coverage 的 checkpoint 后，冻结 cold-start action数据入口，
   转为纯 official env reward 更新 Writer。
4. RL rollout 保存 env/policy/worker RNG、seed schedule、interaction cursor、
   per-task reward rows、optimizer/scheduler和exact-resume状态。
5. 在 train tasks 上按多个任务覆盖和aggregate曲线判断平台，避免只由一两个
   易任务支撑。只对少量平台附近checkpoint运行完整validation。
6. 评测 correct-video，并在 selected best 上做一次 cross-suite wrong-video；
   不读取 validation actions，不用 fixed states训练或选 RL checkpoint。

现有 `train_rl_writer.py` 和 `src/ember/pi05_rl_writer*` 只能复用通用
RNG/checkpoint/env-pool机制；任何绑定旧 Action-Memory schema、旧冷启动含义
或旧数据墙的部分必须适配/替换，不能原样恢复。

## 9. 当前仓库与实物状态快照

以下是交接时只读快照；新 session 启动后必须重新核验：

- canonical checkout：`/data/ymdai/projects/EMBER`
- branch/HEAD：`main` / `b78584ab05e7f639cf1c022fdf457b3a971d64e6`
- 当时 `main == origin/main` 且 worktree clean。
- source base run：
  `/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722`
- source checkpoint：
  `/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`
- tokenizer：
  `/data/ymdai/ember_data/openpi/paligemma_tokenizer.model`
- target data root：
  `/data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`
- 现有17GB cache
  `/data/ymdai/outputs/ember/pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722`
  只存每帧`16×2048`粗空间pool，不是完整SigLIP/projected image tokens，不能
  冒充新架构所需full-token cache。stride5的24-task full-token BF16 cache约
  42GiB（int8/FP8约21GiB）；先实测online路径和量化误差，再决定是否构建，
  并纳入500GB peak预算。
- 交接时 `/data/ymdai` 占用 `278,857,052,160` bytes；`/data` 可用约
  `3,084,411,146,240` bytes。个人硬上限仍为 500GB。
- 交接时 GPU 0–3 各 `0 MiB`，GPU 4–7 被另一用户进程使用约 37GB/card。
  这些都是瞬时状态，launch前必须重查。
- tmux `ember_as_bias_r4_s3200` 仅剩一个空 bash shell，没有训练子进程；
  不依赖它，也不因它假定任何 output仍在写。
- 仓库存在多个历史 worktree。不要因“看起来旧”删除；main仍干净且无并发写
  时直接使用 main。只有实际重叠写、branch切换或活跃进程import同一 checkout
  时才新建隔离 worktree。

## 10. 启动顺序和证据要求

新 session 应：

1. 先 `get_goal`；若无 active Goal，使用本文件第1节和 owner原始目标创建一个
   无 token budget 的完整 Goal。
2. `git pull --ff-only origin main`，检查 status/HEAD/worktrees/活动进程。
3. 完整阅读根 `AGENTS.md` 规定的 authority 文档，再完整阅读本文件。
4. 读取 `code-architecture-gate`；建立 imports/callers/schema/checkpoint map。
5. 先实现最短可运行垂直切片并做 shape/gradient smoke；不要先花数小时只写
   文档或大规模cache。
6. GPU launch 前使用 live GPU/storage preflight；不得 kill/reset/干扰他人。
   owner指定使用0–3不等于授权终止别人进程。
7. expensive canonical training 前使用 formal launch contract，保存 exact
   command、commit、config/hash、topology、预计storage peak和output ownership。
8. meaningful milestone 更新 `task_plan.md`、`findings.md`、`progress.md`；
   不提交 dataset、weights、cache、checkpoint或凭据。
9. 每个可复现里程碑做 task-scoped diff/tests，commit并push main。
10. 只有本文件第1节全部完成才把该子任务 Goal 标记 complete；代码完成、
    smoke、loss下降或一个 validation点都不够。
