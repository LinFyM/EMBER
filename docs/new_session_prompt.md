# Prompt for the next EMBER session

你现在在 BCI A100 主机上接手独立研究项目 EMBER。

工作区：

```text
/data/ymdai/projects/EMBER
```

远程：

```text
git@github.com:LinFyM/EMBER.git
https://github.com/LinFyM/EMBER
```

这次不要重新开始长期 EMBER 主线，也不要从旧对话猜设计。仓库已经封存了
Action-Forecast Writer 的完整架构和执行合同；你需要直接实现、训练、评测并
推进到本 prompt 指定的 AS/RL Writer 子任务终点。

> 2026-07-25更新：当前active实现是单-token
> `Belief_u=[Plan_u(128)|Revision_u(128)]`、Plan-relative all-covering
> Revision、zero-preserving Temporal和content-only LoRA query decoder；AS
> 只用positive functional loss，不得恢复order-contrast。唯一配置为
> `configs/pi05_as_writer_action_forecast_v3.json`，`frame_stride=5`固定。
> Revision显式强度为frozen source normalization下的原始residual RMS：
> `Revision_u=stopgrad(m_u)*RMSNorm(z_u)`；不得加入`tau`、训练集分位数尺度
> 或其他人工强度超参数。
> 新架构fresh连续训练`0→600`，不中途主动切换评测。旧8-scalar/Fourier、
> adjacent Revision、Plan/Revision interleaving、additive/type routing和
> static-query residual描述只作provenance；精确合同以
> `docs/action_forecast_writer_handoff.md`顶部Belief-v3 override为准。

## 1. Goal

首先调用 `get_goal`。当前应已有 active Goal；核验其 objective 与下文完全一致，
且没有 `token_budget`。若确实没有 active Goal，调用 `create_goal`，不要设置
`token_budget`，objective 原文使用：

> 在 EMBER 中完整实现并核查 owner 已认可的单-token Belief_u Action-Forecast Writer 架构；固定 frame stride=5，只优化训练/评测的 batch 与 frame-microbatch 等效率参数；从新架构随机初态连续训练到 600 optimizer steps（期间可密集保存 checkpoint，但不按约半小时 segment 主动停训或切换评测），随后评测多个 validation checkpoint并对 step-600 做视频顺序特异性诊断。若特异性明确，则继续按较大步长充分探索 AS Writer 的 validation observed-best 与显著峰后下降，并在绝对性能良好后推进独立 cold-start RL Writer；若任一科学 gate 经分析和第一性原理架构修正仍无法通过，则停下向 owner 汇报。全程不引入对比损失，不恢复旧 Action-Memory/平行 runner，仅使用 GPU 0、1、2、3且不触碰4–7。

本 Goal 只在上述 focused AS/RL Writer 子任务全部完成后标记 complete。代码完成、
smoke、loss下降、一个validation点、一个train平台或一个方法阶段都不够。

## 2. 启动与 authority

执行：

```bash
cd /data/ymdai/projects/EMBER
git pull --ff-only origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

确认没有用户未提交修改或活跃进程写同一 checkout/output。不要从 Git 历史恢复
旧 runner、旧 prompt、旧 split 或旧 checkpoint。历史 worktrees 不要因为
“看起来没用”就删除；main clean且无并发writer时直接使用main。

先完整阅读根 `AGENTS.md`，再按其顺序完整阅读：

1. `README.md`
2. `docs/execution_brief.md`
3. `task_plan.md`
4. `findings.md`
5. `progress.md`
6. `docs/concept.md`
7. `docs/decisions_and_open_questions.md`
8. `docs/novelty_and_landscape.md`

然后完整阅读：

```text
docs/action_forecast_writer_handoff.md
```

该 handoff 是新架构的完整活动 authority，包含精确 tensor 合同、模块设计、
参数预算、退役边界、profile矩阵、AS/RL训练和停止规则。无需也不得依赖旧对话。
若旧 Action-Memory 文档、配置、测试或历史 ledger 与它冲突，以 handoff 为准；
历史实验结果保留为 provenance，但不再是活动实现。

这次是非平凡结构替换，开始改源码前读取并应用 `code-architecture-gate`。它只用于
控制单一 owner、文件职责和退役边界，不得变成拖延实现的额外流程。

## 3. 当前事实：不要重做已完成工作

- specification-only source audit 已完成：71个active LIBERO-90 tasks、每task
  50条成功episodes、source-only normalization和hashes均已封存。
- 共享 π0.5-LIBERO source base 已从generic `lerobot/pi05_base` fresh full-SFT
  1,000 steps并冻结；40-task快速screen为`46/320`，覆盖13 tasks和全部4 suites。
- 不要重训source base，不要重新做Phase A，不要使用`pi05_libero`。
- Source-SFT当前focused task不重训。四卡rank-128完整validation曲线
  step100–1100为：
  `81,95,68,78,94,99,108,97,95,104,94 / 400`；
  四卡observed-best是step700的`108/400`。
- `122/400`来自旧八卡rank-128 Source-SFT step400，不是四卡结果；它是所有
  SFT候选的全局incumbent和stretch目标。
- 旧Action-Memory temporal-RoPE Writer step400为`108/400`，有视频任务内容
  特异性，但倒序/乱序effective-LoRA相对变化仅`0.00937/0.00699`，近似
  bag-of-states。它是新架构要解决的问题，不是新模型初始化。
- 当前仓库已实现并profile Action-Forecast Belief-v3架构；shape、gradient、
  identity、freeze、OOM与exact-resume gate已经通过。正式AS下一步是直接从
  fresh identity连续训练到step600，不再重做profile或中途评测。

固定实物路径：

```text
source run:
/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722

source checkpoint:
/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000

tokenizer:
/data/ymdai/ember_data/openpi/paligemma_tokenizer.model

target data:
/data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a

old 8-card SFT 122/400 artifact:
/data/ymdai/outputs/ember/pi05_source_sft_rank128_val8x50_step0400_77ec0ae_g67_r5_20260723

four-card SFT 108/400 artifact:
/data/ymdai/outputs/ember/pi05_source_sft_rank128_ceiling_r4_val8x50_step0700_982c115_g01_r5_20260724
```

已有17GB旧feature cache只保存每帧`16×2048`粗空间pool，不能冒充新架构需要的
full projected image tokens。是否构建约42GiB BF16（或约21GiB int8/FP8）
full-token cache，必须先比较online路径、量化误差和实际吞吐，并纳入500GB cap。

## 4. 不可改变的科学合同

- `correct task language + exactly one action-hidden teacher video -> shared Writer
  -> complete rank-16 task-specific LoRA`。
- frozen source base、24 development-train tasks、source normalization、同task内
  独立video/action episode和functional policy接口保持不变。
- Writer不得读取teacher action、proprio、reward、terminal、task ID、filename
  或隐藏normalization；AS action只进入frozen policy functional loss。
- teacher视频当前只用单条`obs/agentview_rgb`；不要静默加入wrist、第二视频或
  真实robot state。
- policy在AS及correct/wrong/shuffled/reversed评测中始终接收正确task language；
  只替换或重排Writer视频。
- 初始AS只用normal positive functional action loss，不先加contrast。
- Writer生成公开完整rank-16 task LoRA；Writer内部Meta-LoRA只服务教师视频
  理解，执行时不携带。
- 不增加独立公共LoRA支路。conditional module bias允许存在，但public LoRA
  输出必须经过当前language/video procedural memory和query decoder。
- 禁止`pi05_libero`、MemLLM、bank、geometry、shared update subspace、
  residual escape、额外shared trainable adapter和未merge source LoRA。
- 四卡就是4个native DDP ranks；不做gradient accumulation模拟8卡。
- 只有一个真实`action_query_batch_size_per_rank`，不要另设不同的
  `functional_policy_microbatch_size`。`frame_microbatch_size`只切同一视频的
  帧，不改变optimizer batch。
- RL训练与selection只用LIBERO official BDDL/random reset；fixed
  `.pruned_init`只作训练隔离的fresh evaluation。

## 5. 必须实现的唯一Action-Forecast Writer

完整细节以handoff第3节为准，不能自行简化成旧Action-Memory。端到端路径是：

```text
frames [T,3,H,W] + true frame indices [T] + full task tokens [L]
  -> frozen SigLIP/projector full image tokens [T,N_img,2048]
  -> content-only virtual state tokens [T,28,2048]
  -> full PaliGemma contextual prefix + VL Meta-LoRA
  -> Action Expert + Action Meta-LoRA + full 10-step flow
  -> final normalized action plans [T,50,7]
  -> same-absolute-time Plan/Revision tokens [U,2,256]
  -> variable-length Temporal Transformer [2U,256]
  -> 320 one-way LoRA queries
  -> exact sealed rank-16 public LoRA state
```

关键实现：

- 固定物理/控制时间stride采帧，视频`T`保持变长；只在batch内padding并携带mask
  和真实frame index。首轮真实比较stride 5与10。
- 每帧完整image tokens与完整task language通过PaliGemma；语言必须使用经过
  backbone上下文化的hidden state，不能只取embedding table。
- State token decoder：image width 2048先无bias投到128；28个routing-only
  slots、`Z_0=0`，经2个content-only self/cross-attention+expansion-4 FFN
  blocks、4 heads直接生成28个2048-d virtual tokens。routing只进入Q/K，
  V/residual/output只读取视频content。
- 28个virtual tokens加一个真实whitespace位置，使state区域长度29，与真实
  train state文本mean 28.8898/median 29对齐；不要恢复scalar、Fourier、
  离散tokenize或把state放进action suffix。
- PaliGemma 18层q/k/v/o用identity-init VL Meta-LoRA rank4；Action Expert
  18层q/k/v/o用identity-init Action Meta-LoRA rank8。A Kaiming、B zero。
- 每帧运行真实π0.5 10-step flow、horizon50；同一个video condition内所有帧
  共享同一可恢复Gaussian `[50,32]`起始noise。PaliGemma prefix每帧只算一次，
  KV在10次flow中复用。
- 只保留最终`[T,50,7]`计划，不保留`10×18` hidden states，也不把imagined
  state另喂temporal encoder。
- pinned LeRobot `sample_actions`带`@torch.no_grad()`，不能直接用于训练。
  在仓库内写可微wrapper，严格复用真实`embed_suffix`、mask/position、KV格式
  和10次`denoise_step` Euler更新；不得修改site-packages或写近似flow。
- 绝对时间`u=t_i+k`。`Plan_u`取时刻u之前最新帧对u的receding-horizon决定；
  `Revision_u`聚合同一u的连续forecast revisions。不要只比较一对相邻chunk，
  也不要平均掉同一未来时刻的多次预测。
- Plan MLP是`8->256->256`；revision event MLP是`24->256->256`，随后一个
  routing-only revision query以单层单向cross-attention+FFN聚合数量可变
  events。directed content独立RMSNorm；count和delta-norm统计只产生范围
  `[0.75,1.25]`的乘法gate，不得additive进入Revision。无event使用learned
  no-revision token。
- Temporal Transformer接受变长`[U,2,256]`，width256、8 heads、2 blocks，
  使用真实absolute-time RoPE、padding mask和token-type embedding。
- 320 routing identities：288个`18 layers×16 rank slots` expert queries、16个
  `action_in_proj` queries、16个`action_out_proj` queries。两层decoder均为
  `Z_0=0`的content-only self-attention、单向读取procedural memory的
  cross-attention、expansion-4 FFN；factor heads只能读取`Z`。
- 8类factor heads严格生成真实identity template中的38 targets/76 tensors；
  heads全部bias-free且final projection weight为zero，fresh public LoRA必须
  functionally identity。
  tensor name/shape从真实`LoraTensorSpec`读取，不手写猜测。

训练参数预算约等于rank-128 Source-SFT的`10,297,344`：

```text
content-only 28-slot state decoder   1,053,440
VL Meta-LoRA rank4                     921,600
Action Meta-LoRA rank8               1,253,376
Plan-relative Belief encoder         1,007,040
zero-preserving Temporal encoder     1,640,192
2-block content-only query decoder   2,191,104
Factor heads                         2,181,120
total                               10,247,872
```

从真实model/config打印逐模块参数量；可微调hidden widths以接近10.297M，但不能
改变上述信息流。public rank-16 LoRA仍为1,287,168 scalars。

## 6. 代码替换与效率优先

先用`rg`建立imports/callers/schema/checkpoint map，然后原位替换：

- `scripts/train_as_writer.py`保留为唯一AS入口；
- `scripts/evaluate_pi05.py`保留为唯一π0.5 rollout入口；
- `src/ember/writer/model.py`继续拥有LoRA specs/decoder；
- 退役`src/ember/writer/action_memory.py`，由单一
  `action_forecast.py`或同等职责owner替换；
- `temporal.py`原位成为Plan/Revision variable-time owner；
- 同步替换`as_contract.py`、checkpoint schema、training/inference/evaluator
  调用点和必要测试；
- 用唯一`configs/pi05_as_writer_action_forecast_v3.json`替换旧active
  action-memory配置；不保留v4/new/experimental平行runner。

优先尽快得到最短可运行垂直切片。一旦shape、梯度、identity/freeze和一个
exact-resume smoke通过，就立即做真实GPU profile/训练；不要先花数小时写大批
脚手架、广泛测试或cache。

校验一律以效率为先：

- 只做会直接防止无效科学结果、OOM、信息墙泄漏、错误冻结、错误LoRA schema、
  不可恢复checkpoint或安全越界的最小检查；
- 不做重复的流程门槛、全仓仪式性测试、无关lint、反复hash或为“更严谨”而增加
  的测试矩阵；
- 真实训练/rollout是研究结论authority，smoke只证机械合同；
- meaningful里程碑自动保存必要config/hash/rows/resume evidence，不要让整理
  工作阻塞下一段GPU任务。

速度与显存：

- BF16、fused SDPA/FlashAttention、静态Meta-LoRA、prefix KV复用、
  `output_hidden_states=False`；
- frozen SigLIP/projector可`no_grad`或缓存；从virtual state起的PaliGemma、
  十次Action Expert flow必须保留到Writer的梯度；
- forecast使用activation checkpoint/rematerialization，只保留最终plans和
  Plan/Revision最小张量；
- 若Writer graph与functional policy graph共驻留OOM，使用严格同sample/RNG/loss
  的two-pass replay/VJP，不改变optimizer step定义；
- evaluation先批量生成/cache固定panel的public LoRA并复用现有
  `per_sample_lora_batched_replan`，绝不退回逐rollout materialize+sequential。

## 7. 四卡profile与正式AS训练

只能使用物理GPU`0,1,2,3`。这不是终止他人进程的授权；每次launch前实时检查
owner、显存、利用率、温度、进程、driver/CUDA、`/data/ymdai`占用和`/data`
容量。不得kill/reset别人任务。个人硬上限500GB，大cache或新run先估峰值。

训练profile已经完成并封存：

1. `frame_stride=5`固定，`frame_microbatch_size=32`、每rank action-query
   batch=`20`。frame-microbatch40更慢；48在首步前达到`81,153/81,920 MiB`
   且无法稳定前进，已拒绝；
2. 选中配置12-step profile稳态中位约`6.49s/step`、全局约
   `12.32 queries/s`，并覆盖rank0采到72帧的长视频；
3. 最终无`tau` raw-RMS实现又完成fresh step1和step1→2 exact-resume：
   resumed step约`6.92s`、全局`11.56 queries/s`，峰值allocated/reserved
   为`77,090,931,200/83,730,890,752` bytes，source policy 0 trainable；
4. 不再运行stride10、frame/action batch profile或未充分训练的specificity；
   直接正式fresh 0→600；
5. 评测从旧稳定点4 replicas/GPU、8 envs/replica附近实测，并测adapter
   预生成/cache；旧6 replicas在旧Writer编码阶段OOM，新路径只有实测通过才用；
6. Long tasks按实际可用GPU数先做cost-balanced shards覆盖每个device，之后所有
   workers从统一dynamic queue接普通tasks并work-steal；不固定切八份；
7. 唯一训练选择已写入canonical config；不得重新引入临时profile开关。

正式AS：

- 4个同角色DDP ranks；GPU0无额外CUDA model/server/controller。
- 从fresh identity一次连续运行step0→600；每75 steps保存完整checkpoint，
  但训练期间不主动停下做validation或specificity。
- step600保存完成后再统一选择多个checkpoint做paired
  `8 tasks×50 fixed states` correct-video closed-loop validation；step600的
  shuffled/reversed特异性先做低成本内部数值诊断。
- val functional loss只能微微参考，不能决定best。最终看完整rollout、
  per-task counts、paired flips和独立复测。
- 内部诊断逐层比较forecast residual、Revision、Temporal memory、query
  content和effective LoRA；只有最终输出差异明确且跨多个tasks/videos稳定，
  才运行昂贵的shuffled/reversed paired validation arms。通过后再依据绝对
  性能和后续大步长训练选择observed-best，并给最终best补齐
  correct/cross-suite-wrong/shuffled/reversed。
  paired task/init/policy/video/noise seeds保持一致，shuffle/reverse使用同一帧集合。

Writer停止标准必须严格执行：

- 必须先找到validation observed-best；
- best之后必须出现幅度非常明显的下降：明显超过400-rollout正常复测波动，
  aggregate上清晰可见，由多个tasks共同贡献，而不是单task崩掉；
- 该明显下降必须在预先封存的独立evaluation seed/panel或重复测量中仍成立；
- 多个后续checkpoint只是略低、paired统计勉强可区分、loss平台、train平台、
  success持平或一个坏点，全部不算饱和；
- 没看到上述明显下降，就按600-step或证据支持的更大step跨度继续，不设总
  wall-clock上限，也不恢复约30分钟segment节奏。

性能门槛：AS不能明显落后四卡SFT`108/400`；超过旧八卡SFT`122/400`是stretch。
还必须做到correct优于cross-suite wrong，且对shuffled/reversed的变化显著优于
旧Action-Memory近似bag-of-states的表现。若不过关，先区分实现故障与科学负
结果，做最小、证据驱动修正后继续；不要为正结果改变split、数据墙或任务。

## 8. AS通过后才做RL-Writer

不要提前推进RL。AS同时通过绝对性能、correct/wrong和帧顺序特异性后：

1. 从新架构规定identity initialization新建独立RL-Writer run；不加载完整AS best。
2. 做短、task-balanced AS cold start，同时持续official random-reset reward
   screen；直到24个development-train tasks每个至少一次真实success。
3. 保存每task first-success step、teacher action query数量和wall；全task覆盖后
   永久关闭action入口，转pure official env reward。
4. 高效profile RL estimator、每rank配置、persistent env pool和rollout拓扑；
   自主选择真实reward signal最好的合规实现，不为普通参数等待owner。
5. 保存optimizer、scheduler、worker/env/policy RNG、seed schedule、interaction
   cursor、reward rows、task coverage、runtime和exact-resume。
6. train曲线必须由多个tasks支撑并训练到平台，但train平台不是停止点。继续对
   合适checkpoint做完整validation，找到observed-best。
7. RL停止标准与AS完全相同：best后必须出现非常明显、远超rollout噪声、由多个
   tasks贡献、独立panel/复测仍成立的validation下降；多个略低点绝对不够。
8. 只在selected RL best做correct-video与一次cross-suite wrong-video。

若AS经过多轮最小合规修正仍无法同时通过，保存完整证据并停下汇报，不伪造RL。
若RL实现或科学信号经多轮合规修正仍无法推进，也保存failure packet后停下汇报。

本focused Goal完成后不要自动继续final-32、test task-local RL、joint oracle或
ViVLA；先向owner汇报。

## 9. 自主推进与交付

除新增权限、必须干扰他人进程、预计突破500GB、删除/覆盖不可恢复数据、实质
改变科学问题或同一具体阻塞经多轮修复仍无法推进外，不要停下来逐项询问。
训练/评测一旦可启动就先启动；运行期间只推进不会修改其import/config/output
contract、也不写同一目录的工作。

每个meaningful里程碑更新`task_plan.md`、`findings.md`、`progress.md`。只提交
task-scoped代码/文档，不提交dataset、weights、checkpoint、cache、凭据或私有
大文件。完成必要的最小验证后commit并push main。最终汇报必须包含：

- 新架构逐模块真实参数量；
- 训练/评测最优配置、显存、吞吐和wall；
- AS/RL全部候选的逐task validation曲线及明显峰后下降证据；
- observed-best、correct/wrong/shuffled/reversed结果；
- cold-start action消耗、24-task first-success coverage、纯RL interactions；
- commands、commit、configs、raw rows、hashes和exact-resume状态；
- 任何未通过项的工程/科学归因与failure packet。

## 10. 2026-07-25最新暂停点

Belief-v3 formal 0→600和全部低成本内部顺序特异性检查已经完成。Revision与
Temporal的time-centered动态分量对reversed/shuffled有明显差异，但当前单路
Temporal memory被task-global/time-constant成分主导，最终effective LoRA差异
只有`0.000297/0.000169`，内部gate失败。仅增加normalization无效；将真实
Temporal memory做masked time-centering后，同一query/factor heads的effective
差异可恢复到`0.0543/0.0401`，故根因已定位为global constant遮蔽temporal
innovation。

owner要求在结论形成后暂停并汇报。当前不得自动运行shuffled/reversed环境
validation、correct-video性能曲线、继续AS、实现下一版或启动RL。下一次继续前
先与owner对齐global/innovation双路、各自归一化且最终仍固定256维的合成接口。
