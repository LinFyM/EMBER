# EMBER

EMBER研究能否把目标机器人的无action教学视频一次性编译为完整task-specific LoRA，使同一个frozen
VLA在该任务不同初始化上闭环工作：

```text
exact task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> one complete rank-16 task LoRA
                    -> frozen π0.5-LIBERO source policy
                    -> closed-loop execution
```

Writer只在rollout前运行一次。它不读取teacher action、proprio、reward、terminal、task ID、filename、
object pose或hidden normalization；video是唯一dynamic value，不能存在language-only LoRA bypass。

## Current status

- canonical workspace：`/data1/user/ymdai/projects/EMBER`；主写分支：`codex/bci-continuation`。
- 长期Goal未完成：同一shared method/single checkpoint strict paired correct必须严格`>150/400`并继续
  提高，同时保持视频时序因果、same-task鲁棒、breadth和低checkpoint漂移。
- 历史最好single checkpoint是v6-fast macro400：五臂
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。
- v6-Prior whole-LoRA objective已完成formal 0→50和同一schedule四点strict correct400：
  macro0/10/25/50=`134/127/105/123`。macro0仍最佳，四点逐task envelope=`147`；该objective已停止，
  不续训、不扫权重、不为loser补五臂。
- 第34节ECP已完成formal0→10→25并退役：strict correct=`133/120`，macro25相对同schedule
  macro0=`134`的paired gained/lost=`13/27`、`p=.038477`。它成功把`a_correct`和expert component推高，
  却伴随更大的expert-orthogonal drift和显著closed-loop退化，因此不续50/100、不扫权重、不补六臂。
- 当前唯一活动候选是第35节v6 Condition-Local Dynamic Expert Tangent Tube Writer：保持historical v6
  初始化、冻结上游、one-shot输入、B20 functional和negative schedule；对correct及当前negative分别用
  同一language/video/order的frozen-v6输出作dynamic baseline，只限制student增量的expert-orthogonal
  分量。anchor仅训练期存在，部署仍只生成一套LoRA且不读取expert bank或feature cache。
- 当前没有运行中的EMBER GPU进程。新canonical v3 config、双臂low-rank objective、training-only anchor、
  trainable-only resume/deployment load、新评测family和CPU oracle已在同一vertical path通过全仓
  `276 passed`、compileall和diff-check。clean frozen`2616773`的六卡gradient/whole-macro profile也已
  exit0：wall=`21.531s`、peak allocated/reserved=`43.354/47.113GB`、0 OOM/nonfinite；macro0两臂
  tube exact zero，唯一projection/ranking weights=`.00686480847114155/.010514453175708578`已写回config。
  下一步是严格后继的fresh0→1/exact-resume1→3/contiguous0→3，再按macro10/25/50门及时跑strict400。
- 首次A40 batch8 smoke只发现普通BF16 batch-shape roundoff（max`.001953125`、mean约`4.70e-5`，direct
  repeat为零）。此前固定batch1和重复direct forward的决定已经撤回；当前吞吐优先，从稳定且有显存
  余量的候选中选择实测LoRAs/s最高的batch，并使用原生BF16/F32 LoRA cache、action prefetch和更少
  host sync。
- 真实fixed-panel profile在同一32 requests/1093 sampled frames上得到batch8/16/32吞吐
  `.911427/.905107/.906432 LoRA/s`，选择实测最快batch8；8-row vertical smoke完整闭环且0异常。
- logical B20保持不变；physical B20和B16已由A40容量实证排除，balanced B10+10以FP32 leaf-gradient
  加权累积完成train24×20=`480/480` queries。旧whole-LoRA gradient seal的expert/ranking weights为
  `.008355172068998324/.28570466890490887`；ECP重新实测的projection/ranking weights为
  `.006883349605446485/.010514451404229894`。当前Tangent Tube从自己的live gradient seal得到
  `.00686480847114155/.010514453175708578`，没有直接继承任一旧seal。
- formal训练root为
  `runs/outputs/pi05_v6_prior_formal_r6_lb20_mb10_eff15db_20260809`；四点paired分析保存在
  `runs/outputs/pi05_v6_prior_checkpoint_curve_strict_paired_eff15db_20260809/analysis.json`。

当前科研结论、完整历史实验谱系和关键不确定性见
[`docs/active_session_handoff.md`](docs/active_session_handoff.md)；精确执行协议见
[`docs/execution_brief.md`](docs/execution_brief.md)；当前计划见[`task_plan.md`](task_plan.md)。

## Cumulative evidence in one view

- frozen source base为`48/400`；privileged mixed-task Source-SFT best为`109/400`。
- v5.2 old的`132/138/74/82/83`仍是最强视频特异性形态；v6-fast task-complete的
  `143/135/125/128/129`是最高absolute。两者recipe交叉结果证明架构与训练方式耦合，不能整体判死某一
  architecture，也不能简单退回old recipe。
- CV-ADR、Target-Bound、Semantic Factor-Basis、variance reduction、Direction Store、Target-Owned、
  Policy-Lane/Atom、Condition-Kernel、K4/few-shot、trace/expert routing和多条reward路线逐步证明：视频
  sensitivity、LoRA健康几何、较低functional loss、独立parameter ownership或few-shot任一项都不是
  closed-loop成功的充分条件。
- 24个task experts统一step2000的development-train direct-expert成绩为`658/1200`、23/24 tasks非零，
  证明它们是有用但不完美的privileged train targets；soft/hard bank在held panel只有`15/80`和`3/80`，明确否定把train experts
  直接当deployment字典。
- 当前最小假设不是重做全部架构，也不是继续追LoRA健康度。whole-LoRA监督主要造成径向收缩；ECP虽
  补足expert分量，却通过共享compiler/heads连带改写大量其他effective方向。新目标以每个实际condition
  自己的historical-v6输出为局部原点，保留expert方向上的必要修正，只抑制增量的正交漂移，同时维持
  原positive functional和bounded时序/wrong ranking。

## Data and evaluation

- 起点是generic`lerobot/pi05_base`，不是读过目标LIBERO-40 actions的`pi05_libero`。
- LIBERO-90 specification-only audit排除19个与目标40 exact semantic/composition重合的source tasks；
  71 tasks×50成功episodes训练frozen source base。
- 目标40固定development split为24 train / 8 validation / 8 test，不按outcome改task IDs。
- Writer训练只读train24 actions；validation/test actions不产生梯度。
- official evaluation严格配对correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video的state、
  env/policy RNG、video ordinal和输入处理；shuffled/reversed真实重排frames后完整forward。
- checkpoint只由真实closed-loop选择。loss、smoke、LoRA norm/rank/cosine和内部路径只能作机制证据。

## Runtime and paths

BCI项目资产按canonical roots归并：

```text
EMBER/
├── data/       # datasets and LIBERO assets
├── models/     # tokenizer/model assets
├── runs/       # training/evaluation/checkpoints/logs
├── evidence/   # migration and retained evidence
├── .venv/
└── .cache/
```

进入仓库后使用项目`.venv`；`.env.local`提供BCI本地默认路径，训练与评测仍通过CLI显式传入关键资产。
主要入口：

```text
scripts/train_task_experts.py
scripts/train_v6_prior_writer.py
scripts/evaluate_pi05.py
```

GPU工作必须实时检查`gpu01/gpu02`，只用空闲A40、合计最多6张，不干扰他人；多卡显式
`NCCL_P2P_DISABLE=1`并遵守NUMA physical/local rank和deferred-NCCL合同。不得为验证身份生成或比较
SHA-256/MD5；吞吐、有效显存利用和尽快获得真实严格评测优先。

## Required reading

修改或运行项目前必须完整遵守[`AGENTS.md`](AGENTS.md)中的阅读清单。最小当前入口是：

1. `AGENTS.md`
2. `README.md`
3. `docs/active_session_handoff.md`
4. `docs/execution_brief.md`
5. `docs/action_forecast_writer_video_expert_manifold_design.md`
6. `task_plan.md`
7. `findings.md`
8. `progress.md`

历史设计保留为证据而非活动实现；改变其拥有的接口前，按handoff实验谱系读对应design到EOF。
