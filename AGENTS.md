# EMBER Repository Instructions

## Authority

本文件和 `docs/execution_brief.md` 是当前活动 authority。2026-07-21 generic π0.5 feasibility 已结束；其后 owner 明确批准继续完整 EMBER 主线，并以本文件记录的共享 π0.5-LIBERO source base、one-video Writer 和 test-task training 口径替换此前“结果后停止”的临时边界。

修改代码、数据、split、模型或实验状态前，完整阅读：

1. `README.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_expert_consultation.md`
5. `docs/action_forecast_writer_design.md`
6. `docs/action_forecast_writer_v4_root_cause.md`
7. `docs/action_forecast_writer_v5_design.md`
8. `docs/action_forecast_writer_v5_1_proposal.md`
9. `docs/action_forecast_writer_v5_2_design.md`
10. `docs/action_forecast_writer_v5_3_design.md`
11. `docs/action_forecast_writer_v6_design.md`
12. `docs/action_forecast_writer_v7_design.md`
13. `docs/action_forecast_writer_v8_design.md`
14. `docs/action_forecast_writer_v10_design.md`
15. `docs/action_forecast_writer_loom_derivation.md`
16. `docs/action_forecast_writer_loom_design.md`
17. `docs/action_forecast_writer_recenter_design.md`
18. `docs/action_forecast_writer_core_program_design.md`
19. `docs/action_forecast_writer_prior_innovation_design.md`
20. `docs/action_forecast_writer_target_spectral_design.md`
21. historical `docs/action_forecast_writer_coherent_procedure_design.md` from
    commit `35fb28f`（current tree中已由后续authority取代；用
    `git show 35fb28f:docs/action_forecast_writer_coherent_procedure_design.md`读取）
22. `docs/action_forecast_writer_semantic_program_grid_design.md`
23. `docs/action_forecast_writer_unified_causal_program_design.md`
24. `docs/action_forecast_writer_amplitude_preserving_dual_read_design.md`
25. `docs/action_forecast_writer_contextual_value_dual_read_design.md`
26. `task_plan.md`
27. `findings.md`
28. `progress.md`
29. `docs/concept.md`
30. `docs/decisions_and_open_questions.md`
31. `docs/novelty_and_landscape.md`

`docs/active_session_handoff.md`是当前跨session恢复入口，集中摘要研究证据链、
v5失败证据、v5.1设计理由、运行状态和下一动作，但不覆盖架构或长期科学
authority；focused AS/RL完成或其不再承担跨session恢复作用时应更新或删除。

旧 `docs/expert_plan.md`、SmolVLA/70-10-10 runner/config/checkpoint 和
Phase A–F 可执行路径已从工作树退役，只由 Git 历史保存 provenance；不得恢复为
活动路径，不得依赖或混入 MemLLM。

## Active objective

以 generic `lerobot/pi05_base` 为起点，先在与目标 LIBERO-40 specification 无 exact semantic/composition 重合的 LIBERO-90 source tasks 上做联合 action-SFT，得到并冻结一个共享、多任务、语言条件的 π0.5-LIBERO source base；随后在固定 24 train / 8 validation / 8 test 目标 split 上完成 AS-Writer、RL-Writer、Source-SFT、seen/wrong-video 机制对照、合并 32 source 后的单 seed 重训和 zero-interaction test；再直接在 8 个 test tasks 上把三种 task-local LoRA RL initialization 训练到各自最佳，最后用 8 个 test tasks × 50 action episodes 联合训练一个 privileged shared-LoRA oracle。ViVLA-style matched baseline 和 source-only outer learning 只在核心闭环之后有时间再做。

任何单一 source base、训练 loss、smoke、局部 seen 结果或一个 Writer 阶段都不能单独触发长期 Goal complete。

## Current focused execution task

2026-08-01 current override：当前可执行canonical Writer仍为
[`docs/action_forecast_writer_amplitude_preserving_dual_read_design.md`](docs/action_forecast_writer_amplitude_preserving_dual_read_design.md)。
以下直到`## Data and split`的UCP/SPG叙述只作紧邻历史证据，不得覆盖本段或恢复
退役可执行路径。

AP-ADR精确参数`10,241,024`，保留mean-backed permutation-invariant Semantic
Core、outgoing raw `[A_f,G_(f+1),G_(f+1)-G_f]` Program、独立target-only Core
reads、38×16 target/rank Program reads和conventional coherent factor heads；
删除terminal normalization/AdaLN/gate、global mixer、谱约束和第二套LoRA。
最长105-frame B20三macro与formal-seed fresh0→1→exact-resume1→3已通过，step1
全部payload逐项未改写；live seal commit为`7dffb6f`。

clean detached `7dffb6f`的fresh首小时已经自然完成macro0→200：

```text
tmux   已自然退出，不得重复启动
frozen /data/ymdai/.codex/worktrees/EMBER-ap-adr-formal-7dffb6f-20260801
root   /data/ymdai/outputs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801
log    /data/ymdai/logs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801.log
```

run summary确认200 optimizer steps/200 cycles、96,000 queries、4,800 one-video
conditions、每task 4,000 queries/200 visits，wall `3898.217s`；validation/test
action读取和test video读取均为0。macro50/100/150/200 paired correct400已经完成，
为`91/81/94/91`，breadth为`6/6/5/7`；single winner macro150只有94，且相邻点
gained/lost为`33/43`、`36/23`、`25/28`，故一小时门失败，不resume到400、不做
五臂。低aggregate不能整体否定AP复用的Core或dual-read思想，但已经直接否定
当前`contextual Program只作K、raw A/E/D直接作V`这一中央职责。

修复分析器中PI05 sampler把attention backend从SDPA永久改为eager的生命周期污染后，
macro150 refs1在8/8 tasks上实现逐层、effective BA和fixed-action严格零误差重放；
修复commit为`5d93af3`。有效内部结果的analysis/summary SHA为
`d42fc4eb...bc2b`/`f2c572c5...e682`。same-task raw Program与第二层contextual
Program relative L2为`.919/1.105`，到Program read/BA/action却只剩
`.0321/.0301/.0167`；shuffled/reversed到BA仅`.00269/.00390`。反转valid
contextual temporal keys时BA/action只变`.000521/.00194`。只保留Effect列即可
在8/8 tasks重建full BA（平均差`.00821`），只保留Action或Change则约差
`.276/.283`；固定full key后结论不变。最早失效接口因此是
`causal contextual Program -> high-entropy key-only routing -> raw Effect-DC value`，
不是whole Program无梯度，也不是视频前端无信号。

下一步先完成预注册endpoint10的18-checkpoint no-gradient关联审计，同时把上述
架构根因与UCP raw/SERIAL训练交互共同用于下一整体设计。不得把endpoint结果用于
选checkpoint或改loss，除非它原封不动通过预注册的全局、family、recipe-direction
和逐task四重门；不得因AP失败而跳过cycle-normalized randomized-group4的受控
训练因果格。下一结构authority已经封存在
[`docs/action_forecast_writer_contextual_value_dual_read_design.md`](docs/action_forecast_writer_contextual_value_dual_read_design.md)：
同一causal contextual Program直接承担K/V，保留mean-backed Core和独立dual reads；
当前main在实现前仍只有AP一条可执行路径。

同曝光UCP raw macro150与SERIAL step900内部对照确认：SERIAL把删除A/D后的
BA/action变化从`.0653/.01269`提高到`.4184/.12999`，所以update granularity
真实控制视频动态写出；但四点correct差值`+7/-17/+21/-3`且漂移未解，SERIAL不
是默认recipe。v7/v8/v10/Loom等仍只否定已有内部反事实定位的局部接口，所有与
full24/B20/fast400混杂的可复用思想继续保留为条件反事实。

exact v5.2 topology在mature task-complete recipe上的缺失因果格已经完整封存。
macro150/200/350/400 paired correct400为`51/91/106/120`；single-checkpoint
winner macro400的correct/same/wrong/shuffled/reversed为
`120/109/107/111/124`。它只覆盖4/8 tasks，reversed高于correct，因此没有正式
行为视频特异性。内部分析确认顺序和视频变化确实从Procedure传到effective
LoRA及fixed-query action，但这种数值差异没有与闭环收益对齐；task-complete
recipe还把same-task视频中心化方差从旧v5.2的`1.6655%`压到`0.6844%`。当前
病灶优先指向functional surrogate与source-policy closed-loop有效流形错位、
task能力轮换和条件创新缩弱，而不是上游完全没读视频。

SPG canonical实现精确参数为`10,633,216`，最长真实105-frame B20 profile、
fresh0→1→exact-resume1→3和formal macro0→200均已完成。macro50/100/150/200
paired correct400为`97/115/77/100`，一小时门失败，不续到400，也不做昂贵五臂。
checkpoint envelope union为162但single point best只有115；macro100→150
lost51/gained13，随后又反向轮换。train functional loss在24 tasks上都改善，
held functional loss和closed-loop不跟随。

SPG不是上游Program失败。macro100 refs2反事实中same/wrong/shuffled/reversed的
Program relative L2为`.967/1.186/1.193/1.202`，到Program coordinates为
`.355/.715/.627/.658`，再到effective BA只剩`.066/.221/.116/.116`。固定Core、
只改变Program仍保留order差异。最早病灶是compiler路由同质化：CoreReader entropy
`.999992`、target-centered attention energy `3.9e-5`，ProgramReader target/rank
centered routing约`4–5e-5`；exact50 LoRA几乎严格rank1且B columns相同。独立Core
加法旁路和跨target/rank mixer共同把强视频Program写成共享方向。CP-24 projected/
raw cosine约`.983`、norm约`1.25×`；raw full24 mean末段只保留平均单task gradient
energy的`4.79%`，说明投影能消负pair但不能恢复近正交task innovations。

下一canonical模型与训练设计统一由
[`docs/action_forecast_writer_unified_causal_program_design.md`](docs/action_forecast_writer_unified_causal_program_design.md)
负责。它把未中心化absolute `X_f=M_f+G_f`、native Action `A_f`和正确outgoing
change `G_(f+1)-G_f`放入同一个causal axial Program；删除独立Core value旁路、
target-Core first hop和global coordinate mixer；归一化target/rank identities后
由38×16 coordinates单级读取raw Program，再使用conventional coherent heads。
训练恢复raw full24 mean，保留Gram诊断但不投影；B20改为边缘仍uniform-row的
20-strata随机jitter，以降低单task update的过程覆盖方差；首版保持fast-decay400，
不把slow2000与拓扑同时混入。

UCP canonical实现已在独立write worktree完成：唯一Program路径为
`[X_f, A_f, G_(f+1)-G_f]`，唯一compiler为单级38×16 raw-value reader；旧SPG
Core add/global mixer和CP投影均已从active executable path删除。真实module
enumeration为`7,683,328`参数；step0 identity、causal prefix、outgoing alignment、
target/rank routing、零内容不造值、raw full24 mean和20-strata exact-resume等CPU
合同已通过，全仓`203 passed`。fresh config为
`configs/pi05_as_writer_unified_causal_program_full24_decay400_v1.json`，现已seal为
B20 formal authority。独立`0d4c271` frozen worktree上的真实最长105-frame三
macro profile已通过：step wall `20.394/18.494/18.504s`，峰值allocated/reserved
`77,127,082,496/83,345,014,784` bytes，72套video conditions和1,440 queries
全finite，step2起四个主块梯度可达。canonical formal seed的
fresh0→1→exact-resume1→3也已通过，step1全部checkpoint payload逐字节不变。

UCP raw-full24 formal macro50/100/150/200 paired correct400已经封存为
`82/117/100/110`；union169、single best117且能力持续轮换，故不resume到400、
不做五臂。macro100 refs1证明reader路由和order/wrong差异可达BA/action，但固定X
只换A/D时BA与action变化仅约`2–5%/.5–.9%`，dynamic教学弱。exact50 analyzer的
rank-gauge异常已经定位：raw A/B置换变化`.74184/.13602`，effective BA误差仅
`1.299e-9`；bf16 fixed action的`.002047`差异来自rank求和次序，而非BA错误。
sanity继续对finite和BA `2e-5` fail-close，action execution drift只作记录。

下一训练反事实已经在main实现：冻结UCP拓扑和完整task/video/query exposure，
改为每update全局4 tasks、六phase覆盖24 tasks的serial-4；LR按full24 cycle重复六次。它同时改变
AdamW/moment/weight-decay时钟，并把long-first变成真实optimizer curriculum，结果
不能单因归为“消除梯度抵消”。clean `c4b85e8` refs2与exact50均已通过；exact50
400 rows确认pooled same-video BA/action variance仅`.09008%/.01656%`。serial
long-first重放的phase/cost Pearson为`-.8331`，task38始终phase0，必须审计真实
optimizer curriculum。serial CPU合同为`233 passed`；clean detached `10a71a1`
最长105-frame、B20、18 updates/3 cycles已通过，formal seed fresh0→1→resume1→3
→跨cycle boundary到7也通过，step1/3文件不变。canonical config已seal；下一动作
已执行：clean frozen `3db82df`从fresh identity启动1,200 updates，tmux
`ember-ucp-serial4-3db82df`；首个六phase cycle合同健康。不得重复启动或从smoke
续接；等待时准备300/600/900/1200 paired correct400与内部联合根因分析。

v7/v8/v10/Loom及后续历史不得整体判死。只能删除由内部反事实独立否定的局部
机制；Action anchors、causal Procedure、双流、Core语义、target-first/rank-last
等与fast task-complete recipe混杂的组件仍可在职责完整的新结构中复用。
匹配150次video exposure时v5.2 old/new=`132/51`而v6 old/new=`95/111`，描述性
DiD=`97`；这进一步证明强architecture×training-bundle交互，但不识别单一recipe
成分。

关键历史基线仍为：v5.2五臂`132/138/74/82/83`；v6 task-complete
single-checkpoint best及五臂`143/135/125/128/129`；v6 old recipe
`121/122/111/84/47`；corrected mixed-task rank-128 Source-SFT `109/400`。
v7、v8、v10、Loom和Recenter各自在当时训练bundle下已经形成正式负结果，旧版
可执行路径不得整套恢复，也不得在原失败checkpoint上继续堆局部scale/gate补丁；
但这不构成对整版思想的独立否定。只有被内部反事实直接定位的接口可以删除，
其余组件必须继续按架构×recipe混杂处理，并可在职责完整、受控的整体设计中复核。

`150`继续作为重要里程碑与强baseline参考，但不再是focused Goal的终点。
即使超过150，只要内部仍有明确漏洞、task漂移、视频学习不足或可信改进方向，
就继续按照“整体设计→一小时训练→评测/内部分析→续训或根因重构”循环推进。
只有agent在其能力范围内已经找不到可信提升空间，才允许focused阶段停止。

当前Writer通过后才做严格配对one-shot baseline与独立
short-AS-cold-start→pure-reward RL-Writer；不得把完整AS best冒充RL cold
start。focused闭环不自动继续final-32、test task-local RL、joint oracle或
ViVLA。

GPU工作固定frame stride=5，只使用物理GPU4–7；0–3不进入visible set。
4–7即使已有他人进程也可按owner授权共卡，但不得杀、暂停、重置或干扰。

## Data and split

- 目标 benchmark 为 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，共 40 tasks。
- 活动 development split 已封存在 `configs/libero_24_8_8_v1/`：每 suite 6 train / 2 validation / 2 test，总计 24/8/8；不得按 outcome 改 task IDs。
- validation 完成方法选择后，将 8 validation tasks 合入 source，形成最终 32 source / 8 test，并从规定初态重训已选方法。
- shared source-base corpus 来自 LIBERO-90。完整3600-pair specification-only
  audit已在看新policy outcome前封存：排除19个与目标40 exact
  semantic/composition重合的source tasks，保留71个active tasks。task44
  （`turn on the stove`）和task77
  （`pick up the book and place it in the back compartment of the caddy`）只是
  其中两项；不得把audit误写成尚待完成，也不得按outcome重开source IDs。
- source base 使用过滤后每个 active LIBERO-90 task 的全部 50 条成功 teacher episodes。不得使用 `pi05_libero`，因为它已读过目标 40 tasks actions。
- source-base action/state normalization 只从过滤后的 LIBERO-90 source actions/states 计算并冻结；所有下游方法共用，validation/test 不单独重算。

## Common frozen source base

活动文档中的 frozen π0.5-LIBERO source base 统一指：

```text
generic lerobot/pi05_base
→ 在过滤后的 LIBERO-90 source tasks × 每 task 50 条成功 episodes 上联合 action-SFT
→ 得到共享、多任务、语言条件的 π0.5-LIBERO policy
→ 若训练 recipe 使用 source LoRA，先 merge 成 base
→ 冻结，作为所有后续方法的共同起点
```

- 先调研官方/成熟 π0.5 fine-tuning 与 LoRA 实现，不自行猜 targets 或 runner 参数。
- source base 不追求高 ceiling；用全部目标 40 tasks 的小型快速 screen 确认它已开始在该 benchmark 上产生跨多个 task 的部分真实成功，不能只靠一个易 task 的 aggregate。这里不要求每个 task 已有高成功率。generic π0.5 的 `0/400` 只作原始校准，新 source base 必须另测。
- owner 于 2026-07-22 将 source-base 正式训练锁定为从 generic base fresh 运行 1,000 optimizer steps；不续接已停止且无 checkpoint 的旧 30k attempt。历史非focused阶段的约120分钟guardrail保留为其原实验合同；当前v5.1 AS/RL按上述focused authority分段探索，不受该旧guardrail限制。
- source base 冻结后，AS-Writer、RL-Writer、Source-SFT、三臂 task-local RL、联合 target-action oracle 和 ViVLA-style baseline（若做）均从它开始。
- 下游只保留一个活动 LoRA 空间；不得叠加未 merge 的 shared source adapter。

## Writer and source baselines

- 核心固定为 `task language + exactly one action-hidden teaching video -> shared Writer -> complete task-specific LoRA`。
- Writer 不得接收 action、proprio、reward、terminal、task ID、filename 或隐藏 normalization；source actions 只能进入 AS functional loss。
- `Action-Supervised Writer (AS-Writer)`：development在24 train tasks上做上述task-complete宏步；每个task只读1条teacher video并生成1套one-shot LoRA，`B_a`条独立同task action queries在该LoRA下各计算一次functional loss、先task内求均值，再让24 tasks等权。下一次macro访问该task时换一条video；video与action episode/chunk不要求同episode配对。frozen source base只通过functional LoRA forward参与，更新Writer。
- 历史task-local RL的总预算合同不影响当前focused v6 AS/RL，但每个新增训练段仍须通过当前证据门。
- `Reward-Trained Writer (RL-Writer)` 是独立路线：按当前 focused task 从新架构规定初态做短、task-balanced AS cold start，直到24个development-train tasks各在官方random-reset rollout中至少成功一次，再关闭action数据入口并跨source tasks做纯reward训练；它不从完整AS-Writer best继续，cold-start消耗必须完整报告。
- RL-Writer rollout 使用 LIBERO 官方随机 reset/BDDL 初态；不使用 `.pruned_init`。只用官方 env reward/success，不从 object pose 等内部状态手工构造 privileged shaping。
- `Source-SFT` 是在同一 frozen source base 上、跨 24 development train tasks fresh训练的一套 shared rank-128 LoRA，test 不看 held video/action。physical batch必须混合tasks，以`task→episode→chunk`分层均匀采样并做task-balanced loss，不得让rank固定为单一task。v6确认后默认重训并根据validation找最佳；它和AS-Writer不要求机械匹配optimizer steps或consumed examples，但必须报告训练数据、steps、GPU-hours、参数量和搜索上限。
- 所有方法共享同一frozen source base、normalization和policy接口，但不再机械要求相同trainable参数化或LoRA rank。Writer继续生成sealed rank-16 public task LoRA；capacity-matched Source-SFT可使用rank128，其10,297,344个trainable参数用于约束Writer本体参数预算。各方法的targets/rank/alpha/dropout与identity初始化都必须显式报告。

## Seen and video-causality evidence

- 必须增加 source/seen-task performance comparison；seen panel 在看 outcome 前按 specification 预声明并覆盖四 suites，不用它替代 validation/test。
- 必须做 wrong-video control：evaluation task、正确 language、init state、policy RNG 均不变，只把 Writer 输入换成另一 suite 的 teacher video。
- 对 AS-Writer 和可用的 RL-Writer均报告 source base、correct-video LoRA、cross-suite wrong-video LoRA；核心视频特异性量是 correct-video 与 wrong-video 的差异，而不是只看两者是否各自高于 base。
- zero-interaction held evaluation 每个 rollout 从正确 task 的 50 条 teacher videos 随机抽一条；不得挑最好视频。

## Final retraining and zero-interaction test

- development 只先跑一个 training seed。AS-Writer、RL-Writer（若成立）和 Source-SFT 在 24 train / 8 validation 上选定配置后，合并成 32 source tasks，从规定初态各自重训一次。
- 在打开最终 test 前先完成 final seen-task comparison。
- zero-interaction test 统一比较新的 frozen source base、Source-SFT、AS-Writer、RL-Writer（若成立）及 correct/wrong-video controls。旧 generic base `0/400` 不可冒充新 source base 结果。
- 旧 test 已做 generic/source-base feasibility audit，owner 明确不把这视为阻塞；不得再以“untouched test”异议停止推进。

## Test-only task-local RL

- task-local RL 不在 validation 上预训练、预冻结或选择算法；在最终 test 阶段打开后，直接把每个 test task 当作 adaptation training domain，在该 task 上调优并训练到 reward/性能曲线接近最佳。
- 三臂为：source base + functionally identity LoRA、AS-Writer LoRA、RL-Writer LoRA。RL-Writer路线失败时如实缺席，不伪造。
- 每个 `(task, adaptation seed)` 开始时随机选一条该 task teacher video；AS/RL Writer 两臂使用同一条并固定生成的初始化 LoRA，随后只原位更新该 LoRA。
- 三臂使用相同 task、env/policy seed schedules、官方随机 BDDL 初态序列、相同 RL 实现和可比的调优/资源上限；保存完整 optimizer、worker RNG、seed schedule、interaction cursor 与 exact-resume state。
- adaptation、调参和 checkpoint 选择可使用该 test task 的官方随机-reset reward rollouts；固定 50 `.pruned_init` states 只作训练分离的 fresh evaluation，仍执行 dummy settling、suite horizon 和成功即终止。

## Privileged direct-action oracle

- direct target-action baseline 不是 task-local per-task LoRA。
- 在三臂 RL 和无 action 方法结果封存后，从同一 frozen source base 出发，使用 8 个 test tasks、每 task 全部 50 条 action episodes，联合训练一套 shared multi-task LoRA；第一轮只做完整 50/task，不做 action-budget 曲线。
- 它是 privileged oracle/reference，不属于与 EMBER 同信息墙的主 baseline，也不得反向修改前面方法。

## Evaluation and efficiency

- official π0.5/LIBERO preprocessing 保持：render 256、model 224、两相机 180° rotate、state/action 7维、10 flow steps、执行前 5 actions后重规划、dummy settling 10、成功即终止、suite horizons 220/280/300/520。
- generic feasibility 已证明固定“一 task/一 GPU”会被两个 horizon-520 tasks拖尾；新 evaluator 必须先调研其他成熟项目，并按预计 `episodes × horizon` 做 cost-balanced state shards、动态任务队列和持久 model/env，而不是静态 task/GPU。
- Writer每 rollout LoRA 不同时，真实 profile batched functional LoRA 与每卡统一 1/2/3 个 policy replicas；选择有效 rollouts/s 最优且稳定的方案。所有卡使用相同 CUDA process count，GPU0 不得额外堆 controller/server/model。
- batch 8→16 只带来约 0.9% per-episode 提升，不能把继续堆同 adapter batch 当作唯一优化。
- 训练最多使用 8 张 A100 80GB，一卡一 DDP rank 为默认；用真实数据尽量利用显存并平均预留约 10GB。评估只优化有效 rollout/s，不用 dummy tensors填显存。
- 任何 GPU launch 前实时检查 GPU owner/telemetry、进程拓扑、CUDA/runtime、storage 和 `/data/ymdai` 500GB cap；不得干扰无关进程。

## Engineering, evidence, and delivery

- 只保留一条 canonical π0.5 path；不恢复旧 runner，不新增平行版本、bank、geometry、shared update subspace、residual escape 或额外 shared trainable adapter。
- smoke 只检查 load、shape、gradient、冻结对象、OOM、resume 和环境；不解释小分母性能。
- checkpoint 保存 model/Writer/LoRA、optimizer、scheduler/scaler、sampler/data cursor、每 rank/worker RNG、env seed schedule、interaction cursor、step、episode和consumed-data state。
- 等待下载、训练或 rollout 时推进不污染运行的后续代码、文档、hash和离线验证；精度细节不改变科学结论时效率优先。
- meaningful state 后更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push。核心闭环完成前不要停在只写脚手架或只报告单一 smoke。
- optional ViVLA-style matched reproduction 和 source-only outer learning 只在核心结果之后有时间再做，不阻塞长期 Goal complete。
