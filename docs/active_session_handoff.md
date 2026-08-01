# EMBER focused active session handoff

最后更新：2026-08-01 UTC。

本文顶部保存当前运行状态、恢复入口和紧邻动作；后续编号章节是按时间保留的
历史快照。当前可执行实现仍是已经封存负结果的AP-ADR；下一架构authority是
`docs/action_forecast_writer_contextual_value_dual_read_design.md`。长期科学边界是
`AGENTS.md`与`docs/execution_brief.md`。任何接手者都必须先只读复核现场，
不能按历史快照重复启动进程。

## 0. 2026-08-01 AP-ADR结果、内部根因与紧邻动作

Amplitude-Preserving Asymmetric Dual Read Writer的fresh首小时已经自然完成；
不得重复启动、不得从profile/smoke续接，也不得让main后续改动污染其frozen源码：

```text
commit  7dffb6f7faa98e049d2cb6bc2410fbfc5d1bf0a9
frozen  /data/ymdai/.codex/worktrees/EMBER-ap-adr-formal-7dffb6f-20260801
root    /data/ymdai/outputs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801
log     /data/ymdai/logs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801.log
tmux    已自然退出
```

合同为4 ranks、仅物理GPU4–7、NUMA node1、B20、每macro 24 tasks恰好一次、
每task一条video生成一套rank16 LoRA、task内20条独立query先mean再task等权、
raw full24 mean、一次clip/AdamW/scheduler、fresh macro0→200、every25 checkpoint。
精确Writer参数`10,241,024`；source policy trainable参数0；validation/test action
gradient与test video读取均为0。首小时候选固定评测macro50/100/150/200 paired
correct400；是否resume到400只由absolute、breadth、趋势和内部主路径裁决。

实际完成200 optimizer steps/200 cycles、96,000 queries、4,800 one-video
conditions、每task 4,000 queries/200 visits，wall `3898.217s`；所有200行finite，
validation/test action读取与test video读取为0。四个paired correct400已经完成：

```text
macro50/100/150/200 correct = 91 / 81 / 94 / 91
breadth                      = 6 / 6 / 5 / 7
winner macro150 per task     = [18,1,0,37,29,9,0,0]
```

四个panel均为8 tasks×50 states、correct videos无放回、36 long-first shards、
6 replicas/6 Writer generators，teacher action读取0，paired authority一致。相邻点
gained/lost为`33/43`、`36/23`、`25/28`，没有共同单调增长；best94明显低于同期
UCP raw117、SERIAL121和v6-fast约133。一小时门失败，不resume到400、不做五臂。

macro150内部分析的最终有效root为：

```text
commit  5d93af39a7724587205e714baa5fc92b2658ea47
root    /data/ymdai/outputs/ember/pi05_as_writer_ap_adr_rawfull24_macro0150_internal_refs1_v10_5d93af3_20260801
analysis SHA d42fc4eb6694031a6a709bb10a7b82fced04cd2a6090109fc42b7d4a3a98bc2b
summary  SHA f2c572c54a37b0bd6c983322b7ae13018497822300b86de4e0c34c689846e682
```

此前refs1重放中的约5% `Q_text`漂移已定位为分析器工程污染：PI05 recursive
sampler会永久把language/expert attention backend从SDPA改成eager。训练functional
forward不调用sampler，正式evaluator又在rollout前先完成全部Writer cache生成，故
正式训练与correct400不受影响。`5d93af3`保存/恢复backend并加入回归测试；新root的
8/8 tasks逐层、effective BA、fixed-action重放误差全部严格0，checkpoint未改变。

有效根因跨8 tasks一致：same-task的`program_raw/block2/program_read/BA/action`
relative L2为`.9188/1.1051/.03210/.03005/.01668`；shuffled/reversed的block2仍为
`.09066/.07112`，到BA仅`.002689/.003903`。反转valid contextual temporal keys
只改变BA/action `.000521/.001944`。Effect-only距full BA仅`.008208`，Action-only/
Change-only距full约`.2761/.2832`，固定完整key后结论不变；Effect缩放0.5/2造成
BA `.141/.289`变化，而Action/Change缩放最多`.008/.001`。AP的causal Program
不是没形成动态，而是只被用作高熵K，真正写出的raw V由Effect DC垄断。

预注册endpoint10的三组历史portable cache已全部自然完成：v5.2-old 64、v6-fast
8×64、v6-old 64，共640套public LoRA；三份信息墙均为environment steps、validation/
test action reads与test video reads全零。对应manifest file SHA为v5.2-old
`ab158969...9de1`、v6-old `988ef3ee...4398`，v6-fast八点依次为
`14086ba7...fbee`、`488989b2...7436`、`5dfb854d...b1fb`、`1d86b51f...492b`、
`44367a8a...26f8`、`a53057ed...e989`、`ea47d859...564b`、`db47ab99...fd0a`。
当前紧邻运行是18-checkpoint no-gradient formal关联审计。它只检验执行前5 action
误差是否可作held monitor；未原封不动通过全局、family、recipe direction和逐task
四重门前，不得进入loss、训练或checkpoint选择。
之后根据AP中央职责失败与UCP raw/SERIAL训练交互，设计contextual-value职责完整的
下一架构，并完成cycle-normalized randomized-group4受控因果格。后续不使用subagent。

正式launch前的live seal已完成。longseed172真实105-frame B20三macro的step wall为
`20.567/18.717/18.644s`，峰值allocated/reserved为
`77,227,462,656/83,523,272,704` bytes；step2起semantic frontend、Core、Program、
compiler、factor全部非零可达。formal seed fresh0→1→exact-resume1→3通过，step1
全部7个payload的size、mtime和SHA逐项未变；seal commit `7dffb6f`已push。

UCP raw macro150 exact50也已补齐并与SERIAL step900严格匹配150次/task exposure。
raw→SERIAL时，删除A/D的BA/action影响由`.0653/.01269`升到
`.4184/.12999`，same-video variance/sample energy由`.1096%/.03230%`升到
`.4865%/.7322%`，证明update granularity强烈控制动态视频信号写出。但四点
correct差值为`+7/-17/+21/-3`、best仅117→121且漂移未解，因此SERIAL不是默认
recipe。后续若AP主路工作但仍漂移，必须做scheduler-only和去除cost-phase
curriculum的cycle-normalized randomized group4，而不是整体处决旧架构思想。

endpoint10 no-gradient诊断代码已在`544c0ef`/`2055a82`合入main并push；
`CUDA_VISIBLE_DEVICES=`全仓`222 passed`。它强制exact ten-step sampler在autocast
外运行、从sampler输入删除ACTION，并对候选配对、finite、sealed512与历史LoRA
provenance fail-close。v5.2-old、v6-fast八点、v6-old共10个历史候选的真实GPU
portable cache已经从三个clean frozen extension commit用四rank生成并核验，所有
tmux自然退出；下一步只剩真实CUDA profile/parity和18-candidate四rank formal。
endpoint metric在预声明关联门通过前不得进入训练。

## 0A. 2026-08-01 UCP历史状态

v5.2 task-complete、SPG、UCP raw-full24和serial-4同曝光正式训练均已完成。
serial-4训练已自然退出，不得重复启动或修改其frozen worktree：

```text
commit  3db82dfd3b42f6b424790ef19716013ac1cf4fce
frozen  /data/ymdai/.codex/worktrees/EMBER-ucp-serial4-formal-3db82df-20260801
root    /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_formal_dev_r4_b20_seed7_3db82df_20260801
log     /data/ymdai/logs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_formal_dev_r4_b20_seed7_3db82df_20260801.log
```

fresh训练完成1,200 optimizer updates/200完整cycles，wall `4197.076s`；严格等于
raw-full24的96,000 queries、4,800单视频条件和每task 200 visits。1200行metrics
连续finite，8个150-step checkpoint均通过manifest完整性校验，validation/test
action读取及test video value读取均为0。raw→serial的4,800个rank/task/visit/demo/
sampled-frame assignment逐项一致，normalized replay SHA为`d406f2f1...80cc`。
held loss在step150..1200为
`.132407/.131304/.133484/.132973/.130352/.132508/.132237/.132918`，不作候选预选。

当前必须继承四条active paired correct400评测；均在tmux
`ember-ucp-serial4-correct400-3db82df`，使用同一8 tasks×50 states panel、
without-replacement videos、dynamic long-first 36 shards、6 persistent replicas和
6 Writer generators：

```text
step0300 -> GPU4 -> /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_correct400_noreplacement_seed7_step0300_3db82df_20260801
step0600 -> GPU5 -> /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_correct400_noreplacement_seed7_step0600_3db82df_20260801
step0900 -> GPU6 -> /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_correct400_noreplacement_seed7_step0900_3db82df_20260801
step1200 -> GPU7 -> /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_correct400_noreplacement_seed7_step1200_3db82df_20260801
```

四个launcher与24个Writer workers已确认存活、prepared contracts各为400 states且
teacher action读取0；当前在生成四套400-entry LoRA cache。下一动作是等待四条自然
完成，严格核验400 rows及pairing，再运行candidate-curve和raw-vs-serial同曝光分析。

以下UCP raw-full24 formal只作已完成provenance，不得按本节重复启动：

```text
commit  c94f1c6bb6479625c6c4ffb1a3b28e3fba7730c1
frozen  /data/ymdai/.codex/worktrees/EMBER-ucp-formal-c94f1c6-20260801
root    /data/ymdai/outputs/ember/pi05_as_writer_ucp_rawfull24_decay400_formal_dev_r4_b20_seed7_c94f1c6_20260801
log     /data/ymdai/logs/ember/pi05_as_writer_ucp_rawfull24_decay400_formal_dev_r4_b20_seed7_c94f1c6_20260801.log
```

run-contract payload/file SHA为`b372fb8f...d13807`/`832a7ba4...9c628a`，config
SHA为`c8202053...2cf12`。fresh macro0→200自然完成，耗时约64.48分钟，共
96,000 queries、4,800个task-video visits；validation/test action gradient和读取
均为0。所有25-step checkpoint完整。

paired correct400为：

```text
macro50 / 100 / 150 / 200 = 82 / 117 / 100 / 110
```

single winner是macro100，但四点union为169，比best高52；三次相邻转移的
gained/lost为`64/29`、`18/35`、`39/29`。breadth nonzero为`7/7/5/7`，top2
贡献为`61.0/66.7/65.0/62.7%`，Spatial始终近零。train loss持续下降而held
functional loss维持`.131–.132`。因此一小时门失败，不resume到400、不做五臂。
正式curve与drift为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_ucp_rawfull24_decay400_candidate_curve_seed7_c94f1c6_20260801/analysis.json
SHA256 6462e6532791df98803db4d7af65ec71755e6cfd4de2bd093d98a136fdf4fc25
drift_analysis.json
SHA256 084b3a100a514eefcb99d29f688a942603eba512e046115a1a5d4c03e3db6675
```

macro100 refs1内部纵向验证已经在analysis commit `a4b06f5`完成。原诊断失败不是
模型状态变化，而是canonical B5与recompute B1触发CUDA BF16 batch-shape数值路径
差异；修复后保持五条件carrier batch、只改/抽row0，原`2e-5` fail-close未放宽，
Program、coordinates、factor、public A/B、effective BA和action全部重算误差0。
refs1 root与analysis SHA为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_ucp_rawfull24_macro0100_internal_ref1_seed7_a4b06f5_20260801
analysis.json SHA256 8cef65bcd1b3f386fb43af9521119125114818bf6aa7c14cfd895de5d791b470
```

reader target/rank centered energy为`.240/.117`，所以SPG同质化已消失；
same/wrong/shuffled/reversed的final Program→effective BA→fixed action relative L2
为`.190/.492/.352/.447 → .040/.190/.065/.107 → .014/.067/.016/.030`。但固定X
只改A/D时effective BA仅`.014/.021/.024/.024`，说明dynamic teaching仍弱；
LoRA norm约`59.5`，q/v跨层cosine约`.917/.923`。

首次exact50零rollout内部分析已经作为工程失败封存；当前没有需要继承的GPU进程。
不得复用该output root或修改其frozen worktree：

```text
commit  a4b06f5dc9f0a5c0fbd75739d7dde2b10e4e2504
frozen  /data/ymdai/.codex/worktrees/EMBER-ucp-analysis-ref1-a4b06f5-20260801
root    /data/ymdai/outputs/ember/pi05_as_writer_ucp_rawfull24_macro0100_internal_exact50_seed7_a4b06f5_20260801
log     /data/ymdai/logs/ember/pi05_as_writer_ucp_rawfull24_macro0100_internal_exact50_seed7_a4b06f5_20260801.log
```

rank1在进入首个seal collective前发生了本地异常；旧代码把异常吞成字符串后等待
仍在计算的其余ranks，最终NCCL 600秒watchdog只留下二次timeout。sequence和
精确video schedule审计排除了collective错序、缺视频与正常负载长尾；该root只有
run contract，不产生任何科学结果。commit `874e5f1`把reference异常补齐
rank/task/reference上下文并立即落盘、直接re-raise，由torchrun fail-fast；成功
路径先原子写rank rows，再用analysis-only两小时Gloo控制组协调，失败路径不做
可能遮盖原trace的process-group cleanup。新refs2已精确定位原异常为rank1的
`libero_spatial task3/reference1` rank-gauge sanity失败；torchrun立即终止其余
ranks。`8f8716b`进一步把BA/action/raw A/B四组实际误差写入异常。instrumented
clean root得到raw A/B relative L2 `.74184/.13602`、effective BA
`1.299e-9`且max absolute`7.45e-9`，证明gauge实现正确；fixed action
`.002047`来自bf16两段LoRA的rank reduction顺序变化。修复仍对finite和BA
`2e-5` fail-close，只把action execution drift降为记录量。下一步用新clean root
验证refs2通过，再以另一root完成exact50。
SPG macro50/100/150/200 correct400为`97/115/77/100`，不续第二小时。SPG
Program本身对same/wrong/shuffled/reversed有`.967/1.186/1.193/1.202` relative
L2，但target/rank reader近均匀，差异到effective BA压成
`.066/.221/.116/.116`；same-task video variance/sample energy从macro50
`.419%`降到macro200`.210%`。CP投影解决负pair但没有解决task轮换。

raw-full24实现历史worktree为：

```text
/data/ymdai/.codex/worktrees/EMBER-unified-program-534064a-20260801
head 0d4c27114991c8887c4dd5479ec42fdd11fd63a3
```

exposure-matched serial-4已从独立写worktree集成main，当前尚未formal launch：

```text
frozen /data/ymdai/.codex/worktrees/EMBER-ucp-serial4-10a71a1-20260801
head   10a71a148001a8d257017c6b4bfa20f2b0c11ac0
main commits ccdf21f / 92548ed / 10a71a1 docs authority
```

实现用六phase精确重建同一full24 task/video/query exposure与rank内long-first顺序；
LR按cycle重复六次，fresh serial config/checkpoint/rank schemas及midcycle resume cursor
不兼容旧full24。formal checkpoint/stage边界强制整除6，profile/formal teacher-video
seed明确为`172/20260722`。全仓`233 passed`，architecture guard无hard violation；
尚未做GPU profile、resume smoke或formal launch。

Unified Causal Program canonical CPU实现已经完成：统一`X/A/outgoing D` causal
grid、单级normalized target/rank raw-value reader、无Core add/mixer；训练为严格raw
full24 mean并使用无偏20-strata B20。真实参数`7,683,328`，全仓`203 passed`；step0
identity、causal prefix、outgoing alignment、target/rank routing、零内容不造值和
raw-gradient/sampler exact resume合同均通过。canonical config为
`configs/pi05_as_writer_unified_causal_program_full24_decay400_v1.json`，现已seal为
B20 fresh macro0→200 formal authority。

最长105-frame profile root为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_ucp_rawfull24_profile_b20_longseed172_r4_0d4c271_20260801
```

task38/demo36的105 sampled frames已真实进入step1。三macro wall为
`20.394/18.494/18.504s`，峰值allocated/reserved为
`77,127,082,496/83,345,014,784` bytes；每步严格24 tasks、480 queries、24
one-video conditions，所有metric与checkpoint finite，step2起四个主块梯度非零。
run contract/metrics/summary SHA分别为`3d375caa...554ab6`、
`06320db8...1dd24`、`bc8e3e4e...d1b18`。

formal-seed exact-resume root为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_ucp_rawfull24_resume_smoke_b20_formalseed_r4_0d4c271_20260801
```

fresh0→1后exact-resume1→3完成；step1 manifest、Writer、trainer与四个rank state
逐SHA不变，三步loss为`.153645/.149302/.149466`，grad-before-clip为
`.014274/.038833/.654902`，LR与data/RNG cursor连续。run contract/metrics/summary
SHA为`31187bf9...7d9d0`、`84681e63...3c2f`、`489ca502...c0c5c`。

clean `c4b85e8` refs2与exact50均已自然完成；exact50 root为
`/data/ymdai/outputs/ember/pi05_as_writer_ucp_rawfull24_macro0100_internal_exact50_v2_seed7_c4b85e8_20260801`。
它严格包含8 tasks×50 references=400 rows、四rank各100、0 rollouts且无failure；
analysis/summary SHA为`a6e40cd6...25a8`/`386a04f5...acaa`。pooled same-task
effective-BA/fixed-action centered variance/sample energy为`.09008%/.01656%`，
确认dynamic教学弱遍及八task。旧exact50 tmux已自然退出。

serial-4最长105-frame B20 profile已从clean detached `10a71a1`自然完成：

```text
root /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_profile_b20_longseed172_r4_10a71a1_20260801
log  /data/ymdai/logs/ember/pi05_as_writer_ucp_serial4_profile_b20_longseed172_r4_10a71a1_20260801.log
```

profile共18 updates=3完整cycles，checkpoint为`1,3,5,6,7,12,18`；首update实际
包含task38/demo36的105 sampled frames。每cycle恰好24 unique tasks，1,440 queries/
72 videos，峰值allocated/reserved为`76,971,835,904/83,647,004,672` bytes，全部
finite且step2起四个主模块梯度可达。run contract/metrics/summary SHA为
`a24c015e...13fa1`/`031f2e31...5995`/`ec44cfd8...103`。

formal seed smoke也已完成：fresh0→1、resume1→3、resume3→7；step1与step3全部
checkpoint文件在后续resume后逐项SHA不变，metrics phase为`0..5,0`，前六步覆盖
24 unique tasks，scheduler只在step6推进且step7使用`.0002275`。root为
`/data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_resume_smoke_b20_formalseed_r4_10a71a1_20260801`，
run contract/metrics/summary SHA为`2c350077...0c28`/`3d09ac35...6cd`/
`eb2c880a...9fdb`。旧profile/resume tmux均已自然退出。serial-4
严格用`cycle,phase=divmod(update,6)`重建同一full24 cost-balanced cycle；六更新
覆盖24 tasks，LR在同cycle六次保持不变。禁止naive连续warmup102/decay2400。
canonical config现已seal。seal commit `3db82df`已push并建立clean frozen
worktree；fresh identity正式0→1,200当前由tmux
`ember-ucp-serial4-3db82df`运行，不得重复启动或从raw-full24/smoke warm-start：

```text
frozen /data/ymdai/.codex/worktrees/EMBER-ucp-serial4-formal-3db82df-20260801
root   /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_formal_dev_r4_b20_seed7_3db82df_20260801
log    /data/ymdai/logs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_formal_dev_r4_b20_seed7_3db82df_20260801.log
config SHA256 e6a604a5bab0e4656c54db2c2fc35608fc55199b42cf22202e5a030f50053cab
run contract SHA256 995f248ae322f7a1b1fef6a11f18b3c7e7a4da8a9471db62b8a4ec8ea68e404b
```

确切命令：

```bash
numactl --cpunodebind=1 --membind=1 env \
  PYTHONPATH=/data/ymdai/.codex/worktrees/EMBER-ucp-serial4-formal-3db82df-20260801/src \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 \
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 \
  scripts/train_as_writer.py \
  --config configs/pi05_as_writer_unified_causal_program_serial4_exposurematched_v1.json \
  --mode formal \
  --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
  --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir /data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_formal_dev_r4_b20_seed7_3db82df_20260801 \
  --total-steps 2400 --stop-after-step 1200 --checkpoint-steps every:150 \
  --batch-size 20 --num-workers 2 --log-every 1 --skip-data-sha
```

首个完整cycle已确认六phase覆盖24 unique tasks；四rank frames总cost为
`207/216/206/204`且rank内严格long-first；每步4 videos/80 queries，step2起四个
主模块梯度可达，全部finite，信息墙读取计数0。训练完成后评测
300/600/900/1200 paired correct400。

历史架构审计必须按组件×recipe解释：v7以后所有正式负结果都使用同一fast
task-complete recipe，没有old recipe反事实。只能删除已被内部机制证据独立否定
的局部模块，不能把anchors、causal Procedure、双流或target-first/rank-last整体
判死。

匹配每task 150次video exposure的正式审计为v5.2 old/new=`132/51`、v6
old/new=`95/111`，recipe effect=`-81/+16`、描述性DiD=`97`；该反差支持强
architecture×training-bundle交互，但optimizer/scheduler/AdamW时钟仍混杂。
serial long-first重放中visit phase与sampled-frame cost Pearson=`-.8331`，task38
始终phase0，结果必须同时按phase/cost审计。

UCP phase-estimator audit：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_action_query_phase_variance_macro0200_seed7_20260801/analysis.json
SHA256 636c3072e59e7ca1df04ad438c0a837de1dcc515e6f370639519ebcbdb875c2f
```

## 0A. SPG launch历史快照（已完成，不得按本节重启）

v5.2 task-complete训练、四候选、winner五臂、exact50 LoRA几何和五条件内部传递
已经全部完成；当前没有需要继承的v5.2训练或评测。candidate correct400为
`51/91/106/120`，macro400 winner五臂为`120/109/107/111/124`。本轮没有行为
视频特异性；内部虽确认顺序变化传到effective BA与policy action，但same-task
视频中心化方差只占sample energy的`.6844%`，且方向未与闭环收益对齐。

SPG独立写worktree仍为：

```text
/data/ymdai/.codex/worktrees/EMBER-spg-60f4508-20260731
```

canonical实现精确参数`10,633,216`。最长105-frame、B20、四rank的三宏步profile
已在CP通信完成性修复后稳定通过，step wall为
`20.5359/18.5778/18.5461s`，峰值allocated/reserved为
`77,203,449,344/83,529,556,160` bytes。72个视频条件、1,440 queries均finite，
每步24 tasks唯一、rank内long-first，macro2起五个主模块都有有限非零梯度。

profile root与三项seal为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_spg_cp24_profile_b20_longseed172_sync_v2_7c1b9fc_20260801
run contract  f4787296e57c1c31f6a011e8e6b3b37e6704b198ef8ea613cf26b607fa87ca17
metrics       b70d7b2920139235c11a1e3028e18cda29416745f7235d8fc84eb3c87eee6c78
run summary   74f3d35eb7a1eae5c901bdac4ea2ca539a51bd38413b1d01d46a74569b34d672
```

首次profile的macro2在共卡条件下stall。phase trace证明NCCL的同步Python接口只
保证all-gather排入CUDA stream，快rank可排入全部bounded chunks而慢rank尚未进入
首chunk。canonical修复在每个CUDA Gram chunk后形成显式stream completion
boundary，并记录all-gather/sync计数；它不改变Gram、PCGrad或optimizer数学。

formal teacher seed `20260722`的fresh0→1→exact-resume1→3已经在clean
`f6d487635abdaf3bd5039df667e20f8730bb2110`上通过。三步loss为
`.152172/.147053/.154108`，gradient norm为`.031343/.072098/.192859`，wall为
`19.466/19.831/19.245s`；每步13次Gram chunk all-gather与13次CUDA completion
严格对应。step1的manifest、Writer、trainer和四rank state在resume后哈希逐项
不变，metrics连续3行、invocations明确为fresh与resume两段，validation/test
action和video读取均为0。root与主要seal为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_spg_cp24_resume_smoke_b20_formalseed_f6d4876_20260801
run contract file  f07ac8ebf6132bced4946d93561b57b588d6fc664ed1918eaf8f3dd823aeedbf
metrics            d04b2dfb40d5183a34aa742220acb3f4a94ff9428684976e8c23b55e772affea
run summary        c27e11c07194e6b72d32baa025f4e71ea525780f3252361c47559f0d022a2ffe
step1 manifest     3a0fe8106ab6fb88dfe310079aef4f7df197dec4f79c578c9d27cf5e6a715b07
step3 manifest     e2d8369c3db7927d3b0a360eefb845643a250cecbeb87bc3f5affbc5edd0617d
```

resume seal已提交并push；`HEAD=origin/main=79fb7ee2bfa191438dd5e83642fe16b499e90e58`。
SPG正式fresh macro0→200当时从以下frozen worktree/tmux/root运行；现已完成且tmux
结束，以下只保存provenance：

```text
worktree  /data/ymdai/.codex/worktrees/EMBER-spg-formal-79fb7ee-20260801
tmux      ember-spg-cp24-79fb7ee
root      /data/ymdai/outputs/ember/pi05_as_writer_spg_cp24_decay400_formal_dev_r4_b20_seed7_79fb7ee_20260801
log       /data/ymdai/logs/ember/pi05_as_writer_spg_cp24_decay400_formal_dev_r4_b20_seed7_79fb7ee_20260801.log
```

exact command为：

```bash
numactl --cpunodebind=1 --membind=1 env \
  PYTHONPATH=/data/ymdai/.codex/worktrees/EMBER-spg-formal-79fb7ee-20260801/src \
  PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
  /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 \
  scripts/train_as_writer.py \
  --config configs/pi05_as_writer_semantic_program_grid_cp24_decay400_v1.json \
  --mode formal \
  --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
  --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir /data/ymdai/outputs/ember/pi05_as_writer_spg_cp24_decay400_formal_dev_r4_b20_seed7_79fb7ee_20260801 \
  --total-steps 400 --stop-after-step 200 --checkpoint-steps every:25 \
  --batch-size 20 --num-workers 2 --log-every 1 --skip-data-sha
```

launch preflight只查询GPU4–7；四卡均空闲，个人占用`359,735,353,342` bytes，
config SHA为`097ed082f27955d9193c6fb4efe376a7f011d8050eabd0d362499c31d4f796a0`。
当时四个rank分别占GPU4/5/6/7且无额外CUDA角色。首macro为`19.431s`、loss
`.152172`、grad norm`.031343`、LR`1.6667e-5`；24 tasks/480 queries/24 videos
合同、rank内long-first和`13 gather=13 CUDA completion`均通过。该run后来完成到
macro200并评测macro50/100/150/200为`97/115/77/100`，不得重复启动。

GPU只可查询和使用物理4–7；0–3不得查询或进入visible set。4–7可按owner授权
共卡，但不得杀、暂停、重置或干扰其他进程。

## 0.1 v5.2最终正式证据

正式训练root：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v52_taskcomplete_decay400_formal_dev_r4_b20_seed7_60f4508_20260731
```

macro150/200/350/400逐task分别为：

```text
macro150  5/0/0/31/12/1/0/2
macro200  20/0/0/35/24/9/1/2
macro350  23/0/0/18/33/31/0/1
macro400  33/0/0/30/25/32/0/0
```

winner五臂逐task为：

```text
Long-1    33/31/20/23/27
Long-2     0/ 0/ 2/ 0/ 0
Goal-3     0/ 0/ 0/ 0/ 0
Goal-6    30/29/37/31/37
Object-1  25/24/14/21/22
Object-3  32/24/33/34/35
Spatial-1  0/ 1/ 1/ 0/ 2
Spatial-3  0/ 0/ 0/ 2/ 1
```

严格paired分析、exact50 geometry和完整内部传递分别封存在：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v52_taskcomplete_decay400_winner_five_arm_paired_seed7_macro0400_60f4508_20260731/analysis.json
pi05_as_writer_v52_taskcomplete_decay400_winner_lora_geometry_correct_macro0400_v2_60f4508_20260801/analysis.json
pi05_as_writer_v52_taskcomplete_decay400_winner_internal_specificity_macro0400_refs2_v4_60f4508_20260801/
```

exact50 effective norm/stable rank/top energy为
`113.5185/1.000305/99.9699%`，q/v/action energy为
`72.701/27.269/.030%`，q/v跨层cosine`.9723/.9837`。functional validation在
macro200更优而behavior到macro400继续上升，且checkpoint间出现明显task轮换；
不得把functional loss作为closed-loop选择器，也不得继续同一v5.2 recipe。

## 0.3 2026-07-31历史恢复合同（不得按此重复启动）

强制authority、代码、Git历史和正式artifact审计已完成。exact v5.2
task-complete正式训练已经完成；当时tmux `ember-v52-candidates-60f4508`在物理
GPU4–7上分别评测macro150/200/350/400 paired correct400。输出root统一前缀为
`/data/ymdai/outputs/ember/pi05_as_writer_v52_taskcomplete_decay400_correct400_`
`noreplacement_seed7_macro*60f4508_20260731`。不得重复启动这些root。

```text
config: configs/pi05_as_writer_language_axial_v5_2_taskcomplete_decay400_v1.json
fresh identity macro0→400                 已完成
every25 checkpoint
paired correct400: macro150/200/350/400    当时正在四卡并行
single-checkpoint winner
winner formal correct/same/wrong/shuffled/reversed 400
winner internal Core/Procedure/LoRA/action/rank/layer/video analysis
```

正式科学尺度是400 macros，不因共卡导致wall-clock略超机械两小时而删减updates。
必须使用一条video生成一套LoRA，不融合checkpoint，不平均多video/LoRA。所有
evaluator worker继续long-task-first；没有long shard后才领取其他task。

仓库、Git历史、正式outputs与内部analysis的完整审计及SPG独立复核均已完成。
SPG canonical CPU实现位于独立写worktree：

```text
/data/ymdai/.codex/worktrees/EMBER-spg-60f4508-20260731
```

无论v5.2新recipe结果好坏，都要实现并实验SPG。SPG及其后任何新整体架构都先
fresh训练约一小时并做paired correct400：若未达到同期一小时v5.2/v6水平且无
明确上升价值，不续第二小时、不做昂贵行为特异性，只做充分无rollout内部分析；
若同档或更好、或趋势明确有价值，才exact-resume第二小时并在强single-checkpoint
上做行为五臂。每个负结果都必须定位前向与训练的最早根因，再整体重构，禁止在
失败结构上叠gate、scale、旁路或局部residual。

focused Goal不是机械`correct400>=150`。150只是里程碑；只要仍存在task漂移、
视频学习不足、跨层/跨rank退化、closed-loop off-manifold或可信改进空间，就持续
执行“整体设计→一小时训练→评测/内部分析→续训或重构”。只有接手agent在其能力
范围内已找不到可信提升空间，才允许停止并向owner汇报。

GPU只可使用物理4–7，0–3不得查询或进入visible set。4–7可按owner授权与他人
共卡，但不得杀、暂停、重置或干扰其他进程。

## 0.4 v5.2正式训练与候选启动历史快照

正式训练root：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v52_taskcomplete_decay400_formal_dev_r4_b20_seed7_60f4508_20260731
```

run contract SHA为
`152c0818f3266e8abbc9ddeca1b07cc1128f32b5063d24f717dc818f1436088e`；
macro0→400完成`400`行metrics、`192,000` action queries、`9,600`单视频条件，
训练wall `9695.1329s`。macro400 train loss `.09633848`、grad norm `.10484845`、
functional validation `.13686878`；这些不构成closed-loop证据。run summary SHA为
`857f0111a3b52472662e3293a5e7b3dc094326eec47b4d665cd23378e6fdee66`。

候选评测启动前现场核验main/origin/frozen均为clean `60f4508`，个人占用
`350,451,040,256` bytes，低于500GB cap；GPU4–7仅GPU4有一项约978MiB外部进程，
按owner共卡授权未干扰。四个candidate启动命令均显式CVD单卡、B-scale1、
without-replacement、6 replicas、6 Writer generators、batch16与long-first dynamic
queue。

## 0.2 v5.2内部几何完成、正式训练前的封存快照

v5.2实现基线commit为`799aa6676b7f94f337d019956366eb7f180ba83a`。
它把exact v5.2 topology接到mature full24 task-complete、B20、
cost-balanced long-first与fast-decay400 recipe；canonical config为：

```text
configs/pi05_as_writer_language_axial_v5_2_taskcomplete_decay400_v1.json
```

精确Writer参数为`10,237,704`；最长105-frame三macro profile和formal-seed
fresh0→1→exact-resume1→3已经通过。以下“本session”叙述是正式训练前历史快照，
不得覆盖上面的当前状态。

本session实际完成了v5.2 step900正式validation 8 tasks×50 correct-video LoRA
的重新生成与内部分析；没有创建LIBERO env，`rollout_shards_executed=0`。永久
保留的分析文件：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v5_2_step900_lora_geometry_529da6b_20260731/analysis.json
```

SHA256为
`9d816baadace851153415a06334efad6f9927bf334f014d5e8ae760be357e1af`。
关键结果：

```text
effective norm mean                              140.441
stable rank / entropy rank                       1.01256 / 1.04467
top singular energy                              99.0244%
q / v energy                                     73.4476% / 26.5502%
q / v cross-layer BA cosine                      .96211 / .98166
same-task centered variance / sample energy      1.6655%
video variation orthogonal / scale-like          89.35% / 10.65%
q / v coordinate participation                   15.83 / 15.92 of 16
q / v negative component pairs                   0% / 0%
```

所以v5.2的16个坐标不是能量失衡或负向相消；它们建设性同向地形成近rank1。
Target-Spectral强制高rank/正交已得到`34/400`，证明不能把rank数本身当病灶。
v5.2比v6保留更大的same-task视频创新，但Visual Transition与task-complete
recipe仍然混杂，必须用缺失的v5.2新recipe cell判定。

新session第一项且唯一预定正式实验：现场只读核验Git、存储、GPU4–7后，从
identity fresh运行上述v5.2 config的macro0→200，默认exact-resume到400；
评测macro150/200/350/400 paired correct400，不做checkpoint融合。当前session
没有替新session启动它。GPU0–3不得查询或使用；GPU4–7可按owner授权共卡，但
不得干扰他人进程。

本节末尾曾引用的Coherent-Procedure/B-only residual提案已经撤回；不得实现。
当前下一架构只认`docs/action_forecast_writer_semantic_program_grid_design.md`。

## 0.2 Target-Spectral负结果历史快照（不得执行）

Target-Spectral fresh macro0→200已自然完成：200个finite optimizer updates、
4,800个single-video LoRA conditions、96,000个action queries和every25的8个
完整checkpoint。macro50/100/150/200 paired correct400为：

```text
30 / 12 / 18 / 34
```

四点均为相同8 tasks×50 states、每task teacher demos 0–49无放回各一次、同一
state/video/RNG配对；36/36 shards、400 LoRA caches、结果与run-summary hash均
完整，worker无OOM/NaN/traceback。macro200虽名义6/8 tasks非零，但31/34个成功
集中在Long-1、Goal-6和Object-1。它低于source base `48`、corrected Source-SFT
`109`、v5.2 `132`、v6 `143`和门`150`，因此没有行为级same/wrong/shuffled/
reversed rollout，也没有续第二小时。

winner macro200的CPU rank/layer/video分析在：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_target_spectral_rank_layer_cpu_aa9d89a_20260731/analysis.json
```

无rollout五条件内部分析在：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_target_spectral_single_checkpoint_macro0200_internal_specificity_refs2_aa9d89a_20260731/summary.json
```

关键对照为：

```text
                              Target m200   v6 m200     v6 m400
correct400                    34            133         143
effective LoRA norm           25.87         94.71       108.91
effective stable rank         3.3245        1.00017     1.00027
q / v energy                  39.0/60.9%    74.5/25.5% 73.8/26.2%
q / v layer-energy CV         1.294/.805    .047/.043   .063/.054
q / v cross-layer cosine      .032/.066     .968/.988   .969/.985
```

Target-Spectral确实消除了伪rank16：约15.6/16个rank有非平凡scale，q/v stable
rank为`1.96/4.66`，A行、B列和component近乎正交。但它同时拆散了v6中高增益、
q-dominant、跨层协调的公共写入方向。16个同向component按16建设性相加，16个
正交component只按sqrt(16)合成；理论4倍与实测v6/Target范数`3.66×`吻合。
same-task视频相对中心化方差从v6 m200 `.44%`升至`.65%`，但因总LoRA大幅缩小，
绝对视频创新RMS约低3倍。

这不是上游失败：Target m200与v6 m200训练functional loss为`.10023/.10043`，
Core/Procedure绝对量和shuffle/reverse order signal同量级。Target内部
shuffle/reverse的Procedure-centered差异为`1.349/1.525`，到effective LoRA
仍有`.302/.352`、到fixed-query policy action为`.0365/.0733`；固定正确Core
只改变Procedure可复现这些差异，Core-only几乎不随顺序变化。它能“对视频敏感”，
却把这种差异写进了闭环无效甚至破坏source skill的方向。近rank1因此不是v6
性能瓶颈的充分证据，Target-Spectral只作负结果，不在其scale/gate上打补丁。

Prior–Innovation fresh macro50/100/150/200 paired correct400最终为
`100/61/89/88`，没有恢复v5.2 `132`或v6 `143`，因此没有续训或做行为级
视频控制。它进一步证明，仅改变Core/Procedure读法不能解决当前主要瓶颈。

跨v6、Core-Program和Prior的CPU复核把更稳定的结构病灶定位在public-LoRA
编译端：

```text
v6 effective BA stable rank                1.00009→1.00028
v6 q/v B-column cosine                     .9974量级
v6跨层q/v effective-delta cosine           .969/.983+
同task视频中心化方差 / mean-LoRA能量        ~0.30%
视频方差中正交方向占比                     ~90.6%
```

视频不是只改变共同方向的scale；它产生了真实但很小的方向创新。直接塌缩主要
发生在B的16列，且38个真实policy targets跨层共享近乎同一更新方向。因此当前
第一实验只修复这一处，不同时改optimizer，避免把decoder与任务梯度冲突混为
一谈。

当前源码仍封存在
[`docs/action_forecast_writer_target_spectral_design.md`](action_forecast_writer_target_spectral_design.md)
定义的 Target-Spectral，直到下一架构决策原位替换；这不构成resume授权：

```text
v6 Q_text + M_f + G_f + Semantic Core
native 50-suffix mean Action + teacher visual transition
→ two-layer causal Procedure

Core/Procedure先形成38个真实policy-target states
→ target-specific value-coordinate transforms
→ 最后才展开16个代数rank coordinates
→ row-orthogonal A basis
→ column-orthogonal U basis × 16 learned spectral scales
→ complete rank16 public LoRA
```

该gauge允许模型诚实选择rank1，但不允许复制16条相同A/B方向伪装rank16。
step0 spectral scale为零，因此effective delta严格identity。精确参数为
`14,495,744`。训练和推理始终是一条teacher video生成一套LoRA；不做
multi-video平均。action queries继续与video同task但跨episode独立。

活动源码/config/schema已原位切换；Prior config和兼容执行路径退役。
canonical实现commit为`f8bbce6`且已push；全仓`196 passed`、compileall、
JSON和diff check通过。

GPU4–7独立最长视频B20 profile已通过：首步包含task38/demo36真实105个
stride-5帧，三步均finite；后两步均值`25.488 queries/s`、
`191.159 macro/hour`，峰值allocated/reserved为
`77,074,980,864/83,649,101,824 bytes`，不触发B16。step1→3的530个
trainable tensors中458个变化，compiler、factor/scale heads、Core、
transition、Procedure和全部主Meta组均finite且可达；72个Action Meta-LoRA A
因新增spectral scale→AdaLN→Meta-B分级zero-init在三步后尚未打开，配对B均已
变化，正式训练会在后续step继续打开。

formal teacher seed`20260722`下的独立root也已完成fresh0→1→exact-resume
1→3：metrics、LR、task/video/query cursor连续，累计72 video conditions与
1,440 queries；validation/test action reads和test video value reads均为0。
resume前后macro1的manifest、Writer、trainer和四rank state共七个文件逐项SHA
完全不变。

紧邻顺序：

```text
保留v6已验证的公共高增益主写入manifold
→ 把额外rank定义为可选、zero-init的视频创新容量
→ 用现有内部证据重新设计compiler与训练更新规则
→ owner讨论后再封存下一fresh架构
```

当前只可使用物理GPU4–7中现场空闲的卡做内部分析，暂时不得启动正式训练；
不得查询或使用GPU0–3。Target-Spectral不得resume，也不做昂贵视频控制臂。

当前活动配置：

```text
configs/pi05_as_writer_target_spectral.json
source policy:
/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000
data root:
/data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a
```

## 1. Core-Program与Recenter历史快照（整节已失效，不得执行其命令）

owner在Loom macro50/100/150/200 correct400仅为`79/106/105/112`且内部
correspondence/confidence/gap缺少可靠锚点后，授权继续从第一性原理重设计，
而不是围绕Loom打补丁。唯一canonical Writer已原位切换为Recenter：
Action-Anchored Core-Keyed Centered Procedure Writer。schema fresh不兼容，
精确trainable参数为`10,709,248`。

当前session-local Goal要求持续执行：

```text
第一性原理重构
→ 约一小时fresh训练
→ 四个checkpoint paired correct400
→ 未恢复同期旧架构：不做行为级特异性，只做详细内部/梯度分析后再重构
→ 恢复或超过同期：续第二小时
→ 达150或稳定145+：才做winner四个额外视频特异性臂
```

视频特异性只用于证明性能确实由输入视频学习，不允许以牺牲absolute换取漂亮
控制臂，也不允许用checkpoint融合掩盖单点漂移。

Recenter恢复v5.2/v6的原生50-token suffix mean Action主干，保留稳定
task-query patch grounding与v6 Semantic Core；task-grounded transition只能
以Action RMS四分之一为上限作残差修正。单路causal Procedure进入新的
Core-keyed compiler：Core只提供slot地址和`[0.75,1.25]`乘性调制，value读取
raw time-centered Procedure，slot mixer混合方向后恢复输入RMS。因此
constant/zero Procedure无论Core为何都保持identity，不存在Loom的raw
correspondence、confidence、Teacher/Policy gap或双流Procedure。

canonical实现、fresh配置和确定性结构测试已经完成；Loom-only
`relations.py`已退役。全仓`196 passed`、compileall、diff check和architecture
guard均通过；额外修复了zero transition反向NaN、zero/near-zero mixer梯度和
bf16非二次幂constant Procedure伪残差。Recenter没有继承Loom证据，而是在
GPU4–7独立完成真实105-frame视频B20三macro profile和正式seed
fresh0→1→exact-resume1→3；全部step finite、step1文件hash不变，
`10,709,248/10,709,248`个trainable参数在step1→3间变化。配置已经seal。
紧邻动作是：

```text
GPU4–7 only
→ live preflight
→ fresh macro0→200，每25 macro checkpoint
→ paired correct400: macro50/100/150/200
```

选择single-checkpoint observed-best后，只有absolute达到同期有效架构水平才
做正式五臂rollout；否则先做详细内部数值分析并重新定位根因。除GPU4–7外不得
查询或使用其它GPU。

Recenter首段正式launch合同：

```text
sealed config  commit 1ef4e08; configs/pi05_as_writer_recenter.json
source policy  /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000
data root      /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a
topology       physical GPU4–7; 4 DDP ranks; NUMA node1; B20; 2 workers/rank
scale          fresh macro0→200; 4,800 videos; 96,000 queries; every25 checkpoint
output         /data/ymdai/outputs/ember/pi05_as_writer_recenter_taskcomplete_decay400_dev_r4_b20_seed7_s2400_1ef4e08_20260730
integrity      sealed manifest + exact sizes + HDF5 schema; no runtime full-data SHA
storage        /data/ymdai 425.60GB before profile; projected profile/resume/formal/eval <8GB
selection      paired correct400 at macro50/100/150/200; no checkpoint fusion
```

该root必须fresh且不存在；启动前再次只查询GPU4–7、核对500GB个人上限和
clean/pushed Git状态。正式命令由run contract逐字段保存。

v6 fast-decay已按owner后续要求从macro400 exact-resume到600。完整
correct400曲线为：

```text
macro       50  100  150  200  250  300  350  400  450  500  550  600
successes  106   64  111  133  132  117  138  143  131  130  132  126
```

macro400不再是右端点，但仍是单checkpoint observed-best。macro600相对400
为`14 gained/31 lost,p=.01609`，已经形成可信post-best下降，不再续训该
task-complete fast-decay recipe。macro400正式五臂为
`correct/same/wrong/shuffled/reversed=143/135/125/128/129`：same同档，
correct相对wrong显著，但shuffled/reversed只保留`+15/+14`且不显著。内部
反事实确认顺序信号仍由Visual Transition进入Procedure并传到LoRA/action，
只是下游差异较弱。

同一套v6拓扑的旧训练范式对照也已fresh完成：每rank每update一个task、全局
4 tasks、连续6 updates完整覆盖24 tasks、无gradient accumulation、固定
`B20`。正式run在约1.01小时内完成900 updates；step100/500/700/900的
paired correct400为`98/121/76/95`，step500是observed-best。其正式五臂为
`121/122/111/84/47`：same同档，shuffled/reversed显著下降，但wrong只低10、
不显著且仅2个tasks正向，因此语义视频门失败。内部检查显示old recipe把
shuffled/reversed的Procedure差异强烈传到effective LoRA/action，Core-only
仍近零。

当前证据排除了两个简单解释：

1. v6拓扑并非天然无法产生顺序特异性；不改拓扑只改训练粒度，就能把
   reversed从task-complete的129压到47。
2. 直接恢复旧训练范式也不是答案；它用更强的Procedure下游增益换来了更低
   absolute、较差breadth和仍然不足的wrong-video语义特异性。

因此当前核心矛盾更准确地位于多任务优化与Core/Procedure→compiler融合增益的
接口：task-complete侧偏向较高absolute但弱化时序条件，old recipe侧过度放大
时序路径并发生剧烈参数/任务能力迁移。尚不能把根因单独归给训练或架构。

上述“对照完成后停下讨论”的旧边界已被owner后续v7授权覆盖。全部v6
checkpoint、评测cache、rows、queue、logs和结果仍保留。v7通过absolute与机制
门前不启动one-shot或RL。下文2–22节保留历史背景，23节是最新恢复入口。

v6 fresh task-complete 正式 run 已在 GPU4–7 从 macro0 完整训练到 macro400，
训练和评测进程均已自然退出：

```text
output:
  /data/ymdai/outputs/ember/
  pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728

log:
  /data/ymdai/logs/ember/
  pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728.log

canonical run-contract SHA256:
  e0d0cf703b596e73552f4150f5abed9f9726a80e5af214095baca33719bdd6a3
```

macro200→400 是同一 root、同一合同和完整 RNG/cursor 的 exact-resume。
`metrics.jsonl` 连续 `1..400` 共400行；每25 macro均有 checkpoint。
run summary 为：

```text
completed optimizer updates       400
global video conditions         9,600
global action queries         192,000
second-segment wall          3,903.024 s
```

24 tasks 各访问400个视频条件，均覆盖全部50个 teacher videos 和50个 action
episodes。macro225..400 checkpoint 的schema、合同、cursor、计数和全文件
manifest均已核验；macro400 Writer SHA256 为
`b4873358...c5742`。owner明确不要求正式run计算整批HDF5内容SHA；不得补做。

训练loss仍缓慢下降，25-step窗口在201..400依次为：

```text
.100447 .101343 .101482 .100144 .099740 .099170 .100669 .098620
```

online functional validation在macro200/400为`.137535/.136083`，只能作数值
监控。Procedure进入compiler的zero-init modulation norm从macro200的
`1.0244`继续增长到250/300/350/400的
`1.1904/1.3206/1.4426/1.5651`，所以该路径没有被冻结。

## 2. 完整 correct400 曲线

macro50..400 的paired、每task 50 videos无放回correct400为：

```text
macro          50   100   150   200   250   300   350   400
successes     114    77   120   129   117   118   125   125
tasks > 0       6     7     7     5     7     5     6     7
```

完整分析artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_full_correct_curve_paired_144c30e_20260728.json
SHA256: 7789350d0cd7d9e36c36b303c112279c4ba3d17a2448380f60e9162e20b472e1
```

八点使用相同400个task/state keys、env/policy/noise seeds和teacher-video
assignment；每task的50 videos均为无放回双射。macro200仍是observed-best。
它相对250/300/350/400的paired lost/gained分别为
`39/27`、`34/23`、`30/26`、`40/36`，exact p分别
`.175/.185/.689/.731`。后四点并非统计显著下降，但成功集合发生了很大交换，
不是多task共同提高。

逐task的200→400变化最能说明问题：

```text
task              macro200  macro250  macro300  macro350  macro400
Long-1                  22        18        17        11        14
Long-2                   0         1         0         0         0
Goal-3                   0         0         0         0         2
Goal-6                  24        32        32        32        40
Object-1                50        42        49        48        45
Object-3                31        19        19        31        21
Spatial-1                2         3         0         2         1
Spatial-3                0         2         1         1         2
```

Goal-6上涨`+16`的同时，Long-1/Object-3分别下降`-8/-10`。第二小时提高了
breadth，却没有提高aggregate；因此不再继续同一full-24 recipe，也不补
every-25稠密评测。

## 3. observed-best的五臂与内部传递

observed-best仍是macro200，所以不对非best macro400重复五臂。macro200的
full400为：

```text
correct / same-task-other / wrong / shuffled / reversed
129     / 131             / 108   / 111      / 105
```

relative-to-correct paired switches和exact p为：

```text
same-task-other  22/24  p=.8830
wrong            42/21  p=.0111
shuffled         36/18  p=.0198
reversed         37/13  p=.00094
```

方向门通过，但correct对后三臂margin仅`21/18/24`，明显弱于v5.2的
`58/50/49`，且主要由少数tasks贡献。

16-reference内部检查的median relative-L2为：

```text
condition          Core  ActionProbe Transition Procedure eff.LoRA action
same-task-other   .0664      .1593      1.2541    .0365    .0856  .0139
wrong             .2897      .4788      1.3395    .1345    .3233  .0501
shuffled          .0000      .3640      2.1241    .0888    .2590  .0282
reversed          .0044      .4323      1.3767    .1167    .2436  .0392
```

fixed-Core Procedure-only几乎复现shuffled/reversed的effective-LoRA与action
差异，Core-only接近零；顺序信号确实来自
visual-transition→Procedure，而不是Semantic Core或静态旁路。相比v5.2，
v6 Procedure差异更强，但LoRA/action差异更弱。

## 4. 当前科学判断

side-chat提出的三个解释经v6第二段与两套SFT对照后更新为：

1. “只是optimizer updates不足”已明显降权。额外200次update、同量数据和
   持续增大的compiler modulation没有带来absolute共同上涨。
2. “full-24 task平均过强”也已降权。global-8 SFT保持global batch、LR和
   task/sample clock，只把24-task平均拆成三个8-task updates，结果仍只有
   `105/400`且明显漂移，没有超过full-24的109。
3. v6拓扑仍不是第一嫌疑。Semantic Set、Visual Transition和Causal
   Procedure的职责分离均被内部反事实支持；若存在架构瓶颈，更可能局限在
   Procedure→compiler增益，而非上游时序表示。

因此不推翻v6，但也不再把cyclic-8 Writer作为默认下一正式run。现有v6八点
逐task envelope为156、单点best为129，说明先检验“同一轨迹的互补能力能否在
参数空间合成”比再烧一小时同构subset recipe更便宜、更直接。若参数平均失败，
下一fresh实验优先改LR/稳定化优化；只有优化路径仍不能把Procedure差异传到
LoRA/action时，才改Procedure→compiler。

但owner随后明确把 corrected mixed-task Source-SFT 提前为紧邻动作：先得到
可信强baseline，避免Writer反复改动后才发现没有超过普通action-SFT。SFT结果
同时提供一个重要诊断：它没有Writer的异构per-video LoRA或六个micro-round，
每个mixed physical batch只做一次普通forward/backward/sync/clip/AdamW。
因此其full-24样本覆盖不能直接等同于v6 task-complete的训练机制。

当前focused Goal的两个absolute硬条件已经统一为：

```text
EMBER correct400 >= 150
EMBER correct400 >= corrected Source-SFT observed-best + 30
```

即最终门是`max(150, SFT_best+30)`；还必须同时满足same≈correct、
correct显著优于wrong/shuffled/reversed、多个tasks共同贡献，并用独立
RNG/video permutation复测。`+30`只是最低研究里程碑，不是达到后强制停止。

## 5. corrected mixed-task Source-SFT已实现并完成profile

canonical实现已fast-forward到main：

```text
branch provenance: codex/source-sft-mixed
implementation commits: 4c527dd / 55ccbcc
profile seal commit: effbd4b
config: configs/pi05_source_sft_rank128_mixed_v2.json
```

它从同一frozen source base fresh训练一套shared rank-128 LoRA。每个rank的
physical batch都含24 tasks等量样本；B144时为每task每rank 6条、全球每task
24条。采样为确定性的
`uniform task -> no-replacement episode cycle -> no-replacement chunk cycle`，
跨rank row不重复且sample identity只由step/rank/task/offset决定。每个physical
batch只做一次普通同步update，没有gradient accumulation、`no_sync`或
task-complete micro-round。targeted tests为`21 passed`。

GPU4–7 B144 profile root：

```text
/data/ymdai/outputs/ember/
pi05_source_sft_rank128_mixed_profile_r4_b144_55ccbcc_s3_20260728
```

fresh step1后从完整checkpoint exact-resume到step3；metrics连续`1,2,3`，
共1,728 unique query rows。每步24 tasks等量，到step3每task的50 episodes
均已覆盖。后两步max-rank wall为`16.684/15.847s`，吞吐
`34.524/36.346 queries/s`；峰值allocated/reserved为
`60,690,811,904/74,065,117,184 bytes`。loss/gradient均finite，source policy
冻结，只有`10,297,344`个rank-128 LoRA参数可训练；B120 fallback未触发。

正式首段由实测吞吐封存为fresh step0→225、每25步checkpoint，约61分钟训练
body；四rank冷加载单独报告。若首段observed-best在右端或峰值明显不稳定，
默认exact-resume到step450；不机械固定400，也不预先授权更后段。

## 6. 清理与空间

四个macro250/300/350/400评测均已完整保留results、rows、queue、logs、
contracts和paired artifact；随后只删除可确定重建的per-rollout LoRA cache，
释放`4,254,614,115` bytes。精确清单：

```text
/data/ymdai/outputs/ember/
cache_cleanup_v6_resume_correct_lora_20260728.json
```

此前已删除的历史完成评测cache清单继续保留在同一outputs root。B144 profile
结束后`/data/ymdai`为`315,415,540,099` bytes，低于500GB operator hard
cap；任何新formal run前仍需重测现场与预计峰值。

## 7. 紧邻动作与恢复命令

当前紧邻动作是：

1. 建立session-local Goal，成功条件为single-checkpoint correct400至少150且
   通过完整视频因果门；
2. 原位实现`docs/action_forecast_writer_v7_design.md`，删除v6 executable
   schema/config，不保留并行Writer或兼容分支；
3. 完成最短shape/mask/identity/freeze/gradient/resume检查；
4. 做live GPU4–7与storage preflight，再用最长真实视频确定训练吞吐与显存
   上限；
5. 按真实证据小步训练、paired correct400筛best，并只对best做五臂和内部传递；
   每次fresh实验只回答一个已定位瓶颈。

corrected Source-SFT best已固定为109，所以focused AS硬门为
`max(150,109+30)=150`。达到absolute和视频因果双门后才做matched action
one-shot与独立cold-start RL。现有及后续checkpoint全部保留。

只查询GPU4–7：

```bash
nvidia-smi -i 4,5,6,7 \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
du -sh /data/ymdai
```

## 8. corrected Source-SFT正式launch合同

本合同只用于fresh首段；同一scientific contract的step225→450 resume复用它。

```text
workspace: /data/ymdai/projects/EMBER
branch: main
Git gate: launch时HEAD == origin/main且worktree clean
devices: physical GPU4,5,6,7
topology: 4-rank DDP, one CUDA process/rank, NUMA node1
config: configs/pi05_source_sft_rank128_mixed_v2.json
source policy:
  /data/ymdai/outputs/ember/
  pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000
data:
  /data/ymdai/ember_data/LIBERO-datasets/
  f13aa24a3da8c43c7225569f28c562979fa0e35a
output:
  /data/ymdai/outputs/ember/
  pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728
log:
  /data/ymdai/logs/ember/
  pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728.log
```

首段为225 optimizer updates、每rank B144、global576 queries/update，总计
129,600 action queries；24 tasks等权，每task 5,400 samples。每25步保存，
预计9个checkpoint约0.57GB，连同合同、metrics和online validation首段新增
低于1GB；完整2400-step探索包络若全部保留仍低于2GB。模型和数据只复用现有
路径，不复制，无额外临时root。

正式命令：

```bash
numactl --cpunodebind=1 --membind=1 env \
  PYTHONPATH=/data/ymdai/projects/EMBER/src \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 \
  OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  /data/ymdai/projects/EMBER/.venv/bin/torchrun \
  --standalone --nproc-per-node=4 scripts/train_source_sft.py \
  --config configs/pi05_source_sft_rank128_mixed_v2.json \
  --stage development --mode formal \
  --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
  --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir /data/ymdai/outputs/ember/pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728 \
  --stop-after-step 225 --num-workers 2 --log-every 10 --skip-data-sha
```

`--skip-data-sha`只取消owner明确拒绝的runtime全量HDF5内容hash；sealed
manifest、精确file size、HDF5 schema、source/tokenizer/LoRA合同仍fail closed。
fresh output不得已存在或覆盖。首段后以online validation loss筛少量候选，再
用同一fixed validation closed-loop panel选observed-best；只有右端/峰值不稳
才从完整step225 checkpoint exact-resume到450。任何batch、sampler、LR、
数据、参数化或信息墙变化都必须fresh root。

## 9. corrected Source-SFT首段完成与当前评测

fresh step0→225已自然完成。`metrics.jsonl`恰好225行且step `1..225`连续，
全部loss/gradient/LR/吞吐字段finite；run summary记录129,600 global action
queries、24 train tasks、每task 5,400 samples、`10,297,344`个trainable
rank-128 LoRA参数和`3,639.436s`训练body wall。step25..225每25步共9个
checkpoint均存在；step225包含LoRA、optimizer/scheduler、四rank sampler/RNG
state和manifest内逐文件SHA256。个人空间现场为约295GB，完整首段root仅551MB，
按owner最终澄清保留原checkpoint cadence，不删除这些checkpoint。

online task-balanced functional loss为：

```text
step        25       50       75      100      125      150      175      200      225
loss   .139748  .134216  .134064  .132966  .133862  .134068  .134527  .135724  .135276
```

step100是online observed-best，但online loss只用于筛候选，不替代closed-loop
success。step50/100/175/225分别在物理GPU4/5/6/7完成相同fixed validation
8 tasks×50 states，结果为：

```text
step          50    100    175    225
successes     60     75     77     56
tasks > 0      5      8      6      6
```

四点每个checkpoint只加载一次、每卡6个persistent policy workers；冷加载约
149–154秒，rollout稳态约72–73GB/卡并接近100% UTL。每点均为400 unique
rows、36/36 shards、6/6 workers正常退出、所有attempt为1且零错误；
task/state/env/policy seeds及共同长度noise prefix完全配对。每个queue首批
6个shard均为horizon-520 long tasks，global long-first通过。

step175相对step100为`39 gained / 37 lost,p=.9088`，aggregate只高2且成功
集合大幅交换；相对step225为`40/19,p=.00864`。逐task step100→175主要是
Long-1 `4→14`、Object-3 `2→6`，同时Goal-6 `29→25`、两个Spatial
`1/6→0/0`。因此首段不是稳定封闭峰：step100/175同档，右端又发生显著下降，
且历史SFT已有下降后恢复。按owner的“峰值不稳定再训一小时”和sealed合同，
下一动作是从完整step225 exact-resume到450，保持B144、全部24 tasks混合、
LR/sampler/参数化与每25步checkpoint不变。暂不补密集rollout，也不删已有
checkpoint。

## 10. full-24 corrected Source-SFT上限已封存

同一正式root已从step225 exact-resume到450。`metrics.jsonl`连续`1..450`，
所有loss/gradient finite；step25..450每25步共18个checkpoint全部保留，
总占用约1.1GB。第二段online loss在step250..450为：

```text
.133605 .133035 .135426 .133542 .135065 .134222 .133433 .132800 .134391
```

fixed paired correct400 dense曲线为：

```text
step          50  100  175  225  275  300  325  350  375  400  425  450
successes     60   75   77   56   77   57   87   71   98  109  107   74
tasks > 0      5    8    6    6    6    6    8    8    7    6    7    7
```

artifact：

```text
/data/ymdai/outputs/ember/
pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728/
paired_correct400_step0050_0450_dense.json
SHA256 5a781a50344b72085ac154b1602a6842cb9bcb6b44a0a957f3da544e5e8791c4
```

12个面板均为400 unique rows、36/36 shards、6/6 workers exit0、paired
state/env/policy/noise与global long-first。step400和425为`24/22` switches、
`p=.883`，属于同一平台；425→450为`45 lost/12 gained,p=1.31e-5`，
400→450为`50/15,p=1.57e-5`，形成明确post-best下降。故full-24
observed-best取step400=`109/400`并停止，不续训到600。

step400相对旧四卡rank-pure step700为`109 vs108`、paired
`37/36,p=1.0`；相对旧八卡历史best为`109 vs122`、`25/38,p=.1299`；
相对v6 macro200为`109 vs129`、`32/52,p=.0375`。该结果说明仅把每个physical
batch改成24-task等权并未提高SFT上限，也复现了明显task能力漂移。

## 11. global-8 cyclic mixed Source-SFT已完成profile

隔离分支`codex/source-sft-mixed8`用唯一canonical sampler替换full-24 sampler：

```text
4 ranks × 2 tasks/rank = 8 disjoint tasks/update
3 updates = 1 complete 24-task cycle
B144 = 72 samples/task/rank-selected update
global576 queries/update
每个task跨完整cycle平均24 samples/update
```

因此它与full-24保持相同global query batch和task/sample clock，只把一次
24-task平均拆为3次8-task AdamW update。task permutation每cycle刷新，
物理rank轮换，episode/chunk无放回；checkpoint和stop只允许落在完整3-step
cycle边界。配置为`configs/pi05_source_sft_rank128_mixed8_v3.json`。

GPU4–7 B144 profile root：

```text
/data/ymdai/outputs/ember/
pi05_source_sft_rank128_mixed8_profile_r4_b144_c25cd5d_s6_20260728
```

fresh step0→3后exact-resume到6；两轮cycle的每轮24 tasks均精确覆盖一次，
全部3,456 query identities唯一。稳态step wall为
`15.833/15.882/15.840/15.883s`，吞吐
`36.381/36.267/36.363/36.266 queries/s`；峰值allocated/reserved为
`60,690,811,904/74,065,117,184 bytes`。loss/gradient均finite，无OOM、
NCCL/CUDA错误或allocator增长，B128 fallback未触发。

正式首段封存为fresh step0→240、每30步checkpoint；约64分钟training body，
筛step60/120/180/240。默认第二段exact-resume到480并筛
300/360/420/480，除非首段已经出现可信的多task绝对下降。全部checkpoint保留；
不做runtime全量HDF5 SHA。正式launch前仍须clean/pushed main、GPU4–7 live
preflight和存储复核。

## 12. global-8 Source-SFT正式结果与停止判断

正式root：

```text
/data/ymdai/outputs/ember/
pi05_source_sft_rank128_mixed8_dev_r4_b144_seed7_s2400_85bfe8e_20260728
```

fresh 0→240后从step240 exact-resume到480；`metrics.jsonl`连续1..480，
loss/gradient全部finite。累计276,480 action queries，24 tasks各11,520
samples、160 visits且覆盖全部50 action episodes。step30..480共16个
checkpoint全部保留；step480的LoRA、trainer和四rank state逐文件SHA256复算
均与manifest一致。两段进程启动至封存合计约`11.32 GPU-hours`，唯一trainable
对象为`10,297,344`参数shared rank-128 LoRA。

八点paired correct400：

```text
step          60  120  180  240  300  360  420  480
successes     63   83   85   98   90   62   90  105
tasks > 0      4    8    6    6    8    7    4    5
```

artifact：

```text
/data/ymdai/outputs/ember/
pi05_source_sft_rank128_mixed8_dev_r4_b144_seed7_s2400_85bfe8e_20260728/
paired_correct400_step0060_0480.json
SHA256 9446b471016dfb99abb18f107de047163f3245cc9d009456673fe42115c8d2be
```

八点均400 rows、36/36 shards、6/6 workers exit0；task/state/env/policy/noise
完全paired，global long-first通过。step480相对420为
`21 lost/36 gained,p=.0627`，相对240为`30/37,p=.464`；八点逐task
envelope为126而best只有105。相对full-24 step400=`109`，step480为
`28 gained/32 lost,p=.699`，只是能力重分配。两个Spatial tasks在两种recipe
best中均为0。因此global-8不续到600，corrected Source-SFT development best
封存为full-24 step400=`109/400`。

隔离候选`codex/v6-cyclic8-training@eb7943b`已通过全仓190 tests，但没有合并、
push或启动。该实现保留作provenance；SFT直接对照不支持把它作为默认下一run。

## 13. v6 checkpoint-average screen与机制结果

实现worktree/branch：

```text
/data/ymdai/.codex/worktrees/EMBER-v6-average-20260729
codex/v6-checkpoint-average
```

screen合同：

```text
configs/pi05_as_writer_v6_checkpoint_average_screen_v1.json
```

四个候选已在原v6 run下生成：

```text
derived_checkpoints/uniform_m150_m200
derived_checkpoints/uniform_m200_m400
derived_checkpoints/uniform_m150_m200_m350_m400
derived_checkpoints/uniform_m150_m200_m250_m300_m350_m400
```

固定映射为前述顺序对应物理GPU4/5/6/7。每份只含单套
`writer.safetensors`和manifest；源raw checkpoint及其它checkpoint全部保留。
派生算法为float32逐tensor均匀算术平均后cast回原dtype。独立复算四份均为
600 tensors、12,064,064 state elements、0 mismatch、全部finite；523个训练
tensor发生平均，77个固定buffer保持一致。formal evaluation adapter检查已
通过，cursor axis明确记为`max_source_optimizer_step`；训练resume/warm-start
仍只接受原始`checkpoints/step_*`。

验证：

```text
focused Writer tests  65 passed
full repository       190 passed
git diff --check      pass
```

评测合同固定为每候选correct400、seed7、teacher video无放回、6 Writer
generators、generation batch16、6 persistent policy workers和global
long-first。四候选结果为：

```text
sources                         correct400  tasks>0
150,200                               129       7
200,400                               140       6
150,200,350,400                       144       7
150,200,250,300,350,400               145       7
```

winner相对raw macro200=`129`为`37 gained/21 lost,p=.04794`。screen artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_checkpoint_average_correct400_paired_ea99f65_20260729.json
SHA256 09d4399662de821a1de0d6f38903eeba60a571fee2594c02fe6a445013dfb8ac
```

winner五臂为`145/134/128/119/122`。correct相对wrong/shuffled/reversed为
`38/21,p=.03634`、`44/18,p=.001299`、`45/22,p=.006741`；正向task数
为5/6/5。same为`30/19,p=.1524`，统计同档但aggregate少11；保守
`|delta|<=10`标志因此差1未过。五臂artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_checkpoint_average_late6_specificity400_paired_ea99f65_20260729.json
SHA256 9244b8db004f4155f9ee254bbddbaf013ee033640b6d9974c2b98cd283579d8b
```

16-reference内部检查的wrong/shuffled/reversed effective-LoRA为
`.3591/.2689/.2923`，action为`.0568/.0576/.0434`。fixed-Core
Procedure-only几乎复现，Core-only近零。summary：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_avg_internal_specificity_late6_refs2_ea99f65_20260729/
summary.json
SHA256 7596fbd5cd03232d99667b5eb5b500995e5b1cbf6d1c01b97bb2c8a8628d169d
```

## 14. v6 fast-decay400已完成

新config：

```text
configs/pi05_as_writer_language_axial_v6_decay400_v1.json
```

唯一科学变化：

```text
cosine decay_steps  2000 -> 400
peak_lr             3e-4 unchanged
warmup              17 unchanged
floor_lr            1e-5 unchanged
```

其余authorities、information wall、Writer、data、task-complete B20、AdamW、
seed与checkpoint cadence逐对象相同。该run从step0 fresh，首段0→200后从
完整macro200 exact-resume到400；未从六点average或任一旧Writer checkpoint
初始化。输出：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729
```

`metrics.jsonl`连续1..400且全部finite，累计192,000 action queries和9,600
video conditions；每25 macro的16个checkpoint全部保留。八点correct400为：

```text
macro             50  100  150  200  250  300  350  400
fast-decay       106   64  111  133  132  117  138  143
original-v6      114   77  120  129  117  118  125  125
```

macro400相对原同点为`46 gained/28 lost,p=.04739`。350→400为
`25 gained/20 lost,p=.5515`；raw右端best仍在400，但后段已近冻结。
fullcurve artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_decay400_correct400_fullcurve_paired_4efa737_20260729.json
SHA256 99b04bf1cf72ad2385119638ca8020c5caf24e2c33075d758ee7f38dcc253d03
```

375→400 update L2为`.12647`、cosine`.1189`；完整dynamics SHA256为
`804689cac6e108357e6977fb1f263cdc7a13611be46eb6bd3e477d6cae805f32`。

## 15. fast-decay checkpoint-average screen已完成

封存config：

```text
configs/pi05_as_writer_v6_decay400_checkpoint_average_screen_v1.json
SHA256 07d115811cf6042d5d0246e9f91c304aed3e5289b53d898d17af0330526951f5
```

四候选/GPU固定为：

```text
GPU4  uniform_m350_m400
GPU5  uniform_m200_m350_m400
GPU6  uniform_m200_m250_m350_m400
GPU7  uniform_m150_m200_m250_m300_m350_m400
```

四份derived checkpoint的600个tensors均经独立float32均值复算，
`max_abs_error=0`；manifest、source provenance和inference-only authority
全部通过。GPU4–7各跑一组correct400，teacher video按task无放回、
6 generators、batch16、6 persistent workers和global long-first。结果：

```text
candidate                                      correct400  tasks>0
uniform_m350_m400                                     139        6
uniform_m200_m350_m400                                135        6
uniform_m200_m250_m350_m400                           129        6
uniform_m150_m200_m250_m300_m350_m400                 130        7
raw macro400                                          143        6
```

相对raw的paired`gained/lost,p`依次为
`18/22,.6358`、`21/29,.3222`、`13/27,.03848`、
`18/31,.08543`。每组均为400 unique rows、36/36 attempt1 shards、
6/6 workers return0、无adopt；每task teacher demos严格`0..49`双射，
四组与raw的state、env/policy seed、noise prefix和video assignment完全相同。
前12 shards均为horizon520，之后无long shard。

完整paired artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_decay400_checkpoint_average_correct400_paired_7c3879c_20260729.json
file SHA256 ac6e15450cf99800cc15202fa90e62f38caa735b0da956c51125cef1cab61d30
canonical payload SHA256 a9ffd347af8504cd46aad5f90fc732c6e6122a4ec3f818ae2e4ef666a39adfdb
```

四个average的episode-level union为174；把raw也加入union为180，说明能力
模式仍高度互补，但单套线性均值没有把它们合成。raw macro400继续是当前
fast-decay observed-best。因owner要求本步后暂停，尚未启动下一分析或实验。

## 16. fast-decay续训与macro400机制证据

同一正式root从完整macro400 exact-resume到600：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729
```

`metrics.jsonl`连续1..600，run summary记录600 optimizer updates、
288,000 action queries、14,400 one-video conditions，test/validation action
reads均为0。新增macro450/500/550/600的correct400为
`131/130/132/126`，四点均低于macro400=`143`；400→600为
`31 lost/14 gained,p=.01609`。完整12点artifact：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_decay400_correct400_m50_m600_paired_4efa737_20260729.json
SHA256 4a2b6121235b08d8b7ce9c1bf72ab93bf6592ffe78c69c0dcbcfe76be5208eef
```

macro400单checkpoint五臂：

```text
correct / same / wrong / shuffled / reversed
143     / 135  / 125   / 128      / 129
```

相对correct的`control-only/correct-only,p`分别为：

```text
same       16/24  p=.26819
wrong      20/38  p=.02475
shuffled   22/37  p=.06744
reversed   23/37  p=.09246
```

wrong方向由4个tasks贡献且显著；shuffled/reversed虽方向正确但未过显著门。
16-reference内部median relative-L2为：

```text
condition   Core  Transition Procedure  ProcSlots  Fused  eff.LoRA action
same       .0748      1.2550    .0363      .4199  .1002    .0627  .0123
wrong      .3290      1.3604    .1113     1.1350  .3987    .3520  .0550
shuffled   .0000      2.1492    .0851     1.1372  .3658    .2367  .0421
reversed   .0044      1.3750    .0952     1.2295  .3331    .2138  .0466
```

fixed-Core Procedure-only几乎复现shuffled/reversed的effective-LoRA/action，
Core-only近零；所以task-complete的控制臂高分不是上游没看到顺序，而是顺序
差异在下游只形成中等增益。

五臂artifact SHA256为
`b299750377461061d13bb3dbb5f9ba38dacebd02b4117dbf1caf52a16b80f488`；
内部summary/run-contract SHA256为
`a91b91a9...ec315/f13706a5...bb6b4`。

## 17. v6旧训练范式对照

canonical config固定为：

```text
configs/pi05_as_writer_language_axial_v6_old_recipe_v1.json
```

该对照只改训练更新范式，不改v6 Writer拓扑、数据、信息墙、public rank-16
LoRA空间或policy。最长105-frame视频的fixed-B20 profile连续3步finite，
稳态约`20.09 queries/s`、`904 updates/hour`，峰值allocated/reserved约
`76.94/83.72GB`。`B21`从未运行，正式入口固定B20且对更大batch fail closed。

正式root与成功log：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v6_oldrecipe_rankrotating_dev_r4_b20_seed7_s12000_bad9a96_20260729

/data/ymdai/logs/ember/
pi05_as_writer_v6_oldrecipe_rankrotating_dev_r4_b20_seed7_s12000_bad9a96_20260729_launchretry1.log
```

第一次launcher在创建有效step/output前因CLI checkpoint列表末端900与sealed
`total_steps=12000`不一致而fail closed；保留失败log。仅移除错误的CLI覆盖后
用同一科学合同重启，900 steps自然完成。run summary为900 updates、
72,000 action queries、3,600 video conditions、训练body
`3,626.731s`；每100步checkpoint均保留。每6 updates精确覆盖24 tasks，
step300时每task已访问全部50 videos一次。

step100/500/700/900的correct400：

```text
step       100  500  700  900
successes   98  121   76   95
tasks>0      5    5    4    6
```

500→700为`50 lost/5 gained,p=2.14e-10`，500→900为
`43/17,p=.00107`，不是400-rollout噪声。参数诊断也显示500→700整体
update L2=`8.013`、相对L2=`.0670`、与前一段cosine仅`.190`；
700→900 cosine仅`.081`，factor heads相对位移约`.144`，说明持续训练仍在
大幅旋转下游解。

step500五臂：

```text
correct / same / wrong / shuffled / reversed
121     / 122  / 111   / 84       / 47
```

same为`23/24,p=1`；wrong为`34/24,p=.237`，只有2个正向tasks且最大单task
占`.867`，语义门失败；shuffled为`58/21,p=3.76e-5`，reversed为
`83/9,p=3.92e-16`，顺序门强通过。内部median relative-L2：

```text
condition   Core  Transition Procedure  ProcSlots  Fused  eff.LoRA action
same       .0496      1.2320    .0377      .3870  .1591    .0794  .0104
wrong      .2101      1.3796    .1597     1.1546  .8390    .5612  .1227
shuffled   .0000      2.2617    .1084     1.0069  .6255    .3635  .0663
reversed   .0034      1.3926    .1664     1.5066 1.0053    .6064  .1485
```

fixed-Core Procedure-only复现shuffled/reversed的
`.3635/.0665`和`.6066/.1487`，Core-only近零。这直接证明训练范式能够改变
Procedure→compiler的有效增益，而不需要改Visual Transition或Semantic Core。

主要artifact：

```text
correct curve:
  /data/ymdai/outputs/ember/
  pi05_as_writer_v6_oldrecipe_correct400_steps0100_0500_0700_0900_paired_bad9a96_20260729.json
  SHA256 712e51dda7371edb6fc09f57973ca6e67ad9d47fe52b95ede2becca0ba2297b6

five-arm:
  /data/ymdai/outputs/ember/
  pi05_as_writer_v6_oldrecipe_single_checkpoint_step0500_specificity400_paired_bad9a96_20260729.json
  SHA256 c0127e652f2f039f4f1982ac3b6f143ce84b46c8424c26ab81a14554a9ebb818

checkpoint dynamics:
  /data/ymdai/outputs/ember/
  pi05_as_writer_v6_oldrecipe_checkpoint_dynamics_s0100_s0500_s0700_s0900_bad9a96_20260729.json
  SHA256 19a6361d5d4c68ed04ab5d431dede9e1e7546ab690c94e25d90b994826361f48

internal:
  /data/ymdai/outputs/ember/
  pi05_as_writer_v6_oldrecipe_single_checkpoint_step0500_internal_specificity_refs2_bad9a96_20260729/summary.json
  SHA256 bdd3145f572fffd5f29e823a354af8e220405930a0f912f9fdfe73ea45ae9963
```

## 18. v7第一性原理设计与当前切换点

owner要求先记录、再建立Goal。已记录的核心决策为：

```text
同一次multimodal prefix
→ task-aligned semantic trajectory G_f
→ frame-mean Semantic Core
+ 8-token Action Expert probes
   × forward semantic change G_(f+1)-G_f
→ joint softmax over all 8×L action–effect pairs
→ high-level action–effect events
→ 3-layer causal Procedure
→ Core-conditioned query / Procedure-only content compiler
→ complete rank-16 LoRA
```

Action Expert teacher suffix是在一次forward中由1个probe扩为8个稀疏原生
positions，不是8次forward；execution policy的50-action chunk不变。v7删除
Text-only Gemma、Core-primary AdaLN和Core到factor的content路径，不加入
null token、Action-only residual、额外视觉encoder或order loss。设计参数
`10,312,192`，真实实现枚举已逐项吻合。

8个Action probes不先mean，也不以8个tokens进入Procedure；它们与`L`个
forward semantic change tokens的全部`8×L` pairs在每个head内joint softmax，
以语义变化为value、Action作zero-init逐通道调制，直接产生一个event。
`D_f=0`时event严格为零，Core与Procedure直到compiler才首次相遇。

设计文档落盘时现场为：main与origin/main均
`f920f4a0e13366864fee3334eb60beb56c4edf6d`、原工作树clean；GPU4–7为0MiB，
GPU0–3有其他用户进程且不得触碰；个人空间约338GB、低于500GB cap；无EMBER
训练/评测进程。上述快照不是launch授权证据，真正GPU启动前必须重新做live
preflight。

session-local Goal已建立。canonical v7源码/config已原位实现，v6
schema/checkpoint刻意不兼容；全仓192 tests通过，architecture guard无hard
violation或parallel family。

GPU4–7真实profile结果：

```text
B32  首个functional policy forward OOM（仍需968MiB）
B24  首个functional policy forward OOM（仍需726MiB）
B20  3/3 macros finite，含105-frame最长视频
      step wall 19.234 / 17.492 / 17.447s
      steady 27.477 queries/s，206.075 macros/hour
      max allocated/reserved 77,020,274,176 / 83,647,004,672 bytes
```

B20 root为
`/data/ymdai/outputs/ember/pi05_as_writer_v7_profile_b20_jointae_r1_20260729`。
真实step1→3 resume root为
`/data/ymdai/outputs/ember/pi05_as_writer_v7_resume_smoke_b20_jointae_r1_20260729`；
checkpoint1恢复后未改写，三步task/video/query/LR/cursor与连续run一致，
独立连续run的最大mean-loss差`2.33e-5`。joint Action–Effect binder
`262,656/262,656`参数在step1→3全部变化，L2位移`0.08944`。

首轮正式配置已改回正式teacher seed `20260722`，选择B20与v6已验证较优的
fast cosine decay400；fresh 0→200 macro、每25 checkpoint，共96,000 action
queries和4,800 one-video conditions。实现/profile seal commit为
`ca7db57d0c2d1ec2e7032a44b58238b6de35b1f4`，已push至`origin/main`。正式root
预声明为
`/data/ymdai/outputs/ember/pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729`，
log为同名文件置于`/data/ymdai/logs/ember/`，tmux为
`ember-v7-formal-200`。该run随后已经完成macro0→400；以下第19节覆盖其结果和
当前切换点。

## 19. v7结果、根因与v8当前切换点

v7正式root为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729
```

macro0→200和exact-resume200→400均正常完成，metrics连续1..400且finite。
correct400为：

```text
macro       50  100  150  200  250  300  350  400
successes   82  106  114  120  101  114  115  106
```

macro200五臂为`correct/same/wrong/shuffled/reversed=120/112/91/100/69`。
correct相对wrong/shuffled/reversed的paired净差为`+29/+20/+51`；v7具备强于
v6的方向特异性，但absolute显著低于v6 best143和focused门150。

refs1全8个validation tasks的内部检查显示：

```text
pair logit std                         ≈ 0.058
pair entropy / theoretical uniform     ≈ 0.9996
effective Action probes                ≈ 7.998 / 8

fixed Procedure, varying Core LoRA L2  ≈ 0.001–0.002
fixed Core, varying Procedure LoRA L2  ≈ full observed difference
```

所以v7的joint `8×L` softmax实际上均匀平均所有Action–Effect pairs；Core只
通过query进入compiler也几乎不起作用，Writer实际退化为Procedure-only。
macro200→400两项内部指标未改善且absolute下降，因此不再续训v7。

当前authority为
`docs/action_forecast_writer_v8_design.md`。v8只修复两个已证实缺口：

```text
each of 8 Action anchors
→ independently attends over L effects
→ 8 bound action–effect tokens
→ Procedure-only EventRead
→ one event / frame interval

Core read
→ Core-conditioned Procedure read
→ bounded multiplicative Core gate on Procedure slots
→ content-only post-fusion
```

没有additive Core residual；`D=0→event=0`、`Procedure=0→public LoRA identity`
仍为结构硬约束。真实参数枚举为`10,706,176`：hierarchical binder
`590,848`、Core-gated compiler`1,469,696`。活动config为
`configs/pi05_as_writer_language_axial_v8.json`。B20 profile和exact-resume均
已封存：

```text
profile root:
/data/ymdai/outputs/ember/pi05_as_writer_v8_profile_b20_hierae_r1_20260729

step wall: 19.243 / 17.506 / 17.450s
steady:    27.463 queries/s, 205.974 macros/hour
peak:      77,035,771,904 allocated / 83,655,393,280 reserved bytes

resume root:
/data/ymdai/outputs/ember/pi05_as_writer_v8_resume_smoke_b20_hierae_r1_20260729
```

resume step1未改写；task/video/query/LR/cursor完全一致，最大mean-loss绝对差
`4.7951e-5`。binder`590,848/590,848`、binding/EventRead、Core modulation
`65,792/65,792`及所有主模块在step1→3全部变化。formal状态已封存为B20、
teacher seed`20260722`、fresh0→200、every25 checkpoint。

## 20. v8结果与停止理由

v8 task-complete fast-decay400已完整训练并完成八点paired correct400：

```text
macro       50  100  150  200  250  300  350  400
successes   90  110   82  110   90  125   98  115
```

single-checkpoint observed-best为macro300。其正式五臂为：

```text
correct / same / wrong / shuffled / reversed
125     / 121  / 110   / 110      / 117
```

v8没有超过v6 best143，也没有恢复v5.2的强视频margin。内部检查进一步表明：

```text
fixed Effect, vary Action event relative-L2   ≈ 8–10%
fixed Action, vary Effect event relative-L2   ≈ 147–300%
Effect token attention entropy ratio          ≈ 97.79%
EventRead attention entropy ratio             ≈ 99.67%
effective Action anchors                      ≈ 7.95 / 8
```

所以问题不是新模块没有梯度，而是缺少局部Action–Effect身份：Action Expert
probe是冻结source policy对当前画面的action hypothesis，视觉差分是未知teacher
action造成的effect；信息墙内没有把二者逐interval一一配对的标签。strict
binding/EventRead把已由v5.2/v6证明有效的Action证据压缩掉，输出被Effect主导。
继续同架构训练没有根据，v8停止。

## 21. v10设计、实现与profile封存

v10完整authority为
[`action_forecast_writer_v10_design.md`](action_forecast_writer_v10_design.md)。
其最短拓扑为：

```text
text-only task axis
├─ multimodal + task-queried patch evidence → permutation-invariant Core
└─ 8 sparse Action probes → Action stream
   task-queried patch forward difference → Visual-Effect stream
   A0,V0,A1,V1,... → 2-layer causal Procedure

CoreRead
→ Core-conditioned ProcedureRead
→ Procedure content + beta + tanh(gamma) × full-rank Core
→ factor heads → complete rank-16 LoRA
```

删除joint `8×L` softmax、strict multiplication和EventRead；Action在`D=0`时
仍保留，Effect在`D=0`时严格为零。Procedure值按Action/Effect分别沿时间
中心化。compiler所有调制线性层bias-free，因此
`Procedure=0→public LoRA identity`，Core不能独自生成adapter。

真实module enumeration为`11,627,520`，全仓`192 passed`。

GPU4–7 B20 profile：

```text
root:
/data/ymdai/outputs/ember/
pi05_as_writer_v10_profile_b20_dualstream_r1_20260730

step wall              20.075 / 18.150 / 18.242 s
steady queries/s       26.446 / 26.313
steady macros/hour     198.346 / 197.345
peak allocated         77,008,402,432 bytes
peak reserved          83,653,296,128 bytes
max sampled frames     105
```

三步全部finite、四rank完成，B16未触发。profile run-contract/metrics/summary
file SHA256依次为：

```text
589fd5007e465c9eae9f546ea9eb6c04311e8ea9e188261dcc0c0cd8f7875f73
8fe8cae1bb224ad64e27dcf3aea726b5d0b484f61259e39ec87963ac3872f520
7cfa19871ac1539e9d8a72ff9a75e02d7a24aaed3499d43f44d9e787d8efce72
```

独立resume root从fresh0→1后exact-resume1→3；step1全部文件大小、mtime和
SHA不变，三步task/video/query/LR/cursor与连续run相同，最大mean-loss差
`2.6332e-6`。Text/VL/Action Meta-LoRA、Semantic Core、Action phase、
Visual Effect、Procedure、Procedure reader、gamma/beta modulation、
Core content和factor heads在step1→3均全参数变化。

正式合同已恢复teacher seed`20260722`，B20、task-complete、fast decay400、
fresh0→400、every25 checkpoint。预计192,000 action queries、9,600
one-video conditions、16 checkpoints。正式run必须从包含本节与profile seal
的clean、已push main启动；完成后评测多个single checkpoints，不做融合。

## 22. v10正式结果与当前暂停点

正式root与log：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v10_dualstream_taskcomplete_decay400_dev_r4_b20_seed7_s2400_5fd0a25_20260730

/data/ymdai/logs/ember/
pi05_as_writer_v10_dualstream_taskcomplete_decay400_dev_r4_b20_seed7_s2400_5fd0a25_20260730.log
```

macro0→400自然完成，`metrics.jsonl`连续1..400且finite；teacher-video cycle、
24 tasks/macro、480 queries/macro、唯一一次DDP sync和rank内long-first均
通过。correct400 curve、macro50五臂和内部结论已在本文第1节及v10 design
第12节记录。恢复时直接使用以下证据，不得重复启动：

```text
training audit:
/data/ymdai/outputs/ember/pi05_as_writer_v10_training_audit_macro400_5fd0a25_20260730.json

correct curve:
/data/ymdai/outputs/ember/pi05_as_writer_v10_correct400_curve_paired_5fd0a25_20260730.json

five-arm:
/data/ymdai/outputs/ember/pi05_as_writer_v10_single_checkpoint_macro0050_specificity400_paired_5fd0a25_20260730.json

internal:
/data/ymdai/outputs/ember/pi05_as_writer_v10_single_checkpoint_macro0050_internal_specificity_refs1_5fd0a25_20260730/summary.json
```

对应file SHA256依次为：

```text
6701ec353433203ef89490f0fe6b179eefddaf9e304fd60c9800e204e70ff97f
6e9d97dcf31afdd7d867e4b3f66646db3efa68df552b625f5db2b3ba05012dfd
a2dbcacdfcfbe4ba2a3a9010c4c28664b2ff8ce4530c532560a24e680474be6b
df5b0271991b6ff95360b138dfe72dd7ab5daf34cc54383b92688acab539ec9f
```

当前终止条件来自owner明确指令，不是Goal达标：v10 absolute只有103，session
Goal的150与SFT+30均未完成。保持Goal active但不继续自动执行；先与owner讨论
架构/训练含义。

## 23. Loom负证据与Recenter历史恢复快照

owner后续授权Loom实现和一小时实验。Loom的macro50/100/150/200 paired
correct400为：

```text
79 / 106 / 105 / 112
```

没有达到v5.2/v6同期水平，因此按owner约定未做昂贵五臂rollout，先完成内部
数值分析。Teacher/Policy gap、raw-patch correspondence和teacher confidence
没有获得可靠、可解释的教学锚点；该结果与v7/v8已经证明的Action–Effect局部
不可辨识问题一致。Loom停止，不续训，不在其上调整confidence或gap scale。

该历史阶段的canonical设计为：

[`action_forecast_writer_recenter_design.md`](action_forecast_writer_recenter_design.md)

活动源码删除Loom-only relations/dual Procedure/gap compiler，恢复原生50-token
Action mean，保留v6 Core与patch grounding，使用有界visual-transition residual、
单路causal Procedure和保留幅度的Core-keyed centered compiler。精确参数为
`10,709,248`。活动配置为：

```text
configs/pi05_as_writer_recenter.json
```

该配置当前已封存：

```text
profile_evidence.status = sealed_b20
formal_run.status       = sealed
```

这是Recenter在commit `93c7e32`上的独立证据，不是复制Loom结果。B20连续
3个完整macro均finite且含真实105-frame视频；后两步均值
`25.808 queries/s`、`193.562 macro/hour`，峰值allocated/reserved为
`76,989,294,080/83,644,907,520 bytes`，B16未触发。正式seed独立root完成
fresh0→1→exact-resume1→3，metrics连续`1,2,3`且step1所有checkpoint文件
hash不变；真实step1→3间全部`10,709,248`个trainable参数变化。下一步只在
GPU4–7 live preflight后启动fresh macro0→200。GPU0–3不得查询或使用。

## 24. 当前恢复点：v5.2 × task-complete fast-decay

owner在Target-Spectral负结果和统一LoRA几何复核后，授权先补齐最关键缺失
因果格：原版v5.2拓扑使用v6成熟的新训练范式跑满约两小时。direct Source-SFT
stable rank约`1.517`、v6约`1.0003`；但v6的16个代数坐标能量实际比SFT
更均匀，且q/v分量高度建设性同向，并不存在主导rank负相关。Target-Spectral
因此是错误地把低有效rank当成必须修复的病灶。

活动源码已原位恢复commit `529da6b`对应的精确v5.2模型拓扑，同时保留当前
成熟的full24 cost-balanced long-first sampler、task内mean/24-task等权、
一次DDP sync/AdamW和macro-boundary resume。新config为：

```text
configs/pi05_as_writer_language_axial_v5_2_taskcomplete_decay400_v1.json
```

固定B20、4 ranks×6 tasks、LR `3e-4`、warmup17、cosine decay400到
`1e-5`、every25；fresh macro0→200后默认resume到400。正式前还需GPU4–7
最长105-frame三macro profile和fresh0→1→resume1→3。评测点为150/200/350/
400，winner若为内部点再补±25；不做checkpoint融合。当前实现聚焦测试
`41 passed`，全仓初轮仅剩一处已同步修正的RL错误消息测试。

上述profile与resume门现已完成并seal：main `62598d3`全仓`189 passed`；
seed172三macro包含105-frame、B20 finite，峰值allocated/reserved
`76,967,302,656/83,638,616,064 bytes`。formal seed `20260722`独立
fresh0→1→resume1→3后，metrics/LR/task/video/query cursor连续且所有主模块
可达；正式配置已切换`formal_run.status=sealed`。当前共卡实测约
`142.074 macro/hour`，所以400 macros约169分钟body；仍保持与v6相同更新数，
不为凑两小时偷偷减少科学预算。下一动作是clean commit/push后在GPU4–7
fresh启动macro0→200，再exact-resume到400，并评测150/200/350/400。
