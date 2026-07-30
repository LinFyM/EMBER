# EMBER focused active session handoff

最后更新：2026-07-30 UTC。

本文只保存当前运行状态、恢复入口和紧邻动作。Loom设计与推导现只作失败
provenance；当前fresh架构正在按第一性原理封存为Recenter
（Action-anchored、Core-keyed centered Procedure）。长期科学边界是
`AGENTS.md`与`docs/execution_brief.md`；历史实验细节在`findings.md`、
`progress.md`和Git history。任何接手者都必须先只读复核现场，不能按本文
快照重复启动进程。

## 1. 当前实时状态

owner已明确要求在当前session持续自主推进，session-local Goal已建立。唯一
成功目标是提高EMBER single-checkpoint absolute：优先达到correct400
`150`，或至少稳定接近、显著超过corrected Source-SFT和同期旧有效架构。
视频特异性只用于证明性能确实由输入视频学习，不允许以牺牲absolute换取漂亮
控制臂。循环固定为：

```text
第一性原理重构
→ 约一小时fresh训练
→ 四个checkpoint paired correct400
→ 未恢复同期旧架构：不做行为级特异性，只做详细内部/梯度分析后再重构
→ 恢复或超过同期：续第二小时
→ 达150或稳定145+：才做winner四个额外视频特异性臂
```

Loom正式fresh macro0→200和四点correct400已经全部完成：

```text
macro       50  100  150  200
correct400  79  106  105  112
```

训练wall`3,855.28s`、`4,800`视频条件、`96,000`action queries，机械完整且
无OOM。macro200是右端observed-best，但同recipe、同macro200比v6
fast-decay的`133`低21，相近一小时比v5.2的`132`低20；因此没有续第二小时。
四个行为级特异性臂在生成任何cache/result前停止，没有运行环境rollout。

macro200的内部五条件检查已完成且没有环境交互。Loom并非没读视频：Core顺序
合同、Teacher差异传递、compiler replay和zero-Teacher identity都成立。
真正根因是：

- raw patch matcher近uniform、mutual consistency约随机，visual confidence
  约`1e-6`；
- shuffled的大相邻变化反而获得高于correct的teacher confidence和adaptation
  scale；
- Teacher–Policy latent近正交，gap strength在所有条件约`.73`，不是source
  competence gap；
- Teacher支配最终LoRA，Policy/Core影响弱，same-task视频方差仍偏高。

所以Loom整体退役，不修matcher、confidence、scale或latent subtraction。
当前写实现隔离在：

```text
worktree /data/ymdai/.codex/worktrees/EMBER-action-core-20260730
branch   codex/action-core-procedure-20260730
```

新Recenter必须保留v5.2/v6已验证的`Q_text→patch`、Semantic Core、native
Action value、task-grounded transition、causal Procedure、320 routing slots和
full-width factors；恢复native 50-suffix mean Action主干，transition只作
zero-preserving有界修正。compiler改为Core寻址、time-centered Procedure供给
主要value、Core只对非零Procedure作identity-init有界乘性调制，并满足
`Procedure read=0→public LoRA identity`。Loom raw correspondence、Events、
confidence、独立Policy stream和Teacher–Policy gap全部从active code退役。

紧邻动作是完成设计/实现/CPU验证，integrate到clean main并push；随后只在
GPU4–7做B20最长105-frame三macro profile和fresh1→resume3。通过后保持v6
task-complete fast-decay训练合同，fresh macro0→200、每25 checkpoint，并发
评测50/100/150/200的correct400。正式GPU动作尚未启动；除GPU4–7外不得查询或
使用其它GPU。

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
门前不启动one-shot或RL。下文2–17节保留历史背景，18节是最新恢复入口。

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
