# EMBER focused active session handoff

最后更新：2026-07-28 UTC。

本文只保存当前运行状态、恢复入口和紧邻动作。架构 authority 是
`docs/action_forecast_writer_v6_design.md`；长期科学边界是 `AGENTS.md` 与
`docs/execution_brief.md`；历史实验细节在 `findings.md`、`progress.md` 和 Git
history。任何接手者都必须先只读复核现场，不能按本文快照重复启动进程。

## 1. 当前实时状态

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

side-chat提出的三个解释经第二段后更新为：

1. “只是optimizer updates不足”已明显降权。额外200次update、同量数据和
   持续增大的compiler modulation没有带来absolute共同上涨。
2. full-24 task平均/优化粒度过强成为第一嫌疑。task-complete没有稳定breadth，
   而是让checkpoint在tasks之间交换能力；相邻25-step总更新方向后期也接近
   正交或反向。
3. v6拓扑仍不是第一嫌疑。Semantic Set、Visual Transition和Causal
   Procedure的职责分离均被内部反事实支持；若存在架构瓶颈，更可能局限在
   Procedure→compiler增益，而非上游时序表示。

因此不推翻v6；若后续直接改Writer训练粒度，当前最小可归因候选仍是：

```text
每rank每update处理2 tasks
4 ranks × 2 = global 8 tasks/update
3个optimizer updates组成一个完整24-task cycle
每task仍1 video、B20 action queries、task内均值、全局task等权
```

它保留多task联合更新，不退回one-task/rank；同时将每24-task cycle的AdamW
updates从1提高到3，并把单次共享梯度平均从24 tasks降到8。每个cycle仍精确
覆盖24 tasks，video/action schedule和long-first cost balancing保持。
这是fresh不兼容recipe，不从macro200/400 resume。

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

当前紧邻动作是：把main与`origin/main`推进到包含`effbd4b`及本handoff的clean
状态；做正式launch前的Git/data/storage/GPU4–7检查；随后只在GPU4–7启动
corrected Source-SFT fresh step0→225。首段结束后先用online loss筛候选，再
对少量checkpoint做同一fixed validation correct400，依据observed-best位置和
breadth决定是否resume到450。

Source-SFT observed-best封存后，回到AS-Writer：优先检验global-8/
three-update cycle是否减轻任务漂移；若Procedure仍强而compiler传递弱，再改
Procedure→compiler。达到上述absolute与视频因果双门后才做matched action
one-shot和独立cold-start RL。不得自动进入final-32、test task-local RL或
joint oracle。

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
