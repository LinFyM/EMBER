# EMBER v6 active session handoff

最后更新：2026-07-28 UTC。

本文只保存当前运行状态、恢复入口和紧邻动作。架构 authority 是
`docs/action_forecast_writer_v6_design.md`；长期科学边界是 `AGENTS.md` 与
`docs/execution_brief.md`；历史实验细节在 `findings.md`、`progress.md` 和 Git
history。任何接手者都必须先只读复核现场，不能按本文快照重复启动进程。

## 1. 当前实时状态

v6 fresh task-complete 正式首段已在 GPU4–7 正常完成：

```text
tmux:
  ember-v6-formal-200

output:
  /data/ymdai/outputs/ember/
  pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728

log:
  /data/ymdai/logs/ember/
  pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728.log

launch code:
  main == origin/main == 149badc00c30f8d230401fa24a5dbfa4299b322c

run-contract canonical SHA256:
  e0d0cf703b596e73552f4150f5abed9f9726a80e5af214095baca33719bdd6a3
```

首段自然停在 macro200；tmux 和训练进程均已退出，GPU4–7 已释放。`metrics.jsonl`
为连续 `1..200` 共 200 行，8 个 checkpoint 为 `25..200` 每 25 一点。
run summary 记录 `4,800` 个 one-video conditions、`96,000` 条 action
queries、wall `3,864.599s`。macro200 时 24 tasks 各有 4,000 queries、
200 次 video visits，且各自已覆盖全部 50 action episodes 与 50 teacher
videos。终点 checkpoint 的 Writer、trainer 与四 rank state 已逐文件重算
SHA256 并与 manifest 一致。

25/50/75/100/125/150/175/200 的 online task-balanced functional loss 为
`.130744/.133971/.133841/.133092/.132344/.133132/.134178/.137535`。这些
loss 只作数值监控，不能替代即将运行的 closed-loop rollout。

正式命令为：

```bash
env PYTHONPATH=/data/ymdai/projects/EMBER/src \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 \
OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
/data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 \
scripts/train_as_writer.py \
--config configs/pi05_as_writer_language_axial_v6.json \
--mode formal \
--source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
--checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
--tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
--data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
--output-dir /data/ymdai/outputs/ember/pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728 \
--stop-after-step 200 --num-workers 2 --log-every 10 --skip-data-sha
```

正式合同是 total axis 2400 macros、当前 stage stop 200、每 25 macros 保存。
owner 明确取消 formal HDF5 全量 SHA；仍核对 sealed manifest、精确文件 size、
HDF5 schema、run/checkpoint state。不要重新计算没有科学收益的全量 SHA。

## 2. 当前隔离改动

训练期间不要修改 canonical main。仓库清理和 macro200→400 单调 stage
extension 修正在隔离 worktree：

```text
/data/ymdai/.codex/worktrees/EMBER-v53-20260728
branch: codex/v53-visual-transition-procedure
```

清理只退役无现行引用的旧 SmolVLA/70-10-10/Phase A–F、flat task-local RL、
flat Writer-RL、对应 config/test 和旧实验 config；保留仍被 source-corpus seal
引用的 `configs/libero90_70_10_10/data_manifest.json`。历史可从
`149badc` 及此前 Git 恢复。当前 canonical `source_sft/`、`task_local/`、
`rl_writer/`、π0.5 evaluator 和 v6 均保留。

隔离分支已通过：

```text
PYTHONPATH=<worktree>/src python -m pytest -q -p no:cacheprovider
177 passed
```

resume 修正只允许在 sealed `total_steps=2400` 内单调增加 operational stage
stop；其它 contract 字段仍 fail closed。它保留原 contract SHA 和 checkpoint
link，并在 `invocations.jsonl` 记录 requested stop、原 stop、代码变更和是否
stage extension。macro200 已退出并通过状态核验；下一步将该 commit
fast-forward 合入 main 后启动评测。

`.codex/tmp/v6_internal_specificity.py` 是待 observed-best 选择后使用的唯一临时
机制诊断脚本；在完成 v6 内部检查前不要清除。

## 3. 当前研究判断

历史结果只保留决定 v6 所需的最小摘要：

| Writer | correct400 / controls | 结论 |
|---|---|---|
| v4 | correct `109`, shuffled `148`, reversed `126` | 低层 phase/translation 旁路，逻辑失败 |
| v5 | best correct `115`; 五臂 `115/108/74/113/114` | Procedure 内部有序，但信号被 downstream 压弱 |
| v5.1 | best correct `127`; 五臂 `127/133/94/107/120` | wrong/shuffle 有方向，reverse 不稳；低 LR 未提高 |
| v5.2 | curve `72/79/120/132`; 五臂 `132/138/74/82/83` | 首次通过 wrong/shuffle/reverse 行为门，absolute/breadth 仍不足 |

v6 不是单纯扩大 v5.2。它把已验证的职责分离收敛为：

```text
task language + one action-hidden video
  -> frozen Gemma task queries / per-frame task-grounded evidence
  -> Semantic Set:
       mean backbone + task-selected centered residual
  -> Procedure:
       frozen Action-Expert probe
       + adjacent task-grounded visual transition
       + causal temporal encoder
  -> Core/Procedure compiler + refiner
  -> 320 routing identities / factor heads
  -> complete sealed rank-16 public LoRA
```

Semantic Set 对 frame permutation 不变；transition 必须在 correct/shuffled/
reversed 各 arm 的实际输入顺序内重新计算。Procedure 没有 absolute patch、
geometry、teacher action、state、reward 或 task-ID 旁路。v6 Writer 精确参数为
`10,775,296`；比 rank-128 Source-SFT 的 `10,297,344` 多约 4.64%，这是 owner
允许的同量级合理分配，不再机械凑相等。

## 4. 当前训练合同与已验证上限

每个 macro：

```text
4 DDP ranks × 6 tasks/rank = 24 tasks
每 task: 1 teacher video -> 1 LoRA -> B20 independent action queries
task 内 loss 求均值；24 tasks 等权
rank 内前 5 task backward 使用 no_sync，第 6 task 同步
每 macro 只做一次 clip / AdamW / scheduler
```

每 macro 精确消费 24 video conditions、480 action queries、24 次 functional
policy forward。task assignment 按当前选中 video 的 stride-5 frame cost 做
四组平衡；每 rank 内 long-first，四组跨 macro 轮换物理 rank。任何未来
checkpoint/卡数配置都继续遵守“worker 先取 long，long 耗尽后再取其它任务”。

B20 profile 已覆盖真实最长 105 帧视频并连续完成 3 macros：

```text
step seconds max-rank: 20.442 / 18.585 / 18.635
steady throughput:    25.793 queries/s
                      193.447 macros/hour
max allocated:        76,985,299,968 bytes
max reserved:         83,644,907,520 bytes
```

因此选择 B20，未运行 B16 fallback。step1→3 resume smoke 恢复了相同
task/video/query/LR/cursor；边界 checkpoint 文件 bitwise 相同。CUDA 后续更新
最大约 `9.82e-5` 非 bitwise 漂移已记录为 kernel 数值行为，不是状态丢失。
visual-transition 参数 step1→3 L2 更新 `0.0111083`，真实 functional gradient
可达。

## 5. 当前固定动作

1. 合入已验证的清理/resume 窄修正；main 与 origin/main 必须再次一致且 clean。
2. 在 GPU4–7 做均匀 fixed correct400 screen，默认 checkpoint 为
   macro50/100/150/200；每张卡只加载一个 checkpoint。若曲线峰值需要更密集
   定位，再利用每 25 保存的点补测，不预先全扫。
3. evaluator 使用 6 个 Writer generators + 6 个 persistent policy workers，
   teacher video 做 50 条无放回随机匹配，所有 worker 全局 long-first；不得用
   静态 task/GPU 分配。
4. 以 closed-loop absolute、paired 多 task 贡献和曲线选择 observed-best。
   对 best 做 correct/same-task-other/wrong/shuffled/reversed full400，并运行
   16-reference 内部 Core/Procedure/effective-LoRA/action 传递分析。
5. 除非 macro0→200 的 closed-loop absolute 明确下降，否则 exact-resume 同一
   recipe 到 macro400。平台、轻微波动或 online loss 上升都不是跳过第二段的
   理由；第三段必须重新由真实曲线决定。
6. v6 通过 absolute 与视频特异性门后，fresh 训练 corrected mixed-task
   rank-128 Source-SFT；再做每 task 预封存一条 episode 的 matched π0.5
   action one-shot baseline，最后进入独立 short-AS cold-start →
   pure-reward RL-Writer。
7. 不自动进入 final-32、test task-local RL 或 joint oracle。

## 6. 只读恢复命令

只查询 GPU4–7，绝不把 GPU0–3 写进命令：

```bash
RUN=/data/ymdai/outputs/ember/pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728

tmux has-session -t ember-v6-formal-200
tail -n 40 /data/ymdai/logs/ember/pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728.log
jq -s '{rows:length,last:.[-1]}' "$RUN/metrics.jsonl"
find "$RUN/checkpoints" -maxdepth 1 -type d -name 'step_*' -printf '%f\n' | sort
cat "$RUN/validation_functional_loss/metrics.jsonl"
nvidia-smi -i 4,5,6,7 \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

`/data/ymdai` 的 operator hard cap 是 500GB。首段完成后个人占用为
`402,806,538,492` bytes；任何新正式 run、批量 rollout cache 或 checkpoint
扩展前都要重新测现场和峰值空间。
