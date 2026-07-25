# EMBER Progress and Handoff

最后更新：2026-07-22。

## 当前状态

- 长期Goal已建立且无token budget；不会因source base、Writer或任一局部阶段提前完成。
- 活动目标split仍为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；seal位于 `configs/libero_24_8_8_v1/`。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已下载，weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 fixed states上为 `0/400`。400 rows唯一、全部到suite horizon，result seal SHA256 `c78e92e9...20c2`；该结果不评价EMBER。
- Phase A source audit、71-task manifest、source-only normalization、pinned official recipe与hash seal已完成；cost-balanced dynamic evaluator及fail-closed contracts已完成，真实1/2/3 replicas吞吐profile选择3 replicas/GPU。
- canonical π0.5 source-base full-SFT runner、atomic checkpoint与exact-resume机制已完成；fresh 1,000-step、333-step warmup、global batch256正式训练及step1000 raw/EMA checkpoint验证已完成，目标是轻量interface acquisition而非LIBERO-90收敛。
- formal attempt1因NUMA affinity缺失在step12终止；attempt2因显式zero右腕被LeRobot误标`mask=true`而在step316终止。两者均无checkpoint、failure packet已封存且永不resume；修正后的训练/评测都通过missing feature key得到OpenPI规定的zero image + `mask=false`。
- 当前里程碑fresh验证为112 tests passed；compileall与diff checks通过；architecture guard为`REVIEW`且无hard violation。ownership、增长理由与旧cache/cold-start/inference路径retirement已记录。
- 第一轮完整流程只跑一个training seed；不提前扩多seed或direct action-budget curve。
- source checkpoint落盘后live `/data/ymdai`占用约446.72GB，离500GB cap约53.28GB；development feature cache按274,523帧BF16×2048维约1.12GB主体估算，不需清理。后续只在预测峰值逼近上限时清理已核验可再生的smoke/profile产物，不删除正式checkpoint、raw rows或来源不明文件。

## Generic feasibility已验证的实现事实

- 使用LeRobot official π0.5 conversion：model chunk50、`n_action_steps=10`、每次执行前5 actions后重规划、10 flow inference steps。
- official evaluator：render256、model224、两相机旋转180°、seed7、50 fixed init states、dummy settling10；horizons Spatial/Object/Goal/Long=`220/280/300/520`。
- OpenPI公开PaliGemma tokenizer已逐token核验；模型/tokenizer manifests与24-train interface stats均封存在当前config目录。
- 首次8卡formal因固定`MUJOCO_EGL_DEVICE_ID=0`使GPU1–7在rollout前失败；修复为每进程物理`CUDA_VISIBLE_DEVICES`后formal全部exit0。失败root与正式root隔离。
- 单卡profile：1 env约27.52秒/episode，8 env约19.76秒/episode，16 env约19.58秒/episode；8→16仅约0.9%提升，峰值显存约20.1→23.2GB。
- 静态一task/GPU使两个horizon-520任务成为最后拖尾：正式最长task rollout约2169秒，而Spatial约1004秒。下一evaluator必须做cost-balanced state sharding/dynamic queue，不得复用静态映射作为效率上限。

## Target split

| suite | train | validation | test |
| --- | --- | --- | --- |
| `libero_spatial` | 0,2,4,5,7,9 | 1,3 | 6,8 |
| `libero_object` | 2,4,5,6,8,9 | 1,3 | 0,7 |
| `libero_goal` | 0,1,2,5,8,9 | 3,6 | 4,7 |
| `libero_10` | 4,5,6,7,8,9 | 1,2 | 0,3 |

算法seed `20260721`；key为 `seed\0suite\0task_name\0language\0bddl_file` 的SHA256排序，前6/中2/后2。test IDs不得按outcome替换。

## Source-base corpus audit：完成并封存

- 只读task language、BDDL、objects、roles、initial predicates与完整ordered composition，逐一审计90×40=3600对；算法为`manual_role_bddl_full_task_equivalence_v1`，不读取action/reward/proprio/terminal/normalization/policy outcome。
- 规则只排除完整有序任务等价，保留primitive/subtask containment与不同multiplicity/source/destination selector；scene、coordinates、distractors和BDDL实例编号不参与等价判断。
- 排除LIBERO-90 IDs `8,9,10,20,25,27,30,31,44,46,47,48,49,50,51,52,53,54,77`，active为其71-task补集。Goal3/4/7/8/9、Object0/1/4/5/6/7/9及Long5的完整映射记录在audit rows中。
- 明确保留near-miss IDs `2,29,12,13,14,15,38`，因为其完整composition或role selector与目标不同。
- active corpus为71×50=3550 successful episodes、529,173 frames、52,710,755,898 bytes；HDF5 aggregate SHA256 `81bdb358...a1a50e`。
- canonical hashes：overlap audit `fe731127...cc003`、manifest `75453a20...2e54`、source-only normalization `e259ee6e...f7c4`、recipe `4c537067...281734`；`sha256sum -c`通过。recipe hash变化只加入pinned OpenPI相机mask authority，不改变source corpus。

不得根据后续source-base/Writer outcome修改这些source IDs。

## π0.5 source-base recipe与工程合同

- 官方anchor固定为OpenPI `15a9616...ccac`、LeRobot `30da8e6...76ce`：full action-SFT、AdamW `(0.9,0.95)`、eps `1e-8`、weight decay `1e-10`、clip1、peak LR `5e-5`、10k linear warmup后constant、EMA `0.999`、30k steps、global batch256。
- source base采用full-SFT并最终直接冻结policy/EMA，不叠shared source adapter；下游统一LoRA合同为18层action expert q/v加action_in/out共38 targets、rank/alpha16、dropout0、B=0 identity init。
- 当前有效8卡profile只取相机mask修正后的m32+EMA smoke：3/3 steps finite，steps2–3平均47.75 examples/s，每卡peak allocated/reserved为67.18/71.18GB，八卡角色与NUMA绑定对称。contract/metrics/summary/log SHA256为`90fbe1da...0458`、`de2d9889...50d9`、`0a590a29...e1bc`、`26bb5aad...c10`；旧batch对比profile只作工程provenance。
- checkpoint包含policy、EMA、optimizer、scheduler、per-rank Python/NumPy/CPU/CUDA RNG、sampler/data/metrics cursor与完整hash manifest。两次从同一step1恢复到step2得到相同loss `0.3457298893481493`、grad norm `4.03354549407959`与逐rank state hashes；独立NCCL启动后policy/EMA最大末位差为`1.49e-8/3.73e-9`。compact evidence SHA256 `16137fa1...b1e`。
- formal fresh launch要求clean且HEAD已推到`origin/main`；resume由保存的commit/contract约束，不因后续`origin/main`前进失效。每次invocation另存当时完整Git观测。

## Canonical code ownership与生命周期

- `pi05_source_corpus.py`只拥有specification filter、data seal和source normalization；`pi05_source_setup.py`只拥有model/data/distributed setup；`pi05_source_contract.py`只拥有launch/resume contract；`pi05_source_checkpoint.py`只拥有atomic state；`pi05_source_training.py`只拥有训练编排与step loop。
- `scripts/train_source_base.py`是唯一活动π0.5 source-base入口；拆分这些owner是因为严格加载、52.7GB一次性校验、DDP训练与33.8GB atomic checkpoint是独立故障边界。
- 旧`source_base.py`/`source_base_checkpoint.py`仍被历史SmolVLA模块import，只作provenance；不得作为活动入口。π0.5 source-base训练与40-task evaluator达到功能对等后，先迁移剩余通用依赖，再删除旧可执行路径，不保留双canonical runner。
- target/Writer新增代码按故障边界分属六个owner：`pi05_target_data.py`负责零数值读取seal，`feature_cache.py`负责PI05 cache schema/store，`cache_pi05_writer_features.py`负责8-rank extraction，`writer/as_contract.py`负责24-task action墙及authority联锁，`writer/training.py`原位替代旧cold-start训练owner，`writer/checkpoint.py`负责atomic exact-resume。新增体量来自target authority、PI05 2048-d cache和完整AS state合同，不是平行算法实现；architecture guard无parallel family、无hard violation。
- 已删除旧`cache_writer_features.py`和`train_writer_cold_start.py`两个可执行入口。历史Smol `writer/inference.py`、RL与direct modules仍为provenance且会被新AS schema/PI05-only store fail-close；当PI05 Writer evaluation、RL-Writer、Source-SFT与task-local owner逐个具备功能对等时，删除对应旧CLI/import，而不是再加兼容分支。Phase F若需test video cache，将扩展同一PI05 cache authority并新增sealed role，不创建第二个cache runner。

## Formal source-base attempt 1：工程失败并关闭（2026-07-21）

- commit/worktree：`236202ed4371f301ef94bf8984aab423eef98db1`，`/data/ymdai/worktrees/EMBER-pi05-source-formal`；model/data/tokenizer全部hash门禁通过，8卡各一个69,130MiB CUDA rank。
- root/log：`/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_236202e_20260721`与同名`.log`；live CPU `PSR`发现rank可跨GPU的本地NUMA node漂移，违反launch contract。
- 在step12、3,072 global examples、首个checkpoint前发送SIGINT，pane exit130；failure packet明确禁止resume或科学使用。run contract/metrics/log SHA256分别为`997af43a...8b2`、`81dbfcbc...4ca`、`7a169300...118`。
- 修复：初始化CUDA device后从PCI sysfs解析NUMA node，把rank及其DataLoader children限制到该node cpulist；formal缺少binding时fail closed。修复smoke记录8个rank的完整affinity，metrics/contract SHA256为`bc01e6ce...6ca`/`42727d8c...18d`。

## Formal source-base attempt 2：工程失败并关闭（2026-07-21）

- canonical workspace：`/data/ymdai/worktrees/EMBER-pi05-source-formal-77ff1ef`；task-owned branch `codex/pi05-source-formal-77ff1ef`；commit `77ff1ef21567e1d5290921ca308cd8792813a504`已在`origin/main`且worktree clean。
- command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-pi05-source-formal-77ff1ef/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=8 scripts/train_source_base.py --config configs/pi05_source_base_v1.json --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90 --foundation-path /data/ymdai/ember_data/lerobot_pi05_base --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --output-dir /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_77ff1ef_20260721 --mode formal --num-workers 4`。
- scale/topology：71×50 successful source episodes；30,000 steps×global batch256=`7,680,000` examples；8-way DDP、每rank microbatch32、EMA0.999、无accumulation。formal代码要求rank0–3绑定NUMA0、rank4–7绑定NUMA1，并把8行affinity写入launch contract；每卡仍恰好一个同角色CUDA process。
- provenance/evidence：model、71 HDF5、tokenizer、audit、normalization和recipe全部重算/核验SHA；只读source action/state，不读取target action/reward/outcome。训练metrics只作source acquisition；完成后另行运行40-task fixed-state screen。
- output/log/storage：全新root为上述output，stdout/stderr为同名`/data/ymdai/logs/ember/*.log`；checkpoint每5,000 steps原子发布并只保留最新。与launch同一preflight shell测得个人占用379,947,614,208 bytes，双checkpoint峰值447,622,427,872 bytes，低于500GB cap且不下载/复制模型数据。
- resume/failure：仅从该root最新完整manifest、相同commit/config/contract恢复；超前metrics隔离到failure packet。attempt1 root永不resume；任何NUMA、hash、设备数、非空output或checkpoint漂移均fail closed。
- live acceptance：8个rank分别为PID 1131909–1131916，逐卡仅一个同角色CUDA进程且NUMA正确；运行至step316、80,896 examples时发现相机合同违反pinned OpenPI recipe，主动SIGINT停止。无checkpoint，不得resume或作科学结果。
- 根因与修复：显式传入zero右腕使LeRobot自动生成`image_mask=true`；官方OpenPI LIBERO transform要求third-camera zero image但mask为false。canonical processor和evaluator现在省略右腕feature key，由LeRobot创建zero padding + false mask；source config对此fail closed。
- evidence：failure packet/run contract/metrics/log SHA256分别为`2d2a9e40...9b80`、`e79e1c84...e7d8`、`fb0b2edc...f918`、`3f0eb65f...76f7`。旧tmux dead shell已在确认无GPU进程后清理。

## Canonical cost-balanced evaluator（实现与source profile完成）

- `scripts/evaluate_pi05.py`取代并删除旧静态`evaluate_pi05_base.py`，是唯一活动π0.5目标评测入口；不保留双runner。
- `pi05_eval_contract.py`拥有authority/final-policy/test-state门，`pi05_eval_queue.py`拥有cost-balanced SQLite WAL队列，`pi05_evaluation.py`拥有persistent policy/env与official rollout，`pi05_eval_results.py`单独拥有worker拓扑证据和strict aggregation；拆分是为隔离调度、runtime与不可变结果故障边界，不是平行runner。
- state shards按`count × horizon`估算cost并动态work-steal；8 GPUs上统一1/2/3 replicas，launcher CPU-only，GPU0无额外CUDA角色。policy noise按`(seed,suite,task,state,replan)`确定性独立，不受batch或worker顺序影响。
- launcher lock覆盖active-worker audit、queue recovery、preflight与spawn；partial spawn/failure只回收本launcher PIDs并封存logs/jobs/hashes。正式吞吐包含worker spawn、model load和首次env/EGL，另报raw shard window。
- formal/screen拒绝非当前完整source config、非final step1000 selected raw policy、相机interface漂移、test init hash漂移及同大小model/tokenizer篡改；aggregate交叉核对raw rows、DB counts、producer、8×replica topology、GPU UUID和NUMA。

- 在同一all-40-task×1-state panel上，1/2/3 replicas/GPU分别得到有效`0.155553/0.181832/0.189738` rollout/s，shard window分别`122.879/79.176/65.454`秒；3 replicas稳定、每卡约31GB，正式source screen据此使用3 replicas/GPU。

## Formal source-base attempt 3：已由owner停止并封存（2026-07-21）

- canonical workspace/commit：`/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055`，branch `codex/pi05-source-formal-aa8b055`，commit `aa8b0556619889480d8d9c129ea2f54af26c9d06`；启动时clean且等于`origin/main`。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=8 scripts/train_source_base.py --config configs/pi05_source_base_v1.json --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90 --foundation-path /data/ymdai/ember_data/lerobot_pi05_base --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --output-dir /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_aa8b055_20260721 --mode formal --num-workers 4`。
- output/log/tmux：`/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_aa8b055_20260721`、同名`/data/ymdai/logs/ember/*.log`、session `ember_pi05_source_aa8b055`。这是从generic base fresh启动的新root，绝不resume attempt1/2。
- launch authorities：source config SHA256 `366a5e12...ee8`、recipe `4c537067...1734`、run contract file `6db50495...1bec`；foundation weights和71个source HDF5均在formal启动时完整重算SHA并通过，right wrist合同为missing feature key→zero padding + `image_mask=false`。
- scale/topology：71×50 episodes，30,000 steps×global batch256；8个同角色rank PID 1264369–1264376，rank0–3绑定NUMA0、rank4–7绑定NUMA1。step7时每卡约69.0–69.2GB、GPU UTL 100%，loss/gradient finite，稳态约47.5 examples/s。
- live preflight：8卡启动前均0MiB且无compute apps；driver570.158.01、CUDA12.8、torch2.11.0+cu128。`/data/ymdai`为379,033,156,799 bytes，按实测33,837,406,832-byte checkpoint的atomic双态峰值估计446,707,970,463 bytes，低于500GB cap；`/data`可用约3.059TB。
- checkpoint每5,000 steps原子发布、只保留最新；完成前不作source competence结论。首个checkpoint预计约7.4小时，完整30k按当前吞吐约44.7小时；等待期间只在另一worktree推进不改其import/config/output的后续代码。
- step48 fresh live check仍为8卡各一个同角色PID、约69.0–69.2GB、100% GPU；loss/gradient finite，稳态约47.2 examples/s。该检查只证明运行健康，不作行为结论。

- owner随后把source预算明确改为fresh 1,000 steps；本attempt在step2880停止且没有checkpoint，`superseded_run.json` SHA256为`7aa78458d9f79885206b850f3c61738a5828019a14ee75ee06be91f1f9ff40a4`。不得resume或用于后续方法。

## π0.5 LoRA / one-video Writer core里程碑（2026-07-21）

- 新活动合同`configs/pi05_lora_v1.json`绑定generic `lerobot/pi05_base`完整revision/weights/config、当前source config与recipe hashes；文件SHA256 `1dcf58f7...cb07`，canonical contract SHA256 `42d5919e...94dd7`。
- 真实foundation safetensors metadata和meta-device `PI05Pytorch`结构均核验38个精确Linear targets；rank16得到76 tensors、1,287,168 parameters。没有加入state/time projections，也没有沿用旧37-target Smol合同。
- 通用`lora.py`只保留PEFT mechanics协议；活动科学拓扑由`pi05_lora.py`单独fail-close加载。这样旧历史imports不进入π0.5 runner，也没有第二个活动训练/评测入口。
- `CompleteLoRAWriter`现在在活动边界只接受一个非空video (`offsets=[0,L]`)；LoRA template和输出逐tensor保留真实BF16/FP32 dtype。functional action loss调用真实PI05 `forward(batch)`，不再传旧接口`noise/time`。
- fresh验证：真实checkpoint target metadata通过；mixed-dtype functional/copy parity精确通过；全套`91 passed`、compileall、diff/checksum通过。architecture guard为`REVIEW`、无hard violation；review仅来自既有Writer构造函数长度和目录密度，owner/lifecycle如上。

## Target40 seal、PI05 feature cache与AS-Writer owner（2026-07-21）

- target40下载已完成且未复制现有cache：四suite各10个HDF5，总计33,784,856,577 bytes。metadata-only seal为40 tasks/2,000 episodes/338,575 frames，24/8/8 IDs逐项一致；manifest/checksum均通过，manifest SHA256 `1b28547f...049d`。
- `configs/pi05_writer_feature_cache_v1.json` SHA256 `2165a2d9...e3ce`，只授权development train+validation视频，禁止test video与任意trajectory action/state/reward/terminal读取；raw source policy上的8卡batch32 smoke完成8 tasks/8 episodes/1,033 frames，critical-path `689.47 frames/s`且无OOM/nonfinite，因此profile已封存。
- smoke从launch到manifest约122秒，其中实际每task抽取1.03–1.50秒；run-contract/cache-manifest/log SHA256为`f166e86b...cda7`/`906e75b3...a178`/`f845a53c...65d3`。development formal cache规模为32 tasks/1,600 episodes/274,523 frames，BF16视觉主体约1.124GB。
- development formal cache已完成32 tasks/1,600 episodes/274,523 frames，实际目录1,126,794,227 bytes，launch到manifest 248秒；32/32 task safetensors逐文件hash复核通过，`test_video_values_read=0`。run-contract file/contract、manifest file/payload、extraction、log SHA256分别为`f803806a...1c7b`/`edcd59da...9410`、`72edc286...e92a`/`4b8f064c...a6a0`、`4ee82d04...2757`、`ce481b0d...5e94`。
- `configs/pi05_as_writer_v1.json`随feature authority更新后SHA256为`eb54c748...ea09`，明确24 train actions、one-video input、independent sampler/video seeds、frozen source normalization及≤120分钟正式wall-clock；formal状态`pending_profile`，当前4-step/batch1仅为未来mechanics profile默认值。
- AS训练已原位替换旧Smol cold-start owner：每rank每step由task-balanced action sampler给出task/visit，再由独立teacher-video schedule选一条demo；`WriterFeatureStore.load_one_video`只暴露pure language、该episode的video features和`[0,L]` offsets。policy、base与encoder冻结，只有shared Writer DDP更新。
- checkpoint先验证canonical manifest和全部file SHA再读取optimizer/RNG pickle，交叉核对manifest/trainer/rank cursor，metrics JSONL按checkpoint cursor隔离orphan rows；formal最终coverage由launch total-step自动推导，调用者不能关闭。rank写盘/发布失败会跨8 ranks一致传播，避免barrier死锁。
- 删除`cache_writer_features.py`和`train_writer_cold_start.py`；相关活动测试改为PI05 schema。fresh全仓`107 passed`，compileall、config SHA和target checksum通过；architecture guard结果`REVIEW`且hard violations为空。review增长理由与retirement trigger见ownership段。
- formal source-base attempt3在上述工作之外的隔离worktree继续健康运行；最近只读观测step1100，8卡各约69GB且97–100%利用率，loss/gradient finite、约46.5–47.4 examples/s。此状态不构成source competence或行为结果。

## PI05 Writer evaluation与wrong-video机械证据（2026-07-21）

- 没有新增平行runner：`evaluate_pi05.py`、`pi05_eval_contract.py`、`pi05_evaluation.py`和`pi05_eval_results.py`原位支持source-base或AS-Writer arm，共用dynamic queue、persistent env、fixed-state rows、resume和aggregate。
- AS evaluator逐字段联锁raw source policy、AS config/run/checkpoint、PI05 LoRA和feature cache；正式screen/formal只接受formal AS run，development cache可用于train/validation且会对test显式fail-close。未来test-open cache必须另行封存，不能把当前`test_video_values_read=0` cache冒充final cache。
- 每个rollout由顺序无关的哈希独立抽一条teacher video；correct与wrong arm共用selection seed/demo ordinal。wrong video按同split role跨suite双射，完整map、map SHA与condition进入run-contract hash。
- materialized backend每episode生成一次完整LoRA，并在该episode每次replan前安装同一state；不同活动env不会被错误合成普通同adapter batch。functional batched backend仍待final source产生后做真实rollouts/s profile。
- raw rows和aggregate保留checkpoint/cache/map/video/LoRA/timing证据，row validator会重算video seed、demo和map。fresh全仓`112 passed`，compileall、diff check通过，architecture guard为`REVIEW`且无hard violation；尚未因source未完成而运行GPU Writer smoke或产生科学结果。
- formal source-base attempt3最近只读观测step1100：8卡约69GB、97–100%利用率，loss/gradient finite、约46.5–47.4 examples/s；仍未到首个checkpoint，不作competence结论。

## PI05 shared Source-SFT owner与静态评测接入（2026-07-21）

- 新增development-only配置`configs/pi05_source_sft_development_v1.json`及checksum（SHA256 `32e927c...8a641`）。它只授权24 train tasks×50 action episodes，四suite各6个；final stage在该config内fail-close，validation选择后必须创建独立final authority，不能续接development LoRA。
- `ember.source_sft`成为单一owner：`contract.py`联锁target manifest、final raw source policy、tokenizer/source normalization和PI05 LoRA；`training.py`在8个对称DDP ranks上只训练一套shared LoRA；`checkpoint.py`原子保存adapter-only exact-resume state；`inference.py`核验formal artifact并一次性安装静态adapter。薄入口为`scripts/train_source_sft.py`，没有复用旧Smol per-task direct runner。
- exact-resume state包含optimizer/scheduler、optimizer=micro cursor、metrics cursor、每rank RNG、sampler/data identity、DataLoader-derived worker seed与50-episode coverage；manifest在pickle前逐文件验SHA。development/final、source checkpoint、config或LoRA合同任一变化均拒绝resume/evaluation。
- canonical `evaluate_pi05.py`新增与AS-Writer互斥的Source-SFT参数；共享LoRA每worker只安装一次，随后继续普通multi-env batch和dynamic queue。raw rows记录固定LoRA state SHA；Source-SFT不生成Writer row，也不生成Writer correct/wrong pairing hash。
- fresh验证：Source-SFT与evaluator聚焦测试`23 passed`，全仓`119 passed`，compileall、config checksum、diff check通过；architecture guard为`REVIEW`且hard violations为空。review增长来自新增baseline的独立data-wall/training/checkpoint故障边界；旧`direct_lora*`的删除触发为本PI05路径完成真实8卡finite loss/grad与exact-resume smoke。
- formal source-base attempt3保持隔离且未被修改；最近只读观测step1450、371,200 global examples，8卡各约69GB、100%利用率，loss/gradient finite、约47.39 examples/s。仍未到step5000 checkpoint，不作source competence结论。

## PI05 reward core、zero-AS RL-Writer与test-only task-local合同（2026-07-21）

- Writer-v2替换活动authority后，`configs/pi05_rl_writer_development_v1.json`与`configs/pi05_task_local_rl_test_v1.json`只做机械rebind；随Writer-v2 formal seal更新后当前SHA256为`6eac8449...b954`/`97a4ce86...b509`。前者formal状态仍为`pending_source_screen_and_real_profile`；后者所有正式budget保持0并写明`blocked_until_zero_interaction_test_and_test_open`，因此当前代码完成不会越过阶段信息墙，也没有启动RL。
- `ember.reward`统一实现official random BDDL reset、10-step settling、suite horizon、显式逐replan PI05 flow-noise、成功即停、immutable raw ledger和三类cursor。成功trajectory只保留真正执行的每个replan前缀；reward loss不会监督未执行的45/50 actions。
- `ember.rl_writer`拆分为contract、runtime、loop和checkpoint owner。fresh Writer在8 ranks上使用共同确定性seed且generated LoRA功能恒等；只有rank0原子发布run contract并跨rank校验digest。Writer-only DDP更新、task/video full-cycle coverage、完整per-rank RNG、optimizer/scheduler、metrics与ledger-bound exact-resume均已接入；micro-AS分支在zero完整负证据前硬拒绝。
- `ember.task_local`已封存8 test tasks、三臂/cohort video/匹配seed、一次性初始化bundle、physical task-LoRA-only executed-prefix update、随机reset reward checkpoint选择和hash-bound resume mechanics；尚未实现或启动test formal runtime，不把机械合同写成结果。
- 为保持一条活动路径，删除旧Smol `evaluate_source_base.py`、`train_writer_only_rl.py`与`train_task_local_lora_rl.py`三个可执行入口。历史模块/配置暂时只供provenance；新π0.5 task-local runtime和canonical fixed-50接入达到功能对等后删除其余旧实现与测试。
- fresh验证：reward/RL-Writer/task-local/evaluator定向`37 passed`，全仓`141 passed`，三份新config checksum、compile与diff check通过；architecture guard为`REVIEW`且无hard violation。约4k新增活动行按上述三个故障边界分属共享reward、shared Writer与task-local mechanics，不存在parallel version/function family；retirement trigger如上一条。
- formal source-base attempt3仍在隔离worktree不受修改；最近只读观测step2050、524,800 global examples，8卡各约69GB且100%利用率，loss/gradient finite、约47.37 examples/s。该状态仍不是source competence或行为结论。
- `configs/pi05_seen_panel_v1.json`在上述outcome之前封存四suite各2个development-train tasks，文件SHA256 `6f96b28e...0aee`；选择只使用task language/identity/BDDL SHA与固定seed，global IDs为`0,2,15,12,21,28,39,37`，policy outcome和trajectory value reads均为0。

## Frozen RL-Writer canonical evaluator接入（2026-07-21）

- `src/ember/rl_writer/inference.py`核验RL run contract、8-rank checkpoint全部file hashes、source/cache/config联锁，并从checkpoint cursor重算task-balanced schedule和teacher-video coverage；development config未seal时formal evaluation继续fail-close。
- `writer/inference.py`把AS/RL两种Writer统一到同一video mapping、pairing hash、per-rollout LoRA生成和raw evidence；row evidence现在显式记录`writer_method`、checkpoint axis/cursor。AS axis为optimizer step，RL axis为reward update。
- `evaluate_pi05.py`新增互斥RL-Writer资产参数，但仍只使用原dynamic queue/persistent worker runner；prepare与runtime adapter ownership集中在既有`eval_adapters.py`，脚本由797行降至755行，没有第二套评测入口。
- fresh验证：相关`38 passed`，全仓`143 passed`，compile/diff check通过；architecture guard为`REVIEW`且hard violations为空、无parallel family。formal source-base attempt3最近只读观测step2250、576,000 global examples，loss/gradient finite、约47.17–47.42 examples/s；仍不作行为结论。

## Fresh 1k source base完成（2026-07-22）

- canonical workspace/commit为`/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055`、`e2cc238b6423d3c41c681e3764fca96d64203a16`，启动时clean且等于`origin/main`。全新root为`/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722`，没有续接任何旧权重。
- 8×A100、global batch256、333-step warmup、EMA0.999，恰好完成1000/1000 optimizer steps与256,000 global examples；训练loop wall-clock `5494.806s`（91.58分钟），启动至结束约95分钟，低于120分钟guardrail。
- loss的50-step means从steps1–50的`0.26213`降至951–1000的`0.08659`；末段仍缓慢下降但已经明显趋平。按owner指定budget停止并记录budget-censored，不自动追加训练。
- final checkpoint为`checkpoints/step_00001000`，完整file/manifest验证通过；原始contract/summary/metrics/checkpoint-manifest/log SHA256分别为`ae05c077...e374`、`921ea45a...2032`、`b77c8498...6d3c`、`c236cb2d...d6bf`、`d2f17289...2d0c`。checkpoint contract canonical hash为`b6090341...e867`；原summary选择EMA已被后续诊断纠正并保留原hash作provenance。
- checkpoint apparent bytes为`33,837,823,088`；trainer state确认step1000、raw与EMA都存在、1000条metrics step唯一且无invalid rows。下游共同起点现明确为raw `policy/`与同一source-only normalization；不重训source base。
- 先前40-task×8-state EMA screen为0/320，机械执行虽完整，但该模型选择无效：EMA0.999从generic初始化，在1k短训后仅走完raw参数更新位移的28.62%。同一4个active LIBERO-90 tasks、相同init/noise下raw 4/4、EMA 0/4；同一32 source samples flow loss为raw `0.06165`、EMA `0.17775`、generic `0.29302`。
- raw 40-task×8-state正式screen得到`46/320 = 14.375%`，成功覆盖13/40 tasks及全部4 suites；逐suite为Long `2/80`（1 task）、Goal `28/80`（5 tasks）、Object `1/80`（1 task）、Spatial `15/80`（6 tasks）。因此跨多task partial competence门槛正式通过，不是0 competence或单易task支撑。
- screen共有320个唯一task/state rows、24/24 workers exit0、40 shards完整、无错误；wall-clock `412.372s`、有效`0.775999 rollout/s`。results/run-contract/run-summary/log SHA256分别为`4e2defaf...db3a`、`e496dc6a...1d7b`、`7befa655...ed71`、`ab553c0e...4e9d`；旧EMA 0/320仅保留为superseded engineering evidence。

## 已对齐的后续方法

- frozen source base：过滤后LIBERO-90×50 action-SFT，必要source LoRA merge，source-only normalization冻结；快速screen全部目标40 tasks，需开始在多个tasks有部分真实成功，不能只靠一个易task aggregate。
- 全部适用训练阶段：先短profile学习速度/吞吐，用固定廉价screen与曲线斜率淘汰候选，仅对少量候选做完整validation；约2小时只是guardrail而非目标，到上限仍未充分训练则保存证据后停止自动追加。
- AS-Writer：24 train/8 val开发，one video，video/action episode同task独立采样；loss驱动稀疏val与早停。
- RL-Writer：随机Writer、零AS warm-up起步；无reward再极少warm-up，仍失败则关闭。
- Source-SFT：24/32 source tasks联合一套shared LoRA；独立val选最佳，不匹配AS steps/data。
- seen comparison：specification-only预声明覆盖四suites的source panel。
- wrong-video：直接另一suite，正确language/task/state/RNG不变。
- final：合并为32 source，单seed分别重训后先seen、再zero-interaction test。
- test-only RL：不碰validation；test task上训练identity/AS/RL Writer三臂到接近最佳，官方random resets，fixed50只fresh eval。
- task-local RL预算按每种初始化方法跨全部8个test tasks的合计训练wall-clock约束，不是每task各给2小时；逐task记录是否因总预算截断。
- direct oracle：最后使用8 test tasks×50 actions联合一套shared LoRA，不是per-task LoRA。
- optional：核心后有时间再做ViVLA；outer learning不阻塞。

## 当前后续动作

1. 进入final 32-source单seed fresh重训；development Source-SFT、AS-Writer和RL-Writer均不再追加训练或额外消融。
2. final模型先完成sealed seen comparison，再打开8个test tasks做zero-interaction correct/wrong对照。
3. RL-Writer后续只保留既定task-local RL-init与final correct/wrong用途；其development视频因果证据较弱，不补generic arm或micro-AS追正结果。

## AS-Writer短周期profile与正式seal（2026-07-22）

- 最初4-step/batch1 mechanics probe使用了4-step scheduler horizon，LeRobot的自动缩放把1,000-step warmup取整为0，首步直接施加`3e-4`并产生无代表性的loss/gradient上冲；该run仅作mechanics evidence，不参与正式步数选择。
- 修正后的profile保持1,000-step schedule horizon，只执行前128 steps；8卡每rank batch16、global action queries/step为128。稳态约1.05秒/步、约122 queries/s，每rank峰值allocated/reserved为63,534,307,840/68,167,925,760 bytes，符合约10GB稳定余量目标且无需继续batch sweep。
- 16-step mean loss依次从`0.14714`、`0.13880`、`0.12935`降到末段`0.11930`；后64步线性斜率为`-1.576e-4/step`，仍在学习且无nonfinite/冻结越界。由此选择完整1,000-step schedule，而不是按120分钟上限倒推；预计净训练约17.5分钟。
- v1正式AS配置曾封存batch16、1,000 steps、checkpoints 250/500/750/1000；这些profile与训练结果只作v1 provenance。v2仍处于真实profile前，不能继承v1 batch/step结论或冒充已封存配方。

## 历史边界

旧SmolVLA 70/10/10曾完成到旧Phase F并留下真实结果，但与当前π0.5、split、one-video和source-base合同不兼容。只能用作经验/provenance，不能复用checkpoint、normalization或runner。

## AS-Writer validation与matched-scale Source-SFT seal（2026-07-22）

- AS cheap screens为step250/500/750/1000=`24/18/15/15`（各64 rollouts），由预封存规则只将250和500送入完整validation。完整8×50结果为step250 `119/400`、step500 `99/400`，development AS-Writer据此冻结step250。
- owner要求补测的同配置source-base validation完成`48/400`，逐任务为Spatial 1/3=`0/0`、Object 1/3=`5/0`、Goal 3/6=`0/41`、Long 1/2=`2/0`；AS step250逐任务为`0/0, 40/36, 0/27, 16/0`。
- 在RL-Writer前先运行Source-SFT。首版不调step，只匹配已选AS checkpoint约32,000 action-query总训练量；保持实测高效的8卡每rank batch64，固定63 steps=`32,256` queries，只保留step63。此前batch64/128 profile仅作稳定性与吞吐证据，不作为科学结果。
- 下一动作：封存并推送Source-SFT配置后完成该固定训练；仅对step63运行一次完整8-task validation，与source base `48/400`和AS `119/400`比较。
- Source-SFT固定run已完成：63 steps、32,256 queries、训练wall `450.263s`，唯一checkpoint完整exact-resume核验通过；首/末8-step mean loss `0.15475/0.13883`，仍下降但按固定预算停止，不追加或精调step。
- step63完整8×50 validation得到`61/400`，逐任务Spatial 1/3=`5/1`、Object 1/3=`20/0`、Goal 3/6=`0/32`、Long 1/2=`1/2`。相对source base为`+13`，相对AS step250为`-58`；400 rows的env/policy seed与noise-seed共享前缀全部配对。
- 训练root为`/data/ymdai/outputs/ember/pi05_source_sft_development_equalq32k_99b6020_b64_s63_20260722`，validation root为`/data/ymdai/outputs/ember/pi05_source_sft_val8x50_equalq32k_step0063_99b6020_r3_20260722`。results/comparison SHA256为`92e3e667...3f6d`/`c376ef9c...a1f`。
- AS step250 cross-suite wrong-video正式对照完成`115/400`，correct-video为`119/400`，source base为`48/400`。paired flips为both-success 102、correct-only 17、wrong-only 13、both-fail 268；correct−wrong仅+4，当前development AS checkpoint未显示强视频内容依赖。
- wrong-video root为`/data/ymdai/outputs/ember/pi05_as_writer_val8x50_cross_suite_wrong_step0250_fa635cc_r3_20260722`；24 workers exit0、38 shards和400 rows完整，results/comparison SHA256为`0e6ee518...a9ce`/`d4a4f9f7...eaac`。
- canonical `evaluate_pi05.py`现直接支持已封存的8-task `seen_panel` derived role：入口核验`pi05_seen_panel_v1.json`及checksum、target-data authority、四suite各2个train task和零outcome/value reads；Source-SFT同一静态adapter路径仅接受该精确subset。没有新增runner或改写旧checkpoint/config authority，相关聚焦43 tests及全仓144 tests通过。

## Writer-v2组合修订已实现、待真实profile（2026-07-22）

- 旧v1 correct/wrong为119/115，且400/400视频虽不同、生成LoRA hash虽不同，有效`B@A`的correct/wrong相对差中位数只有`7.52e-6`；当前状态判定为科学shortcut negative而非cache/evaluator故障。seen与RL-Writer继续暂停。
- 新活动authority为`configs/pi05_writer_feature_cache_v2.json`和`configs/pi05_as_writer_v2.json`；真实cache smoke封存后当前SHA256为`3dc3557d...396c`/`5ffd85d4...b93c`。旧v1配置由schema fail-close，只保留artifact/Git provenance。v2仍使用同一raw source policy、24 train actions、development train/validation videos、38-target complete LoRA和同一canonical训练/评测入口。
- cache从每帧全局mean改为固定4×4 spatial grid，tensor为`frames×16×2048 BF16`；预计完整274,523-frame cache主体约17.99GB。当前`/data/ymdai`实占约232GB，峰值远低于500GB cap，无需删除既有证据。
- Writer-v2为14,403,200 trainable parameters；language/video使用独立固定token memory，所有层级attention去除query-only residual，decoder只以parameter query乘性调制conditional memory，output heads无bias且identity init不变。CPU full-shape direct check生成76个三condition tensors、全部finite，人工开启head后condition mean diff非零。
- owner最终口径已在任何artifact产生前覆盖初版三分支：固定中性language `perform the demonstrated task`经正常tokenizer/embedding进入Writer，policy在所有分支始终收到正确task language。训练按`normal → full-language contrast → generic-language contrast`三步循环；contrast用半批query复制为correct/wrong两臂并共享policy RNG，总policy samples/step不变，两个correct臂都有绝对functional action loss。action query仍来自同task独立episode；paired negative只来自预封存的另一个development-train task并采用对称配对。
- v2 cache smoke完成8 tasks/8 episodes/1,033 frames，8 ranks全部exit0；最慢rank task wall为1.550秒，对应critical-path约666.25 frames/s。每个task tensor包含普通task language、同一个8-token generic language embedding和`frames×16×2048 BF16`视频；8份generic embedding逐byte SHA256均为`62172105...7d27`。run-contract/cache-manifest/log SHA256为`b4313579...d2d7`/`bcadf191...17b`/`b0f8124c...7045`，batch32据此封存。
- v2 formal cache现覆盖32个development tasks、每task50条视频、共1,600 episodes/274,523 frames；32/32 task tensors、episode/frame counts与SHA256复核通过，generic embedding在全部tasks逐byte一致，test-video/action/state/reward/terminal读取均为0。cache root为`/data/ymdai/outputs/ember/pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722`；run-contract/manifest/log SHA256为`219920cc...1f3a`/`b98a934c...ade2`/`06cb5f6d...51a6`。
- 真实8卡batch16学习profile执行30步且三种mode各10步；normal positive首/末3步均值从`0.16161`降至`0.13554`，full-language gap从`-3.44e-6`移至`+9.58e-5`，generic gap从`-1.04e-5`移至`+4.13e-5`。最大allocated/reserved为`63,748,333,032/68,543,315,968` bytes，全部finite、policy prompt始终为正确language。profile只说明机制开始向正确方向移动，尚不构成视频特异性结论。
- 据此封存首轮250 steps、batch16/rank、checkpoints 50/100/150/200/250；每个checkpoint先做固定functional specificity诊断，step250仍未饱和则按owner规则记录undertrained而不自动追加。`configs/pi05_as_writer_v2.json`当前SHA256为`65383ab8...40ab`；下游RL-Writer/task-local仅机械rebind为`6eac8449...b954`/`97a4ce86...b509`，均未启动。不同mode的wall-clock和吞吐只作资源记录，不作为删减分支、缩小对照或拒绝启动的门槛。
- 正式首轮root为`/data/ymdai/outputs/ember/pi05_as_writer_v2_development_seed7_dcfb206_b16_s250_20260722`，完成250/250 steps与五个完整checkpoint；训练段wall `334.476s`，24 tasks各覆盖全部50 action episodes与50 teacher videos。run-contract/metrics/summary/log SHA256为`08e58005...af4c`/`592212b6...289`/`8b3e456f...37b`/`dfd866b1...a7d`，250 rows连续、唯一、全部finite。
- 首/末10步positive loss为normal `0.14501→0.11478`、full `0.14337→0.12463`、generic `0.13617→0.12040`；full/generic末10步wrong-minus-correct均值为`+0.00707/+0.00788`。这是比30-step profile更强的训练内信号，但不同step query不相同，尚不能替代fixed-query或closed-loop验证。
- 同一canonical evaluator已增加`generic_correct/generic_cross_suite_wrong`条件；generic只替换Writer language为正常cache得到的中性embedding，policy始终接收正确task language。聚焦测试`33 passed`；下一步在clean/pushed commit上对step200/250做四分支cheap validation screen。

## Writer-v2首轮validation与双方法ceiling任务（2026-07-22）

- Writer-v2 step250的64-state screen为full-language correct/wrong=`12/8`、generic correct/wrong=`12/8`；完整full-language correct/wrong=`83/63`（各400），paired correct-only/wrong-only=`40/20`、exact McNemar `p=0.01349`。视频特异性已跨5个正向tasks成立，但correct绝对性能低于v1的119/400。
- owner将当前执行范围收敛为：充分训练并独立选择Writer-v2与Source-SFT的validation最强checkpoint，然后比较并暂停。Writer候选选择只看correct-video；唯一最佳checkpoint确定后才补一次correct-language + cross-suite-wrong-video完整control，不再运行generic full或候选wrong arms。
- formal seal已通过全仓`150 passed`：Writer-v2 fresh 1,500 steps、batch16/rank、50 warmup/1,500 decay、250-step checkpoints，配置SHA256 `34c5a1f8...34b7a`；Source-SFT fresh 800 steps、batch64/rank、100 warmup/800 decay、checkpoints 100/200/400/600/800，配置SHA256 `3f5a2c93...15818`。下游未启动的RL-Writer/task-local仅机械rebind authority hash；两者若到120分钟guardrail仍未饱和，保存曲线并明确标记budget-censored，不自动追加。

## AS-Writer与Source-SFT development选择完成（2026-07-22）

- Source-SFT ceiling run完整validation为step200/400/600=`74/87/73`（各400）；step400是明确峰值，故停止在最近完整checkpoint并冻结step400，不恢复到800。development Source-SFT至此完成，后续不重复。
- Writer-v2原1,500-step run的correct-video完整曲线为step500/750/1000/1500=`99/92/75/72`。为排除未保存细峰，另从identity fresh运行相同训练合同、只加密350–750 checkpoint retention；16-state screens将600/700/750送入完整validation，结果为`90/85/95`。没有候选超过原run step500，且原run后段持续下降，因此冻结原run step500=`99/400`。
- 唯一最终wrong-video arm在同一step500上得到`55/400`；paired both-success/correct-only/wrong-only/both-fail=`43/56/12/289`，exact McNemar `p=6.21e-8`。逐task correct−wrong为Long `-1/-2`、Goal `+1/+11`、Object `+17/+12`、Spatial `+2/+4`，视频正效应来自6/8 tasks而非单任务。
- development主比较为source base `48/400`、Source-SFT `87/400`、Writer-v2 correct `99/400`。v1虽有`119/400` correct，但wrong仍为`115/400`；它更像公共LoRA。v2牺牲部分绝对correct性能，却把wrong降到55并建立强视频因果差异，按owner口径暂时通过。
- peak-scan run summary/metrics SHA256为`31e36942...f29`/`171acd12...9c1c`；最终correct/wrong results SHA256为`e55cdf8e...66e4`/`3f9d6f1e...8b6f`。下一阶段直接进入zero-AS RL-Writer，不补不影响当前判断的AS消融。

## RL-Writer真实profile与formal seal（2026-07-22）

- 第一次profile在reset前发现64-bit hash seed不满足LIBERO内部`np.random.seed`的uint32范围；0 rollouts/0 actions。seed owner现只对environment seed做uint32映射，policy/update seeds与跨arm配对不变，相关22 tests通过。
- 第二次profile完成首批8 rollouts（4 successes）后暴露成功/失败ranks的DDP分支互等；成功轨迹26–44 replans被整批反传，峰值约80GB。修复为8-chunk policy replay microbatch：先累积生成LoRA叶梯度，再一次回传Writer，并按固定顺序手工all-reduce；逐成功episode全局等权不变，没有新增算法分支。
- 第三次profile完整覆盖24/24 train tasks各1条official random-reset rollout，得到7 successes、7167 environment actions、3/3 optimizer updates；成功覆盖Spatial 0/2/5/7、Goal 1/8和Long 5。单cycle max-rank wall `133.471s`，三update分别`46.668/42.297/44.510s`，峰值reserved `40,842,035,200` bytes。
- profile checkpoint update3全文件hash、24-task no-replacement coverage和cursor复核通过；run-contract/metrics/summary/checkpoint-manifest SHA256为`487fee45...4d28`/`d7e22bc7...cdea`/`3bb5b550...008a`/`2a27f673...c95`。零warm-up已有明确reward signal，不进入micro-AS。
- formal最大合同为120 updates=40 full cycles，按profile约89分钟净循环，并为分段模型加载、checkpoint与判断保留约31分钟；checkpoint为3/6/12/24/36/54/72/96/120。runner现允许只在这些sealed checkpoints暂停并保持同一contract exact-resume；首段预声明只跑到update12。

## RL-Writer development完成（2026-07-23）

- canonical root为`/data/ymdai/outputs/ember/pi05_rl_writer_zero_formal_seed7_r2_376ac0f_20260722`。此前resume启动前的metrics reconciliation错误已定位为AS轴`optimizer_step`与RL轴`next_update`混用；修复后rank0异常会广播到全部ranks，fresh全仓`154 passed`，旧失败run只保留failure evidence，未跨代码commit恢复。
- 新run先到update3，再从update3真实8-rank exact-resume到24，随后从24到36、36到54；最终432 rollouts、81 successes、131,354 environment actions、44个有效optimizer updates，净训练wall `2261.716s`。run-contract/invocations/metrics/summary SHA256分别为`d637959f...073`/`637f4878...946`/`d5a064c7...22d`/`ed5aee66...4d4`。
- 固定64-state screens为update12/24/36/54=`6/11/15/14`，results SHA256分别为`97c78986...e78`、`59dacc72...17e`、`d7e73252...3eb`、`ae641ec8...2a1`；据此冻结update36，不继续训练。selected checkpoint manifest/writer SHA256为`85577446...596`/`57f9b12c...2af`。
- selected update36完整correct-video validation为`94/400`，root为`/data/ymdai/outputs/ember/pi05_rl_writer_validation8x50_r2_update00000036_correct_376ac0f_r1_20260722`，results SHA256 `d1d4b1cf...aa5`。唯一cross-suite wrong arm为`87/400`，root suffix为`...update00000036_wrong_376ac0f_r1_20260722`，results SHA256 `6601221a...325`；两者`paired_control_sha256=57e3985c...321`。
- paired correct-only/wrong-only为`10/3`，exact McNemar `p=0.092285`。因此RL-Writer held competence成立但视频特异性较弱；已有zero-AS reward signal，不启用micro-AS，不补generic或更多wrong arms。下一阶段是预封存seen panel比较。

## Phase E sealed seen-panel完成（2026-07-23）

- 预封存8 train-task panel的四个必要arms均完成8×50 official fixed-state评测：source base `137/400`、Source-SFT step400 `182/400`、AS-Writer-v2 step500 `204/400`、RL-Writer update36 `164/400`。逐task原始counts与runtime已写入各自`results.json`；AS最强，SFT次之，RL仍高于base。
- results roots依次为`pi05_source_base_seen8x50_raw_e92b482_r3_20260723`、`pi05_source_sft_seen8x50_step0400_e92b482_r3_r2_20260723`、`pi05_as_writer_v2_seen8x50_step0500_e92b482_r3_20260723`、`pi05_rl_writer_seen8x50_update00000036_e92b482_r3_20260723`；SHA256依次为`91a9a31f...fb833`、`05c4c0d1...d889b`、`3d640e57...d97479`、`92a958a3...3f2c8`。
- Source-SFT原训练被owner在选择step400后于step600停止，未发布terminal summary；当前evaluator的fail-closed检查在0 rollout处暴露该缺口。只依据不可变contract、600条metrics和step600 manifest确定性重建summary并另存recovery provenance，SHA256为`887ae816...ab2e`/`c7f29ae7...803c`，未运行训练或改写checkpoint。
- Phase E的必要seen比较至此封存；不补seen wrong-video或额外checkpoint。下一步为final 32-source fresh AS/SFT/RL重训合同与运行。

## Final 32-source合同已封存（2026-07-23）

- 单一AS与RL canonical runner现按immutable config支持development/final stage；Source-SFT已有final owner。32-source角色严格为`train+validation`，每suite8 tasks，test不进入训练。
- final AS保持1,500-step scheduler horizon并机械停在选定step500；final Source-SFT保持800-step horizon并停在选定step400，避免LeRobot因缩短`total_steps`自动改变warmup/decay。final RL固定12 full source-task cycles，即48 updates/384 rollouts。
- final AS/SFT/RL配置SHA256依次为`ebe269ea...e299e`、`25e99628...d10c2`、`32dd979b...2ab30`；32-task Writer cache可直接复用。聚焦测试与全仓fresh验证为`28 passed`/`159 passed`，下一步在clean pushed commit上依次启动三套fresh单seed训练。

## Final AS-Writer训练完成（2026-07-23）

- root为`/data/ymdai/outputs/ember/pi05_as_writer_v2_final32_seed7_a5173a1_b16_stop0500_20260723`；fresh训练实际500 steps、原scheduler horizon 1,500、wall `634.671s`，32 tasks各覆盖50 action episodes与50 videos，500 rows全部finite。
- checkpoint/run-contract/metrics SHA256为`b30b2e1d...c395`/`36207182...2de`/`0d208b15...b619`；末20 full/generic matching gap为`0.00729/0.00783`，训练机械与信息墙通过。
- 修正了run summary沿用development `validation_action_reads=0`的报告错误；权重和optimizer改动为0，corrected summary/correction provenance SHA256为`a4f76fb2...9de7`/`ebc1bed8...414e`。全仓fresh仍为`159 passed`；下一步启动final Source-SFT。

## Final Source-SFT训练完成（2026-07-23）

- root为`/data/ymdai/outputs/ember/pi05_source_sft_final32_seed7_5922c61_b64_stop0400_20260723`；fresh shared LoRA实际完成400 steps、保留原800-step scheduler horizon，wall `2857.608s`（metrics训练loop `2852.793s`），共204,800 queries。
- 32 tasks各用满50条action episodes，400 metrics连续finite；首/末20-step mean loss `0.15139→0.11531`。checkpoint/run-contract/metrics/summary SHA256为`0012ffb6...52bd`/`bc136964...da31`/`c0d91c9b...6211`/`ff0a33f7...d472`，exact-resume文件全部校验通过。
- final AS与Source-SFT现均完成；下一步从fresh zero-AS Writer初态启动48-update final RL-Writer，不启用micro-AS或额外分支。

## Action-Memory AS-Writer实现与formal profile（2026-07-23）

- owner将当前执行范围重置为新Action-Memory AS-Writer闭环，暂不推进RL。新canonical路径直接对每个stride-4采样帧运行冻结PaliGemma图文prefix，再用16个可学习memory tokens读取Action Expert全部18层hidden states；encoder-only rank8 Meta-LoRA只在teacher-video编码时安装，执行policy不携带它。
- 变长视频按padding mask批处理；`B×18×16`条时间序列在同一temporal Transformer中并行但互不注意，随后批量layer/slot mixing并直接解码完整76-tensor、1,287,168-scalar task LoRA。Writer共10,097,601个训练参数，是rank128 Source-SFT 10,297,344的98.1%。
- 最终源码8卡profile root为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_v1_profile_refactor_20260723`：per-rank batch16、global128，两步耗时3.851/2.477秒，loss与gradient全部finite；峰值allocated/reserved为75,991,897,600/78,873,886,720 bytes。
- profile run-contract/metrics/summary SHA256为`21f15cab...24c`/`1979da53...7e9`/`6175b068...061`。全仓fresh `166 passed`，architecture guard无hard violation。
- 正式训练合同为fresh normal-positive-only，scheduler horizon 1,200；先到step500并每100步保存。若loss/validation未饱和，避免每100步重载模型，改为每次约300步exact-resume至800、1100，累计训练wall约60分钟后无论是否充分均封存曲线并停止追加。

## 四卡bias-restored Action-Memory新Goal启动（2026-07-23）

- owner将本轮设备权限明确收窄为2026-07-28前只用物理GPU `0,1,2,3`；GPU4–7即使空闲也不进入训练、评测或额外controller角色。此前8卡profile与bias-free checkpoint只作provenance，不作为新trajectory的resume初态。
- 当前模型拓扑保持冻结PaliGemma prefix、16个Action-Expert memory tokens、encoder-only Meta-LoRA、变长temporal/layer/slot聚合和完整task-specific rank16 LoRA不变。仅把conditional path内部Linear/MHA/factor-head普通bias恢复；final factor-head bias和weight都从零初始化，fresh adapter仍为functional identity，且不存在独立公共LoRA支路。
- Writer构造字段已收敛到单一代码合同，AS训练、checkpoint评测和RL runtime共同使用；同时修复RL新Action-Memory Writer初始化漏传冻结`action_in_proj`的问题。当前只影响未来fresh RL启动，不修改任何旧artifact。
- 在读取validation action值前封存`pi05_validation_functional_loss_panel_v1`：8个validation tasks各8个teacher-video groups、每组8个不同episode action queries，共512 rows/checkpoint；query seed、task-equal aggregation和“closed-loop覆盖loss”的冲突规则均固定。新增评测只读取validation label算loss，不向Writer或optimizer暴露，不读取test action/video值。
- 四卡AS预合同使用per-rank batch16/global64、每100步checkpoint；首个候选阶段到step3200，对应204,800 action queries，和rank128 Source-SFT step400的query量相同，但只是首个判断点。可exact-resume到4800/6400，最终是否继续由validation loss与完整closed-loop峰后持续下降证据决定，不设wall-clock上限。
- fresh全仓验证为`169 passed`；配置/面板加载、CLI、compile和diff check通过。architecture guard为`REVIEW`但无hard violation，新增validation owner位于`ember.writer`且没有第二套policy evaluator。下一动作是live四卡/storage preflight与真实profile，profile后再封存并启动fresh formal AS。
- live preflight确认物理GPU0–3均为0 MiB/0%且无compute process；GPU4–7属于其他用户的活动作业，未读取其任务状态以外的数据也未干扰。个人占用`275,195,838,464` bytes，profile与后续checkpoint峰值预算低于500GB cap。
- bias-restored四卡profile root为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_profile_r4_b16_580793a_20260723`：per-rank batch16/global64，两步max-rank wall为`3.8103/1.9300s`，loss与gradient全部finite；peak allocated/reserved为`75,992,039,424/78,871,789,568` bytes。
- 恢复bias后的Writer为`10,119,297` trainable parameters，是rank128 Source-SFT `10,297,344`的`98.27%`。run-contract/metrics/summary SHA256为`819bbbb6...45a1`/`19b8fd85...de28`/`950ff995...3ef`；四rank均绑定NUMA0对应GPU-local CPU集合，GPU0额外CUDA角色为0。正式四卡AS合同据此封存。

## bias-restored四卡AS正式首阶段launch contract（2026-07-23）

- canonical workspace为`/data/ymdai/worktrees/EMBER-as-valdiag-r6`，代码/config必须clean且等于launch时`origin/main`。fresh run不继承任何bias-free、v1/v2/v3或profile权重；冻结source仍为`pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000` raw policy，数据为sealed 24 train tasks，tokenizer与normalization沿用source authority。
- 规模为4-way DDP、物理GPU `0,1,2,3`、每卡一个同角色policy CUDA process、per-rank batch16/global64、scheduler horizon6400；首段fresh运行到step3200，即204,800 action queries，每100步原子保存。若需继续，只允许从该root的完整step3200 checkpoint在同一合同下exact-resume到4800/6400。
- output为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723`，log为`/data/ymdai/logs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723.log`，tmux session为`ember_as_bias_r4_s3200`。按profile每checkpoint约124.4MB，32个checkpoint连同临时原子副本预计低于4.5GB；preflight个人占用275.2GB，因此预计峰值低于280GB和500GB cap。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_as_writer.py --config configs/pi05_as_writer_action_memory_v1.json --mode formal --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723 --stop-after-step 3200 --num-workers 2 --log-every 10`。
- acceptance/monitoring：首个optimizer step必须loss/gradient finite；四rank GPU/NUMA对称且GPU0无额外CUDA角色；metrics cursor连续、每100-step checkpoint完整；不得使用GPU4–7或读取validation/test action用于更新。checkpoint筛选另用预封存512-row validation-loss panel，最终以完整8×50 closed-loop success为authority。

## rank128 Source-SFT四卡上限探索准备（2026-07-23）

- 在AS正式进程运行期间，只在隔离worktree`/data/ymdai/worktrees/EMBER-sft-ceiling-r4`修改尚未启动的Source-SFT owner；没有改动AS已import的代码、config或output。
- Source-SFT runtime不再硬编码8 ranks，而是从profile/formal sealed contract读取world size。当前profile待物理GPU0–3可用后以4 ranks×batch128运行，从而保持旧比较的global batch512；此前8×batch64 profile与训练只作provenance，不能跨world-size exact-resume。
- rank128 fresh formal预合同保留原100-step warmup和800-step cosine decay，使前800步LR轨迹定义不变；最大horizon扩至2400，只在800后增加`1e-5`低LR tail。每100步保留checkpoint，首段到800，只有validation仍未建立真实峰值与持续峰后下降时才exact-resume到1600/2400；owner明确不设wall-clock上限。
- 当前只封存“pending four-rank profile”，不得直接formal启动。相关Source-SFT/LoRA聚焦测试`16 passed`，architecture guard无hard violation；AS释放四卡后先跑真实batch128 profile，再写入吞吐/显存/hash并封存正式合同。
- Source-SFT现复用同一封存512-row panel并支持两种互补生命周期：active formal训练在每个checkpoint后用常驻policy原地测loss、恢复完整RNG后继续；独立只读入口仅用于历史checkpoint backfill或训练进程结束后的复核，不执行训练、也不构成第二套policy evaluator。两者共享panel manifest、task-equal summary和Source-SFT adapter owner。

## checkpoint内联validation-loss监控（2026-07-23）

- owner澄清validation functional loss应在训练进程保持π0.5、Writer和显存常驻时原地计算，而不是每个checkpoint暂停并重载模型。bias-restored AS正式run已在完整step300 checkpoint后做一次必要代码升级；step100/200/300均已原子保存，训练轨迹未回滚。
- canonical AS runner现在只在development formal checkpoint后切换`eval/inference_mode`，换入封存512-row panel，四rank各处理16个`task×video-group`，随后恢复Writer train mode和进入monitor前的Python/NumPy/CPU/CUDA RNG。validation actions不进入Writer、optimizer或反向传播，policy/Writer不复制、不重载，训练更新合同不变。
- 每个step目录保存512 raw losses、逐task等权mean/std、相邻checkpoint loss delta、wall和checkpoint hash；连续下降用于继续训练，平台或连续多点回升用于收缩候选区间。单点波动不触发早停，完整8×50 closed-loop success仍覆盖loss诊断选择最终best。
- 独立历史backfill一次模型加载同时复核step100/200/300/400/500，task-balanced loss依次为`0.135237/0.138363/0.134698/0.141123/0.134224`；其中step300/400/500共1,536条逐query loss与训练进程内结果逐值完全相同。backfill只作旧checkpoint补齐，后续checkpoint全部由常驻训练进程原地测量。
- step500 exact-resume后，step600/700/800 loss为`0.138690/0.139285/0.140583`，相对step500形成连续三点上升；同期100-step train-loss mean从step401–500的`0.117381`降至step601–700的`0.111555`，显示train/validation开始分叉。run已在完整step800 checkpoint和在线validation后干净暂停，没有未checkpointed metrics；step500是当前loss谷底，但最终best仍待closed-loop确认。

## bias-restored AS closed-loop候选launch contract（2026-07-23）

- canonical workspace为`/data/ymdai/worktrees/EMBER-as-valdiag-r6`，branch `codex/as-valdiag-r6`，launch前必须clean且等于`origin/main`。上游训练root为`pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723`，source checkpoint、tokenizer、sealed validation split、fixed-50 states、policy/video seeds与旧正式AS评测完全相同；本轮只改变Writer checkpoint。
- val-loss用于收缩区间而不替代closed-loop：首批完整8-task×50 correct-video候选为谷底前step300与谷底step500，并行使用GPU `0,1`和`2,3`；每张物理卡统一5个persistent policy replicas，两个CPU-only launcher不在任何GPU上增加模型。完成后再测峰后肩部step800，确认loss趋势与success是否一致。
- 输出分别为`/data/ymdai/outputs/ember/pi05_action_memory_as_bias_val8x50_step0300_correct_18c9e9a_g01_r5_20260723`与`...step0500_correct_18c9e9a_g23_r5_20260723`，均为不存在的新root，禁止覆盖或混入bias-free结果。每个正式结果预计约3MiB，连同queue/log远低于当前258GiB个人占用和500GB cap。
- step300 exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src /data/ymdai/projects/EMBER/.venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --output-dir /data/ymdai/outputs/ember/pi05_action_memory_as_bias_val8x50_step0300_correct_18c9e9a_g01_r5_20260723 --role validation --mode formal --state-count 50 --replicas-per-gpu 5 --gpu-indices 0,1 --as-writer-config configs/pi05_as_writer_action_memory_v1.json --as-writer-checkpoint /data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723/checkpoints/step_00000300 --writer-video-data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --writer-video-condition correct`。
- step500 command仅将output、`--gpu-indices 2,3`与Writer checkpoint改为对应step500路径。两臂各400 rollouts，raw rows必须唯一、38个cost-balanced shards必须完整、10 workers均exit0；结果按相同task/init/policy/video seeds配对，任何partial queue只能按canonical resume恢复。

## bias-restored AS首轨迹closed-loop与scheduler修正（2026-07-23）

- decay-6400轨迹的step300/500/800完整8×50 correct-video结果已完成，分别为`62/400`、`77/400`、`80/400`，raw rows均为400且无worker error。step800用GPU0–3、每卡5 replicas和32个cost-balanced shards，wall `1282.51s`、吞吐`0.31189 rollouts/s`；动态队列先按实际四卡拆Long shards，再由全部workers接普通任务。
- step300/500/800 results SHA256依次为`3c2643cf...fa8334`、`db01087c...4fce6`、`f2ef8786...8af10`。step500与800 paired flips为27/30，说明两者闭环持平；functional val loss的连续回升适合收缩候选区间，但不能替代完整closed-loop选best。
- 审计发现四卡global batch64沿用了warmup100，却把旧八卡global batch128的decay1200错误拉长到6400。正确query-equivalent日程是warmup100/decay2400；现有step500–800长期停在peak LR附近，因此该首轨迹保留为scheduler-confounded provenance，不用于否定bias或Action-Memory架构。
- canonical配置现只修正scheduler/阶段合同：total/decay为2400，fresh首段step1200，必要时exact-resume到1800/2400；每100步仍由同一驻留模型原地运行封存512-row validation-loss panel。配置SHA256为`cf031d79b6273ba71f3b5969a491cd6cf5d13d9c17a85e989cfc1c2d8f6ac69f`；聚焦验证`15 passed`、全仓fresh验证`172 passed`，下一步commit/push与live preflight后在GPU0–3启动不存在的新root。

## query-equivalent四卡AS fresh launch contract（2026-07-23）

- scheduler-only修正已在commit `43608a7e8d48e0f03a425d32a6b9152c67dfff0f`验证并推送；训练架构、bias、source checkpoint、24-task数据、sampler、optimizer、loss和validation panel均未改变。新run必须fresh identity，不能从decay-6400轨迹resume。
- output为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723`，log为`/data/ymdai/logs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723.log`，tmux session为`ember_as_bias_qscaled_r4_s1200`；三者启动前必须不存在。4 ranks只绑定物理GPU0–3，每卡一个同角色训练进程；GPU4–7不进入visible set。
- 每100步保存checkpoint并在同一驻留进程运行512-row task-balanced validation loss；首段stop为1200。checkpoint按既有实测约124.4MB估算，12份加原子临时副本、metrics和validation rows低于2GB，远低于500GB个人cap。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_as_writer.py --config configs/pi05_as_writer_action_memory_v1.json --mode formal --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723 --stop-after-step 1200 --num-workers 2 --log-every 10`。

## query-scaled四卡AS早停与closed-loop候选（2026-07-23）

- fresh run root为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723`。step100–800 validation loss为`0.135237/0.141384/0.135191/0.134058/0.134964/0.135579/0.141342/0.139462`；step400后形成持续上升区间，故在step800完整checkpoint/val后早停。
- step800训练elapsed为`1951.68s`，累计51,200 queries；最大allocated/reserved为`79,540,023,808/82,288,050,176` bytes，loss/gradient均finite。run contract、当前完整metrics和validation metrics SHA256依次为`b5053300...e978347`、`2279f960...18461e`、`4d5936b0...722079`；latest checkpoint仍严格指向step800。
- 中断前step801–809已完成并保留在metrics中，但没有checkpoint也不进入候选；不删除或改写该证据。step300/400/500/600各用两张物理GPU、每卡5个persistent replicas，step800用四卡同样每卡5 replicas；所有run均有400条唯一rows、全部shards完成、workers exit0。
- step300/400/500/600/800完整correct-video success依次为`57/91/86/87/88`，results SHA256依次为`57ea1be8...e8179`、`07847743...99197`、`60b41c56...1b35d`、`6cb2d5af...4768`、`70b33252...74745`。对应wall为`2190.35/2135.77/2197.09/2146.22/1231.29s`；四卡step800有效吞吐`0.32486 rollouts/s`，两卡runs约`0.182–0.187`。
- step400相对300/500/600/800的paired `400-only/other-only`为`49/15`、`33/28`、`26/22`、`29/26`；只有300显著较差。val-loss最低点与真实峰值同在400，但五点总体相关性很弱，因此在线loss只负责趋势早停和候选筛选。该轨迹best `91/400`显著低于旧rank128 SFT的`122/400`（paired `25/56`，`p=7.52e-4`），下一步先完成SFT真实ceiling，再对AS做训练统计效率修正。

## rank128 Source-SFT四卡profile与正式首段合同（2026-07-23）

- profile root为`/data/ymdai/outputs/ember/pi05_source_sft_rank128_profile_r4_b128_22fae58_s4_20260723`；仅使用GPU0–3，一卡一rank，batch128/rank、global512。4步全部finite，稳定吞吐约`36.18 queries/s`，峰值allocated/reserved为`54.998/67.979GB`。
- profile artifact约61MB；正式run到step800保留8个约61MB checkpoint、在线validation rows及原子临时副本，峰值新增低于1GB。当前个人占用约259GB，远低于500GB cap。
- 配置已封存首段step800、每100步checkpoint与驻留512-query val-loss monitor；允许的后续stop为`1100/1400/1700/2000/2300/2400`。val-loss连续下降则续训，连续上升且train loss仍降则停止；单点不决策，完整8×50 closed-loop覆盖loss选择最终best。

## AS condition-balanced累计实现待profile（2026-07-23）

- canonical `as_step.py`现支持同一generated adapter跨多个policy microbatches复用；128 queries按`16×8`顺序执行、按chunk实际样本数加权loss/adapter-gradient，最后只反传一次Writer。新增聚焦测试覆盖尾部不等长切片和加权梯度等价性；既有normal/contrast owner、checkpoint和sampler cursor未分叉。
- owner澄清batch size不应成为方法门槛后，候选profile收敛为四卡每rank顺序2个独立conditions、每condition 16 queries，即8 conditions/128 global queries；这恢复旧八卡AS的逻辑训练单位，不再机械匹配rank128 SFT的global512。配置为`pending_condition_balanced_profile`，formal仍不能启动；SFT正式进程期间只做代码和CPU测试，不抢占或改写其输出。

## 四卡rank128 Source-SFT在线早停与候选评测修复（2026-07-23）

- fresh formal root为`/data/ymdai/outputs/ember/pi05_source_sft_rank128_ceiling_r4_b128_af658c4_s2400_20260723`。step100/200/300/400的task-balanced validation functional loss依次为`0.133067/0.133336/0.134167/0.137131`，同期100-step train-loss mean为`0.138862/0.117804/0.109806/0.106153`；validation连续且加速回升而train继续下降，故在完整step400 checkpoint/validation后暂停，后续可从step400 exact-resume。
- 首次候选评测在任何rollout前被旧机械合同拒绝：adapter检查仍硬编码formal world-size为8，并要求整个训练run已有最终`run_summary.json`，与当前四卡协议及“暂停→评中间checkpoint→必要时续训”冲突。
- 最小修复改为读取sealed formal config中的`expected_world_size`；仅`development + validation`允许用已完整发布、manifest校验通过的formal checkpoint在run summary尚不存在时评测，并显式记录`published_checkpoint_before_run_completion`。seen/final/test仍要求完成run summary；相关Source-SFT与evaluator回归`51 passed`。

## Source-SFT四卡step100–800结果与继续训练（2026-07-24）

- 四卡fresh rank128 SFT的完整8-task×50 validation曲线为step100/200/300/400/500/600/700/800=`81/95/68/78/94/99/108/97`。step700是该轨迹当前best，但600/700/800之间的paired差异均未形成明确峰后持续下降；旧八卡step400的`122/400`仍是全局SFT incumbent。
- 旧八卡8×64与当前四卡4×128都为global batch512，所以同一step的optimizer updates与总queries相同；两个step400 checkpoint也都实际记录`204,800` examples，每task覆盖范围仅相差一个128-example小批，确认训练量大体可比。每次更新内task小批数量不同只作为次要梯度方差信息。四卡step800的updates和queries已经是旧step400的两倍，不能因condition visits相同就称为等价点。后续仍记录拓扑与条件覆盖，但不再因GPU数量或batch变化机械缩放step或从零训练。
- 当前四卡run已从完整step800 checkpoint在相同四卡合同下exact-resume到step1100，首个新finite metric为step802、loss `0.0867007`、gradient norm `0.0237412`、吞吐`36.26 queries/s`。step900/1000/1100保存后仍以完整closed-loop决定是否继续；functional val loss只作微弱参考。
- AS累计实现最终采用每rank每optimizer update处理2个独立task/video conditions、每condition 16 queries；四卡每update合计8 conditions/128 queries，与旧八卡AS一致。此前每condition64/global512仅是未启动的SFT-batch-matched提案，已在owner澄清后退役。该设置用于恢复条件覆盖与梯度稳定性而非把batch设为科学门槛，真实profile后优先warm-start现有best。
- canonical AS runner新增显式`--initialize-writer-checkpoint`阶段初始化：可从已封存且source/authority/Writer/LoRA完全兼容的最佳Writer权重启动新优化阶段，同时重新初始化optimizer、scheduler、sampler和RNG；run contract记录源checkpoint、源step及三类hash，并明确标为warm-start而非exact-resume。这样GPU数或训练统计合同变化时无需重跑0→best，也不伪造跨合同exact-resume。旧qscaled step400 checkpoint已通过只读manifest/architecture/authority兼容检查；是否采用仍由真实低方差profile后决定。
- evaluator新增受控`6 replicas/GPU, OMP=1`运行profile；它只扩展同一dynamic-queue/persistent-policy owner，不改变task、seed、rows或结果聚合。按既有3 replicas约31GB估算6 replicas仍低于80GB，但必须由下一轮900/1000 checkpoint正式评测实测显存稳定性与有效rollout/s；若OOM或吞吐不优于5，立即退回5并保留failure/profile证据。

## Action-Memory temporal-RoPE快速实验合同（2026-07-24）

- owner要求快速验证新的时间聚合架构：保留有bias的冻结PaliGemma/Action Expert memory/Meta-LoRA/完整rank16 LoRA owner，只将乘性time gate和单一对称pool替换为实际采样位置上的1D RoPE、4个condition-only learned temporal memory queries；不加入transition token、order auxiliary loss、contrast或shared adapter。
- 真实GPU0–3 profile使用一rank一condition、16 queries/rank、global64且无梯度累计。两步loss/gradient均finite，step wall为`3.8037/1.9552s`，峰值allocated/reserved为`76,119,387,136/78,928,412,672` bytes；Writer参数`11,252,737`，为rank128 Source-SFT参数的`1.092781×`。
- fresh formal只训练500 optimizer steps，保存step400/500；两者使用同一paired 8-task×50 validation panel，随后只对observed-best做视频、单帧、倒序与打乱诊断。output为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_rope_mem4_native_r4_seed7_s0500_20260724`；两份checkpoint和原子临时副本预计新增低于0.5GB，启动前个人占用`278,289,612,800` bytes，峰值远低于500GB cap。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_as_writer.py --config configs/pi05_as_writer_action_memory_v1.json --mode formal --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data/ymdai/outputs/ember/pi05_as_writer_action_memory_rope_mem4_native_r4_seed7_s0500_20260724 --num-workers 2 --log-every 10`。

## temporal-RoPE Writer快速子任务完成（2026-07-24）

- 实现commit `182a038`：canonical bias-enabled Action-Memory temporal owner原位加入frame-index RoPE与4个condition-only memory queries；聚焦Writer tests为`41 passed`。四卡native-global64 profile通过后，fresh formal root `pi05_as_writer_action_memory_rope_mem4_native_r4_seed7_s0500_20260724`完成500 steps，body wall `1188.6s`，step400/500 checkpoint均保存完整四rank RNG/optimizer/scheduler/sampler state。
- step400/500在线512-query validation functional loss为`0.1364674/0.1369167`。正式8×50 correct-video validation使用commit `d2997ed`的`per_sample_lora_batched_replan`，不是旧`materialized_per_rollout_sequential_replan`。6 replicas/GPU在Writer视频编码阶段OOM且0 rows，作为failure evidence保留；稳定拓扑为4 replicas/GPU、8 envs/replica。
- step400 root `pi05_as_writer_rope_mem4_batched_val8x50_step0400_correct_d2997ed_g01_r4_20260724`：`108/400`，wall `1712.444s`，`0.233584 rollout/s`，results SHA256 `163b0df72a14523ada233a8f846c1eaac9fac2cc6db9df822af47f8a5cff6d81`。
- step500 root `pi05_as_writer_rope_mem4_batched_val8x50_step0500_correct_d2997ed_g23_r4_20260724`：`98/400`，wall `1764.352s`，`0.226712 rollout/s`，results SHA256 `f8e0e1b299ac2b468cd0f46945f27c39daa25b7da1e0908b3922be7c8a4087c0`。paired step400-only/step500-only为`24/14`，故冻结step400。
- evaluator优化将此前AS约`36.6min/0.182 rollout/s`改善为最佳run的`28.5min/0.234 rollout/s`；旧sequential失败目录无有效rows，不参与科学结果。优化commit `d2997ed`及诊断commit `85962bf`均已推送`origin/main`。
- step400视频/帧特异性artifact为`pi05_as_writer_rope_mem4_step0400_video_frame_specificity_85962bf_20260724/diagnostic.json`，SHA256 `4f8110c1bf719d2ff07b220b5965af8d98d818ad7ae5e85c95c3216dc03a9316`，wall `173.316s`。跨suite错误视频、同task另一demo、倒序、乱序的有效LoRA相对变化中位数分别为`0.2267/0.0403/0.00937/0.00699`；单首/中/末帧为`0.1745/0.1124/0.3339`。
- 快速子任务到此按owner要求停止：新Writer对视频任务内容有明确特异性，但对时间顺序仍近似不敏感；correct峰值`108/400`未超过rank128 Source-SFT incumbent `122/400`。本轮不启动contrast、额外AS训练、SFT或RL。

## Action-Forecast Writer跨session交接（2026-07-24）

- owner决定将实现和实验交给新的独立session；当前session停止改代码和GPU
  launch，没有产生半成品实现，也没有修改正在运行进程的import/config/output。
- 新的活动设计、参数预算、退役边界、profile矩阵、AS分段训练和RL cold-start
  接续合同已封存在`docs/action_forecast_writer_handoff.md`，并由根
  `AGENTS.md`与`docs/execution_brief.md`显式引用。
- 交接只读快照：main/`b78584ab05e7f639cf1c022fdf457b3a971d64e6`
  当时clean且等于origin/main；GPU0–3空闲，GPU4–7为其他用户进程；
  `/data/ymdai`占用`278,857,052,160` bytes。新session必须重新核验所有live
  状态，不能把该快照当launch许可。
- 最终交接审计确认旧tmux `ember_as_bias_r4_s3200`只剩空bash且没有训练/eval
  子进程，随后已删除该空session；main外15个历史worktree均clean且无活跃写
  进程，因此全部保留provenance。新session在main clean且无并发writer时直接
  使用main，不因历史worktree数量另造平行canonical路径。
- owner要求确认新session无需读取旧对话即可实现后，又对交接文档做逐层完整性
  审计：补齐了端到端tensor shapes、single-agentview信息墙、state-width128
  coordinate-query head、连续state token在PaliGemma文本state位置的插入方式、
  冻结backbone仍需保留的梯度路径、per-condition flow noise、Plan/Revision
  精确MLP/attention聚合、query identities、8个factor heads及真实输出宽度、
  长Long-shard动态调度、近似候选复测规则和已封存`122/400` SFT artifact。
- owner进一步纠正比较口径：`122/400`来自旧八卡rank128 SFT，不是四卡成绩。
  四卡step100–1100为`81/95/68/78/94/99/108/97/95/104/94`，best为step700
  `108/400`。当前AS硬比较是“不明显落后于108”，超过122为stretch。
- owner随后明确禁止把“多个峰后点略低”的判断套给Writer：AS/RL都必须在
  validation找到best，并在其后看到幅度非常明显、明显超过rollout噪声、由
  多个tasks贡献且独立panel复测后仍成立的下降趋势；否则继续训练。
- 最终仓库审计发现旧`docs/new_session_prompt.md`仍错误要求8卡、重做Phase A/
  source base、RL零warm-up和AS约2小时上限；现已原位替换为唯一Action-Forecast
  prompt。当前prompt明确source/SFT已封存、只用GPU0–3、AS/RL无总时限和上述
  强下降停止标准，并加入owner最新效率要求：仅做防止无效实验/不可恢复浪费的
  最小shape/gradient/identity/freeze/resume检查，通过后立即GPU profile/训练。

## Action-Forecast Writer实现与formal训练收口（2026-07-24）

- 新session已将旧Action-Memory活动owner原位替换为唯一Action-Forecast路径：
  imagined-state/PaliGemma融合、Writer内部VL和Action Meta-LoRA、每帧完整
  10-step flow action plans、同绝对时刻Plan/Revision tokens、变长temporal
  Transformer及单向LoRA query decoder均已接入AS训练、checkpoint、inference和
  canonical evaluator。旧source/config/schema/tests及独立specificity runner
  已删除；历史文档和artifact只保留provenance，不再是可执行入口。
- Writer真实训练参数为`10,161,217`，相当于rank128 Source-SFT
  `10,297,344`的`98.68%`；生成的public task adapter保持完整38-target
  rank16合同，共76 tensors、`1,287,168` scalars，初始化为严格functional
  identity，source policy trainable parameter count为0。
- GPU0–3真实profile已封存stride5、frame-microbatch32、每rank batch16。
  17-step长profile覆盖全部24个train tasks与1088 action queries，steady
  step中位数/p95为`6.1183/9.0442s`，吞吐中位数`10.4611 queries/s`，最大
  allocated/reserved为`67.08/70.18GB`。frame-microbatch64令rank1达到
  `80,821/81,920 MiB`并失去前进，已拒绝；owner决定不再扩测stride10。
- profile checkpoint完成step1→2 exact-resume；flow-noise cursor、sampler、
  optimizer/scheduler及各rank RNG均恢复，contract SHA256为
  `c7a3dc88ae840d386b9d825e6f71f2f9613fccf0f37adf85b29c5a577d0ecd68`。
  两组提交前focused tests分别为`30 passed`和`51 passed`，相关Python模块
  `py_compile`及`git diff --check`通过。
- formal AS配置现为四卡、每rank batch16、每75 steps checkpoint、每300 steps
  exact-resume segment。首段之后只先评测step150/300 correct-video；owner明确
  撤销前置的单卡最小顺序诊断，wrong/shuffled/reversed只能在充分训练、找到
  validation observed-best且验证明显稳健峰后下降之后，对最佳checkpoint执行。

## Action-Forecast AS首段与两阶段evaluator（2026-07-24）

- fresh formal root
  `/data/ymdai/outputs/ember/pi05_action_forecast_as_development_seed7_49cef59_r4_s5_fm32_b16_20260724`
  已完成step0→300：300条metrics、19,200 action queries、1,200独立video
  conditions，训练body wall `2022.169s`；step wall中位数/p95为
  `6.1408/9.1705s`，loss范围`0.089996–0.186392`、最大grad norm
  `1.845997`，峰值allocated/reserved为`67,092,966,912/
  70,185,385,984` bytes，全部finite。
- step75/150/225/300四个约124.8MB checkpoint均通过完整manifest与file SHA
  核验。512-query在线functional monitor依次为
  `0.137364/0.133570/0.137465/0.134575`，只作弱候选信号，不作为closed-loop
  selection或停止依据。
- step150耦合式correct screen仅作实现provenance：GPU0–1 r3和GPU2–3 r4各跑
  16 episodes，分别为`1/16`，execution window约`94.3/100.5s`，显存约
  `40/53GB`每卡；小分母不作科学解释，也不能用于确定rollout replicas。
- owner指出LoRA生成和rollout并发必须解耦、共同模型应尽量常驻复用。现已在
  唯一`evaluate_pi05.py`内实现两阶段cache/handoff：generator数量和batch独立，
  cache完成后原generator进程只释放Writer并保留source π0.5直接rollout，随后
  再启动额外rollout-only workers；相同cache可跨rollout replica profile复用。
- cache身份包含完整adapter/model/task-state与生成batch/grouping但排除rollout
  replicas；逐entry safetensors+evidence以目录原子发布，final manifest核验
  coverage、LoRA SHA及file SHA，完整400-entry panel估计约1.03GB。结果新增
  rollout-only吞吐，同时保留end-to-end wall。
- 结构门无hard violation（`REVIEW`仅保留既有大函数与可接受的新模块审阅项）；
  新逻辑收口在`src/ember/pi05_eval/`和`src/ember/writer/evaluation_*`，没有
  平行runner。全仓fresh验证为`177 passed`，`py_compile`、`git diff --check`
  通过。下一步commit/push后在GPU0–3分别profile生成batch与纯rollout replicas。

## Action-Forecast AS step300→600与正式validation（2026-07-24）

- formal AS从完整step300 checkpoint同合同exact-resume到step600；本段300
  optimizer steps消费19,200 queries和1,200 video conditions，body wall
  `1978.45s`，step wall中位数`6.0985s`。step375/450/525/600四个checkpoint
  均含Writer、optimizer/scheduler、sampler/data cursor与四rank RNG；累计到
  step600为38,400 queries和2,400 video conditions，loss/gradient均finite。
- step150/300/450/600的完整correct-video validation依次为
  `75/99/93/118`。step600逐任务为Long `13/2`、Goal `1/34`、Object
  `46/17`、Spatial `0/5`；32/32 shards、400 rows、118 successes、所有24
  workers exit0且无重试。results SHA256为
  `bf3c98dc9a9df0e067b6589d7627b02863197528555ac6c11964799dfd7733be`。
- step450→600 paired为450-only `29`、600-only `54`（exact
  `p≈0.00804`），多个tasks共同贡献净提升。step600成为新observed-best并超过
  四卡Source-SFT的`108/400`；尚未出现任何峰后下降，故不做特异性诊断、不停训。
- commit `493917e`将cache identity原位改为可见视频级去重，并让aliases复用同一
  LoRA tensor与Writer生成随机流；全仓fresh测试`182 passed`。正式400-episode
  panel现为259个唯一LoRA和141个aliases，生成约53–56秒；每卡6 replicas稳定，
  step600 end-to-end/rollout-only吞吐分别为`0.45836/0.61045 episode/s`。
- 启动下一段前GPU0–3全空闲、GPU4–7仍为其他用户进程且未进入visible set；
  main/`493917e` clean并等于origin/main，个人占用约285.6GB。现已从step600
  启动四卡exact-resume到step900，仍按75步保存并优先正式评测step750/900。

## Action-Forecast AS step600→900与继续训练（2026-07-24）

- 四卡从step600 exact-resume到900，300步body wall `1994.75s`，step wall
  中位数/p95为`6.1495/9.0615s`；本段mean functional loss `0.114818`、
  最大grad norm `0.360586`，峰值allocated/reserved为
  `67,084,895,744/70,176,997,376` bytes，全部finite。
- step675/750/825/900在线functional monitor为
  `0.135046/0.134811/0.134446/0.134562`，仅显示弱平台。step900 checkpoint
  逐文件SHA核验通过，累计57,600 queries、3,600 video conditions；24 tasks
  各150次视频访问且均覆盖全部50 videos，完整optimizer、sampler与四rank RNG
  可恢复。
- step750/900正式validation为`104/113`，均为32/32 shards、400 rows、24
  workers exit0。600→750下降14但paired `p≈0.0814`；900随后回升9，且600与
  900 paired `p≈0.6254`。因此step600仍是observed-best，但没有满足owner停止
  条件的峰后下降，不能做特异性诊断。
- step750/900 results SHA256分别为
  `584a5c2164b631eb96fc6d60589720ad4ad297626ac750548b78b953c664ea22`
  和`4c1d62d0b3fbc847b776cdbcce0558d502b12a70381fa7f72d0913112d32a1cf`。
  在GPU0–3实时空闲、GPU4–7仍隔离、个人占用约287.4GB时，已从step900启动
  下一段exact-resume到step1200，继续优先评测step1050/1200。

## Action-Forecast AS step900→1200与新observed-best（2026-07-24）

- 四卡从step900同合同exact-resume到1200；step975/1050/1125/1200在线
  functional monitor为`0.134745/0.134612/0.134946/0.134434`，只显示微弱
  摆动。step1200累计76,800 queries、4,800 video conditions，24 tasks各
  3,200 queries与200次视频访问且50/50 videos全覆盖；checkpoint全部文件SHA
  校验通过。
- step1050/1200正式8×50 correct-video validation为`117/125`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。step1200逐task为Long
  `6/3`、Goal `1/38`、Object `45/20`、Spatial `1/11`，7/8 tasks非零。
- step600→1050 paired flips为`31/30`、exact `p=1.0`；step600→1200为
  `31/38`、净`+7`；step1050→1200为`15/23`、净`+8`。step1200成为新的
  AS observed-best `125/400`，因此不存在持续峰后下降，不能停止或提前做
  wrong/shuffled/reversed诊断。
- step1050/1200 results SHA256为
  `b88303cbf2a170315a1d5523f58cb1b0b3346d4671a9e37f024a0dda23f339a7`
  和`c575591ba36d949578061aa164f59572fcd59c81952a3f301c4c66b4afd38dd0`；
  rollout-only吞吐为`0.62721/0.60638 episode/s`。两次均按视频级cache只
  生成259个唯一LoRA，generator batch100与每卡6 rollout replicas解耦。
- GPU0–3再次实时空闲、GPU4–7仍为其他用户进程且未触碰、个人占用约289.8GB
  时，已从step1200 exact-resume到1500；仍按75步保存并正式评测
  step1350/1500。若没有出现明显、多个tasks共同贡献且独立复测成立的峰后
  下降，继续下一段而不设总wall-clock上限。

## Action-Forecast AS step1200→1500与继续探索（2026-07-24）

- 四卡从step1200同合同exact-resume到1500，本段wall `2019.83s`；
  step1275/1350/1425/1500在线functional monitor为
  `0.135102/0.134932/0.134596/0.135012`，仍只是弱平台。step1500累计
  96,000 queries、6,000 video conditions，24 tasks各4,000 queries与250次
  视频访问且50/50 videos全覆盖；最终checkpoint逐文件SHA验证通过。
- step1350/1500正式8×50 correct-video validation为`120/119`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `10/2, 8/1`、Goal `0/32, 1/33`、Object `43/19, 43/18`、Spatial
  `0/14, 0/15`。
- 相对step1200，1350/1500的paired净差仅`-5/-6`，exact
  `p≈0.6029/0.4614`；1350→1500净差`-1`、`p=1.0`。因此
  `125→120→119`只是两个略低且彼此持平的点，不满足强峰后下降停止条件。
- step1350/1500 results SHA256为
  `edf5b889eb4d6fdc0da9554966f97e8f9e5417cae250597526b2da7336337327`
  和`1a9232906b30d1d2ae679d8b726f332af5279aaff4a8e2ea1d8873981c035cc5`；
  rollout-only吞吐为`0.60530/0.63953 episode/s`。r6完成两次全panel，
  step1350瞬时显存约80.6GiB但未OOM。
- GPU0–3实时空闲、GPU4–7仍隔离、个人占用约291.6GB时，已从step1500
  exact-resume到1800；继续正式评测step1650/1800，仍不提前做specificity。

## Action-Forecast AS step1500→1800与继续充分探索（2026-07-24）

- 四卡从step1500同合同exact-resume到1800，本段wall `2051.35s`；
  step1575/1650/1725/1800在线functional monitor为
  `0.135525/0.134902/0.134651/0.134745`，仍是弱平台。step1800累计
  115,200 queries、7,200 video conditions，24 tasks各4,800 queries与300次
  视频访问且50/50 videos全覆盖；完整checkpoint和四rank恢复状态已核验。
- step1650/1800正式8×50 correct-video validation为`120/114`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `6/3, 4/2`、Goal `0/33, 1/34`、Object `43/21, 45/17`、Spatial
  `0/14, 0/11`。
- 相对step1200，1650/1800 paired为`29/24`与`31/20`，净`-5/-11`，
  exact `p≈0.5831/0.1608`；1650→1800为`30/24`，净`-6`、
  `p≈0.4966`。该幅度未远超400-rollout噪声，任务方向也混合，不能把
  `125→120→119→120→114`解释成充分确认的峰后下降。
- step1650/1800 results SHA256为
  `e800361b3bcdf57d57f39f635b20136f043a73d80197560098f0e087b5c35f9a`
  和`5c0de70f6b75c63d332e6e6e35ece5f2f4a57041cf364111123b6d73f61654d3`；
  rollout-only吞吐为`0.61633/0.61129 episode/s`。两次均以batch100生成
  259个唯一视频LoRA，随后每卡6 replicas稳定完成。
- GPU0–3再次实时空闲、GPU4–7仍为其他用户进程且未触碰、个人占用约274GB
  时，已从step1800 exact-resume到2100；下一步正式评测step1950/2100，
  specificity继续推迟。

## Action-Forecast AS step1800→2100与再次续训（2026-07-25）

- 四卡从step1800同合同exact-resume到2100，本段wall `2033.67s`；
  step1875/1950/2025/2100在线functional monitor为
  `0.134469/0.134724/0.135176/0.134929`，仍只有弱摆动。step2100累计
  134,400 queries、8,400 video conditions，24 tasks各5,600 queries与350次
  视频访问且50/50 videos全覆盖；最终checkpoint文件逐SHA与manifest一致。
- step1950/2100正式8×50 correct-video validation为`110/114`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `4/0, 4/0`、Goal `1/28, 1/32`、Object `45/19, 44/14`、Spatial
  `0/13, 0/19`。
- step1200→1950 paired为`34/19`，净`-15`、exact `p≈0.0534`，但主要由
  Goal-6贡献；step1200→2100为`36/25`，净`-11`、`p≈0.2000`。
  step1800→2100恰为`28/28`、净`0`，1950→2100反而净`+4`。所以
  `125→…→114→110→114`仍未建立明显、持续、多task共同贡献的峰后下降。
- step1950/2100 results SHA256为
  `c62e75973b8196e4e6052cecde8e0add00dd948f0536385ac5be44d0a158a576`
  和`934382c211027c3b6407b46898e65f708dfda136a1bc6cbef8a60f18cacf3905`；
  rollout-only吞吐为`0.61188/0.61723 episode/s`。两次均以batch100生成
  259个唯一视频LoRA，随后每卡6 replicas稳定完成。
- GPU0–3再次实时空闲、GPU4–7持续隔离、个人占用约276GB时，已从step2100
  exact-resume到2400；继续正式评测step2250/2400，仍不提前做specificity。

## Action-Forecast AS step2100→2400与峰值平台回访（2026-07-25）

- 四卡从step2100同合同exact-resume到2400，本段wall `2037.47s`；
  step2175/2250/2325/2400在线functional monitor为
  `0.135174/0.134400/0.134857/0.134759`，仍是弱平台。step2400累计
  153,600 queries、9,600 video conditions，24 tasks各6,400 queries与400次
  视频访问且50/50 videos全覆盖；最终checkpoint完整保存。
- step2250/2400正式8×50 correct-video validation为`123/111`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `5/3, 8/0`、Goal `0/34, 0/36`、Object `45/20, 43/18`、Spatial
  `1/15, 0/6`。
- step1200→2250 paired为`30/28`，净`-2`、exact `p≈0.8957`，2250实质
  追平observed-best；step1950→2250反而净`+13`。step1200→2400为
  `32/18`、净`-14`、`p≈0.0649`，2250→2400为`32/20`、净`-12`、
  `p≈0.1263`，但这是接近峰值后的一次单点回落，且task方向混合，不能停止。
- step2250/2400 results SHA256为
  `35ff55e3f8c2a5f8ed8885cf2a335862879b255189907a098e63d7ad61525655`
  和`f5c9a77b40048e6826a8b667c887e6d796c14f71be68fe9d9a249329bdc036df`；
  rollout-only吞吐为`0.60369/0.61011 episode/s`。两次均以batch100生成
  259个唯一视频LoRA，随后每卡6 replicas稳定完成。
- GPU0–3再次实时空闲、GPU4–7持续隔离、个人占用约277GB时，已从step2400
  exact-resume到2700；继续正式评测step2550/2700，specificity继续推迟。

## Action-Forecast AS observed-best特异性门与顺序修正profile（2026-07-25）

- step2550正式correct-video完成`124/400`，再次回到step1200的`125/400`
  峰值平台；step2700 checkpoint完整存在但尚未评测。owner随后要求先对当前
  最高AS做视频特异性，若通过再推进RL。
- step1200四个同seed、同state、同policy-RNG的8×50 arms全部完成32/32 shards、
  400 rows、24 workers exit0且无重试：correct/cross-suite-wrong/shuffled/
  reversed=`125/67/121/124`。correct-vs-wrong paired为`71/13`、
  `p=7.8639e-11`，内容特异性通过；correct-vs-shuffled/reversed分别为
  `17/13`、`15/14`，顺序特异性失败。故没有越过RL硬门槛。
- canonical `src/ember/writer/as_step.py`现原位支持一个最小order-contrast训练
  mode：正例与shuffle/reverse负例共享物理action batch、policy language与
  Writer flow noise，两个functional forward串行执行以保持峰值显存；负例只在
  loss低于`correct+0.01`时施加`-0.5`梯度。source policy仍为0 trainable，
  Writer输入与禁入信息不变，没有新增runner或恢复Action-Memory。
- 从step1200 Writer权重、fresh optimizer/scheduler/RNG启动的四卡2-step真实
  profile已完成；batch16/rank、frame-microbatch32，双forward全局
  128 policy samples/step，峰值allocated/reserved为
  `67,077,086,720/69,250,056,192` bytes，首步/第二步
  `19.4614/11.6213s`，无OOM或非finite。focused tests为`13 passed`，
  config解析、`py_compile`和`git diff --check`通过。配置已封为正式首段
  300 steps；训练后只对新轨迹的validation候选选best，再在best上复测完整
  四arm特异性，通过前不启动RL。

## Action-Forecast Writer v2实现与正式前检查（2026-07-25）

- 按owner最新口径原位完成28-state-token、directed Revision bounded-gate、
  content-only LoRA decoder；删除order-contrast活动配置与`as_step.py`分支，
  schema/checkpoint/config统一升级到v2。focused tests `26 passed`，
  `py_compile`和`git diff --check`通过。
- Revision反事实诊断完成8 tasks×2 videos，未读actions/reward/outcome：
  新合成time-centered reversed/shuffled相对L2中位数
  `0.3554/0.2418`，旧Revision为`0.0281/0.0316`。
- GPU0–3实时空闲、个人占用`301,090,004,992` bytes、总盘可用约3.06TB时，
  运行fresh step1后从
  `/data/ymdai/outputs/ember/pi05_action_forecast_v2_profile_resume_r4_s5_fm32_b16_20260725/checkpoints/step_00000001`
  exact-resume到step2。contract SHA256为
  `5afbb65786f70ab67c131a78ca59959fde3284dd9bbbbb4932f35eec1ddc83a6`；
  四rank state、Writer、optimizer/scheduler与trainer state均在checkpoint
  manifest中逐文件封存。
- profile保持`stride=5`、`frame_microbatch_size=32`、
  batch16/rank；step2为`6.5025s`、全局`9.8424 queries/s`，峰值
  allocated/reserved为`67,088,471,040/69,235,376,128` bytes，无OOM或
  nonfinite。下一步提交并push唯一canonical代码/配置，再从fresh identity
  启动正式step0→600，checkpoint间隔75，完整评测step300/600。
