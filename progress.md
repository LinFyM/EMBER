# EMBER Progress and Handoff

最后更新：2026-07-21。

## 当前状态

- 长期Goal已建立且无token budget；不会因source base、Writer或任一局部阶段提前完成。
- 活动目标split仍为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；seal位于 `configs/libero_24_8_8_v1/`。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已下载，weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 fixed states上为 `0/400`。400 rows唯一、全部到suite horizon，result seal SHA256 `c78e92e9...20c2`；该结果不评价EMBER。
- Phase A source audit、71-task manifest、source-only normalization、pinned official recipe与hash seal已完成；cost-balanced dynamic evaluator代码和fail-closed contracts已完成，真实1/2/3 replicas rollout/s profile待final source checkpoint。
- canonical π0.5 source-base full-SFT runner、atomic checkpoint与exact-resume机制已完成；相机mask修正后的真实8卡m32+EMA smoke为47.75 examples/s、71.18GB reserved/卡，formal配置锁定为global batch256、30,000 steps。
- formal attempt1因NUMA affinity缺失在step12终止；attempt2因显式zero右腕被LeRobot误标`mask=true`而在step316终止。两者均无checkpoint、failure packet已封存且永不resume；修正后的训练/评测都通过missing feature key得到OpenPI规定的zero image + `mask=false`。
- 当前里程碑fresh验证为112 tests passed；compileall与diff checks通过；architecture guard为`REVIEW`且无hard violation。ownership、增长理由与旧cache/cold-start/inference路径retirement已记录。
- 第一轮完整流程只跑一个training seed；不提前扩多seed或direct action-budget curve。
- 最新live `/data/ymdai`占用约412.84GB，已包含LIBERO-90和完整target40。按单checkpoint实测33,837,406,832 bytes计算，旧checkpoint+新partial的替换峰值连同约1.4GB target feature cache预计约482GB，低于500GB cap；不得再引入重复dataset/model/cache副本。

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

## Canonical cost-balanced evaluator（实现完成，profile待checkpoint）

- `scripts/evaluate_pi05.py`取代并删除旧静态`evaluate_pi05_base.py`，是唯一活动π0.5目标评测入口；不保留双runner。
- `pi05_eval_contract.py`拥有authority/final-EMA/test-state门，`pi05_eval_queue.py`拥有cost-balanced SQLite WAL队列，`pi05_evaluation.py`拥有persistent policy/env与official rollout，`pi05_eval_results.py`单独拥有worker拓扑证据和strict aggregation；拆分是为隔离调度、runtime与不可变结果故障边界，不是平行runner。
- state shards按`count × horizon`估算cost并动态work-steal；8 GPUs上统一1/2/3 replicas，launcher CPU-only，GPU0无额外CUDA角色。policy noise按`(seed,suite,task,state,replan)`确定性独立，不受batch或worker顺序影响。
- launcher lock覆盖active-worker audit、queue recovery、preflight与spawn；partial spawn/failure只回收本launcher PIDs并封存logs/jobs/hashes。正式吞吐包含worker spawn、model load和首次env/EGL，另报raw shard window。
- formal/screen拒绝非当前完整source config、非final step30000 EMA、相机interface漂移、test init hash漂移及同大小model/tokenizer篡改；aggregate交叉核对raw rows、DB counts、producer、8×replica topology、GPU UUID和NUMA。

## Formal source-base attempt 3：运行中（2026-07-21）

- canonical workspace/commit：`/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055`，branch `codex/pi05-source-formal-aa8b055`，commit `aa8b0556619889480d8d9c129ea2f54af26c9d06`；启动时clean且等于`origin/main`。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=8 scripts/train_source_base.py --config configs/pi05_source_base_v1.json --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90 --foundation-path /data/ymdai/ember_data/lerobot_pi05_base --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --output-dir /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_aa8b055_20260721 --mode formal --num-workers 4`。
- output/log/tmux：`/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_aa8b055_20260721`、同名`/data/ymdai/logs/ember/*.log`、session `ember_pi05_source_aa8b055`。这是从generic base fresh启动的新root，绝不resume attempt1/2。
- launch authorities：source config SHA256 `366a5e12...ee8`、recipe `4c537067...1734`、run contract file `6db50495...1bec`；foundation weights和71个source HDF5均在formal启动时完整重算SHA并通过，right wrist合同为missing feature key→zero padding + `image_mask=false`。
- scale/topology：71×50 episodes，30,000 steps×global batch256；8个同角色rank PID 1264369–1264376，rank0–3绑定NUMA0、rank4–7绑定NUMA1。step7时每卡约69.0–69.2GB、GPU UTL 100%，loss/gradient finite，稳态约47.5 examples/s。
- live preflight：8卡启动前均0MiB且无compute apps；driver570.158.01、CUDA12.8、torch2.11.0+cu128。`/data/ymdai`为379,033,156,799 bytes，按实测33,837,406,832-byte checkpoint的atomic双态峰值估计446,707,970,463 bytes，低于500GB cap；`/data`可用约3.059TB。
- checkpoint每5,000 steps原子发布、只保留最新；完成前不作source competence结论。首个checkpoint预计约7.4小时，完整30k按当前吞吐约44.7小时；等待期间只在另一worktree推进不改其import/config/output的后续代码。
- step48 fresh live check仍为8卡各一个同角色PID、约69.0–69.2GB、100% GPU；loss/gradient finite，稳态约47.2 examples/s。该检查只证明运行健康，不作行为结论。

## π0.5 LoRA / one-video Writer core里程碑（2026-07-21）

- 新活动合同`configs/pi05_lora_v1.json`绑定generic `lerobot/pi05_base`完整revision/weights/config、当前source config与recipe hashes；文件SHA256 `1dcf58f7...cb07`，canonical contract SHA256 `42d5919e...94dd7`。
- 真实foundation safetensors metadata和meta-device `PI05Pytorch`结构均核验38个精确Linear targets；rank16得到76 tensors、1,287,168 parameters。没有加入state/time projections，也没有沿用旧37-target Smol合同。
- 通用`lora.py`只保留PEFT mechanics协议；活动科学拓扑由`pi05_lora.py`单独fail-close加载。这样旧历史imports不进入π0.5 runner，也没有第二个活动训练/评测入口。
- `CompleteLoRAWriter`现在在活动边界只接受一个非空video (`offsets=[0,L]`)；LoRA template和输出逐tensor保留真实BF16/FP32 dtype。functional action loss调用真实PI05 `forward(batch)`，不再传旧接口`noise/time`。
- fresh验证：真实checkpoint target metadata通过；mixed-dtype functional/copy parity精确通过；全套`91 passed`、compileall、diff/checksum通过。architecture guard为`REVIEW`、无hard violation；review仅来自既有Writer构造函数长度和目录密度，owner/lifecycle如上。

## Target40 seal、PI05 feature cache与AS-Writer owner（2026-07-21）

- target40下载已完成且未复制现有cache：四suite各10个HDF5，总计33,784,856,577 bytes。metadata-only seal为40 tasks/2,000 episodes/338,575 frames，24/8/8 IDs逐项一致；manifest/checksum均通过，manifest SHA256 `1b28547f...049d`。
- `configs/pi05_writer_feature_cache_v1.json` SHA256 `3e3a8ea7...429e`，只授权development train+validation视频，禁止test video与任意trajectory action/state/reward/terminal读取；当前profile状态`pending_source_base`，formal cache必须先完成final source EMA上的真实8卡profile。
- `configs/pi05_as_writer_v1.json` SHA256 `971cac43...f807`，明确24 train actions、one-video input、independent sampler/video seeds、frozen source normalization及≤120分钟正式wall-clock；formal状态`pending_profile`，当前4-step/batch1仅为未来mechanics profile默认值。
- AS训练已原位替换旧Smol cold-start owner：每rank每step由task-balanced action sampler给出task/visit，再由独立teacher-video schedule选一条demo；`WriterFeatureStore.load_one_video`只暴露pure language、该episode的video features和`[0,L]` offsets。policy、base与encoder冻结，只有shared Writer DDP更新。
- checkpoint先验证canonical manifest和全部file SHA再读取optimizer/RNG pickle，交叉核对manifest/trainer/rank cursor，metrics JSONL按checkpoint cursor隔离orphan rows；formal最终coverage由launch total-step自动推导，调用者不能关闭。rank写盘/发布失败会跨8 ranks一致传播，避免barrier死锁。
- 删除`cache_writer_features.py`和`train_writer_cold_start.py`；相关活动测试改为PI05 schema。fresh全仓`107 passed`，compileall、config SHA和target checksum通过；architecture guard结果`REVIEW`且hard violations为空。review增长理由与retirement trigger见ownership段。
- formal source-base attempt3在上述工作之外的隔离worktree继续健康运行；最近只读观测step1100，8卡各约69GB且97–100%利用率，loss/gradient finite、约46.5–47.4 examples/s。此状态不构成source competence或行为结果。

## PI05 Writer evaluation与wrong-video机械证据（2026-07-21）

- 没有新增平行runner：`evaluate_pi05.py`、`pi05_eval_contract.py`、`pi05_evaluation.py`和`pi05_eval_results.py`原位支持source-base或AS-Writer arm，共用dynamic queue、persistent env、fixed-state rows、resume和aggregate。
- AS evaluator逐字段联锁source EMA、AS config/run/checkpoint、PI05 LoRA和feature cache；正式screen/formal只接受formal AS run，development cache可用于train/validation且会对test显式fail-close。未来test-open cache必须另行封存，不能把当前`test_video_values_read=0` cache冒充final cache。
- 每个rollout由顺序无关的哈希独立抽一条teacher video；correct与wrong arm共用selection seed/demo ordinal。wrong video按同split role跨suite双射，完整map、map SHA与condition进入run-contract hash。
- materialized backend每episode生成一次完整LoRA，并在该episode每次replan前安装同一state；不同活动env不会被错误合成普通同adapter batch。functional batched backend仍待final source产生后做真实rollouts/s profile。
- raw rows和aggregate保留checkpoint/cache/map/video/LoRA/timing证据，row validator会重算video seed、demo和map。fresh全仓`112 passed`，compileall、diff check通过，architecture guard为`REVIEW`且无hard violation；尚未因source未完成而运行GPU Writer smoke或产生科学结果。
- formal source-base attempt3最近只读观测step1100：8卡约69GB、97–100%利用率，loss/gradient finite、约46.5–47.4 examples/s；仍未到首个checkpoint，不作competence结论。

## PI05 shared Source-SFT owner与静态评测接入（2026-07-21）

- 新增development-only配置`configs/pi05_source_sft_development_v1.json`及checksum（SHA256 `32e927c...8a641`）。它只授权24 train tasks×50 action episodes，四suite各6个；final stage在该config内fail-close，validation选择后必须创建独立final authority，不能续接development LoRA。
- `ember.source_sft`成为单一owner：`contract.py`联锁target manifest、final source EMA、tokenizer/source normalization和PI05 LoRA；`training.py`在8个对称DDP ranks上只训练一套shared LoRA；`checkpoint.py`原子保存adapter-only exact-resume state；`inference.py`核验formal artifact并一次性安装静态adapter。薄入口为`scripts/train_source_sft.py`，没有复用旧Smol per-task direct runner。
- exact-resume state包含optimizer/scheduler、optimizer=micro cursor、metrics cursor、每rank RNG、sampler/data identity、DataLoader-derived worker seed与50-episode coverage；manifest在pickle前逐文件验SHA。development/final、source checkpoint、config或LoRA合同任一变化均拒绝resume/evaluation。
- canonical `evaluate_pi05.py`新增与AS-Writer互斥的Source-SFT参数；共享LoRA每worker只安装一次，随后继续普通multi-env batch和dynamic queue。raw rows记录固定LoRA state SHA；Source-SFT不生成Writer row，也不生成Writer correct/wrong pairing hash。
- fresh验证：Source-SFT与evaluator聚焦测试`23 passed`，全仓`119 passed`，compileall、config checksum、diff check通过；architecture guard为`REVIEW`且hard violations为空。review增长来自新增baseline的独立data-wall/training/checkpoint故障边界；旧`direct_lora*`的删除触发为本PI05路径完成真实8卡finite loss/grad与exact-resume smoke。
- formal source-base attempt3保持隔离且未被修改；最近只读观测step1450、371,200 global examples，8卡各约69GB、100%利用率，loss/gradient finite、约47.39 examples/s。仍未到step5000 checkpoint，不作source competence结论。

## PI05 reward core、zero-AS RL-Writer与test-only task-local合同（2026-07-21）

- 新增`configs/pi05_rl_writer_development_v1.json`（SHA256 `cde828db...0afd`）和`configs/pi05_task_local_rl_test_v1.json`（SHA256 `cd594a97...1550`）。前者formal状态仍为`pending_source_screen_and_real_profile`；后者所有正式budget保持0并写明`blocked_until_zero_interaction_test_and_test_open`，因此当前代码完成不会越过阶段信息墙。
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

## 已对齐的后续方法

- frozen source base：过滤后LIBERO-90×50 action-SFT，必要source LoRA merge，source-only normalization冻结；快速screen全部目标40 tasks，需开始在多个tasks有部分真实成功，不能只靠一个易task aggregate。
- AS-Writer：24 train/8 val开发，one video，video/action episode同task独立采样；单次训练≤约2小时，loss驱动稀疏val与早停。
- RL-Writer：随机Writer、零AS warm-up起步；无reward再极少warm-up，仍失败则关闭。
- Source-SFT：24/32 source tasks联合一套shared LoRA；独立val选最佳，不匹配AS steps/data。
- seen comparison：specification-only预声明覆盖四suites的source panel。
- wrong-video：直接另一suite，正确language/task/state/RNG不变。
- final：合并为32 source，单seed分别重训后先seen、再zero-interaction test。
- test-only RL：不碰validation；test task上训练identity/AS/RL Writer三臂到接近最佳，官方random resets，fixed50只fresh eval。
- direct oracle：最后使用8 test tasks×50 actions联合一套shared LoRA，不是per-task LoRA。
- optional：核心后有时间再做ViVLA；outer learning不阻塞。

## 当前后续动作

1. 保持formal attempt3的隔离worktree/config/output不变；继续实现task-local PI05 runtime与canonical fixed-50 adapter接入，但formal budget/test gate保持关闭，不提前读取validation/test outcome。
2. 继续周期性只读监测attempt3；首个step5000 checkpoint产生后校验manifest/hash/exact-resume状态，完成30k前不作source competence结论。
3. final checkpoint产生后先对Source-SFT/AS-Writer做真实8卡loss、freeze、OOM/吞吐及resume profile，再实测evaluator统一1/2/3 replicas的有效rollouts/s并选择唯一拓扑。
4. 用选定evaluator快速screen全部40 target tasks；只有跨多个tasks出现真实成功才冻结共同base并进入正式Phase C开发。

## 历史边界

旧SmolVLA 70/10/10曾完成到旧Phase F并留下真实结果，但与当前π0.5、split、one-video和source-base合同不兼容。只能用作经验/provenance，不能复用checkpoint、normalization或runner。
