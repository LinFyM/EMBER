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
- 当前里程碑fresh验证为88 tests passed；py_compile、JSON/checksum与diff checks通过；architecture guard为`REVIEW`且无hard violation。ownership、增长理由与旧静态runner retirement已记录。
- 第一轮完整流程只跑一个training seed；不提前扩多seed或direct action-budget curve。
- 最新live `/data/ymdai`占用379,028,677,098 bytes。单checkpoint实测33,837,406,832 bytes；`keep_latest=1`的atomic替换峰值约446.70GB，低于500GB cap。

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

1. fresh运行完整tests、JSON/checksum/architecture checks，commit并push相机mask修复和唯一dynamic evaluator。
2. 从新push commit创建隔离clean worktree；fresh GPU/storage/process preflight后从generic base启动全新formal attempt3，绝不resume attempt1/2。
3. 训练等待期间推进π0.5 Writer/functional LoRA；final checkpoint产生后实测evaluator统一1/2/3 replicas的有效rollouts/s并选择拓扑。
4. 用选定evaluator快速screen全部40 target tasks；只有跨多个tasks出现真实成功才冻结共同base并进入Phase C–H。

## 历史边界

旧SmolVLA 70/10/10曾完成到旧Phase F并留下真实结果，但与当前π0.5、split、one-video和source-base合同不兼容。只能用作经验/provenance，不能复用checkpoint、normalization或runner。
