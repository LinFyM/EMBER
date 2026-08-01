# EMBER Progress Ledger

最后更新：2026-08-01。

阅读规则：本文按时间顺序保留真实执行状态。早期段落中的“当前”“下一步”、
GPU范围和训练步长是当时快照；活动状态只取
`docs/active_session_handoff.md`、
`docs/action_forecast_writer_unified_causal_program_design.md`和本文顶部最新段落，
不能用旧快照覆盖后续owner决定。

## 2026-08-01 SPG门失败、内部根因与UCP设计

- UCP live seal以`c94f1c6`提交并push；新的detached formal worktree只读加载该
  commit，fresh macro0→200已在tmux `ember-ucp-formal-c94f1c6`启动。run root为
  `pi05_as_writer_ucp_rawfull24_decay400_formal_dev_r4_b20_seed7_c94f1c6_20260801`。
- 首macro wall`19.319s`，loss/grad/LR为`.153645/.014274/1.6667e-5`；严格24
  tasks、480 queries、24单视频条件、rank内long-first，10组raw-gradient
  allgather/completion/CUDA sync对应。四个rank仅驻留GPU4–7，训练继续到200。
- UCP live seal完成：detached `0d4c271`上最长105-frame、B20、四rank三macro
  连续通过，step wall `20.394/18.494/18.504s`，峰值allocated/reserved
  `77,127,082,496/83,345,014,784` bytes；72个视频条件、1,440 queries和全部
  checkpoint finite，step2起frontend/Program/reader/factor梯度均可达。
- canonical formal seed fresh0→1→exact-resume1→3通过；step1七个文件SHA不变，
  metrics/LR/task-video-query/RNG cursor连续，10个gradient chunks逐项
  gather/completion/CUDA sync。config已seal B20；下一步从新clean commit fresh
  启动macro0→200，不从smoke续训。
- UCP canonical CPU实现完成：`[X_f,A_f,G_(f+1)-G_f]`两层causal axial Program，
  单级normalized target/rank raw-value reader，无独立Core add和global mixer。
  semantic frontend/Program/reader/factor参数分别为
  `3,453,440/1,838,592/212,224/2,179,072`，总计`7,683,328`。
- CP active path已删除；每macro仍逐task求梯度并严格组成raw full24等权mean，
  overall和semantic_frontend/program/compiler/factor Gram仅作诊断。B20每visit覆盖
  全20个normalized-progress strata，episode边缘仍由permutation+jitter保持均匀。
- fresh config/checkpoint/evaluation schema均不兼容旧SPG；step0 identity、causal
  prefix、outgoing alignment、target/rank置换、padding、raw gradient world1/2和
  sampler exact resume通过；`CUDA_VISIBLE_DEVICES=''`全仓`203 passed`。
- 当前没有活动训练、评测或tmux；下一步提交现场seal并从新frozen commit只在
  GPU4–7启动formal macro0→200。

- SPG fresh macro0→200完成，macro50/100/150/200 correct400为
  `97/115/77/100`；按一小时门停止，不resume、不跑五臂。
- 四候选exact50几何、candidate gained/lost、Gradient Gram/energy和macro100
  refs2内部反事实完成。Program对order和wrong-video有强差异，但CoreReader与
  ProgramReader几乎不区分target/rank，最终LoRA近严格rank1/B-column相同；最早
  失效接口锁定为compiler，而非evidence extraction。
- 审计v5.2/v6 old/new及v6 slow/fast后，降低“full24或scheduler单独解释一切”
  的置信度；CP能消负pair但不能恢复近正交task innovations，functional loss与
  closed-loop仍错位。
- 完成现有B20前200 macros的96,000-query phase audit；长期无偏但单task visit
  覆盖方差可观，支持无偏20-strata estimator作为训练联合设计，而非新监督。
- 新write worktree
  `/data/ymdai/.codex/worktrees/EMBER-unified-program-534064a-20260801`已从clean
  `534064a`创建；设计authority为
  `docs/action_forecast_writer_unified_causal_program_design.md`。

## 2026-08-01 v5.2正式评测、五臂与内部分析封存

- paired correct400候选macro150/200/350/400完成，为`51/91/106/120`；选择
  single-checkpoint macro400，不做checkpoint融合。
- macro400正式五臂完成：`120/109/107/111/124`。逐task、逐suite、gained/lost
  state和严格pairing审计封存；四个控制臂都没有证明correct的行为优势。
- exact50 LoRA几何与五条件Core/Procedure/BA/fixed-action反事实完成；数值顺序
  信号可下传，但same-task视频方差缩至sample energy的`.6844%`且方向未与行为
  收益对齐。v5.2新recipe cell因此完成并停止，不再训练或评测。

## 2026-08-01 SPG最长profile与CP通信修复

- 初始最长profile macro1完成后，macro2在CP Gram交换处stall；只终止本任务
  tmux，未触碰任何外部进程。最小NCCL/Gram probe健康，逐phase trace把故障定位
  为分块all-gather仅入队、缺少逐chunk CUDA completion boundary。
- canonical CP实现加入每CUDA Gram chunk的stream completion，并记录
  all-gather/sync计数；CPU/Gloo路径保持0 sync。修复后原始105-frame/B20 profile
  三macro连续通过，step wall `20.536/18.578/18.546s`，峰值reserved
  `83,529,556,160` bytes。
- 72个单视频条件、1,440 queries全finite；每步24 tasks唯一且long-first，
  macro2起所有五个主块梯度可达。
- clean `f6d4876`上的formal-seed fresh0→1→exact-resume1→3已完成；step1六个
  状态文件与manifest在resume后哈希不变，metrics三行连续，72 videos/1,440
  queries、LR、task/video/RNG cursor和信息墙均核验。下一步提交seal并从最终
  frozen commit fresh启动macro0→200，不从profile/smoke warm-start。
- resume seal已push至`79fb7ee`；detached frozen worktree的正式fresh0→200已在
  tmux `ember-spg-cp24-79fb7ee`启动。首macro `19.431s`，24 tasks/480 queries/
  24 videos、B20、long-first、4 CUDA ranks和CP `13 gather=13 sync`全部通过。

## 2026-07-31 v5.2 task-complete macro400与候选启动

- frozen worktree commit `60f4508`上的正式root自然完成macro400；run contract
  SHA `152c0818...6088e`，run summary SHA `857f0111...ee66`，未提前截断或融合。
- `400` macros消费`192,000` action queries、`9,600` teacher-video conditions；
  wall `9695.1329s`，最后train/validation functional loss为
  `.09633848/.13686878`，全程finite，validation/test action reads为0。
- 一次live preflight确认main/origin/frozen均为clean `60f4508`、个人占用
  `350,451,040,256` bytes；只查询GPU4–7，未触碰0–3。随后tmux
  `ember-v52-candidates-60f4508`用GPU4/5/6/7分别启动macro150/200/350/400
  correct400，四个launcher存活且命令显式B-scale1、without-replacement、
  6 replicas/6 generators/batch16。

## 2026-07-31 SPG canonical CPU实现

- 独立写worktree `EMBER-spg-60f4508-20260731`已把canonical Writer切换为
  Semantic Program Grid，并删除活动`temporal.py`/v5.2 320-slot执行路径；历史
  与正在运行的v5.2由Git及独立frozen worktree保存。
- 新owner边界为`video_program.py`证据前端、`semantic_program.py` Core/Program、
  `program_compiler.py` target/rank compiler、`conflict_projection.py` CP-24，
  `model.py`只负责编排与public LoRA合同。
- 精确参数`10,633,216`；fresh incompatible schema/config为
  `configs/pi05_as_writer_semantic_program_grid_cp24_decay400_v1.json`。
- 全仓回归`201 passed in 26.18s`，`git diff --check`通过，architecture guard无
  hard violation。当前配置临时使用teacher-video seed172，只为确保首个三macro
  profile覆盖task38/demo36的最长105-frame视频；profile后必须改回formal seed
  `20260722`，再在同一干净commit完成fresh/exact-resume seal。

## 2026-07-31 当前交接

- main `799aa66`已恢复exact v5.2 topology并封存task-complete fast-decay400
  config；最长视频B20 profile和fresh0→1→exact-resume1→3通过。
- v5.2 step900的400套correct-video LoRA已完成生成和内部几何分析，未启动env
  或rollout；永久analysis SHA256为
  `9d816baadace851153415a06334efad6f9927bf334f014d5e8ae760be357e1af`。
- 结论：v5.2 q/v 16坐标能量均匀、建设性同向，effective近rank1不是负相消；
  其same-task视频创新明显高于v6既有估计。
- Coherent-Procedure/B-only residual已撤回；下一整体架构SPG已封存在
  `docs/action_forecast_writer_semantic_program_grid_design.md`。
- 新session先立即启动v5.2 task-complete macro0→200→400，然后充分审计仓库；
  无论v5.2结果好坏都实现SPG并进入每版一小时的持续根因迭代。
- 当前没有正式训练、rollout或tmux。v5.2 task-complete macro0→200→400属于
  新session第一实验，本session未启动。

## 当前状态

- 当前session-local Goal以工具实时状态为准；不会因source base、Writer或任一局部阶段提前完成长期主线。
- 活动目标split仍为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；seal位于 `configs/libero_24_8_8_v1/`。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已下载，weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 fixed states上为 `0/400`。400 rows唯一、全部到suite horizon，result seal SHA256 `c78e92e9...20c2`；该结果不评价EMBER。
- Phase A source audit、71-task manifest、source-only normalization、pinned official recipe与hash seal已完成；cost-balanced dynamic evaluator及fail-closed contracts已完成，真实1/2/3 replicas吞吐profile选择3 replicas/GPU。
- canonical π0.5 source-base full-SFT runner、atomic checkpoint与exact-resume机制已完成；fresh 1,000-step、333-step warmup、global batch256正式训练及step1000 raw/EMA checkpoint验证已完成，目标是轻量interface acquisition而非LIBERO-90收敛。
- formal attempt1因NUMA affinity缺失在step12终止；attempt2因显式zero右腕被LeRobot误标`mask=true`而在step316终止。两者均无checkpoint、failure packet已封存且永不resume；修正后的训练/评测都通过missing feature key得到OpenPI规定的zero image + `mask=false`。
- source/evaluator阶段的112-test里程碑是历史证据；v5初版曾完成`187 passed`，
  当前单视频合同切换后只运行防止无效正式实验所需的focused 25 tests并全部
  通过。没有用重复全仓仪式性校验延迟GPU训练。
- 第一轮完整流程只跑一个training seed；不提前扩多seed或direct action-budget curve。
- v5已完成step0→1800与正式五臂并因顺序行为门失败退役。当前focused Writer
  为v5.1 Language-Axial Semantic Core + Causal Action Procedure +
  Slot-Normalized Fusion；实时接手信息见`docs/active_session_handoff.md`。
- 2026-07-27交接审计时`/data/ymdai`占用约337.34GB，低于500GB cap；该值是
  live快照，新launch仍须重查。只清理已核验可再生的smoke/profile产物，不删除
  正式checkpoint、raw rows或来源不明文件。

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

## Action-Forecast Writer v1设计封存（历史，2026-07-24）

- owner决定将实现和实验交给新的独立session；当前session停止改代码和GPU
  launch，没有产生半成品实现，也没有修改正在运行进程的import/config/output。
- 当时的v1设计、参数预算、退役边界、profile矩阵、AS分段训练和RL cold-start
  口径已执行并由后续结果封存；相关旧辅助文档现已被2026-07-25 canonical
  design覆盖并删除，不再作为活动入口。
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
- 当时还发现旧启动提示错误要求8卡、重做Phase A/source base、RL零warm-up和
  AS约2小时上限；该提示随后又被后续架构演进反复覆盖，现已删除。当前口径只
  由根authority和`docs/action_forecast_writer_design.md`定义。

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

## Belief-v3实现、效率选择与正式启动前封存（2026-07-25）

- 唯一canonical Writer已升级为Belief-v3：一个absolute-time token内concat
  Plan128/Revision128；Revision比较所有更早covering forecasts与最新Plan；
  Temporal和LoRA query decoder均为content-only、zero-preserving路径。
- owner最终取消所有人工Revision强度尺度。活动公式为
  `Revision=stopgrad(raw source-normalized residual RMS)*RMSNorm(direction)`；
  routing strength也detach，`tau`与分位数只作诊断、不参与前向。
- 固定GPU0–3和stride5的效率profile选择frame-microbatch32、batch20/rank。
  12-step参考output为
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_profile_r4_s5_fm32_b20_20260725`；
  稳态中位`6.4942s`、`12.3188 global queries/s`。fm40较慢，fm48在首步前
  达到`81,153/81,920 MiB`且失去稳定前进。
- 最终raw-RMS实现的fresh+resume output为
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_rawrms_resume_r4_s5_fm32_b20_20260725`。
  fresh step1后从`checkpoints/step_00000001` exact-resume到step2；contract
  `352f7409d671d97399262b46afe0d415b4b6c145bcca66cbe43725474fa8e234`，
  resumed step `6.9184s`、`11.5634 queries/s`，峰值allocated/reserved
  `77,090,931,200/83,730,890,752` bytes，无OOM/nonfinite。
- checkpoint schema v3逐文件封存Writer、trainer/optimizer/scheduler及四rank
  RNG/sampler状态；flow-noise cursor从global visit 4准确推进到8。Writer
  `10,247,872` trainable parameters，source policy trainable count为0。
- focused测试、JSON解析、compile和diff检查通过后，下一步不再重做profile：
  提交/push当前唯一路径，实时复核GPU/storage，然后用同一配置从fresh identity
  一次连续训练0→600，每75步保存且不中途评测。
- step600顺序特异性先跑低成本内部数值诊断；只有normal/shuffled/reversed在
  effective LoRA等最终输出上已有明确、跨多个tasks/videos的稳定差异，才启动
  昂贵的paired validation arms。正常correct-video多checkpoint评测仍保留。

## Belief-v3正式step600与内部特异性failure packet（2026-07-25）

- commit `3363345`上的formal run
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_as_development_seed7_3363345_r4_s5_fm32_b20_s600_20260725`
  已一次连续完成0→600；run contract SHA256为
  `afbdea64b3b660baaa7576bc544c37f44575b9e001715ebea5191726a65a5071`，
  run-summary file SHA256为
  `110b45d521d61a4b35e933906d733e6a749a86379ad48a5fa2945d01bef2fc50`，
  wall `4157.74s`。75/150/225/300/375/450/525/600八个checkpoint均存在；
  step600完整manifest校验通过。
- 8 tasks×2 videos正式schedule子集的paired内部诊断位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_step0600_internal_order_val8x2_3363345_20260725`。
  Revision和time-centered Temporal均有明显顺序差异，但normalized query与
  effective LoRA分别只剩reversed/shuffled
  `0.0000719/0.0000448`和`0.000297/0.000169`相对L2，故内部gate失败。
- 两个8 tasks×1 video无训练反事实分别位于
  `pi05_action_forecast_belief_v3_step0600_bounded_counterfactual_val8x1_3363345_20260725`
  与
  `pi05_action_forecast_belief_v3_step0600_centered_memory_counterfactual_val8x1_3363345_20260725`。
  normalized-V/bounded-output不能解决；仅去除Temporal masked时间均值即可把
  query/effective LoRA差异恢复到`0.1053/0.0825`与`0.0543/0.0401`，
  定位为global constant遮蔽temporal innovation，而非Revision或query容量不足。
- owner要求完成特异性检查和归因后停下汇报。按先前两级门，未启动
  shuffled/reversed environment rollout；也未启动多checkpoint correct-video
  validation、后续AS续训、架构改写或RL。GPU0–3已释放，4–7始终未触碰。

## 32-token Visual-State canonical design已记录（2026-07-25）

- owner最终对齐的完整设计已集中记录在
  `docs/action_forecast_writer_design.md`：32-token native state anchor、
  初始帧锚点加非递归anchor/local有向变化、可学习identity-init双Meta-LoRA、
  future-action forecasts、Plan/Revision、单-token Belief、两层Temporal、
  content-conditioned query decoder和完整rank-16 LoRA。
- 旧Action-Forecast辅助提示和handoff文档已删除；根`AGENTS.md`、
  `README.md`、`docs/execution_brief.md`、
  `docs/decisions_and_open_questions.md`与`task_plan.md`的活动引用统一指向
  canonical design。旧v1/v2/v3结果继续留在findings/progress作为历史证据，
  但相关段落已明确标为历史，不再形成平行活动口径。
- 当前下一步是原位实现并做必要mechanical checks，然后固定stride5用GPU0–3
  fresh训练75 step，先完成低成本内部顺序与直接换视频特异性闭环。通过后才
  启动fresh 0→1200正式AS；未通过则按最早失效层级快速迭代，不使用contrast
  loss。

## 32-token Visual-State v4实现与profile（2026-07-25）

- canonical v4已原位实现：32个原生anchor tokens、8坐标的initial/anchor/local
  visual-state reader、可学习VL/Action Meta-LoRA、Plan/Revision单-token
  Belief、两层identity-safe Temporal、routing/content分离query decoder及完整
  rank-16 LoRA。旧v3 config/schema已退役。
- Writer实测`10,299,072`个训练参数，和rank128 Source-SFT
  `10,297,344`相差`1,728`（`0.017%`）；public LoRA仍为76 tensors、
  `1,287,168` scalars。focused CPU checks为20 passed。
- GPU0–3真实profile选择stride5、frame-microbatch32、batch20/rank。连续step2
  吞吐约`11.83 queries/s`，峰值allocated/reserved为
  `76,926,757,376/83,703,627,776` bytes，无OOM或nonfinite；现有reserved
  已无batch22或frame-microbatch40的安全余量，因此不做故意OOM试验。
- 75-step specificity训练保留正式1200-step scheduler时间轴，只把本次
  `selected_stop_step`设为75；不得把scheduler总步数压缩成75后冒充正式轨迹
  的前75步。
- step1 checkpoint恢复到step2后，loss、gradient norm、数据/视频/flow-noise
  游标与四rank RNG均匹配连续运行；rank-state文件bitwise一致。CUDA进程重启后
  Writer仅6个tensor出现最大`4.28e-8`的浮点差异，因此checkpoint完整可恢复，
  但不把跨进程CUDA计算误称为bitwise deterministic。

## 32-token Visual-State v4 step75特异性门（2026-07-25）

- 有效fresh轨迹位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_gate75_fb280b3_r4_s5_fm32_b20_20260725`；
  它保留正式1200-step scheduler时间轴，连续完成0→75，消费6000 action
  queries和300个task-video conditions，24 tasks各覆盖12–13条teacher
  videos。step50/75 checkpoint均完整发布，训练wall为`542.04s`。
- 8 validation tasks×2 reference videos×4反事实的内部诊断位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0075_internal_specificity_val8x2_fb280b3_20260725`。
  正确language、flow noise和order反事实的frame indices均固定；实际重算
  reversed/shuffled forecasts，action/reward/outcome reads均为0。
- reversed/shuffled在effective LoRA上的相对L2中位数为
  `0.0420/0.0468`，16/16 comparisons均非零，8/8 tasks均有贡献；旧Belief-v3
  failure只有约`0.000297/0.000169`。同task换demo为`0.0250`，cross-suite
  wrong为`0.0714`，直接换视频特异性同样成立。
- 差异没有在下游再次坍缩：reversed/shuffled从Belief
  `0.8217/0.7852`到Temporal `0.6902/0.6428`，query output仍有
  `0.0528/0.0593`，最终effective LoRA为`0.0420/0.0468`。Revision strength
  中位数分别增加约`11.9%/20.0%`，并由13/16与14/16视频对同向贡献。
- 该低成本门判定通过。step75尚不要求绝对correct success，环境paired
  rollout推迟到已有绝对能力的正式候选，避免低成功率地板把机制检查变成无效
  证据。下一步从fresh identity连续训练到1200。

## v4正式轨迹终止与现有checkpoint选择完成（2026-07-26）

- 正式v4 run
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_development_seed7_ad0db5f_r4_s5_fm32_b20_s1200_20260725`
  已完成step2400并停止；不再续训。2400-step run无OOM/nonfinite/error，
  step2400 checkpoint完整，训练过程共消费`192,000` policy samples。
- owner取消80-episode快筛。固定400 panel评测
  step675/825/900/1200/1275/1500/1875/2100/2400分别为
  `100/109/82/96/94/92/90/90/89`；现有observed-best为step825。
- step825 correct结果位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_correct_ad0db5f_g0123_gen1_b100_roll6_20260725`，
  `results.json` SHA256为
  `92434e9df8e25fdd85f4b09b8102c7410cce32c758e196df196ff6a025222a82`。

## v4 step825完整特异性评测完成并停止（2026-07-26）

- canonical evaluator新增`same_task_other`条件：实际teacher demo固定为paired
  correct demo的`+17 mod 50`，task/language/init/env/policy seeds及Writer
  flow/order随机性保持配对。400/400 rows均核验为同task、不同demo。
  当前fresh复核eval contract/runtime/cache tests为`34 passed in 4.98s`；
  实现已在commit `64af8b0` push到`origin/main`。
- step825内部16-reference特异性证据位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_internal_specificity_val8x2_ad0db5f_20260725`。
  effective LoRA相对L2中位数same/wrong/shuffled/reversed为
  `0.0955/0.8762/0.2598/0.3255`。
- 五个固定400结果为correct/same-task-other/cross-suite-wrong/shuffled/
  reversed=`109/104/99/148/126`。新增四臂output及`results.json` SHA256：
  - same：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_same_task_other_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `36be0c368f278ae1f36a863c672bf890566366f7c25e2b966f27fcc96aeb38f1`；
  - wrong：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_cross_suite_wrong_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `a5f302da57a8a6d19d102f6ac05e7f21249838221f10f251e94658cfcabf501e`；
  - shuffled：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_shuffled_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `d466374207e32adfdb33ccedee093bfc7bf3f8ff167bcb1f551d53ae710057db`；
  - reversed：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_reversed_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `d17c9d66aab8f4f46163e914ef64ffcb1b409d93fa151d2642b4ae8ab66bb101`。
- same-task other只净降5且行为churn最小；但shuffled显著净增39
  (`p=3.48e-5`)，reversed净增17，收益集中在object tasks。实际行为
  特异性硬门失败，当前不进入cold-start RL，也不修改架构或继续训练。
- 所有训练/评测进程已结束。GPU0–3实时均为`0 MiB`且无
  `train_pi05/evaluate_pi05`进程；4–7未触碰。`/data/ymdai`当前占用约
  `321.61 GB`，低于500GB cap。本轮按owner要求在记录、验证、commit、push后
  停止，等待后续讨论或外部专家意见。

## step825固定首帧shuffle快速归因完成（2026-07-26）

- owner授权一个scoped anchor ablation。commit `6b5923f`新增
  `shuffled_keep_first` canonical evaluator条件：复用原full-shuffle
  permutation，只把原始frame 0移回首位。eval contract/runtime/cache focused
  tests为`35 passed in 4.90s`，compile和diff检查通过，commit已push。
- GPU0–3预检均为`0 MiB`、0% utilization；个人占用约`321.61 GB`、预计新增
  `672 MB`，未触碰GPU4–7。固定400 run以4 generators、batch100、24 rollout
  workers一次完成，wall `864.49s`、有效`0.4627 rollouts/s`。
- 结果为`136/400`，逐task
  `9/1/0/45/45/26/1/9`。相对correct `109`为`18`条correct-only与
  `45`条keep-only，`p=8.98e-4`；相对full-shuffle `148`为`32`条
  full-only与`20`条keep-only，`p=0.126`。
- full-shuffle相对fixed-anchor直接净高12，且主要集中在Object-3；两项干预
  可能非线性交互，不能严格做因果加法分解。固定anchor后仍相对correct显著
  净增27，因此当前不再把随机anchor视为必要条件或主要根因；后续专家分析应
  优先审查非首帧order/local-transition/forecast-Temporal映射。
- run output：
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_shuffled_keep_first_6b5923f_g0123_gen1_b100_roll6_20260726`；
  results SHA256
  `0ec198d1438bdb85d9eccb41ac5f6796a470903b963576f29260c048b453ac99`。
  完成后GPU0–3均释放为`0 MiB`。

## 外部专家咨询材料已收敛（2026-07-26）

- 新增`docs/action_forecast_writer_expert_consultation.md`作为只能访问远程
  GitHub的专家唯一自包含入口，按“EMBER思想→全部关键架构演进→当前v4模块与
  完整结果→未解问题”组织，并附远程代码/配置/证据阅读路径。
- 文档嵌入source-base、各历史Writer、v4参数预算、step75内部量、step825
  fixed400逐任务/paired结果及fixed-anchor归因；不要求专家访问历史聊天或
  `/data/...`本地输出。
- README、`docs/execution_brief.md`和
  `docs/decisions_and_open_questions.md`已从旧“75→1200→600续训”未来式更新为
  当前事实：v4停止于2400、observed-best为825、行为特异性失败、RL暂停。
- 本次只整理远程可见的科学上下文，没有启动训练、rollout或新架构修改。

## 外部复核后的v4第一轮因果诊断（后被全面复审覆盖，2026-07-26）

- owner授权自主推进到“决定下一版架构”并明确后续只使用物理GPU4–7；0–3上
  他人进程未被停止、重置或干扰。本轮所有新增GPU launch均只把4–7放入
  `CUDA_VISIBLE_DEVICES`，最终阶段探针峰值reserved为
  `12,530,483,200` bytes/GPU。
- 新增本地一次性forecast-order transplant诊断，固定step825与16条validation
  references，完成`N→N/N→S/S→N/S→S`逐层和policy-function检查。summary：
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_forecast_order_transplant_val8x2_2fa1a1d_20260726`
  与
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_forecast_order_policy_function_val8x2x2_2fa1a1d_20260726`。
- 只对Object-1/Object-3各50 states运行新增四臂rollout，不做full400。结果
  `correct/S→N/N→S/S→S=49/47/72/82`，output为
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_forecast_order_cross_rollout_object13_2fa1a1d_20260726`。
- 完成Plan、Revision direction、value strength和Q/K routing四因子内部/
  policy-function交换，以及两个Object定向rollout。Plan-only/
  strength-only/direction-only/full-Revision=`61/54/67/75`；主要行为中介为
  Revision direction，strength单独与routing均非主因。
- 完成五条Object轨迹、25个经图像与gripper qpos核验阶段、12个LoRA反事实的
  action probe。异常主要改写pre-grasp/close/transport的end-effector
  translation；Revision=0会产生更大且常反向的动作变化，不能直接删除。
- 当时根据仍不充分的证据，过早把v4根因判定为未经识别的shared robot
  absolute-time forecast alignment及其Revision direction，并排除了
  visual-state；下一节的全面复审已撤回“唯一根因”和visual-state排除结论。
- 当时新增的文档（现已改名为
  `docs/action_forecast_writer_v4_root_cause.md`）曾拍板原位删除absolute-time
  Plan/Revision/Belief，改为256D frame-local Intent和adjacent ordered
  Transition；保留32-token visual-state、两个Meta-LoRA、两层content-only
  Temporal及decoder。该架构决定已被下一节撤回为局部候选；从未实现或训练。
- 当时的诊断summary SHA256仍作provenance；当前根因和未决合同以重写后的
  v5 decision文档及下一节为准。

## v4根因全面复审完成，旧v5决定撤回（2026-07-26）

- owner指出上一轮分析过早结束后，继续固定v4 step825并只使用物理GPU4–7完成
  更细粒度诊断；0–3上的他人进程未停止、重置或干扰。个人存储峰前占用约
  `305 GB`，低于500GB cap。
- 完成24 train tasks×4 demos的step75/300/825 hidden forecast semantics审计。
  teacher action/proprio只在inference完成后作measurement target，不进入Writer、
  optimizer或validation/test。summary SHA256依次为
  `99f341c2...b2baa`、`de5a4529...b763c`、`a1633aa5...edb4bf`。
- 完成三个checkpoint的same-task demo geometry及Writer参数演化。证据显示
  visual-state由弱demo信号退化为主要progress code，而raw-image/Meta forecasts
  越来越贴近低层demo translation；AS loss下降时latest-is-best和
  residual-correction语义持续恶化。
- 完成既有400 LoRA consensus、64×8 random permutations、endpoint/time-warp、
  AS loss/gradient和forecast component分解。summary SHA256为
  `390fcad1...f9a6`、`edbb86c8...916e`、`2bd6ae54...7186`。
- 生成Object-1/Object-3共100 episodes的五种root-cause LoRA cache，并运行
  official fixed-state rollout。no-VL/no-Action/lead-only/frame-main-only/
  translation-only为`48/50/40/72/79`；translation-only几乎复现true shuffled
  `82`。LoRA geometry/rollout summary SHA256为
  `3d0b6679...65c1`/`d384219c...662d`。
- 全面结论不再是“absolute-time唯一主因”。当前因果链为AS可识别性不足、
  visual-state旁路、Meta低层phase/translation化及absolute-time Revision放大。
  此前Intent+Transition v5只能修最后一层，已撤回为局部候选。
- 原位重写该根因文档（现名
  `docs/action_forecast_writer_v4_root_cause.md`），并同步README、
  execution brief、task plan、findings、decisions和v4 provenance。当前没有
  v5代码或训练；不继续AS、不进入RL，停在下一版重新设计前。

## correct/shuffled成败翻转行为复放完成（2026-07-26）

- 只使用物理GPU4–7、sealed step825 correct/shuffled LoRA cache和原固定
  Object-1/Object-3各50 states，完成四个condition/task的exact replay；
  success与termination step均`50/50`复现。0–3上的他人进程未触碰，结束后
  4–7均释放。
- 只为47条成败翻转保存每5 steps的agentview/wrist和每步EEF/gripper/action；
  未读取object pose、teacher action/state或隐藏目标。输出为
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_correct_shuffle_flip_replay_object13_g4567_20260726`。
- Object-1/Object-3的`shuffle-only/correct-only`分别为`9/2`和`31/5`。
  Object-3的31条shuffle-only中，correct有23条明确选择深绿色干扰瓶；两臂
  首次闭合点配对距离中位`0.1119 m`。Object-1收益主要来自更早到达和更可靠
  抓取/抬升；反向翻转证明shuffle也会破坏有用控制。
- Object-3收益跨22个teacher demos；四个相同cached LoRA在不同init geometry
  上出现相反翻转。结论从“正号只是无法解释的偶然补偿”细化为：正序视频的
  低层translation controller bias会压过物体语义，shuffle破坏该bias后让已有
  高层任务信息重新主导；不是shuffle生成更多语义或释放参数容量。

## Semantic Core + Causal Procedure v5设计封存（2026-07-26）

- owner批准新focused Goal并要求持续推进到AS特异性/性能与独立cold-start RL
  全部完成。Goal无token budget；完成focused AS/RL后停止，不自动进入
  final-32、task-local RL、joint oracle或ViVLA。
- `docs/action_forecast_writer_v5_design.md`现为唯一活动设计authority，完整记录
  teacher无state prompt、language-conditioned image-position Core、fixed
  native suffix、双Meta-LoRA、per-frame robot-semantic hidden、两层global
  causal Procedure、Core compiler、zero-init Procedure refiner、320
  routing-only identities、factor heads、公开LoRA schema和退役边界。
- 原`action_forecast_writer_v5_decision.md`已改名为
  `action_forecast_writer_v4_root_cause.md`，明确只保存v4根因证据；v4完整设计
  仍为历史provenance，不再定义当前代码。README、AGENTS、execution brief、
  decisions、concept、task plan、findings和progress已开始统一到v5。
- v5已原位实现并删除活动`visual_state.py`、`action_forecast.py`和v4 config；
  AS training、checkpoint、online validation、inference、evaluation cache及
  canonical evaluator均切换到v5 schema，不保留运行时兼容分支。
- 真实构造打印v5 trainable参数为`10,301,440`，比rank128 Source-SFT多
  `4,096`；公开LoRA保持rank16、76 tensors、`1,287,168` scalars。全套
  `187 passed`，Core permutation invariance、causal prefix、zero-content、
  identity、视频条件梯度和固定suffix buffer均通过。
- AS初版曾固定每action独立`N=4`条同task videos、`B_a×4`个逻辑LoRA/loss；
  推理严格one-shot。后续只使用物理GPU4–7，frame stride5固定，重新profile
  `B_a`与frame microbatch后按约一小时segment训练，每段均匀保存6个checkpoint。
- GPU4–7真实profile完成`B_a=1/4/8/12/20`及`m40/B8`边界。最终选择
  `B_a=8`、`N=4`、frame microbatch32；step2→12真实exact-resume通过，
  11个稳态steps中位/均值/范围为`61.39/59.78/38.99–92.08s`，峰值
  allocated/reserved为`60,319,360,000/67,471,671,296 bytes`。`B_a=12/20`
  及`m40/B8`均因reserved跳到约80GB、余量不足3GB而淘汰。
- 正式AS因此封存为每约一小时60 steps、每10 steps一个checkpoint，每段6个；
  下一步是fresh identity第一段和resident validation functional-loss选择。当前
  profile checkpoint只作mechanics/吞吐证据，不作科学性能结论。

## v5 AS首轮训练、step40/120特异性与续训（2026-07-27）

- fresh formal训练已完成step0→60并exact-resume到120；每10步均保留完整
  Writer/optimizer/scheduler/data cursor/4-rank RNG checkpoint。functional
  validation observed-best暂为step40 `0.136874`，step100曾反弹至
  `0.137017`，step120为`0.139036`；尚无足够峰后下降证据。
- step40内部顺序路径存在但很弱，五条件fixed-400为
  correct/same/wrong/shuffle/reverse=`45/52/52/51/51`，行为硬门失败。
  checkpoint未被误判为最终架构上限；step10/40/60内部纵向比较表明task语义
  分离和Procedure顺序差异仍在演化，因此保持架构不变继续训练。
- step120内部反事实通过结构门：Core set对shuffle/reverse数值不变；
  fixed-Core Procedure-only有效LoRA差异为`0.626%/1.087%`，Core-only伪差仅
  `0.073%/0.074%`；8/8 tasks均贡献。same-task-other与wrong有效LoRA差异为
  `1.235%/15.963%`。
- step120完整fixed-400为`65/59/57/61/65`。correct相对step40净增20，
  `41`条new-only、`21`条old-only、exact `p=0.0151`，且跨多个task提升；
  但correct相对same/wrong/shuffle/reverse的净差仅`+6/+8/+4/0`，均未显著，
  所以行为特异性和`125/400`性能门仍未通过。
- 五条件采用每卡5个模型副本并发：每条件仍是完整400 panel、4个持久
  policy/env workers；GPU4–7实测约`64GB`峰值、约`60GB` rollout稳态，五个
  panel约48分钟同时完成。没有触碰GPU0–3。
- 曾从step120按原合同续训，但owner在step128停止；没有生成step120之后的原子
  checkpoint，旧合同科学证据止于step120。

## v5共享四视频训练合同启动（2026-07-27）

- 复核确认旧`B_a=8,N=4`并非每rank只生成4套LoRA：每条action独立采视频，
  实际每step/rank在demo碰撞去重后仍生成约24–32套；step126–128 sampled
  frames为`537–799`，这是约一分钟一步的主要根因。
- owner现拍板：每rank每step一个task，确定性抽4条不同teacher videos，只生成
  4套one-shot LoRA；`B_a`条独立action queries均匀分给4套LoRA，每条action
  只对应一条video，形成`B_a`个普通均值functional losses。4 ranks全局
  task-balanced轮转。
- canonical sampler、AS step、checkpoint schedule identity、run metrics和
  config已开始原位切换到共享set；不新增runner或兼容分支。旧step120不可按
  新合同resume，后续使用fresh root。
- frame stride5保持不变。单video sampled frames为P50/mean/max
  `30/35.6/105`，所以保留`max_frames_per_encoder_call=32`显存安全块；末块
  改为真实长度、不再padding。profile只搜索`B_a`，不做optimizer accumulation。
- 该段记录的是共享合同刚切换时的待办；其focused tests、exact-resume与
  GPU4–7 `B_a` profile已在下一节完成。首轮结束仍先做absolute fixed-400
  validation，达到约`110–120/400`后再做特异性。

## v5共享四视频一对一分组profile封存（2026-07-27）

- canonical映射已改为每rank每step一个task、4条不同teacher videos生成4套
  one-shot LoRA，`B_a`条action按`i mod 4`均分；每条action只对应一条video，
  总functional losses为`B_a`而不是`4B_a`。focused tests直接锁定
  `[0,1,2,3,...]`映射、等分计数和不可整除fail-close。
- GPU4–7真实选择`B_a=16`。canonical 12-step profile先fresh到step2，再从完整
  checkpoint exact-resume到step12；合同SHA256
  `8dd6dfe6...263fb2`，metrics SHA256 `a570c916...09cea`，step12 manifest
  SHA256 `1ef4bde3...61bdd`。24 tasks两轮均覆盖，每task恰好32条action和
  8次video visits。
- 11个稳态steps的wall中位/均值/范围为
  `10.347/10.043/7.072–14.341s`，全局有效pairs/s中位`6.185`；每step始终
  64个全局policy samples、16个Writer video conditions、1次policy forward。
  峰值allocated/reserved为`63,736,767,488/68,415,389,696 bytes`，观察到的
  rank0四视频sampled frames范围`82–240`。
- B20虽完成3步，但reserved跳到`83,732,987,904 bytes`，只余约1.3GiB；
  B24/B32均在首个policy forward OOM，故拒绝。按
  `3600/10.347≈348 steps`取整，正式segment封存为400 steps、每50步一个
  checkpoint，预计约67–69分钟。下一步从fresh identity启动step0→400；
  首段后先做fixed-400绝对性能选择，不先跑特异性。

## v5单视频完整action-batch切换与profile（2026-07-27）

- owner终止共享四视频分组合同，活动训练改为每rank每step一个task、1条video、
  1套LoRA，完整action batch监督这套LoRA；后续task visit轮换video。
- canonical data/as-step/checkpoint/functional路径已原位简化，删除四视频
  schedule、round-robin映射和batched per-sample LoRA执行器；无新runner。
- 最长真实stride5视频为105帧。F32/B1完整一步`5.93s`；F105/B1占
  `79,873 MiB`且超过90秒不完成，因此保留F32显存安全分块。
- GPU4–7联合profile：F32/B20三步为`6.956/3.109/3.527s`，峰值
  allocated/reserved `76,937,901,056/83,630,227,456 bytes`；F32/B24和
  F24/B24 OOM；F40/B20无收益。owner接受最长视频少量余量，选择F32/B20并
  停止B21。
- 正式配置改为fresh step0→900、每100步checkpoint，使用物理GPU4–7；
  fixed validation和后续特异性均等待首段完成。

## v5单视频正式首段启动与跨session交接（2026-07-27）

- canonical单视频实现、focused tests和F32/B20 profile已在commit
  `0b4e00696113cf6601d6e63b4c73734f3cea1073`封存并push；正式launch前
  `HEAD==origin/main`且worktree clean。
- fresh formal已在tmux `ember-v5-as-sv900`启动，只见物理GPU4–7；output为
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727`。
  start-event contract SHA256为
  `03186c57ac736ac82398400676ff10c33eb46ab3e5f9bcbbe44064305944787c`。
- 首步确认每rank B20、1 task/1 video/1 LoRA、全局80 unique actions与4个
  video conditions、一次policy forward、无optimizer accumulation；source
  policy trainable params为0。首步`6.209s`，随后常规步约3–4秒，四卡物理
  显存约77.9GB。
- step100/200/300/400均已原子发布完整checkpoint；step1–400全部finite。
  step400训练body为`1,534.14s`，累计32,000 action queries；24/24 tasks每task
  覆盖1,320–1,340 examples、66–67次video visits和全部50条unique videos。
  常驻模型写出的512-row validation functional loss为step100/200/300/400
  `0.1360107/0.1349113/0.1332633/0.1324333`，无gradient、无optimizer update
  且test action reads为0；正式run仍继续到900。
- 跨session当前状态、精确launch、tmux/log/output、实时复核命令、step900后
  absolute-first评测顺序和禁止事项集中记录在
  `docs/active_session_handoff.md`。该文件是临时live-state ledger，不覆盖
  v5设计authority；新session不得据快照重复启动run。

## v5单视频首段封存、评估加速与轻量特异性（2026-07-27）

- 正式fresh run正常完成step0→900，训练body `3,485.15s`，累计72,000 action
  samples与3,600 one-video conditions；9个每100步checkpoint全部atomic且
  exact-resume state完整。step900每个train task恰有3,000 examples、150次
  video visits并覆盖全部50 videos与50 action episodes。
- correct-video fixed400代表点step100/400/700/800/900为
  `62/64/92/76/103`，首段observed-best为step900。虽然尚低于absolute预门，
  step800→900 paired净提升`+27`、exact `p=0.00155`，没有持续峰后下降证据。
- owner要求任何GPU/checkpoint分配下都先处理long。canonical evaluator已在
  commit `3b6d9d1`实现worker-slot级long-first；step800四卡24 workers先取完
  48个long shards后才取24个普通shards，400 rollouts用`921.60s`，
  `0.4340 rollouts/s`，约为首轮单卡吞吐`2.66×`。focused tests `27 passed`，
  commit已push。
- step900内部16-reference检查显示Core顺序不变性保持，fixed-Core
  Procedure-only effective-LoRA shuffle/reverse差异为`3.689%/5.764%`，
  policy action差异为`0.921%/1.406%`；顺序通路比step120明显增强并到达policy。
- 四个80-rollout反事实臂分别独占GPU4/5/6/7并行，correct直接复用full400的
  init-state 0–9。五臂correct/same/wrong/shuffled/reversed为
  `21/25/14/23/23`；配对净差correct-other为`-4/+7/-2/-2`，exact p为
  `0.344/0.143/0.688/0.688`。这只支持wrong-video方向性，尚无顺序优势；
  按owner定义不把80样本screen冒充full400特异性结论。
- 下一正式动作已封存为同一root从step900 exact-resume到step1800；新增900步、
  72,000 samples、3,600 video conditions与9个checkpoint，预计约一小时和
  `1.2GB`新增存储。GPU仍只用4–7，F32/B20与全部scientific contract不变；
  训练代码/config相对原run commit无diff，当前main仅多了评估调度改动，故使用
  fail-close的`--allow-contract-compatible-code-resume`。

## v5单视频step900→1800正式续训启动（2026-07-27）

- 续训launch前`HEAD==origin/main==db2a690`、worktree clean；step900 checkpoint
  的Writer、trainer与四rank state逐文件SHA256全部通过。个人存储
  `342.21GB`，预计峰值约`343.42GB`；物理GPU4–7均空闲，0–3未进入查询或
  visible set。
- tmux `ember-v5-as-sv1800`已用同一formal root exact-resume。start event为
  原contract `03186c57...94787c`、`resume_step=900`、
  `stop_after_step=1800`、24 tasks、source policy trainable参数0；
  invocation记录runtime commit `db2a690`与
  `contract_compatible_code_resume=true`。
- resume resident validation重算step900仍为`0.1370745508`，optimizer updates
  为0、无parameter gradient、test action reads为0。step901起metrics连续，
  初始核验至step917全部finite；常规step约`3–4s`，每步全局80 actions、
  4个one-video conditions与1次policy forward。GPU4–7约`77.9GB`且UTL接近
  100%。

## v5训练封存与step1400正式特异性（2026-07-27）

- step900→1800 exact-resume正常结束；metrics连续finite、每100步atomic
  checkpoint完整，旧训练tmux已退出。fixed400 correct在step1000与1400并列
  `115/400`，step1700/1800降至`71/86`，选择step1400 observed-best完成唯一
  正式机制检查，不再补1100/1200/1300/1500。
- step1400内部16-reference检查已在GPU4–7完成，16/16 rows、最大wall
  `27.12s`、peak reserved `19.316GB`。Core对同帧集合保持不变，
  Procedure shuffle/reverse中位差`64.30%/72.56%`，但effective LoRA仅
  `2.93%/4.77%`、policy action仅`0.49%/0.75%`；下游融合衰减是最早失效层。
- correct复用既有115/400；same/wrong/shuffled/reversed分别独占GPU4/5/6/7
  同时正式运行，每卡3 Writer generators + 6 persistent rollout workers。
  259个unique LoRA cache/臂只各生成一次；前12个long shards在普通task之前
  全部领取，随后动态分配。四臂均36/36 shards、400/400 rows、六worker
  return code全0、无错误，GPU已释放。
- 最终五臂为`115/108/74/113/114`。相对correct的
  correct-only/arm-only和exact p：
  same `23/16, p=.337`；wrong `58/17, p=2.18e-6`；
  shuffled `14/12, p=.845`；reversed `12/11, p=1.0`。
  视频内容门通过，same方向可接受，顺序门明确失败；v5停止且不进入RL。

## v5.1 authority切换（2026-07-27）

- 已完整保存side-chat收敛方案到
  `docs/action_forecast_writer_v5_1_proposal.md`。v5正式失败触发owner的直接
  推进授权；`AGENTS.md`、`docs/execution_brief.md`和
  `docs/active_session_handoff.md`已切换为v5.1唯一下一架构。
- profile前不预设900/1800 steps：实现与必要smoke后，先在GPU4–7用真实105帧
  视频重新profile显存、action batch和step吞吐，再换算约一小时fresh formal
  stop。实测后来得到首段900；它不规定下一段到1800。首段后先查内部五条件与
  轻量paired行为；第二/第三段只有在特异性、absolute和曲线共同支持时才单独
  启动，绝不自动续训。

## v5.1 canonical实现与CPU合同验证（2026-07-27）

- 已在既有`CompleteLoRAWriter`、AS training/evaluation/checkpoint入口内原位
  替换v5，没有新增runner或并行Writer。活动配置改为
  `configs/pi05_as_writer_language_axial_v5_1.json`；v5 config、constructor
  key和checkpoint/eval/generation schema均已从活动代码删除，v5结果只由Git
  与文档保存。
- tokenizer从完整权威prompt的SentencePiece immutable piece offsets提取task
  span；Text路只输入`BOS + 同一组task-span IDs`，不重新分词也不含模板。
  `video_program`现有Text/VL/Action三套独立rank4 Meta-LoRA，共享
  `2048→256`语言投影；Core value只来自multimodal task-token hidden，raw
  image-position hidden不再进入下游。
- `temporal`现实现token-aligned、跨frame无序的mean-anchored attention，
  两层language-axis Core、两层causal Procedure、centered Procedure reader、
  zero-init AdaLN和一个post-fusion slot block；factor head hidden为240。
  逐模块真实计数与设计表完全一致，总计`10,244,872`。
- CPU验证覆盖真实tokenizer round-trip、可变L/T shape、Core frame permutation
  invariance、Procedure prefix causality、routing/value隔离、fresh identity、
  三阶段gradient opening、固定suffix与不兼容schema。全仓
  `PYTHONPATH=src .venv/bin/pytest -q`为`189 passed`；architecture guard无
  hard violation，既有大owner仅保留review flag。

## v5.1 GPU合同、训练/推理上限与首段seal（2026-07-27）

- GPU4–7真实policy smoke完成step1并从完整checkpoint exact-resume到step2；
  两个cursor、四rank state、Writer/optimizer/scheduler/sampler/video schedule
  与RNG均通过原生checkpoint校验，source policy trainable参数为0。step1/2
  分别约`4.49/2.67s`，没有nonfinite或OOM。
- F32/B20重新在v5.1上实测，不是继承v5：105帧真实最长video步为`7.248s`、
  `11.04` global queries/s；三步profile为`7.322/3.249/3.664s`，常规吞吐
  `24.63/21.84` queries/s。峰值allocated/reserved为
  `76,926,205,440/83,638,616,064` bytes；B20保留约`8.36GiB` allocated
  headroom，按实测batch斜率不再冒险启动B21。
- 推理profile在GPU4–7每卡一次性启动6个worker，24个worker共同按确定性分片
  生成47个LoRA并保留source policy直接rollout。最大单worker generation wall
  `5.203s`，峰值allocated/reserved约`11.63/12.81GB`；48 episodes的
  rollout-only吞吐`0.3799/s`（`1367.6/h`），现场整卡约63–65GB且GPU利用率
  `99–100%`。首次24-policy并发load耗时约`146–162s`，是主要固定成本。
- evaluator queue已进一步修正为全局long-first：任何GPU的未领取max-horizon
  shard都压过ordinary；GPU affinity只决定long内部先取本卡还是偷取他卡。
  新回归覆盖“本卡long耗尽但他卡仍有long且preferred task为ordinary”的情况。
- v5.1首段按新吞吐换算为step900约一小时：4-rank DDP、F32/B20、每step
  80 action queries/4 one-video conditions、每100步checkpoint并做512-query
  online validation。`total_steps=12000`只保留scheduler/最大探索包络；
  `selected_stop_step=900`为唯一当前launch边界。第二/第三段的停止点未预定，
  step900后必须先看早期特异性、absolute和train/validation曲线，不能自动resume。

## v5.1首段训练、step700选择与特异性封存（2026-07-27）

- 正式训练根目录：
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_1_language_axial_dev_r4_seed7_s12000_c199ad3_20260727`。
  fresh step0→900已正常结束，900条metrics连续、checkpoint100..900完整，
  总wall `3622.358s`；contract payload/run-summary file/metrics file
  SHA256分别为
  `acc57fd96cace6d3a9d38a7dbfe6d8593cd29bdce1a0ff10e1f2b4239de46227`、
  `327ba70c9fc9854441a1ce75bb8b6bba103299ae4b49add8dd8c3aa361e96cb0`、
  `0fe5d2490d2d692b98b9c3e8f70177f7839ad0a4e6cdcd5cb943f179d74d4a86`。
- 有放回80-rollout screen全部通过aggregate验证：
  step100/200/300/400/500/700/800/900=`19/18/15/7/21/17/19/14`。
  随后按一张物理卡负责一个checkpoint，同时在GPU4/5/6/7完成
  step100/500/700/900的正式correct400，结果为`82/96/98/84`。四个root：
  `pi05_as_writer_v5_1_correct400_withreplacement_step{0100,0500,0700,0900}_c199ad3_20260727`；
  results SHA256依次为
  `023a9c5fb98fe4b937a1c760a2fa74bb9bb5ba944098af48d593b4cb4ac98577`、
  `23f5032f32d0e95b301ee4b11146efe06a8c955b9e56ad86c7bf735aab9defd5`、
  `cb42f0e7802463cb2e4a26efffc0ce5e41abdb72dad44b750ff2764bb2f9049b`、
  `1b0e28b1afedf133dd43585e9a3b4e6e2a9711e2b436ba5d2ee65c1eaef26ab2`。
  每个root均400 rows、8 tasks×50 states、36 shards、6 workers、return code全0。
- step700轻量五臂复用既有correct80=`17/80`；same/shuffled/reversed分别为
  `20/11/6`，正式root为
  `pi05_as_writer_v5_1_specificity80_withreplacement_step0700_{same_task_other,shuffled,reversed}_c199ad3_20260727`，
  results SHA256分别为
  `ecee24fd84d15d23bf512da8e60316f0224d7c47e3c959d1c7b841ad8bc3fd9b`、
  `a12dda7d65cad76dc2f808bde2f0883969b0fe04e45ec6e5e47500d2ff409324`、
  `52553b14073e8dcca16301bd0e5b0f0ac537e016c9656dd991620d4fd34703a5`。
- 初次wrong root
  `pi05_as_writer_v5_1_specificity80_withreplacement_step0700_cross_suite_wrong_c199ad3_20260727`
  遭遇单worker EGL 0x8cdd；resume后aggregation按launcher timing证据
  fail-close，未产生可信`results.json`，只作失败provenance。正式fresh root
  `pi05_as_writer_v5_1_specificity80_withreplacement_step0700_cross_suite_wrong_fresh2_c199ad3_20260727`
  使用GPU4–7一次调用、24 workers、26 shards完成80/80，wrong=`7/80`，
  results SHA256为
  `e11c0daa1994420dd24b7d52bff5e153a2f1628396527468f1cccda0b5406b75`。
- 五臂逐row exact-pair分析保存在
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_1_specificity80_withreplacement_step0700_paired_analysis_c199ad3_20260727.json`，
  SHA256为
  `6fecb53d051104b72698b5f776eb588240ee5931520bf233985e3b72e2984316`。
  correct-only/control-only为same `4/7`、wrong `12/2`、shuffled `10/4`、
  reversed `13/2`。
- 内部16-reference检查保存在
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_1_internal_specificity_step0700_refs2_c199ad3_20260727`，
  summary SHA256为
  `7a0ced20700b38cd8500396453c7958d94dedde04bd53d5a9c562dda032ec0fe`；
  4 ranks、16/16 rows、8 tasks×2 reference videos全部通过，无validation action
  target或teacher state value读取。
- 所有训练、rollout和内部probe进程完成后均已释放GPU4–7；没有启动第二段、
  第三段、无放回重测、full-400五臂或cold-start RL。当前按owner要求停在
  v5.1首段特异性结论，等待讨论。封存前fresh重读全部上述JSON与SHA、验证逐row
  paired统计和内部counterfactual；全仓`PYTHONPATH=src .venv/bin/pytest -q`
  为`190 passed`，物理GPU4–7均为`0 MiB`且没有活动EMBER tmux/process。

## v5.1无放回五臂与持续探索恢复（2026-07-27）

- owner明确解除上述停止点并创建开放式AS性能Goal；只要absolute没有提高到
  可信满意水平，或提升存在v4-shuffled式逻辑漏洞，就继续探索，不需逐项审核。
- step700新的无放回paired full400已全部完成：
  correct/same/wrong/shuffled/reversed=`88/97/75/65/45`。五个root均400 rows、
  8 tasks×50 states；每task teacher demo0..49恰好各用一次，所有worker
  exit0。结果SHA256依次为
  `d3391e3a...ae1b`、`2369d50c...f388`、`1e295154...a12c`、
  `11a98c83...37ca`、`622b0bca...d598`。
- 逐row配对分析封存在
  `pi05_as_writer_v5_1_specificity400_noreplacement_seed7_step0700_paired_analysis_92b1e03_20260727.json`
  （SHA256 `c4a62c4c...31fa`）。same净`-9,p=.2221`；correct相对wrong
  净`+13,p=.1766`；相对shuffled净`+23,p=.00762`；相对reversed净
  `+43,p=8.91e-7`。新结果消除了v4式shuffle获益，但absolute与wrong breadth
  均未达到停止标准。
- 根据四卡24-worker reversed现场尾部，canonical queue现在保持long全局优先
  的同时，把ordinary工作至少保留两个worker波次。实际标准panel从
  48 long + 24 ordinary变为48 + 48；覆盖仍为400/400且long领取顺序不变。
  focused evaluator测试`48 passed`、全仓`194 passed`，commit
  `73f171a`已push。
- step900 checkpoint重新核验：manifest/canonical payload/writer/trainer
  SHA256分别为`6958498b...b828`、`2971d3a4...8fe`、
  `17da429d...7ac`、`a7057a84...cda`；原训练合同SHA256
  `acc57fd9...227`。下一动作是只在GPU4–7上同root exact-resume至step1800，
  随后按一张卡一个checkpoint并发建立无放回correct400密集曲线。

## v5.1 step900→1800正式resume启动（2026-07-27）

- preflight时HEAD/origin均为`a92850f`且tree clean；个人占用
  `361,804,259,328 bytes`，预计新增约1.2GB；物理GPU4–7均0MiB、无计算
  进程，GPU0–3未查询。正式命令与`task_plan.md`记录一致，在tmux
  `ember-v51-as-sv1800`启动，runtime只见GPU4–7。
- start event为`resume_step=900`、`stop_after_step=1800`、4-rank DDP、
  Writer `10,244,872`参数、source policy trainable count0；
  `contract_compatible_code_resume=true`，原formal合同SHA256保持
  `acc57fd96...227`。resident step900 validation精确复现
  `0.13314267079249476`。
- step1000已生成第一份新增完整checkpoint并继续训练：manifest/canonical
  payload/writer/trainer SHA256为`61e7e66a...6cfc`、
  `b1c7f209...9d2e`、`ea249c56...f065`、`ff391fcc...e9b3`；累计80,000
  action queries，24 tasks均已读全50 action episodes与50 unique videos。
  step1000 online functional validation为`0.1373837591`，比step900高
  `0.0042411`；它只作诊断，不用于替代无放回rollout选择。

## v5.1 1800-step封存、稠密曲线与scale扫描启动（2026-07-27）

- 同一formal root已经完整到step1800并正常退出；新增900步耗时约一小时，
  checkpoints1000..1800每100步完整保留。最后一步applied LR仍为
  `2.84213e-4`，所以没有把“训练结束”误写成“学习率已充分退火”。
- 无放回correct400曲线全部完成：
  `100/500/700/900/1000/1100/1200/1300/1400/1500/1600/1700/1800`
  对应
  `83/98/88/86/114/111/114/92/127/95/92/65/126`。step1400为全局
  observed-best。step500与1600原先因EGL失败后resume聚合规则不完整而没有
  `results.json`，现已从不可变raw shards正式补聚合为`98`和`92`。
- step1400内部16-reference root为
  `pi05_as_writer_v5_1_internal_specificity_step1400_refs2_42a9707_20260727`；
  run-contract/summary/rows SHA256依次为
  `39cb5206...9d3`、`1749a354...d78`、`56b3314d...342`。16/16 references
  表明Core、Procedure、fusion、effective LoRA和policy function的信息路径
  均按v5.1合同工作。
- commit `082090f`完成三项canonical evaluator改进：同GPU EGL transition
  flock、跨resume累计launcher证据、LoRA-B rollout scale。targeted
  `37 passed`、全仓`196 passed`、architecture guard无hard violation；
  feature branch与main均已push，HEAD=origin/main=`082090f`。
- tmux `ember-v51-scale`已启动四个step1400 full400：
  GPU5=`1.25×`、GPU6=`1.50×`、GPU4=`1.75×`、GPU7=`2.00×`。四者均使用
  6 policy workers、long-first queue、无放回state/video双射并复用原
  400-entry LoRA cache；preflight只查询GPU4–7。启动前个人占用
  `375,770,816,512 bytes`，低于500GB cap且scale roots不复制1GB cache。

## v5.1 scale封存与step1400全量控制启动（2026-07-27）

- 四个scale full400均完整退出：
  `1.25/1.50/1.75/2.00 = 124/119/99/82`，均低于原`1.00=127`。
  相对1.00的逐row`new/lost`依次为`21/24, 26/34, 19/47, 14/59`；
  不是一致增益。results SHA256依次为
  `b22e7854...6c48`、`88d84a3d...1964b`、
  `d8e025ac...2c378`、`075f9d3f...0f0b`。选择保留scale 1.00。
- 为避免卡间等待，每个scale所在GPU一释放就自动接入一个step1400控制臂。
  tmux `ember-v51-step1400-specificity`当前在GPU5/6/4/7分别运行
  `same_task_other/cross_suite_wrong/shuffled/reversed`；每臂full400、
  每卡6 generators后原进程切换为6 rollout workers、无放回配对、
  long-first。四臂预计新增约`4.27GB`，个人占用峰值仍显著低于500GB。
- 新配置
  `configs/pi05_as_writer_language_axial_v5_1_stabilization.json`
  已在commit `52503e1`封存并push。它不改Writer拓扑、F32/B20、数据或信息墙，
  只从原step1400加载Writer权重，开启fresh AdamW和
  `peak_lr=1e-4,warmup=50,decay_steps=1800,decay_lr=1e-5`的新阶段；
  首段只运行phase step0→900。完整控制结束后做live GPU/storage preflight，
  再用GPU4–7正式启动。

## step1400五臂完成、低LR运行、v5.2实现（2026-07-27）

- step1400四个控制root均400/400正常退出；五臂总分
  `127/133/94/107/120`，逐row pairing、video bijection、env/policy RNG及
  noise prefix全部通过。统一paired分析artifact已原子写入outputs，SHA256
  `51c19b66...1579`。
- 低LR preflight确认main/origin=`756bdaa`、tree clean、个人占用
  `379,485,047,888 bytes`，GPU4–7均0MiB且无compute process；GPU0–3未进入
  查询或visible set。
- tmux `ember-v51-stabilize1400`已正式启动4-rank F32/B20 phase0→900。
  run contract canonical SHA256为`b19937ce...c95a`，初始化精确记录原
  step1400 manifest `a503eaac...26cb`与Writer
  `22da8417...5d1a`，optimizer/scheduler/RNG均fresh。step100完整checkpoint
  manifest SHA256为`5387b2cf...0a9`，训练继续。
- 隔离worktree `EMBER-v52-20260727`完成canonical v5.2实现、schema/config、
  参数预算与设计文档；focused Writer测试`61 passed`、`git diff --check`
  通过。当前commit `4011966`已push至`origin/codex/v52-patch-grounding`。
- 训练结束后的固定动作是用GPU4/5/6/7各负责phase100/300/600/900一个
  checkpoint，四点同时做无放回correct400；每卡6个Writer generators完成
  cache后转6个persistent rollout workers，queue保持全局long-first。

## v5.1低LR首段完成与并发correct400启动（2026-07-27）

- phase0→900已正常结束：72,000 action queries、3,600 one-video conditions、
  wall `3616.478s`；run summary SHA256 `238853ad...e7b`，最终checkpoint
  manifest SHA256 `c0bf283d...e8d`。
- 九个online validation loss没有持续改善；权重漂移artifact
  `writer_drift_analysis.json`显示100-step update逐渐变小，但后续相邻方向
  持续负余弦，SHA256 `7564fff2...ddc3`。这把低LR描述为待rollout判定的
  稳定化尝试，而不是已成功的新best。
- tmux `ember-v51-stabilize-correct400`已将phase100/300/600/900分别分配到
  GPU4/5/6/7。每个checkpoint只加载一次，每卡6 generators→6 persistent
  rollout workers；全部固定validation 8×50、无放回、全局long-first。
- tmux `ember-v51-stabilize-analysis`等待四个结果并自动生成相对原step1400
  的逐row paired artifact。评测期间main保持`756bdaa` clean，不合并v5.2。

## v5.1低LR封存与v5.2正式合同（2026-07-28）

- 低LR phase100/300/600/900四个无放回correct400全部完成，结果
  `119/115/123/104`，没有超过原step1400=`127`；paired artifact SHA256
  `f52c9b78...543`。phase600仍由Goal-6与两个object task构成，Spatial两task
  均0，故v5.1停止。
- v5.2 branch已推进至`849e622`并完成真实GPU4–7 profile。B20三步含一次
  exact-resume；Task-Queried Patch evidence相对task evidence RMS均值`.429`。
  B21连续三步finite，最大allocated/reserved
  `80,283,666,944/83,892,371,456` bytes；B22四rank对称OOM。
- 配置现封存F32/B21、4-rank、global84、scheduler探索包络12000 steps；
  只授权fresh首段stop=900、每100步checkpoint/512-query online validation。
  900之后必须先做无放回correct400 checkpoint选择与机制检查，不自动进入
  第二或第三段。

## v5.2首段封存、原recipe续训与v5.3实现（2026-07-28）

- v5.2 step0→900、四点correct400、step900内部检查和五臂full400全部完成。
  correct曲线`72/79/120/132`；五臂`132/138/74/82/83`。same无显著差异，
  correct相对wrong/shuffled/reversed均为极显著优势；paired artifact
  SHA256 `d8e2f4b...7ae7`。
- owner明确先测原版v5.2训练上限，并将v5.3设为默认下一fresh架构；v5.3仍用
  原版one-task-per-rank update，不采用task-complete。main
  `529da6b`已从step900 exact-resume到本次segment边界1800，tmux
  `ember-v52-resume-1800`；每100步checkpoint。训练结束后自动在GPU4–7并行
  评测step1200/1400/1600/1800的无放回correct400。
- v5.3设计封存在`docs/action_forecast_writer_v5_3_design.md`。隔离分支
  `c1e3777`已实现task-grounded adjacent visual transition、fresh schema和
  参数预算搬移，全仓回归`198 passed`并push。它不影响当前v5.2训练
  commit；待v5.2上限封存后再做真实GPU profile。

## v6设计封存（2026-07-28）

- owner把默认下一fresh架构提升为EMBER Writer v6，并批准
  Task-Grounded Semantic Set + Visual-Transition Procedure整体方案。
- 新authority已写入`docs/action_forecast_writer_v6_design.md`：Core采用
  mean backbone + centered residual，Procedure采用按actual arm order重算的
  adjacent task-grounded transition，compiler保持v5.2已验证的传递路径，
  factor hidden恢复为256。手算总参数`10,775,296`。
- `AGENTS.md`、active handoff、task plan与findings已同步版本定位。当前只完成
  文档封存；没有修改code/config，没有启动profile、训练或评测。

## v6 canonical实现与task-complete CPU封存（2026-07-28）

- owner提供最终训练合同并覆盖旧v6 recipe：K6 task-complete、首选B20、
  OOM/连续不稳定才退B16；首段后除非absolute明确下降，默认续第二小时。
- 隔离worktree
  `/data/ymdai/.codex/worktrees/EMBER-v53-20260728`已完成唯一canonical v6：
  Semantic Set mean backbone + centered residual、Visual Transition、
  hidden256 factor heads、总参数`10,775,296`。
- 训练入口原位改为每rank六个task-pure micro-round：前5轮DDP`no_sync`，
  第6轮同步；每个task loss乘`1/6`立即backward；一个macro一次zero_grad、
  clip、AdamW和scheduler。B20计数为24 video conditions、480 queries、
  24 functional forwards。
- sampler根据本次video长度做四组cost balance、rank内long-first并跨macro
  轮换物理rank；checkpoint和resume只在macro边界，run contract与metrics封存
  全部计数及24个task/video assignment。
- `configs/pi05_as_writer_language_axial_v5_3.json`已由唯一v6 config替换；
  v5.2/v5.3 checkpoint/eval artifact fail closed。全仓
  `PYTHONPATH=src .venv/bin/pytest -q`为`200 passed`，architecture guard无
  hard violation，`git diff --check`通过。
- corrected mixed-task Source-SFT合同已写回authority，待v6完成后fresh实现/
  重训并寻找validation best。

## v6 B20 profile、resume smoke与正式配置封存（2026-07-28）

- GPU4–7只读preflight均为空闲后，在commit `d66e726`完成B20三步真实
  task-complete profile。root为
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_taskcomplete_profile_b20_d66e726_r2_20260728`；
  3 macro共1,440 queries和72 video conditions，wall `58.730s`。
- 三步max-rank wall=`20.442/18.585/18.635s`，后两步平均
  `25.793 queries/s`与`193.447 macro/hour`；峰值allocated/reserved
  `76,985,299,968/83,644,907,520 bytes`。最长105帧条件成功，loss/grad
  全finite，故选择B20且不运行B16。
- run-contract/metrics/summary/final manifest SHA256依次为
  `5f9b66fc...161e0e`、`e13f250d...16df6`、
  `30bb3798...401a`、`282825c4...733b`。
- 独立resume root从bitwise相同step1边界继续到step3；任务、视频、query、
  LR和cursor一致。GPU kernel非确定性使两步后Writer最大参数漂移约
  `9.82e-5`，不影响exact-state resume合同。visual-transition step1→3
  L2更新`0.0111083`，真实梯度路径成立。
- 正式config封存B20、首段200 macro、每25 macro checkpoint、默认第二段到
  400（除非absolute明确下降）。owner取消正式run全量HDF5 SHA；启动仍核对
  manifest、文件size和HDF5 schema。下一动作是验证、commit/push、集成main，
  再在GPU4–7启动fresh macro0→200。

## v6 task-complete正式首段完成（2026-07-28）

- 正式root
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728`
  已自然停在 macro200；tmux/训练进程退出，run summary 为 200 metrics、
  4,800 video conditions、96,000 queries、wall `3,864.599s`。
- checkpoints `25/50/75/100/125/150/175/200` 全部存在；终点24 tasks各
  4,000 queries、200次video visits、50/50 action episodes和50/50 teacher
  videos。macro200 Writer/trainer/四rank state逐文件SHA256与manifest一致；
  正式log无OOM、nonfinite、CUDA/NCCL error或traceback。
- 全段平均 `18.668s/macro`、`25.720 queries/s`；峰值
  allocated/reserved `76,986,335,232/83,642,810,368` bytes。训练后GPU4–7
  均回到0MiB。
- 训练等待期的仓库清理在隔离分支完成：101 files changed，
  431 insertions/18,853 deletions；tracked tree降至约3.17MB，退役临时
  `.codex/tmp` 另清除108 files/1,820,301 bytes。正确worktree
  `PYTHONPATH`下全仓`177 passed`、Markdown link audit与`git diff --check`
  通过；正式run退出后可安全fast-forward合入。
- 下一动作是合入清理提交，再用GPU4/5/6/7分别运行macro50/100/150/200
  no-replacement correct400；每卡6 Writer generators和6 persistent policy
  workers、全局long-first。

## v6 四点 correct400 启动（2026-07-28）

- 清理/evidence commits `24bdc5d/aecb100` 已 fast-forward 到 main 并push；
  main现场全仓 `177 passed`，status clean。
- live preflight 时GPU4–7均0MiB、个人占用`402,806,314,951` bytes。
  tmux `ember-v6-correct400` 已把macro50/100/150/200依次映射到GPU4/5/6/7；
  每点一个checkpoint、400 episodes、6 generators、generation batch16、
  6 persistent workers、无放回video。
- 四点各自400-entry LoRA cache均已完成；同一进程保留source policy切换
  rollout，避免第二次约150秒模型加载。首批每点6个claimed shards全部来自
  两个horizon-520 `libero_10` tasks，global long-first现场核验通过。

## v6 correct曲线完成与macro200特异性启动（2026-07-28）

- macro50/100/150/200 的 no-replacement correct400 全部自然完成：
  `114/77/120/129`；对应成功 task 数为 `6/7/7/5`。所有 launcher workers
  exit 0、queue 36/36 complete、400 rows、零错误。
- paired artifact
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_correct_curve_paired_aecb100_20260728.json`
  核验四点 state/env seed/policy seed/noise/video assignment 完全一致，
  每 task 50 teacher videos 无放回双射。macro200 是 aggregate best，但与
  macro150 的 9-success 差异不显著，且 breadth 从 7 降到 5。
- GPU4–7 清空后，tmux `ember-v6-specificity400` 已将 macro200 的
  same-task-other/cross-suite-wrong/shuffled/reversed full400 映射到
  GPU4/5/6/7；每臂仍使用 6 generators、batch16、6 persistent workers、
  无放回视频与 global long-first。
- 等待期额外清除约 3.8 MiB Python/pytest/editable-install 可再生缓存。
  Git 仍 clean；9.1 GiB 唯一活动 `.venv` 和实验证据未动。
- 进一步审计发现 117 组 Writer LoRA cache 共 91.74 GB。只删除其中 113 组
  已有 matching results/launcher completion 且所有 worker exit 0 的历史
  cache，共 `87,487,144,566` bytes；当前四个 control cache 被硬性排除。
  rollout rows/results、queue、日志、contract、checkpoint 全部保留，个人占用
  从 `411,326,994,567` 降到 `323,840,205,468` bytes。

## v6 macro200五臂完成与内部检查启动（2026-07-28）

- 五臂结果为 `129/131/108/111/105`；same相对correct switches
  `22/24,p=.8830`，wrong为`42/21,p=.0111`，shuffled为
  `36/18,p=.0198`，reversed为`37/13,p=.00094`。行为方向通过，但后三臂
  margin明显弱于v5.2，且correct只覆盖5/8 tasks。
- 四个control均400 rows、36/36 shards、workers exit 0、零错误；视频无放回
  双射和global long-first复核通过。paired artifact位于
  `pi05_as_writer_v6_specificity400_noreplacement_seed7_macro0200_paired_analysis_faf6e33_20260728.json`。
- control结果封存后删除其4组可重建LoRA cache，额外释放
  `4,254,855,093` bytes；结果/rows/queue/log/contract/checkpoint均保留，个人
  占用降至`319,598,037,816` bytes。
- tmux `ember-v6-internal-m200` 已在GPU4–7启动16-reference内部传递检查；
  输出root为
  `pi05_as_writer_v6_internal_specificity_macro0200_refs2_aecb100_20260728`。

## v6 macro200内部检查完成（2026-07-28）

- 16/16 rows和四个rank输出正常完成，五条件、fixed-Core Procedure-only与
  Core-only反事实均齐。新增visual-transition让shuffled/reversed的Procedure
  median relative-L2达到`.0888/.1167`，并传到effective LoRA
  `.2590/.2436`和policy action`.0282/.0392`。
- fixed-Core结果几乎复现全部顺序差异，而Core-only接近零，排除了Semantic
  Core顺序旁路。相对v5.2，上游Procedure差异增强但下游LoRA/action差异减弱；
  结合macro200仍是absolute右端最高点，按合同exact-resume到macro400。

## 第二轮可重建cache清理（2026-07-28）

- 审计并删除17个旧v5.1 standalone LoRA cache：16个已有完整results/
  launcher completion，另1个结果缺失wrong-video run已被保留的fresh2重跑
  替代，共释放`5,282,177,024` bytes。
- 删除清单为
  `/data/ymdai/outputs/ember/cache_cleanup_legacy_v51_standalone_lora_20260728.json`；
  结果、rows、queue、日志、合同及全部Writer/source checkpoint未删。至此
  outputs中不再残留`writer_lora_cache`或顶层`*_cache`目录。

## v6第二段exact-resume启动（2026-07-28）

- tmux `ember-v6-formal-400` 从`step_00000200`续到macro400；invocation记录
  `contract_compatible_code_resume=true`、`monotonic_stage_extension=true`，
  canonical contract SHA仍为
  `e0d0cf703b596e73552f4150f5abed9f9726a80e5af214095baca33719bdd6a3`。
- GPU4–7各一DDP rank，稳态约78.0GB/卡、约25.25queries/s；resume后metrics
  从201连续追加，没有重放首段。

## v6第二段、四点correct400与focused判断封存（2026-07-28）

- macro200→400 exact-resume自然完成；metrics连续1..400，225..400每25步
  checkpoint和四rank state完整。第二段wall `3,903.024s`，累计9,600 video
  conditions与192,000 action queries；训练/评测进程均退出，GPU4–7释放。
- macro250/300/350/400的paired无放回correct400为
  `117/118/125/125`，没有超过macro200=`129`。完整八点curve artifact SHA256
  `7789350d...72e1`；所有点task/state、env/policy/noise和50-video双射一致。
- 第二小时没有显著aggregate下降，但成功能力在tasks间大幅迁移；因此停止
  full-24 v6 recipe，不补every-25 rollout。macro200仍为observed-best，已有
  `129/131/108/111/105`五臂和16-reference内部证据继续有效。
- owner把corrected mixed-task Source-SFT提前为下一实验，并统一focused AS门：
  `correct400 >= max(150, corrected SFT_best+30)`，同时保留全部视频因果、
  same-task、多task breadth和独立paired复测条件。

## corrected mixed-task Source-SFT集成与profile（2026-07-28）

- 隔离分支`codex/source-sft-mixed`在commits
  `4c527dd/55ccbcc/effbd4b`实现并封存hierarchical mixed sampler、checkpoint
  v2与B144正式合同，随后fast-forward合入main。
- 每rank B144 physical batch固定为24 tasks×6 samples；每个batch一次普通
  同步optimizer update，无gradient accumulation。task→episode→chunk分层
  均衡、跨rank row disjoint、absolute-step sample identity和exact resume由
  focused tests锁定；profile seal后`21 passed`。
- GPU4–7 fresh step1→resume step3完成；root为
  `/data/ymdai/outputs/ember/pi05_source_sft_rank128_mixed_profile_r4_b144_55ccbcc_s3_20260728`。
  后两步wall `16.684/15.847s`、吞吐`34.524/36.346 queries/s`，峰值
  allocated/reserved `60.69/74.07GB`；三步共1,728 unique rows且24 tasks
  每步等量，step3已覆盖每task全部50 episodes。B120 fallback未触发。
- config已封存formal fresh step0→225、每25步checkpoint，约61分钟训练body；
  冷加载单独报告。首段峰值在右端或不稳定时exact-resume到450，之后不做机械
  续段。正式launch前仍需main/origin clean、GPU4–7 live preflight和存储复核。

## corrected mixed-task Source-SFT首段完成与四点correct400启动（2026-07-28）

- main/origin均为`64622795314ab2223b7948f526e7e32767c468df`且launch时
  worktree clean。正式root
  `/data/ymdai/outputs/ember/pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728`
  已自然完成step225；225条metrics连续finite，累计129,600 queries、每task
  5,400 samples，训练body wall `3,639.436s`。
- step25..225每25步的9个checkpoint均保留；step225的LoRA、trainer state、
  四rank state与manifest SHA256复算一致。run root为551MB，个人占用约295GB，
  因而按owner最终澄清不做checkpoint删除。
- online validation step25..225为
  `.139748/.134216/.134064/.132966/.133862/.134068/.134527/.135724/.135276`；
  step100暂为online best，但closed-loop排序尚未得出。
- tmux `ember-source-sft-mixed-val4`将step50/100/175/225映射到
  GPU4/5/6/7，每点fixed validation 8×50、6 persistent workers、一个
  checkpoint只在一张卡加载。四卡冷加载约149–154秒，rollout时约
  72–73GB/卡、接近100% UTL；四个queue首批6个shard均为horizon-520 long
  tasks。
- 四点自然完成为`60/75/77/56`，每点400 unique rows、36/36 shards、
  6/6 workers exit0、全attempt1、零错误；paired state/env/policy/noise
  合同完整。results SHA256依次为
  `760bca21...7976/346100c7...e8a/a3f95801...b546/76687676...a863`。
- step175与100仅差2，paired为`39/37,p=.9088`；175相对225为
  `40/19,p=.00864`。能力在Long/Object上涨的同时从Goal/Spatial迁出，峰值
  不稳定且训练量仅129,600 queries。按sealed规则从完整step225
  exact-resume到450；不改recipe、不补当前密集rollout、不删checkpoint。

## full-24 Source-SFT封顶与global-8 profile（2026-07-28）

- full-24 formal从step225 exact-resume到450自然完成；metrics连续1..450，
  step25..450的18个checkpoint全部保留，root约1.1GB。12点paired
  correct400为
  `60/75/77/56/77/57/87/71/98/109/107/74`；step400为observed-best，
  step450相对400显著下降`50 lost/15 gained,p=1.57e-5`，故停止该recipe。
- dense analysis artifact为
  `paired_correct400_step0050_0450_dense.json`，SHA256
  `5a781a50344b72085ac154b1602a6842cb9bcb6b44a0a957f3da544e5e8791c4`；
  12个面板均400 rows、36 shards、6 workers exit0并满足paired seeds/noise和
  global long-first。
- 隔离分支`codex/source-sft-mixed8`提交`c25cd5d`实现唯一global-8 cyclic
  sampler：4 ranks×2 tasks/update、3 updates覆盖24 tasks，保持
  B144/global576、rank-128 LoRA、LR/scheduler及平均task/sample clock。
- GPU4–7 profile root
  `pi05_source_sft_rank128_mixed8_profile_r4_b144_c25cd5d_s6_20260728`
  完成fresh0→3与exact-resume3→6。两轮cycle均精确覆盖24 tasks，3,456
  samples全唯一；稳态wall `15.833–15.883s`、吞吐
  `36.27–36.38 queries/s`，峰值allocated/reserved `60.69/74.07GB`，
  loss/gradient finite，无OOM/NCCL/CUDA错误。B128 fallback未触发。
- config封存formal fresh0→240、每30步checkpoint、closed-loop
  60/120/180/240；除非可信下降，否则exact-resume至480并评测
  300/360/420/480。owner最终要求保留原checkpoint，不做删除；当前个人占用
  约296GB，远低于500GB cap。

## global-8 Source-SFT正式上限与下一AS判别实验（2026-07-29）

- global-8正式root
  `/data/ymdai/outputs/ember/pi05_source_sft_rank128_mixed8_dev_r4_b144_seed7_s2400_85bfe8e_20260728`
  已从identity fresh完成0→240并从完整step240 exact-resume到480。
  `metrics.jsonl`连续1..480，loss/gradient全部finite；累计276,480 action
  queries，24 tasks各11,520 samples、160次task visits，并覆盖全部50 action
  episodes。step30..480共16个checkpoint全部保留，终点LoRA、trainer和四rank
  state逐文件SHA256复算与manifest一致。两段从进程启动到终点封存合计约
  `11.32 GPU-hours`（4×A100），唯一trainable对象为`10,297,344`参数shared
  rank-128 LoRA。
- step60/120/180/240/300/360/420/480的fixed paired correct400为
  `63/83/85/98/90/62/90/105`，成功task数为
  `4/8/6/6/8/7/4/5`。八点均400 rows、36/36 shards、6/6 workers exit0，
  task/state/env/policy/noise完全paired且global long-first通过；analysis
  artifact SHA256为
  `9446b471016dfb99abb18f107de047163f3245cc9d009456673fe42115c8d2be`。
- step480相对420为`36 gained/21 lost,p=.0627`，只是一次边缘显著的反弹；
  相对step240为`37/30,p=.464`。逐task envelope为126，而任一checkpoint
  最好仅105，仍有21-success能力错位。它相对full-24 step400=`109`为
  `28 gained/32 lost,p=.699`，两个Spatial tasks同为0；因此global-8没有
  提高SFT上限，也没有消除能力漂移，不续到step600。
- corrected Source-SFT development observed-best最终封存为full-24
  step400=`109/400`。focused AS absolute门仍为
  `max(150,109+30)=150`。
- 隔离分支`codex/v6-cyclic8-training@eb7943b`已实现与SFT同构的Writer
  cyclic-8候选并通过正确worktree下全仓190 tests，但尚未合并或启动。
  SFT的直接负对照使“8-task update本身解决漂移”降权，因此不因沉没成本运行。
  下一步先做现有v6 checkpoint的单权重参数平均screen：显式记录源checkpoint/
  权重/hash，derived checkpoint只允许inference、禁止resume/warm-start；
  若不能同时改善absolute和breadth，再fresh改LR/优化器，最后才动
  Procedure→compiler。

## v6 checkpoint参数平均实现与screen封存（2026-07-29）

- 新增唯一的inference-only derived checkpoint owner与薄CLI。派生目录只含
  `writer.safetensors`和canonical manifest，记录source checkpoint路径、
  cursor、manifest/Writer SHA、均匀有理权重与tensor合同；训练resume和
  warm-start仍只接受原始`checkpoints/step_*`。
- outcome前封存四组候选与固定GPU映射：
  `{150,200}→GPU4`、`{200,400}→GPU5`、
  `{150,200,350,400}→GPU6`、
  `{150,200,250,300,350,400}→GPU7`。评测固定为correct400、seed7、
  50-video无放回、6 generators、batch16、6 persistent workers和global
  long-first；合同位于
  `configs/pi05_as_writer_v6_checkpoint_average_screen_v1.json`。
- 四份真实派生权重均已生成，原始及派生checkpoint全部保留。每份包含600个
  state tensors、12,064,064个元素；独立重算确认523个可训练浮点tensor按
  float32均值后回写原dtype，77个固定buffer保持一致，逐元素mismatch为0且
  全部finite。四份formal evaluation adapter authority均通过。
- focused Writer tests为`65 passed`，全仓为`190 passed`；
  `git diff --check`与screen JSON语法通过。下一动作是集成clean main并在
  GPU4–7启动四点paired correct400。

## v6 checkpoint-average评测、五臂和内部传递完成（2026-07-29）

- commit `ea99f65`已合入并push到clean main。四个derived候选分别在
  GPU4/5/6/7完成correct400，结果为`129/140/144/145`。全部输出均400 rows、
  36/36 shards、6/6 workers exit0、无重试/OOM；每个queue前12个claim均为
  horizon-520。winner六点late average相对raw macro200净增16，
  `37 gained/21 lost,p=.04794`，screen paired artifact SHA256为
  `09d4399662de821a1de0d6f38903eeba60a571fee2594c02fe6a445013dfb8ac`。
- winner的same/wrong/shuffled/reversed在GPU4/5/6/7各自完成full400，
  与已有correct合成`145/134/128/119/122`。四个run wall为
  `2279.47/2295.95/2315.81/2338.50s`；所有cache、results、rows、queue、
  logs、contract以及原始/派生checkpoint均保留，不做删除。
- paired checks全部通过：五臂同400个state keys、env/policy/noise，
  teacher video每task无放回双射；same为`+17` demo offset，另三臂复用
  correct demo；所有worker全attempt1且无adopt。correct相对后三臂精确
  p为`.03634/.001299/.006741`，各有5/6/5个正向tasks。same为
  `30/19,p=.1524`，aggregate差11，按预先写入的`<=10`保守边界记borderline。
  artifact SHA256为
  `9244b8db004f4155f9ee254bbddbaf013ee033640b6d9974c2b98cd283579d8b`。
- tmux `ember-v6-avg-late6-internal`在GPU4–7完成16-reference五条件内部检查，
  自然exit0；max allocated/reserved为`11.69/19.33GB`，probe wall
  `26.92s`。fixed-Core Procedure-only保留shuffled/reversed的
  effective-LoRA/action，Core-only近零；summary/rows/run-contract SHA256为
  `7596fbd...169d/b678403f...25d/4ed4aa43...639`。

## v6 fast-decay400 fresh正式合同准备（2026-07-29）

- 新sealed config
  `configs/pi05_as_writer_language_axial_v6_decay400_v1.json`只改变scheduler
  `decay_steps 2000→400`。authorities、information wall、完整v6 Writer、
  data、task-complete conditioning、B20、AdamW和seed逐对象核对完全相同；
  config loader通过。实际LR核验为macro
  `50/100/150/200/250/300/350/400 =
  2.8896e-4/2.5753e-4/2.1049e-4/1.55e-4/9.951e-5/
  5.247e-5/2.104e-5/1e-5`。
- 首段仍fresh0→200、每25 checkpoint且全部保留；评测50/100/150/200。
  除非首段出现可信多taskabsolute下降，否则exact-resume到400并评测后四点。
  该run不从raw或derived checkpoint warm-start。本文记录的是提交前合同；
  正式launch只使用包含这些变更的clean/pushed main，并紧邻执行live GPU4–7、
  storage和新output root核验，最终事实以run contract为准。

## v6 fast-decay400正式训练与八点评测完成（2026-07-29）

- commit `4efa737`的fresh run在GPU4–7完成0→200，随后从完整macro200
  exact-resume到400。两段各约一小时，metrics恰好1..400且全部finite；
  16个25-step checkpoint、optimizer/scheduler、4-rank sampler/RNG和
  trainer state全部保留。累计192,000 action queries、9,600 one-video
  conditions，信息墙记录test/validation action reads均为0。
- macro50/100/150/200和250/300/350/400分别在GPU4/5/6/7完成correct400；
  每点400 rows、36/36 shards、6 workers return0、全attempt1且无adopt。
  每task 50 teacher videos为无放回双射；每个queue前12个shards全部为
  horizon520，清空后才领取普通任务。八点结果为
  `106/64/111/133/132/117/138/143`。
- macro400相对原v6同点125为`46 gained/28 lost,p=.04739`；相对SFT109
  高34但仍低于absolute150。350→400仅`25/20,p=.5515`。fullcurve artifact
  SHA256为`99b04bf1...53d03`，完整checkpoint dynamics SHA256为
  `804689ca...05f32`。
- 训练和八点评测结束后GPU4–7均释放。个人空间约332GB；全部原checkpoint、
  derived checkpoint、LoRA cache、raw rows、queue、logs和results均保留，
  没有执行删除。

## v6 fast-decay checkpoint-average screen封存（2026-07-29）

- outcome前新增
  `configs/pi05_as_writer_v6_decay400_checkpoint_average_screen_v1.json`，
  固定四候选/GPU为
  `{350,400}→4`、`{200,350,400}→5`、
  `{200,250,350,400}→6`、
  `{150,200,250,300,350,400}→7`。
- 所有派生权重继续使用已验证的float32均匀平均、原dtype回写和
  inference-only manifest；不得用于resume/warm-start。评测固定correct400、
  无放回video、6 generators、batch16、6 persistent workers与global
  long-first。config SHA256为
  `07d115811cf6042d5d0246e9f91c304aed3e5289b53d898d17af0330526951f5`。
- screen只从包含本文与config的clean/pushed main执行；先生成四份derived
  checkpoint并完成CPU逐tensor复算/authority检查，随后GPU4–7并行评测。
  所有源checkpoint、派生checkpoint及评测cache/rows/results继续保留。

## v6 fast-decay checkpoint-average screen完成并暂停（2026-07-29）

- commit `7c3879c`的sealed screen在GPU4–7完成。四份derived checkpoint均
  独立复算为`max_abs_error=0`，formal authority通过；四卡各负责一份
  checkpoint，每卡6 generators、batch16和6 persistent workers。
- `{350,400}`、`{200,350,400}`、`{200,250,350,400}`、
  `{150,200,250,300,350,400}`的correct400为
  `139/135/129/130`，均低于raw macro400=`143`。最佳两点average相对raw
  为`18 gained/22 lost,p=.6358`；没有candidate达到absolute150。
- 所有run均400 rows、36/36 attempt1 shards、6 workers return0、无adopt；
  video无放回双射、paired seeds/noise与global long-first全部通过。
  artifact file/canonical SHA256为
  `ac6e1545...1d30/a9ffd347...9fdb`。GPU4–7已释放。
- owner明确要求本步后稍停讨论；未启动五臂、内部传递、gradient-conflict
  分析、第三训练段或下一fresh实验。全部checkpoint和评测证据继续保留。

## fast-decay续训、五臂与内部检查完成（2026-07-29）

- 同一fast-decay正式root从完整macro400 exact-resume到600；metrics连续
  1..600且finite，累计288,000 action queries和14,400 one-video conditions。
  新增450/500/550/600 correct400=`131/130/132/126`，每点400 rows、
  36/36 attempt1 shards、6 workers return0、无adopt；long-first与无放回
  video双射均通过。macro400仍为best，400→600显著下降`p=.01609`。
- macro400五臂在GPU4–7完成：
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。五臂pairing、
  state/env/policy/noise、video和global long-first审计全部通过。
- 16-reference内部检查自然exit0；fixed-Core Procedure-only复现
  shuffled/reversed的effective-LoRA/action，Core-only近零。全部原checkpoint、
  cache、rows、queue、logs和结果保留。

## v6旧训练范式实现、profile与正式run（2026-07-29）

- commits `36f1cf4/86438ab/42ac8c0/a937e52/bad9a96`实现并封存唯一的
  rank-rotating v6 control。canonical config为
  `configs/pi05_as_writer_language_axial_v6_old_recipe_v1.json`；固定B20，
  对更大action-query batch fail closed。`B21`只曾作为未运行的候选名出现，
  没有profile、训练或评测。
- 最长105-frame真实视频完成fresh step0→1和exact-resume1→3；后两步稳态
  `20.091 queries/s`、约904 updates/hour，峰值allocated/reserved
  `76.94/83.72GB`。全仓192 tests通过后才启动正式run。
- 第一次formal launcher因额外CLI checkpoint列表终点900与sealed
  `total_steps=12000`不一致而在创建有效step/output前fail closed。失败log
  保留；移除该错误覆盖后以不变科学合同重启。
- 正式root
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_oldrecipe_rankrotating_dev_r4_b20_seed7_s12000_bad9a96_20260729`
  完成900 updates；`metrics.jsonl`连续1..900，run summary记录72,000 action
  queries、3,600 video conditions和`3,626.731s`训练body。每100步checkpoint
  全部保留；每6 updates完整覆盖24 tasks。

## v6旧训练范式评测与内部传递完成（2026-07-29）

- step100/500/700/900分别映射GPU4/5/6/7并行correct400，结果
  `98/121/76/95`。每卡一checkpoint、6分布式Writer generators、
  batch16、6 persistent rollout workers；每个queue前12 shards均为
  horizon520，之后才领取普通task。四点均400 rows、36 shards、全attempt1。
- step500 single-checkpoint best完成五臂：
  `121/122/111/84/47`。same同档；wrong不显著且贡献集中，语义门失败；
  shuffled/reversed强显著通过。
- checkpoint dynamics与16-reference内部检查完成。old recipe显著放大
  Procedure→effective-LoRA/action的顺序差异，fixed-Core反事实完整复现，
  Core-only近零。训练和全部评测自然退出，GPU4–7释放。
- 当前按owner要求停下讨论；没有修改v6架构，没有启动后续fresh训练、
  one-shot或RL。

## v7第一性原理设计封存（2026-07-29）

- owner解除上一轮暂停边界，要求先记录需求/设计，再创建session-local Goal
  自主推进absolute performance。完整v7 authority已新增为
  `docs/action_forecast_writer_v7_design.md`。
- v7定义唯一的Task-Aligned Semantic Trajectory、frame-mean Core、8-token
  sparse Action Expert probes、forward Action–Effect binding、三层causal
  Procedure与Procedure-content-only compiler。
- owner进一步把8→1聚合收敛为单步joint action–effect pooling：全部`8×L`
  pairs直接归一化并形成每区间一个event，删除独立EventRead；Core直到
  compiler才首次与Procedure相遇。真实模块枚举为`10,312,192`，与更新后设计
  预算逐项吻合。
- root `AGENTS.md`、README、execution brief、active handoff、task plan和
  findings已同步下一fresh架构定位。architecture guard修改前baseline为pass，
  无hard violation、parallel version family或活动source diff。
- 设计落盘前现场只读核验：HEAD与origin/main均为
  `f920f4a0e13366864fee3334eb60beb56c4edf6d`，原worktree clean；GPU4–7为
  0MiB，GPU0–3存在其他用户进程且未触碰；个人空间约338GB；无EMBER训练/
  评测进程。
- session-local Goal已经建立。canonical Writer源码/config已原位切换到v7，
  v6 schema/checkpoint不兼容且没有并行可执行分支；全仓192 tests通过，
  architecture guard无hard violation或parallel family。

## v7 B20真实profile、resume与正式合同（2026-07-29）

- GPU4–7上B32、B24均在首个functional policy forward明确OOM；不再扫描中间
  batch。B20连续3个完整macro finite，首步含105-frame最长视频，三步wall为
  `19.234/17.492/17.447s`，后两步均值`27.477 queries/s`、
  `206.075 macros/hour`；峰值allocated/reserved为
  `77,020,274,176/83,647,004,672 bytes`。
- B20 root为
  `/data/ymdai/outputs/ember/pi05_as_writer_v7_profile_b20_jointae_r1_20260729`；
  run-contract/metrics/summary SHA256为
  `c0f1becf...e0ee3/fc1f361d...9dc8/6da42ada...fc25`。
- 独立resume root fresh0→1后exact-resume 1→3；checkpoint1未改写，task、
  video、query、LR与cursor身份和连续run一致，最大mean-loss绝对差
  `2.33e-5`。joint binder的`262,656/262,656`参数在真实step1→3全部变化，
  L2位移`0.08944`。
- 正式配置封存task-complete B20、teacher seed`20260722`和fast cosine
  decay400；首段fresh0→200、每25 checkpoint，共96,000 queries与4,800
  one-video conditions。实现/profile seal commit为
  `ca7db57d0c2d1ec2e7032a44b58238b6de35b1f4`，已push至`origin/main`。
  正式root预声明为
  `/data/ymdai/outputs/ember/pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729`；
  尚未启动正式训练或评测。

## v7正式训练、评测与内部根因完成（2026-07-29）

- 正式root完成fresh macro0→200及exact-resume200→400，metrics连续1..400、
  loss finite；16个every-25 checkpoint及完整resume状态保留。
- macro50/100/150/200/250/300/350/400 correct400为
  `82/106/114/120/101/114/115/106`。macro200五臂为
  `120/112/91/100/69`。
- refs1内部检查覆盖8个validation tasks。Action–Effect pair attention熵约为
  理论均匀熵的99.96%，有效8 probes约7.998；fixed-Procedure/vary-Core的
  effective-LoRA差异只有约0.1–0.2%，fixed-Core/vary-Procedure几乎复现全部
  差异。v7停止且GPU4–7释放。

## v8设计与canonical CPU实现（2026-07-29）

- 新authority`docs/action_forecast_writer_v8_design.md`记录hierarchical
  Action–Effect binding、Procedure-only EventRead与Core multiplicative
  gate。v7 source/config已原位替换，不保留并行可执行路径或checkpoint兼容。
- 每个Action anchor独立读取task-token effects，得到8个bound tokens后再
  聚合成一个event；Core gate只乘性调制Procedure slots，不增加Core-only
  value path。
- 真实枚举：binder`590,848`、compiler`1,469,696`、Writer
  `10,706,176`。聚焦38 tests和全仓192 tests通过；Markdown link audit零
  缺失、`git diff --check`通过，architecture guard无hard violation或
  parallel version/function family。shape/mask、Action/effect gradient、
  identity、`D=0`与`Procedure=0`硬约束均成立。
- 活动config为`configs/pi05_as_writer_language_axial_v8.json`，profile和
  formal状态均为pending。下一步是全仓回归/clean push，然后只在GPU4–7做
  B20三macro最长视频profile；失败才直接B16。

## v8 B20 profile、resume与formal seal（2026-07-30）

- live preflight：GPU4–7均0MiB/0%，无进程；个人`/data/ymdai`用量352GB，
  总盘余量2744GB。GPU0–3未进入visible set。
- B20 root完成3/3 macros，含105-frame最长视频；三步
  `19.243/17.506/17.450s`，稳态`27.463 queries/s`、
  `205.974 macros/hour`，峰值allocated/reserved
  `77,035,771,904/83,655,393,280 bytes`。B16未触发。
- 独立resume root完成fresh0→1→resume3；step1未改写，scientific cursor
  全同，最大loss差`4.7951e-5`。全部binder和Core modulation参数以及所有主
  模块step1→3均变化。
- config已恢复正式teacher seed`20260722`并封存B20、fresh0→200、
  every25 checkpoint；下一步是CPU复验、clean commit/push和正式launch。

## v8完成并切换v10（2026-07-30）

- v8 macro0→400正式训练与八点correct400完成；曲线
  `90/110/82/110/90/125/98/115`，best macro300五臂
  `125/121/110/110/117`。内部检查确认event被Effect主导，v8停止。
- owner批准Evidence-Preserving Dual-Stream v10并创建session-local Goal。
  `docs/action_forecast_writer_v10_design.md`、唯一canonical源码和不兼容v10
  config已完成；尚未封存的v9草案与v8 executable config原位退役。
- v10真实参数`11,627,520`；全仓192 tests与`git diff --check`通过。
- GPU4–7 B20 profile三步finite且包含105-frame视频，后两步约
  `26.38 queries/s`、`197.85 macros/hour`，峰值约`77.01/83.65GB`。
  fresh0→1→resume3通过，最大loss差`2.63e-6`，所有新增路径梯度可达。
- config已恢复正式teacher seed`20260722`，封存task-complete B20、
  fast-decay400、fresh0→400、every25 checkpoint。紧邻动作是clean
  commit/push和正式两小时训练。

## v10正式序列完成并按owner要求暂停（2026-07-30）

- main `5fd0a25`上的v10正式run在GPU4–7从identity fresh自然完成
  macro0→400；400行metrics finite，累计`9,600`个Writer视频条件、
  `192,000`个action queries，wall约`7,832.8s`。训练与全部rollout进程均已
  退出，GPU4–7释放。
- 12个single checkpoints的paired、每task 50 teacher videos无放回
  correct400为`95/103/84/89/82/90/96/96/89/96/97/91`；
  observed-best是macro50=`103/400`，未使用checkpoint融合。
- macro50五臂完整完成：`103/94/75/67/43`。same同档；
  wrong/shuffled/reversed相对correct的exact p依次为
  `.001762/1.01e-5/5.63e-13`，视频行为门通过，但absolute低于
  corrected Source-SFT `109`且未达150。
- refs1内部检查覆盖8/8 validation tasks。Core顺序不变、Procedure差异可完整
  传到LoRA/action、Procedure=0严格identity；同时Action变化远强于Effect变化，
  Effect attention近均匀，compiler将很小的Procedure slots通过RMSNorm调制为
  高增益Core content。同task换正确video的Procedure/action方差也显著高于
  v5.2，解释了强特异性与低absolute并存。
- 中间因main存在owner写入、未跟踪的Loom文档，评测clean-worktree guard按
  设计fail closed；失败调用未创建queue/run contract，输出和log已原样移入
  `.codex/tmp/v10_dense_failed_clean_guard_20260730/`。随后从同一commit的
  detached clean eval worktree完成全部正式评测，科学合同与结果未污染。
- owner最新指令为“v10做完就先停下”。没有启动Loom、one-shot或RL；Loom
  相关未跟踪文档和隔离worktree中的未提交草案均保持原样，未接入main。

## Loom canonical实现与正式启动门（2026-07-30）

- owner后续明确授权Loom，因此v10后的暂停边界已解除。Loom原位替换唯一
  canonical Writer/config；真实枚举参数为`12,855,552`，不从旧Writer
  checkpoint resume。
- 全仓191项CPU测试、compileall、diff check通过；architecture guard无
  hard violation或parallel version/function family。GPU4–7的B20三macro最长视频profile
  含105-frame条件，三步`20.463/18.397/18.367s`，稳态约
  `26.112 queries/s`与`195.843 macro/hour`；峰值allocated/reserved为
  `77,566,232,064/83,732,987,904 bytes`，B16未触发。
- 正式seed`20260722`下fresh0→1→exact-resume1→3通过。step1 checkpoint
  全文件未改写，task/video/query和LR逐步等于uninterrupted profile，最大
  mean-loss差`1.5891e-6`。首次尝试因两次进程加载之间修改config文件而被
  SHA合同正确拒绝，未污染checkpoint。
- 正式首段封存为GPU4–7、task-complete B20、fresh macro0→200、每25 macro
  checkpoint；结束后只比较single checkpoint 50/100/150/200，不做融合。

## Loom正式首段、correct曲线与内部停止判定（2026-07-30）

- main `1e5870f`上的Loom fresh macro0→200自然完成，wall`3,855.28s`、
  `4,800`视频条件和`96,000`action queries，训练机械、task-complete覆盖、
  checkpoint和online validation均完整。
- macro50/100/150/200的paired无放回correct400为
  `79/106/105/112`。每点400 rows、36/36 shards、无failure；macro200为
  observed-best，但比同macro、同recipe v6的`133`低21，也比同期v5.2的
  `132`低20，未触发第二小时。
- owner要求先做内部数值分析且暂停rollout。已停止自动启动的四个特异性臂；
  停止时cache=0、results不存在，未运行任何same/wrong/shuffled/reversed环境
  rollout。
- macro200内部五条件检查完成且没有环境交互：Core顺序合同、差异传递、
  compiler replay和zero-Teacher identity均通过；同时matcher近uniform、
  visual confidence近零、shuffled confidence/scale高于correct、
  Teacher–Policy gap近常数、Teacher支配LoRA且same-video方差偏高。Loom据此
  作为科学non-pass停止，不继续修补或续训。

## Recenter canonical实现与CPU合同（2026-07-30）

- owner明确授权在同一session持续自主推进，已创建session-local Goal；目标为
  single-checkpoint correct400至少150或稳定接近且显著高于旧架构，达到
  absolute门后才做行为级视频特异性。
- Loom首段四点correct400为`79/106/105/112`，内部gap/confidence/
  correspondence缺少可靠锚点。按owner“不得打补丁、必须从根因重设计”的
  要求，新增
  `docs/action_forecast_writer_recenter_design.md`并把Loom降为provenance。
- 唯一canonical源码已原位切换：恢复原生50-token Action mean，保留v6
  Semantic Core，新增25%径向上限的task-grounded transition residual与单路
  causal Procedure；compiler改为Core-keyed、time-centered raw Procedure
  values和amplitude-preserving slot mixer。Loom-only `relations.py`、双
  Procedure和gap compiler已退役，无平行可执行路径。
- 活动fresh config为`configs/pi05_as_writer_recenter.json`，schema、checkpoint、
  launch、eval adapter和episode evidence均切换为Recenter。Loom的profile/
  resume/gradient evidence没有复制；profile与formal状态保持pending。
- 精确参数枚举为`10,709,248`。确定性测试覆盖Core permutation、transition
  cap、Action-zero无旁路、causality、constant Procedure identity、Core gate、
  Procedure scale、step0 identity、staged gradient及零点finite backward。
- 修复审查发现的zero-RMS反向NaN：transition分母直接使用mean-square，
  diagnostic RMS detach；slot mixer用`torch.linalg.vector_norm`物理RMS及
  零subgradient处理零输入。
  targeted tests和全仓`196 passed`；额外覆盖bf16非2幂长度constant Procedure
  精确零与near-zero mixer有界梯度。compileall和diff check通过，architecture
  guard只有既有大文件review提示，无hard violation，active source净删约
  1,100行。紧邻动作是clean commit/push；之后只在GPU4–7重新做Recenter
  B20/B16 profile和exact-resume，不继承Loom seal。

## Recenter B20 profile、resume与formal seal（2026-07-30）

- main已在`93c7e32`封存canonical Recenter实现。GPU4–7独立完成B20三macro
  最长视频profile：包含真实105-frame条件，3/3 finite；后两步均值
  `25.808 queries/s`、`193.562 macro/hour`，峰值allocated/reserved
  `76.99/83.64GB`，B16未触发。
- 正式seed `20260722`下fresh0→1→exact-resume1→3通过；metrics、LR、
  task/video/query cursor连续，step1 checkpoint各文件hash在resume前后
  完全不变，validation/test action reads为0。
- profile step1→3间全部`10,709,248`个Writer参数变化，覆盖11个主模块组。
  config现已恢复正式teacher seed并seal为B20、fresh macro0→200、每25 macro
  checkpoint；紧邻动作是clean push后在GPU4–7启动约一小时正式段。

## Core-Program canonical CPU实现（2026-07-30）

- 新authority为
  `docs/action_forecast_writer_core_program_design.md`。canonical源码/config/
  launch/checkpoint/eval schema已原位切换；删除
  `configs/pi05_as_writer_recenter.json`，不保留兼容执行路径。
- compiler现为raw Core value reader、Core-keyed full raw Procedure reader、
  width512 bias-free strict bilinear和zero-preserving slot block；transition
  恢复v6 uncapped `A+R`。精确参数枚举为Writer `10,905,856`、compiler
  `1,665,792`。
- 模型合同tests `14 passed`；config/checkpoint/evaluation相关affected tests
  `34 passed`；最终全仓`194 passed`、compileall与diff check通过。
  architecture guard为REVIEW、无hard violation，active source净删643行。
- 按single-checkpoint合同删除checkpoint-average module/CLI，evaluation明确
  拒绝`derived_checkpoints`；RL-Writer在raw-video Core-Program接口完成重建和
  fresh retrain前于任何GPU/data加载前fail closed。GPU profile、resume smoke
  与formal训练尚未开始，不得继承Recenter evidence。

## Core-Program B20最长视频profile（2026-07-30）

- main `4769b36`在GPU4–7以profile-only teacher seed `172`独立完成3个
  task-complete B20 macros；首步覆盖task38/demo36的真实105个stride-5帧，
  三步loss/gradient均finite。
- 三步wall为`20.4094/18.5197/18.5874s`；后两步均值
  `25.8712 queries/s`、`194.0340 macro/hour`。峰值allocated/reserved为
  `76,993,247,232/83,644,907,520 bytes`，因此选择B20且不触发B16。
- profile step1→3的523个trainable tensor全部发生变化且finite，覆盖
  Meta-LoRA、Semantic Core、transition、Procedure、strict bilinear compiler
  和factor heads。配置已恢复正式teacher seed`20260722`；下一步为独立
  fresh0→1→exact-resume1→3 smoke。

## Core-Program exact-resume与正式seal（2026-07-30）

- 当前schema/commit的独立root先fresh到macro1，再从完整macro边界resume到3；
  metrics严格为`1,2,3`，累计queries为`480/960/1440`、video conditions为
  `24/48/72`，LR、task/video/query cursor和四rank RNG均连续。
- resume前后macro1的manifest、Writer、trainer和四个rank state逐文件hash
  不变；三步均finite，validation/test action reads保持0。配置现已封存B20、
  formal teacher seed`20260722`、fresh macro0→200和every25 checkpoint。

## Core-Program首段正式launch合同（2026-07-30）

- sealed config commit为`d67d9f5`；fresh identity、GPU4–7四rank、NUMA1、
  B20、2 workers/rank，macro0→200，每25保存。首段精确消费4,800个
  one-video LoRA conditions和96,000 action queries，不继承profile/smoke权重。
- output为
  `/data/ymdai/outputs/ember/pi05_as_writer_core_program_taskcomplete_decay400_dev_r4_b20_seed7_s2400_d67d9f5_20260730`；
  启动前个人目录`433.62GB`，formal与四点correct400预计新增低于7GB。

## Core-Program首段完成与correct400启动（2026-07-30）

- fresh macro0→200自然完成：200行连续finite metrics、8个every25 checkpoint、
  4,800个single-video LoRA conditions和96,000 action queries；training body
  `3,858.26s`，last-50 steady约`25.742 queries/s`，终点loss`0.10119`。
- 24 tasks每macro恰好各出现一次，DDP每macro一次同步；validation/test action
  reads保持0。8个checkpoint manifest、trainer和四rank state均通过校验。
- macro50/100/150/200现分别在GPU4/5/6/7并行做paired、无放回correct400；
  每卡6 workers、4 Writer generators、generation batch8，全局long-first。
  四个run均已完成prepare，绑定各自唯一raw checkpoint。
- macro150首次launcher的source-run参数误填，合同校验在模型/GPU加载前拒绝；
  删除508B空壳后已按正确source-base合同fresh重启，不影响其它三点。

## Core-Program评测完成与停止（2026-07-30）

- macro50/100/150/200 fixed correct400全部完成，结果为`84/75/60/76`；四个
  roots均为400 rows、8 tasks×50，全部50 teacher videos无放回恰好一次，
  mapping和seed schedule完全相同，teacher action reads为0。
- observed-best macro50相对v5.2 step900与v6 macro200分别净低48/49，
  paired p值为`3.88e-7/4.76e-7`。按预定门停止Core-Program，不续训、不做
  行为级视频控制。
- GPU4–7完成macro50 refs2内部检查：16条件、无rollout/reward、validation
  actions0、teacher states0。结果定位为compiler压缩强AC顺序信号和bilinear
  梯度耦合，不是上游完全忽略视频。

## Prior–Innovation canonical CPU实现（2026-07-30）

- 新authority为
  `docs/action_forecast_writer_prior_innovation_design.md`。唯一canonical
  compiler已整体替换为Core semantic prior + centered Procedure innovation；
  Core-Program config/schema/class退役，不保留并行执行路径。
- 新fresh config为`configs/pi05_as_writer_prior_innovation.json`，所有profile、
  resume、gradient和formal证据重置pending；不能继承旧硬件证据或checkpoint。
- 精确Writer/compiler参数为`10,643,968/1,403,904`。focused不变量、全仓
  `195 passed in 16.13s`、compileall、JSON、diff check全部通过；
  architecture guard无hard violation。尚未使用GPU。
- `/data/ymdai`当前占用约`438.61GB`，本轮profile、训练和四点correct400预计
  新增`6–8GB`，低于500GB hard cap。紧邻动作是clean commit/push，然后只在
  GPU4–7做最长105-frame B20三macro profile和exact-resume。

## Prior–Innovation B20 profile与formal seal（2026-07-31）

- main `7b7abf1`在GPU4–7完成独立最长105-frame B20三macro profile，全部
  finite；稳态约`25.818 queries/s`、`193.635 macro/hour`，峰值
  allocated/reserved约`76.99/83.64GB`，B16未触发。
- 正式teacher seed下fresh0→1→resume1→3通过；step1逐文件SHA未改写，
  metrics/LR/cursor/RNG连续，累计72 videos和1,440 queries，validation/test
  信息墙读数均为0。所有主模块finite且可达。
- config已恢复正式seed并seal为B20、fresh macro0→200、every25；下一步为
  clean push sealed evidence后，从identity启动约一小时正式段。

## Prior–Innovation首段正式launch（2026-07-31）

- sealed config commit为`807266b`；live preflight确认Git clean/pushed、
  GPU4–7各约81.1GB free且无compute process，个人目录`439.39GB`，正式段及
  四点correct400仍低于500GB hard cap。
- tmux `ember-prior-innovation-m200`已从fresh identity启动：GPU4–7四rank、
  NUMA1、B20、2 workers/rank、macro0→200、every25，精确预算4,800个
  one-video conditions和96,000 action queries。
- output为
  `/data/ymdai/outputs/ember/pi05_as_writer_prior_innovation_taskcomplete_decay400_dev_r4_b20_seed7_s2400_807266b_20260731`；
  首段完成后只做macro50/100/150/200 paired correct400，不融合checkpoint。

## Prior完成与Target-Spectral CPU实现（2026-07-31）

- Prior formal macro0→200已自然完成；macro50/100/150/200 paired correct400
  为`100/61/89/88`。没有启动第二小时或行为级视频控制。
- 新authority为
  `docs/action_forecast_writer_target_spectral_design.md`。唯一canonical
  compiler已从320个rank-level semantic slots替换为38-target-first、
  rank-last spectral compiler；Prior config/schema退役。
- Target-Spectral Writer精确参数`14,495,744`。A/U采用FP32 reduced-QR并
  固定R对角符号；已补强共同方向压力测试、effective-LoRA视频条件测试、
  38-target拓扑guard、BF16输入的FP32 Procedure centering以及不手工开权重的
  三步gradient staging。
- 当前训练合同没有变化：一条video生成一套LoRA，action query跨episode；
  full24等权、B20、一次AdamW。profile/resume/formal evidence全部重新置为
  pending，不能继承Prior。
- 下一步是完成全仓验证、clean commit/push，只在GPU4–7做最长视频B20 profile
  和exact-resume；通过后fresh macro0→200并评测四个single checkpoints。

## Target-Spectral B20最长视频profile（2026-07-31）

- main `f8bbce6`在GPU4–7以profile teacher seed172完成三个task-complete
  B20 macros；首步包含task38/demo36真实105-frame条件，三步loss和gradient
  均finite。
- 后两步均值`25.488 queries/s`、`191.159 macro/hour`；峰值allocated/
  reserved为`77,074,980,864/83,649,101,824 bytes`，因此B16不触发。
- step1→3的530个trainable tensor中458个变化，所有主模块finite且变化。
  唯一整组暂未变化的是72个Action Meta-LoRA A；这是spectral scale、
  Procedure AdaLN与Meta-LoRA B连续zero-init造成的预期四步staging，配对B已
  全部变化。配置已恢复formal seed，下一步做独立fresh0→1→exact-resume1→3。

## Target-Spectral exact-resume与formal seal（2026-07-31）

- formal seed `20260722`的独立root先fresh0→1，再从完整macro1恢复到3；
  metrics严格为1/2/3，LR、task/video/query cursor、累计queries
  `480/960/1440`和video conditions`24/48/72`连续且finite。
- resume前后macro1的manifest、Writer、trainer和四rank state共七个文件SHA
  完全不变；validation/test action reads和test video value reads均为0。
- formal-seed step1→3同样为530个trainable tensors中458个变化，所有主模块
  finite且可达；72个Action Meta-LoRA A按分级zero-init延迟，配对B已变化。
  config现已seal为B20、fresh macro0→200、every25。

## Target-Spectral训练、correct400与内部分析完成（2026-07-31）

- sealed commit `aa9d89a`的fresh run自然完成macro0→200：200行finite metrics、
  4,800个single-video conditions、96,000 queries、every25的8个完整checkpoint；
  training body `3920.15s`，终点loss/grad为`.10023/.06443`，峰值allocated/
  reserved约`77.08/83.65GB`。全部checkpoint manifest通过校验。
- macro50/100/150/200在相同8×50 fixed states、每task teacher video无放回
  0–49和同一RNG配对下得到`30/12/18/34`。macro200逐task为
  `12/0/0/6/13/1/1/1`；31/34成功集中在三个tasks。独立审计确认四份结果、
  36/36 shards、400 LoRA caches、worker return codes和hash链完整。
- 按门停止行为评测与续训。CPU rank/layer/video分析完成，产物SHA256
  `4d7dfc68efa84b9863b8a6d9b7d4ab717f529018992b6c316c06320631d10a89`；
  Target m200 stable rank/norm为`3.3245/25.87`，v6 m200为
  `1.00017/94.71`。Target q/v跨层余弦仅`.032/.066`、layer-energy CV高达
  `1.294/.805`，确认强制正交拆散了v6高增益协调方向。
- GPU4–6现场有他人进程，按owner要求没有挤占；只在空闲GPU7单卡完成16条件
  non-rollout内部探针。首轮被旧probe字段`transition_norm`拦截，核对当前模型
  后只把capture更新为`transition_key_norm`，原始复现随后成功。该故障仅影响
  disposable instrumentation，不影响训练或正式correct400。
- 内部结果证明Core/Procedure工作且order差异传到LoRA/action；失败集中在
  compiler写入几何和functional-loss→closed-loop错位。当前暂不正式训练，
  只允许使用GPU4–7中现场空闲卡分析；下一步由owner讨论后封存保留v6主方向、
  仅增加可选视频innovation rank的架构。

## v5.2 task-complete控制实现（2026-07-31）

- owner解除临时训练暂停，要求先完成原版v5.2拓扑与成熟full24 fast-decay400
  训练的两小时对照，再基于全部证据设计下一架构。
- 当前main原位恢复v5.2 Core/Procedure/320-slot AdaLN compiler，删除
  Target-Spectral-only compiler源码；保留现有cost-balanced long-first
  task-complete训练、raw-video信息墙、checkpoint/resume与evaluator。
- 新config固定B20、4 ranks×6 tasks、24 tasks/macro、480 queries/macro、
  LR`3e-4`、warmup17、decay400到`1e-5`、every25、0→200→400。
- 精确参数预算`10,237,704`；聚焦模型/训练/checkpoint/evaluator合同
  `41 passed`，全仓首轮`188 passed/1 message-only failure`，该消息测试已
  同步修正。architecture guard为REVIEW、无hard violation，active source
  净删约609行。

## v5.2 task-complete B20 profile与formal seal（2026-07-31）

- main `62598d3`全仓fresh回归为`189 passed`；核心model/temporal/
  video-program逐文件SHA与正式v5.2 commit `529da6b`完全一致，随后已push。
- profile seed172在GPU4–7完成三个full24 macros；首步包含task38/demo36的
  105 sampled frames。三步loss/gradient finite，峰值allocated/reserved为
  `76,967,302,656/83,638,616,064 bytes`；B20通过，不触发B16。
- 现场GPU4和6有未干扰的他人轻量进程；后两步均值`18.943 queries/s`、
  `142.074 macro/hour`。因此相同400-macro科学预算在当前共卡吞吐下约需
  169分钟body，formal wall上限如实放宽到190分钟，而不是偷偷减少updates。
- 独立formal teacher seed `20260722`先fresh到macro1，再从完整checkpoint
  exact-resume到3。metrics、LR、task/video/query cursor连续为`1/2/3`、
  `480/960/1440 queries`和`24/48/72 videos`，validation/test action与test
  video读取均为0。
- formal-seed step1→3的519个trainable tensors中447个变化，Core、Procedure、
  compiler、factor heads和三条semantic projection/Meta-LoRA主路径均可达；
  72个Action Meta-LoRA A因zero-B与BF16短profile分级暂未量化变化，配对B全部
  变化，且旧v5.2 step100→900证实72/72 A随后均变化。配置已seal为fresh
  `0→200→400`、B20、every25。
- 清理108个已核验无进程引用、可由评测重建的`writer_lora_cache`，删除约
  `105.77GB`；正式result rows、contracts和checkpoints均保留。个人占用从
  `453.12GB`降到`347.35GB`，这些缓存未进入回收站、只能重新生成。
