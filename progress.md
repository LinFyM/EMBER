# EMBER Progress and Handoff

最后更新：2026-07-21。

## 当前状态

- 长期Goal已建立且无token budget；不会因source base、Writer或任一局部阶段提前完成。
- 活动目标split仍为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；seal位于 `configs/libero_24_8_8_v1/`。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已下载，weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 fixed states上为 `0/400`。400 rows唯一、全部到suite horizon，result seal SHA256 `c78e92e9...20c2`；该结果不评价EMBER。
- Phase A source audit、71-task manifest、source-only normalization、官方recipe研究与hash seal已完成；高吞吐evaluator仍待完成。
- canonical π0.5 source-base full-SFT runner、真实8卡batch/EMA profile、atomic checkpoint与exact-resume分支核验已完成；formal配置锁定为8×microbatch32、global batch256、30,000 steps。
- 当前里程碑验证为57 tests passed、py_compile/JSON/checksum/diff checks通过；architecture guard为`REVIEW`且无hard violation，ownership、增长理由与旧路径retirement trigger已记录。
- 下一关键动作是提交/推送该可复现里程碑，隔离正式训练worktree并启动约45小时的8卡source-base训练；运行期间在不改其import/config/output的前提下继续推进dynamic evaluator与π0.5 Writer适配。
- 第一轮完整流程只跑一个training seed；不提前扩多seed或direct action-budget curve。
- 清理三套已封存证据的32GB probe checkpoint后，`/data/ymdai`为379,942,686,720 bytes。单checkpoint实测33,837,406,832 bytes；`keep_latest=1`的atomic替换峰值约447.62GB，低于500GB cap。

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
- canonical hashes：overlap audit `fe731127...cc003`、manifest `75453a20...2e54`、source-only normalization `e259ee6e...f7c4`、recipe `5772a136...b89494`；`sha256sum -c`通过。

不得根据后续source-base/Writer outcome修改这些source IDs。

## π0.5 source-base recipe与工程合同

- 官方anchor固定为OpenPI `15a9616...ccac`、LeRobot `30da8e6...76ce`：full action-SFT、AdamW `(0.9,0.95)`、eps `1e-8`、weight decay `1e-10`、clip1、peak LR `5e-5`、10k linear warmup后constant、EMA `0.999`、30k steps、global batch256。
- source base采用full-SFT并最终直接冻结policy/EMA，不叠shared source adapter；下游统一LoRA合同为18层action expert q/v加action_in/out共38 targets、rank/alpha16、dropout0、B=0 identity init。
- 8卡profile从microbatch1/4/16/32提升到约13.82/31.55/44.69/47.45 examples/s；选定m32时每卡allocated/reserved为67.18/71.30GB，loss/gradient finite，八卡角色与进程数对称。最终smoke使用pinned OpenPI精确的`q99-q01+1e-6` quantile分母，metrics SHA256 `bf952859...aa96`。
- checkpoint包含policy、EMA、optimizer、scheduler、per-rank Python/NumPy/CPU/CUDA RNG、sampler/data/metrics cursor与完整hash manifest。两次从同一step1恢复到step2得到相同loss `0.3457298893481493`、grad norm `4.03354549407959`与逐rank state hashes；独立NCCL启动后policy/EMA最大末位差为`1.49e-8/3.73e-9`。compact evidence SHA256 `16137fa1...b1e`。
- formal fresh launch要求clean且HEAD已推到`origin/main`；resume由保存的commit/contract约束，不因后续`origin/main`前进失效。每次invocation另存当时完整Git观测。

## Canonical code ownership与生命周期

- `pi05_source_corpus.py`只拥有specification filter、data seal和source normalization；`pi05_source_setup.py`只拥有model/data/distributed setup；`pi05_source_contract.py`只拥有launch/resume contract；`pi05_source_checkpoint.py`只拥有atomic state；`pi05_source_training.py`只拥有训练编排与step loop。
- `scripts/train_source_base.py`是唯一活动π0.5 source-base入口；拆分这些owner是因为严格加载、52.7GB一次性校验、DDP训练与33.8GB atomic checkpoint是独立故障边界。
- 旧`source_base.py`/`source_base_checkpoint.py`仍被历史SmolVLA模块import，只作provenance；不得作为活动入口。π0.5 source-base训练与40-task evaluator达到功能对等后，先迁移剩余通用依赖，再删除旧可执行路径，不保留双canonical runner。

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

1. 验证、commit、push Phase A seal与source-base canonical runner。
2. 创建隔离worktree，fresh live GPU/storage preflight后启动8卡formal source-base full-SFT。
3. 训练运行期间在另一worktree推进cost-balanced dynamic evaluator、persistent env/model与统一1/2/3 replicas真实rollout/s profile。
4. source base checkpoint产生后按预声明快速screen全部40 target tasks；只有跨多个tasks出现真实成功才进入共同冻结地基。

## 历史边界

旧SmolVLA 70/10/10曾完成到旧Phase F并留下真实结果，但与当前π0.5、split、one-video和source-base合同不兼容。只能用作经验/provenance，不能复用checkpoint、normalization或runner。
