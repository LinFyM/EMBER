# EMBER Progress and Handoff

最后更新：2026-07-21。

## 当前状态

- generic π0.5 feasibility Goal已完成；结果提交为 `27593d1`。本次 authority/handoff 更新在其上承接新的长期协议。
- 活动目标split仍为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；seal位于 `configs/libero_24_8_8_v1/`。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已下载，weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 fixed states上为 `0/400`。400 rows唯一、全部到suite horizon，result seal SHA256 `c78e92e9...20c2`；该结果不评价EMBER。
- owner已决定下一步先从generic π0.5训练共享π0.5-LIBERO source base，再推进AS-Writer、RL-Writer、Source-SFT、seen/wrong-video、final32、test RL和联合direct oracle。
- 第一轮完整流程只跑一个training seed；不提前扩多seed或direct action-budget curve。
- `/data/ymdai` formal launch前约354GB，500GB个人cap下约146GB headroom；任何新source data/cache/model构建前重新实时测量。

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

## Source-base corpus audit起点

只读本地官方suite已确认：

- LIBERO-90 task44：`KITCHEN_SCENE9_turn_on_the_stove`，language与target `libero_goal` task7相同；target task7为当前test。
- LIBERO-90 task77：`STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`，language与target `libero_10` task5相同；二者scene/BDDL不同。
- Spatial/Object与LIBERO-90没有exact language/name/BDDL overlap；Goal/Long各有上述1条exact language overlap。当前只检查了直接字符串重合，尚未完成semantic/composition audit。

下一session必须先完成完整specification-only audit，再封存过滤后的active source IDs。至少不能未经处理直接把LIBERO-90写成90/90 source tasks。

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

## 下一session第一批动作

1. `get_goal`；若无active Goal，按 `docs/new_session_prompt.md` 创建长期Goal，不设token budget并复核。
2. `git pull --ff-only origin main`、status、完整阅读authority。
3. live检查8卡owner/process、CUDA/PyTorch、storage和已有LIBERO-90数据/cache；不得启动训练前盲目复制。
4. 调研并锁定成熟π0.5 action-SFT/LoRA recipe与官方参数。
5. 完成LIBERO-90↔目标40 overlap audit、active source manifest、normalization/data hashes。
6. 并行推进高吞吐evaluator的cost-balanced scheduler；随后profile并启动source-base短训练/40-task快速screen。

不要停在计划复述或只写第二套runner；安全检查通过后直接推进Phase A/B。等待下载、索引或训练时继续推进互不污染的后续代码。

## 历史边界

旧SmolVLA 70/10/10曾完成到旧Phase F并留下真实结果，但与当前π0.5、split、one-video和source-base合同不兼容。只能用作经验/provenance，不能复用checkpoint、normalization或runner。
