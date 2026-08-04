# EMBER Active Session Handoff

更新时间：2026-08-04 UTC。本文只记录迁回 BCI 前后的当前真相。历史执行流水仍在
`progress.md`，证据与解释仍在`findings.md`及各架构设计文档；不要用其中旧的
“当前”“下一步”覆盖本文。

## 0.0 当前状态：step50 coverage仍未过门，AS准备exact-resume到75

- owner已恢复持续推进并要求科学/工程问题自行深入分析。当前唯一活动方法为
  `docs/action_forecast_writer_relative_flow_credit_design.md`：恢复v6条件生成路径做
  fresh独立AS cold start，随后关闭teacher action入口，以full24 official random-reset
  reward、同task K4 leave-one-out advantage和per-CFM-sample PPO/SPO ratio训练Writer。
  one-shot、信息墙、single checkpoint、不使用subagent和最多6张live空闲A40不变。
- canonical源码已原位完成替换：旧success-only self-imitation、flat task-local RL、
  Target-Owned与Direction Store活动实现均退役；success/failure executed prefixes、
  deterministic Nmc4 flow credit、实际world-size full24 assignment、deferred NCCL、完整
  cycle checkpoint/resume和raw-video evaluator接线已落在唯一Writer/RL路径。聚焦
  RL/reward/eval 43项通过；全仓按内存边界拆为135+75项，合计210项通过。
- v6 AS profile root为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_profile_r6_bci_20260804`：6 ranks×4
  tasks、logical B20、policy microbatch2、16-frame chunk，三步wall
  `33.464/30.886/30.977s`，峰值allocated/reserved
  `34,948,858,880/44,816,138,240` bytes，最长105帧、0 OOM/clip、0 validation/test
  action reads。step1按zero-init只有factor梯度，step3五个声明主block均finite/nonzero。
- 独立resume root为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_resume_smoke_r6_bci_20260804`：fresh
  0→1再exact-resume1→3，合同`1d2290ea...d457a87`不变，metrics严格1/2/3、累计
  1,440 queries与72 one-shot videos，source policy trainable=0。profile结束后
  `gpu02:1,2,3,4,5,7`已自然回到10--11MiB；0和6始终属于其他用户且未触碰。
- 正式AS root
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_formal_r6_b20_seed7_b75cb19_20260804`
  已完成fresh0→25并exact-resume25→50：累计24,000 queries、1,200 videos、50个finite
  macros、0 OOM/clip，两个segment wall=`810.991/816.191s`，完整checkpoint=
  `checkpoints/step_00000050`。这是独立v6 cold start，不含profile或历史权重。
- 有效reward profile root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro025_r6_bci_retry2_6ff7599_20260804`：
  96条K4 ledgers、28,085 actions、25 successes、12/24 task success coverage、9 mixed、
  3 all-success、12 all-failure；coverage失败。两epoch ratio/clip/grad健康，峰值reserved
  `45,183,139,840` bytes，说明机制与A40负载通过，但其cycle1 checkpoint禁止作为正式
  reward cold start继续。
- step50有效reward profile为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro050_r6_bci_retry1_e5bca71_20260804`：
  96条K4 ledgers、25,878 actions、38 successes、14/24 task coverage、10 mixed、4
  all-success、10 all-failure；suite success spatial/object/goal/libero10=`9/12/11/6`，
  coverage=`4/4/4/2`，仍未过门。相对step25的96个task/cursor，env seed、初态hash、
  policy seed、teacher demo和共同noise prefix全部严格一致；gained/lost/retained=
  `19/6/19`，coverage`12→14`，说明总体积累同时仍有task换手。
- step50首次profile的96条pre-update ledger与上述有效root逐文件字节级一致，但其
  outcome-skewed local credit耗时使0-mixed rank提前进入NCCL all-reduce，480秒后被
  watchdog终止，没有metrics/checkpoint。`e5bca71`加入每epoch本地反向后的独立
  FileStore all-rank-ready；在原六卡、96 rollout、两epoch规模重放后完成finite更新和
  cycle1 checkpoint，0 watchdog/traceback。两epoch ratio范围=`[.9905,1.0094]`和
  `[.8555,1.0559]`，clip=`0/0`，grad norm=`.02872/.02697`，峰值reserved
  `40,342,913,024` bytes。
- 下一步只从上述AS step50 exact-resume同root到step75，再以新的pre-update K4 cycle
  重做coverage。不得借历史macro400/best或reward checkpoint，也不得按outcome改变task
  或seed合同。
- 本轮根修RL环境池未绑定sealed asset cache，以及非连续选卡时把local rank误当物理EGL
  card的问题；有效run contract已记录physical GPU=`1,2,3,4,5,7`。相关长期规则已写入
  `AGENTS.md`，诊断root不进入科研结论。

## 0.0a 历史裁决：Policy-Target-Owned Factor已负裁决

- owner授权下的本轮架构、profile、fresh正式训练、四点rollout和全部预注册内部分析
  已完成；按owner此前要求，现在暂停，不启动下一架构、训练或评测。长期
  single-checkpoint `correct>150/400`目标未完成；strict one-shot、不使用subagent、
  效率优先和每次live选择`gpu01/gpu02`最多6张空闲卡的边界继续有效。
- clean pushed`34be4a0`在frozen worktree从fresh identity完成macro0→200：200次
  full24 update、96,000 logical queries、4,800 one-video conditions、8 checkpoints，
  wall`6678.957s`；0 clip/OOM、峰值allocated/reserved`33.696/38.729GiB`、0
  validation/test action reads。正式root为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_34be4a0_20260804T051244Z`，runtime contract
  `6af3b4fe...904b`；profile或历史Writer权重均未进入。
- 50/100/150/200 strict paired correct400=`99/76/86/68`，breadth=`6/6/7/5`；逐task
  为`9/0/1/44/38/6/1/0`、`5/0/4/33/28/2/0/4`、
  `7/0/1/26/39/10/1/2`、`7/0/0/31/27/2/1/0`。相邻gained/lost=
  `15/38,35/25,18/36`，union/intersection=`136/37`、envelope gap37。winner
  macro50=99，低于Direction Store129和v6-fast143；Long-2四点全0，故不续400。
  四个sealed roots为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_correct400_noreplacement_seed7_macro{050,100,150,200}_34be4a0_20260804`；每个root均有400 unique rows、42 shards、
  9 workers exit0、50个teacher demos/task且无retry/adoption。
- macro50 refs1内部root为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_macro050_internal_refs1_seed7_34be4a0_20260804`：
  六rank、8 tasks、correct/same/wrong/shuffled/reversed完整，wall`100.864s`，0 rollout、
  0信息墙违规、strict replay/rank gauge/checkpoint unchanged全通过。分析完成后六张GPU
  自然释放。
- 76 heads确实解除旧policy-target硬共享：q/v cross-layer effective-BA cosine从
  Direction Store`.9319/.9666`降到`-.00011/-.00030`。但correct LoRA norm均值仅
  `19.0257`，layer-energy CV=`1.9607`，q/v top-4占`.7329/.8529`，比直接SFT的
  `.464--.469/.544--.589`更过度集中；action heads能量占比仅`.000085`。
- same-task Program/factor/BA/action relative-L2为
  `.90933/.05842/.09119/.03161`：独立heads把BA差异放大，却没有写入等比例的
  policy-action有效方向。A/E、Core mean、Core-only、Program-only和memory reversal
  都能到BA/action，动态路径未断。高分Goal-6/Object-1对视频很不敏感，最敏感的
  Object-3只有6/50，也说明condition dependence没有与competence绑定。
- factor承担单task梯度能量中位数`69.25%`，24-task median cosine`.0040`、负pair
  `.4457`、full24能量保留`.0484`。CountSketch里task identity只解释factor方向方差
  `.0168`（随机基线约`.0048`），同task+demo隔50 macros的重现余弦仅`.0046`。
  正式拒绝policy-target sharing作为主要task-drift根因；最早剩余接口更新为
  condition-to-policy credit缺少稳定、闭环有效、跨随机query可累积的task/video方向。
  下一轮不得继续加heads、layer gate/scale、强制SFT profile或监督专用trick。

## 0. BCI运行交接（优先于下文旧A100操作描述）

- EMBER已迁至`/data1/user/ymdai/projects/EMBER`并使用项目`.venv`。source、data、
  tokenizer、checkpoint和output继续由CLI显式传入；每次进程还需显式设置
  `EMBER_STORAGE_ROOT=/data1/user/ymdai`、owner容量上限和项目内
  `EMBER_LIBERO_ASSETS_ROOT`，不能假定`.env.local`自动提供这些值。不要再把
  `/data/ymdai`绝对路径写入新命令或新artifact。
- 当前VR A40配置是
  `configs/pi05_as_writer_semantic_factor_basis_variance_reduced_long105_profile_v1.json`；
  它使用6 ranks×4 tasks、16-frame encoder microbatch、逻辑B20和policy microbatch2，
  不固定物理GPU编号。一个LoRA仍读取完整B20随机样本，只把frozen-policy forward
  切成10个B2并按样本数加权；full24 raw mean、一次AdamW与scheduler合同不变。
- BCI四卡迁移验收已完成：NCCL/BF16 collective通过，真实Writer fresh 0→1通过，
  exact resume 1→2通过，最长真实视频105帧，峰值CUDA reserved
  `44,853,886,976` bytes；随后8/8 validation smoke rollouts完成并聚合。
- 当前torch/NCCL在gpu02直接P2P传输会挂死；EMBER环境已自动设置
  `NCCL_P2P_DISABLE=1`使用稳定的共享内存传输。无需在每条命令里重复设置。
- 评测preflight已移除对整个个人目录的递归`du`和个人容量硬门，只保留快速文件系统
  余量及所选GPU现场检查。不要恢复全目录扫描或A100的固定GPU4--7约束。
- 验收root为
  `/data1/user/ymdai/projects/EMBER/runs/acceptance/ember_bci_gpu_acceptance_20260803T1232`；
  迁移证据在
  `/data1/user/ymdai/projects/EMBER/evidence/migration/20260803/gpu-acceptance/`。
  这些profile/smoke checkpoint只证明运行链路，后续VR正式实验仍须fresh identity，
  不得从验收权重warm-start。
- 验收结束后无EMBER训练、评测worker或tmux进程，四张验收GPU均已释放。
- owner现授权每次实时比较`gpu01`与`gpu02`，只用空闲卡且总数最多6张。2026-08-03
  07:00 UTC附近快照中`gpu01`八卡均忙，`gpu02`的0/1/2/3/4/7空闲，因此工程profile
  只使用这六卡；5/6有他人任务且从未触碰。该分配是易变快照，每次launch必须重查。
- 六卡NCCL/BF16 smoke通过；未冻结工程profile在
  `runs/acceptance/ember_bci_vr_effective_b20_micro2_r6_profile_20260803T1600/train`。
  fresh0→1再exact-resume1→3完成；每步24 tasks、480 logical queries、240 physical
  forwards，三步wall为`33.973/31.686/31.240s`，loss为
  `.157415/.152420/.148585`。峰值allocated/reserved为
  `34,970,270,208/47,108,325,376` bytes，五个主block从macro2起finite/nonzero，
  validation/test action reads为0，step1/2/3 checkpoint齐全。由于运行时源码未提交，
  这里只算工程证据；提交后必须fresh重放0→1与exact-resume1→3再seal。
- 实现经23项focused、226项全仓CPU回归、compileall和architecture guard无hard
  violation后提交/push为`391f183`。同一logical-B20配置随后从clean pushed commit在
  `runs/acceptance/ember_bci_vr_effective_b20_micro2_r6_profile_391f183_20260803T0735Z/train`
  完成fresh0→1与exact-resume1→3：三步`33.514/32.050/31.326s`，loss
  `.157415/.152418/.148564`，峰值allocated/reserved
  `34,970,270,720/47,108,325,376` bytes；最长105帧、1440 queries、72 videos、
  五主block从macro2起finite/nonzero、validation/test action reads为0。contract为
  `31ea4bc9...55de0`，step3 payload为`2b50bafd...618f7`，profile已seal。
- 第一次frozen resume尝试在第二条invocation前出现一次15分钟setup collective卡死；
  只终止本方进程。随后相同六卡`all_gather_object`/`broadcast_object_list`最小探针
  通过，同一原命令重试也完整通过，因此目前只能标记为未复现的一次性runtime观察，
  不能伪称软件根因。formal fresh0→200不读取profile权重，launch保留live timeout与
  进程/GPU监控。
- commit`6f18499`的首次BCI formal在macro10前发现配置仍使用longest105 profile专用
  `teacher_video_seed=172`，而同一配置的sealed字段及ordinary SFB正式基线都要求
  formal seed`20260722`。本方在任何checkpoint前主动终止，六张卡完整释放；partial
  root和log只作aborted合同审计，禁止resume、评测或性能引用。修复把实际seed切回
  `20260722`，并在config loader增加sealed formal seed不一致即fail-close的回归门。
  root内`aborted_contract_incident.json`记录10 rows/0 checkpoint及四份证据hash，文件
  SHA256=`9d5d03b8...cf9907`。

### 0.1 BCI VR fresh 0→200 formal retry1 launch contract

- canonical workspace为`/data1/user/ymdai/projects/EMBER`；launch必须使用包含本段
  记录的clean commit，且现场核验`HEAD == origin/main`。分支名不改变run identity，
  精确branch/commit由自动`run_contract.json`记录。
- sealed config为
  `configs/pi05_as_writer_semantic_factor_basis_variance_reduced_long105_profile_v1.json`，
  当前SHA256=`333e4d6a...044492`，实际`teacher_video_seed=20260722`并由loader与
  `formal_teacher_video_seed_after_profile_seal`强制一致。source step1000 manifest SHA256=
  `c236cb2d...cd6bf`，tokenizer SHA256=`8986bb4f...8fc6`；source selected raw policy
  identity仍取sealed manifest的`60ea7ee8...df36`。data root为项目内迁移核验后的
  filtered LIBERO数据；launch执行sealed size/schema检查并按合同跳过重复全量SHA。
- output root固定为
  `/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_semfactor_vr_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_retry1_20260803`；
  启动前必须不存在。log固定为
  `/data1/user/ymdai/projects/EMBER/runs/logs/ember_vr_bci_rawfull24_r6_b20_micro2_seed7_formalvideo20260722_retry1_20260803.log`，
  tmux固定为`ember_vr_bci_r6_b20_seed7_retry1_20260803`。错误seed的旧root/log原位
  保留并明确禁止作为retry输入。
- 规模为fresh macro0→200：96,000 logical action queries、4,800 one-video conditions、
  48,000 physical B2 policy forwards、8个every25 checkpoints。6-rank DDP每rank 4 tasks，
  logical B20、full24 raw mean、一次clip/AdamW/scheduler不变；profile checkpoint绝不
  warm-start。estimated peak新增容量按1.5GiB计；2026-08-03 08:03 UTC `/data1`
  personal quota为`256,638,532/1,073,741,824 KiB`，共享余量86TiB。
- exact inner command固定为：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,7 torchrun --standalone --nproc-per-node=6 \
  scripts/train_as_writer.py \
  --config configs/pi05_as_writer_semantic_factor_basis_variance_reduced_long105_profile_v1.json \
  --mode formal \
  --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model \
  --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir runs/outputs/pi05_as_writer_semfactor_vr_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_retry1_20260803 \
  --skip-data-sha
```

- GPU assignment不是永久保留：launch前重新比较`gpu01`/`gpu02`，只在六张目标卡仍
  candidate时使用上式；若设备集合变化则先更新实际`CUDA_VISIBLE_DEVICES`与现场记录。
  `.venv` activation提供`NCCL_P2P_DISABLE=1`。启动后监控commit/root/device、invocation、
  metrics推进、finite/clip/OOM、quota和他人进程；15分钟内无invocation/start则只停止
  本方进程组并保留证据。
- formal自然完成后只评测single checkpoints 50/100/150/200的paired correct400；不
  根据loss挑点、不融合checkpoint。只有absolute/breadth/trend/internal path过门才
  exact-resume同一root到400；任何合同改变都fresh。profile和失败启动root不得作为
  resume来源。

### 0.2 BCI VR 0→200正式结果与阶段暂停

- 有效训练从clean pushed`d9130c9`和fresh identity自然完成到预注册stage stop200。
  canonical root仍为0.1节所列retry1 root；contract SHA256=
  `0f9ed99d...13599a`，config SHA256=`333e4d6a...044492`。200个macro严格对应
  optimizer step1--200，累计96,000 logical action queries、4,800 one-video
  conditions、48,000 physical B2 forwards，wall=`6619.670s`；all finite、0 clip、
  source trainable=0，validation/test action reads=0。8个every25 checkpoints的
  64/64 payload size与SHA均复验通过，8个512-row held functional panels完整。
- 在每次live GPU/quota preflight后，只使用`gpu02`物理0/1/2/3/4/7；5/6上他人进程
  从未触碰。A40 evaluator采用3 GPUs/panel、3 persistent replicas/GPU、3 Writer
  generators/GPU、generation batch4；正式root为：

```text
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0050_d9130c9_20260803
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0100_d9130c9_20260803
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0150_d9130c9_20260803
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0200_d9130c9_20260803
```

  四个root均为400 unique rows、42/42 complete shards、9/9 workers return0；每task
  50 states和teacher demos 0--49各一次，results/launcher-completion hash均通过。
  checkpoint之间以及各自ordinary SFB comparator的state、teacher video、env/policy
  RNG均400/400配对；成功提前终止只缩短noise数组，公共前缀逐项相同。
- paired correct400曲线与逐task结果如下；task列顺序为Long-1/2、Goal-3/6、
  Object-1/3、Spatial-1/3：

| macro | correct | breadth | per-task successes |
| ---: | ---: | ---: | --- |
| 50 | 76 | 7 | `3/1/0/37/29/4/1/1` |
| 100 | 88 | 4 | `4/0/0/37/25/22/0/0` |
| 150 | 126 | 7 | `4/2/1/41/42/34/0/2` |
| 200 | 107 | 5 | `8/0/0/39/33/24/0/3` |

  single winner为macro150=`126`，比严格最低通过分151少25、比v6-fast winner143少17，
  也未超过ordinary SFB winner127。相邻gained/lost为50→100=`30/18`、
  100→150=`49/11`、150→200=`21/40`；四点success union/intersection=`158/49`，
  single envelope gap=`32`。breadth在`7/4/7/5`之间切换，VR没有解决task漂移。
- 对同点ordinary SFB的paired delta为`+7/-3/+8/-20`；对应gained/lost为
  `23/16`、`18/21`、`33/25`、`21/41`。macro200下降覆盖7/8 tasks，只有
  Object-1和Spatial-3略升。VR winner150相对source base gained/lost=`83/5`，说明
  Writer产生真实新能力；相对v6-fast macro400为`27/44`，没有形成新上限。
- 与ordinary SFB完全matched的前200步诊断中，VR仅把same-task successive all-block
  CountSketch cosine平均提高`.002634`、factor提高`.005104`，raw mean/sample energy
  retention提高`.001914`、factor提高`.001121`；51--100段方向cosine反而更差，
  151--200段energy retention也没有改善。这个效应小、分阶段反复，不构成material
  gradient stabilization。
- held functional loss在macro200达到VR四点最好`.129146`，且优于同点SFB
  `.131776`，但closed-loop同时从VR macro150的126跌到107、并比SFB macro200少20。
  逐task loss与绝对成功的相关性主要反映任务难度；24个相邻task变化的Spearman仅
  `.263`。因此正式拒绝“可约flow Monte Carlo方差是主要漂移根因”，把最早剩余接口
  升级为functional action surrogate与source-policy closed-loop有效流形错位。
- owner要求在四点rollout与全部分析完成后先暂停。当前无EMBER tmux、训练/eval
  worker或本方GPU占用；不续到400、不做五臂、不启动下一架构/训练目标，等待owner
  看完本次状态后继续指示。长期`>150` Goal仍未完成。

### 0.3 owner恢复推进与Semantic Direction Store

- owner随后已解除上述阶段暂停：继续严格one-shot，取消Writer参数量软上限，优先
  重构条件生成方向的存储/组合，并允许配套修改训练方式；仍不使用subagent。推进以
  效率优先，只做shape、信息墙、identity、freeze、gradient、OOM、resume和正式结果
  所需的聚焦检查，不重复全量hash或无关历史扫描。
- 当前不再把“held functional loss不能预测rollout”本身称为task漂移根因。更直接的
  内部边界是SFB route已task-conditioned，而shared factor仍占约97% gradient energy、
  task mean只保留约4.2%条件能量且一阶方向持续轮换。
- 新设计authority为
  `docs/action_forecast_writer_semantic_direction_store_design.md`：frozen text-only
  language anchor只用24 train languages建立8个固定semantic centers；每task稳定等权
  top2，每个store拥有完整独立1024→256→factor-width参数。完整Core/A/E/D仍是唯一
  factor value，Writer实际参数37,355,776。在进入formal前，canonical实现、
  24-train-language center authority、61项focused CPU合同与clean六卡profile已完成；
  当时尚无效果结论。
- 当时封存的下一执行顺序是从sealed formal seed`20260722`和clean origin-main
  fresh0→200，再做
  50/100/150/200四点paired correct400；不复用profile/VR checkpoint或
  Latin/antithetic estimator。
- clean `7b13b6c`首次六卡profile在训练循环前复现NCCL 480秒heartbeat失败；六rank日志
  均明确`only active collectives: 0`，且当时只有rank-local source CUDA构造在运行，
  不是Direction Store collective、OOM或科学non-pass。该root已停止且禁止resume。
  根因修复为延后NCCL生命周期：rank先完成local policy/Writer/optimizer CUDA构造，
  经独立FileStore all-rank-ready rendezvous后才允许任何rank建process group；不得让
  快rank提前创建NCCL，也不得用放宽heartbeat或timeout封口。
- `78d8b4f`重放确认生命周期修复后，六rank统一进入`SeqNum=1/ALLREDUCE/Numel=1`，
  随后暴露BCI迁移期已裁决的第二层transport合同：显式launch漏传
  `NCCL_P2P_DISABLE=1`，direct P2P/CUMEM在600秒超时。相同`gpu02:0--5`六卡加该变量
  后，scalar sum=`21`、BF16 matmul finite及第二次all-reduce在10.5秒内全部通过。
  因此BCI A40 launcher与代码现同时显式/fail-fast要求SHM transport；第二root同样
  aborted且禁止resume。
- clean `eaa8bce`随后在精确空闲拓扑`gpu02:1,2,3,4,5,7`完成两次collective sum21、
  all-rank CUDA-ready、NCCL与run-contract发布，证明两层多卡根因均已越过；进入step0
  后由`as_step.py`一份退役的重复method白名单拒绝新Direction Store method。canonical
  `as_config.py`此前已完整验证该conditioning合同，因此修复是删除第二份字符串白名单，
  让step owner只执行已验证合同；该root无metric/checkpoint，不跨commit resume。
- clean pushed`1d0507e`最终在`gpu02:0--5`完成fresh0→1和exact-resume1→3，contract
  `749773d8...8fd6`。三步`33.451/31.823/31.025s`，loss
  `.150377/.152492/.142434`，最长105帧，峰值allocated/reserved
  `35,827,363,840/47,129,296,896` bytes；1,440 queries、72 one-video conditions，
  validation/test action reads=0且无clip/OOM。step2起五个主块全部finite/nonzero；
  配置现切回formal seed`20260722`并seal，正式run必须fresh identity。

### 0.4 Semantic Direction Store正式结果、内部裁决与当前暂停

- clean pushed`91feeef`从fresh identity在`gpu02:0--5`完成macro0→200。canonical
  root为
  `/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_direction_store_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_91feeef_20260803`。
  200个macro累计96,000 logical action queries、4,800 one-video conditions、8个
  every25 checkpoints，wall=`6619.255s`；all finite、0 clip/OOM、0 validation/test
  action reads，峰值CUDA reserved=`39,806,042,112` bytes。profile/VR/SFB权重均未
  warm-start。
- 只用live preflight后`gpu02`六张空闲卡完成macro50/100/150/200的strict paired
  correct400；gpu01持续有他人任务，gpu02:6有他人进程且从未触碰。四个root依次为：

```text
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0050_91feeef_20260803
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0100_91feeef_20260803
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0150_91feeef_20260803
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0200_91feeef_20260803
```

  state、teacher demo和policy RNG公共前缀全部严格配对，0 retry/failure。task顺序为
  Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3：

| macro | correct | breadth | per-task successes |
| ---: | ---: | ---: | --- |
| 50 | 129 | 7 | `7/2/0/42/45/31/1/1` |
| 100 | 107 | 7 | `5/1/1/37/37/22/0/4` |
| 150 | 120 | 7 | `9/2/0/40/40/26/2/1` |
| 200 | 129 | 5 | `10/0/0/38/41/36/0/4` |

  macro50与200同分，按更高breadth和更早成本选macro50为唯一winner。相邻gained/lost=
  `17/39,43/30,27/18`，四点union/intersection=`174/65`、single envelope gap45。
  相比SFB macro50提高60，但未超过v6-fast143或严格门151，且后续仍明显换手，因此不
  续到400、不做五臂。
- step133的task-pair梯度分层显示shared0/1/2 stores的factor cosine均值为
  `-.00043/.00664/.02249`：fixed semantic stores局部化了干扰，但store内部仍近正交。
- winner macro50的完整refs1五条件内部分析成功root为
  `runs/outputs/pi05_as_writer_direction_store_bci_macro0050_internal_refs1_seed7_retry2_a115b06_20260803`。
  8 tasks的ordered top2数组均不同（其中`1,5`与`5,1`是同一无序组合），且route跨
  video固定；same-task-other的Program/factor/
  effective-BA relative-L2为`.93377/.01935/.03242`，shuffled为
  `.81049/.04731/.07193`，reversed为`.93086/.09808/.15963`。A/E与Core mean
  carrier均传到BA/action，动态路径没有断路，但其差异在compiler后被强压缩。
- 全部16个rank坐标active，effective LoRA norm均值`43.86494`，但rank90/rank99均为
  1、stable rank=`1.000043`、entropy rank=`1.000371`、top singular energy=
  `.999957`、B-column cosine=`.999971`。Direction Store改善了早期acquisition和参数
  ownership，却仍把public rank16写成几乎同一B方向；正式拒绝
  “factor parameter coexistence是主要完整根因”。
- 内部分析首次重放暴露assignment隐藏4-rank默认，第二次暴露final seal固定4 payload/
  每rank2 tasks。`f82c7cd`与`a115b06`分别把LPT ownership和Cartesian sealing绑定
  实际`world_size`；8项定向测试及clean六卡真实规模均通过。该根修与BCI transport/
  process-group生命周期规则均写入`AGENTS.md`并push到branch/main。
- owner要求rollout和全部分析后暂停了解现状。当前正式训练、四点rollout与winner内部
  分析均结束；没有EMBER训练/eval/analysis进程或本方GPU占用。不得启动下一架构、
  training target或GPU工作，等待owner明确继续指示；长期`>150` Goal仍未完成。

## 1. 当前边界

- owner此前授权在当前BCI上继续环境适配、架构/训练设计、profile、正式训练、严格配对
  评测和内部分析；目标是缓解task漂移，并使同一single checkpoint的correct aggregate
  严格超过`150/400`后继续提高。Direction Store rollout与全部内部分析完成后，owner
  最新边界是先暂停了解现状；当前不得自动启动下一实验。推进期间仍不使用subagent。
- 当前写分支为`codex/bci-continuation`，BCI新增输出只写项目`runs/`，证据写
  `evidence/`。下列A100窗口、旧分支和`/data/ymdai`只保留历史provenance。
- owner在迁移由另一session启动后重新开放约十小时A100 post-seal研究窗口，允许在
  原信息墙/split/安全合同和物理GPU4--7边界内继续架构、训练、评测与分析。窗口以
  `2026-08-02 19:18 UTC`起算，约`2026-08-03 05:18 UTC`硬停；操作上最迟`03:45 UTC`
  冻结新实验，为二次迁移留出时间。
- 已迁移封存基线为`f9a144c`；本轮所有Git与artifact都是post-seal delta，外部登记根
  为`/data/ymdai/migration_manifests/ember_postseal_20260802/`。迁移仍由另一session
  执行，本session不修改其现有副本，只提供增量清单。
- 本A100研究窗口的训练、评测、内部分析和GPU profile均已结束；当前没有需要继承的
  tmux、torchrun、评测worker或GPU实验。MemLLM同样没有活动实验。
- EMBER迁移封存基线为`f9a144c94e71bb44373d7247ed0fded2ed835305`；Semantic
  Factor-Basis仍是canonical Writer；A100最后push的VR实现commit为`50662a8`。
- Target-Bound Role-Preserving Program 已在远端分支
  `origin/codex/target-bound-role-program`实现，commit
  `b260a57a94dc21bd3446b212bfa42f71b037ce13`。它只完成 CPU shape、identity、
  causality、gradient、checkpoint 等结构验证；没有做 B20 profile、resume、训练或
  rollout。不得把它写成实验结果。
- Target-Bound已完成fresh0→200；macro50/100/150/200 paired correct400为
  `75/120/90/110`，winner macro100仍明显轮换，因此不续训、不做行为五臂。winner
  refs1证明remove-A、remove-D、causal-memory reversal均8/8过门，Core-only与
  Program-only都不能复现full BA；视频主路径真实到达BA/action，最早剩余失败接口是
  shared factor conditional coexistence。
- Semantic Factor-Basis只替换这一接口：Core以Q/K软选择四个unit-mean factor value
  bases，完整Core/A/E/D仍作为value；不加task ID、gate、scale或额外loss。精确参数
  11,159,296。`e87363f`的longest105 B20三macro及formal-seed fresh0→1/
  exact-resume1→3均通过，五个主block从macro2起finite/nonzero；seal/push commit为
  `f5ddfe3`。
- clean frozen`f5ddfe3`从fresh identity完成0→400、every25；不从profile/smoke
  warm-start。完整paired correct400为`69/91/118/127/117/81/126/120`，single
  winner仍是macro200。八点success union/intersection=`193/39`、single envelope
  gap=`66`；250→300 lost52、300→350 gained60，第二小时明确证明能力轮换而非成熟化。
  formal root：
  `/data/ymdai/outputs/ember/pi05_as_writer_semfactor_postseal_rawfull24_decay400_formal_r4_b20_seed7_f5ddfe3_20260802`；log：
  `/data/ymdai/logs/ember/pi05_as_writer_semfactor_postseal_resume200to400_r4_b20_seed7_f5ddfe3_20260803.log`。
- variance-reduced estimator保持SFB拓扑、objective期望、B20/full24/optimizer不变，
  只对flow time做exact-Beta Latin分层并对Gaussian noise做随机antithetic pairing。
  BCI正式0→200与四点correct400=`76/88/126/107`均已完成；机制改善小且非持续，
  held functional loss与closed-loop在macro200明确错位，方法已负裁决，不续到400。
- 迁移步骤、路径映射、资产分流和新 Codex 接手顺序统一看
  [`a100_to_bci_migration_handoff.md`](a100_to_bci_migration_handoff.md)。

## 2. 最新 closed-loop 结论

### 2.1 CV-ADR RAW

canonical root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_rawfull24_taskcomplete_decay400_formal_dev_r4_b20_seed7_254ade4_20260802_retry1
```

- fresh identity 完成 macro0→400，192,000 action queries、9,600 one-video
  conditions，all finite、0 clip，validation/test action reads为0。
- paired correct400 在 macro50/100/150/200/250/300/350/400 为：

```text
76 / 111 / 99 / 117 / 77 / 69 / 80 / 82
```

- single winner 是 macro200=`117/400`。第二小时不是成熟化：200→250为
  16 gained / 56 lost，后段 LoRA norm没有坍缩而行为持续退化，因此未做五臂。
- macro200与400的matched梯度方差分解显示，video主效应仅约
  `.1211%/.1060%`且0/24 tasks主导；query约`48.59%/49.53%`，flow及
  query×flow约`48.78%/48.50%`。24/24 matched train functional loss继续下降，
  correct400却`117→82`。
- 晚期factor block约占task-gradient energy的`94%`；参数段方向在低LR仍不稳定，
  held functional loss横盘。最可信根因是视频条件梯度低SNR、query/flow噪声、
  shared compiler写出与closed-loop有效流形错位共同作用，不是单纯LR、rank或norm。

内部根：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_rawfull24_macro0200_internal_exact50_seed7_ff988dc_20260802
```

exact50确认Core与Program两路都必要，但Action/order仍弱：remove-A只在1/8 tasks达
预注册门，remove-D为5/8；same-task effective-BA centered variance/sample energy
约`.10494%`，fixed-action中位变化约`.00856%`。LoRA norm`64.24`、stable rank
`1.0072`，所以不是Target-Spectral式增益或coherence坍缩。

### 2.2 CV-ADR normalized GROUP4

canonical root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_group4_taskcomplete_decay400_formal_dev_r4_b20_seed7_51c0ba5_20260802
```

- 完成1200 physical updates=200 cycles、96,000 queries、4,800 videos，all finite、
  1 clip；cycle50/100/150/200 paired correct400为：

```text
82 / 77 / 73 / 110
```

- single winner cycle200/step1200=`110`，低于RAW winner`117`，四点均值
  `85.5<100.75`；breadth6、top2占`71.82%`，未解决能力轮换，不做五臂。
- GROUP4比RAW保留更多source successes（42/48 vs 34/48），但没有共同获得更多新
  能力。effective norm反而`64.24→72.06`，held loss略低而closed-loop更差。
- exact50中A+D、remove-A、remove-D职责门由RAW的`8/1/5 of 8`降为`0/0/0`；
  Effect-only到full BA的relative L2由`.06744`降为`.01882`，contextual-memory
  reversal由`.00607`降为`.00311`。它学到更大、更coherent、却更static和
  off-manifold的写入。

内部根：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_group4_cycle0200_step1200_internal_exact50_seed7_51c0ba5_20260802
```

结论：normalized GROUP4、CP式负冲突解释和“减少optimizer gain即可稳定”均没有
获得支持。full24 raw mean很少直接对task candidate为负，pairwise negative cosine
不能自动解释漂移。

## 3. 架构×训练方法的关键反事实

以下四个single winner均从正式400-row paired artifacts逐行重验：

| 架构×recipe | correct | same | wrong | shuffled | reversed |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5.2 old recipe, step900 | 132 | 138 | 74 | 82 | 83 |
| v5.2 task-complete, macro400 | 120 | 109 | 107 | 111 | 124 |
| v6 old recipe, step500 | 121 | 122 | 111 | 84 | 47 |
| v6 fast task-complete, macro400 | 143 | 135 | 125 | 128 | 129 |

必须继承的解释：

1. task-complete在v5.2和v6上都压弱Procedure→effective BA/action与顺序margin，
   但correct absolute分别`-12/+22`；架构和recipe不能独立判死。
2. matched 150-video visits时，v5.2 task-complete相对old为`-81`，v6为`+16`；
   v6的Visual Transition/Core-conditioned transition是正证据，但其selected
   `+22`又几乎由一个Object task的`+24`贡献，不能说漂移已解。
3. old recipe每task-cycle六次Adam会恢复更强slots/AdaLN/动态写出，也会产生低
   breadth和近正交参数轨迹；退回old recipe不是解法。
4. post-v5的v7、v8、v10、Loom、Recenter、Core-Program、Prior、Target-Spectral、
   SPG、UCP、AP和CV负结果都与训练operator混杂。它们各自的局部失败接口有正式
   证据，但不能据此宣布其全部思想在任意训练方式下无效。
5. functional loss下降不等于closed-loop改善；强行高rank/正交、全局scale、gate、
   B-only residual、多video/LoRA平均或checkpoint融合都没有当前依据。

四格正式联合审计root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v52_v6_recipe_video_causality_audit_seed7_20260802
```

analysis SHA256：
`98371337e2cf1f7cec09d04e81445b419fc21c654fe173cb081a4b5e63092efa`。

### 3.1 Semantic Factor-Basis最终裁决

完整曲线与逐task（Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3）为：

| macro | correct | per-task |
| ---: | ---: | --- |
| 50 | 69 | `3/1/0/39/17/7/1/1` |
| 100 | 91 | `7/0/1/38/26/15/2/2` |
| 150 | 118 | `14/1/0/40/32/28/3/0` |
| 200 | 127 | `13/2/1/44/31/32/3/1` |
| 250 | 117 | `14/1/0/42/30/27/1/2` |
| 300 | 81 | `12/0/1/42/17/9/0/0` |
| 350 | 126 | `22/0/1/40/33/29/0/1` |
| 400 | 120 | `20/0/1/43/23/32/1/0` |

macro200相对source base paired gained/lost=`84/5`，证明Writer提供真实新能力；相对
v5.2-old为`49/54`、v6-fast macro200为`33/39`，没有提高现有上限。后半段
raw-full24 candidate-negative tasks始终为0，但gradient energy retention从
201--250的`.04443`降到351--400的`.04203`，factor share从`.9586`升到`.9691`，
same-task successive cosine从`.0676`降到`-.0099`。相邻checkpoint Adam一阶moment
近正交而二阶moment高度稳定，说明主要现象是条件方向/functional sample持续轮换，
不是全局mean即时伤害某些task，也不是scale统计失控。

macro200既有内部root仍是本版winner机制authority：

```text
/data/ymdai/outputs/ember/pi05_as_writer_semfactor_postseal_macro0200_internal_refs1_seed7_18d3e89_20260802
```

它证明route与A/E/D→BA→action工作，但task routing只部分解决shared-factor共存。由于
absolute未达到strong门，第二小时不新增same/wrong/shuffled/reversed 1600个rollout。

## 4. 当前代码与下一实验边界

Semantic Direction Store已原位替换为canonical Writer path；历史SFB、Target-Bound、
CV-ADR与VR由Git、frozen config和artifacts保存，不保留并行活动模型。核心职责为：

- 38个真实policy targets先读Core；
- target-bound地读取Action、Effect与Change；
- A/E/D使用private causal temporal channels和private rank reads；
- 16 rank coordinates最后展开；
- identities只进入Q/K，raw evidence进入V；
- frozen language anchor减去train24均值后固定等权选择top2/8 stores；
- 每个store独立拥有八个完整factor input/output heads，所有value仍来自完整Core/A/E/D；
- factor heads保持coherent near-rank1高增益，不加谱/正交/entropy约束。

当前A100临时授权窗口、BCI VR 0→200和四点正式评测均已完成。owner已恢复推进并
取消Writer参数量上限；紧邻候选不是继续给SFB加窄basis，而是用固定language语义地址
组合完整独立factor direction stores。设计见
`docs/action_forecast_writer_semantic_direction_store_design.md`；实现必须fresh schema，
当前fresh schema、37,355,776参数、center authority和focused CPU合同已完成，不复用
VR checkpoint或estimator。

不得从smoke/profile权重warm-start。当前完整设计为
`docs/action_forecast_writer_semantic_factor_basis_design.md`；VR设计及其正式负结果在
`docs/action_forecast_writer_variance_reduced_functional_estimator_design.md`。
Target-Bound设计与正式负结果保留在Git、该文档和post-seal artifacts中。

## 5. 迁移时必须保留的EMBER科学资产

- frozen source raw policy：
  `/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722`；
  policy SHA256
  `60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36`。
  rejected EMA和训练resume状态已清理；它现在是inference/source asset，不是完整
  source-SFT resume包。
- canonical feature cache v2：
  `/data/ymdai/outputs/ember/pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722`。
- 原迁移封存的60个正式/历史训练checkpoint roots、406个完成evaluation roots，
  加上post-seal的2个正式训练root、12个formal correct400 roots及内部analysis。
  它们是训练漂移与架构×recipe复核的唯一证据，不能只迁winner；精确增量只取
  `/data/ymdai/migration_manifests/ember_postseal_20260802/assets.tsv`。
- `/data/ymdai/logs/ember`、tokenizer、精确revision的426.57MB LIBERO simulation
  assets和`/data/ymdai/migration_manifests`。

cleanup已删除的profile/resume/reseal/cache路径若仍出现在历史文档中，表示工程
provenance，不表示artifact损坏，也不授权重跑。精确删除清单和SHA都在：

```text
/data/ymdai/migration_manifests/a100_cleanup_20260802
```

## 6. 新Codex接手顺序

本机Codex sessions、archive、auth、cache和worktree不迁移；它们不是authority。
新Codex在BCI上应先：

1. 核验Git HEAD、origin、工作区和迁移资产hash；
2. 完整阅读`AGENTS.md`要求的authority文件；
3. 优先读本文件、迁移handoff、`docs/execution_brief.md`、CV与Target-Bound设计；
4. 检查BCI实际路径并设置`EMBER_STORAGE_ROOT`、owner cap及
   `EMBER_LIBERO_ASSETS_ROOT`，所有source/checkpoint/tokenizer/data/output路径继续
   通过CLI显式传入；
5. 在owner恢复实验授权前保持无GPU作业状态。

旧Codex对话不能代替上述Git文档。任何与本交接冲突的历史“live”段落均视为过期。
