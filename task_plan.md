# EMBER Task Plan

## 当前交接计划：Video-Conditioned Expert-Manifold（2026-08-08）

- [x] 完成K4 Phase-Aligned正式训练、四点、winner五臂与8-task refs1：correct=
  `88/108/80/99`，winner五臂=`108/115/94/101/121`。wrong显著更差且BA/action变化
  material，证明视频未被旁路；但reversed更高、LoRA stable rank约`1.00021`、最后
  50步gradient retention约`.04`，task轮换未解。本方法负裁决，不resume/warm-start。
- [x] 封存
  `docs/action_forecast_writer_video_expert_manifold_design.md`：视频是唯一dynamic value，
  用frozen π0.5 joint-video innovation到phase16；先训24套policy-effective task experts，再用
  168个`16×512`topological chunks和交替chunk/rank axial decoder重建完整LoRA。
- [x] 实现task-local sampler/checkpoint、六worker ownership与单卡多task builder；clean`174d292`
  在A40 B16完成fresh0→1、resume1→3和contiguous0→3，科学metrics及step3 adapter精确一致，
  峰值reserved`21,313,355,776` bytes，formal config已seal。
- [x] clean`81101fe`用六个independent workers完成24个rank-16 experts统一step1000；正式root=
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`，24/24 completion、
  72个step250/500/1000 checkpoints、约562MiB，不读任何held action。
- [x] 实现task-expert bank canonical evaluator与统一step几何分析；实现只含action-hidden
  phase16×3072 task-span+Action-Expert video innovation的hashless feature cache、168-chunk axial decoder、direction/scale
  reconstruction、六rank task-complete exact-resume meta trainer和one-shot strict paired evaluator。
  retained实现已并入`codex/bci-continuation`；当前尚未作A40 profile，formal cache/meta仍由config
  阻塞；K4 executable待新Writer profile通过后才退役。
- [ ] 完成expert bank统一step250/500/1000 official development-train rollout与几何裁决；若曲线仍
  有充分上升依据，从clean`81101fe`的frozen worktree沿原root统一resume到2000，再选择唯一expert
  step；不得按task挑不同checkpoint。
- [ ] live profile并封存train24×50 frozen feature cache；完成meta-Writer六卡fresh0→1、
  exact-resume1→3、finite/OOM/梯度与任务等权合同后才seal formal。
- [ ] 完成A40 profile、identity-fresh meta训练、strict paired correct400曲线、五臂视频因果、
  task drift和expert→generated LoRA→action机制分析；根据最早失效接口迭代。
- [ ] 同一single checkpoint strict correct必须`>150/400`且继续提高absolute、breadth、
  稳定积累与视频特异性。
- [x] owner已完成讨论并恢复持续自主执行；长期Goal为同一single checkpoint strict correct
  严格超过`150/400`，同时保持真实视频时序因果性、same-task鲁棒性、breadth与低checkpoint
  漂移。只有实质性阻塞才回报owner，GPU工作仍逐次执行live空闲卡与BCI多卡合同。

## 已完成并负裁决：K4 Phase-Aligned v6（2026-08-07）

- [x] 完成Grounded-Video Expert四点、winner五臂与8-task refs1内部分析：correct=
  `76/88/77/82`、breadth>=5=`3/4/3/3`，winner五臂=`88/87/82/86/86`。视频与时序能material
  改变LoRA/BA/action且expert-local retention约`.5`，但correct无视频margin、task轮换未解；hard
  route/expert isolation正式负裁决，不resume、不warm-start。
- [x] 封存`docs/action_forecast_writer_k4_phase_aligned_v6_design.md`：恢复v6 trainable PI05高层
  semantic/procedure图；K4逐video phase16对齐，Core读取联合无序证据，Procedure逐video causal后
  按phase组合，exact v6 compiler只生成一套LoRA；AS与未来RL共用同一图。
- [x] 原位切换唯一canonical Writer，删除Grounded `fewshot_m2p.py`活动实现，建立fresh config/
  launch/checkpoint family；step0 identity、K4 set permutation、phase causal、source freeze和五个
  gradient owner合同闭合，全仓`190 passed`。
- [x] clean`e1d0b62`在live空闲gpu01六卡、3+3 NUMA、显式`NCCL_P2P_DISABLE=1`完成
  longest105 K4/B20/B2 fresh0→1与same-root exact-resume1→3：0 clip/OOM，peak reserved
  `47,016,050,688` bytes，step3五个owner全可达；profile权重弃用，formal config已seal。
- [x] clean`2356d33`从identity完成formal0→200与50/100/150/200 correct400=
  `88/108/80/99`；union/intersection=`157/36`，single winner=macro100=108，不resume400。
- [x] winner五臂=`108/115/94/101/121`；correct相对wrong gained/lost=`28/14,p=.04356`，
  视频task identity进入closed loop，但order不对齐。LoRA norm/stable-rank/top-energy中位=
  `91.12/1.00021/.99979`，最后50步factor/program retention=`.04634/.04363`；负裁决。

最后更新：2026-08-07 UTC。

本文件只保存尚未完成的长期闭环与当前执行顺序。历史实验过程见
`findings.md`、`progress.md` 和 Git；实时进程见
`docs/active_session_handoff.md`；迁移状态见
`docs/a100_to_bci_migration_handoff.md`。不得从历史 ledger 中恢复已退役 recipe、
runner、split、路径或 GPU 权限。

## 已完成并负裁决：Grounded-Video Semantic-Expert Route（2026-08-07）

- [x] 完成Sparse routefix identity-fresh0→200、四点、winner五臂与production-batch内部分析；
  correct=`74/74/78/75`，winner五臂=`78/85/90/83/92`，correct最低。确认视频trace真实改变
  Reader→program→BA→action，LoRA gain/rank充足；拒绝“视频被忽略”与“继续加参数/步数”。
- [x] 定位最早接口：language-only route让同task五臂固定同两个owners，parameter isolation改善
  expert-local retention却没有让高层视频语义决定credit存储位置。
- [x] 封存
  `docs/action_forecast_writer_grounded_video_expert_route_design.md`：冻结PI05 multimodal task-token
  video innovation形成K4 semantic address，train24-only 8-center route；20-group traces仍是
  dynamic value，八完整experts与single-LoRA不变，AS/RL共用同一图。
- [x] 原位实现grounded address与input-only route generator；clean`563089a`用gpu01六张空闲A40
  完成train24×50 action-hidden videos提取。初始top2的primary/exact/overlap=
  `1.0/.984833/.992417`，但task35 secondary使batch4/singleton仅`23/24` exact；没有放宽门，
  根据完全稳定的primary收敛为top1 one-hot。
- [x] 最终route gate为随机K4稳定率`1.0`、batch4/singleton=`24/24` exact、primary usage=
  `2/6/7/3/1/1/2/2`且8 experts全非空；全程teacher action/state/reward/terminal与validation/test
  video reads均为0。artifact为`configs/pi05_grounded_video_expert_route_v1.json`。
- [x] 建立fresh incompatible architecture/config/checkpoint family并退休language-route executable
  path；grounded route/model/config/checkpoint聚焦`30 passed`、py_compile与real route load通过。
- [x] clean`0be3627`、live比较`gpu01/gpu02`后，以`gpu01:0,1,2|4,5,7`六张空闲A40完成
  longest105、K4/B20/B2 fresh0→1和same-root exact-resume1→3。三步约
  `42.63/41.72/41.20s`，0 clip/OOM/nonfinite，peak reserved`45,237,665,792` bytes；step2起
  16 blocks全可达且train24真实route与authority完全一致。profile权重弃用。
- [x] clean`a758bba`按sealed B20/world6合同从identity自然完成fresh0→200、每25 checkpoint：
  200 finite macros、96,000 queries、19,200 K4 action-hidden video conditions、8个完整checkpoint、
  0 clip/OOM/nonfinite，wall=`8828.911s`，peak allocated/reserved=
  `36,708,964,864/42,727,374,848` bytes，source trainable=0且validation/test action reads=0。
- [x] canonical evaluator去除checkpoint/authority/shard/result的SHA-256/MD5内容校验：launch v2只保留
  显式run UUID reference、path/schema/size/direct paired identity；policy-noise RNG与deterministic job ID
  的既有小型SHA算法保持科学配对不变。聚焦`55 passed`、py_compile、hashless prepare vertical path通过。
- [ ] 严格评50/100/150/200 correct400，single winner再做五臂和内部route/path/gradient分析；评测
  contract必须使用新的hashless v2且保持四点direct paired-control字段逐项相同。
- [ ] 同一single checkpoint strict correct必须`>150/400`且继续提高；未过门时只根据最早失败
  接口继续迭代，不用旧best、checkpoint融合、挑video或延长失败schedule救点。

## 已完成并负裁决：Sparse Semantic-Expert Trace（2026-08-07）

- [x] 完成Evidence-Factorized macro200五臂与8-task refs1内部分析：五臂=
  `84/85/66/83/78`，correct-wrong gained/lost=`36/18,p=.01983`；视频task identity已到closed
  loop，same/order仍无有效margin。
- [x] 闭合`physical/direction/evidence→attention→dual values→Reader→axis→BA→fixed action`；
  两value branch都material，LoRA norm/stable-rank/top-energy=`60.31/1.291/.847`。最早故障定位
  为shared Reader/axis最后50步近1/24 credit cancellation，不再改频谱、scalar、rank或loss。
- [x] 封存
  `docs/action_forecast_writer_sparse_semantic_expert_trace_design.md`：train24-only frozen semantic
  top2 route选择两个完整独立Reader+axis experts，video trace仍是唯一动态value，top2 memory只
  生成一套LoRA；AS/RL共用同一图。
- [x] 生成route authority并审计8 experts的primary/top2 train usage；不得读取held action、
  validation/test input或rollout选择route。
- [x] 原位实现唯一canonical sparse-expert Writer、fresh config/schema/checkpoint/task-gradient
  owner，退休single-expert executable path；完成聚焦合同、全仓回归、compileall与real load。
- [x] 首次profile完成后启动formal，并在macro28由expert-local Gram发现task9 runtime secondary
  owner与route artifact不一致；精确停止本项目进程，否决旧profile与中断formal，不把工程合同
  失败误当科学结果。
- [x] 根修task anchor的co-batch shape依赖：逐exact language独立forward，以singleton anchors
  重生成route，实测最大anchor差`1.49e-8`且co-batch/singleton top2完全一致。
- [x] clean`bbe5cf2`、live比较`gpu01/gpu02`后，以gpu01六张3+3 NUMA空闲A40和新root重做
  longest105、K4/B20/B2、fresh0→1及same-root exact-resume1→3；真实route与authority一致，
  0 clip/OOM，step2起16 blocks全可达，profile重新seal。旧profile/中断formal权重永久弃用。
- [x] clean`3820f27`从functional identity完成fresh0→200与50/100/150/200 correct400：
  `74/74/78/75`、breadth=`6/5/5/5`；single winner macro150=78，不续400。
- [x] 完成winner五臂和production-batch内部分析：五臂=`78/85/90/83/92`，correct最低；
  video path与LoRA leverage成立，但language route无法让视频语义决定owner。Sparse方法正式负裁决，
  不resume、不warm-start、不恢复language-only route。

## 已完成并负裁决：K4 Energy-Preserving Policy-Layer Trace M2P（2026-08-06）

> 本节方法已完成并负裁决；当前活动设计见下一节。

- [x] 完成上一版macro100五臂与8-task内部probe：
  `correct/same/wrong/shuffled/reversed=99/92/57/94/105`，correct相对wrong的paired
  gained/lost=`61/19,p=2.73e-6`，证明video task identity真实进入LoRA与closed loop；
  但shuffled/reversed不低于correct，时序差异未对齐任务程序。
- [x] 定位最早失效接口：原始DCT trace的DC能量占比中位`.95664`，高频8项
  总占比仅`.003592`；旧实现却把每个`group × frequency`独立归一为单位norm，
  将高频相对放大约140倍。这会制造强reversal操纵却不产生有效closed-loop顺序语义。
- [x] 封存
  `docs/action_forecast_writer_energy_preserving_layer_trace_design.md`：保留exact language、K4
  action-hidden videos、20 groups、DCT16、Reader/M2P、rank16与full24 B20，只用每视频
  一个全局scalar匹配旧总trace energy，完整保留group/frequency间原始相对能量和符号。
- [x] 原位替换canonical trace normalization，建立新architecture/config/checkpoint family并严格
  拒载旧checkpoint；完成聚焦CPU合同、全仓回归与real config load。
- [x] clean/push后live比较`gpu01/gpu02`，只用最多6张空闲A40完成fresh0→1与
  exact-resume1→3 profile；权重弃用，config已seal。
- [x] 从functional identity完成formal fresh0→200：200 finite macros、96,000 action
  queries、19,200 K4 action-hidden video conditions、8个every25 checkpoints、0 clip，
  source trainable=0且validation/test action reads=0。wall=`7373.955s`，peak
  allocated/reserved=`18,096,449,024/20,478,689,280` bytes。
- [x] 严格评macro50/100/150/200 paired correct400：correct=`67/83/74/85`、breadth=
  `5/6/7/7`，single winner固定macro200=85；固定panel无mismatch。
- [x] 对macro200完成same-task-other/cross-suite-wrong/shuffled/reversed四个paired
  correct400和内部trace→Reader/M2P→BA→action分析；五臂=`85/85/80/74/87`，视频task
  specificity消失，energy-preserving方法正式负裁决。
- [ ] 按single-checkpoint correct、breadth、video causality与task-gradient证据继续迭代；最低严格
  `>150/400`，达到后仍继续提高。若频谱修复后credit仍接近1/24抵消，才打开
  frozen-semantic routing驱动的condition-specific sparse value experts。

### Energy-Preserving Layer-Trace A40 profile launch合同（2026-08-06）

- implementation/config seal=`22234c4`，已push branch/main；启动时必须clean且新root不存在。
  不传`--initialize-writer-checkpoint`，首段不传`--resume`，禁止加载上一版macro100或任何
  profile/formal Writer。
- 唯一config为
  `configs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_bci_v1.json`；profile root/log/tmux固定为
  `runs/outputs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_profile_r6_b20_22234c4_20260806`、
  `runs/logs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_profile_r6_b20_22234c4_20260806.log`、
  `ember_k4_energy_trace_profile_22234c4`。
- 启动前live比较`gpu01/gpu02`，只用同一node满足3+3 NUMA的6张空闲A40，显式
  `NCCL_P2P_DISABLE=1`。longest105、K4/B20/B2、16-frame chunk先fresh0→1，再从同root
  `step_00000001` exact-resume1→3；profile权重永久弃用。
- `/data1`当前quota为`343,430,876/1,073,741,824 KiB`；参数拓扑与上版相同，三个
  checkpoint、原子临时副本与log预计峰值新增低于4GiB，距离独立配额充足。
- profile只裁决finite/OOM、zero/identity、source freeze、reader/axis可达、多卡和exact-resume；
  不用三步loss做科研结论。通过后回写config封存证据，才允许formal。

profile已按上述合同自然完成：三步loss=`.150377/.152822/.148504`，grad norm=
`.000589/.000636/.000639`，0 clip/OOM/nonfinite；step2起reader/axis update L2均非零，
peak allocated/reserved=`18,113,258,496/20,375,928,832` bytes，累计1,440 queries/288 videos，
source trainable=0，六rank、3+3 NUMA和exact-resume闭合。config seal=`3b7eb4a`，profile权重弃用。

### Energy-Preserving Layer-Trace formal0→200 launch合同（2026-08-06）

- sealed config commit=`3b7eb4a`，已push branch/main；启动前必须clean且`HEAD==origin/main`。
  只从functional identity fresh启动，不传`--resume`或`--initialize-writer-checkpoint`，不加载
  任何profile、macro100或历史Writer。
- formal root/log/tmux固定为
  `runs/outputs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_formal_fresh0_200_r6_3b7eb4a_20260806`、
  `runs/logs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_formal_fresh0_200_r6_3b7eb4a_20260806.log`、
  `ember_k4_energy_trace_formal_3b7eb4a`。scale=`200×24×B20=96,000` action queries、
  `200×24×K4=19,200` action-hidden videos、8个every25 checkpoints。
- profile实测约36.9s/macro，训练主体预计约123分钟；单checkpoint约664MiB，8点、原子
  临时副本、metrics/log预计峰值新增低于8GiB。`/data1` quota为
  `343,430,876/1,073,741,824 KiB`，容量足够。
- world6、logical B20、policy B2、16-frame encoder chunk、full24等权、source freeze、K4和
  `NCCL_P2P_DISABLE=1`不变。启动前live比较双节点，只用同一node满足3+3 NUMA的
  6张空闲A40；任一卡变忙即更换合法组合或延后，不共享、不干扰。
- 自然完成后才并行两波评macro50/100/150/200 strict paired correct400；不用functional
  loss、训练期gradient或中途任务结果挑checkpoint。single winner再做五臂和内部分析。

精确命令：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4,5,7 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py --config configs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_bci_v1.json --mode formal --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_formal_fresh0_200_r6_3b7eb4a_20260806 --stop-after-step 200 --num-workers 0 --log-every 1 --skip-data-sha
```

上述formal已从clean/pushed launch commit`d833961`自然完成，六张A40已释放。四个50步
窗口的full24 gradient retention/cosine/negative-pair中位依次为
`.12497/.07199/.35870`、`.08564/.04393/.42391`、`.08050/.02884/.44022`、
`.05079/.00555/.48007`；频谱修复使前150步coexistence明显优于上一版，但最后50步再次
接近抵消，只作机制证据。

### Energy-Preserving Layer-Trace strict correct400 launch合同（2026-08-06）

- 唯一训练root为
  `runs/outputs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_formal_fresh0_200_r6_3b7eb4a_20260806`；
  固定评`step_00000050/100/150/200`，不替换候选、不加载其他Writer。
- 每点固定validation 8 tasks×50 sealed states、formal、correct video、K4
  without-replacement；source checkpoint/tokenizer/video dataset和state/env/policy RNG panel
  与历史strict400相同。每root 3 GPUs、3 replicas/GPU、3 Writer generators/GPU、generation
  batch4；两点并行一波，跨两点总计6张空闲卡，第二波前重新live检查。
- 输出root固定为
  `runs/outputs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_bci_correct400_noreplacement_seed7_macro{0050,0100,0150,0200}_d833961_20260806`；
  完成合同为每root 400 unique rows、42 shards、9 workers exit0和paired panel mismatch=0。
- 四点只按single-checkpoint correct、breadth、per-task与换手选择winner；winner之后才做
  same-task-other/wrong/shuffled/reversed和内部trace→Reader/M2P→BA→action分析。

命令模板：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --role validation --mode formal --state-count 50 --replicas-per-gpu 3 --writer-generators-per-gpu 3 --writer-generation-batch-size 4 --as-writer-config configs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_bci_v1.json --writer-video-data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --writer-video-condition correct --writer-video-sampling without_replacement --gpu-indices GPUS --as-writer-checkpoint runs/outputs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_formal_fresh0_200_r6_3b7eb4a_20260806/checkpoints/step_STEP --output-dir OUT
```

四点已自然完成：每root 400 rows、42 shards、9 workers exit0。逐task顺序为
Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3：`3/0/0/29/33/0/1/1`、
`1/0/2/38/37/0/2/3`、`4/3/3/32/28/0/2/2`、`6/1/3/37/34/0/3/1`；相邻
gained/lost=`28/12,18/27,28/17`，union/intersection=`122/40`，envelope gap37。
macro200以最高correct和并列最高breadth成为single winner，但85远低于上一版99、v6-fast143
与严格门151；不得续训或用更低loss选点。

### Energy-Preserving macro200五臂/内部分析合同（2026-08-06）

- correct=85已封存，只对同一macro200 checkpoint追加`same_task_other`、
  `cross_suite_wrong`、`shuffled`、`reversed`。四臂保持同一400 state/env/policy RNG panel、
  K4 without-replacement与其各自canonical video mapping/order；不挑video或checkpoint。
- 两臂一波，每root 3 GPUs，总计最多6张live空闲卡；输出root分别为
  `runs/outputs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_bci_{same_task_other,cross_suite_wrong,shuffled,reversed}400_noreplacement_seed7_macro0200_69bb0f0_20260806`。
- 五臂后在winner上复用既有hashless refs1内部probe，比较raw trace、Reader、axis、effective
  BA、fixed action、LoRA谱/能量和task-gradient；若真实幅度保留改善gradient却损害closed-loop，
  判断频谱幅度本身是否包含不可简单衰减的任务线索，再决定下一架构。

上述合同已全部完成。correct相对same/wrong/shuffled/reversed的control gained/correct lost=
`16/16,25/30,12/23,18/16`，exact p=`1,.590,.0895,.864`；没有一个control形成可信
视频因果margin。相对旧逐频率单位化，same/wrong/shuffled/reversed的trace relative-L2中位从
`.995/1.319/1.375/1.414`降到`.135/.310/.251/.335`，Reader从
`.135/.547/.406/.342`降到`.030/.297/.060/.079`，effective BA从
`.167/.715/.450/.452`降到`.049/.478/.092/.117`。Reader effective groups也从约13.97
降到10.63。

新LoRA并未低增益或重新rank collapse：norm中位`58.71`、stable rank`1.410`、top singular
energy`.793`，identity→correct fixed-action差异中位`.581`。因此最早失效明确是raw
amplitude让DC和高能policy groups淹没弱但task-discriminative的direction，而不是Writer容量、
LoRA leverage或先前预注册的shared expert credit。下一方法必须同时保留direction与physical
support，不能在逐token单位化和全局raw amplitude两个破坏性极端之间二选一；暂不打开sparse
  experts。

## 已完成并负裁决：K4 Evidence-Factorized Policy-Layer Trace M2P（2026-08-06）

- [x] 用逐频率单位化与raw-amplitude两个严格反事实定位破坏性二选一：前者保留video
  direction但放大低能order噪声，后者改善早期gradient coexistence却消除video task
  specificity与policy-group diversity。
- [x] 封存`docs/action_forecast_writer_evidence_factorized_trace_design.md`：从同一raw DCT同时
  保留normalized direction与physical coefficient，以group/frequency energy share和K4
  leave-one-out direction consensus只作key evidence，再用shared-attention dual vector values
  和bias-free fusion进入原axis M2P。
- [x] 原位实现新唯一architecture/config/checkpoint family，退休Energy-Preserving活动path；
  normalized direction、raw physical value和3维evidence按同一K4 token顺序进入single-attention
  dual-value Reader。Writer参数=`60,926,976`；BCI assets下全仓`192 passed`、compileall、real
  config load、fresh family和diff check闭合，formal仍blocked。
- [x] clean/push后live比较双节点，在`gpu01:0,1,2|4,5,7`六张空闲A40完成longest105 B20
  fresh0→1、exact-resume1→3 profile，权重弃用。三步loss=`.150377/.152820/.148508`，
  0 clip/OOM/nonfinite，step2起evidence key、双value、vector fusion、Reader与axis均finite可达；
  peak allocated/reserved=`18,218,217,984/20,470,300,672` bytes。
- [x] 从identity formal0→200：200 finite macros、96,000 queries、19,200 K4 videos、
  8 checkpoints、0 clip，GPU自然释放。
- [x] strict correct400固定评50/100/150/200：`74/59/65/84`，breadth=`6/6/5/5`；
  macro200固定single winner但未过严格门。
- [ ] 对macro200完成五臂和全部内部分析；按最早失败接口继续迭代，single checkpoint严格
  `>150`且继续提高。

### Evidence-Factorized Trace formal0→200 launch合同（2026-08-06）

- sealed config commit=`692ab5e`；启动前必须clean、已push且`HEAD==origin/main`。只从functional
  identity fresh启动，不传`--resume`或`--initialize-writer-checkpoint`，不得加载任何profile或
  历史Writer。
- formal root/log/tmux固定为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_formal_fresh0_200_r6_692ab5e_20260806`、
  `runs/logs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_formal_fresh0_200_r6_692ab5e_20260806.log`、
  `ember_k4_evidence_trace_formal_692ab5e`。scale=`200×24×B20=96,000` action queries、
  `200×24×K4=19,200` action-hidden videos和8个every25 checkpoints。
- profile实测39.1--41.4s/macro，主体预计约135分钟；峰值reserved20.47GB。新Writer的8个
  checkpoint、原子临时副本、metrics/log预计峰值新增低于9GiB；`/data1` live quota为
  `359,273,924/1,073,741,824 KiB`，容量充足。
- world6、logical B20、policy B2、16-frame encoder chunk、full24等权、source freeze、K4、
  3+3 NUMA和显式`NCCL_P2P_DISABLE=1`不变。启动前重新live比较双节点；只用同node满足3+3
  NUMA的6张空闲卡，任一卡变忙就更换合法组合或延后，不共享、不干扰。
- 自然完成后固定评50/100/150/200 strict paired correct400；不用functional loss、训练期
  gradient或中途task结果挑checkpoint。四点完成后按single winner做五臂与全部内部分析。

精确命令：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4,5,7 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py --config configs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_bci_v1.json --mode formal --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_formal_fresh0_200_r6_692ab5e_20260806 --stop-after-step 200 --num-workers 0 --log-every 1 --skip-data-sha
```

上述formal已从clean/pushed launch commit`7e3559f`自然完成。wall=`7272.774s`，peak
allocated/reserved=`18,203,289,600/20,304,625,664` bytes；source trainable=0，validation/test
action reads=0。四个50步窗口的full24 gradient retention/cosine/negative-pair中位依次为
`.10601/.06078/.36957`、`.08578/.05152/.38949`、`.06065/.02493/.44746`、
`.05227/.00727/.47645`；不得用该机制曲线或functional loss挑checkpoint。

### Evidence-Factorized Trace四点strict correct400 launch合同（2026-08-06）

- 唯一训练root固定为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_formal_fresh0_200_r6_692ab5e_20260806`；
  只评`step_00000050/100/150/200`，不得替换checkpoint或加载其他Writer。
- 每点为validation 8 tasks×50 sealed states、formal、correct K4 action-hidden videos、
  without-replacement；state/env/policy RNG、K4 set、source checkpoint、tokenizer、video dataset
  与历史strict400 panel不变。每root 3 GPUs、3 replicas/GPU、3 Writer generators/GPU、generation
  batch4；两点并行一波，总计最多6张live空闲卡，第二波前重新检查双节点。
- 四个root固定为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_bci_correct400_noreplacement_seed7_macro{0050,0100,0150,0200}_c23195d_20260806`；
  log同名位于`runs/logs/`，tmux为`ember_k4_evidence_eval{0050,0100,0150,0200}_c23195d`。
  每root完成门为400 unique rows、42 shards、9 workers exit0和paired panel mismatch=0。
- 四点完成后只按single-checkpoint correct、breadth、per-task与换手选winner；winner之后才追加
  same-task-other/wrong/shuffled/reversed与内部direction/physical/evidence→Reader→axis→BA/action分析。

命令模板：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --role validation --mode formal --state-count 50 --replicas-per-gpu 3 --writer-generators-per-gpu 3 --writer-generation-batch-size 4 --as-writer-config configs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_bci_v1.json --writer-video-data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --writer-video-condition correct --writer-video-sampling without_replacement --gpu-indices GPUS --as-writer-checkpoint runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_formal_fresh0_200_r6_692ab5e_20260806/checkpoints/step_STEP --output-dir OUT
```

四点已自然完成：每root 400 rows、42 shards、9 workers exit0。按Long-1/2、Goal-3/6、
Object-1/3、Spatial-1/3顺序，逐task为`5/0/1/39/26/0/2/1`、
`3/1/0/27/22/3/0/3`、`6/1/0/32/25/0/0/1`、`9/2/0/36/34/0/0/3`；相邻
gained/lost=`19/34,23/17,29/10`，union/intersection=`122/32`。K4 set、state、env seed、
teacher order和policy-noise common prefix跨四点均0 mismatch。macro200以最高correct固定为
single winner=84，但低于raw-only85、unit-only99、K4 invariant108与v6-fast143；不续训练。

### Evidence-Factorized Trace macro200五臂与内部分析合同（2026-08-06）

- correct=84沿用既有macro200 root，不重跑。其余只改变`--writer-video-condition`为
  `same_task_other`、`cross_suite_wrong`、`shuffled`、`reversed`；保持同一400 state/env/policy
  RNG panel、K4 without-replacement、source与macro200 checkpoint。
- 两臂一波，每root 3 GPUs、3 replicas/GPU、3 generators/GPU、generation batch4；每波启动前
  live比较`gpu01/gpu02`，最多使用6张空闲卡。四root固定为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_bci_{same_task_other,cross_suite_wrong,shuffled,reversed}400_noreplacement_seed7_macro0200_b1d3156_20260806`。
- 五臂后只对同一macro200做8-task refs1内部probe：必须分别量化raw physical/direction/evidence、
  direction/physical value read、attention/effective groups、fusion/axis、effective BA、fixed action、
  LoRA谱/能量及identity/leave-one-out/alternate-set；训练期gradient直接用sealed metrics。

## 已完成K4 Policy-Layer Trace M2P（2026-08-06）

- [x] 按K4内部证据与SHINE/Doc-to-LoRA的结构原则封存
  `docs/action_forecast_writer_k4_layer_trace_m2p_design.md`。保持四条action-hidden视频联合
  生成一套LoRA，改为读取冻结PI05 action expert的20组all-layer video innovation，以
  layer×parameter-slot双轴M2P直接生成完整public LoRA；不走language-only value旁路，
  不在未解决policy拓扑对齐前直接复制多expert。
- [x] clean`a2c6d94`原位替换旧final-layer随机128维descriptor、32×256 invariant program和608-token
  通用decoder；同步唯一architecture/config/checkpoint/task-gradient/test owner，删除活动
  executable旧K4 path并拒载旧family。全仓BCI assets下`190 passed`、compileall与diff check
  通过；20×64 trace、K4 permutation、zero-video identity、68-slot slicing、step1→step2
  梯度可达和完整参数ownership闭合。
- [x] 完成聚焦CPU合同与A40 longest105 K4/B20/B2 fresh0→1、exact-resume1→3 profile；只在
  live最多6张空闲卡、3+3 NUMA、显式`NCCL_P2P_DISABLE=1`下运行，profile权重永久弃用。
  - [x] 首个`89f5384` diagnostic因一次跑0→3不满足resume程序，并暴露axis FFN pre-LN把
    极小bootstrap memory放大到O(1)：loss`.150→58.93→96.82`且三步clip；root禁止resume。
  - [x] clean`ed4f46e`移除FFN value-path normalization，保留route只进Q/K，新增zero邻域
    2×幅度合同；全仓`191 passed`。下一次另起fresh root执行正式profile程序。
  - [x] clean`44e248b`在`gpu01:0,1,2|4,5,7`严格分段通过：loss稳定约`.148--.153`、
    grad norm约`.001`、0 clip/OOM/nonfinite，step2起两个block均可达；三步约34.6--34.7秒，
    peak reserved20.38GB，六rank exact-resume闭合，profile权重弃用。
- [x] 从functional identity fresh0→200，strict correct400固定评50/100/150/200；按
  layer trace、reader、axis M2P、BA/action leverage、task-gradient coexistence与闭环换手的
  最早失效接口分析并继续迭代，single checkpoint必须严格`>150`且继续尽可能提高。
  - [x] fresh formal0→200自然完成：200 finite macros、96,000 action queries、19,200
    action-hidden videos、8个every25 checkpoints、0 clip/OOM/nonfinite，source trainable=0且
    validation/test action reads=0。wall=`7350.114s`，peak reserved=`20,478,689,280` bytes。
  - [x] 用同一paired K4/state/RNG panel完成macro50/100/150/200 strict correct400：
    correct=`69/99/88/94`、breadth=`5/6/6/6`，single winner为macro100。四点完成前没有按
    functional loss或内部梯度选择checkpoint。
  - [x] 对macro100完成same-task-other/wrong/shuffled/reversed四个paired full400，并完成
    layer trace→reader→axis M2P→BA→fixed-action、LoRA谱/能量与task-gradient内部分析；只按
    最早失败接口决定下一架构。

### K4 Policy-Layer Trace四点strict correct400 launch合同（2026-08-06）

- formal training seal=`535123a`且已push branch/main；四个checkpoint必须来自同一fresh root
  `runs/outputs/pi05_as_writer_k4_layer_trace_m2p_formal_fresh0_200_r6_d3f568d_20260806`。
  role=`validation`、state-count50、formal、correct、without-replacement、K4 set/state/env/policy
  RNG完全配对；不能改变video、挑状态或复用旧K4 LoRA cache。
- 两波各并行两个独立root，每root使用3张物理GPU、3 replicas/GPU、3 Writer generators/GPU、
  generation batch4。第一波macro50使用`gpu01:0,1,2`、macro100使用`gpu01:4,5,7`；自然结束
  并确认释放后，第二波同样拓扑运行macro150/200。跨节点合计始终最多6张，启动前仍live
  比较双节点；任一卡变忙就更换为同node三张空闲卡或延后，不共享、不干扰。
- 四root依次为
  `runs/outputs/pi05_as_writer_k4_layer_trace_m2p_bci_correct400_noreplacement_seed7_macro{0050,0100,0150,0200}_535123a_20260806`；
  log同名位于`runs/logs/`，tmux同step命名为`ember_k4_trace_eval{0050,0100,0150,0200}_535123a`。
  既有K4 correct400每root约1.07GB，四点加原子临时cache预计峰值新增低于6GB，当前quota余量充分。
- 每root必须封存400 rows、42 shards、9 workers exit0、400个unique K4 sets及完整results/
  completion；聚合报告correct、breadth、per-task、gained/lost、union/intersection。任何launcher
  故障只在同root按canonical resume恢复，不能用另一checkpoint或减少panel冒充。

每点命令模板（替换`STEP`、`GPUS`与`OUT`）：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --role validation --mode formal --state-count 50 --replicas-per-gpu 3 --writer-generators-per-gpu 3 --writer-generation-batch-size 4 --as-writer-config configs/pi05_as_writer_k4_layer_trace_m2p_bci_v1.json --writer-video-data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --writer-video-condition correct --writer-video-sampling without_replacement --gpu-indices GPUS --as-writer-checkpoint runs/outputs/pi05_as_writer_k4_layer_trace_m2p_formal_fresh0_200_r6_d3f568d_20260806/checkpoints/step_STEP --output-dir OUT
```

四点已自然完成并满足上述完整性门。逐task顺序为Long-1/2、Goal-3/6、Object-1/3、
Spatial-1/3：macro50=`4/0/0/33/29/2/0/1`、macro100=`5/3/0/34/41/12/0/4`、
macro150=`12/1/0/34/26/13/0/2`、macro200=`15/1/0/34/27/11/0/6`。相邻
gained/lost=`42/12,28/39,28/22`，四点union/intersection=`145/37`；跨checkpoint的
K4 set、state、env seed与policy-noise common prefix为0 mismatch。layer alignment只把旧K4
macro100的94提高到99，之后下降并换手；不续同一schedule或挑其他checkpoint。

### K4 Policy-Layer Trace macro100五臂与内部分析合同（2026-08-06）

- winner固定为同一fresh root的`step_00000100`；correct arm沿用已封存的99/400 root，不重跑。
  其余四臂只改变`--writer-video-condition`，保持validation、state-count50、formal、
  without-replacement、K4 schedule、state/env/policy RNG、3 replicas/GPU、3 generators/GPU与
  generation batch4不变。
- 两波各并行两个独立root、每root 3张空闲A40：第一波`same_task_other`与
  `cross_suite_wrong`，第二波`shuffled`与`reversed`。每波启动前live比较`gpu01/gpu02`，总计
  不超过6张；每root仍必须400 rows、42 shards、9 workers exit0。
- 四root固定为
  `runs/outputs/pi05_as_writer_k4_layer_trace_m2p_bci_{same_task_other,cross_suite_wrong,shuffled,reversed}400_noreplacement_seed7_macro0100_535123a_20260806`；log同名位于
  `runs/logs/`。不复用correct cache，因为不同condition的实际Writer输入必须独立完整forward。
- 行为五臂完成后，8 validation tasks各取一个paired K4 condition做hashless内部probe：至少覆盖
  correct/alternate-set/wrong/shuffle/reverse/leave-one-out/zero、raw pre-normalization DCT
  layer×frequency能量、reader/axis memory、effective BA谱与固定action。训练期task-gradient
  直接读取既有200步sealed metrics，不重算functional panel。

### K4 Policy-Layer Trace M2P fresh formal0→200 launch合同（2026-08-06）

- implementation/config/profile seal=`d3f568d`，已push branch/main；正式启动前必须保持clean且
  `HEAD==origin/main`。只从functional identity fresh启动，不传`--resume`或
  `--initialize-writer-checkpoint`，不加载两个profile root、旧K4或任何历史Writer权重。
- fresh output/log/tmux固定为
  `runs/outputs/pi05_as_writer_k4_layer_trace_m2p_formal_fresh0_200_r6_d3f568d_20260806`、
  `runs/logs/pi05_as_writer_k4_layer_trace_m2p_formal_fresh0_200_r6_d3f568d_20260806.log`、
  `ember_k4_trace_formal_d3f568d`。scale=`200×24×B20=96,000` action queries、
  `200×24×K4=19,200` action-hidden videos、8个every25 checkpoints；sealed profile约
  `34.6s/macro`，预计训练主体约115分钟。
- source checkpoint、tokenizer、train24 dataset、normalization与LIBERO assets沿用sealed
  canonical路径；world6、logical B20、policy B2、16-frame encoder chunk、full24等权和
  `NCCL_P2P_DISABLE=1`不变。upstream为generic frozen source step1000，不使用held action、
  reward或历史Writer初始化。
- profile单checkpoint约686MiB；8点checkpoint、原子临时副本、metrics/log预计峰值新增低于
  8GiB。2026-08-06同日live quota观测约`316,176,688/1,073,741,824 KiB`，本次重查
  `xfs_quota`被权限拒绝；共享`/data1`仍有充足容量。该已知权限限制不触发重复扫描或hash。
- live GPU选择仍须在启动前即时比较`gpu01/gpu02`；优先使用与profile一致且满足3+3 NUMA的
  `gpu01:0,1,2|4,5,7`，仅当六卡均空闲才启动。任一卡出现他人进程、显存或利用率占用即
  延后或改用同一node满足3+3的六张空闲卡，跨节点合计始终不超过6张。
- 训练结束后固定评macro50/100/150/200 strict paired correct400，与旧K4
  `70/94/99/108`、v6-fast`143`和严格门`>150`比较；functional loss或内部几何不用于选点。
  single winner再做五臂、另K4 set/leave-one-out及layer trace→reader→M2P→BA→action和
  task-gradient coexistence分析；失败只按最早失效接口重构。
- 上述fresh formal已由launch commit`1b868ed`自然完成。训练期full24 gradient
  retention/cosine/negative-pair按50步段中位为
  `.07229/.03067/.4112 → .06074/.01741/.4094 → .05466/.01100/.4457 →
  .04573/.00400/.4746`；层对齐带来的早期coexistence优势在晚期明显衰减，行为四点仍是
  必须完成的裁决，不能仅凭该内部量提前判负。

精确命令：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4,5,7 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py --config configs/pi05_as_writer_k4_layer_trace_m2p_bci_v1.json --mode formal --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_k4_layer_trace_m2p_formal_fresh0_200_r6_d3f568d_20260806 --stop-after-step 200 --num-workers 0 --log-every 1 --skip-data-sha
```

## 已完成的Few-Shot Invariant-Program M2P（2026-08-06）

- [x] owner明确指出EMBER不能忽略视频，并允许在根因需要时从one-shot切换few-shot。完成
  `docs/action_forecast_writer_fewshot_invariant_m2p_design.md`：exact language + K4条
  action-hidden same-task videos联合形成一个video-value-only invariant program，再由
  608个target/layer/rank tokens的policy-wide M2P生成一套完整rank16 LoRA。
- [x] 原位替换canonical Writer、full24 AS、checkpoint、K4 schedule、live/cached evaluation
  与paired evidence；删除已退役Condition-Kernel及其method-specific validation/analysis
  executable paths。新架构禁止language-only value bypass、逐视频LoRA平均/挑选、历史
  Writer warm-start和held validation functional-loss选点。
- [x] CPU合同完成：step0与zero-video identity、K4 set permutation equality、38 targets×16
  lanes、target-owned A/B、完整四block梯度ownership、B20 action/video episode排斥、实际
  world-size full24、hashless checkpoint/evaluation和旧family拒载。项目全仓`188 passed`，
  compileall与diff check通过。
- [x] clean commit/push后live比较`gpu01/gpu02`，只在满足3+3 NUMA的最多6张空闲A40上做
  longest105、logical B20/B2、16-frame chunk的fresh0→1和同root exact-resume1→3 profile；
  显式`NCCL_P2P_DISABLE=1`，profile权重永久弃用。sealed root为
  `runs/outputs/pi05_as_writer_k4_invariant_m2p_profile_r6_b20_8807ae0_20260806`：正式200-step
  scheduler clock下三步`34.055/33.955/33.831s`，peak reserved`19,690,160,128` bytes，
  0 OOM/clip/nonfinite，step2起四block全部可达，fresh0→1→exact-resume1→3闭合。
- [x] profile通过后从functional identity新root正式训练0→200、每25保存；固定50/100/150/200
  strict correct400，并对single winner完成correct/same/wrong/shuffled/reversed、另K4 set、
  leave-one-video-out、LoRA谱/能量、Program→BA→fixed-action和task漂移分析。
  - [x] fresh formal0→200自然完成：200 finite macros、96,000 queries、19,200 videos、8个
    checkpoints、0 clip/OOM/nonfinite、0 validation/test action reads。
  - [x] macro50/100 strict correct400完成：`70/94`、breadth=`6/6`；K4 sets/state/RNG严格配对。
  - [x] macro150/200完成；完整曲线=`70/94/99/108`、breadth=`6/6/6/7`，macro200为single
    winner但未过门。refs1内部分析确认K4共同program、高增益LoRA与order→action路径成立；
    最后50步四个Writer block的task-gradient均接近1/24正交抵消。
- [x] 按最早接口裁决K4：descriptor/invariant/M2P/policy leverage均通过，剩余失败定位为
  condition-specific credit在共享Writer参数中的coexistence；禁止续同一schedule、warm-start、
  loss挑点或退回one-shot/video忽略。
- [x] 下一轮保留K4 video-owned single-LoRA合同，并先修复frozen policy layer与public LoRA
  topology未对齐的更早接口；只有layer-aligned结果仍显示分组task-gradient抵消时，才依据
  证据打开稀疏共享或experts。

### K4 M2P fresh formal0→200 launch合同（2026-08-06）

- implementation/config/profile seal=`dd3b854`且已push branch/main；launch-record后续只改文档，
  真实Git commit由run contract记录。启动必须clean且`HEAD==origin/main`，不传`--resume`或
  `--initialize-writer-checkpoint`，不加载任何profile、Condition-Kernel或历史Writer权重。
- fresh output/log/tmux为
  `runs/outputs/pi05_as_writer_k4_invariant_m2p_formal_fresh0_200_r6_dd3b854_20260806`、
  `runs/logs/pi05_as_writer_k4_invariant_m2p_formal_fresh0_200_r6_dd3b854_20260806.log`、
  `ember_k4_m2p_formal_dd3b854`。scale=`200×24×B20=96,000` action queries、
  `200×24×K4=19,200` teacher videos、8个checkpoint；profile约34秒/macro，预计主体约114分钟。
- live双节点检查后`gpu01`八卡全空，`gpu02:5/6`为他人进程；选择
  `gpu01:0,1,2|4,5,7`六卡、3+3 NUMA、single-node DDP，显式`NCCL_P2P_DISABLE=1`。最终启动
  前仍即时复查；任一卡变忙则不共享、不干扰，改选同node仍满足3+3的空闲卡或延后。
- `/data1` live quota=`316,176,688/1,073,741,824 KiB`；Writer+trainer单checkpoint约
  `377.6MiB`，8点加原子临时副本、metrics/log预计峰值新增低于4GiB，余量充分。source、
  tokenizer、train24 data与LIBERO assets沿用sealed路径并由real load验证，不做内容hash。
- 只在25倍数保存，50/100/150/200完成后才启动strict correct400。训练loss、梯度几何或
  中途单task reward不选择checkpoint；任何异常只从同一完整checkpoint exact-resume，不能
  从profile或历史best替代。

精确命令：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4,5,7 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py --config configs/pi05_as_writer_k4_invariant_m2p_bci_v1.json --mode formal --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_k4_invariant_m2p_formal_fresh0_200_r6_dd3b854_20260806 --stop-after-step 200 --num-workers 0 --log-every 1 --skip-data-sha
```

## 已完成并负裁决：Factorized Condition-Kernel Program Memory（2026-08-05）

- [x] 完成Program-Credit train24内部分析与400-held LoRA复核，正式定位最早失效接口：
  exact task cotangent近正交而共享Writer参数更新后的condition program delta高度同向，
  same-task更新又由task mean主导；不是functional surrogate、semantic tie-break、LoRA rank或
  decoder完全无响应。
- [x] 封存
  `docs/action_forecast_writer_factorized_condition_kernel_memory_design.md`：固定foundation
  task×video RFF address、完整P1024 Program Value Memory、24×24 regularized kernel
  correction、fresh FactorHead bootstrap0→50、memory-only AS50→200与同一memory direct reward。
  方法从generic source全新训练，不加载AS125/v6-fast/Policy-Lane或任何历史Writer。
- [x] 删除已满足retirement trigger的Program-Credit method-specific analysis runtime；只有确有
  当前第二用途的pure gauge-invariant metrics才迁入既有owner。
- [x] 原位实现唯一condition-kernel Writer、custom memory update、fresh config/checkpoint
  family与AS/RL阶段freeze；不保留v6并行model/loader。
- [x] 完成action-hidden train24×50 descriptor/Gram audit：seal text/video bandwidth、fixed
  seed、Phi/K spectrum、same-task video与reversed/shuffled距离；validation 8 tasks只apply，
  不参与拟合或调参。
- [x] 完成identity、fixed address、kernel predicted/observed equality、full24 multi-rank、
  macro50 freeze和exact-resume的聚焦CPU合同与architecture gate。
- [x] live比较gpu01/gpu02后，在最多6张空闲A40上完成longest105、B20/B2、fresh0→1→
  exact-resume1→3 profile；profile权重永久弃用。
- [x] 从functional identity完成正式AS0→200与固定50/100/150/200 strict correct400；曲线
  为`46/46/45/49`、breadth始终`3`，macro200未过`correct≥120且breadth≥6`门。
- [x] 按预注册合同禁止进入direct reward cycle1；当前实验只完成全部内部分析后负裁决，
  不用RL掩盖AS substrate失败。长期single-checkpoint严格`>150`且尽可能更高不变。

### Condition-Kernel fresh AS0→200正式launch合同（2026-08-05）

- code/config seal=`4038960`，已push branch/main；全仓带BCI LIBERO assets为`198 passed`，
  compileall与diff check通过。profile root和任何历史Writer checkpoint均不得用于初始化。
- live比较两节点后选择全空闲`gpu01:0,1,2|4,5,6`，六卡14--90MiB、0% util且无compute
  process，保持profile的3+3 NUMA；`gpu02:5/6`有他人进程，不使用。`/data1` quota为
  `310,538,532/1,073,741,824 KiB`，formal预计新增小于2GiB。
- fresh root=
  `runs/outputs/pi05_as_writer_condition_kernel_memory_formal_fresh0_200_r6_4038960_20260805`，
  log同名位于`runs/logs/`，tmux=`ember_ck_formal_r6_4038960`。world6、logical B20、policy
  B2、full24、显式`NCCL_P2P_DISABLE=1`；checkpoint只在50/100/150/200完整边界保存。
- exact command为
  `PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py --config configs/pi05_as_writer_condition_kernel_memory_bci_v1.json --mode formal --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_condition_kernel_memory_formal_fresh0_200_r6_4038960_20260805 --stop-after-step 200 --num-workers 0 --log-every 1 --skip-data-sha`。

### Condition-Kernel四点rollout与内部分析合同（2026-08-06）

- formal自然完成200 macros、96,000 queries、4,800 videos、四个checkpoint与200 metrics；
  wall=`3951.928s`、峰值reserved=`19,344,130,048` bytes、0 validation/test action reads。
- 四点strict correct400 roots统一为
  `runs/outputs/pi05_as_writer_condition_kernel_memory_bci_correct400_noreplacement_seed7_macro{0050,0100,0150,0200}_4b04c90_20260806`；
  每点400 rows、42 shards、9 workers、400个无放回video、一次启动且全worker exit0。逐task
  结果只集中在Goal-6、Object-1和Long-1，四点breadth均为3，reward gate已失败。
- internal analysis owner由clean/pushed`2972f8f`补充same-task五video相邻checkpoint的Program
  与exact effective-BA update task-mean能量占比；全仓`200 passed`、compile与diff check通过。
  live比较后`gpu01`八卡全空，`gpu02:5/6`有他人进程，选择`gpu01:0,1,2,3,4,5`六卡；
  output/log/tmux为
  `runs/outputs/pi05_as_writer_condition_kernel_memory_internal_all4_r6_2972f8f_20260806`、
  同名`runs/logs/*.log`与`ember_ck_internal_r6_2972f8f`。分析覆盖24 train tasks×4 checkpoints、
  demos0--4、reversed/shuffled及8-task fixed-action panel，target/validation/test action reads均为0。
- [x] 内部分析完整结束：96/96 rows、6/6 payload、wall=`273.968s`、peak reserved=
  `19,277,021,184` bytes。same-task feature/Program/BA差异约`.786/.783/.767`且order反事实
  BA约`1.36--1.39`，但LoRA norm仅`.176→.178`、fixed-action效应仅`.19--.24%`。explicit
  kernel修复credit混合，却被macro50冻结的low-gain fresh decoder锁在接近identity的policy
  tangent；完整负裁决见design第11节与internal `experiment_analysis.json`。
- [x] 全部GPU自然释放。当时的讨论暂停已由owner在2026-08-06解除；Condition-Kernel保持
  负裁决且禁止reward/resume，当前执行顺序只取上方K4方法。长期single-checkpoint
  `>150/400`仍未完成，不是Goal完成。

## 已完成Antithetic Program-Credit推进（2026-08-05）

- [x] 完成Policy-Lane fresh0→200、四点strict correct400与全部内部分析；虽然LoRA stable
  rank、lane参与和跨层专门化真实改善，correct仍为`70/63/37/61`且same-task video
  hidden/BA能量仅约`.05%/.02%`，正式否定继续加容量/几何为主路线。
- [x] 封存`docs/action_forecast_writer_antithetic_program_credit_design.md`：恢复v6的
  `320×256` compiler program作为高层动作；K4做两组同随机性antithetic扰动；binary-first
  pair差直接反传program，不再使用CFM action surrogate。cold start固定为fresh AS125，
  冻结semantic encoder、完整FactorHeads、source policy和normalization。
- [x] 原位恢复唯一v6 Writer和显式`encode_program/decode_program`接口，删除Policy-Lane
  executable family；原位重写`src/ember/rl_writer/`的schema、loop、checkpoint与evaluation
  adapter，不保留旧CFM/Tangent并行路径。
- [x] 完成forward等价、antithetic seed/配对、binary-first credit、freeze/gradient ownership、
  checkpoint/resume与实际world-size的聚焦合同；运行architecture gate并保持单一owner。
- [x] 首次原六卡profile在68/96 rollout、0 update处fail-fast并定位真实CRN根因：LIBERO同一
  hard-reset env的placement history不被重复seed清空。实现每task两条lockstep persistent
  lanes，plus/minus固定分lane；实机连续三次验证XML/state/双相机逐字节一致，v2 ledger绑定
  environment lane，且不调用`set_init_state`。失败root仅作诊断并永久禁止resume。
- [x] live比较`gpu01/gpu02`后完成AS125-fresh独立六卡one-cycle A40 profile及fresh0→1→
  resume1→2；K4/24 tasks、pair randomness、finite cotangent、四个block可达、0 frozen grad、
  NCCL ready和原子checkpoint全部通过后才seal formal。
- [x] formal从AS125阶段边界fresh0→1并完成与AS125严格配对correct400；cycle1=`106/400`、
  breadth5，相对AS125净`+9`、breadth不降且三suite改善，但未达到预注册净`+10`门，故禁止
  resume cycle2/4/8，先完成内部机制分析再设计下一fresh方法。

### 双lane v2 A40重放合同（2026-08-05）

- clean implementation commit=`5ad9db5`且已push branch/main；全仓`221 passed`、compileall和
  diff check通过。失败的v1 root禁止resume，新root为
  `runs/outputs/pi05_antithetic_program_credit_profile_as125_r6_crn_v2_20260805`，log同名位于
  `runs/logs/`，tmux=`ember_program_credit_profile_crn_v2`。
- live比较后选`gpu01:0,1,2|4,5,6`，六卡均无compute process、18--93MiB、0% util，保持
  `3+3 NUMA`；`gpu01:7`及`gpu02:0,5,6`已有他人进程，不使用。沿用本阶段已核验的/data1
  独立quota（约294GiB used/1TiB），新增仅ledger与Writer checkpoint，预计远小于1GiB。
- fresh段只运行profile cycle0→1：world6、24 tasks×K4、每task两条lockstep env lanes、一次
  full24 update，显式`NCCL_P2P_DISABLE=1`。必须96/96 rollout、24 credit、pair初态/noise一致、
  finite cotangent/梯度、0 frozen grad/OOM/watchdog并原子保存cycle1；通过后同root仅以
  cycle1 checkpoint exact-resume到cycle2，任何合同变化都改用新root。
- 上述两段均已完成：每轮96/96 rollout、24/24 credits、48/48 CRN pairs、54 successes，
  四上游block非零且冻结梯度0；wall=`431.709/431.367s`、peak reserved=
  `19,308,478,464/19,331,547,136` bytes，cycle1/2 checkpoint各9文件，0错误。formal已seal，
  profile checkpoint永不warm-start。

### Antithetic Program-Credit formal cycle1合同（2026-08-05）

- seal commit=`219ab4e`并已push branch/main；fresh root=
  `runs/outputs/pi05_antithetic_program_credit_formal_as125_r6_crn_v2_219ab4e_20260805`，log同名
  位于`runs/logs/`，tmux=`ember_program_credit_formal_crn_v2`。唯一cold start仍是sealed
  fresh-AS125 checkpoint，禁止载入任一profile/reward/历史best权重。
- live比较两节点后原选`gpu01:0,1,2|4,5,6`；启动前即时复查发现物理GPU6被他人新进程
  占用约9.1GiB/100% util，因此未启动且不触碰该进程，机械改为仍保持已验证`3+3 NUMA`的
  空闲`gpu01:0,1,2|4,5,7`。`gpu02:0,5,6`也有他人进程，不使用。沿用本阶段已核验的/data1
  quota，formal cycle1新增远小于1GiB。
- 只运行formal cycle0→1：24 train tasks×K4、48 lockstep CRN pairs、一次full24 equal-task
  direct-program update，world6且显式`NCCL_P2P_DISABLE=1`。完成后先封存并释放GPU，再用既有
  strict panel评AS125与cycle1；仅当cycle1严格`>150`，或相对AS125 net+10、breadth不降且
  至少两suite改善时，才exact-resume cycle2。
- formal cycle0→1已完成：96/96 rollouts、24/24 credits、48/48 valid CRN pairs、54
  successes、6 binary-discordant pairs，一次finite update；四上游block非零、冻结梯度0，
  wall=`418.692s`、peak reserved=`19,308,478,464` bytes，完整cycle1 checkpoint、0错误。

### Program-Credit cycle1 strict correct400合同（2026-08-05）

- AS125 baseline直接复用已经封存且同一state/video/env/policy-noise panel的
  `runs/outputs/pi05_as_writer_v6_coldstart_as125_bci_correct400_noreplacement_seed7_df413de_20260805`
  （400 rows、correct97）；不重复消耗GPU。cycle1新root为
  `runs/outputs/pi05_antithetic_program_credit_cycle001_bci_correct400_noreplacement_seed7_219ab4e_20260805`，
  log同名位于`runs/logs/`，tmux=`ember_program_credit_cycle1_correct400`。
- live选最多6张空闲卡并由evaluation launcher再次检查；使用validation 8 tasks×50 sealed
  init states、correct same-task video、without-replacement seed7、3 replicas/GPU、完整400 rows。
  输出预计约1.04GiB，适用/data1 quota已在本阶段核验。结束后按task/state/video/env seed和
  policy-noise prefix与AS125严格配对，报告correct、breadth、suite/task、gained/lost和继续门。
- strict panel已完整结束：400 rows、8 tasks×50、18/18 worker exit0、0 retry/error、每task
  50 unique无放回videos。cycle1 correct/breadth=`106/5`，相对AS125=`97/5`的
  gained/lost/retained/both-fail=`18/9/88/285`，union/intersection=`115/88`，paired
  p=`.12208`。per-task AS125→cycle1为Long`10/0→11/0`、Goal`0/43→0/44`、Object
  `24/19→27/23`、Spatial`1/0→0/1`；三suite净增但Spatial内部换手，净增只为9，续训门失败。

### Program-Credit cycle1内部分析合同（2026-08-05）

- 唯一只读analysis owner按authority/runtime/pure metrics拆为三个模块；复用canonical Writer、
  source policy和既有gauge-invariant指标，不增加训练/模型路径。architecture guard无hard或
  parallel family；review仅为一次性正式analysis体量，删除触发是artifact与下一design封存。
- 分析固定比较sealed AS125与formal cycle1，train24每task读取demo0--4及cross-suite wrong、
  reversed、shuffled；8-task panel做相同fixed-action probe。另从24份credit ledger重建exact
  `320×256` cotangent，报告task-pair cosine、负pair、full24 retention和binary/semantic能量。
- source/target action reads为0，validation/test reads为0；532个semantic encoder、FactorHeads
  与template tensors必须逐元素不变。聚焦`12 passed`，带BCI assets的全仓`223 passed`、
  py_compile/diff check通过。正式launch仍须clean pushed代码、live双节点GPU检查和空root。
- clean implementation/authority commit=`129cab6`并已push branch/main。正式root/log/tmux固定为
  `runs/outputs/pi05_antithetic_program_credit_internal_as125_cycle1_r6_129cab6_20260805`、
  同名`runs/logs/*.log`和`ember_program_credit_internal_r6_129cab6`；只用`gpu02:0,1,2,3,4,7`
  六张实时空闲卡，`torchrun --nproc-per-node=6`、显式`NCCL_P2P_DISABLE=1`。24 tasks×2
  checkpoints必须形成48 rows与6份ownership；formal artifact预计远小于1GiB，沿用本阶段
  已核验的/data1 quota。任一卡在最终复查时被占用则不启动并重新选卡，不共享或干扰他人进程。
- 内部分析已完整结束：48/48 rows、6/6 payload、wall=`272.876s`、peak reserved=
  `19,304,284,160` bytes、0 action-wall/validation/test reads，GPU已释放。task cotangent
  pair cosine mean/median=`.000107/0`、retention=`.041874`，post-update task-mean program
  delta=`.5801/.6128`、retention=`.55537`且无负pair；same-task update的task-mean energy
  fraction在program/BA=`.82990/.91623`。Program-Credit正式负裁决，旧cycle2/4/8禁止。

## 已完成Policy-Lane Coupled Hyperdecoder推进（2026-08-05）

- [x] 完成PWAD fresh0→200、四点strict correct400与24×4内部分析；曲线
  `77/71/80/80`、breadth=`5/6/5/5`、union/intersection=`115/44`。64 atoms广泛使用但
  A/B mixing row stable rank约`1.000002`，正式负裁决并禁止resume到400。
- [x] 结合direct SFT复核与PWAD数学接口封存
  `docs/action_forecast_writer_policy_lane_hyperdecoder_design.md`：取消独立A/B atom
  mixing，让16个public lanes各自用一个32维condition hidden共同生成全部38 targets的
  A/B向量；不强制高rank/正交，不加入监督或LIBERO专属loss。
- [x] 原位替换canonical PWAD runtime与analysis owner，建立fresh incompatible
  launch/checkpoint/config family；完成真实38-target shape、identity、lane ownership、
  condition写出和BA梯度阶段的聚焦CPU合同。完整Writer=`49,041,664`参数，聚焦Writer
  合同`84 passed`，architecture guard无hard/parallel family；formal已在live profile后seal。
- [x] live比较`gpu01/gpu02`，在6张空闲A40上完成longest105/logical-B20三步profile与
  独立fresh0→1→exact-resume1→3；0 OOM/clip，step2起五个主块全部可达，完整训练与
  resume状态闭合，formal config已seal。
- [x] 从clean/pushed代码的独立fresh root训练0→200：200 finite macros、96,000 queries、
  4,800 one-video conditions、8个checkpoint，0 OOM/clip/nonfinite/stall。
- [x] strict评测50/100/150/200完整结束：correct=`70/63/37/61`、breadth=
  `6/4/6/6`，相邻gained/lost=`17/24,14/40,40/16`，四点union/intersection=
  `117/14`、single envelope gap=`47`。四点均400 rows、42 shards、一次启动、全部
  worker exit0、每task 50个无放回视频；macro50 single winner=`70`，正式禁止续400。
- [x] 用既有cold-start analysis owner完成50/100/150/200四点内部分析：96/96 cells、
  6/6 payload、0 target-action/validation/test reads。Policy-Lane确实形成约10个有效输出
  lanes、stable rank `1.34→1.54`及与direct SFT相同量级的跨layer专门化，但same-task
  video hidden/BA能量仅约`.05%/.02%`，且上述漂亮结构与`70/63/37/61`闭环负结果错位。
- [x] 基于完整负证据封存下一credit方法：必须直接在Writer/LoRA生成层获得闭环相对信用，
  不再增加lane/store/rank、强制SFT几何或用functional loss选点；design authority完成前
  不实现或launch。

### Policy-Lane fresh0→200 formal launch contract

- 功能实现与profile seal=`fbc320a`；本段launch-record为docs-only delta，实际启动commit
  由`run_contract.json`记录。启动时worktree必须clean且等于`origin/main`；不传`--resume`
  或`--initialize-writer-checkpoint`，不加载PWAD、v6、profile或resume-smoke Writer权重。
- output=`runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805`；
  log=`runs/logs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805.log`；
  tmux=`ember_policy_lane_formal_r6_fbc320a`。root已核验不存在。
- scale=`200 macros × 24 tasks × B20 = 96,000` logical queries、4,800 one-video conditions，
  every25共8个checkpoint；profile三步稳态约31秒，预计主体约105分钟，正式wall上限190分钟。
- live 2026-08-05启动快照：`gpu01`八卡无compute process，`gpu02`只有4张空闲；选择
  `gpu01:0,1,2,4,5,7`、single-node six-rank、`3+3 NUMA`，显式
  `NCCL_P2P_DISABLE=1`。`/data1` quota使用`299,102,944KiB/1TiB`，单checkpoint实测
  约565MiB，含8个保留点与原子临时副本预计峰值新增低于6GiB。
- 只在50/100/150/200 checkpoint完成后启动strict paired correct400；训练loss不选择
  checkpoint。0→200完成前不续400，也不对中间weak metric增加trick或旁路。

精确命令：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4,5,7 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py --config configs/pi05_as_writer_policy_lane_hyperdecoder_bci_v1.json --mode formal --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805 --stop-after-step 200 --num-workers 0 --log-every 1 --skip-data-sha
```

### Policy-Lane four-point strict correct400 launch contract

- 固定四点`50/100/150/200`，每点8 validation tasks×50 sealed states，correct video
  without-replacement；四点共享task/state/video ordinal/env seed/policy seed与policy noise，
  不得按中途结果替换checkpoint。
- live 2026-08-05快照选择`gpu01:0,1,2`与`gpu01:3,4,5`两组三卡；第一波并行50/100，
  全部结束释放后第二波并行150/200。每点3 replicas/GPU、3 generators/GPU、generation
  batch4；跨节点总占用始终不超过6卡。
- output统一为
  `runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_bci_correct400_noreplacement_seed7_macro{0050,0100,0150,0200}_244b677_20260805`。
  source/config/tokenizer/data与训练合同一致，结果只认single-checkpoint correct、breadth与
  严格配对gained/lost；functional loss不参与选择。
- live quota使用`303,808,444KiB/1TiB`，每root预计LoRA cache与results低于1.1GiB，四点
  峰值新增低于5GiB。prepare后用canonical`evaluate_pi05.py start`启动，失败只按同root
  `resume`恢复，不重建面板。

### Policy-Lane four-checkpoint internal analysis launch contract

- 使用clean/pushed canonical analysis owner，同时读取正式50/100/150/200 checkpoint；覆盖
  24 train tasks、same-task demos0--4、每suite两task fixed-action panel及
  reversed/shuffled。只读task language与action-hidden teacher video，固定action probe不读
  target action，validation/test reads必须为0。
- 必须报告lane hidden跨task/same-task-video差异、16-lane storage/output participation、
  effective BA谱与视频方差、checkpoint churn及fixed-action传递；结合训练ledger中后段
  same-task相邻梯度复现，而不是用functional loss选择checkpoint。
- output固定为
  `runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_internal_all4_r6_20260805`；预计96 cells、
  新增小于0.1GiB，参考同owner既有规模峰值低于20GiB/卡。启动前仍需live确认六张空闲卡、
  `/data1`独立quota与output root不存在；显式`NCCL_P2P_DISABLE=1`。

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/analyze_relative_flow_coldstart.py --mode formal --training-run runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805 --checkpoints runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805/checkpoints/step_00000050 runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805/checkpoints/step_00000100 runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805/checkpoints/step_00000150 runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805/checkpoints/step_00000200 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_internal_all4_r6_20260805
```

## 已封存的Policy-Wide Atom Dictionary推进（2026-08-05）

- [x] 基于direct SFT的跨target组织、v5.2视频传递、v6 absolute及全部负裁决，封存
  `docs/action_forecast_writer_policy_wide_atom_dictionary_design.md`：K64 policy-wide
  atoms + condition-dependent rank16 A/B mixing，fresh identity，不加载历史Writer。
- [x] 原位替换canonical Writer，删除旧320-slot compiler/factor decoder；新增独立config、
  launch/checkpoint schema和五block ownership。聚焦CPU合同确认13,033,728参数、38-target
  shapes、identity、条件写出与真实BA loss的三阶段梯度开启；formal保持profile前blocked。
- [x] live比较`gpu01/gpu02`后用`gpu01:1,2,3,4,5,7`的3+3 NUMA六卡完成最长105-frame、
  logical B20、microbatch2三步profile；独立fresh0→1→exact-resume1→3在补齐新family的
  optimizer restore ownership后通过，1,440 queries/72 videos、六rank状态与五block可达。
- [x] profile证据seal后从独立fresh root完成0→200：200 macros、96,000 queries、4,800
  one-video conditions、every25共8个完整checkpoint；0 OOM/clip/nonfinite，validation/test
  action reads均为0。strict correct400曲线为`77/71/80/80`、breadth=`5/6/5/5`；明显
  低于v6-fast143和严格门，不续200→400。四点配对与内部condition→atom→BA/action
  分析已全部完成：字典广泛active但mixing/public LoRA塌成单一lane方向，正式负裁决。

### fresh0→200 formal launch contract

- 功能代码/config authority=`a924477`；branch=`codex/bci-continuation`，启动时必须clean且
  与`origin/main`一致，最终docs-only launch-record commit由runtime `run_contract.json`
  精确记录。
- output=`runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805`；
  log=`runs/logs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805.log`；
  root必须全新，不传`--resume`或`--initialize-writer-checkpoint`，失败root不作warm-start。
- scale=`200 macros × 24 tasks × B20 = 96,000` logical queries、`4,800` one-video conditions，
  every25共8个checkpoint；profile估时约100分钟，预计peak新增storage小于2GiB。
- device=`gpu01:1,2,3,4,5,7`、单节点six-rank torchrun、3+3 NUMA、显式
  `NCCL_P2P_DISABLE=1`；启动前仍须live确认双节点ownership/显存及`/data1` quota。
- 后续只以50/100/150/200 strict paired correct400、breadth/换手、视频因果与
  condition→atom→effective BA→action内部路径裁决；functional loss不选择checkpoint。

精确命令：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py --config configs/pi05_as_writer_policy_wide_atom_dictionary_bci_v1.json --mode formal --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805 --stop-after-step 200 --num-workers 0 --log-every 1 --skip-data-sha
```

### correct400 rollout launch contract

- 固定四点`50/100/150/200`，不得按中途结果替换；每点8 validation tasks × 50 sealed
  init states，correct video without-replacement，state/video ordinal/policy RNG严格配对。
- live preflight选择`gpu01:0,1,2`与`gpu01:3,4,5`两组三卡；`gpu01:6`为他人进程，
  不接触。第一波并行50/100，完成释放后第二波并行150/200，跨节点总使用始终为6卡。
- evaluator固定formal、50 states、3 replicas/GPU、3 Writer generators/GPU、generation
  batch4；source policy/tokenizer/data/config与训练contract一致。结果选择只认single
  checkpoint correct、breadth和严格配对换手，不用functional loss。
- output roots统一为
  `runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_bci_correct400_noreplacement_seed7_macro{0050,0100,0150,0200}_69563a0_20260805`。

### four-checkpoint internal analysis launch contract

- analysis code=`941c5e3`，只扩展既有cold-start analysis owner；同时读取50/100/150/200，
  覆盖24 train tasks、same-task demos0--4、reversed/shuffled与每suite两task fixed-action panel。
- 新增PWAD专属证据为raw A/B mixing、atom与storage-weighted participation、mixing row rank、
  跨task和same-task-video mixing variance及public target energy profile；已有effective BA谱、
  checkpoint churn和fixed-action传递保持不变，target/validation/test action reads均为0。
- live preflight选择`gpu01:0,1,2,3,4,5`六张空卡；output=
  `runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_internal_all4_r6_941c5e3_20260805`，
  预计新增小于0.1GiB。启动前`/data1` quota用量约282GiB/1TiB。

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 .venv/bin/torchrun --standalone --nproc-per-node=6 scripts/analyze_relative_flow_coldstart.py --mode formal --training-run runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805 --checkpoints runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805/checkpoints/step_00000050 runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805/checkpoints/step_00000100 runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805/checkpoints/step_00000150 runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_formal_fresh0_200_r6_20260805/checkpoints/step_00000200 --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_internal_all4_r6_941c5e3_20260805
```

## 已完成的SFT-Anchored Policy Tangent-Basis消融（2026-08-05）

- [x] 完成AS125→cycle2固定参数hybrid：BA层upstream贡献较大、action层factor-output贡献
  较大且随suite反转；24×7×4与8-task action panel完整，0信息墙读取。
- [x] 封存
  `docs/action_forecast_writer_sft_anchored_tangent_basis_design.md`：从v6-fast macro400=
  143启动，冻结semantic encoder和8个factor-output policy basis，只用reward训练上游/
  factor-input coefficients；不同时加anchor/store/router/rank loss。
- [x] 原位升级canonical RL config/runtime/checkpoint/evaluator ownership，并让历史A100
  cold-start按既有source identity跨host匹配；删除一次性hybrid分析入口。
- [x] live preflight后完成macro400六卡只读progress diagnostic；通用内容、binary、
  all-failure、counterfactual与non-pixel门全过才进入profile。
- [x] 完成独立one-cycle A40 profile：96 rollout/two finite updates、五block全可达、
  8 basis与440 semantic tensors完全不变、76个预注册系数张量全部改变，peak
  reserved`19.48GB`、0 OOM/watchdog，cycle1 checkpoint完整；profile权重弃用。
- [x] fresh formal macro400→cycle1与strict correct400完成：`143→142`、breadth`6→7`、
  gained/lost=`20/21`、intersection/union=`122/163`、`p=1.0`。Spatial净增3但其余
  三suite净降；预注册门失败，禁止resume cycle2。
- [x] 将`142`明确封存为v6强SFT warm-start上的factor-basis冻结消融，而非fresh新Writer
  成绩；它否定basis旋转为唯一/主要漂移根因，不证明LoRA质量或视频特异性改善。
- [x] 由全部证据设计fresh policy-coordinated条件LoRA架构；结果为上节Policy-Wide Atom
  Dictionary，首阶段以AS学习，RL只作为后续可选校准。

## 已封存的Progress-Credit裁决与condition-to-policy分解（2026-08-05）

- [x] 复核历史RL：确认旧PI05 Writer-RL只是success-filtered executed-prefix BC，没有
  failure、advantage、old/current ratio或trust region；不得恢复或改名复用。
- [x] 封存task-relative flow-credit design：fresh v6 AS cold start，随后teacher action
  永久关闭；full24、K4 official random resets、task-local LOO advantage、Nmc4、PPO正项
  与SPO负项，Writer-only update。
- [x] 原位恢复唯一v6 Writer并替换canonical RL/checkpoint/evaluator；删除Target-Owned、
  flat task-local与旧RL并行活动路径。210项全仓分组回归与architecture hard gate通过。
- [x] 在live空闲`gpu02:1,2,3,4,5,7`完成longest105 AS profile：logical B20/B2，峰值
  allocated/reserved`34,948,858,880/44,816,138,240` bytes，三步finite、0 OOM/clip。
- [x] 独立fresh0→1/exact-resume1→3通过；1,440 queries、72 videos、五主block到step3
  可达、source policy trainable=0；AS config已seal。
- [x] clean commit/push至origin main，从fresh identity完成AS 0→25：12,000 queries、
  600 videos、wall810.991s、0 OOM/clip并保存完整step25 checkpoint。
- [x] 用canonical reward cycle完成pre-update K4 coverage：96 rollouts、25 success，但仅
  12/24 tasks有success，coverage未过；9 mixed tasks提供非零credit。
- [x] 完成最长failure、Nmc4、两epoch RL profile；ratio/clip/grad健康，峰值reserved
  45,183,139,840 bytes。修复asset runtime binding和physical EGL card mapping。
- [x] exact-resume同一AS root 25→50；累计24,000 queries/1,200 videos，step50完整。
- [x] step50 K4严格配对得到38/96 success、14/24 coverage、10 mixed；相对step25
  gained/lost=`19/6`，净积累但仍有task换手，coverage未过。
- [x] 根修outcome-skewed local credit使快rank提前进入NCCL的问题：每epoch先用独立
  FileStore all-rank-ready。原六卡96-rollout两epoch重放，96/96 ledger字节级一致、
  finite更新/完整checkpoint/0 watchdog。
- [x] exact-resume同一AS root 50→75；累计36,000 queries/1,800 videos、75 finite macros。
- [x] step75 K4严格配对得到47/96 success、18/24 coverage、13 mixed；相对step50
  gained/lost=`21/12`，coverage净增4但task4换出，门仍未过。两epoch profile完整且0 watchdog。
- [x] exact-resume同一AS root 75→100；累计48,000 queries/2,400 videos。step100 K4为
  52/96 success、17/24 coverage、11 mixed，相对step75 gained/lost=`14/9`且task20换出；
  aggregate上升但breadth回落，正式RL不启动。
- [x] 完成AS step25/50/75/100的96-row真实BA谱、rank/能量、same-video与固定action
  审计：near-rank1为历史复核，video能量约`.13%`平台且失败tasks变化不更小；order到
  action路径有效。拒绝rank/scale/store小改，正式RL仍受coverage门约束。
- [x] exact-resume同一AS root 100→125并重做严格K4：50/96 success、19/24 coverage、
  14 mixed；相对step100 gained/lost=`10/12`，task5/29新获coverage但5个task仍全失败。
- [x] 完成step100/125两点条件写出审计：norm继续增大但video-energy和demo→action差异
  不增；success变化与video-energy变化显著负相关，binary-only Flow-Credit正式负裁决，
  不继承profile update、不续AS150、不启动formal outer cycles。
- [x] 封存
  [`docs/action_forecast_writer_task_grounded_progress_credit_design.md`](docs/action_forecast_writer_task_grounded_progress_credit_design.md)：
  冻结AS125 semantic encoder，以task-token patch/Action-Expert interaction的teacher与
  rollout首尾内容delta建立bounded start-relative potential；mixed binary优先、
  all-success零梯度、仅all-failure semantic LOO。禁止normalized-video-time、teacher
  action、privileged state和LIBERO特化规则。
- [x] 原位实现fresh incompatible config/schema与唯一launcher的`diagnostic`模式：显式
  保留rollout起点/terminal agentview，冻结AS125 semantic encoder，计算correct/wrong/
  shuffled/reversed utility并按实际world size封存；0 optimizer/backward/checkpoint，
  未过门前profile/formal fail-close。聚焦CPU合同`23+53 passed`且architecture无hard。
- [x] 完成AS125严格配对六卡只读机制裁决：50/96 outcomes逐项复现，mixed agreement
  `13/14`、pair AUC`.8913`，五个all-failure task均有非退化utility，三种视频反事实和
  pixel nuisance门全部通过；0 optimizer/backward/checkpoint。
- [x] 完成AS125-fresh、不可续训的two-epoch Writer-update profile：5/5 all-failure
  task非零LoRA梯度、五block可达、observer grad0、ratio/NCCL/A40全部健康；95/96完整
  rollout配对，唯一成功终止时刻少7 actions不改变outcome/credit。
- [x] formal首次fresh0→1完整收集96 rollout与24 task credit后，在第一轮gradient sum
  暴露旧`FileStore` ready的rank seq18/17分裂；600秒watchdog终止，0 update/metrics/
  checkpoint。定位为CUDA完成语义和临时store生命周期竞态，不以timeout或少卡绕过。
- [x] 根修为CUDA synchronize→launch-unique/cycle/epoch原子rank markers→NCCL；同一输出
  目录连续两个新torchrun session的真实六卡探针均6/6 markers、sum21，旧marker不污染。
- [x] clean/pushed`30977b5`用全新root完成原96-rollout/two-epoch formal fresh0→1：两轮
  6/6 CUDA-complete markers、2 finite updates、完整双ledger checkpoint、0 watchdog/
  OOM；5/5 all-failure梯度与五block可达，observer grad0。
- [x] 同一strict panel完成AS125 baseline与cycle1 correct400：`97→104`，gained/lost=
  `22/15`、breadth=`5→4`；净增集中Object-1，Spatial-1失去唯一成功。400对effective
  BA只变化`.01677`且near-rank1几何不变，不能宣称task drift已解。
- [x] live preflight后从同一formal checkpoint exact-resume cycle1→2，保持两epoch、
  task/video schedule、3+3 topology与全部信息墙；cycle2为49/96 train successes、
  21 active-credit tasks，完整checkpoint/双ledger且0 watchdog/OOM。
- [x] 同一strict correct400完成cycle2=`102`、breadth4；相对cycle1 gained/lost=
  `15/17`，Object-1`31→26`、Object-3`19→22`且无新task coverage。按预注册门停止
  cycle4/8，同recipe续训轴负裁决。
- [x] 在固定train-task/video/action panel上完成AS125/cycle2参数hybrid因果分解，区分
  factor-output policy basis与上游condition composition对effective BA/action更新的贡献；
  先裁决再选择冻结basis、全task policy-distance anchor或新basis/coefficients接口，
  不同时改多个变量。长期single-checkpoint strict`>150/400`不变。

## 历史BCI Policy-Target-Owned Factor推进（2026-08-04）

- [x] owner恢复长期`>150`目标，要求科学问题自行深入分析后继续推进，不再为中间判断
  请求确认；继续禁用subagent、保持one-shot与效率优先。
- [x] 重新分析两套direct rank-128 Source-SFT step400的effective LoRA：两者同样以
  q-dominant低秩更新为主，排除“Writer near-rank1本身导致漂移”。
- [x] 确认SFT的policy-target specialization跨配方稳定：q/v layer-energy profile
  Pearson=`.9931/.9904`、rank correlation均`.9835`，对应target BA cosine均值
  `.8450/.8529`；而Direction Store的q/v跨层余弦`.93--.97`、能量近uniform。
- [x] 封存
  [`docs/action_forecast_writer_target_owned_factor_design.md`](docs/action_forecast_writer_target_owned_factor_design.md)：
  保留Target-Bound Core/private A-E-D/rank-read，删除task Direction Store；76个公开
  A/B tensors各自拥有完整bias-free`1024→256→width`head，不加正交/rank/energy约束。
  机制与AS objective解耦，后续可直接由rollout reward训练。
- [x] 原位替换唯一canonical Writer、config/schema/checkpoint/inference/internal
  decode路径；删除只服务Direction Store的额外frozen text-anchor forward和专用测试，
  不保留并行模型/runner。精确参数预算预期`47,857,920`。
- [x] compileall、config/fresh checkpoint family和89项Writer focused tests通过；
  architecture diff净减少334行，无新增hard signal。环境完整评测合同在显式加载
  `.env.local`资产路径后52/52通过。
- [x] authority与canonical替换以`20479d3`clean commit/push；live六卡formal-seed
  fresh0→1工程smoke finite且显存通过，但最长仅82帧，已明确不冒充longest105。
- [x] 根修profile/formal seed切换：磁盘config固定formal seed20260722，profile mode
  自动解析声明的seed172并写入run contract，不再手工改seed；24项聚焦测试通过。
- [x] 实时重查后在`gpu02:1,2,3,4,5,7`完成clean seed172 longest105、logical B20/B2、
  fresh0→1/exact-resume1→3；峰值reserved`43.936GiB`，step2起五主块finite/nonzero，
  config已恢复formal seed20260722并seal，profile权重不进入正式轨迹。
- [x] profile封存后从fresh identity完成0→200、每25保存；50/100/150/200严格配对
  correct400=`99/76/86/68`，breadth=`6/6/7/5`，union/intersection=`136/37`，
  envelope gap37。winner macro50=99，低于Direction Store129、v6-fast143和门151，
  按预注册判据不续400。
- [x] winner完成六卡refs1五条件内部分析。76 heads已把q/v跨层BA余弦降到约0，证明
  policy-target ownership生效；但层能量过度集中、LoRA norm下降，Program差异扩大的
  BA没有转成等比例action差异。完整训练梯度又显示factor task directions近随机正交，
  同task+demo没有稳定重现。正式拒绝policy-target sharing为主要根因。
- [x] 按owner此前要求，在本轮rollout与全部分析封存后暂停；不启动下一架构、训练或
  评测。下一讨论边界是重构condition-to-policy credit，使video条件获得policy-aware、
  闭环有用且可由AS或未来reward共同训练的累积方向，而不是继续加head/gate/scale或
  强制SFT几何。

## 已完成并负裁决：BCI Semantic Direction Store（2026-08-03）

- [x] owner解除VR结果后的阶段暂停：继续one-shot，取消Writer参数量软上限，优先
  重构条件生成方向的存储/组合，允许配套修改训练；继续禁用subagent并保持效率优先。
- [x] 重新按内部证据区分task drift与functional-loss不可预测性：后者不能单独解释
  漂移；SFB的核心缺口是已学会activation routing，但约97%梯度仍进入共享factor且
  方向持续轮换。
- [x] 完成`Semantic Direction Store Writer`设计authority：frozen text-only semantic
  anchor在24 train languages上建立8个固定centers，每task稳定top2；每store拥有完整
  1024→256→factor-width input/output参数，预计Writer为37,355,776参数。保持完整
  Core/A/E/D value、one-shot、single LoRA与信息墙。
- [x] 原位替换canonical SFB factor path，不保留并行架构；增加BCI B2切片可重建的
  keyed independent Beta/Gaussian sampler，退役VR estimator的活动配置。
- [x] 建立仅基于24 train language的center authority；完成route、独立W_out、sparse
  gradient、identity、freeze、B20/B2 parity与fresh/resume聚焦验证。
- [x] live比较`gpu01`/`gpu02`后用`gpu02:0--5`六张空闲卡完成longest105真实profile；
  fresh0→1/exact-resume1→3通过，峰值reserved`43.893GiB`，无需改变logical B20、
  full24 raw mean或一次AdamW。根治rank-local构造与NCCL生命周期错位，并封存BCI
  A40/NCCL2.28显式SHM transport fail-fast到代码和`AGENTS.md`。
- [x] clean pushed`91feeef`从fresh identity完成0→200：200 finite macros、96,000
  queries、4,800 videos、8 checkpoints；四点paired correct400为
  `129/107/120/129`，breadth=`7/7/7/5`。macro50以同分更高breadth成为single
  winner129，未超过v6-fast143或严格门151，不续到400、不做五臂。
- [x] 完成macro50 refs1五条件内部分析：固定route和A/E/Core→BA/action路径成立，
  但same-task Program relative-L2 `.93377`到factor/BA只剩`.01935/.03242`；16个
  active rank坐标的stable rank仅`1.000043`、首奇异值能量`.999957`。正式拒绝“只靠
  独立完整factor stores解决漂移”，定位到Program→public A/B的多维方向形成失败。
- [x] 根修复六卡内部分析的历史4-rank假设：任务LPT分配与最终Cartesian sealing均绑定
  实际`world_size`，clean`a115b06`六rank/8-task/5-condition真实分析完整封存；规则写入
  `AGENTS.md`。当前无GPU工作，按owner要求暂停讨论。

- [x] 完整阅读authority、迁移handoff、架构/recipe设计、代码与历史证据；核清
  data/model/tokenizer/source checkpoint、formal outputs、环境与simulation assets。
- [x] 核验A100→BCI迁移清单、hash、223项旧环境CPU回归和四卡训练/评测验收；新资产
  统一位于项目`data/`、`models/`、`runs/`和`evidence/`。
- [x] 实时比较`gpu01`/`gpu02`；首轮仅`gpu02`的0/1/2/3/4/7空闲，六卡collective
  通过，未触碰有他人任务的卡。
- [x] 在不改变逻辑B20、full24 raw mean或优化器合同的前提下，实现policy B2物理
  microbatch与6 ranks×4 tasks拓扑；23项focused CPU测试通过。
- [x] 未冻结工程profile完成fresh0→1/exact-resume1→3；峰值allocated/reserved约
  `34.97/47.11GB`，三步finite，五主block从macro2起可达。
- [x] 实现提交/push为`391f183`；从clean pushed commit重放同一最长路径与
  exact-resume并seal。一次setup stall未复现，最小collective和原命令重试均通过；
  profile checkpoint不得warm-start。
- [x] 预注册BCI VR formal launch contract：fresh 0→200、六卡、logical B20/B2、
  96,000 queries、every25、1.5GiB峰值预算、fresh root、tmux/log、启动与resume门。
- [x] 首次formal在checkpoint前发现A40 overlay误保留profile teacher seed`172`并安全
  停止；修回正式`20260722`、新增sealed seed fail-close，partial root禁止resume/评测。
- [x] 从fresh identity完成VR 0→200、every25；200步/96,000 queries/4,800 videos、
  8 checkpoints和全部hash/信息墙门通过。严格配对correct400为
  `76/88/126/107`，breadth=`7/4/7/5`，single winner126；四点漂移与matched
  VR→SFB机制/held-loss分析完成，正式负裁决，不续到400、不做五臂。
- [ ] 长期Goal仍是同一single checkpoint correct严格超过`150/400`并尽可能提高；
  不复用VR checkpoint、不使用subagent、checkpoint融合、挑video或信息捷径。

## Post-seal A100研究窗口（2026-08-02 19:18 UTC起）

- [x] owner重新授权约十小时GPU4--7研究窗口；以`f9a144c`为迁移封存基线，创建外部
  delta ledger与独立`codex/postseal-target-bound`写worktree。
- [x] 完整复核CV-ADR、historical Coherent-Procedure与Target-Bound设计；判定首项
  真实证伪应保持mean-backed Core，并让task/Core语义在38 targets和A/E/D私有时序
  读取之前进入，而不是再加gate/scale或硬task-ID experts。
- [x] 将远端Target-Bound实现`b260a57`无冲突移植到`f9a144c`，得到`fbbb784`。
- [x] 恢复frozen Python/CUDA环境，完成48项focused CPU vertical path并立即push。
- [x] 只对GPU4--7做一次live preflight；完成longest105 B20三cycle、fresh/exact resume。
- [x] Target-Bound clean frozen`cfd26df`完成fresh0→200并并行评测
  50/100/150/200 correct400=`75/120/90/110`；不续到400、不做昂贵五臂。
- [x] winner macro100内部反事实证明A、D、causal memory、Core与Program均传到
  effective BA/action；最早剩余失败接口定位为shared factor conditional coexistence，
  而不是视频路径断路。
- [x] 完整实现Semantic Factor-Basis并push`e87363f`；11,159,296参数、55项聚焦
  回归、longest105 B20三macro及fresh0→1/exact-resume1→3均通过，seal为`f5ddfe3`。
- [x] clean frozen`f5ddfe3`完成fresh0→200；paired correct400为
  `69/91/118/127`，macro200 breadth8，50→200 gained/lost=`68/10`，但
  150→200仍为`38/29`。
- [x] 同一root exact-resume 200→400并并发评测250/300/350/400；完整曲线
  `69/91/118/127/117/81/126/120`，single winner仍为macro200，第二小时明确轮换。
- [x] 在不替换SFB架构的前提下实现variance-reduced functional estimator；mode接线
  修复与回归commit为`50662a8`，只改变exact-marginal flow time/noise批内依赖。
- [x] 完成VR longest105 B20三macro与formal-seed fresh0→1/exact-resume1→3；matched
  早期梯度稳定性小幅改善，但尚无fresh0→200或closed-loop证据。
- [x] 02:42 UTC停止全部GPU工作；不在剩余窗口启动无法完成paired评测的新训练。
- [x] 封存并交付最终代码/文档及34行、16,483,938,529 bytes post-seal
  `must-transfer`增量清单；迁移后由owner重新授权才运行VR fresh0→200与
  50/100/150/200 paired correct400。

## A100清理与BCI迁移准备（2026-08-02）

- [x] 核验EMBER/MemLLM Git、工作区、tmux和训练/评测进程；没有活动实验需要继承。
- [x] 创建并验证EMBER 138-ref全量bundle；复验MemLLM 186-ref历史bundle与SHA。
- [x] 清理52个Writer LoRA caches、138个profile/smoke/resume/WIP roots、退役
  SmolVLA outputs/numeric数据、旧feature cache、endpoint LoRA tensors、reseal/
  capacity roots和Codex archive；每批有外部manifest与SHA。
- [x] 定向测试发现active `hf-libero`的simulation-assets symlink依赖原
  `ember_assets`；按精确HF revision只恢复426.57MB必需snapshot，原4.28GB多余缓存/
  revision不恢复，4个原始contract失败测试重新通过。
- [x] 将source step1000精简为selected raw policy inference asset；保留policy、
  trainer state和manifest，formal inspector通过，明确不再支持source exact resume。
- [x] 删除可按精确revision重下的generic `lerobot/pi05_base`，封存revision、bytes和
  SHA；LIBERO exact dataset、tokenizer、formal outputs与feature cache v2保留。
- [x] 保守保留原封存60个正式checkpoint roots和406个complete evaluation roots，
  并登记post-seal新增2个正式训练root和12个formal correct400 roots；它们是task
  漂移、checkpoint轮换和架构×recipe混杂的唯一证据，不只留winner。
- [x] 封存EMBER/MemLLM完整dependency freeze；验证后删除EMBER venv/package cache；
  owner关闭MemLLM venv消费者后也删除其7.60GB环境，两者都列为BCI重建项。
- [x] 删除55个clean辅助worktree、36个本地实验branch和obsolete stash；历史由
  bundle保存，Target-Bound仍在GitHub远端分支。
- [x] 评测preflight支持`EMBER_STORAGE_ROOT`，不再写死`/data/ymdai`；定向测试通过。
- [x] 重写README、AGENTS、active handoff和execution brief，新增迁移handoff与机器
  可读资产表；A100 Codex不迁移，新Codex从Git authority接手。
- [x] 完成最终全量验证、cleanup manifest总SHA、Git commit/push与两repo状态核验。

迁移由后续智能体执行。本计划不授权跨机写入或迁移后GPU实验。

## 历史交接顺序（2026-08-02）

- [x] 恢复exact v5.2 topology到mature task-complete/B20/long-first/
  fast-decay400 config，并完成最长视频profile与exact-resume smoke。
- [x] 对v5.2 step900重新生成400套correct-video LoRA并完成零rollout几何分析；
  证明近rank1来自建设性coherence，不是坐标能量失衡或负向相消。
- [x] 撤回Coherent-Procedure/B-only residual，封存完整SPG模型与CP-24训练设计。
- [x] 新session完成全部authority、代码、Git历史和正式artifact审计，并从独立
  frozen `60f4508` worktree启动v5.2 task-complete fresh macro0→400。
- [x] v5.2 macro150/200/350/400 paired correct400完成，为
  `51/91/106/120`；winner macro400五臂`120/109/107/111/124`，exact50几何和
  Core→Procedure→LoRA→action内部传递完成，不融合checkpoint。
- [x] v5.2 run挂起后充分阅读全部文档、代码、Git历史与正式outputs，形成完整
  v1→SPG证据模型。
- [x] 独立复核并实现canonical SPG+CP-24；精确参数`10,633,216`，全仓
  `201 passed`，architecture guard无hard violation。
- [x] SPG最长105-frame B20四卡三macro profile通过；定位并修复共卡NCCL
  chunk只入队导致的CP-24 starvation，修复后step为
  `20.536/18.578/18.546s`且全部主模块梯度可达。
- [x] 同一clean `f6d4876`和formal seed完成fresh0→1→exact-resume1→3；step1
  文件bitwise不变，三步metrics/LR/cursor连续，CP chunk gather/sync严格对应。
- [x] resume seal已提交并push；从最终clean `79fb7ee` frozen worktree fresh启动
  macro0→200，首macro的24-task/B20/long-first/CP同步合同通过。
- [x] SPG macro0→200与四个paired correct400完成：`97/115/77/100`；一小时门
  失败，不续到400、不做五臂。
- [x] 完成SPG exact50四候选几何、macro100 refs2分层反事实、24-task gradient
  coherence和checkpoint drift分析；最早失败定位为compiler routing同质化，
  CP-24无法恢复近正交task innovation。
- [x] 按组件×recipe重审v7/v8/v10/Loom/Recenter/Core-Program/Prior/Target-
  Spectral；只封存局部强负机制，不整体判死与fast recipe混杂的架构思想。
- [x] 封存Unified Causal Program设计authority和现有B20 phase-variance审计。
- [x] 实现UCP canonical路径、raw full24 Gram诊断和无偏20-strata B20；删除旧SPG
  Core add/global mixer与CP投影active path。真实参数`7,683,328`，全仓
  `203 passed`；fresh formal config先保持pending，由下一项live evidence解封。
- [x] 完成shape/mask/identity/freeze/gradient/resume和最长105-frame B20 profile；
  三macro峰值reserved约77.62GiB，formal-seed fresh0→1→exact-resume1→3逐文件
  不变，选择B20。
- [x] 提交并push UCP live seal `c94f1c6`，从新的clean detached commit建立formal
  frozen worktree并fresh启动macro0→200；首macro合同健康，未从smoke warm-start。
- [x] 完成clean frozen `c94f1c6` fresh macro0→200和50/100/150/200 paired
  correct400：`82/117/100/110`；union169、single best117，门失败，不续到400、
  不做五臂。
- [x] 完成macro100 refs1内部纵向和CUDA batch-shape诊断；保持B5 carrier后
  canonical重算各层严格一致，reader路由健康，但dynamic A/D写出仅约2–5%。
- [x] 定位首次`a4b06f5` exact50失败为rank1本地异常被NCCL gather掩盖；失败root
  只有run contract。实现reference上下文、rank failure artifact、torchrun
  fail-fast和analysis-only Gloo控制组，不修改训练protected owners。
- [x] 新refs2精确暴露rank1异常为`libero_spatial task3/reference1`的rank-gauge
  sanity失败，并验证failure artifact与torchrun立即收割；加入BA/action/raw A/B
  判别量；确认BA误差仅`1.299e-9`、bf16 action drift为`.002047`，修正错误的
  位级动作硬门而保持BA `2e-5` fail-close。
- [x] 用新clean `c4b85e8` root验证refs2通过；四rank共16 rows、无failure，随后
  用另一root启动exact50并封存clean provenance。
- [x] 完成exact50：8 tasks×50 references共400 rows、四rank各100、0 rollout、
  无failure；封存逐task same-video variance、Program→BA→action、消融和有效LoRA
  几何，确认pooled BA/action条件方差仅`.09008%/.01656%`。
- [x] 在独立write worktree完成并集成UCP exposure-matched serial-4单路径：六phase
  重建同一full24 cycle，LR按cycle阶梯重复；全仓`233 passed`，fresh incompatible
  config/checkpoint及midcycle cursor通过，architecture guard无hard violation。
- [x] 完成serial-4最长105-frame B20 profile：18 updates/3完整cycles、B20 finite、
  task38/demo36真实105 sampled frames；formal seed fresh0→1→resume1→3→跨cycle
  boundary到7，step1/3文件不变、scheduler/cursor连续，canonical config已seal。
- [x] 完成跨v5.2/SPG/UCP/v6新旧recipe的strict surrogate审计；确认同一held panel
  无法选择closed-loop checkpoint或追踪逐task漂移，并把历史结论分成局部机制否定、
  recipe混杂和现有证据不可识别三层。
- [x] 完成architecture×training mechanics审计；量化old/full24的6×LR integral、
  Adam记忆、clip/WD/重线性化差异和v6 matched参数路径，明确它仍是多因素bundle。
- [x] serial-4从clean frozen `3db82df` fresh identity完成1,200 optimizer updates；
  96,000 queries/4,800 videos/200 cycles与raw-full24逐项同曝光，8个checkpoint和
  信息墙合同完整。
- [x] serial step300/600/900/1200 paired correct400完成，为
  `89/100/121/107`；同曝光raw为`82/117/100/110`，差值
  `+7/-17/+21/-3`，best仅+4且漂移未解。
- [x] 补齐raw macro150与SERIAL step900 exact50同曝光内部对照；SERIAL将
  x-only→full BA/action差异从`.0653/.01269`提高到`.4184/.12999`，证明
  update granularity强烈控制视频动态写出，但task能力仍轮换。
- [x] 独立复核、实现并集成AP-ADR；精确参数`10,241,024`，保留mean-backed
  Core、outgoing A/E/D raw Program、独立Core/Program reads和coherent heads，
  删除terminal amplifier、global mixer、谱约束与并行旧路径。
- [x] AP-ADR最长105-frame B20三macro、formal-seed fresh0→1→exact-resume1→3
  全部通过；step1七个payload逐项不变，seal `7dffb6f`已push。
- [x] clean frozen `7dffb6f`的AP-ADR formal fresh macro0→200自然完成：200 cycles、
  96,000 queries、4,800 one-video conditions、wall `3898.217s`，信息墙读取0。
- [x] macro50/100/150/200 paired correct400完成，为`91/81/94/91`，breadth
  `6/6/5/7`；winner macro150仅94且四点持续能力轮换，一小时门失败，不resume、
  不做五臂。
- [x] 修复PI05 action sampler永久切换attention backend造成的内部分析重放污染；
  `5d93af3`后8-task refs1逐层/BA/action严格零误差重放。AP根因定位为contextual
  Program只作高熵K、raw Effect DC主导V：Effect-only距full BA仅`.82%`，反转
  temporal keys仅改变BA `.052%`。
- [x] 封存下一整体CV-ADR设计：保留mean-backed Core与separate dual reads，删除
  Program key/raw-value二轨，让同一causal contextual Program直接作为K/V；无新
  gate/scale/loss，预期参数不变，fresh schema。实现前先完成UCP normalized
  randomized-group4训练因果格。
- [x] endpoint10实现已合入`544c0ef`/`2055a82`：exact ten-step sampler无
  autocast/ACTION输入，sealed512 pairing、finite和historical provenance均
  fail-close；CPU全仓`222 passed`。
- [x] 在生成任何endpoint数值前预注册executed-first5主指标、18-candidate
  global/within-family/per-task关联门及两个matched-recipe方向；secondary不能
  覆盖主门。
- [x] 从三个clean frozen历史extension commits生成并核验portable-v2 LoRA cache：
  v5.2-old 64、v6-fast 512、v6-old 64，信息墙全0；所有tmux自然退出。
- [x] 完成真实CUDA profile/parity与18-candidate四rank formal endpoint诊断：
  9,216 rows，global Spearman `.258398`、permutation `p=.298447`，预注册all gate
  失败；endpoint10永久只作负诊断，不进入训练、loss或checkpoint选择。
- [x] 在独立UCP worktree实现fresh raw-full24 vs cycle-normalized randomized-group4
  受控cell：task/query-keyed stateless policy noise/time、随机Latin group4、LR/beta/
  decay/scheduler exposure composition、fresh checkpoint family与midcycle cursor，
  聚焦测试`31 passed`。
- [x] 完成group4 longseed172真实105-frame B20四卡18-update/3-cycle profile；每cycle
  24 tasks恰好一次、step2起主块梯度可达、峰值reserved `83,647,004,672` bytes。
- [x] 完成group4正式seed 0→1→3→7与raw 0→1→3 exact-resume；cycle0 teacher-video
  assignments逐项一致，step1/3未改写，scheduler/cursor连续，两份fresh配置已seal。
- [x] 将封存`b52cb54` UCP运行面逐blob恢复为唯一canonical；退役AP/endpoint可执行
  路径，聚焦`107 passed`、compileall/JSON/diff check通过。
- [x] task-query raw `configured-decay400/autoscaled200`及50/100/150/200 paired
  correct400完成：`81/72/107/78`；best macro150后lost43/gained14，不续训、不做
  五臂，只封存为scheduler ablation（analysis SHA `bfd580d4...0993`）。
- [x] 提交并push formal scheduler fail-close与corrected total=400/2400；从clean
  `cfc2ad1` fresh完成真正fast400 raw及50/100/150/200 paired correct400：
  `89/71/82/117`。winner macro200只有117、breadth7且仅4 tasks达到5次成功，
  不做五臂；candidate/scheduler-interaction SHA分别为
  `7b7d9822...dd3`/`81eca3cc...ab7e`。
- [x] 审计task/query RNG-v1并定位真实合同偏差：CUDA Gaussian noise按query锁定，
  但LeRobot PI05 Beta flow timestep仍从ambient CPU generator采样。step0 identity的
  四个跨rank重叠task在action rows/video/seed完全一致时loss仍改变，直接证伪跨
  rank/phase stateless合同。
- [x] 正常停止`cfc2ad1` GROUP4 formal于physical step307/51 complete cycles；保留
  metrics与step150/300 checkpoint为invalid-contract provenance，禁止resume/eval。
- [x] 完成CPU+CUDA task/query RNG-v2修复：同时fork/seed/restore两个generator，
  升级config、checkpoint family及三类state schema；`dae13bf`已push，CPU全回归
  `241 passed`。
- [x] 从`dae13bf` frozen authority仅在GPU4--7完成RAW 0→1→3与GROUP4
  0→1→3→7 fresh/exact-resume及跨rank manipulation；四task loss/gradient逐位相等，
  CountSketch最大差`5.82e-11`，两份formal config重新seal。
- [x] 从全新root和fresh identity完成RNG-v2 RAW与cycle-normalized GROUP4，固定
  cycle50/100/150/200 paired correct400为`72/87/86/89`与`77/76/66/100`；
  行为门false，GROUP4不迁移为CV默认。v1 RAW只作observed bundle，v1 GROUP4不进入
  裁决。
- [x] 完成RAW/GROUP4 matched cycle200 exact50：GROUP4将A/D→BA从`.058999`压到
  `.013291`且8/8 tasks一致，唯一fixed-action反向异常来自0-success task；paired
  analysis SHA为`7201364a...11fd`。
- [x] 在RNG-v2 closed-loop outcome前冻结operator裁决层次：cycle200 endpoint与四点
  cycle-AUC、single-best breadth、success-set churn/envelope gap、逐task/phase-cost
  方向及A/D→BA→action传递共同决定；含混时CV首跑保留更简单RAW。
- [x] 完成RAW RNG-v1/v2单变量训练噪声审计：仅CPU Beta timestep identity变化即使
  四点曲线差`-17/+16/+4/-28`、matched梯度草图余弦中位降至`.163--.193`；将其
  定位为optimizer-basin敏感性而非v1优越性或seed-general估计。
- [x] 在独立worktree实现并以merge `b97960f`集成CV-ADR canonical路径；参数
  `10,241,024`、完整CPU回归`226 passed`、结构门无hard violation；旧UCP executable
  path已退役，历史由Git/artifact保存。
- [x] 完成CV-ADR teacher-seed172最长105-frame B20三macro profile与formal-seed
  fresh0→1→exact-resume1→3；真实105帧、B20、五主块可达、step1七文件不改写，
  RAW config已解除profile blocker。
- [x] 从post-fix clean frozen `254ade4`与fresh identity启动CV-ADR RAW macro0→200；
  tmux健康、首macro合同通过，每25保存，未从profile/smoke warm-start。
- [x] CV-ADR RAW macro0→200及paired correct400完成：50/100/150/200为
  `76/111/99/117`，macro200为右端best、breadth6、top2占`57.26%`；与完全相同
  RAW recipe/exposure/panel的UCP相比四点均为正增益`+4/+24/+13/+28`。
- [x] 完成CV-ADR macro200 exact50和34项结构反事实。Core-only/Program-only距
  full BA为`.606/.812`，Effect-only距full`.0674`，证明双路与新contextual value
  都真实工作；但remove-A只在1/8 tasks达门、contextual-memory order在0/8达门，
  same-task BA variance仅`.1049%`，固定action中位仅`.00856%`。
- [x] 完成UCP RAW/GROUP4/SERIAL source-capability配对审计及UCP→CV同RAW架构审计：
  SERIAL cycle150同时改善source retention与新能力但到200回落；CV macro200的
  `+28`由多保留9个source successes和多获得19个新successes共同构成。由此拒绝
  “optimizer gain只会破坏旧能力”和“CV增益只是遗忘更多”两个简单解释。
- [x] 完成v5.2/v6 old×task-complete五臂、source retention、内部传递与matched
  optimizer dynamics联合审计：新recipe在两架构上都压弱Procedure→BA/action与
  顺序margin，却对absolute产生`-12/+22`的相反winner effect；因此后续按
  architecture×training整体根因裁决，不整体否定与recipe混杂的post-v5思想。
- [x] 将联合审计展开到逐task内部transfer与全部历史checkpoint：v6 matched150的
  recipe收益为source retention -1/new gain +17，但selected +22被Object task3
  单项+24主导；保留v6语义/transition bundle证据，明确重构下游reader/compiler。
- [x] CV-ADR RAW已从step200 exact-resume到400：400 cycles、192,000 queries、
  9,600 videos、every25 checkpoints、all finite、0 clip；full400动力学审计降低
  CP负投影与低LR自然止漂移解释。
- [x] 完成250/300/350/400 paired correct400=`77/69/80/82`；八点winner保持
  macro200=`117`，200→250 lost56/gained16，LoRA norm不坍缩。RAW不做五臂。
- [x] 完成macro200/400固定visit397--399的24-task video/query/flow梯度方差分解：
  video主效应约`.1%`且0/24主导，query/flow支配；macro400 task-mean SNR继续下降，
  24/24刚曝光train条件的matched functional loss改善但held loss横盘、闭环崩落。
- [x] 完成CV GROUP4最长105-frame B20 profile与formal-seed fresh0→1→3→7 exact
  resume；B20 finite、cursor/scheduler连续、step1/3未改写，canonical config已seal。
- [x] 从post-seal clean frozen commit与fresh identity完成GROUP4 0→1200正式控制；
  cycle50/100/150/200 paired correct400=`82/77/73/110`，低于RAW四点均值与winner，
  漂移未解且不做五臂。formal root
  `pi05_as_writer_cvadr_group4_taskcomplete_decay400_formal_dev_r4_b20_seed7_51c0ba5_20260802`；
  config/launcher SHA为`a8dd6c83...da79`/`bd7d3210...4082`。
- [x] 完成GROUP4 cycle200 exact50并封存RAW×GROUP4职责对照：A+D/remove-A/remove-D
  门从`8/1/5→0/0/0`，norm上升而视频职责下降；GPU4--7和本任务tmux/process均已
  清空。这是owner本阶段暂停前最后一项GPU工作。
- [ ] Target-Bound Role-Preserving Program已在隔离feature branch完成CPU实现和
  architecture gate，并以`b260a57a94dc21bd3446b212bfa42f71b037ce13` push；按owner
  暂停边界不做最长视频profile、resume或正式训练，下一session现场复核后才决定启动。
- [ ] 后续每版整体架构只有达到同期有效旧架构水平或显示明确续训价值才开第二
  小时和行为五臂。
- [ ] 持续定位task漂移、视频学习和closed-loop off-manifold根因，禁止补丁式
  gate/scale/bypass；150不是自动完成线。

当前UCP受控格实现与实时状态只认
`docs/active_session_handoff.md`。

## 长期完成定义

只有以下核心项全部完成，长期 Goal 才可完成：

1. 使用过滤后的 LIBERO-90 corpus 训练并冻结共享 π0.5-LIBERO source base；
2. 在固定 24 train / 8 validation 上完成并选择 AS-Writer、RL-Writer（若成立）
   与 corrected mixed-task Source-SFT；
3. 完成 source/seen、correct/same-task-other/wrong/shuffled/reversed 机制证据；
4. 合并 validation 后在 32 source tasks 上从规定初态重训已选方法；
5. 完成 final seen comparison 与 8-task zero-interaction test；
6. 在 8 test tasks 上完成 identity/AS/RL Writer 三臂 task-local RL；
7. 用 8 test tasks × 50 action episodes 联合训练一套 shared target-action
   privileged oracle；
8. 原始 rows、逐 task counts、learning curves、seeds、interaction/data counts、
   GPU-hours、参数量、runtime 与关键 hash 齐全，代码验证、commit、push。

ViVLA-style matched reproduction 和 source-only outer learning 是核心闭环后的
可选项，不阻塞 Goal complete。

## 已封存基础

- [x] 固定 LIBERO-Spatial/Object/Goal/Long 40-task benchmark 和 24/8/8 split。
- [x] 完成 LIBERO-90 × target40 的 3,600-pair specification-only audit：
  排除 19 个 exact semantic/composition 重合 tasks，封存 71 active source
  tasks × 50 successful episodes。
- [x] 从 generic `lerobot/pi05_base` fresh 训练 1,000-step shared source base；
  raw step1000 在 target40×8 screen 为 `46/320`，覆盖 13 tasks 与四 suites。
- [x] 冻结 source-only action/state normalization、source policy、tokenizer、
  model/data manifests 和 canonical π0.5 evaluator。
- [x] evaluator 支持 cost-balanced dynamic queue、persistent model/env、
  Writer per-rollout LoRA、无放回 video schedule 与逐 row paired RNG evidence。
- [x] v4/v5/v5.1 失败根因和 v5.2 五臂成功证据已封存；旧可执行路径已退役。

## 历史执行：EMBER Core-Program Writer

- [x] 完成Loom首段和内部负证据：macro50/100/150/200 correct400为
  `79/106/105/112`；correspondence/confidence/Teacher–Policy gap缺少可靠
  锚点，因此停止且不围绕其scale打补丁。
- [x] Recenter fresh macro50/100/150/200 correct400仅
  `55/84/79/85`；所有tasks低于v6 best且Object-3坍塌。内部更新/幅度证据把
  根因定位为time-centering和弱Core造成的semantic-basis starvation，而非
  简单训练不足。
- [x] 从根因重新封存Core-Program设计：v6 Semantic Core提供slot semantic
  basis，uncapped transition+native Action形成full raw causal Procedure，
  width512 bilinear严格要求两分支共同产生content。
- [x] 原位替换canonical Writer/config/schema，退役Recenter可执行配置；
  fresh不兼容，精确参数`10,905,856`。
- [x] 建立Core permutation、uncapped transition、causality、Core/Procedure
  双必要性、constant Procedure DC、zero-preserving slot block与step0 identity
  的确定性模型合同。
- [x] 全仓`194 passed`、compileall与diff check通过；architecture guard仅有
  既有大文件review提示、无hard violation，active source净删643行。
- [x] 集成canonical commit并push。
- [x] GPU4–7最长105-frame B20三macro独立profile；真实覆盖105帧，
  后两步约`25.871 queries/s`、`194.034 macro/hour`，选择B20。
- [x] fresh0→1→exact-resume1→3通过；metrics/LR/task-video-query/RNG cursor
  连续，step1文件不变，全部523个trainable tensor可达；formal config已seal。
- [x] fresh task-complete macro0→200完成：200行finite metrics、4,800 videos、
  96,000 queries、8个every25 checkpoint，未从profile/smoke warm-start。
- [ ] paired、无放回correct400正在GPU4–7并行评测macro50/100/150/200，
  每卡一个single checkpoint。
- [ ] 一小时best若未达v5.2/v6同期`132–133`同档，不做行为级特异性rollout，
  只做Action/transition/Core/Procedure/compiler/LoRA/action反事实和per-task
  gradient conflict分析后重构下一版；达到同档则默认续训第二小时。
- [ ] 第二小时达到`150`，或至少两个相邻checkpoint稳定`145+`且多task共同
  贡献，才对single-checkpoint winner补same/wrong/shuffled/reversed full400。
- [ ] 在上述absolute与视频因果证据完成前不启动one-shot或RL。

## Phase C：v6 AS-Writer development

- [x] 封存
  [`docs/action_forecast_writer_v6_design.md`](docs/action_forecast_writer_v6_design.md)：
  Task-Grounded Semantic Set + Visual-Transition Procedure。
- [x] 在唯一 canonical Writer/runner 内实现 v6，参数 `10,775,296`，step0 public
  LoRA 精确 identity；不保留 v5.3 平行 executable path。
- [x] 实现 task-complete macro：4 ranks × 6 tasks/rank、每 task 一视频一 LoRA
  和 B20 queries、24 tasks 等权、前 5 次 `no_sync`、一次 DDP sync/AdamW。
- [x] 实现 selected-video cost balancing、rank 内 long-first、跨 macro rank
  rotation、macro-boundary checkpoint/resume 和 every-25 retention。
- [x] GPU4–7 最长 105-frame B20 profile 连续 3 macros finite；稳态约
  `25.793 queries/s`、`193.447 macros/hour`，选择 B20，不触发 B16。
- [x] step1→3 resume smoke 恢复 task/video/query/LR/cursor；真实
  visual-transition gradient 可达。
- [x] fresh macro0→200 正式段完成：200 条连续 finite metrics、8 个 every-25
  checkpoint、24-task 等权消费与终点全文件 SHA 均已核验。
- [x] 在 GPU4–7 对 macro50/100/150/200 做并行 fixed correct400；每卡一个
  checkpoint，6 Writer generators + 6 persistent workers，视频 50 条无放回，
  全局 long-first。结果为 `114/77/120/129`，paired 输入合同全部通过。
- [x] 选择 macro200 为 absolute observed-best；其 129/400 仅覆盖 5/8 tasks，
  与覆盖 7/8 tasks 的 macro150（120/400）差异不显著，保留 breadth 风险。
- [x] 对 macro200 做 correct/same-task-other/wrong/shuffled/reversed
  full400：`129/131/108/111/105`；same同档，correct对后三臂 paired
  `p=.011/.0198/.00094`，方向门通过但 margin 弱于 v5.2。
- [x] 完成 macro200 的 16-reference 内部传递分析：顺序差异由新增
  visual-transition 路径进入 Procedure，并在 fixed-Core 反事实下传到
  effective LoRA/action；无 Semantic Core 顺序旁路。相对 v5.2，Procedure
  差异更强但下游 LoRA/action 差异更弱，需由续训判断是成熟度还是新瓶颈。
- [x] exact-resume macro200→400 与 macro250/300/350/400 correct400 已封存；
  后四点为 `117/118/125/125`，均未超过 macro200=`129`。第二小时提升部分
  breadth但aggregate不涨，能力继续在tasks间迁移；不继续同一full-24 recipe。
- [x] v6拓扑与机制证据封存：Semantic Set、Visual Transition、Causal
  Procedure职责成立，macro200五臂通过方向门；但absolute、margin和跨task
  稳定性未达最终满意门，后续训练粒度/下游compiler仍需改进。
- [x] 实现显式provenance、inference-only的derived Writer checkpoint：
  导出单套平均权重并保持一次Writer前向；原始checkpoint全部保留，training
  resume/warm-start对derived路径fail closed。真实四候选逐tensor独立重算
  完全一致，formal evaluation authority检查通过。
- [x] 按outcome前封存的四候选
  `{150,200}`、`{200,400}`、`{150,200,350,400}`、
  `{150,200,250,300,350,400}`在GPU4–7各跑paired correct400；结果为
  `129/140/144/145`，最后一组相对raw macro200净增16、
  `37 gained/21 lost,p=.04794`，覆盖从5/8增至7/8 tasks。
- [x] 对六点平均winner完成full400五臂与16-reference内部传递：
  `correct/same/wrong/shuffled/reversed=145/134/128/119/122`；
  correct对后三臂均显著且各由至少5个tasks正向贡献，fixed-Core
  Procedure-only保留到effective LoRA/action，Core-only近零。same差11且
  `p=.152`，只比预封存的保守差值阈值多1；absolute仍比150硬门少5。
- [x] fresh运行唯一的v6 fast-decay400稳定化对照：只把cosine
  `decay_steps 2000→400`，其余架构、task-complete B20、AdamW、数据与seed
  全部保持；先0→200并评测50/100/150/200，除可信absolute下降外默认
  exact-resume至400并评测250/300/350/400。八点结果为
  `106/64/111/133/132/117/138/143`；macro400比corrected SFT高34但仍比
  absolute150少7，末段参数位移已很小且350→400净增不显著，不机械续第三段。
- [x] 按outcome前sealed合同筛选四个fast-decay checkpoint-average：
  `{350,400}`、`{200,350,400}`、`{200,250,350,400}`和
  `{150,200,250,300,350,400}`；GPU4–7各负责一组，跑paired correct400。
  结果为`139/135/129/130`，均未超过raw macro400=`143`；只有局部两点平均
  恰好达到SFT+30，四者均未达absolute150。所有源checkpoint、派生checkpoint、
  评测cache/rows/results均保留；完整paired与long-first审计通过。
- [x] 按owner后续决定，把fast-decay从macro400 exact-resume到600并评测
  450/500/550/600；结果`131/130/132/126`均低于macro400=`143`，
  400→600为`31 lost/14 gained,p=.01609`，形成可信post-best下降。
- [x] 对fast-decay单checkpoint best macro400完成正式五臂与内部传递：
  `143/135/125/128/129`；wrong显著，shuffled/reversed方向正确但不显著。
  顺序信号存在于Procedure并能传到LoRA/action，只是task-complete下游增益弱。
- [x] 在不改v6拓扑的前提下实现并封存旧rank-rotating训练范式；最长视频只做
  fixed-B20 profile，`B21`从未运行且正式入口拒绝更大batch。
- [x] v6旧范式fresh训练900 updates并评测step100/500/700/900：
  `98/121/76/95`，step500为single-checkpoint observed-best，后续有显著下降。
- [x] 对旧范式step500完成五臂与16-reference内部分析：
  `121/122/111/84/47`；顺序门强通过、wrong语义门失败，Procedure-only几乎
  完整复现shuffled/reversed的LoRA/action差异。
- [x] 按owner要求在上述证据完成后停下讨论；owner随后已批准v7第一性原理
  设计与自主迭代，因此该临时停止边界结束。

## Phase D：corrected mixed-task Source-SFT

- [x] 从 frozen source base 实现唯一canonical corrected rank-128 Source-SFT；
  每个physical batch一次普通同步forward/backward/clip/AdamW，无gradient
  accumulation或Writer式micro-round。
- [x] 每rank physical batch包含全部24 tasks等量样本；按
  task→episode→chunk分层无放回周期采样，跨rank row不重复、exact resume，
  task-balanced普通batch mean。
- [x] GPU4–7 B144真实fresh step1→resume step3通过；每步全球576 queries、
  峰值allocated/reserved `60.69/74.07GB`，稳态`34.52–36.35 queries/s`。
  B144稳定，未触发B120 fallback。
- [x] 用sealed config从identity fresh训练step0→225（约一小时训练body，
  冷加载另计），每25步checkpoint；225条metrics连续finite，9个checkpoint
  和完整resume state均已核验。
- [x] GPU4/5/6/7各加载step50/100/175/225之一，并行完成四个fixed
  validation correct400；结果为`60/75/77/56`，每点400 rows、36 shards、
  6 workers、零错误，paired seeds和noise prefix完全一致。
- [x] 从step225 exact-resume到450并完成12点dense correct400；
  step400/425为`109/107`同档，step450降到`74`且paired显著，封存full-24
  observed-best step400=`109/400`，不再续训该recipe。
- [x] 实现global-8 cyclic mixed替代sampler：4 ranks×2 tasks、每update
  8个disjoint tasks、连续3 updates完整覆盖24 tasks；保持B144/global576、
  rank-128 LoRA、LR/scheduler与平均task/sample clock不变。
- [x] GPU4–7完成global-8 B144 fresh0→3→resume6 profile；两轮完整cycle、
  3,456 query identities唯一，稳态`36.27–36.38 queries/s`，峰值
  allocated/reserved `60.69/74.07GB`，无OOM或nonfinite。
- [x] global-8从identity fresh训练step0→240并exact-resume到480；16个每30步
  checkpoint全部保留，累计276,480 queries、每task 11,520 samples。
- [x] global-8八点paired correct400为
  `63/83/85/98/90/62/90/105`；step480=`105`为该recipe observed-best，
  但相对step420仅`+15,p=.0627`，相对full-24 step400=`109`为
  `28 gained/32 lost,p=.699`。它没有解决task漂移或提高SFT上限，故不续到
  600；最终corrected Source-SFT development best仍为full-24 step400
  `109/400`。
- [x] 与 v6 使用同一 frozen source base、normalization、policy interface 和
  validation rows；不机械匹配 optimizer steps。

## Phase C2：v7第一性原理Writer

- [x] 封存
  [`docs/action_forecast_writer_v7_design.md`](docs/action_forecast_writer_v7_design.md)：
  明确Core、Action–Effect Procedure与Procedure-content-only compiler的需求、
  已有prefix/suffix信号、最少结构、参数预算和可证伪判据。
- [x] 原位替换唯一canonical Writer：删除Text-only分支与Core-primary AdaLN；
  一次Action Expert forward使用8个原生稀疏suffix anchors；不保留v6/v7
  parallel executable path或checkpoint兼容分支。
- [x] 完成task-span、shape/mask、Core permutation invariance、forward
  transition、D=0 binder、causality、Core-only identity、freeze/gradient、
  public-LoRA schema和checkpoint-resume最短验证；全仓192 tests与真实
  step1→3 exact-resume均通过。
- [x] 只在物理GPU4–7完成最长真实视频profile：B32/B24 OOM，B20三步finite，
  含105-frame视频；稳态约27.48 queries/s、206.08 macros/hour。
- [x] task-complete B20、fast-decay400从identity fresh完成macro0→200并
  exact-resume到400；每25 checkpoint，metrics连续且finite。
- 正式首段launch contract：

  ```text
  workspace  /data/ymdai/projects/EMBER
  branch     main
  commit     ca7db57d0c2d1ec2e7032a44b58238b6de35b1f4
  devices    physical GPU4,5,6,7; 4-rank DDP; NUMA node1
  input      frozen source-base raw step1000 + sealed 24 train tasks
  scale      200 macros = 96,000 queries = 4,800 one-video conditions
  output     /data/ymdai/outputs/ember/
             pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729
  retained   8 every-25 checkpoints; projected peak additional storage <1.3GB
  selection  paired fixed correct400; full five-arm only for current best
  resume     only exact same-contract complete macro checkpoint
  ```

  Exact command:

  ```bash
  numactl --cpunodebind=1 --membind=1 env \
    PYTHONPATH=/data/ymdai/projects/EMBER/src \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 \
    OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    /data/ymdai/projects/EMBER/.venv/bin/torchrun \
    --standalone --nproc-per-node=4 scripts/train_as_writer.py \
    --config configs/pi05_as_writer_language_axial_v7.json --mode formal \
    --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
    --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
    --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
    --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
    --output-dir /data/ymdai/outputs/ember/pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729 \
    --stop-after-step 200 --num-workers 2 --log-every 10 --skip-data-sha
  ```
- [x] paired fixed correct400八点为
  `82/106/114/120/101/114/115/106`；macro200五臂
  `120/112/91/100/69`。内部检查定位joint `8×L` attention近均匀以及
  Core→LoRA影响近零，v7停止。
- [x] 未达门后按“表示→传递→多task优化→闭环目标错位”的最早瓶颈做
  fresh单变量迭代；不使用checkpoint融合、ensemble、contrast/order loss或
  信息墙捷径。
- [ ] 最低成功门仍为single-checkpoint correct400≥150、same≈correct、
  correct显著优于wrong/shuffled/reversed、多task共同贡献，并在独立
  RNG/video permutation下复测成立。150不是自动停止点。

## Phase C3：v8 Hierarchical Action–Effect + Core-Gated Procedure

- [x] 封存
  [`docs/action_forecast_writer_v8_design.md`](docs/action_forecast_writer_v8_design.md)：
  v7 joint attention和Core弱影响的定量根因、hierarchical 8→1 binder、
  bounded multiplicative Core gate、参数预算及可证伪判据。
- [x] 原位切换唯一canonical源码/config到不兼容v8 schema；没有v7并行
  executable或checkpoint兼容分支。
- [x] 聚焦CPU合同通过：Writer总参数`10,706,176`、binder`590,848`、
  compiler`1,469,696`；`D=0→event=0`、`Procedure=0→LoRA identity`、
  Action/effect梯度和Core有效调制均成立。
- [x] 全仓`192 passed`、Markdown link audit零缺失、`git diff --check`通过；
  architecture guard无hard violation、无parallel version/function family。
- [ ] clean commit/push。
- [x] live核验GPU4–7和存储后，B20完成最长105-frame真实视频连续3个
  task-complete macros；全finite且稳态约205.97 macros/hour，不触发B16。
- [x] B20完成fresh0→1、exact-resume1→3；step1未改写，任务/视频/query/LR/
  cursor相同，binder/EventRead/Core gate和所有主模块梯度可达。
- [x] clean commit/push后保持task-complete、fast decay400，从identity fresh
  完成macro0→400；八点correct400的observed-best为macro300=`125/400`。
- [x] macro300五臂为`125/121/110/110/117`；内部检查显示Action变化仅贡献
  约`8–10%` event差异，Effect变化贡献约`147–300%`，EventRead近均匀。
  strict local binding缺少teacher-action身份，v8停止。

## Phase C4：v10 Evidence-Preserving Dual-Stream Writer

- [x] 封存
  [`docs/action_forecast_writer_v10_design.md`](docs/action_forecast_writer_v10_design.md)：
  恢复text-only task axis；保留独立Action hypothesis与Visual-Effect streams；
  interleaved causal Procedure；Procedure提供content并门控full-rank Core。
- [x] 原位替换唯一canonical源码/config到不兼容v10 schema；删除尚未封存的
  v9草案和v8 executable config，不保留strict binding/EventRead并行路径。
- [x] 真实参数枚举`11,627,520`；全仓192 tests通过，覆盖Core置换不变、
  dual-stream shape/mask/order、`D=0→Effect=0`、Action保留、
  `Procedure=0→LoRA identity`、完整rank-16 target与freeze/gradient staging。
- [x] GPU4–7最长105-frame B20连续3个macro finite；后两步约
  `26.38 queries/s`、`197.85 macros/hour`，峰值allocated/reserved约
  `77.01/83.65GB`，B16未触发。
- [x] 完成fresh0→1→exact-resume1→3；step1未改写，cursor/采样/LR完全一致，
  与独立连续run最大mean-loss差`2.63e-6`；正式teacher seed与profile已封存。
- [x] clean commit/push后从identity fresh按task-complete fast-decay400训练
  至macro400，共`9,600`个one-video LoRA conditions、`192,000`个action
  queries，约`7,832.8s`；每25保存。
- [x] 对12个single checkpoints完成paired fixed correct400；曲线
  `95/103/84/89/82/90/96/96/89/96/97/91`，observed-best为macro50
  `103/400`，不做checkpoint融合。
- [x] 对macro50完成五臂`103/94/75/67/43`和内部
  Text/Core→Action/Effect→Procedure→slots→effective-LoRA→policy检查。
  same同档且wrong/shuffled/reversed行为门均通过，但absolute低于Source-SFT
  `109`并距硬门150为47；Action主导、Effect近均匀读取和高增益compiler使
  same-task视频方差被放大。v10判为absolute负结果。
- [x] 按owner要求完成v10后暂停：不续训、不改canonical架构、不启动Loom、
  one-shot或RL，等待共同讨论。

## Phase C5：Core-Program负结果与Prior–Innovation重构

- [x] Core-Program fresh macro0→200训练合同完整成立；fixed correct400
  `84/75/60/76`，四点逐task envelope仅`95`，不续训、不做行为级特异性。
- [x] 对macro50完成无rollout内部数值分析：Procedure已有强顺序差异，但到
  effective LoRA/action压缩两个数量级；raw DC主导readout、AC被压弱，
  bilinear形成moving basis，Procedure/Core梯度比约`.36`。
- [x] 从根因撤销strict double-necessity，封存
  [`docs/action_forecast_writer_prior_innovation_design.md`](docs/action_forecast_writer_prior_innovation_design.md)：
  Core提供稳定semantic prior，Core-only query读取time-centered Procedure
  innovation，二者在固定slot坐标直接相加。
- [x] 原位替换唯一canonical源码/config/schema；退役Core-Program活动config，
  不保留兼容resume或并行compiler。
- [x] 精确参数枚举Writer`10,643,968`、compiler`1,403,904`；全仓
  `195 passed`、compileall、diff check通过，architecture guard无hard
  violation。
- [x] canonical实现clean commit `7b7abf1`并push。
- [x] 只在GPU4–7完成最长105-frame B20三macro profile、全参数
  reachability与formal-seed exact-resume；不继承旧证据。
- [x] seal后fresh macro0→200、every25；固定评测50/100/150/200
  correct400为`100/61/89/88`，不融合checkpoint。
- [x] 未恢复同期旧架构，未续第二小时、未做行为级视频特异性；跨架构内部分析
  将最稳定瓶颈定位为B列、rank和跨层effective update塌缩。

## Phase C6：Target-Spectral Writer

- [x] 封存
  [`docs/action_forecast_writer_target_spectral_design.md`](docs/action_forecast_writer_target_spectral_design.md)：
  保留v6 Core/Procedure上游；把320个rank伪语义slots改为38个真实policy
  targets，target-first融合、rank-last展开，并固定A/U spectral gauge。
- [x] 唯一canonical源码/config/schema原位切换；Prior executable config退役，
  不兼容旧Writer checkpoint。
- [x] 精确参数`14,495,744`；step0 effective identity、38-target拓扑、
  target/rank坐标、FP32 Procedure centering、强共同方向QR稳定性和真实三步
  gradient staging均有CPU合同。
- [x] 全仓`196 passed`、compileall、JSON、diff和architecture guard复核后
  clean commit/push。
- [x] 只在GPU4–7完成最长105-frame B20三macro profile；三步finite，稳态约
  `25.488 queries/s`、`191.159 macro/hour`，峰值约`77.07/83.65GB`，
  B16未触发。
- [x] 正式teacher seed下fresh0→1→exact-resume1→3；steps/LR/query/video
  cursor连续、全部finite、validation/test reads为0，step1七个文件逐项SHA
  未改写。
- [x] fresh macro0→200、every25自然完成；固定评测50/100/150/200
  correct400为`30/12/18/34`。四点完整审计通过，best低于source base、SFT、
  v5.2和v6；未续训、未做行为级控制。
- [x] macro200完成无rollout rank/layer/video与五条件内部分析。强制spectral
  gauge把stable rank从约1提高到3.32，却把LoRA范数缩小3.66倍、打散跨层方向、
  翻转q/v能量并造成极端layer不均；Core/Procedure和order传递保持工作。
- [ ] 基于负结果重新设计：保留v6高增益、q-dominant、跨层协调公共主方向，
  把额外rank作为可选zero-init视频innovation；不得在Target-Spectral的
  orthogonal scale/gate上打补丁或resume。

## Phase C7：v5.2 × task-complete fast-decay因果格

- [x] owner授权补齐此前缺失的`v5.2 + 新训练`单元；不得把v6的143直接归因
  于模型拓扑。
- [x] 在成熟long-first task-complete训练框架中原位恢复正式结果对应的v5.2
  拓扑；参数`10,237,704`、step0 identity、信息墙和public rank16不变。
- [x] fresh config固定B20、4 ranks×6 tasks、full24等权、LR
  `3e-4`、warmup17、cosine decay400到`1e-5`、every25。
- [x] GPU4–7完成最长105-frame三macro profile和formal-seed
  fresh0→1→resume1→3；B20 finite、所有主模块可达，配置已seal。
- [ ] clean push后fresh macro0→200，再默认exact-resume到400。
- [ ] 并行评测macro150/200/350/400 correct400；winner若在内部点，只补±25。
- [ ] 对winner与旧v5.2/direct Source-SFT做有效BA谱、范数、q/v、layer/target、
  视频中心化变化和policy action对照；达到absolute门后才补行为控制臂。

## Phase E：matched π0.5 action one-shot baseline

- [ ] 在看 outcome 前，每个 validation task 用固定 seed 从 50 episodes 中抽
  1 条 action episode。
- [ ] 对每 task 只训练一次 one-shot LoRA，不做 50 次 one-shot。
- [ ] EMBER 比较臂使用与该 episode 对应的 action-hidden video；保持 task、
  state、env/policy RNG 和评测预算 paired。
- [ ] 比较 absolute performance、训练/适配 wall、GPU-hours、action supervision、
  trainable parameters 和 deployment-time forward 成本。
- [ ] EMBER 只看 video 且一次 Writer forward，因此不把“必须绝对超过 action
  one-shot”设成唯一成立条件；若能超过则作为更强结果。

## Phase F：RL-Writer development

- [ ] 从 v6 架构规定初态做独立、短且 task-balanced AS cold start；不得从完整
  AS observed-best 继续。
- [ ] 直到 24 个 development-train tasks 各在 official random-reset rollout
  中至少成功一次，才关闭 action 入口并进入 pure-reward。
- [ ] reward 阶段只用官方 binary reward/success；不读 object pose、
  privileged shaping、validation/test reward 或 `.pruned_init`。
- [ ] 保存 Writer/optimizer/scheduler、worker RNG、env/policy seed schedule、
  interaction cursor、video schedule、完整 reward ledger 与 exact-resume state。
- [ ] 用 correct/wrong-video、source/seen 和 absolute validation 选择；
  RL 不能用来掩盖 AS 的绝对性能或逻辑漏洞。

## Phase G：32-source final 与 zero-interaction test

当前 focused v6/SFT/one-shot/RL 完成并向 owner 汇报后才进入；不得自动启动。

- [ ] 将 8 validation tasks 机械合入形成 32 source / 8 test。
- [ ] AS-Writer、Source-SFT、RL-Writer（若成立）各自从规定初态单 seed 重训。
- [ ] 打开 test 前完成 final seen comparison。
- [ ] zero-interaction test 比较 source base、Source-SFT、AS-Writer、RL-Writer
  及 correct/wrong-video；每 rollout 随机抽正确 task 的一条 teacher video。

## Phase H：test-only task-local RL 与 oracle

- [ ] test 打开后，identity/AS/RL Writer 三臂在每个 test task 上使用相同
  official random-reset sequence、同一 cohort video 和可比预算训练到各自最佳。
- [ ] fixed 50 `.pruned_init` states 只作训练分离的 fresh evaluation。
- [ ] 三臂结果封存后，才读取 8 test tasks × 50 action episodes，联合训练一套
  shared multi-task target-action LoRA oracle；不是 8 套 task-local LoRA。

## 每次 GPU 运行前

- [ ] 只读核验 workspace/branch/HEAD/origin/status、现有进程和输出根。
- [ ] 实时比较`gpu01`与`gpu02`，只用空闲卡且合计最多6张；记录owner/进程/显存/
  利用率，不reset、kill、pause或干扰他人。
- [ ] 检查目标`/data1`个人quota、项目占用、峰值新增量和共享filesystem余量。
- [ ] 封存 exact command、config/model/data paths、output root、process topology、
  checkpoint cadence、停止与继续判据。
- [ ] 正式昂贵 run 前做 live GPU preflight；不杀、暂停、reset 或干扰他人进程。
- [ ] output 不覆盖；resume 核验完整 state。stage stop 只可在 sealed total axis
  内单调延长，其它 scientific contract 变化必须 fresh。
- [ ] 评测按 `episodes × horizon` 动态调度；所有 worker 先处理 long task，
  long 耗尽后再取其它 task；任何 checkpoint/GPU 分配都遵守。
- [x] 等训练/rollout 时推进不污染运行的代码、分析和仓库清理；已退役旧路径
  18,853 行、约 3.8 MiB 仓库缓存和 87.49 GB 已完成评测 LoRA 中间 cache；
  活动运行环境、checkpoint、rollout rows/results 和 contract 证据完整保留。

## 长期继续/停止判据

- absolute 低于可信满意区间或尚未形成充分峰后下降时，继续训练、诊断或 fresh
  架构实验；不能因单点略涨结束。
- correct 提升若依赖 wrong video、shuffle/reverse、validation 泄漏或其它违反
  EMBER 映射的捷径，一律判为机制失败。
- focused AS-Writer的absolute硬门统一为
  `correct400 >= max(150, corrected Source-SFT observed-best + 30)`；两个条件
  必须同时满足。还必须same≈correct、correct显著优于wrong/shuffled/reversed、
  多tasks共同贡献并通过独立RNG/video permutation复测。
- `122/400`旧八卡Source-SFT只是背景，不是独立硬门；`+30`是最低研究里程碑，
  不是达到后强制停止。新corrected Source-SFT必须重新训练和选峰后再比较。
