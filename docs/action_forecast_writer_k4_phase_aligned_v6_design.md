# K4 Phase-Aligned Language-Axial Semantic-Procedure Writer

状态：2026-08-07已完成identity-fresh正式训练、四点strict rollout、winner五臂和全部
内部分析，并作为负结果封存。本方法不得resume、warm-start或恢复为活动路径；
当时从functional identity重新训练，没有加载任一历史Writer checkpoint。

## 1. 证据链与最早接口

Grounded-Video Expert的完整parameter隔离已经成功把局部gradient retention提高到约`.5`，内部
LoRA stable-rank也提高到`1.463`，但correct曲线只有`76/88/77/82`，winner五臂几乎无margin。
因此parameter sharing、rank collapse、视频完全旁路和单纯训练方差都不再是最早解释。

历史v6-fast仍是目前最强单checkpoint原始架构：correct最高`143`，并具有相对更强的wrong/order
视频差异。它唯一同时实证过三段接口：trainable PI05 language/multimodal/action high-level encoder，
task-grounded visual transition + causal Procedure，以及Core/Procedure到320 LoRA slots的高增益compiler。
此前K4各版都替换了至少其中一段，因而不能用它们的负结果否定“v6高层语义图 + few-shot共同程序”。

本设计只改one-shot歧义的根接口：每条演示的长度与速度不同，不能直接拼接时序，也不能先生成四套
LoRA再平均。四条视频先各自对齐到统一进度轴，然后形成一个共享语义Core和一个共享Procedure，最终
只编译一次LoRA。

## 2. 输入、信息墙与部署图

- 输入严格为exact task language + exactly four action-hidden same-task videos；
- 禁止teacher action、proprio/state、reward、terminal、task ID、filename、object pose、hidden
  normalization和policy outcome；
- K4按既有deterministic complement schedule选取，四条视频无shot identity且集合置换不变；
- 输出仍是一套完整38-target public rank16 LoRA，不挑video、不平均LoRA、不ensemble；
- AS与未来reward训练调用完全相同的`language+K4 videos -> shared program -> one LoRA`图，只有末端
  cotangent来源不同，不包含SFT-only辅助loss或LIBERO heuristic。

## 3. Trainable high-level video semantics

恢复历史v6的Writer-owned PI05 language-axial encoder：

1. text-only branch产生authoritative task-token queries；
2. 每帧通过trainable rank4 VL Meta-LoRA读取multimodal task-span hidden；
3. task queries读取256个image positions形成task-grounded patch evidence；
4. trainable rank4 Action-Expert Meta-LoRA用固定、持久的`t=1` Gaussian suffix产生每帧interaction；
5. 所有source policy参数和normalization继续冻结。

这些量不是低层视觉descriptor；它们直接从冻结π0.5的语言、视觉和Action-Expert内部状态提取可由
Writer训练的高层语义。视频不是route或附加gate，而是Core和Procedure全部动态value的来源。

## 4. K4 phase alignment与组合

每条视频保持实际输入顺序，独立把per-frame semantic evidence、grounded patch evidence和Action-
Expert interaction用可微线性插值重采样到16个normalized progress slots。插值只消除长度/速度差，
不跨视频做transition，也不使用动作或状态对齐。

- Semantic Core把`4×16`个phase-aligned frame evidence作为一个无序集合读取；每条视频恰有16项，
  因而四条演示等权，视频集合置换严格不变；
- visual transition和causal Procedure分别在每条视频的16个slot内运行，绝不跨video boundary；
- 四条Procedure sequence在同一phase slot逐点等权mean，得到一个K4-common ordered Procedure；
- 共享Core与共享Procedure进入历史v6 exact slot-normalized compiler，生成320个module/layer/rank slots，
  再由八个zero-output FactorHeads产生唯一LoRA。

这不是多LoRA融合：模型只在高层语义程序空间组合多次演示，parameter write只发生一次。

## 5. 初始化、训练与漂移假设

FactorHeads末层和compiler AdaLN保持zero initialization，step0精确等于template-A/zero-B source
identity。训练保持K4、logical B20、full24 task-equal raw mean、AdamW、policy microbatch2、每25
checkpoint和task-query-keyed policy randomness；恢复v6-fast的warmup17/decay400。fresh0→200先作
四点裁决；若曲线没有可信的持续下降，可exact-resume同一root到400，而不得从历史best warm-start。

该架构对task漂移的假设不是强制task梯度同向，而是四条同task演示先在共享semantic/procedure
space提取稳定公共程序，减少one-shot偶然视觉对单次full24更新的旋转；同时保留所有task共享的v6
semantic/compiler参数，使跨task可迁移结构可以累积。是否成立只由strict correct curve、breadth、
gained/lost、五臂和内部Core→Procedure→BA→action证据裁决。

## 6. 工程与A40 gate

一个canonical实现原位替换Grounded-Video Expert；删除其`fewshot_m2p.py`活动实现，历史由Git和
artifact保存。fresh checkpoint family严格不兼容旧方法。A40先在live空闲六卡、3+3 NUMA、显式
`NCCL_P2P_DISABLE=1`下完成longest105、K4/B20/B2 fresh0→1与same-root exact-resume1→3。
保持16-frame encoder chunk和activation checkpointing；只有实测OOM才把physical chunk降到8或
policy microbatch降到1，logical B20/K4/full24/phase16不变。

## 7. 预注册裁决

formal从identity fresh0→200，评50/100/150/200 strict paired correct400。single winner再做
correct/same-task-other/cross-suite-wrong/shuffled/reversed和内部8-task refs1分析。functional loss只作
finite证据，不选择checkpoint。最低门仍是同一single checkpoint correct严格`>150/400`，并在过门
后继续提高absolute、breadth、稳定累积和视频因果性。

禁调项：不加scalar/global scale、rank/diversity loss、task-ID route、success filtering、SFT-only
reconstruction、multi-video LoRA平均、checkpoint融合、挑video或held-out outcome调参。

## 8. A40 profile seal

clean`e1d0b62`在live空闲`gpu01:0,1,2|4,5,7`完成fresh0→1及same-root exact-resume1→3。
三步loss=`.150377/.152774/.147865`、step time=`86.20/87.52/87.47s`，峰值
allocated/reserved=`34,968,286,720/47,016,050,688` bytes，0 clip/OOM/nonfinite。step1 Factor
可达，step2 Semantic Frontend/Core/Compiler可达，step3五个参数owner全部可达；累计1,440 queries/
288 videos，source trainable=0且held reads=0。logical K4/B20/B2/full24/phase16没有降低，profile权重
弃用。该证据已seal formal fresh0→200，不提供任何性能结论。

## 9. 正式训练与strict correct曲线

clean/pushed `2356d33`从functional identity完成fresh0→200：200个finite macros、
96,000 logical action queries、19,200 K4 action-hidden videos、8个every25 checkpoints，
0 clip/OOM/nonfinite。wall=`16,228.904s`，peak reserved=`39,187,382,272` bytes；source
trainable=0，validation/test action reads均为0。唯一训练root为
`runs/outputs/pi05_as_writer_k4_phase_aligned_v6_formal_fresh0_200_r6_2356d33_20260807`。

50/100/150/200的strict paired correct400为`88/108/80/99`，breadth>=5=`4/4/3/4`。
相邻gained/lost=`40/20,27/55,47/28`，四点union/intersection=`157/36`，
single envelope gap=`49`。macro100是single winner，但仅`108/400`，且它之后先失去55个
成功state再回获47个；能力仍在checkpoint间大幅轮换。因此不resume到400，也不使用
functional loss选点。

## 10. winner五臂和视频因果性

macro100五臂`correct/same-task-other/cross-suite-wrong/shuffled/reversed`=
`108/115/94/101/121`，全部400 rows与state/env/policy RNG字段配对一致。correct相对
wrong的gained/lost=`28/14`，exact McNemar `p=.04356`，说明视频任务identity没有被
完全忽略；但same高7、reversed高13，shuffled只低7，没有形成正确时序应优于
反转/打乱的closed-loop语义。五臂union/intersection=`162/66`。

## 11. 内部机制与最终负裁决

8-task refs1的正式内部root为
`runs/outputs/pi05_as_writer_k4_phase_aligned_v6_macro0100_internal_refs1_r6_2356d33_20260807`。
wrong从`Core/Procedure/Program/effective-BA/fixed-action`的relative-L2中位依次为
`.3283/.1912/.3181/.3296/.0765`；shuffled/reversed的BA中位为`.1880/.1653`。因此视频
和时序变化能够material改变Program、LoRA和policy action，不能把失败写成“视频被
旁路”。

但correct LoRA norm中位已达`91.12`，mean-target stable rank只有`1.00021`，首奇异值
能量中位`.99979`；K4集合置换对BA仅有`.00141`的数值差，符合集合合同。
最后50步full24 factor/program gradient retention仅`.04634/.04363`，pair cosine约
`.00337/-.00118`，负pair约`.452/.503`。即使恢复v6高层视频图并用K4对齐，
functional action surrogate仍把一个高增益、近单方向的LoRA在24 tasks间来回旋转。

因此本方法的最早剩余接口是：如何让视频中的高层任务/程序信息直接对应
到policy-effective的参数流形，而不是继续改K、phase、scalar、rank loss或延长同一
functional recipe。下一方法必须保持视频为central dynamic value，不允许language-only
LoRA bypass。
