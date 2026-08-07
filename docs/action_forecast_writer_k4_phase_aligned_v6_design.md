# K4 Phase-Aligned Language-Axial Semantic-Procedure Writer

状态：2026-08-07建立的当前唯一活动设计authority。它从functional identity重新训练，不加载任一
历史Writer checkpoint；旧Grounded-Video Expert只保留Git、正式artifact和负裁决文档。

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
