# EMBER Execution Brief

更新时间：2026-08-16。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

最新完成实验是**V6-LPCP Direct Joint Native-Factor Residual**（DJNFR），authority=
`docs/action_forecast_writer_v6_lpcp_direct_joint_native_factor_residual_design.md`。clean frozen `49a4129`从sealed LPCP
fresh完成full24 cycle1：24 tasks/48 paired states/96 rollouts，candidate/reference=`34/33`、gains=`5/4`、9 active
tasks，cycle=`717.940s`；完整world4 checkpoint、completion与0禁读/OOM/nonfinite均保留。

同一cycle1 checkpoint的K4 strict paired correct400=`136/400`、breadth7、per-task=
`1/2/44/35/0/35/18/1`、per-suite=`3/79/35/19`。相对LPCP143严格=`120 retained / 16 gained / 23 lost /
241 both-fail`、churn39、net`-7`、Jaccard`.754717`；suite net=`-2/-4/-3/+2`。correct≥142与lost≤15两门失败，
按authority终局：不resume cycle2，不补same/wrong/shuffled/reversed/no-video，不做参数小扫。

本轮真正推进了因果定位。post-train task4与validation8 four-view BA cosine/energy=`.802835/.800444`和
`.790242/.785834`，held8 8/8过门；raw factor/action cosine=`.677321/.686125`，held/train BA L2=`1.08903x`。
all400相对LPCP effective-BA relative-L2 mean/median=`7.773e-5/5.080e-5`且q/v/action均非零。因此SJNV的
coherent hidden到frozen W2/native public LoRA断裂已被关闭，carrier、视频顺序、direct写出与跨视频共同方向工作。

但写入内容与reward不对齐：gained/lost/retained-failure BA relative-L2 mean=`4.522e-5/7.248e-5/8.987e-5`，
持续失败样本最大。active-task gradient pairwise cosine mean仅`.002470`。所以最早失败接口是**selected-success
only credit怎样在一个shared direct-factor update中形成held on-policy reward-useful方向**，而不是carrier、LoRA
幅度、量化、W2或跨视频coherence。

PAFS-NV随后因validation8仅`.168111/.372863`、3/8过门在formal前终局。SJNV-Gate进一步把固定地址换成共享
joint gate；clean `913d3d3` task4达到`.472272/.597814`，但validation8只有`.201903/.396448`、2/8过门，action
cosine=`.042986`，同样没有full24/strict。关键新证据是stage localization：joint gate/continuous hidden cosine
约`.94`，冻结W2后的raw factor delta却只有`.021353/.265925`，action factor cosine=`.002672`。因此最早断点已
锁定为**coherent video-language hidden -> frozen W2 -> native public A/B**，不是carrier、时序或joint input。

最新终局successor是**V6-LPCP Direct-Factor Paired Common-State Preference**（DF-PCSP），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_paired_common_state_preference_design.md`。它从sealed LPCP fresh并
完整保留DJNFR生成图，唯一把成功整轨迹正蒸馏改为candidate/reference discordant pair在分叉前同一初始观测处的
winner-vs-loser首段flow preference。两臂共享flow time/noise，只比较共同执行长度；四个disjoint correct K4
views等权。它与历史OPPP的不同是状态严格配对且不比较分叉后的occupancy。memory token、rank8与dynamic K仍
开放，但本轮没有同时改变。首次clean `de6c812` task4 smoke在credit前确认重复相同seed/reset
不能保证两臂首观测逐元素相同；`07b764b`又确认flattened state恢复后，hard reset仍改变未被state覆盖的model pose，
差异只出现在两相机，language/state tokens完全相同。两次都只是工程失败、没有科学结果。canonical现每lane只做
一次hard reset+settling，每臂以deterministic soft reset清空controller/observables后恢复相同qpos/qvel；不增加
rollout或forward。exact task4/task7均变为tie；task9/15/18分别有1/2/1个discordant pairs，margin均真实下降且
八head/q-v-action全通，但task9 held/train BA幅度仅`.105x`，task18 train跨video仅`.290/.428`，三个有效anchors
只有task15全门通过。按门终局，不full24、strict或cycle2。当前没有active successor；下一步必须针对“final
success被错误归因给第一prefix而形成task-dependent update”建立新的单变量authority。
稳定约145资格高于单点分数：至少需要相邻single checkpoints低churn/high-overlap、same-task-other鲁棒，并在
同一final checkpoint上证明correct相对wrong/shuffled/reversed/no-video的明确paired优势。

DJNFR训练root=
`runs/outputs/pi05_v6_lpcp_direct_joint_native_factor_residual_formal_cycle0to1_r4_k4_views4_nmc4_b8_49a4129_gpu02_20260815`；
strict与终局分析root=
`runs/outputs/pi05_v6_lpcp_direct_joint_native_factor_residual_cycle1_k4_correct400_noreplacement_seed7_trainr4_evalr6_49a4129_gpu01_20260816`。

## 2. Completed CV-CSD variable and exact conclusion

- 部署继续是exact language + dynamic K ordered action-hidden videos一次生成一套38-target rank16 LoRA；
- anchor K4只产生AS139/LPCP paired唯一成功trajectory；同一trajectory在4个disjoint same-task correct K4
  conditions下分别通过完整Writer→LoRA→policy CFM；
- 四view只在65,536参数`query_delta.weight`梯度处等权汇合，不平均video、Program或LoRA；
- 只训练query commitment，LPCP carrier、AS139 tail、source policy、optimizer与rollout数不变；
- 结果只否定“query-only map + four-view exact selected-success mean”，不否定few-shot、memory、reward或生成LoRA。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138、ADSP138、V6-LPCP、PCSD、CV-CSD、SFMC、Gradient-Open与CCT均已终局且
不得resume或小扫。
各自训练root、strict root、逐episode transitions和必要BA/跨video evidence均已封存。`>150`仍是更高性能追求；约145
也可成为有效结果，但必须由相邻single checkpoints低churn、same-task-other鲁棒和correct相对wrong/shuffled/
reversed/no-video的明确优势共同认证。单点145或151都不算完成。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

SFMC与CCT均因cycle1预注册门失败而没有相邻checkpoint与controls；这不是缺失分析，而是预注册停止规则。下一架构仍须先以
cycle1 absolute/breadth/retention筛选；过门后必须评相邻checkpoint，再做same/wrong/shuffled/reversed/no-video。
单点145或151都不能跳过稳定性与视频因果资格。

## 4. Continuous adjudication loop

每轮strict结果完成后，按以下顺序分析：

1. absolute、per-task/per-suite、breadth；
2. 相对最接近方法和历史强基线的retained/gained/lost与能力集中；
3. 若有相邻checkpoint，分析persistent/gained/lost与union gap；
4. 沿`input evidence -> Core/Procedure -> set/compiler -> effective BA -> fixed action -> rollout`定位最早失效接口；
5. 分离科学non-pass与明确工程合同违约；
6. 只对最早接口提出一个主要因果变量，写可证伪authority后实现；
7. 做最小必要CPU/机制验证和吞吐profile，尽快回到真实paired400。

不得用loss、cosine、rank或漂亮五臂margin代替absolute，也不得通过rank/scale/seed/dtype/temperature小扫救一个
失败checkpoint。owner的局部建议不能导致整套已认可方案无证据重写。

## 5. GPU and efficiency

- launch前同时live检查gpu01/gpu02；单节点至多6张，有多少真正合适就用多少；
- 允许在显存峰值余量充足、低util且不明显干扰他人的卡上共驻；不抢占、kill、reset或dummy占卡；
- evaluator当前保守共驻门为util≤10%、已用≤8GiB且剩余≥32GiB；任一越界即拒绝；
- 多卡训练设置`NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL；
- fresh可用world1--6；exact resume锁原world topology；
- evaluator用动态cost queue和persistent model/env，不静态拆task；
- 以真实samples/s、LoRA/s、最长视频稳定性和显存峰值选择batch；
- 接受正常BF16/TF32低位差异，不重复forward、固定batch1、扩dtype、逐tensor scan或增加hash。

## 6. Storage, Git and artifacts

- 大run前查询`strg01`上的独立user quota，测canonical root并估计checkpoint/cache/temp峰值；
- formal训练与评测来自clean pushed commit的detached worktree；
- 新run使用fresh output root，不覆盖或部分复用中止/不兼容root；
- formal保留run contract、metrics、macro checkpoints、completion、400 raw rows、aggregate和decision analysis；
- profile/smoke roots只作机制/吞吐证据，不冒充formal；
- meaningful结果更新current state、execution brief、current design、task plan、findings和research history；
- 不把历史命令重新复制进多个文档，精确命令以run contract/invocations为准。

## 7. Collaboration boundary

owner授权在核心目标、信息墙与效率原则内持续自主迭代。当前暂不使用subagents。只有真正需要改变核心目标、
扩大权限、处理破坏性操作或遇到无法从本地证据解决的关键歧义时才向owner停下来请求决定。
