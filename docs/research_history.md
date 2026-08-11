# EMBER Research History

本文是历史实验的唯一精炼索引。它记录“测了什么、严格结果是什么、实际证明了什么、不能重复什么”；
不是活动设计，也不提供可直接启动的命令。删除的旧设计与逐日运行日志完整保存在 Git commit
`3a6f801d08facb3e855ab24f84e0b53cb8802e88`及其祖先，正式结果保存在`runs/outputs/`。

当前真相只取`AGENTS.md`、`docs/active_session_handoff.md`和`docs/execution_brief.md`。截至
2026-08-11，历史最好single checkpoint仍是v6-fast的`143/400`，长期严格`>150/400`目标未完成；PICK-GC
formal=`138/400`、breadth6并已退役。当前唯一active design是尚未实现的OSG-PC，只检验成功on-policy
occupancy可行锥能否保护blind proposal的已有support。

## 1. Stable problem definition

EMBER研究：给定exact task language与一条action-hidden teacher video，shared Writer一次生成一套完整
rank-16 task LoRA；该LoRA挂到同一个frozen π0.5-LIBERO source policy后，应在未见初始化上闭环完成任务。
视频是唯一dynamic value，不允许language-only LoRA bypass、teacher action/reward/proprio读取、task ID、
文件名、object pose、multi-video/LoRA/checkpoint平均或held oracle。

最终方法必须由同一single checkpoint的strict paired closed-loop裁决，并同时满足高absolute、task breadth、
低checkpoint换手、same-task鲁棒和correct优于wrong/shuffled/reversed/no-video。训练loss、functional loss、
LoRA能量/秩/cosine、重建误差和漂亮内部margin都只能作机制证据。

## 2. Fixed baselines

| 基线 | 严格证据 | 解释边界 |
| --- | ---: | --- |
| generic π0.5 | `0/400` | 原始policy缺少LIBERO embodiment能力，不是EMBER方法失败 |
| frozen source base | `48/400` | 过滤LIBERO-90训练后的共享起点；不读目标视频 |
| mixed-task Source-SFT rank128 | `109/400` | privileged target-action shared LoRA参照，不是同信息墙baseline |
| v5.2 old | `132/138/74/82/83` | 最强correct-vs-negative视频特异性形态之一，absolute不足 |
| v5.2 task-complete | `120/109/107/111/124` | recipe改变Procedure传递，absolute与margin都退化 |
| v6 old | `121/122/111/84/47` | 强时序差异可传到闭环，但absolute低且task旋转 |
| v6-fast task-complete | `143/135/125/128/129` | 历史最佳eligible single checkpoint；仍未达到151且视频margin弱 |

五臂顺序统一为`correct/same-task-other/cross-suite-wrong/shuffled/reversed`。v5.2与v6的recipe交叉结果
证明架构和训练配方强耦合：不能把“task-complete”或“old recipe”当成普遍好坏结论。

## 3. Cumulative intervention ledger

| 方法/干预 | 最强strict证据 | 实际证明 | 失败接口与保留结论 |
| --- | ---: | --- | --- |
| Action-Forecast v4 | best`109`; five-arm`109/104/99/148/126` | 视频顺序可显著改变LoRA/action | 学到absolute-time/action-phase shortcut，shuffle反而更好；时序敏感不等于正确理解 |
| Semantic Core + Procedure v5 | best`115`; `115/108/74/113/114` | task语义与有序Procedure可分离，wrong-video margin可形成 | Procedure差异在fusion/compiler后衰减，correct与shuffle/reverse等价 |
| v5.1/v5.2 | best`132` | language-axis Core、task-token evidence与Procedure能提高absolute或视频margin | 配方/架构耦合，不能只续训或按一个margin选方法 |
| v6-fast | best`143` | 高增益decoder和task-complete能达到历史最高absolute | 后续450/500/550/600=`131/130/132/126`，训练不是稳定累积 |
| v7/v8/v10 | 低于v6 | 更强内部Procedure/时序结构可以落到LoRA | 更漂亮内部因果不保证closed-loop，可能放大same-task demo nuisance |
| Loom/Recenter/Core/Prior | 均未过143 | patch correspondence、去DC、静/动态分解等接口可独立实现 | 均未解决policy-effective conditional credit；不再以confidence或单一fusion补丁重跑 |
| Target-Spectral | best`34` | 强制谱/秩形态会明显改变LoRA | “更均匀、更高秩”不是健康的同义词，可破坏q-dominant policy manifold |
| CV-ADR RAW/GROUP4 | best`117/110` | 更大、更coherent更新可构造 | video梯度主效应约`.1%`，query/flow variation和credit错位主导；大更新不等于好闭环 |
| Target-Bound | best`120` | remove-A/D与memory reversal 8/8达门，动态路径确实工作 | shared factor coexistence与checkpoint漂移仍失败；不能再把首因写成“视频未使用” |
| Semantic Factor-Basis | best`127`, union`193`, gap`66` | 一度形成更多共同能力 | 单checkpoint远低于union，核心现象是能力换手而非积累 |
| Variance-reduced estimator | best`126` | exact-Beta/antithetic可略稳gradient | held functional loss变好但closed-loop更差；flow MC方差不是主因 |
| Semantic Direction Store | best`129` | 独立store改善早期acquisition | 同分checkpoint breadth不同，Program→factor压缩与漂移仍在 |
| Policy-Target-Owned Factor | best`99` | 解除38-target共享改善跨层异质性 | action效果和absolute差；target ownership/健康几何不是充分条件 |
| Policy-Lane Hyperdecoder | best`70` | 形成约10条有效lanes和SFT量级专门化 | video BA能量约`.02%`，容量健康不能替代动态credit |
| Policy-Wide Atom Dictionary | best`80` | 64 atoms被广泛使用 | mixing/effective LoRA仍近rank1；不靠增atom/rank/正交loss救活 |
| Factorized Condition-Kernel | best`49` | full-rank stable kernel与跨video差异可形成 | LoRA约比direct SFT小200×、近identity；低增益decoder是局部瓶颈 |
| Few-Shot Invariant K4 | best`108` | K4置换、same/LOO/wrong/order路径可工作，能削弱单video偶然性 | full24 gradient retention约`.043`；few-shot不自动解决共享credit或正确时序 |
| K4 Policy-Layer Trace | best`99` | all-layer trace产生correct>wrong | 逐频单位化放大低能DCT高频约`140×`，reversal仍高 |
| Energy-Preserving Trace | best`85` | 修复真实频率能量比例 | correct/wrong从`99/57`缩到`85/80`；能量保真不等于语义保真 |
| Evidence-Factorized Trace | best`84` | trace→BA→action闭合且correct>wrong | shared Reader retention约`.05`，参数隔离仍非答案 |
| Sparse Semantic Expert | best`78` | expert-local retention提高 | language route固定owner，wrong/order更成功；language-only ownership不足 |
| Grounded-Video Expert | best`88` | video route、Reader、BA、action与rank均material | correct无margin且task轮换；视频敏感+隔离仍不充分 |
| K4 Phase-Aligned v6 | best`108`, reversed`121` | 视频未被忽略 | 近rank1、高能量、program retention约`.04`；phase alignment不足 |
| AS125 + semantic-progress RL | `97→104→102` | failure trajectory可提供非零action-free credit | breadth下降、继续训练换手；reward信号存在但共享更新不稳 |
| Program-Credit RL | `106` | CRN与Program gradient可到达 | task cotangent近正交却被shared condition map压成common update |
| SFT-Anchored Tangent Basis | `143→142` | 强warm-start上小幅reward update可运行 | gained/lost=`20/21`，没有净提升；保持分数不能冒充Writer改进 |
| task experts step2000 | train`658/1200`, 23/24非零 | task-local SFT LoRA是policy-effective task-level target | task9仍0，且不提供held泛化、视频特异性或时序证据 |
| addressless Expert-Manifold | `48/400` | raw-expert reconstruction可训到SFT量级norm | decoder后topology identity坍缩，nearest expert cosine约`.008` |
| topology-address binding | `75/400` | 静态chunk/rank地址可调制video dynamic value并进闭环 | 输出仍task-common、absolute低；只调address不够 |
| Causal Barycentric | `63/400` | temporal coefficients和raw-factor组合可运行 | raw A/B组合有`k≠j` cross terms，不保持effective update |
| policy-effective soft/hard bank | `15/80` / `3/80` | hard compiler可近精确复现所选expert | 当前24-expert deployment dictionary无held support；不外推所有流形方法 |
| v6-prior whole-LoRA objective | `134→127→105→123` | 冻结上游、只训写出端可高吞吐运行 | norm/方向吸引主要径向收缩，macro0仍最佳；objective退役 |
| Expert-Component Projection | `134→133→120` | `a_correct`与expert component按构造提高 | 正交漂移增大，macro25 net`-14`, p=`.038477`；不续/不扫权重 |
| Condition-Local Tangent Tube | `134→131` | relative tube中位`.01390/.01408`，半径约束工作 | direction ratio`108.93/126.88`、completion`0/24`，只压小未旋正 |
| Expert-Flow Teacher Audit | no rollout | gradient residual`.6864/.8387`，expert方向非冗余 | flow loss仅`2/24` tasks、`0/4` suites优于baseline；CEFD否决 |
| Frozen-v6 residual v1 | no rollout | correct retention`.807966`与A/B/action closure成立 | DC key condition`1315.33`、null15/24；不训练、不调lambda/seed/P |
| Balanced DC-Causal v2 | `134/140/139`, union`153` | 13/13机制门、24/24 null、部署/吞吐闭合 | 10→25=`12/13`换手；50-video correction近随机正交，blind-add退役 |
| Exact Anchored Reconciliation | `134→140` | RLS/历史row保留机制可运行 | full400 lost15，correct80误导；offline row保留不保护held occupancy |
| Reward-Credit Program Cotangent | cycle1`134`, `14/14`换手 | on-policy reward可形成有内容Program与continuous tangent | q/v约`1e-8 RMS`运动低于非零BF16 factor约`1e-4` ULP；不续cycle2 |
| Q/V uniform pivot-rank14 | online`128`; compiler-only`138` | 去混杂后可分离compression与regeneration影响 | old→compiler`119/19/15`，compiler→online`115/13/23`；两者独立换手，统一rank14退役 |
| Policy-Innovation Consensus Key | no rollout | raw same/order、full48 correct/null、Program→LoRA→action与吞吐全部闭合 | exact full48 condition=`483.61515>200`；static common mode导致key collision，未获formal训练资格 |
| Policy-Innovation Goal-Causal Key | `138/400`, breadth6 | full48 condition修到`152.61`，FP32 Program与effective BA切向写出闭合 | macro0→macro10=`118/20/16`、churn36；blind offline source-action credit不覆盖held on-policy support，组合退役 |

## 4. Final rank14 adjudication

immutable old full-rank macro0为`134/400`，per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
`0/5/48/34/0/35/11/1`。

online-regenerated rank14 root得到`128/400`、breadth7、per-task=
`1/1/47/29/0/36/13/1`，old→online retained/gained/lost=`113/15/21`。由于old/new使用18/12个
generator且旧调度在worker内局部拼B8，它是真实端到端non-pass，但不是干净compression反事实。

一次性compiler-only root：

`runs/outputs/pi05_v6_qv_rank_reserved_compiler_only_old134_to_rank14_correct400_20260811`

它从old exact cache做50×B8 q/v transform，action 1600 tensors与400 video identities exact，0 Writer/
teacher read/policy forward/update。strict=`138/400`、breadth7、per-task=`1/1/46/32/0/35/22/1`；
old→compiler retained/gained/lost=`119/19/15`、net`+4`、churn34。预注册lost上限是10，因此hard gate失败。
Long1净`+11`掩盖Spatial/Object净`-3/-4`；aggregate提高不是稳定共同积累。compiler→online又是
`115/13/23`、net`-10`，证明regeneration是第二个独立换手源。

正式状态：`original_gate_b_passed=false`、`counterfactual_gate_passed=false`、
`retroactively_changes_original_gate_b=false`、`authorizes_cycle1=false`。不能恢复Gate C、cycle1、
controls或rank14训练，也不能把该结果外推成“视频/Reward/continuous tangent整体无效”。

## 5. Task experts and few-shot

正式task-expert root：

`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`

24 tasks统一续到step2000；step250/500/1000/1500/2000 direct-expert closed-loop=
`432/557/624/638/658` of 1200，step2000为23/24 tasks非零。它们定义policy-effective task-level parameter
manifold，并提供SFT LoRA能量、rank坐标和跨target参考；但同一task的expert target对所有video恒定，所以不
包含same-task video差异或时间顺序。soft/hard bank held=`15/80`/`3/80`进一步否定直接部署字典。

K4实验说明few-shot能过滤单条示范的部分偶然低层细节、改善内部same/LOO稳定性，但未解决full24 shared
credit retention、正确顺序的policy-effective方向和single-checkpoint漂移。未来few-shot只能作为明确变量，
不能因直觉自动替换one-shot；动态视频数量也不是当前已验证能力。

## 6. Stable cross-experiment cognition

1. **视频被使用不等于被正确使用。** hidden/LoRA/action对视频敏感仍可让wrong/shuffle/reverse更好。
2. **正常顺序有因果含义。** shuffled/reversed真实破坏展示顺序，模型不能靠原时间戳恢复；correct必须沿有效
   policy update胜过negative，而不只是把negative推坏。
3. **LoRA健康度是约束，不是目标。** 低能量、过度rank1可解释局部失败，但SFT量级能量、多lane、高秩和
   正确expert cosine都没有自动带来高closed-loop。
4. **functional surrogate长期错位。** loss、gradient consistency、reconstruction和MC方差改善可与closed-loop
   退化并存；checkpoint选择必须及时跑真实paired400。
5. **task drift有多个来源。** query/flow variation、full24正交抵消、shared parameter coexistence、
   Program→factor压缩、condition-map common update、compression和regeneration都只解释部分换手。
6. **架构与recipe耦合。** 新结果应与最接近历史架构的per-task成功集合比较，不能按aggregate 180度转向。
7. **small panel与union会误导。** correct80曾与full400给出相反保留结论；checkpoint union不能代表single model。
8. **新topology不能用held outcome设计。** target/rank/routing必须由train24机制和policy geometry推导，不能
   因某个validation task得失而手调。
9. **吞吐优先于低位复现。** 原生BF16/TF32、batch shape和reduction order的正常微差不是科学精度；不得为了
   `.001953125`级roundoff固定batch1、重复forward、扩dtype或做逐tensor/内容hash门禁。
10. **负结果只淘汰实际假设。** rank14失败淘汰uniform pivot14 support合同，不淘汰所有rank reservation；
    expert bank失败淘汰当前reader+24-expert字典，不淘汰所有task-level manifold监督。

## 7. Do-not-repeat registry

- 不续任何已封存non-pass checkpoint，不扫仅为挽救单点的scale、seed、rank、dtype、dither或ULP参数。
- 不恢复language-only LoRA bypass、multi-video/LoRA/checkpoint平均、validation-task expert routing或held oracle。
- 不用80-row screen、training loss、held functional loss、LoRA norm/rank/cosine或union选择正式checkpoint。
- 不把强制正交、均匀能量、高stable rank、更多atom/lane/expert当成性能目标。
- 不把task expert reconstruction写成视频时序学习，也不把same-task恒定target写成video specificity。
- 不恢复旧SmolVLA、70/10/10 split、`pi05_libero`、flat task-local RL、progress proxy、SPSA或success-only replay。
- 不为普通BF16低位差异降低batch、显存利用或吞吐；不新增SHA-256/MD5或大量防御性校验。
- 不按held task得失选择mixed topology。pivot15+1或mixed `{16+0,14+2}`只可作为未来未授权问题；其中
  rank1 tangent capture历史仅约`.9185`，不能因多保留一列就直接授权Reward cycle1。

## 8. Historical detail retrieval

精确旧设计可通过下式读取，而不需要在active tree保留几十个互相矛盾的文件：

```bash
git show 3a6f801:docs/<historical-design>.md
git show 3a6f801:findings.md
git show 3a6f801:progress.md
```

保留的深证据文档是：

- `docs/action_forecast_writer_v4_root_cause.md`：错误时序shortcut的第一性原理定位；
- `docs/action_forecast_writer_v6_design.md`：历史143强基线的结构；
- `docs/action_forecast_writer_video_expert_manifold_design.md`：压缩后的task-expert/Expert-Manifold/Reward链；
- `docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`：最终rank14设计与闭环否决。

正式artifact不得因代码或文档退役而删除。若要追查精确root、per-episode row、manifest或旧命令，优先读
对应`runs/outputs/*/{run_contract,results,analysis,completion,evidence}.json`和上述Git快照。
