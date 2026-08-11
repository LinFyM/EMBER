# EMBER Execution Brief

## 0. Current operation

PICK只因full48 condition=`483.61515>200`退役；PICK-GC随后把condition降到`152.61`并通过
Program→LoRA→action、world4吞吐和zero-memory部署门，但formal fresh`0→10`后的single-checkpoint strict
paired correct只有`138/400`、breadth6。相对immutable macro0 retained/gained/lost=`118/20/16`、churn36，
未过`correct>=144`与`lost<=8`门，因此PICK-GC+blind offline source-action credit也已退役，不resume、不补
controls、不sweep。OSG-PC随后在唯一world6 profile中因rank-local长尾触发600s NCCL watchdog，wall lower bound
至少是matched baseline的`1.912x>1.25x`，未到full48 report。当前唯一active design是SKNC：只用K4 4/4
binary success key在最终shared Program solve施加nullspace equality，不保存trajectory或求reward VJP。首个
world3 root因TF32 diagnostic measurement单项non-pass并保留；clean `f4fdac7` reprofile已`16/16`通过，
11个4/4 anchors、rank=`48→37`、condition=`29.65`、projected energy=`.778`、Program ratio=`8.95e-8`、
step=`478.627s`、scaled wall ratio=`.47173`。B8/16/32 deployment profile均稳定并选择B32=`.47166 LoRA/s`，
无OOM/nonfinite或hidden teacher read。config已`active_formal_ready`；SKNC尚无训练或paired成绩，下一合法动作是
fresh`0→5`并立即strict paired400。

长期成功条件是同一single checkpoint strict paired correct严格`>150/400`，并具备breadth、低换手、
same-task鲁棒和correct优于wrong/shuffled/reversed/no-video。历史最好仍为v6-fast`143/400`。

## 1. Active SKNC design gate

OSG-PC及任何后继在实现前必须回答：

1. 它改变的唯一主要变量是什么？
2. 它针对`docs/research_history.md`中的哪个最早失效接口？
3. 与最接近历史架构相比，保留了哪些已验证优势？
4. 什么内部证据能快速判定机制是否接通？
5. 何时做真实paired400，什么结果立即停止？
6. 如何避免按held task outcome反向选择target/rank/route？
7. 预计GPU、wall、显存、存储峰值和可恢复状态是什么？

最新接口裁决是：PICK-GC key、condition-local FP32 Program和native rank16 compiler已接通；最早科学失败仍在
blind train24 offline cotangent→held on-policy useful support/coexistence。OSG-PC试图用成功train24 executed-prefix
half-space保护support，但current full-replay per-success VJP graph先在吞吐/长尾接口失败，尚未产生shared guard
transfer证据。后继必须保留信息墙与已通过接口，同时正面限制on-policy credit的cost/occupancy length。
SKNC以每task first all-success key和本macro current all-success keys约束最终shared write，测试“完整conditioned
LoRA零运动”能否保护support，同时把on-policy cost限制为outcome-only K4。完整单变量公式、owner替换和hard
gates见`docs/action_forecast_writer_success_key_nullspace_consolidation_design.md`。

## 2. Fixed information contract

- 输入：exact task language + exactly one action-hidden teacher video。
- video是唯一dynamic value；language不能形成LoRA bypass。
- 禁止teacher action/proprio/reward/terminal、task ID、filename、object pose和hidden normalization。
- 输出：一套完整38-target public rank-16 LoRA；Writer在rollout前运行一次后释放。
- source policy、normalization、split、frame stride5、LIBERO preprocessing和policy interface固定。
- 每rollout随机无放回取正确task的一条teacher video；不挑最好video。
- 不做video/LoRA/checkpoint平均、融合或第二套LoRA。

few-shot若被新设计采用，必须显式改变本节合同：固定`k`或定义动态集合语义，仍action-hidden，不挑video，
同时保留matched one-shot arm和计算成本报告。

## 3. Data and assets

- split：`configs/libero_24_8_8_v1/`，24 train / 8 validation / 8 test。
- source corpus：`configs/pi05_source_corpus_v1/`，过滤后71 tasks。
- target manifest：`configs/pi05_target_data_v1/`。
- source policy：
  `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`。
- task experts：
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`中的统一step2000。
- tokenizer、model、dataset与simulation asset由CLI/`.env.local`指向canonical BCI roots。

不复制大型资产；优先复用canonical path/symlink/manifest。历史A100绝对路径只作provenance。

## 4. GPU selection and throughput

每次launch都重新：

1. 同时检查`gpu01`、`gpu02`的GPU index/UUID、memory、utilization、health和compute process；
2. 选择一个节点；
3. 使用该节点至多6张健康、低利用率、显存余量足够且能提高吞吐的A40；非零显存或低利用率进程不自动排除；
4. 在进程spawn前再次核对；
5. 结束后确认本次进程退出并释放设备。

上限6卡，但不要求6卡。不等待凑满、不dummy occupancy、不跨节点拼碎片，不抢占或明显干扰他人进程。单卡
mechanism/profile按其科学目的使用单卡；strict evaluator按selected node至多6张有益卡动态扩展，不使用NCCL。
多卡训练必须：

- `NCCL_P2P_DISABLE=1`
- physical GPU到local rank显式映射
- 每rank绑定GPU-local NUMA CPU/memory
- 在大资产加载/CPU准备完成后deferred NCCL init
- exact-resume保持原world size/topology

吞吐优化规则：

- 从能充分利用显存的batch开始profile，再向上扩到吞吐平台、allocator抖动或OOM边界；
- 选择真实longest-video panel上samples/s最高的stable点，不按空闲显存或理论batch选择；
- 原生BF16/TF32、SDPA、batched env、persistent workers、prefetch和集中D2H默认开启；
- 合并重复host sync、per-row forward、重复token/video处理和小tensor传输；
- 不为底层微差固定batch1、重复single forward、扩dtype或逐tensor扫描。

## 5. Storage preflight

expensive profile/training/evaluation/cache前必须：

- 在`strg01`查询目标filesystem的`ymdai`独立quota；
- 测量canonical root当前使用；
- 估计peak新增，包括checkpoints、cache、shards、logs、temporary和partial resume；
- 确认峰值不超过quota；`df -h`不能替代user quota。

`/data0`与`/data1`预算分开。正式outputs默认放`/data1/user/ymdai/projects/EMBER/runs/outputs`并通过CLI显式
指定。profile/smoke checkpoint在完成裁决且无consumer后不长期保留；formal checkpoints和paired raw rows保留。

## 6. Formal launch contract

expensive retained run前登记一个简洁合同：

- clean pushed source commit与detached frozen worktree；
- exact config/command/env；
- source model/tokenizer/data/split/normalization/video assets；
- output root必须fresh或由exact-resume schema唯一允许；
- node、physical GPU、world size、NUMA/rank topology；
- batch、precision、optimizer/stateful estimator、sampler和RNG；
- peak storage/wall估计；
- 科学问题、hard gate和下一评测点。

同一run未改变合同的resume复用原记录；command、scale、inputs、devices、overwrite、cost或scientific contract
变化时更新。不得在dirty checkout启动formal，不得让两个writer写同一root。

smoke只证明load、shape、freeze、gradient、OOM、resume和env；profile只选吞吐；mechanism只定位接口。三者都
不能解释closed-loop或冒充formal。

## 7. Training semantics

任何future AS-like训练默认：

- 24 train tasks task-complete等权；
- 每task一条video生成一套LoRA；
- B20 logical same-task cross-episode action queries，先task内mean再task等权；
- video/action episode独立，不制造逐帧低层对应；
- frozen source policy没有trainable参数；
- validation/test actions或reward不产生梯度；
- checkpoint含完整model/stateful update、cursor/sampler、RNG和topology。

新objective若改变task aggregation、video数量、query语义、policy interaction或optimizer，必须在design中明确，
不能冒充旧合同exact resume。

训练时：

- 及时记录per-task loss/gradient、norm/clip、update方向、task coexistence和wall；
- 不因弱指标改善自动延长；
- 到预注册checkpoint尽快跑paired400；
- absolute提高但lost/churn恶化时按能力换手处理，不写成稳定进步；
- 若趋势和内部路径不足，尽早停止，保留正式non-pass。

## 8. Official closed-loop evaluation

preprocessing固定：render256/model224、两相机180° rotate、state/action 7D、10 flow steps、执行前5 actions后
replan、dummy settling10、成功即终止、horizon 220/280/300/520。

strict controls：

- correct
- same-task-other teacher
- cross-suite-wrong
- shuffled
- reversed
- no-video

每个arm严格配对task/state、env seed、policy RNG、video ordinal和初始化。shuffle/reverse必须对真实输入frames
重排后完整forward。evaluator使用cost-balanced dynamic queue、long-first和persistent model/env；卡数只影响
吞吐，不改变request batch membership或科学输入。

至少报告：

- aggregate、per-suite、per-task和breadth；
- 与closest baseline的retained/gained/lost、net、churn、Jaccard和McNemar；
- gained/lost在哪些tasks/suites集中；
- correct与五个controls的同checkpoint差异；
- representation→compiler→effective BA→fixed-action传递。

80-row只作工程screen。正式checkpoint选择必须使用400 rows；不使用checkpoint union、平均、融合或挑task
checkpoint。严格`>150`后仍须补完整controls才能支持视频因果claim。

## 9. Numerical and verification policy

必须验证：

- shape、dtype contract与finite；
- no forbidden read；
- source freeze、no-video/step0 identity；
- request/video/task/state/RNG pairing；
- cache/manifest/completion与resume cursor；
- OOM、nonfinite、stale asset、cross-sample contamination；
- retained code的import/targeted tests和CLI parse。

不验证或不热路径门禁：

- 普通BF16/TF32最后几位；
- batch1相对batchN逐tensorbitwise；
- 大量SHA-256/MD5/content hash；
- 为漂亮数字重复forward或降吞吐；
- 与scientific decision无关的广泛防御性test harness。

## 10. Git, docs and lifecycle

- 一个canonical active implementation；新路线替换旧路线时删除旧executable/runtime/config/tests，历史由Git、
  `docs/research_history.md`和formal artifacts保存。
- 主分支保持clean；并发写用独立worktree，集成后删除临时worktree/branch。
- meaningful状态只更新`AGENTS.md`、handoff、execution brief、research history、task plan和findings，不再向
  数十个历史design和逐日ledger重复追加。
- 删除临时、profile checkpoint或worktree前核验进程、dirty状态、unique commits和consumer。
- 不提交datasets、checkpoints、cache、大binary、credentials或private host信息。

## 11. Current stop boundary

截至本brief，以下全部禁止启动：

- rank14 Gate C/cycle1/controls/fresh training；
- Reward-Credit cycle2或超参扫描；
- RLS、ECP、Tangent、Expert-Flow、old v*/K4/Expert-Manifold的resume；
- pivot15+1、mixed topology、few-shot或其它新候选的profile/training/rollout。

它们只能在新的单变量design authority明确授权后重新获得执行资格。任何旧artifact中的
`formal_ready`只描述历史时点，不构成当前授权。

PICK-GC的formal macro10与strict400已经完成并封存为138/breadth6/lost16 non-pass；其resume、controls、额外
checkpoint与参数sweep全部关闭。OSG-PC profile exit1且无checkpoint/mechanism report；同配置重跑、formal、
deployment和评测全部关闭。SKNC canonical实现、fresh schema、CPU机制验证、clean world3 `16/16` profile与
B32 deployment smoke均已sealed，config为`active_formal_ready`。下一合法动作是从新clean pushed seal head
fresh训练`0→5`并立即strict paired400；launch仍须重新复核gpu01/gpu02、quota、fresh root、物理/NUMA
topology与现有进程，有多少合适的同节点卡就用多少、最多6张，不等待凑卡。

2026-08-11 22:23+08:00资源快照选择`gpu02:0--5`的单节点world6/local4；`:6/:7`属于他人。GPU1只有历史已
纠正ECC/remap且当前无pending/failure。OSG-PC attempt结束后物理0--5均0MiB且volatile/uncorrectable ECC全0；
该空闲快照不是未来设备预约。
