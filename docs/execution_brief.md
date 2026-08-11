# EMBER Execution Brief

## 0. Current operation

PICK只因full48 condition=`483.61515>200`退役。当前唯一active Writer successor PICK-GC已经完成
implementation阶段`345 passed`及seal后完整回归`346 passed`、exact raw full48（condition=`152.45803`）、world6 discarded mechanism
（condition=`152.61008`、retained/null=`24/24`）、B8/16/32吞吐选择和zero-memory deployment vertical；
四suite native LoRA/action均bit-exact，canonical 8-entry cache与8/8 rollout完整。配置现为formal-ready，尚无
formal训练或strict paired400结果，不能把这些机制门写成方法成功。

长期成功条件是同一single checkpoint strict paired correct严格`>150/400`，并具备breadth、低换手、
same-task鲁棒和correct优于wrong/shuffled/reversed/no-video。历史最好仍为v6-fast`143/400`。

## 1. Active design gate

PICK-GC design已经回答下列问题；实现和每次authority更新必须保持这些答案不漂移：

1. 它改变的唯一主要变量是什么？
2. 它针对`docs/research_history.md`中的哪个最早失效接口？
3. 与最接近历史架构相比，保留了哪些已验证优势？
4. 什么内部证据能快速判定机制是否接通？
5. 何时做真实paired400，什么结果立即停止？
6. 如何避免按held task outcome反向选择target/rank/route？
7. 预计GPU、wall、显存、存储峰值和可恢复状态是什么？

未经新的单变量authority，不修改split、信息墙、source policy、normalization、public LoRA topology或official
evaluator。PICK-GC前序CPU/cache/live mechanism/deployment门已经通过；formal训练只能从封存这些证据的
clean pushed commit与新detached frozen worktree启动。

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
3. 使用该节点所有真正空闲、健康且能提高吞吐的A40；
4. 在进程spawn前再次核对；
5. 结束后确认本次进程退出并释放设备。

没有6-card hard cap。不等待凑卡、不dummy occupancy、不跨节点拼碎片、不触碰他人进程。单卡mechanism/profile
按其科学目的使用单卡；strict evaluator按selected node全部有益空卡动态扩展，不使用NCCL。多卡训练必须：

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

它们只能在新session建立明确的单变量design authority后重新获得执行资格。任何旧artifact中的
`formal_ready`只描述历史时点，不构成当前授权。

PICK-GC的CPU、exact raw full48、discarded mechanism、B32 throughput和zero-memory deployment vertical已
通过并封存。当前唯一获准的下一GPU阶段是从新clean pushed seal commit做formal fresh`0→10`，随后立即
strict paired correct400；macro10必须`correct>=144`、breadth`>=6`、lost`<=8`且gained>lost。未看到该结果前
不得resume到25；旧方法、额外candidate、controls或参数sweep仍未授权。每次launch仍须同时复核gpu01/gpu02、
quota、fresh root、world6 topology与他人进程。
