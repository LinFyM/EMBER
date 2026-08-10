# Video-Conditioned Expert-Manifold Topological Writer

状态：本文件是Video-Conditioned Expert-Manifold总路线的当前设计authority。第1--32节记录从
identity-fresh topology Writer到v6 warm-start诊断的历史推导；第33节whole-LoRA、第34节ECP与第35节
Tangent Tube均已正式退役，不能从其中的旧“当前/下一步”恢复训练。第36节matched Expert-Flow Teacher
Audit已正式证明expert flow teacher仅在`2/24` tasks、`0/4` suites过门并否决CEFD；第37节v1又由唯一
macro49 profile定位并淘汰DC-dominated temporal key。第38节Balanced DC--Causal v2随后完成机制、部署、
zero-memory identity、formal0→25与macro10/25 strict400，曲线为`134/140/139`；blind-add没有保留共同能力，
因此已退役。2026-08-10唯一活动设计是第39节Exact Anchored Reconciliation：部署图完全继承v2，只替换
training-time reconciliation与checkpoint sufficient state；canonical实现和CPU合同已完成，尚无RLS GPU、
training或strict结果。下一动作只能是clean pushed/frozen authority上的fresh0→3 A40 profile，通过后才
formal0→10。K4、online expert bank和所有旧Writer只由Git、
正式artifact及其负裁决文档保留。

## 1. 结论先行

本方法不忽略视频，也不用language-only route直接生成LoRA。唯一dynamic value是
frozen π0.5对“exact task language + action-hidden teacher video”的联合高层内部表示相对
matched text/no-image baseline的视频innovation。该innovation保留时序轴，被直接编译为
一套rank-16 public LoRA；把视频值置零必须精确回到source identity。

第1--32节的历史根本变化，是先训练24套task experts再让Writer从language+action-hidden video重建
policy-effective LoRA；该路线建立了expert evidence，但其whole-LoRA/ECP/Tangent/flow-teacher用法现均已
由正式结果退役。**当前第39节不重建或读取task expert**：它冻结historical v6，以correct train24 action
functional cotangent和action-free counterfactual-null rows更新video-keyed Program memory，并用anchored
RLS显式保留此前condition输出。Writer仍不
读取teacher action，validation/test actions始终不得读取；task experts只保留为历史policy-effective参照。

## 2. 直接证据与最早故障

K4 Phase-Aligned macro50/100/150/200 correct=`88/108/80/99`，四点union/intersection=
`157/36`。winner五臂=`108/115/94/101/121`：correct显著优于wrong，证明视频task
identity已进入closed loop；但reversed反而更高，且曲线仍大幅换手。

内部也排除了“视频没有改变LoRA”：wrong/shuffled/reversed的effective-BA relative-L2
中位分别为`.330/.188/.165`。真正失效的是参数流形：correct LoRA norm中位
`91.12`，stable rank却只有`1.00021`、首奇异值能量`.99979`；最后50步
factor/program的full24 gradient retention仅`.0463/.0436`。视频信号有传递、LoRA有
高增益，但functional credit不能把它稳定组织成多task共存的policy-effective方向。

direct Source-SFT的对照说明目标不是强制谱更高：两套SFT的mean-target stable rank
也只有`1.505/1.517`，但q/v跨层effective方向近零相关、layer energy profile高度重现。
所以新目标是学习真实policy expert的组织方式，而不是再加rank/diversity loss。

## 3. 与相关hypernetwork方法的结构对应

WIZARD的有用启示不是其benchmark数字，而是三个可迁移结构：先训每个task expert，
再以prompt+video生成expert LoRA；把LoRA按topological parameter order分成固定网格；
权重重建需要direction与scale同时对齐。SHINE的有用启示是：不用小型shared
output MLP逐段猜测权重，而让容量至少与生成LoRA同阶的memory在层/参数和rank轴
交替全局交换，最后直接slice/reshape为A/B。

本设计只采用这些通用结构，不复制其数据split、benchmark特例或仅适用于监督学习的
deployment bypass。同一video-to-LoRA graph后续可直接接收functional或reward cotangent。

## 4. Task-expert target bank

24个development-train tasks各自从同一frozen source step1000和同一rank-16 identity初态开始：

```text
one train task's 50 teacher-action episodes
  -> task-local PI0.5 action SFT
  -> one complete 38-target rank-16 expert LoRA
```

- 各expert只用自己task的actions，不做mixed-task梯度，从源头去掉task更新抵消；
- 24个expert使用相同A-template/B-zero、optimizer、schedule、batch与global checkpoint step，
  不按单task outcome选不同训练长度；
- 专家阶段只作为建立policy-effective parameter manifold的teacher，不是held-task oracle；
- validation/test actions不读，不训held expert，不用validation outcome调expert；
- 先用development-train official random-reset closed loop对同一global-step expert bank做质量门。
  若expert本身不能稳定提升自task policy，不进入meta-Writer训练，先在同一recipe上
  裁决是欠训练还是38-target topology不足。

首轮保留现有38 targets，因为v6-fast=`143`已证明这个topology有接近过门的
closed-loop能力，可以把“credit target”与“LoRA范围”分开裁决。只有task experts在
足够训练后仍存在明确结构上限，才依据真实inference-active Linear枚举扩大topology。

## 5. 视频是唯一dynamic value

首版保持one-shot：exact task language + exactly one action-hidden teacher video生成一套LoRA。
视频每帧用frozen source π0.5进行一次joint prompt+image forward，取最后高层task-span
multimodal hidden和Action-Expert interaction。对同一language再计算matched text/no-image baseline，
定义：

```text
video innovation_t = joint_hidden(language, frame_t) - baseline_hidden(language)
```

每条video的innovation只按normalized progress可微重采样到16个phase tokens，保留真实输入顺序。
language只决定“从图像中读什么”的query/context，不提供可单独生成LoRA的value或
task memory。因此：

- 换wrong video必须改变全部dynamic input；
- shuffled/reversed必须在phase轴上重排真实innovation；
- zero/no-image innovation必须输出A-template/B-zero identity；
- cache只保存action-hidden high-level features、frame indices和必要identity metadata，不保存
  action/state/reward/terminal。

K4不作为首版：当前K4系列已多次证明“增加shots”本身不会修复credit/
parameter manifold，且WIZARD的K=1/3/5/10也没有显示稳定aggregate收益。如果新
policy-effective target下的same-task跨video方差仍被证明是最早限制，owner已授权把同一图
扩展为few-shot；但不在证据前支付4倍视频计算。

## 6. Topological LoRA tokenization

对每个expert，预测目标是`delta-A = expert-A - shared template-A`与`expert-B`，
不是带gauge任意性的effective-BA分解。所有expert从同一A-template开始，使原始
factor坐标尽可能保持可比；同时报告gauge-invariant BA误差，不用raw sign作方法结论。

38 targets的76个A/B tensors按真实policy拓扑排序：action-in/action-out在前，再按
expert layer0→17和q/v排列。A保持`rank × input`，B转置成`rank × output`，每个宽轴
分512长chunk。当前合同唯一导出：

- 168个`[16,512]`topological chunks；
- 1,287,168个valid values；
- 1,376,256个padded values，padding只用mask隔离。

该规则从真实LoRA contract枚举形状，不写死4/8卡、固定layer分片或每rank任务数。

## 7. Bottleneck-free axial hyperdecoder

视频条件为`16 phase × 3072`：每帧包含2048维joint multimodal task-span hidden与
1024维Action-Expert suffix hidden的时间均值，两者都减去matched no-image baseline。将它投影到
512后，与168个chunk identities和16个
public-rank identities相加，得到`[batch,168,16,512]`memory。多个block在两轴交替：

1. 固定rank、跨168 topological chunks的全局交换；
2. 固定chunk、跨16 public ranks的全局交换。

最后使用square `512→512` zero-output projection直接成为chunk values，不经过小型shared
factor head、atom mixing、scalar gate或language residual。另一个per-chunk scale predictor只预测量纲；
direction和scale都来自同一video-conditioned memory。zero-output时delta-A/B都为0，部署LoRA
精确是shared template-A/B-zero identity。

CPU原型已验证真实38-target round-trip：168 chunks、1,287,168 valid values，两个axial
blocks约7.70M参数，输出shape和zero identity正确。该原型不是formal实现，正式代码
必须按单一canonical owner重写并退役K4 executable path。

## 8. Meta-Writer objective与训练单元

每个训练sample是：

```text
(one train-task language, one action-hidden video feature, that task's expert LoRA)
```

24 tasks先各自mean，再等权聚合，不按episode长度或expert性能改变task权重。loss由
masked raw-factor reconstruction、chunk-direction cosine和log-scale误差组成；effective-BA误差先作
机制监控，不在第一版叠加多个辅助项。目标是同时学到expert方向、层组织和
真实量纲，不强制高rank、正交、多样性或人工energy profile。

首轮使用cached frozen features，使gradient只优化topological Writer，把“视频表示随
functional noise漂移”与“参数生成失败”分开。如果权重重建和held closed loop证明
decoder成立，后续可对同一图使用functional或official reward继续训练；不新增监督
专用的deployment分支。

## 9. 为什么这不是“用监督trick替代RL”

task expert提供的是policy parameter坐标系与初始credit target，不是额外输入。在部署、
functional AS或reward训练中，Writer仍只看language+video并输出一套LoRA。换到新任务或
RL时，可以用task-local reward experts、functional gradients或PPO/SPO cotangent替换expert-SFT
teacher，不需要改输入、hyperdecoder、LoRA topology或rollout adapter。

本方法解决的是更一般的hypernetwork问题：先把条件信息对齐到真正有用的参数流形，
再用任意可用credit在这个流形上微调。它不依赖LIBERO task ID、成功脚本、
reward shaping、特定动作heuristic或只在uniform gradient descent下成立的optimizer trick。

## 10. A40与分阶段门

### 10.1 Task experts

用live空闲A40最多6张，每卡常驻一个frozen policy并串行训自己的task子集，
避免24次重复加载模型。各task独立optimizer/checkpoint/RNG并可exact resume，不使用
DDP/NCCL聚合不同experts。先在单卡profile physical batch、gradient checkpointing和三步finite/
resume，然后才训全24个expert。

### 10.2 Frozen feature cache

六个independent extractors只读train24的action-hidden videos；validation features只在rollout生成
LoRA时按零交互协议读取，不读actions/outcomes。cache预计为低GB级，在创建前依
当时`/data1` quota再做一次聚焦预检。

### 10.3 Meta-Writer

cached-feature训练不常驻数十亿参数policy，首先用最小单卡profile实测batch、显存、
throughput和exact resume；如果单卡足够快，不为了用满6卡引入新的同步故障。formal
从zero-output identity fresh训，checkpoint保存model/optimizer/scheduler/sampler/RNG完整状态。

## 11. 预注册行为裁决

1. 先封存task-expert bank的development-train closed-loop曲线和LoRA谱/层组织；
2. meta-Writer在固定24-task训练上报告expert reconstruction、same-task跨video方差、
   wrong/shuffled/reversed feature-to-LoRA传递，但不用这些选held checkpoint；
3. strict validation仍只认single checkpoint paired correct400、breadth、gained/lost、
   success union/intersection与five-arm video causality；
4. 达到strict `>150/400`后不自动停止，继续提高absolute、breadth、能力稳定积累与
   视频特异性；
5. 若expert bank强而meta-Writer重建准确、held rollout仍弱，再定位是24-task元学习样本不足、
   38-target范围不足或frozen representation缺少所需高层信息；不盲目加scale/rank/loss。

## 12. 禁调项和退役触发

禁止language-only LoRA bypass、task-ID route、挑video、multi-LoRA平均、checkpoint融合、
validation/test expert、强制rank/正交/diversity、从历史Writer warm-start、以functional loss代替
closed-loop选择，以及为了A40降低logical data/task coverage。

实现期间允许expert-bank builder与旧K4评测artifact loader短暂共存，仅用于构建新方法
所需teacher与读取已封存结果。当新meta-Writer完成profile后，K4 model/training/checkpoint/
live-generation executable path必须原位退役；历史由Git和formal artifacts保存，仓库只留一个
canonical active Writer。

## 13. Task-expert builder实现状态

首个retained实现已建立sealed config、task-local deterministic sampler、独立checkpoint与
单GPU多task串行worker。每个worker只加载一份frozen source policy，各task开始前把
public rank-16 LoRA重置到严格identity；不同task不共享optimizer、scheduler、sampler或
RNG状态，也不建立DDP/NCCL。formal拓扑固定为6 workers × 4 tasks，保持24 tasks全覆盖。

每个task checkpoint保存adapter、optimizer、scheduler、sampler cursor、CPU/CUDA RNG和
截至该步的metrics；run contract只记录路径、schema和文件大小，不新增SHA-256、MD5等
内容校验。formal当前仍由config显式阻塞，不能在A40 profile完成前启动。

聚焦CPU验证覆盖config/topology、exact sampler与stage-resume ownership、scheduler和
LoRA identity工具，共14项通过。下一边界是同一task root在live A40完成fresh0→1与
exact-resume1→3，实测finite、OOM、冻结参数、峰值显存和续训等价性；通过后才解除
formal阻塞并训练完整expert bank。

首次profile在模型加载后由scheduler contract正确拒绝：三步profile horizon被误当成
formal cosine horizon，因而小于25步warmup；没有完成optimizer step或写checkpoint。
修复后profile仍只执行三步，但复用formal 2000-step scheduler的前3步，避免用缩短的
诊断horizon悄悄改变真实学习率语义。

修复后的fresh0→1已完成finite step并写出adapter/trainer/RNG完整checkpoint；首次
resume在加载trainer时发现`map_location=cuda`把CPU RNG state也搬到GPU，因而在任何
续训step前由PyTorch拒绝。checkpoint内容本身完整；loader现统一在CPU反序列化trainer，
optimizer再按参数设备恢复，CPU/CUDA RNG分别从CPU ByteTensor恢复。该修复后必须用新
commit和新root重做完整fresh/resume证据，不能沿用旧run contract冒充通过。

## 14. A40 profile seal与formal边界

clean`174d292`最终在live空闲`gpu01:0`完成三条证据链：fresh0→1、同root
exact-resume1→3，以及独立root contiguous0→3。physical B16、rank16、38 targets、完整action
query与formal 2000-step scheduler前缀均未降低。三步loss=
`.221725/.283785/.259915`、gradient norm=`.029505/.032996/.035243`；峰值
allocated/reserved=`15,082,000,384/21,313,355,776` bytes，0 OOM/nonfinite，base policy没有
梯度。resume与contiguous的科学metrics完全一致，step3 adapter逐字节一致，不使用内容hash。

因此`configs/pi05_video_expert_manifold_v1.json`已seal formal：6个independent单卡workers，
每worker严格4 tasks，每task B16，统一先训到step1000并保存250/500/1000；不同task不做
collective。只有development-train official closed loop和LoRA内部组织可以裁决是否在同一root
exact-resume到2000；不得按单task结果选择不同step或训练held experts。

## 15. Expert评测、视频cache与meta-Writer实现边界

task-expert正式训练运行期间，后续实现放在独立worktree，避免修改formal checkout。当前已完成：

- canonical evaluator可直接安装完整train24 task-expert bank，并按task切换对应rank-16 LoRA；
  只允许`development_train`，统一global step由250/500/1000 official closed loop裁决；
  bank身份绑定同一config相对authority、schema、sealed task-expert runtime与source，而不绑定某个
  worktree绝对前缀，因此formal main可保持冻结、评测从clean隔离worktree读取同一bank；
- expert几何分析按统一step测effective LoRA谱、target/layer能量、跨task方向与checkpoint位移；
- frozen feature cache只读取action-hidden视频帧，保存每task 50条`[16,3072]`BF16 innovation；
  sealed manifest记录source、path/schema/size和零action/state/reward/terminal reads，不做内容hash；
- topological decoder严格覆盖168个`[16,512]`chunks，direction先按每chunk有效坐标归一，量纲由
  同一video-conditioned state的动态scale和训练expert导出的静态per-chunk scale prior共同表达；
  静态prior自身没有direction，zero video仍精确identity；
- meta训练每macro覆盖train24，每rank固定4 tasks并先形成local task mean；随后按固定参数顺序拼接
  单一flat gradient，以固定Ring/Simple NCCL做一次六rank all-reduce mean，因而严格等于24-task
  等权mean且没有未封存的DDP reducer状态。模型、optimizer、scheduler、每rank RNG和macro cursor原子
  checkpoint；模型与cache完成local CUDA构造后才建立NCCL，BCI仍显式要求
  `NCCL_P2P_DISABLE=1`。profile/formal均fail-fast要求GPU-local NUMA affinity，run contract逐rank
  记录local/physical GPU、NUMA node与CPU affinity，不能只记录`CUDA_VISIBLE_DEVICES`字符串；
  每个macro的step wall以及累计peak allocated/reserved显存均在全部rank上取`MAX`，避免rank0
  偶然较快或较省显存时形成错误profile seal。
- canonical evaluator新增独立Expert-Manifold adapter：每个rollout按50-state无放回schedule只取一条
  action-hidden video，correct/same/wrong/shuffled/reversed共享state、policy RNG、video ordinal与
  frame-order seed；online frozen encoder和topological Writer先生成episode LoRA cache，随后释放
  Writer并复用同一source policy做cost-balanced rollout。validation/test只开放video，不开放expert
  或action。

以上只完成CPU合同与代码，不构成A40 profile或性能证据。feature extraction与meta训练formal仍由
config阻塞；必须先完成live profile、fresh0→1、exact-resume1→3和原始六rank规模验证。K4旧
executable只在新meta-Writer通过profile后按第12节退役。

meta训练profile checkpoint只允许在`smoke`评测中按`profile_defaults.checkpoint_macros`读取，用于
在formal seal前验证online frozen encoder、Writer generation batch、每卡generator/rollout并存与
显存释放；formal评测仍只接受sealed `formal_run.checkpoint_macros`。两种mode不能互相冒充。

## 16. Task-expert bank完成与2026-08-08交接边界

clean`81101fe`的正式expert阶段已自然完成统一step1000。唯一root为
`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`：6个independent workers、
24/24 tasks完成，每task均保存step250/500/1000，共72个checkpoint，约562MiB。三个统一点最后
50步的24-task等权mean action loss为`.115355/.107207/.105372`；它只记录task-local拟合，不能
代替official development-train closed-loop选择。

第15节列出的retained evaluator/cache/meta-Writer实现已并入`codex/bci-continuation`，不再位于
第二个活动工作树。当前full24三checkpoint正式geometry已完成：step250/500/1000的
effective-LoRA norm中位为`2.792/3.652/4.170`，stable rank中位为`1.126/1.129/1.129`，
跨task effective cosine中位为`.108/.095/.100`。16个rank coordinates全部active且top4
coordinate energy约`.262/.260/.258`，但q/v B-column cosine仍约
`.828/.843→.861/.853`，所以“坐标都活跃”不能被解释成16个独立有效方向。仍没有
expert-bank closed-loop、A40 meta profile、正式feature cache、meta checkpoint或新strict
rollout；geometry不足以封存统一expert step。

若后续证据支持step1000后仍有material上升，必须从`81101fe`创建独立frozen worktree，并沿同一
root把全部24 tasks统一exact-resume到2000；当前分支新增的meta/evaluator实现不能改变该正式训练
合同。若不续训，则选择一个统一expert step后再进入feature profile/cache与meta profile/formal。
owner已在新session完成讨论后恢复持续自主执行。K4 executable的removal trigger仍是新meta-Writer
A40 profile通过，不因代码已实现而提前触发。

## 17. Direct-expert闭环裁决与feature-cache profile seal

clean pushed`1362d15`的唯一有效development-train三点roots均使用每卡3 replicas、每点6 workers。
每点覆盖24 tasks×50 fixed states，108/108 shards attempt1、worker exit0、0 retry/failure，三点的
task/state/env seed/policy seed与执行到共同长度的noise序列严格配对。step250/500/1000=
`432/557/624`，四suite依次为Spatial=`123/147/170`、Object=`125/191/208`、Goal=
`142/163/164`、Long=`42/56/82`。500→1000是`143/76` paired gains/losses，18/4/2 tasks
升/降/平；nonzero breadth=`23→24`、成功至少25次的task=`11→14`。

三点state union/intersection=`731/332`，逐task任选最优checkpoint的privileged oracle=`636`，只比
统一step1000高12。这既拒绝按task挑checkpoint，也说明step1000是强而广的统一中间点。与此同时，
Goal在500→1000为`163→164`却有`21/20` gains/losses；last50 loss变化与success变化Spearman仅
`.094`，LoRA norm变化与success变化为`-.108`。独立expert内部也存在surrogate-to-closed-loop
边界轮换，因此正式决定沿原root把全部24 tasks统一exact-resume到2000，并用1500/2000 closed loop
选择target，不用更低loss或更大norm自动选择。

并发边界也已实测：初始总36 replicas在0 scientific rows前使gpu02主机内存不安全；总24 replicas
时每卡约37.7GB静态占用并在首个inference activation OOM；两批roots均标记ABORTED且不得resume。
有效总18 replicas约30.3GB/卡，主机与A40均稳定。这个结果只修正评测资源合同，不改变1200-state
科学覆盖。

feature cache profile在同一clean`1362d15`、`gpu02:4`完成task0的4条action-hidden videos：
task extraction wall=`4.372s`，raw/sampled frame count=`84--98/18--21`，输出
`[4,16,3072]` BF16；peak allocated/reserved=`10,468,548,096/19,232,980,992` bytes，
teacher action/state/reward/terminal reads、OOM与nonfinite均为0。由此正式seal六worker×4 tasks×
50 videos cache，不降低`videos_per_batch=4`、frame stride5、phase16或feature width3072。

## 18. Formal feature-cache封存

clean pushed`222d3ac`上的train24×50正式cache已由6个独立workers自然完成，root为
`runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808`。它包含
24/24 task records、6/6 worker summaries和24个`[50,16,3072]` BF16 feature tensors，task ordinals
恰好覆盖`0--23`、每task demo ordinals恰好覆盖`0--49`，总量约113MiB。peak
allocated/reserved=`10,504,039,936/19,232,980,992` bytes，teacher action/state/reward/terminal
reads合计0，worker logs无error。仓库seal入口已生成canonical `cache_manifest.json`，其
training commit为`222d3ac72591bf44fa46ff436ace22d8cd5afa35`，information-wall计数全0。该cache
只是冻结的action-hidden Writer输入，不是新Writer性能或视频因果证据。

## 19. Phase-centered causal-prefix dynamic value与no-video裁决

正式cache完成后，对全部24 tasks×50 videos=`1,200`条`[16,3072]`innovation做了只读CPU
分析。每条视频的phase-DC能量占比中位`.98057`，真正temporal residual只有`.01943`；然而
同task leave-one-video-out的ordered temporal-template cosine中位仍为`.88284`，reversed为
`-.32402`，16-phase shuffle proxy为`-.02194`。因此frozen encoder既保留了稳定、有方向的
顺序证据，也同时提供了一个能量大约50倍、足以让constant-target重建绕过时序的静态task捷径。

expert targets本身没有坍缩：step1000 raw target的task-centered effective rank为`19.45`，raw
mean只占平均target能量`.41465`，B-factor跨task cosine中位`.10245`。temporal task geometry与
raw/B target geometry的Spearman为`.46046/.45087`。在leave-one-task linear transfer proxy中，
phase-centered one-shot对B target的cosine中位`.38607`，与DC-only的`.39500`同档；但reversed/
phase-shuffled分别降到`.20667/.26386`。同一proxy改为3-shot/5-shot只升到`.39051/.39290`，所以
当前首要限制不是shot数量，而是允许静态DC直接成为LoRA value。shuffle数字只属于sealed phase
cache上的诊断proxy；正式评测仍须先重排raw sampled frames再完整encoder forward。

phase centering单独仍有一个严格结构缺口：如果训练学会忽略learned phase keys，attention对成对置换的
centered keys/values仍是frame-set permutation invariant，而同task恒定target不会惩罚这条解。CPU
最小反例把phase keys置零后，ordered/reversed输出只剩浮点求和级绝对差异。固定的sqrt-normalized
causal-prefix transform消除此原始set path；其uniform-pool template correct/reversed/shuffled=
`.96263/-.94287/-.04463`，B-target proxy=`.38820/.06042/.19110`，correct几乎不损失，order
margin由`.17940/.12221`提高到`.32778/.19709`，且四suite均为正。3/5-shot causal correct只到
`.39379/.39558`，仍不支持当前切换few-shot。carrier RMS中位从简单centered的`.14446`提高到
causal-prefix的`.19388`，能量比中位`1.8875`；它增强的是无DC的时序carrier，不是生成LoRA能量
已经健康的结论，后者仍须profile/training geometry验证。

据此，在第一次meta GPU profile前原位收紧唯一canonical decoder，不保留旧可执行分支：

```text
routing_memory_t = W(video_innovation_t)
K_t = RMSNorm(routing_memory_t) + phase_key_t
centered_t = routing_memory_t - Mean_phase(routing_memory)
V_t = Sum_{s<=t}(centered_s) / sqrt(t + 1)
LoRA = topological_axial_decode(Q_chunk,rank, K, V)
```

完整joint+Action-Expert特征仍参与phase key/routing，但只有phase-centered、固定causal-bound的
video dynamics能提供content value。任意zero innovation或所有phase完全相同的非零输入都必须精确
输出template-A/zero-B identity；即使模型忽略phase keys，reversal/shuffle也不再只是value的集合
置换。CPU原型和retained tests验证了
constant/zero identity、ordered-vs-reversed差异，以及fresh第一步打开output/scale、第二步梯度到达
input projection、cross-attention和phase keys。这个变化不增加language-only route、scalar gate、
第二套LoRA或额外输入，也不改变已封存feature cache。

严格评测同时新增第六个`no_video`反事实：保留correct arm的task/state/policy RNG、teacher demo
ordinal和exact task language，但不读取teacher frames，直接把数学上matched baseline-minus-baseline
的video innovation置零，再完整运行Writer。它必须逐episode生成identity LoRA，并应与同panel frozen
source policy逐row一致。正式方法输入仍是恰好一条action-hidden video；`no_video`只作因果control，
不成为部署或训练分支。

可复现CPU入口为`scripts/analyze_expert_manifold_feature_dynamics.py`。在full24 step1500/2000完成后，
同一脚本将复算target evolution；统一expert step仍以development-train closed-loop为主证据，若晚期
增益进入平台而B-target可迁移性继续下降，则优先较早的near-max统一step，不按task混合checkpoint。

## 20. Cached-rollout evidence schema纵向合同

profile前对`online generation → release Writer → cached policy rollout`做静态纵向审计时，发现统一
`expected_writer_episode` adapter wrapper在接入Expert-Manifold dispatch后没有暴露和转发调用方既有的
`evidence_schema`关键字。结果不是数值偏差，而是LoRA cache生成成功后、第一条scale-out episode
evidence构造即`TypeError`；因此不能用只测generation的smoke替代完整vertical smoke。

canonical wrapper现在对旧Writer继续原样转发该schema，对Expert-Manifold则生成expected evidence后
显式比较`schema_version`并在不一致时fail-close。回归同时覆盖正确schema等价与错误schema拒绝；聚焦
62/62、全仓220/220及changed-file `py_compile`通过。这个修复不改变视频读取、Writer forward、LoRA
cache内容、source policy或任何训练/rollout随机数，只清除profile前的工程阻塞；仍须真实A40 online
generation和cached rollout共同通过后才能seal formal。

## 21. Expert2000终态与统一target闭环选择

clean`81101fe`的原root已严格exact-resume完成全部24 experts到2000，step1500/2000各24个checkpoint、
6/6 summaries与24/24 completion齐全。最后50步等权action loss在1000/1500/2000为
`.105372/.103881/.103526`，但parameter manifold已明显平台：effective norm中位=
`4.170/4.212/4.212`、stable rank约`1.129`、跨task cosine约`.100`；1500→2000 effective update
energy接近零。

视频到target的CPU proxy也没有随晚期训练改善。causal-prefix one-shot B correct=
`.38820/.38685/.38678`，reversed=`.06042/.06399/.06425`，phase-shuffled=
`.19110/.19195/.19199`；raw centered effective rank与B跨taskcosine基本不变。因此不能因2000的loss
更低就选2000，也不能只因1000更易预测就跳过真实行为。

clean pushed`1362d15`上的1500/2000 direct closed loop现已自然完成。两点各覆盖1200 unique rows，
因本轮每点3 GPUs×3 replicas而各为126个queue jobs、9 workers；全部attempt1、exit0、0 retry/failure。
它们与既有250/500/1000在task/state/env seed/policy seed及共同长度noise prefix上逐row严格配对。五点
success=`432/557/624/638/658`，四suite在1500→2000依次为Spatial=`178→181`、Object=`216→228`、
Goal=`164→166`、Long=`80→83`；paired gains/losses=`77/57`，24 tasks为17升/5降/2平。step2000相对
step1000为`91/57` gains/losses、净增34。尽管nonzero breadth从step1000的24变为晚期23，且
1500→2000的微小parameter位移仍引发明显state turnover，2000同时取得最高absolute、四suite全净增和
多数task改善，已达到预注册的material behavior证据。因此统一target正式选择step2000；不得按task
混点。该658/1200是privileged development-train direct-expert target质量证据，不是Writer validation
成绩，也不计入`>150/400`长期门。

## 22. Meta-Writer profile前边界

config现只封存统一expert step2000，formal状态仍blocked，不能因此启动正式训练。2026-08-09 00:01
CST实时比较两节点：gpu01的物理`0,1,2|4,5,7`为空闲A40且构成3+3 NUMA，物理3有`nlge` VLLM不触碰；
gpu02物理6、7已有他人进程，空闲0--5只能形成4+2 NUMA，故不用于六rank DDP profile。gpu01 available
host memory约516.5GB；`/data1` quota=`552,249,764/1,073,741,824 KiB`，profile新增保守低于2GiB。
下一步必须从clean pushed seal执行六rank fresh0→1、同root exact-resume1→3及独立contiguous0→3，
逐rank记录physical/local GPU、NUMA/affinity并比较三步科学metrics与step3 Writer tensors；不使用hash。
三条证据通过后，profile macro3只用于online frozen encoder→Writer LoRA generation→释放Writer→
cached policy rollout纵向smoke，profile权重永久不得进入formal。

科学config/target seal为clean pushed`d96f0fb`，profile exact roots、三条命令和验收门只取
`task_plan.md`顶部合同。root后缀标识该科学seal，实际运行commit由run contract记录且必须clean/pushed。

首次真实六卡profile的fresh0→1与resume1→3均finite、NUMA正确、峰值reserved低于1GiB；独立
contiguous0→3的macro1 checkpoint与resume root逐字节完全一致，optimizer/scheduler及六rank RNG也
完全一致。但macro3有45个Writer tensors分叉，最大绝对差约`1.30e-5`，因此profile按预注册exact门
否决，未进入online smoke或formal。deterministic algorithms、cuBLAS deterministic workspace及
math-SDPA诊断仍复现“resume路径A/contiguous路径B”，排除漏存state和随机CUDA kernel；分叉始终在
恢复后的首个optimizer update之后出现。

当前唯一有代码差异支持的working root cause是：新meta trainer的DDP构造遗漏了仓库
source-base/Source-SFT已有的`static_graph=True`合同，并
反向保留了无必要的buffer broadcast。DDP reducer的首次迭代自适应状态不在checkpoint中，重启后的
macro2与连续run的macro2处于不同reducer生命周期。canonical修复只让固定训练图使用
`static_graph=True`、`broadcast_buffers=False`、`find_unused_parameters=False`并把前两项写入run
contract；模型、数据、loss、optimizer、task平均、RNG和LoRA topology均不变。这个解释仍必须由
新profile的逐字节parity证伪或确认；必须从新clean/pushed
commit和全新roots重做完整profile，旧profile/probe权重全部弃用。

候选修复已由clean pushed`12727b8`封存；新static-graph reprofile的固定roots、三条exact command和
不放宽的byte-parity门取`task_plan.md`顶部。旧`ac56ab8` roots及deterministic/math-SDPA probes均只作
失败证据。

真实reprofile在第一个macro、0 optimizer step处触发PyTorch 2.11 DDP
`expect_autograd_hooks_`内部断言，证明static graph与当前四次`no_sync` backward不兼容；随后只关闭
buffer broadcast的dynamic-graph probe仍精确复现原A/B分叉。因此buffer不是根因，static graph也不是
可用修复；`12727b8` root不得resume。

根因结论收紧为“DDP reducer生命周期是未checkpoint的隐藏训练状态”，而不是某个buffer或attention
kernel。canonical现删除DDP wrapper：各rank仍以physical microbatch1顺序反传4个task loss形成local
task mean，再按parameter registration order拼成一个flat gradient，以显式固定
`NCCL_ALGO=Ring/NCCL_PROTO=Simple`做一次all-reduce mean；随后六rank执行相同clip与AdamW step。这个
算子严格保持原24-task等权梯度期望、模型/loss/optimizer/RNG不变，同时把跨resume分布式状态缩减为
一个无历史collective。run contract记录无model wrapper、reduction语义及NCCL算法。必须从新commit/
新roots重做byte parity；此前所有profile/probe权重弃用。

retained CPU合同现为聚焦49/49、全仓223/223；architecture guard无hard violation和parallel family。
这些只证明接口与结构闭合，不代替六卡exact-resume profile。

implementation/config seal为clean pushed`c33a16b`；新flat-reduction roots、带Ring/Simple的三条exact
command及byte门取`task_plan.md`顶部。实际run commit必须clean/pushed并由run contract登记。

clean pushed launch-record`b00024b`的真实六卡flat-reduction profile通过预注册core门。resume路径与
独立contiguous路径三步`loss/raw/direction/log_scale/gradient/LR`逐值相同，macro1和macro3 Writer及
六rank macro3 RNG逐字节一致。macro3 `trainer.pt`原始序列化bytes不同，但load到CPU后的optimizer和
scheduler逐项0差异；因此结论只宣告训练语义与Writer byte-exact，不宣告trainer容器byte-exact。
resume/contiguous峰值allocated/reserved=`736,117,760/876,609,536`与
`735,831,552/815,792,128` bytes，0 OOM/nonfinite；run contract封存正确3+3 NUMA、physical/local
映射、`distributed_model_wrapper=none`、single-flat reduction、P2P disable与Ring/Simple。这个结果确认
DDP reducer生命周期就是先前不可续训的隐藏状态；profile权重仍永久不得进入formal。

core profile通过后只剩一个工程门：用macro3 checkpoint在单张A40执行online frozen video encoder→
batch4 Writer generation→8套完整LoRA cache→释放Writer/encoder→保留source policy并以3 replicas完成
validation 8 tasks×1 state smoke。它只验证部署纵向路径、显存和evidence，不用8-row success判断方法。
2026-08-09 00:48 CST live选择`gpu02:0`，忙碌的`gpu02:6/7`和`gpu01:3`不触碰；exact command与验收
门写入`task_plan.md`。通过前formal仍blocked，旧K4 executable也暂不删除。

首次纵向smoke没有启动CUDA：prepare比较profile training source与evaluation source时，把formal检查记录的
非空`source_run_summary`和smoke模式对同一final checkpoint给出的`null`误判为source policy变化；除此
以外所有source字段相同。修复没有放宽模型身份，只在这一模式相关字段缺省时，从training contract补回该
descriptor并重新检查summary文件path/bytes/schema；checkpoint、model files、run contract等任一真实差异
仍拒绝。失败root标记ABORTED且不可resume；聚焦58/58、正式assets环境全仓224/224和真实macro3 smoke
authority检查通过。replacement fresh root写入`task_plan.md`，仍须重新live看卡。

replacement随后从clean pushed`31d41d8`在live空闲`gpu02:0`自然完成。一个generator按两个batch4生成
8套完整rank-16 LoRA并写入8个唯一cache entries，generation wall=`12.634s`；Writer/encoder随后释放，
已加载source policy不重载地由3个workers完成8/8 unique rows，全部attempt1/exit0、0 retry/failure/
OOM/nonfinite。peak allocated/reserved=`10,576,054,272/11,182,014,464` bytes，release后为
`9,391,467,520/9,651,093,504` bytes；teacher action/state/reward/terminal reads全0。`1/8` success
只记作纵向execution smoke，不解释模型质量。

由此六卡stateless flat-reduction exact-resume和单卡online generation/cache/release/rollout两道工程门
全部满足。`configs/pi05_video_expert_manifold_v1.json`现以两组精确evidence seal meta formal；profile
checkpoint继续永久禁止warm-start。第12节K4 executable移除触发已经满足，必须在identity-fresh formal
前原位退役旧model/training/checkpoint/live-generation路径，历史只由Git、文档和formal artifacts保存。

## 23. Canonical退役完成边界

2026-08-09已按第12节完成原位退役：旧K4/AS/RL model、training、checkpoint、live-generation、CLI与
专属配置/测试从当前工作树删除；Git和formal artifacts保留历史。共享数据读取、functional LoRA、public
topology、evaluation cache/runtime仍保留，但动态Writer dispatch、one-shot schedule、episode evidence和
live adapter现在只由Expert-Manifold拥有。统一evaluator同时保留静态Source-SFT和task-expert对照，不再
接受旧AS/RL动态adapter或rollout全局B-scale。CPU-only全仓`186/186`与compileall/diff通过，architecture
guard无hard violation或parallel family。该收口不改变模型数学、sealed config、训练target或任何科研结果；
formal仍必须从identity fresh开始。

## 24. Identity-fresh formal分段边界

首个formal轨迹从zero-output identity fresh启动，统一step2000 expert target、train24×50 frozen video
cache、one-shot sampler、world6 task ownership、single-flat Ring/Simple mean、AdamW与800-macro scheduler
全部保持sealed。首段只用`--stop-after-macro 50`形成第一个正式checkpoint，不压缩scheduler、不读取
held actions、不加载任何profile或历史Writer权重；每macro仍是24-task等权和24条独立teacher videos。

macro50必须先通过formal completion/finite/NUMA/NCCL/checkpoint合同，再做validation strict paired
correct400和expert→generated LoRA→fixed action传递分析。reconstruction loss、task-expert proximity和
LoRA几何只能定位接口，不能代替closed-loop结果决定续到100；后续exact resume仍须保持同一root、
commit科学合同、3+3 NUMA topology、sampler/RNG和scheduler cursor。

clean pushed launch-record`446cd42`已按该边界自然完成0→50：50/50 finite metrics、macro50完整
Writer/trainer/六rank RNG checkpoint、0 OOM/nonfinite；训练body=`10.239s`，peak allocated/reserved=
`737,273,344/815,792,128` bytes，3+3 NUMA与全部collective字段通过。该结果只解封macro50 strict
correct400，不构成性能门通过，也不自动授权resume到100。

formal checkpoint的config身份必须允许training和evaluation处于不同clean frozen worktree：比较同一
仓库相对authority路径、schema和bytes，并继续逐项比较method、information wall、topological writer、
meta training、source与checkpoint manifest；不得把机器上的worktree绝对前缀当作科学身份，也不得只按
basename放宽。首次macro50评测暴露并根修了该工程冲突，失败发生在0 CUDA worker/0 row，不构成科研结果。
根修已由clean pushed`d59841e`封存；replacement只能使用全新root，并保持原macro50 correct400的全部
scientific pairing和资源边界。

## 25. Macro50负裁决与zero-preserving topology-address修订门（2026-08-09）

replacement formal correct400已自然完成：`48/400`，72/72 jobs、400 unique rows、18 workers
attempt1/exit0、0 retry/error/OOM/nonfinite，teacher frame与信息墙证据完整。它与source base在同一
task/state/env/policy-RNG panel上同分，paired gained/lost=`5/5`，所以不能把Goal-6的42次成功解释为
Writer新能力。原macro50 checkpoint永久停止，不exact-resume到100，也不warm-start新结构。

失败不是“LoRA能量仍太小”。400套generated LoRA effective norm中位=`4.54899`，已接近step2000
expert的`4.21249`；但stable rank=`1.00000144`、top singular energy=`.99999856`，q/v/action
B-column cosine全部约`.99999`，nearest-of-24 expert effective cosine中位仅`.007974`。train24
自身demo0的raw/effective target cosine中位也只有`.02326/.01081`，因此最早故障发生在训练域内。

纵向结构probe给出精确机制：学习到的chunk/rank query仍有约`.486/.481` centered energy，但query只
作为cross-attention权重，不进入value或residual。16-phase causal dynamic values输出到2,688个query后，
rank/chunk centered energy中位只剩`1.04e-6/1.08e-6`；四个无位置地址的共享axial blocks及output
projection后进一步为`2.51e-8/4.67e-10`，而expert target为`.936/.994`。一旦cross output近同，后续
permutation-equivariant算子没有信息可重建topology identity；这比closed-loop、scale或video encoder
更早失效。

下一canonical修订只允许在现有唯一Writer内加入乘性地址绑定：dynamic video latent与静态
`chunk_query + rank_query`逐元素结合后再写出，使地址参与value但没有独立静态输出。绑定必须满足：

1. zero或任意phase-constant video innovation仍逐tensor精确identity；
2. exact language不能单独生成LoRA，video仍是唯一dynamic value；
3. 非常量ordered video在output owner前保留material chunk与rank centered energy；
4. 完整rank16、168 chunks、expert2000 target、train24×50 cache、one-shot sampler、objective、
   optimizer、world6 task-complete mean与strict evaluator全部不变；
5. fresh schema/checkpoint family，旧macro50不得加载；先CPU shape/identity/gradient/address-retention，
   再六卡fresh/exact-resume profile，最后identity-fresh formal。

本轮不同时加入few-shot、reversed/shuffled negative loss、RL、scale gate或新的expert target。这样若
target cosine、SFT-like几何和closed-loop改善，可以归因于最早地址接口；视频时序五臂若随后仍不通过，
再单独裁决是否增加显式order-negative训练。

## 26. Zero-preserving topology-address binding implementation seal（2026-08-09）

上述单变量修订已由clean pushed`cd95281`在唯一canonical model中原位实现，没有增加Writer family、
runner、训练objective或部署输入。当前forward的地址接口为：

```text
D[b,c,r,:] = AxialBlocks(CrossAttention(video causal values))[b,c,r,:]
A[c,r,:]   = chunk_query[c,:] + rank_query[r,:]
Z[b,c,r,:] = RMSNorm(D[b,c,r,:]) * RMSNorm(A[c,r,:])
LoRA value = SharedOutputProjection(Z)
```

`D`仍是唯一dynamic value；`A`只是public LoRA topology的静态坐标，不能单独到达output。
chunk scale owner仍读取未绑定的动态`D`，静态per-chunk offset也只能缩放已经非零的direction；因此
zero或phase-constant video令`D=Z=direction=0`，完整输出仍是template-A/zero-B identity。

新回归覆盖四个关键谓词：共同dynamic值经绑定后chunk/rank centered energy均`>.1`；zero dynamic
精确保持零；ordered与reversed video仍不同；zero-output bootstrap打开projection后address norm获得
非零梯度。聚焦合同47/47、正式LIBERO assets环境全仓192/192、compileall与diff check通过；
architecture guard为REVIEW但无hard violation、无parallel family。

该forward图新增`address_norm`参数，旧macro50 checkpoint必须strict-load失败；旧flat-reduction profile
和online smoke也只属于已拒绝decoder。config已移除两组旧证据并重新设为
`blocked_until_live_a40_profile_and_online_generation_smoke`。未来seal时profile与smoke evidence都必须
显式记录`normalized_dynamic_times_normalized_chunk_plus_rank_address`，以防旧证据被机械复制。

下一执行顺序固定为：从clean/pushed launch-record和全新roots重做六卡fresh/resume/contiguous
exact-resume profile；再用profile macro3做单卡online generation/cache/release/rollout smoke；两门均过后
才允许identity-fresh formal。feature cache、step2000 expert target、one-shot schedule、reconstruction
loss、optimizer、world6 task mean与strict evaluator保持不变。当前尚未启动新GPU工作。

本修订的A40工程门已预注册：只使用启动前live空闲的`gpu01:0,1,2|4,5,7`，依次执行fresh0→1、
同root exact-resume1→3和独立contiguous0→3。验收同时要求scientific metrics、Writer/RNG bytes、
optimizer/scheduler语义、`address_norm`梯度、NUMA/physical-local/deferred-NCCL与0 OOM/nonfinite；命令和
全新roots取`task_plan.md`顶部。profile权重不进入formal，当前仍未启动GPU。

该工程门随后由clean pushed`a3666ba`通过：三步科学指标、macro1全文件、macro3 Writer/RNG精确
一致；trainer语义一致但容器raw bytes不同。`address_norm`具有非零finite Adam状态并在macro1→3
发生`1.62e-5`最大权重变化；两root峰值reserved均低于`.9GB`，0 OOM/nonfinite，六卡已释放。
profile权重继续弃用，config仍blocked；下一门只做8-row online generation/cache/release/rollout smoke。

## 27. Address-binding execution seal与fresh formal门（2026-08-09）

单卡online generation/cache/release/rollout smoke已在clean pushed`eb32f3f`自然完成：8个validation
tasks各一行、8套唯一完整FP32 LoRA、2个batch4、3 workers attempt1/exit0、0 retry/failure/OOM/
nonfinite。Writer/encoder释放后复用同一source policy且没有reload；每行teacher frames used，teacher
action/state/reward/terminal reads均为0。generation wall=`9.731s`，peak allocated/reserved=
`10,576,056,320/11,182,014,464` bytes。`1/8` success只作执行证据。

8套macro3 LoRA的CPU只读审计给出0 nonfinite、effective norm中位`.70069`、stable rank中位
`1.98260`、top singular energy中位`.51202`、16/16 coordinates active、top4 coordinate energy中位
`.31274`。这说明新address-value接口在训练早期已避免旧图的必然近rank1输出，但不把高stable rank
当作目标，也不声称macro3已接近experts；不同task pairwise effective cosine中位`.54184`仍表明方向
分离尚未成熟。

六卡profile与单卡smoke evidence现共同绑定
`normalized_dynamic_times_normalized_chunk_plus_rank_address`，config formal状态重新seal。唯一被解封的
科研动作是从identity fresh训练到macro50；profile权重永久弃用，旧macro50也因schema不兼容且已负
裁决而禁止加载。macro50后先看strict correct400、train24 expert proximity、chunk/rank retention和
完整LoRA谱；只有absolute/breadth与内部传递共同支持时才resume。若absolute提高而顺序五臂仍失败，
下一单变量候选才讨论显式order-negative credit；本段不提前混入few-shot、RL或新的target。

## 28. Address-binding macro50内部证据与closed-loop裁决门（2026-08-09）

clean pushed launch-record`925e7b1`已从identity fresh完成0→50：50/50 finite、1,200 train24 one-shot
conditions、完整Writer/trainer/六rank RNG checkpoint、0 OOM/nonfinite；训练body=`10.204s`，peak
reserved=`836,763,648` bytes。末步复合loss/raw reconstruction=`.083826/6.903e-5`仍只作surrogate。

同checkpoint的train24 demo0纵向证据精确分离了“上游动态地址”与“显式绑定”的作用。cross与axial
chunk/rank centered energy中位分别只有`4.60e-6/4.47e-6`和`5.64e-6/6.14e-6`；乘性address后跃迁为
`.4930/.4765`，final output为`.4669/.6159`，target为`.9936/.9364`。这说明结构修复确实位于旧图最早
断点，并没有假称cross-attention自己学会了topology。

raw token与own-expert effective cosine中位=`.1177/.1342`，相较旧图train24 demo0约
`.0233/.0108`形成material改善；nearest expert cosine=`.1393`且8/24 own-nearest。generated LoRA
norm/stable-rank/top-energy中位=`3.360/1.349/.757`、16 coordinates active，故旧“同能量但近rank1且
近正交expert”的失败形态已改变。

新风险是高task共线：24个generated LoRA两两effective cosine中位`.8686`，远高于expert bank约`.100`；
top4 coordinate energy也为`.8694`。因此新图可能学到一个较健康但过于公共的方向，仍不足以稳定承载
24个task。唯一下一裁决是macro50 strict correct400：不过absolute/breadth门就不resume；若通过才继续
训练，并用same/wrong/shuffled/reversed/no-video区分task公共方向与真实视频时序知识。本阶段不因内部
几何漂亮而改变one-shot、target或loss。

## 29. Address-binding负裁决与Causal Barycentric Topological Writer（2026-08-09）

### 29.1 闭环与最早剩余断点

macro50 strict correct400自然完成为`75/400`、breadth=`4/8`，逐task按Long/Goal/Object/Spatial为
`[2,0]/[1,47]/[25,0]/[0,0]`。相对exact同teacher-video schedule的旧图gained/lost=`31/4`，说明
address binding不是形式修复；但它仍明显低于v6-fast`143`和长期`>150`门，而且新增能力集中于两个task。
该checkpoint永久停止，不resume100、不做五臂。

400套LoRA的norm/stable-rank/top-energy中位=`3.20095/1.31757/.77753`且16 coordinates active，
所以旧能量与近rank1故障已消失。新的决定性失败是same-task不同video、cross-task和task-mean
effective cosine中位分别为`.99791/.94197/.94270`；最近train expert仅`.12734`。真实experts跨task
中位约`.100`。结合macro3八task pairwise `.54184`，训练到50反而向公共方向收缩。raw expert mean
本来占约`.414` target energy，而centered target仍有19.54 effective dimensions；让高容量decoder在
1.29M raw factor坐标上做普通重建，最容易先学公共均值，不能靠继续训练证明会自然恢复task residual。

### 29.2 选择的闭式流形坐标

下一canonical不再训练完整权重decoder。固定train24 step2000 experts及其50条action-hidden视频的
causal centroids。对一条部署视频，仍先由frozen π0.5计算exact-language-conditioned
`video_innovation[16,3072]`，再定义唯一dynamic query：

```text
x = mean_phase(phase_centered_causal_memory(video_innovation))
```

将24个train centroids单位化；以它们在当前fold/部署basis中的均值为原点，求ridge `.3`的kernel
barycentric coordinates。对basis matrix `C`和query `x`：

```text
w = (C C^T + 0.3 I)^-1 C (x - mean(C))
alpha = w + (1 - sum(w)) / K
```

其中正式部署`K=24`，LOO证据每折`K=23`。正常非零video下`sum(alpha)=1`；若causal representation
精确为零（no-video或phase-constant），直接令`alpha=0`，完整输出必须是template-A/zero-B identity。
exact language只通过frozen joint video innovation决定从画面读什么，不作为独立coefficient或LoRA
value，因此没有language-only bypass。

### 29.3 完整topological LoRA重构

每个expert仍按既有layout tokenized为`T[k,168,16,512]`。对每个chunk分别保存其有效值RMS
`s[k,c]`和unit-RMS方向`U[k,c,:,:]`。视频坐标只进行：

```text
D[c] = sum_k alpha[k] * U[k,c]
direction[c] = unit_rms(D[c])
log_scale[c] = clamp(sum_k alpha[k] * log(s[k,c]), expert_min[c], expert_max[c])
token[c] = direction[c] * exp(log_scale[c])
```

padding继续mask，随后用同一layout detokenize为完整38-target rank-16 public LoRA。静态experts提供的是
训练形成的policy-effective basis，不是第二套部署LoRA；每个episode只产生和挂载一套最终LoRA。
chunk-wise scale envelope是expert manifold坐标的一部分，不是失败checkpoint后的global scale、B-only
residual或confidence gate。one-hot coefficient必须精确重建对应expert；zero coefficient必须精确identity。

### 29.4 CPU LOO证据与边界

artifact=
`runs/outputs/pi05_expert_manifold_causal_barycentric_loo_step2000_cpu_20260809/analysis.json`。每个fold完全
拿掉一个task及其expert，仅用其余23个basis预测held task的50条视频。直接raw-factor affine的
effective target cosine为`.38838`，但norm仅`1.740`，证实近正交experts线性相消，故不采用。
topological direction/log-scale重构给出：

| arm | target cosine median | LoRA norm median | stable rank | top singular energy |
| --- | ---: | ---: | ---: | ---: |
| correct | `.38302` | `3.84385` | `1.15056` | `.89540` |
| phase-shuffled proxy | `.18539` | `3.82694` | `1.18181` | `.87672` |
| reversed | `.09900` | `3.82310` | `1.21235` | `.86204` |

correct相对reversed/shuffled margin=`.28403/.19763`；16 coordinates始终active，correct top4 coordinate
energy=`.27048`，与expert约`.26`同档。该证据同时解决当前首要的task direction与energy形态，而且
顺序破坏明确远离held expert。它仍不是closed-loop：LOO只模拟unseen task，16-slot shuffle不是formal
raw-frame shuffle，Goal/Long若干task margin弱，不能据此宣称达标。

### 29.5 实现、证据门与后备路线

1. 唯一canonical runtime原位替换learned address-binding decoder；旧model/trainer/checkpoint只由Git和
   artifacts保留，不并行执行。首版没有meta optimizer或Writer checkpoint，identity由固定basis资产定义。
2. CPU必须覆盖basis/task identity、one-hot exact expert、zero/phase-constant identity、deterministic
   solve、coefficient sum、scale envelope、完整LoRA shape/finite及ordered/reversed不同。
3. clean/push后只做单张live空闲A40的8-task online feature→LoRA cache→release→rollout smoke；不需要
   六卡训练profile。通过后才允许全新strict correct400。
4. correct400必须同时提高absolute、breadth和held LoRA task separation才继续。达到可信候选后，用
   same/wrong/shuffled/reversed/no-video的严格raw-frame paired five-arm裁决视频因果性。
5. 若闭式坐标方向正确但held interpolation不足，下一单变量才是在同一24-dimensional coordinate target上
   训练小型video coefficient reader；不恢复129万坐标hyperdecoder。few-shot只在one-shot same-task
   方差成为最早限制时加入，现有1/3/5-shot proxy几乎持平，首版继续one-shot。

### 29.6 Canonical实现封存（2026-08-09）

clean pushed`1d9d030`已经完成29.5第1--2项。唯一活动config为
`configs/pi05_video_expert_manifold_causal_barycentric_v1.json`；evaluation显式接收固定expert bank与
feature cache，不再接收learned Writer checkpoint。旧trainer、checkpoint和learned decoder owner已删除，
没有并行可执行版本，也没有meta optimizer、scheduler或可选择的Writer checkpoint。

CPU合同为全仓`180/180`通过，另对真实24-basis逐项只读：24个one-hot coefficient的完整expert最大
绝对重建误差`2.235e-8`，zero representation逐tensor exact identity，affine coefficient sum最大误差
`1.192e-7`，24/24 demo0 ordered/reversed coefficients不同，所有完整LoRA finite，Writer parameter数为0。
architecture guard无hard violation或parallel family，active source相对前一canonical净删941行。
本29.6 CPU封存时formal状态有意保持`blocked_until_live_a40_online_smoke`；这些CPU证据只解封单卡在线
工程smoke，不构成validation性能或视频因果性成绩。后续状态由29.7覆盖。

### 29.7 Online工程门与formal seal（2026-08-09）

implementation commit`3c8ce25`已在live空闲`gpu02:0`完成validation8×1-state纵向smoke，唯一root为
`runs/outputs/pi05_expert_manifold_causal_barycentric_online_smoke_gpu02_3c8ce25_20260809`。输入为correct/
without-replacement的一条action-hidden video；8套唯一FP32 LoRA、8 cache entries、2个batch4和3个
rollout workers均首次完成，0 retry/failure/OOM/nonfinite，teacher action/state/reward/terminal reads均
为0。Writer/encoder随后释放，source policy原位复用且没有reload；GPU自然释放。`1/8` success只证明
execution，不进入性能比较。

8套LoRA全finite，norm/stable-rank/top-energy中位=`3.9802/1.1555/.89243`，16/16 coordinates active，
top4 coordinate energy=`.27103`，cross-task effective cosine中位=`.69277`，nearest step2000 expert
cosine中位=`.65624`。相对已拒绝learned Writer的`.94197/.12734`，闭式重构确实产生更分离且更落在
expert manifold上的held LoRA；但8 rows不能证明absolute、breadth或视频因果性。

精确smoke evidence现已写入canonical config，formal状态由29.6的临时blocked门切为`sealed`；对真实
24-basis、train24×50 cache、canonical video data和validation8 panel的`require_formal=True`检查通过。
下一门严格保持29.5：从clean pushed frozen worktree做fresh correct400；若absolute、breadth或400-LoRA
task separation不成立，不运行其余五臂，而先定位representation、coordinate solve、expert support或
topological reconstruction的最早断点。正式branch/worktree/root、设备与exact command已预注册在
`task_plan.md`的Causal Barycentric strict correct400 launch合同。

## 30. Causal Barycentric负裁决与Policy-Effective编译接口（2026-08-09）

### 30.1 正式结果与排除项

29.7解封的strict correct400已完整结束为`63/400`、breadth=`5/8`；逐task按Spatial/Object/Goal/Long
为`[0,6]/[38,0]/[0,17]/[1,1]`。400 unique state/video/LoRA rows、72 jobs、18 workers、forbidden-read
和资源释放合同全部成立。相对same-video source/addressless panel的gained/lost=`46/31`、exact
`p=.1100`，不构成可靠提升；相对address-binding `75`为`27/39`。因此该candidate正式停止，不运行
其余五臂，也不通过结果选择某个task expert或video。

full400 LoRA norm/stable-rank/top-energy中位=`3.95796/1.15488/.89374`，16/16 coordinates active，
top4 coordinate energy=`.27061`，q/v/action B-column cosine=`.71200/.74440/.35062`。这些与step2000
experts的`4.21249/1.12877/.90846`及rank-coordinate形态同档，故当前失败不是低能量、近rank1、
inactive rank或全列共线。坐标反演误差足够小，可从400个生成LoRA可靠恢复其24维barycentric状态。

### 30.2 factor manifold并不保持policy update

29节要求对每个chunk分别混合expert A/B factor的unit direction和log scale。即使暂时忽略该
normalization，分别线性组合两因子也满足：

```text
A(c) = sum_k c_k A_k
B(c) = sum_k c_k B_k
B(c) A(c) = sum_k c_k^2 B_k A_k + sum_{k != j} c_k c_j B_k A_j
```

目标expert manifold真正有意义的对象却是：

```text
DeltaW_target(c) = sum_k c_k (B_k A_k)
```

两者连同同expert项的权重都不同；chunk direction归一化和log-scale插值只会增加非线性。one-hot时
交叉项消失，所以one-hot exact regression没有测到此错误。部署400个query的effective abs support中位
`13.75`、negative coefficient count中位`6`，正是交叉项最严重的区域。最终same-task不同video、
cross-task、task-mean effective cosine中位=`.98808/.68514/.69685`，仍远高于真实experts跨task
`.09996`。因此最早剩余断点不是“video能否区分顺序”，而是
`expert coefficients -> policy-effective rank-16 update`没有语义守恒。

### 30.3 下一单变量：Policy-Effective Barycentric Topological Writer

保留29.2的phase-centered causal video representation、ridge `.3` coefficient rule、step2000 expert
bank、train24×50 cache、one-shot输入、38 targets和public rank-16。唯一变化是每个target先构造：

```text
DeltaW(c) = sum_k c_k B_k A_k
```

直接线性组合虽然语义正确，但真实400 queries的effective norm中位只有`2.220`，是expert中位的
`.527`；这会重新引入已知的幅度稀释，故CPU门拒绝pure affine版本。选定重构改为每个policy target
分别混合unit-Frobenius effective direction，并对该target的expert log-Frobenius norm作affine插值与
train24 envelope限制。它不是事后global scale：38个target各自的direction与量纲都由同一24维video
coordinates决定，one-hot退化为该expert effective update，zero仍显式返回template-A/zero-B identity。

随后把每个target的effective matrix压回一套rank-16 factors。CPU门从24 experts的left/right covariance
求共享子空间并比较内部rank `16/32/64/96/128`；选定`96`，保存
`C_k = U^T B_k A_k V`，online只混合最多`96×96` core并作best-rank16 SVD。factor gauge用template-A在
动态row space中的orthogonal Procrustes orientation及train-expert A-RMS静态尺度固定；它不改变`BA`、
不提供静态policy update，也避免SVD任意gauge产生不健康factor幅度。

CPU artifact为原correct400 root下
`policy_effective_compiler_feasibility_full400_rank128_v2.json`。rank96对24 experts的global captured-energy
中位/最小=`.99677/.99331`；对400个选定direction+log-norm targets经public rank16后的captured-energy
中位/最小=`.99365/.99065`，对应cosine中位/最小=`.99682/.99532`、relative-L2中位`.07969`。8个task各
一个full-span exact compression样本的captured-energy中位=`.99523`，说明rank96相对不可压缩上界只损失
约`.16%`能量。生成effective norm中位=`4.155`、expert ratio中位=`.986`；pure affine版本只有`.527`。
最差单target是低总能量的`action_in_proj`，rank64时captured-energy中位`.9578`，但global rank96门仍
满足。该证据通过实现门；实现仍原位替换29节compiler，不保留并行可执行family。

### 30.4 与时序reader、few-shot和v6的顺序关系

现有correct/reversed/shuffled feature proxy已证明时序信号存在。新增CPU reader反事实进一步显示：
contrastive coefficient reader可令held target direction约`.394/-.392/-.008`，但correct norm ratio仅
`.106`；rectified prototype reader几乎把reversed置零，却仍不能提高held direction或解决迁移。
所以时序识别与生成足够强、可泛化的policy update是两个接口。本轮只修后者；同时改reader会使闭环
结果无法归因。只有effective compiler闭环absolute过门，才在同一compiler上引入order-negative credit并
跑raw-frame five-arm/no-video。few-shot仍等待one-shot same-task方差成为最早限制；恢复v6只在该单变量
候选失败后作为后续初始化/先验选择讨论，不在本轮混入。

## 31. Policy-Effective canonical实现与CPU运行门（2026-08-09）

30节选定的compiler已在唯一Writer owner内原位实现。旧
`configs/pi05_video_expert_manifold_causal_barycentric_v1.json`及raw-factor deployment compiler删除，
新authority为`configs/pi05_video_expert_manifold_policy_effective_barycentric_v1.json`；旧方法只由Git和
formal artifacts保留。每个target预计算rank96 left/right energy subspaces、24个effective cores、exact
expert Gram与norm；online以同一coefficients混合direction/log-norm，core SVD截rank16，再用template-A
row-space Procrustes与train-expert geometric-mean A-RMS选择纯gauge。该gauge不改变`BA`，zero coefficients
显式覆盖为template-A/zero-B。

全仓`182/182` CPU tests与真实资产inspector通过。artifact=
`runs/outputs/pi05_expert_manifold_policy_effective_cpu_real_assets_20260809/analysis.json`：Writer 0 learned
parameters、persistent buffers=`68,863,192` bytes，CPU build=`2.33s`、batch24 compile=`.85s`，zero identity
逐tensor exact。24 one-hot experts的effective cosine中位/最小=`.99838/.99665`；train24 demo0对未压缩
intended target为`.99836/.99657`，ordered/reversed coefficient L2最小`1.268`。

因子健康度也闭合：demo0 norm/stable-rank/top-energy中位=`4.179/1.125/.910`，A/B RMS=
`.01891/.00846`，q/v/action B-column cosine=`.815/.813/.455`，16/16 coordinates active。train24 demo0
不同task effective cosine中位`.203`，不再是旧learned/raw-factor Writer的公共方向。这里仍只是CPU机制
证据，不是validation closed loop。专属工程门随后由clean pushed`321bded`在live空闲`gpu02:0`通过：
validation8×1-state的feature→LoRA cache→release→rollout得到8 unique rows/LoRAs/cache entries，3 workers
全exit0、0 retry/failure/OOM/nonfinite/forbidden reads；generation wall=`11.070s`、peak allocated=
`10,576,896,000` bytes，Writer释放后source policy原位复用，GPU自然释放。root=
`runs/outputs/pi05_expert_manifold_policy_effective_online_smoke_retry1_gpu02_fb5b367_20260809`。`1/8` success
只作execution smoke；精确evidence写回后formal=`sealed`，下一步才是预注册小规模strict correct panel。

## 32. Policy-Effective correct80裁决与hard-route support判别（2026-08-09）

### 32.1 预注册screen结果

31节candidate在validation8×states0--9、correct、seed7、teacher video无放回的固定screen得到
`15/80`、breadth=`5/8`；逐task Long/Goal/Object/Spatial=`[1,0]/[0,2]/[8,0]/[1,3]`。36 jobs、80
unique state/video/LoRA rows、9 workers与信息墙均完整，故这是有效的科学负结果。它低于预注册
strong门`28`和ambiguity门`22`，不得扩跑160/400或五臂。

相对exact-same-video raw-factor compiler为`6 gained/3 lost`，只净增3且`p=.5078`；相对source为
`13/7`、`p=.2632`。相对v6-fast same-state/different-video为`5/18`、`p=.01062`。因此修复
effective algebra带来方向正确的小改善，但不能解释当前与143上限之间的巨大差距。

### 32.2 有效空间根因收缩

80套LoRA在38 targets上的exact effective`BA`几何为：norm/stable-rank/top-energy中位=
`4.148/1.234/.847`，A/B RMS=`.018909/.008413`，16 coordinates active，q/v/action B-column cosine=
`.610/.626/.372`。它与matched raw-factor输出的cosine中位仍`.958`、relative-L2=`.302`、norm ratio=
`1.055`。rank96/public-rank16不是瓶颈，cross-term错误也只是次要效应。

same-task不同video/cross-task/task-mean cosine中位=`.989/.703/.712`，最近train expert cosine中位=
`.641`；每task video-centered effective variance仅`.56%--2.54%`。路由已能找到语义合理expert：
Object-1最接近chocolate-pudding-to-basket并有`8/10`，Object-3最接近tomato-sauce-to-basket却为`0/10`。
这把剩余不确定性分成两个可区分解释：

1. broad signed soft mixture稀释了一个本来能跨对象迁移的expert；
2. 即使语义最近的train expert也不能在held object/scene上执行，24-task bank只定义训练任务局部方向，
   不构成held-task policy support。

### 32.3 下一唯一单变量

下一candidate保持phase-centered causal representation、ridge`.3`、step2000 bank、train24 cache、
exact-language+one action-hidden video、38 targets、public rank16、policy-effective compiler和zero identity。
唯一变化是在正常非零video下把24维affine coefficients确定性变为其最大坐标的one-hot；该one-hot经
现有compiler近似重建对应expert。选择完全来自视频representation，不读取task ID、validation action、
rollout outcome或文件身份。

它首先是support判别，同时若闭环通过也可成为稀疏Expert-Manifold起点。必须在CPU验证zero identity、
24 one-hot重构、finite/full topology、不同ordered/reversed输入可选择不同expert；然后完成单卡online
generation/cache/release/rollout smoke，最后只跑与32.1完全相同的80-row panel。若没有实质超过15并
维持/提高breadth，停止在24-expert mixture内部继续调scale、temperature、top-k或rank，转向以历史v6
高性能表示作初始化、但仍由视频提供唯一dynamic value的可迁移policy-effective Writer。若明显提高，
才设计可微sparse reader与order-negative训练。few-shot继续后置：当前same-task输出近乎相同，且此前
3/5-shot proxy对one-shot仅`.39379/.39558`对`.38820`，平均更多视频不能修复当前最早接口。

### 32.4 Hard-route canonical实现与CPU门（2026-08-09）

32.3已作为唯一runtime原位实现：新authority=
`configs/pi05_video_expert_manifold_hard_routed_policy_effective_v2.json`，implementation=`1619631`。旧soft
config不保留为并行可执行family。对任意非零video，先计算原ridge`.3` affine scores，再按最大signed
score确定性选择一个expert；tie固定最低ordinal。部署coefficients恰有一个1，其余为0。affine scores只
能由`affine_coefficients()`用于审计，runtime没有soft/top-k/temperature/scale模式。zero或phase-constant
representation仍显式返回全零coefficients和template-A/zero-B identity。

真实资产CPU artifact为
`runs/outputs/pi05_expert_manifold_hard_routed_cpu_real_assets_20260809/analysis.json`。24个train centroids与
每task 50条、共1,200条ordered videos全部self-route，selection histogram严格为每expert 50次；top1-
top2 affine margin中位`.63037`。全部video反转时1,200条均改变选择；每task固定随机phase permutation时
699/1,200=`58.25%`改变。故reader在训练支撑上同时具有task识别与强order sensitivity，不是恒定路由。

24 one-hot LoRA经相同rank96/public-rank16 compiler后，912个target级effective cosine中位/最小=
`.998982/.961962`；0 parameters、persistent buffers=`68,863,192` bytes，zero exact且所有states finite。
已有correct80 soft run保存的系数表明held panel将选择11个experts，逐task路由不是单一全局expert；其
top1-top2 margin中位`.01932`，远小于train videos，量化了held representation接近task边界这一风险。

本CPU门满足工程解封条件但不提供closed-loop成功证据。formal状态固定为
`blocked_until_live_a40_online_smoke`：只允许下一步先在clean pushed/frozen authority上做一张空闲A40的
validation8×1-state correct纵向smoke。只有generation/cache/release/source-policy reuse和信息墙全部
成立后才seal，并运行与32.1完全一致的80-row panel；不得在看到结果前加入其他变量。

### 32.5 Online smoke、真实hard-route确认与held边界风险（2026-08-09）

clean pushed launch`12c8d1e`在live空闲`gpu02:0`完成validation8×1-state correct/without-replacement
工程smoke，root=
`runs/outputs/pi05_expert_manifold_hard_routed_online_smoke_gpu02_14495d9_20260809`。8个唯一video各生成一套
唯一LoRA/cache，3 workers全部attempt1/exit0，0 retry/failure/OOM/nonfinite，四类forbidden reads为0。
Writer/encoder在cache后释放，resident source policy原位复用且没有reload；generation wall=`10.497s`、
peak allocated/reserved=`10,576,896,000/11,238,637,568` bytes，结束后GPU为0MiB/P8。`0/8`不能解释性能。

为了验证线上确实走hard path而非仅schema改名，CPU posthoc把8个cache LoRA与同一compiler的24个one-hot
输出逐一比较。artifact=`hard_route_online_smoke_route_audit_v1.json`。8/8 nearest one-hot effective cosine
最小`.999999799`，nearest与second的factor-relative-L2 gap最小`.38936`，因此选择无歧义；共覆盖7个
experts。7/8选择与旧soft correct80保存的argmax一致。唯一不一致是Long-2 state0：旧ordinal12/13 score=
`.136743/.136079`，margin仅`.000664`，live选择13。这不是soft混合残留，而是held representation位于
argmax边界、微小跨run数值差异可翻转专家的直接证据。

formal现已由CPU门的blocked状态切到`sealed`。下一correct80仍只改变soft→hard这一变量，panel、video
schedule、state/env/policy RNG与32.1一致。判据预注册为：score`>=28`、breadth`>=5`且相对soft15的paired
net gain至少10为strong，通过后进入correct400；score`22--27`且breadth`>=5`只扩到states0--19的160-row
消歧；score`<=21`或breadth`<=4`则判24-expert support不足，停止top-k/temperature/scale/rank修补并转向
v6先验的可迁移Writer。任何route flip都作为稳定性证据报告，不新增confidence/abstention gate。

### 32.6 Strict correct80结果与24-expert部署字典淘汰（2026-08-09）

固定panel自然完成为`3/80`、breadth=`2/8`，逐task Long/Goal/Object/Spatial=
`[0,2]/[0,1]/[0,0]/[0,0]`。36 jobs、80 unique rows/LoRAs/cache、9 workers和信息墙全部闭合；三张A40
自然释放。该结果同时满足`score<=21`和`breadth<=4`，所以预注册裁决为reject，不扩到160/400，
不运行same/wrong/shuffled/reversed/no-video。

hard与32.1 soft screen严格共享80个state、env seed、policy RNG、teacher demo和frame-order seed。
paired retained/gained/lost/both-fail=`1/2/14/63`，净`-12`，双侧exact McNemar `p=.0041809`。该退化
已经足够明确，不可解释为80-row方差。artifact=
`hard_route_strict_screen_and_policy_effective_route_audit_v1.json`。

为排除hard实现错位，审计在全部38 targets的exact effective`BA`空间将每个cache LoRA与24个raw
step2000 experts比较。最近expert cosine中位/最小=`.998544/.997096`，nearest-second gap最小
`.35133`；共选择11个experts。79/80与32.2保存的soft coefficient argmax一致，唯一flip仍是Long-2
state0的ordinal12→13，旧margin`.000664`。因此hard path真实生效，且失败不能归给路由全塌缩或边界
不稳定主导。

更强的机制反事实来自Object-1：十条held videos全部选择ordinal10
Chocolate-pudding-to-basket expert，结果`0/10`；soft组合在相同十条却为`8/10`。Object-3十条全部选择
ordinal8 Tomato-sauce-to-basket同样`0/10`。单个task expert虽在自己的task random reset上有效，却不
提供held对象、布局或组合任务的可直接复用策略；soft mixture偶尔能合成有用的新方向，但整体15分也
远不足。因此24 experts仍可作为policy-effective监督目标，却不能继续作为部署时的hard、soft或稀疏
选择字典。

本节关闭Expert-Manifold内部的mixture-support分支：禁止继续尝试top-k、temperature、global scale、
confidence、rank或few-shot平均来修复它。下一方法仍属于Video-Conditioned Expert-Manifold总体目标，
但部署生成器必须是**v6-prior transferable policy-effective Writer**：从历史v6-fast已验证的动态视频
表示与生成轨迹出发，直接学习held可迁移的完整LoRA；task experts只提供稳定的policy-effective目标/
先验，不再被在线选择。具体结构、初始化映射和损失必须先由CPU与历史checkpoint合同确定，再开新的
GPU profile或训练门。

## 33. v6-Prior Policy-Effective Temporal-Ranking Writer（2026-08-09）

### 33.1 最早失效接口与最小结构修改

当前候选不再发明新的video encoder、latent topology或expert mixture。历史v6-fast macro400已经达到
strict correct=`143/400`，并在同一checkpoint内把ordered/reversed/shuffled差异从Procedure传到完整
LoRA；但task-complete相对old recipe把Procedure→effective-LoRA与Procedure→action传递压到约
`.42--.61`与`.34--.56`。hard-route又证明24个task experts不能作为held部署字典。因此最小可证伪假设
是：**v6上游已经含有可迁移语义与时序证据，失效发生在共享compiler把这些证据写入policy-effective
LoRA时；task experts适合监督这个接口，但不适合替代生成器。**

唯一部署路径恢复历史one-shot v6拓扑：exact task language加恰好一条action-hidden raw video，经过
task-grounded per-frame evidence、permutation-invariant Semantic Core、adjacent-transition Causal
Procedure、320-slot compiler和8个factor heads，直接生成完整38-target rank16 LoRA。部署不读取expert
bank，不做expert选择/混合、language-only residual、scale/confidence gate或第二套LoRA。

初始化固定为历史v6-fast macro400：

`runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729/checkpoints/step_00000400`

其600个Writer tensors和旧v6 schema只允许经本方法的load-only初始化入口进入全新optimizer、scheduler、
sampler与checkpoint合同，不能冒充历史exact-resume。CPU已只读确认参数ownership：semantic encoder/
Core/visual transition/Procedure/compiler/factor heads分别为`440/22/5/16/25/16` tensors、
`3,455,040/1,836,544/197,120/1,573,888/1,535,232/2,179,072` parameters。

本轮冻结前四个上游block，只训练compiler与全部factor heads，共41 tensors、`3,714,304` parameters。
这不是永久宣称上游最优，而是把第一次干预严格放在证据定位到的最早接口；若输出端目标和梯度均成立
却闭环不升，才允许重新打开更早接口。

### 33.2 Policy-effective expert监督

每个train task使用统一step2000 expert作为常量目标`E_t`。对任意generated LoRA `G`，比较对象不是
raw A/B factor，而是全部38 targets上的effective update `BA`。每个target的norm与inner product均用
rank16 factors精确计算，不物化大矩阵：

```text
||BA||_F^2        = sum((B^T B) * (A A^T))
<BA, Be Ae>       = sum((B^T Be) * (A Ae^T))
cos(G,E)          = sum_j <G_j,E_j> /
                    sqrt(sum_j ||G_j||^2 sum_j ||E_j||^2)
```

correct expert loss由`1-cos(G_correct,E_t)`和对global effective norm ratio的bounded Smooth-L1组成。
它同时约束方向与能量、对LoRA gauge不敏感；不得替换成factor MSE、rank/正交loss或只看q/v的漂亮谱。
step2000 experts的global effective norm为`3.02--5.82`，能量中约`78.2--87.9%`在q、
`11.7--21.2%`在v、`.29--.57%`在action I/O；positive functional action loss继续负责真实policy敏感度，
避免小energy action targets被几何统计误当成不重要或被手工等权放大。

### 33.3 正确、乱序、倒序和错误视频的训练语义

每个macro仍覆盖train24，每task一条correct video和同task跨episode B20 action queries；先task内mean，
再24-task等权。correct臂保留完整positive functional PI05 loss，video/action episode继续错开。

同一correct视频的真实frames另构造reversed或seeded shuffled顺序；第三种negative按封存的cross-suite
对称映射提供wrong-task video，但Writer language与policy language始终保持当前action task的exact
language。三种negative按确定性macro/task schedule轮换；same-task-other不作为negative，50条同task
videos继续通过无放回correct schedule共同逼近同一个task expert，因而其预期是鲁棒而不是被排斥。

negative不运行“最大化action error”，也不被强制到任意恶意LoRA或无限norm。它只进入bounded
policy-effective ranking：

```text
margin = cos(G_correct,E_t) - cos(G_negative,E_t)
L_rank = softplus((m - margin) / tau) * tau
```

当正确臂已比negative更接近有效expert方向达到margin后，梯度自然饱和。这样不能只靠language令所有
视频得到同一task LoRA，也不能只把negative能量放大；correct同时必须通过absolute action loss、expert
direction和expert norm三门。reversed/shuffled复用同一批per-frame frozen evidence，只重算adjacent
transition、causal Procedure和下游compiler；wrong video需要独立action-hidden frame evidence，但不读取
其language、action、state、reward或task ID作为Writer value。

第一轮总目标固定为：

```text
L = L_positive_functional + lambda_expert * L_correct_expert
                          + lambda_rank   * L_counterfactual_rank
```

`lambda_expert/lambda_rank`不按validation outcome sweep。独立profile先在预注册真实train24 macro上记录
三项未加权的compiler+factor gradient norm，再按固定目标比例各不超过positive gradient norm的`.25`
封存一次常数；formal不得在线自适应或按closed-loop改权。若某auxiliary在初始化已满足、梯度为零，则不
人为放大。optimizer使用全新AdamW和低LR短continuation；不加载macro400 Adam moments。

### 33.4 历史CPU、online与正式裁决门（由33.7覆盖）

实现前CPU必须通过：

1. macro400 600 tensors逐名/shape严格加载，除明确记录的历史compiler命名映射外不允许missing/extra；
2. frozen/trainable ownership精确为前四block/后两block，source policy全冻结；
3. effective norm/inner/cosine与显式小矩阵`BA`数值一致，gauge变换前后loss不变且梯度finite；
4. identical correct/negative state给零ranking advantage，ordered/reversed/shuffled真实输入能产生不同
   Procedure和LoRA；same-task correct schedule仍覆盖50 videos；
5. warm-start生成与历史portable macro400 LoRA manifest中的至少一条同task/demo输出一致；若只因
   BCI浮点kernel不能逐byte一致，必须达到逐tensor数值门并说明，不能直接进入训练；
6. checkpoint包含Writer、optimizer、scheduler、sampler/video/counterfactual cursor、六rank RNG和完整
   frozen-block/initialization schema，fresh与exact-resume不可混用。

随后只在clean pushed frozen worktree做一张空闲A40的warm-start reproduction generation smoke和六卡最长105-frame
train macro profile；profile依次完成fresh0→1、exact-resume1→3和独立contiguous0→3，并封存上述两个
loss weight。正式从macro400 warm-start fresh训练50 macros，保存10/25/50；step0/10/25/50在同一当前
without-replacement seed7 strict80 panel上全部评测，避免用different-video历史143选择点。三个训练点
随后全部跑paired correct400，不以functional loss挑winner。

只有single checkpoint满足以下任一条件才做完整same/wrong/shuffled/reversed/no-video：correct严格超过
`150/400`；或correct至少不低于当前同schedule step0且breadth不降、同时多个task净增并有可信上升趋势。
若三点均低于step0或只发生单task换手，则停止，不用更长训练、loss sweep或解冻上游挽救。若correct提高
但顺序/错误margin仍弱，下一单变量才调整counterfactual credit；若margin提高而absolute下降，则降低的
是该目标本身而不是“训练不够”。最终仍只认同一single checkpoint的strict paired闭环结果。

### 33.5 历史实现封存（2026-08-09，runtime事实保留）

训练侧实现已由clean pushed`dd57edc`封存。历史v6 macro400仍按600 tensors strict load；前四block
冻结为483 tensors、`7,060,992` parameters，compiler+factor heads为41 tensors、`3,714,304`
trainable parameters。runtime按6 ranks×4 tasks完成train24等权宏步，只允许DataLoader workers=`0`，
一次flat all-reduce后更新；checkpoint保存完整Writer、optimizer、scheduler、cursor与六rank RNG。

CPU与真实数据门为全仓`215 passed`、24 tasks、206,346 query rows；profile macro49恰好覆盖每task
B20、全局480 unique rows，并包含最长105 sampled-frame视频。config当前状态为
`blocked_until_single_a40_warmstart_reproduction_smoke`，因此尚不能运行六卡gradient profile。rejected
hard-route evaluator/runtime将在下一提交原位替换，完成唯一部署路径和单卡复现smoke后才允许改变该
状态。

### 33.6 历史evaluator复现边界（2026-08-09，batch1裁决已撤回）

部署替换已由clean pushed`bca3f6d`完成。旧hard-route config、online expert-bank/feature-cache资产入口、
HardRouted Writer类和拓扑专属测试已从canonical runtime删除；旧参数即使由历史命令传入也会fail closed。
task-expert evaluator和feature几何工具只保留其仍然有效的训练监督/历史分析职责，不再参与Writer部署。

新adapter schema=`ember_pi05_v6_prior_eval_adapter_v5`，episode schema=
`ember_pi05_v6_prior_episode_v5`。Writer asset允许两类且显式区分：configured historical v6 macro400是
`historical_v6_macro400_load_only`、method macro0；本方法后续`macro_XXXXXXXX` checkpoint是
`v6_prior_trained_checkpoint`并携带真实method macro。两类都严格检查600-tensor state、source contract、
config/objective/ownership和checkpoint schema；不做文件内容hash。

online路径用当前frozen source policy构造同一`CompleteLoRAWriter`，逐名strict-load历史或本方法state。
每个episode只从raw HDF5读取一条固定stride视频；current task language始终不变。reversed/shuffled只改变
frame content的展示顺序，position indices保持新的顺序坐标，因此不是把原始时间戳一起打乱后泄漏原
顺序。batch内可含不同长度视频，offsets显式切分；batch1的无batch输出和batchN的leading batch维均按
合同转成76-tensor LoRA。no-video提前返回template-A/zero-B，不tokenize language、不读frames、
不调用Writer。episode LoRA写入通用cache后删除Writer/store/tokenizer并保留同一source policy做rollout。

全仓`211 passed`；真实asset只读解析确认historical state为600 tensors、12,064,064 values，validation8
映射为8个one-shot requests且deployment expert-bank reads为0。真实CLI prepare也已通过，但这些仍只是
CPU/合同证据。单卡A40 reproduction smoke必须同时验证完整cache/release/rollout链路和batch-vs-single
direct forward数值等价；不能仅因checkpoint能load或8个rollout能结束就seal。该证据写回前
`evaluation.formal_status`与`gradient_profile.status`继续blocked，六卡profile和训练不得启动。
该比较已接入historical-correct smoke runtime：每个batch写cache前逐episode重跑direct forward，state
names/shapes、finite与max-abs门任一失败都会中止，不允许事后人工补判。

首次`30b2ccf` A40 smoke确实在cache前触发该历史门。独立诊断中single-direct repeat逐元素差为零，而
同一样本复制batch8与8个异构样本batch8相对single-direct的max-abs均为`.001953125`、mean约`4.70e-5`，
定位为BF16 batch-shape数值路径。随后把model固定batch1的结论已经由33.7明确撤回；本段只保留故障定位，
不得恢复其执行决定。失败root仍保留0 cache/0 rollout证据且不resume。

### 33.7 Throughput-first修正与当前正式门（2026-08-09）

33.6最后一段把`.001953125`级BF16 batch-shape低位差异提升为scientific reproduction变量，并据此固定
model batch1、每个staged LoRA额外重跑single direct forward。Owner随后明确裁决：所有为了这种底层
微小误差降低效率的行为都不可取，EMBER必须吞吐优先、尽可能利用A40显存并尽快获得真实闭环证据。
因此33.4--33.6中的batch1、`1e-5` direct comparison、重复forward和相关blocked status自本节起撤回。
失败root与`batch_equivalence.json`只保留“不是串样/padding/randomness”的诊断价值。

该修正不改变33.1--33.3的科学假设、information wall、one-shot、v6 macro400 initialization、冻结边界、
train24/B20/full24 objective或strict evaluator。改变的只是等价实现与执行效率：

1. Writer generation batch从8起在真实A40上扫`8/16/32/...`。所有候选处理同一个由最大候选确定的
   32-request longest-first panel和同一总sampled frames，只改变实际forward分批；选择LoRAs/s最高、
   最长异长video连续运行稳定且有显存余量的点。接受正常BF16 kernel/batch/reduction low-bit roundoff。
2. 不再逐episode重复direct Writer forward，也不逐tensor比对。只门禁shape、finite、信息墙、明显
   cross-sample污染、cache identity/dtype、OOM和runtime合同。
3. 生成LoRA保持历史template原生storage：72个BF16 tensors、4个F32 tensors、总`2,641,920` bytes/
   entry；不统一`.float()`到`5,148,672` bytes。一个batch集中nonblocking D2H并按device只同步一次。
4. B20在单physical batch可容纳时直接求functional output gradient，不创建76个FP32 accumulation
   buffers；只有真实microbatch才保留FP32 gradient accumulation，避免小而policy-effective的梯度丢失。
5. correct effective alignment只计算一次并由expert/ranking共享；task scalars、profile norms和step
   metrics批量host transfer，删除热路径逐tensor finite scan与显式重复CUDA sync。宏步clip norm、loss/
   metric finite及真实rollout继续提供必要故障证据。
6. action DataLoader默认2个spawn persistent workers、prefetch2。sampler由global step/rank确定且dataset
   `__getitem__`无随机性；serial、prefetched和prefix+resume row序列必须一致。profile若GPU仍等待data，
   再实测更高worker/prefetch。

当前实现进一步把Writer offsets置于CPU合同边界，合并frame ordinal/order、task span和condition
ownership的重复host barrier，并对token packing做固定shape向量化；PI05 formal functional路径使用与原
forward同loss/LoRA leaf gradients的loss-only调用，删除纯日志型host sync。这些只改变执行路径，不改变
objective或允许的信息。

新的CPU门是：historical 600 tensors strict-load、frozen/trainable ownership、effective-BA数学与gradient、
one-shot/negative/no-video信息墙、native cache schema、DataLoader resume rows、真实validation8 inspect/CLI
prepare和聚焦/全仓测试。CPU门不要求batched与single逐元素同一。

单卡A40门分两段：先通过canonical `profile-writer-generation`子命令在同一loaded model上完成真实异长
video batch/VRAM/end-to-end wall sweep；再用选定batch从
fresh root完成validation8×state0 correct的8 video→8 LoRA→cache→Writer release→source-policy reuse→
8 rollout vertical smoke。要求0 retry/failure/OOM/nonfinite/forbidden reads、native cache dtype/bytes正确、
GPU自然释放；success count只作execution信息。config seal必须记录选定batch、候选吞吐比较、peak
allocated/reserved、redundant Writer forwards=`0`和release/reuse证据，不再记录direct max-abs门。
profile与vertical public入口都在任何模型load/worker spawn前要求目标卡无compute applications且为
`NVIDIA A40`；普通evaluator合计最多6卡。profile还要求clean pushed、validation/correct/
without-replacement、单卡单replica/generator及至少`8/16/32`真实候选，并在独立单卡worker里再次live
preflight和核对checkout。evaluation seal只能由两个retained roots的artifact assembler生成，要求各候选
完整fixed panel一致，不能手填。

六卡profile仍使用预注册macro49和train24×B20=480 queries，封存两个auxiliary weights并完成fresh0→1、
exact-resume1→3、contiguous0→3；同时profile physical microbatch、loader/prefetch、GPU utilization、input
wait和step phase wall。scientific metrics/cursor/RNG/optimizer语义必须一致，正常parallel roundoff不作
逐bit门。

config合法状态固定为：初始全部blocked；单卡vertical artifact seal后仅evaluation sealed并令gradient
ready；六卡gradient artifact seal后同时固定auxiliary weights并令profile ready；fresh/resume/contiguous
artifact比较通过后才令formal ready。后两次状态转换也必须由结构化verifier完成，不能靠人工改status跳门。

formal从同一historical macro400 load-only状态建立当前schedule macro0，训练0→50并保存10/25/50；
macro0/10/25/50先跑固定correct80 screen，但四点都必须跑paired correct400，不能由screen或loss选点。
如果多个task共同上升且50仍有可信趋势，可按同合同续到100/200并及时correct400。single winner严格
`>150/400`或形成可信共同提升时跑correct/same/wrong/shuffled/reversed/no-video。每次与macro0、历史
143、v5.2 old和v6 recipe交叉结果比较per-task gained/lost、breadth、churn与Core→Procedure→BA→action
传递；只改最早失效接口，不因单次结果整体换架构。

### 33.8 单卡吞吐与纵向链路实证（2026-08-09）

clean pushed/frozen `ded0c80`在live空闲`gpu02:0`完成canonical fixed-panel profile。三个候选严格共享
32 requests、1093 sampled frames和最长67-frame样本；batch8/16/32的实际forward分组为
`8×4/16×2/32×1`，两次measured repeat合计wall=`70.220/70.710/70.606s`，吞吐=
`.911427/.905107/.906432 LoRA/s`。peak reserved分别为
`12,824,084,480/12,847,153,152/12,847,153,152` bytes，全部稳定且余量大。batch8按预注册规则最快；
大batch没有吞吐上升，因此未把“占更多显存”误作性能优化，也没有为低位数值选择batch。

同提交fresh vertical smoke以batch8完成validation8×state0 correct：model load=`111.469s`，8套LoRA
一次forward生成，generation=`10.597s`、peak allocated/reserved=
`11,651,564,544/12,811,501,568` bytes；Writer随后释放，source policy原位复用且未reload。8 rows、
single attempt、0 retry/failure/OOM/nonfinite/forbidden reads，总wall=`325.540s`、rollout window=
`196.816s`，进程退出后GPU回到0MiB。cache仍为72 BF16+4 F32、每entry `2,641,920` bytes。
`4/8` success只作execution smoke，不是新strict性能。

当时的evaluation-smoke assembler已从profile与vertical retained roots重建seal；该退役入口现由Git和
artifact保留，不再属于canonical runtime。该实证只关闭“当时实现是否能高效完成完整闭环”
的不确定性；冻结上游、expert辅助和temporal ranking能否共同超过143/150仍完全待六卡训练与paired
closed-loop裁决。下一门是结构化gradient/resume verifier和macro49六卡profile，不能直接跳到formal。

### 33.9 六卡artifact seal与无扰动吞吐观测（2026-08-09）

六卡gradient与fresh/resume/contiguous的结构化verifier已经实现并通过CPU回归，但尚未运行真实六卡
artifact。gradient evidence只能从一个完整retained root重建：canonical tracked config、clean pushed
Git、6-rank A40/NUMA/affinity映射、macro49、train24×B20=`480/480` unique queries、24条deterministic
teacher demo/counterfactual records、8/8/8 negative counts、未加权block norms、推荐weights、显存和
零OOM/nonfinite必须同时成立；task/path/bytes/demo frame count还要回查同frozen commit的target manifest和
真实HDF5 metadata。只改status、使用stale tracked config、复制外部config或手写summary均不能解锁profile。

resume evidence要求gradient commit是profile commit的strict ancestor；fresh0→1+same-root resume1→3
与independent contiguous0→3必须共享科学contract和拓扑。只读checkpoint inspector核对manifest size、
cursor/contract、600 Writer tensors、41 trainable tensors、6-rank RNG、AdamW `exp_avg/exp_avg_sq`、
scheduler/AMP；Writer与optimizer moments以及三宏步scientific metrics只要求预注册容差内等价，普通BF16/
parallel reduction low-bit差异不是失败。

吞吐观测不在每个video encode、policy forward、backward或all-reduce后增加CUDA同步。retained artifact只
借用原本宏步末端的一次同步，记录whole-step wall、`next(DataLoader)`累计input wait与peak allocated/
reserved。若这些证据显示data wait显著，才实测2/4 workers等候选；若显存或计算证据指向activation
checkpointing或physical microbatch，才做对应单变量profile。否则保持B20单物理batch、2 persistent
workers/prefetch2，避免为了更细的profile数字降低正式训练效率。

### 33.10 physical B20容量实证与logical-B20微批合同（2026-08-09）

33.9最后一句关于“否则保持B20单物理batch”的条件已被真实A40证据触发并覆盖。clean frozen
`a17805c`在当时空闲`gpu01:0,1,2,4,5,7`的3+3 NUMA拓扑运行macro49。默认allocator在第一条PI05
functional B20的Gemma MLP请求`606MiB`时OOM，PyTorch allocated=`42.29GiB`、reserved-unallocated=
`1.29GiB`、free=`395.31MiB`。一次预先有依据的`expandable_segments:True`重试把reserved-unallocated降到
约`157MiB`，但active allocated升至`43.43GiB`且仍无法取得`606MiB`。因此physical B20超过该A40所报
`44.42GiB`总容量的有效可用边界，碎片已经排除；不得继续allocator盲重试。两个root均在首条task
objective失败，只含contract/
invocation而无gradient/completion，不能seal、resume、合并或解释方法。

这不允许减少33.1--33.3的scientific batch。每task仍固定20条跨episode action queries，先task内mean再
train24等权，全局仍是`480/480` unique rows；positive/expert/ranking objective、video schedule、optimizer
和信息墙不变。physical implementation改为：

1. 从optimization seed、global task、task visit及完整20条`(demo_index, frame_index)`有序identity导出一
   个rank/phase-independent局部policy seed；不依赖全局CUDA RNG前序消费。它准确地是logical-panel
   keyed而非“单query重排后仍保持draw”的承诺，使用固定SplitMix64整数mix，不调用SHA/MD5。
2. 对logical B20一次定义20个独立exact Beta(1.5,1) flow times和20个独立Gaussian noises。每个physical
   slice重放同一个logical draw tensor并只取自己的offset，保证B16+4与B10+10使用同一query/draw集合。
3. 每slice得到对76个detached LoRA leaves的梯度；BF16/F16 leaves在FP32 buffer中按slice size/20加权累积，
   最终再转回leaf dtype并通过原chain-rule bridge传给Writer。loss也按相同权重合成。
4. 首选physical B16，因为仍只有两次policy forward；但clean frozen`eddba96`的真实六卡macro在第一条
   functional eager-attention统一OOM，请求`254MiB`时allocated=`42.49GiB`、reserved-unallocated=
   `1.25GiB`、free=`235.31MiB`。因此B16并未留下whole-step吞吐点，预声明的A-B-A分支不再适用。当前
   直接运行balanced B10+10；同logical panel、同keyed draws、同两次policy forward和同objective不变。
   B10成功即作为A40可行gradient point封存wall/input wait/peak及0 OOM/nonfinite，失败才重新打开
   policy activation checkpointing，不做allocator retry或宽batch sweep。

policy gradient checkpointing不是首选：现有`writer.activation_checkpointing`不覆盖OOM所在的frozen
PI05 Gemma，打开policy checkpointing需要改变当前eval/frozen执行路径并重算Transformer，可能比两个
physical forwards更慢。只有B10也无法形成稳定配置时，才把它作为独立、显式设计变量；不得暗中与
microbatch混合。`expandable_segments:True`失败retry只证明碎片减少仍不足，不固化为当前scientific/
runtime合同；B16与B10都从default allocator开始并在run contract记录实际观察值。

### 33.11 balanced B10 gradient seal（2026-08-09）

clean pushed/frozen `9c814ff`在live空闲`gpu01:0,1,2,4,5,7`完成logical B20/physical B10+10 macro49。
whole-step wall=`21.095109596s`、input wait=`.076318255s`（`.36%`），peak allocated/reserved=
`43,305,942,016/47,093,645,312` bytes；24 tasks、480/480 unique queries、最长105帧、8/8/8 counterfactual
和0 OOM/nonfinite完整。assembler从tracked frozen config、clean pushed Git、真实HDF5和六rank topology
独立重建通过，因此B10成为当前A40唯一可行physical implementation；不再扫workers4、microbatch或
allocator。

positive/expert/ranking在compiler的unweighted norm=`.0110556/.330800/.00967394`，factor=
`.105556/.663721/.0147533`。逐aux逐block`.25`门给出expert/ranking=
`.008355172068998324/.28570466890490887`；applied compiler fractions均`.25`，factor=
`.052536/.039932`。macro0 generated/expert effective norm mean=`140.52/4.182`且cosine=`.02196`；wrong
margin=`.00225`，reversed/shuffled仅`.000832/.000634`。这说明能量/方向和时序分离仍是实质缺口，但
不允许据内部指标宣告方法有效。下一证据严格是同一seal commit的fresh0→1、same-root resume1→3与
independent contiguous0→3，之后才启动formal并用closed-loop裁决。

### 33.12 B10 exact-resume profile seal与数值门修正（2026-08-09）

gradient seal的严格后继clean pushed commit为`5fbcb27`。第一次在
`gpu01:0,1,2,4,5,7`完成fresh0→1后，GPU0在释放后被其他用户取得；该root保留为完整macro1工程证据，
但没有混入比较。随后重新live比较两节点，只使用当时空闲的`gpu02:0--5`，按每张physical GPU所属
NUMA绑定形成合同允许的4+2拓扑，在两个全新root完成：

```text
runs/outputs/pi05_v6_prior_profile_resume_r6_lb20_mb10_5fbcb27_retry1_20260809
runs/outputs/pi05_v6_prior_profile_contiguous_r6_lb20_mb10_5fbcb27_retry1_20260809
```

resumed root为fresh0→1再same-root exact-resume1→3，contiguous root为独立fresh0→3；两者run contract
逐字相同，均为B10+10、workers2/prefetch2/default allocator、Ring/Simple、
`NCCL_P2P_DISABLE=1`和deferred-NCCL。三次invocation均exit0，两个root各3条metrics、macro1/3完整
checkpoint和macro3 completion；0 OOM/nonfinite/clip。contiguous三宏步总step wall=`61.367943s`、
input wait=`.203131s`；resumed合计=`64.449543/1.152898s`，额外时间主要来自新进程首个macro的loader/
kernel冷启动。steady-state macro3分别=`20.0175/19.6982s`且input wait均约`.0006s`，因此不扫workers4。
全体profile峰值allocated/reserved=`43,265,769,984/47,118,811,136` bytes，restart后reserved只一次增加
约24MiB后平台，不存在容量漂移。

metrics row在optimizer step前计算，所以macro1 row是warm-start基线，macro2/3分别反映一、两次既往
更新。generated correct norm mean从`140.973→138.738→136.066`，expert loss从
`1.79433→1.79036→1.78536`，说明小权重expert norm纠偏方向已生效；cosine始终约`.022`，ranking margin
随预注册negative panel轮换而波动，三步不能解释为方向对齐或性能改善。

resume与contiguous的cursor、checkpoint contract、6-rank RNG、scheduler、AMP以及559个frozen Writer
tensors均精确相等；全部41个trainable Writer tensors和82个Adam moment tensors通过逐tensor
`atol=2e-4, rtol=2e-3`。scientific metrics最大tolerance ratio=`.233773`。macro3 Writer最大绝对差=
`4.6033e-5`、global relative L2=`1.06393e-5`，其总差异只占macro1→3两步Writer更新L2的`1.023%`；Adam
一阶/二阶moment最大绝对差仅`2.6865e-6/1.1353e-8`，但因moment本身接近零，其relative L2为
`.007719/.005237`。

最初verifier错误地把Writer的`7.5e-6/1e-5` aggregate门同时套给Adam；这在所有语义状态和逐tensor
科学门均通过后产生false negative。按owner的吞吐优先数值政策，修正只发生在离线比较器，并改成与
状态语义一致的尺度无关门：Writer只硬门global relative L2`≤.002`，max-abs只记录；Adam对
`exp_avg/exp_avg_sq`分别要求symmetric norm ratio`≥.99`和cosine`≥.999`，raw max-abs/relative-L2只
诊断。macro3两moment的ratio/cosine为`.999632/.999970`和`.999820/.999986`，有明确余量；清零、明显
缩放或方向破坏仍会fail。schema、frozen exact、RNG/cursor/scheduler/AMP exact及逐tensor scientific
allclose继续fail-closed。该修正没有改变训练kernel、dtype、reduction、batch、objective或artifact，也
没有重跑GPU追逐低位一致。evidence schema升到v2，原三root重新组装通过并原样写入config，profile与formal状态现已
`sealed_from_live_a40_resume_profile_evidence`。

本profile只证明B10训练容量、吞吐和resume语义成立，不是新strict性能证据。下一步从新的clean pushed
seal commit创建formal frozen worktree，先跑同schedule macro0，然后fresh0→50并及时评测0/10/25/50；
不得用上述三步loss或energy趋势替代paired closed-loop裁决。

### 33.13 formal 0→50与whole-LoRA objective裁决（2026-08-09）

clean frozen `eff15db`的formal root完整完成0→50；50 macros wall约`1080.75s`，peak allocated/reserved
约`43.266/47.094GB`，0 OOM/nonfinite/clip。current-schedule macro0/10/25/50的strict correct400为：

```text
macro       0    10    25    50
correct   134   127   105   123
correct80  26    26    24    27
```

macro50的小panel看似最好，但full400仍比macro0低11，故screen不能选checkpoint。完整paired 0→10、
0→25、0→50 gained/lost=`19/26`、`19/48`、`20/31`；0→25的exact McNemar
`p=.000522`。四点success union=`172`、intersection=`77`、逐task envelope=`147`，后者仍低于150；
这些只作漂移诊断，不能融合checkpoint。macro50 breadth从6到7主要来自Spatial-1的2个新state，同时
Object-1/Object-3相对macro0净失`5/9`，Goal-6净得6，仍是任务换手而非共同积累。

内部机制与闭环方向一致。metrics是pre-update row：从warm-start到state49，generated norm约
`140.97→107.00`，expert norm约`4.18`，cosine仅`.02194→.02630`；expert loss下降约`.0752`，其中
direction只贡献约`.00436`，即约`5.8%`，其余约`94.2%`来自log-norm径向收缩。绝对投影系数

```text
a_t^x = <G_t^x, E_t> / ||E_t||^2
```

的correct mean却从`.7362`降到`.6623`，23/24 tasks下降；negative从`.6284`降到`.5332`，margin的
上升主要因为negative缩得更多，不是correct补足expert分量。严格同video/state0 cache比较中，macro50
相对macro0的norm ratio/cosine/radial coefficient/orthogonal residual over base/delta over base均值为
`.7180/.9755/.7007/.1551/.3373`。因此这轮训练首先破坏的是已有v6 LoRA的幅度，而非学到更有效的
expert方向。

裁决：停止第33节whole-LoRA direction+global-norm objective；不续100/200、不扫auxiliary weight、
不为loser补六臂。这不证伪v6上游表示，也不证伪task expert可提供局部policy-effective方向；它只证伪
“整套generated LoRA应收敛到task-local expert的方向和能量”这一假设。

## 34. v6-Initialized Policy-Effective Expert-Component Projection Writer

### 34.1 单变量假设与边界

保持第33节的完整Writer架构、historical v6 macro400初始化、冻结encoder/Core/transition/Procedure、
只训练compiler+factor、train24 logical B20/physical B10+10、positive functional、video/negative schedule
和one-shot部署合同。唯一科学修改是expert辅助几何：expert不再是整套LoRA终点，只定义一个应补足的
policy-effective分量。

“v6-initialized”不等于hard anchor。首轮不加入frozen macro0 shadow branch、zero-init residual、
orthogonal drift penalty、`G0+ΔG`重参数化或rank16 retraction；这些都会成为第二个结构变量并增加显存/
计算。若objective-only版本能增加projection却仍发生有害正交漂移，才有证据单独打开hard anchor。

### 34.2 gauge-invariant目标

对task`t`、condition`x`和全部38个public LoRA targets `l`：

```text
G_t,l^x = B_t,l^x A_t,l^x
E_t,l   = B_t,l^E A_t,l^E
a_t^x   = sum_l <G_t,l^x,E_t,l>_F / (sum_l ||E_t,l||_F^2 + epsilon)
```

`a=1`表示generated update在expert方向的最小二乘分量等于一个完整expert；它不要求`G=E`，也不约束
expert-orthogonal分量或global norm。训练目标为：

```text
L_projection = mean_t SmoothL1(a_t^correct - 1; beta)
L_ranking    = mean_t softplus((m - (a_t^correct-a_t^negative))/tau) * tau
L_total      = L_positive_functional + lambda_p L_projection + lambda_r L_ranking
```

negative仍按task visit轮换reversed/shuffled/cross-suite wrong；same-task-other始终是positive分布，不作
negative。ranking达到margin后梯度平滑趋零，不最大化negative action error。`a`只在训练时作为
gauge-invariant测量和loss，不是部署scalar gate或global scale，因此不违反禁止的scale救火。

低秩恒等式直接复用现有effective alignment contraction，不materialize dense`BA`、不增加Writer或policy
forward：

```text
<BA, B_E A_E> = tr[(B^T B_E)(A_E A^T)]
||B_E A_E||^2 = tr[(B_E^T B_E)(A_E A_E^T)]
```

expert denominator可在当前task objective中复用；FP32只用于最终rank-level reductions。每macro记录
`a_correct/a_negative/a_margin`、generated/expert norm和per-target contribution聚合，禁止逐target
`.item()`热路径同步。

### 34.3 与旧路线去重

- 不同于Recenter/Prior-Innovation：不删除DC、不规定Core/Procedure latent分工，也不引入新的prior+
  innovation双分支；实际逐视频v6输出仍由同一个Writer自由生成。
- 不同于Target-Spectral/健康度路线：没有stable-rank、奇异值、正交或rank diversity目标。
- 不同于Barycentric/Policy-Effective bank：expert不参与部署routing、mixing、字典或rank16重压缩；Writer
  仍直接输出完整A/B。
- 不同于第33节：删除`||G||→||E||`和`cos(G,E)→1`，只补expert component，明确允许大规模有用的
  non-expert v6分量继续存在。

### 34.4 证伪门与执行顺序

CPU代数门：macro0 exact-load/no-video identity/信息墙/functional path不变；projection gradient在
effective空间只沿`E`，不能直接压缩`E`正交分量；24 task denominator、coefficient、gradients finite；
记录per-target numerator占比，若少数target垄断只作为后续证据，不在首轮改target平衡。

clean push/frozen后只做一次六卡B10 gradient profile，沿用第33节吞吐图和`.25` compiler/factor
auxiliary预算，分别封存`lambda_p/lambda_r`，不做weight sweep。最早训练内部门：`a_correct`至少18/24
tasks向1移动，绝对expert component上升，generated norm不再系统塌缩，negative不爆炸，positive
functional和fixed-action transfer不出现广泛退化。

closed-loop门使用同一current schedule macro0=`134`：

- macro10若`≤129`且净损失分散于多个tasks，立即停止；
- macro10为`130--133`时，仅当projection、norm、breadth和右端斜率共同健康才允许到25；
- macro25若仍不超过134或只是task换手，停止，不跑50/100、不扫权重；
- 只有strict超过134且多个tasks净获益，才继续50/100；single winner超过134并有可信趋势后做完整
  correct/same/wrong/shuffled/reversed/no-video，最终目标仍是同一checkpoint严格`>150/400`。

若`a≈1`、projection ranking和fixed-action transfer均成立而closed-loop仍不升，必须干净证伪
expert-component假设，下一候选才是单独的v6动态hard-anchor/tangent retraction或policy-output behavior
distillation；不得继续加大projection weight或回到whole-LoRA健康度优化。

### 34.5 canonical v2实现边界（2026-08-09）

ECP已在原v6 vertical path内原位替换旧objective，canonical config为
`configs/pi05_v6_ecp_policy_effective_writer_v2.json`。实现保留同一Writer、functional forward和
counterfactual生成；FP32 low-rank contraction同时产出global coefficient及38-target contribution，
task record只做一次bulk device-to-host copy，不新增dense BA、Writer/policy forward或逐target同步。

config/run、raw gradient/evidence、resume、checkpoint/trainer/RNG、eval adapter/episode均使用独立v2
schema。旧v1 optimizer、aux weights、resume evidence和trained checkpoint不能进入live path；历史v1
aggregate只由analysis的显式legacy family读取已有`results.json`，不能加载Writer、恢复checkpoint或生成
新cache。未改变的historical macro400初始化和evaluation throughput smoke仍可继承，因为它们只约束同一
Writer推理图，不含训练objective状态。

CPU定向门已覆盖dense BA oracle、独立generated/expert gauge、expert-orthogonal energy invariance、
SmoothL1解析gradient、ranking符号、batch/shared-target broadcast、output-gradient chain rule、v1 checkpoint
拒绝和v1/v2 analysis混合拒绝。本节实现封存时尚无GPU证据；后续gradient实证取34.6，性能表述仍只认
34.4的closed-loop门。

Architecture gate对两个既有超大协议owner报告legacy-ratchet escalation，按cohesive exception处理而不
机械拆分：`v6_prior_checkpoint.py`只增加4行schema常量接线；`v6_prior_contract.py`净增56行用于ECP v2
objective/metric/artifact fail-closed验证；对应大测试文件净增26行。它们分别继续唯一拥有checkpoint和
formal artifact协议，没有新增runner、执行分支、versioned objective或parallel function family。把同一
原子协议拆到第二模块会增加跨文件状态耦合，却不减少活动责任；待该协议出现第二个真实消费者或下一轮
替换缩小schema时再提取共享声明。本轮真正的objective owner `effective_objective.py`反而净减5行。

### 34.6 gradient profile seal（2026-08-09）

clean frozen`de28157`在`gpu01:0,1,2|4,5,7`完成唯一一次B10六卡profile：train24×B20=
`480/480` unique queries，8/8/8 counterfactual，最长105帧，wall/input wait=`20.42496/.17998s`，peak
allocated/reserved=`43,316,129,280/47,093,645,312` bytes，0 OOM/nonfinite。结束后六卡释放；启动前
双节点、UUID/process、NUMA和`/data1`quota均live闭合。

unweighted compiler/factor gradients为positive=`.0110556/.105556`、projection=`.401533/1.667382`、
ranking=`.262866/.269814`。预注册逐aux逐block`.25`规则唯一给出
`lambda_projection=.006883349605446485`、`lambda_ranking=.010514451404229894`；加权后compiler各`.25`，
factor仅`.10873/.02688`。旧whole-LoRA ranking weight不可继承。artifact assembler证据已原样嵌入config，
只解锁fresh0→1/exact-resume1→3/contiguous0→3；profile checkpoint仍禁止进入formal。

初始化24/24 tasks的`a_correct<1`，mean `.73453`；correct-negative margin mean `.10324`且23/24为正，
shuffled均值`.05050`最弱并有一个`-.00968`反向。top1/top4 absolute numerator fraction median=
`.18084/.52988`，不存在单target垄断。generated norm仍远大于expert是本设计保留正交v6能力的预期，
不能据此重加norm loss。这里仍只有gradient机制证据，没有ECP训练或closed-loop成绩。

### 34.7 resume/throughput profile seal（2026-08-09）

gradient seal的strict后继clean frozen`fea3f40`在live空闲`gpu01:0,1,2|4,5,7`完成两条ECP v2轨迹：
resumed root fresh0→1后从macro1 exact-resume到3；contiguous root独立fresh0→3。三次tmux launcher均
自然exit0，两root各3 metrics、macro1/3 checkpoints和completion；三步step wall分别=
`62.36865/61.01677s`，input wait=`.17581/.23105s`，peak allocated/reserved=
`43,275,957,248/47,118,811,136` bytes，0 OOM/nonfinite。

artifact assembler确认run contracts逐字相同，gradient commit是profile commit的strict ancestor；cursor、
checkpoint contract、六rank RNG、scheduler、AMP和559 frozen tensors语义闭合。scientific metrics最大
tolerance ratio=`.429003`；macro3 Writer maxabs/global relative L2=`1.30399e-5/4.84507e-6`，Adam
`exp_avg/exp_avg_sq` symmetric norm ratio=`.999977/.999743`、cosine=`.999976/.999990`，均有充足门限
余量。这里接受并行reduction的普通低位误差，不追逐逐bit一致，也没有为此改变kernel、batch或并行度。

三步pre-update row给出早期方向证据：macro1→3的mean `a_correct=.736184→.754337`，absolute expert
component=`3.06189→3.13618`，generated norm=`140.973→142.359`；23/24 tasks的`a_correct`向1移动、
23/24 component上升、17/24 norm上升，gradient norm约`.0956→.1058`且远低于clip1。negative每步轮换，
因此aggregate margin不作三点单调门。上述只满足fresh formal 0→10的工程/早期机制授权，不构成held
性能证据；profile checkpoint永久禁止warm-start formal。

吞吐优先的首个行为裁决只跑ECP macro10 correct400。历史current-schedule macro0=`134`保留immutable
native-family rows；用显式cross-family historical-baseline transition逐row验证后比较，避免无科学增量地
重跑400条macro0。该CPU-only入口已在canonical evaluator原位实现，分别native验证legacy immutable
results与current raw reaggregation，且不放宽checkpoint-curve；全仓`262 passed`。只有确需正式
same-family ECP curve时才补跑v2 macro0，旧rows不得重标或混入curve。

### 34.8 formal0→25与strict closed-loop（2026-08-09）

clean pushed/frozen`450e688`的formal root=
`runs/outputs/pi05_v6_ecp_formal_r6_lb20_mb10_450e688_20260809`。fresh0→10后根据34.4的
grey-zone门进行同root exact-resume10→25；25条metrics、macro10/25 checkpoints、completion、
optimizer/scheduler/sampler和六rank RNG完整，0 OOM/nonfinite/clip。两段step wall均值=
`20.447/20.631s`，peak reserved约`47.1GB`，保持logical B20/physical B10+10。

macro10内部`a_correct=.828442`、component=`3.443939`、generated norm=`151.342566`；24/24 tasks
的`a`较macro1向1移动、component上升。strict correct400=`133`、breadth6、per-task=
`1/2/45/28/0/38/19/0`；对同schedule macro0=`134`严格paired gained/lost=`22/23`、net=`-1`。
因机制健康、top3 share从`.87313`降至`.83459`且exact对旧whole-LoRA macro10=`127`
有`+6`，按34.4只允许一次短resume到25，未声称改善。

macro25内部`a_correct=.884127`、component=`3.672251`、generated norm=`159.816612`；23/24 tasks
的`a`向1移动、24/24 component与norm上升。但macro10→25的expert-orthogonal norm约
`151.303375→159.774416`（`+8.471041`），component仅`+0.228312`。strict correct400 root=
`runs/outputs/pi05_v6_ecp_correct400_noreplacement_seed7_method_macro0025_450e688_20260809`，exit0、
72/72 shards、400 rows、18 workers return0、400 LoRAs、54 batches、batch上限8、0 retry/reuse/
redundant forward。结果=`120/400`、breadth6、per-task=`0/1/43/27/0/33/15/1`，suite=
`1/70/33/16`。对macro0严格paired gained/lost=`13/27`、net=`-14`、McNemar
`p=.0384773083`，suite net=`-4/-12/-2/+4`。macro10→25亦是`18/31`、net=`-13`且四suite
全部净下降。

### 34.9 退役裁决与最早后继接口

ECP的数学目标、gradient、resume和formal训练均按构造工作，held closed-loop却从
`134→133→120`。因而不能将失败解释为“expert loss没动”、“权重太小”或“训练不够”。
该实验证伪的是：在当前共享compiler/factor参数化下，只规定expert component而对其余
effective BA放任自由，不足以产生held-task共同改善。

按34.4停止ECP：不续50/100、不扫projection/ranking weight、不为loser补六臂。后继若继续
检验expert component，唯一有证据的新变量是以同一exact language + correct video的frozen
v6输出作dynamic baseline，直接限制增量的expert-orthogonal drift。这个dynamic anchor不得变成
parameter weight decay、static/language-only bypass、B-only residual、第二套部署LoRA、expert-bank deployment
或rank/checkpoint融合。它必须先与历史SFT-Anchored Tangent-Basis、短LR/weight decay及behavior
distillation去重，并用新schema在CPU dense oracle中证明只约束所声称的effective增量。

## 35. v6 Condition-Local Dynamic Expert Tangent Tube Writer

状态：**2026-08-09 ECP负裁决后的唯一活动design authority；canonical实现、CPU oracle与
formal-lineage guard已通过全仓277项回归，clean frozen`2616773`的六卡gradient/throughput seal及
strict后继`c1bdcae`的fresh/resume/contiguous seal均已通过；formal仍须等待当前evidence/config
authority的clean pushed严格后继。**

### 35.1 根因链与单变量假设

ECP的correct expert component从macro1到25增加`.61036`，但correct expert-orthogonal norm
增加`18.83485`，约为前者`30.9×`。negative的macro1→10汇总也显示component只增
`.25062`、expert-orthogonal norm却增`8.26660`；这两点使用轮换schedule中的不同输入，不能当作
same-input checkpoint drift的精确量，只能作为共享更新伴随大幅非expert位移的风险信号。结合
correct臂的持续增量与closed-loop净退化，当前最早未被隔离的因果变量是：
共享compiler/factor为改变一个expert coefficient，同时重排了大量与expert无关的原v6
effective LoRA。

新假设是：对每个已有训练condition，保留historical v6对同一exact language和同一
actual video/frame order生成的完整dynamic LoRA，只允许当前task expert方向上的局部修正。若
这个修正被干净隔离后仍不能超过macro0，才能证伪expert-component completion本身，
而不是再次把失败归因于未约束漂移。

### 35.2 condition-local gauge-invariant tangent tube

对task`t`和当前condition`x in {correct, negative}`：

```text
G_t,l^x   = B_t,l^x A_t,l^x                 current student effective LoRA
G0_t,l^x  = B0_t,l^x A0_t,l^x               frozen v6, same language/input/order
E_t,l     = B_t,l^E A_t,l^E                 step2000 task expert
D_t       = sum_l ||E_t,l||_F^2 > 0
Δ_t,l^x   = G_t,l^x - G0_t,l^x
d_t^x     = sum_l <Δ_t,l^x,E_t,l>_F / D_t
Δ⊥_t,l^x  = Δ_t,l^x - d_t^x E_t,l
```

correct的原ECP projection coefficient为保持初始gradient identity，仍使用
`a_t^correct=<G_t^correct,E_t>/(D_t+epsilon)`；tube geometry则使用expert-bank合同已保证
非零的exact `D_t`（实现仅以`clamp_min(epsilon)` fail-close）。唯一替换项：

```text
L_tube = SmoothL1(a_t^correct - 1; beta)
       + mean_x [ sum_l ||Δ⊥_t,l^x||_F^2 / (2 * beta * D_t) ]

L_total = L_positive_functional
        + lambda_projection * L_tube
        + lambda_ranking * L_existing_projection_ranking
```

condition mean固定为`correct`+当前轮换的一条`reversed/shuffled/wrong`，不因多一个臂把
anchor权重机械翻倍。ranking的score、margin、temperature和正负号完全不变；纯E-orthogonal
输出变化本身不会改变ranking scalar，但ranking沿`E_t`的共享参数更新可能连带破坏negative的
其他LoRA方向，negative tube正是隔离这种collateral drift，防止它在后续行为层伪造video margin。

低秩实现不materialize dense`BA`：

```text
||Δ||^2  = ||G||^2 + ||G0||^2 - 2<G,G0>
||Δ⊥||^2 = max(0, ||Δ||^2 - <Δ,E>^2 / D)
```

全部norm/inner product仍由rank16 Gram contraction得到。`G/G0/E`各自任意LoRA gauge变换不改变
loss。method macro0时`G=G0`，新anchor loss与gradient为0；因而新projection的未加权初始
gradient应与ECP一致，ranking/positive也完全不变。CPU oracle先证明这一点，随后一次live train24
gradient profile既验证真实BF16路径，也应复现ECP权重；不做weight sweep。由于correct与negative
tube取算术均值，该目标是condition-local anisotropic tube，**不**声称等价于到某个单一dense target
的平方距离。student始终由原A/B heads直接生成唯一rank16 LoRA，部署不做两LoRA相加或SVD融合。

### 35.3 训练时dynamic anchor与部署边界

runtime在historical macro400 warm-start刚加载、任何resume checkpoint覆写student之前，复制并冻结
恰好`compiler+factor_heads`。encoder/Core/transition/Procedure本来冻结且始终与macro0相同，因此
correct和negative均复用已经构造的condition-local memories，只各增加一次小型frozen decoder
forward；不重跑PI05/video encoder、Core/Procedure或B20 policy functional forward。anchor参数不进
optimizer/checkpoint，exact resume每次从immutable historical warm-start重建，并与其provenance合同联锁。

anchor仅是train24 auxiliary teacher：

- 只读与student arm完全相同的exact language + action-hidden video/frame order；
- task expert仍只在train loss中出现，不进Writer input、held routing或deployment；
- evaluator与checkpoint中仍只有一个student Writer，一次生成一套76-tensor/38-target rank16
  LoRA；no-video仍是template-A/zero-B identity；
- 无language-only/static bypass、scalar/global scale、gate/confidence、B-only residual、expert route、
  新parameter-space L2/anchor、rank diversity、multi-video或checkpoint融合；optimizer仍保留既有
  AdamW decoupled weight decay，不把它冒充本方法的function-space约束。

首版在唯一canonical vertical path上原位升级config/run/checkpoint/metrics/adapter family，旧ECP v2由
Git和frozen artifacts保存，不保留live双objective或兼容resume。historical-baseline transition可以只读验证
旧ECP结果，但新family只接受自己的fresh checkpoint。

### 35.4 与历史路线去重

- SFT-Anchored Tangent-Basis冻结全局8个factor-output matrices并用RL改coefficients，结果
  `143→142`、gained/lost=`20/21`；它没有保留每个language/video的完整`G0(x)`。
- Program-Credit与Condition-Kernel证伪了冻结共享decoder足以阻止upstream/common drift的假设，但没有
  condition-local effective-output anchor。
- v5.1低LR、v6 slow schedule及weight decay只改变步幅；ECP前25步decoupled WD上界远小于
  实测漂移，不能代替function-space tube。
- Recenter/Prior-Innovation是fresh hard decomposition，分别移除DC或建新prior/innovation分支，没有从
  强v6 same-video output出发。
- policy-output behavior distillation仍未做过，但它需要额外PI05 policy forward，且只约束
  sampled B20 states。只有dynamic tube机械成立而strict仍失败，才转向这个更贵的policy-space
  单变量。

### 35.5 CPU、A40与closed-loop证伪门

CPU必须先证明：

1. 三状态low-rank contraction与dense BA oracle一致，对`G/G0/E`独立gauge invariant；
2. `G=G0`时tube loss/gradient为0，完整初始projection/ranking gradient与ECP一致；
3. `Δ=cE`时tube为0，加入任意E-orthogonal perturbation时只产生回锚gradient；
4. correct/negative anchor与student的language、video/frame order逐项相同，anchor无grad/
   optimizer/checkpoint/deployment ownership；
5. source policy、frozen upstream、functional B20、information wall、no-video和evaluator输出均不变。

clean push/frozen后不重扫B10/workers或aux weights。只做一条fresh0→1、same-root exact-resume1→3与
独立contiguous0→3 A40 profile，实测两次小decoder的wall/VRAM；不得为低位数值一致降低
physical B10、六卡并行或BF16吞吐。只有在B10+10不可避免OOM，或把cache构建成本摊入预期训练
宏步后end-to-end实测更快，才启用training anchor cache；不能仅因某个局部overhead百分比就牺牲
简单直接的在线路径。cache key必须覆盖language、actual video identity、frame order/seed、condition和
macro schedule；100 macros的correct+negative upper-bound约`12.68GB`，创建前按独立`/data1` quota
与峰值预算复核，不改科学目标。

机制门：

- `a_correct`至少18/24 tasks向1移动，expert component上升；
- 只有在要把闭环负结果解释为“干净证伪expert-component completion”时，还必须满足裁决checkpoint上
  task median `|a_correct-1|≤.05`；未达到时只能证伪当前权重/训练窗内的完整recipe，不能把
  component假设本身写死；
- correct和negative的`||Δ⊥||/||G0||` task median均`≤.02`，至少20/24 tasks各自`≤.03`；
- 对方向增量非近零的tasks，`||Δ⊥||/(|d| ||E||+epsilon)`中位`≤1`；不得重演ECP
  的数十倍正交漂移。

closed-loop仍只认single checkpoint strict correct400：

- fresh只到macro10就评测；`≤129`且多task净损失立即停止；
- `130--134`只有在tube机械成立、breadth`≥6`、对macro0的churn显著低于ECP macro10的`45`
  且数值`≤35`、最近3个macro的projection方向斜率`≥0`时才允许到25；
- macro25必须strict`≥135`、至少3个task和2个suite净正增，否则停止当前recipe，不扫anchor
  weight/LR/WD。只有同时满足上一段`.05` completion门和tube门，才能把负结果升级为对component
  假设本身的干净证伪并直接转policy-output behavior distillation；否则先按component deficit与
  action-space错位的实证差异选择下一单变量；
- 超过134才到当前sealed macro50；若50仍形成共同上升趋势，再以显式config更新决定是否exact-resume
  到100。任何single checkpoint首次达到`≥144`即补
  correct/same/wrong/shuffled/reversed/no-video六臂；若另一个checkpoint首次达到`≥151`，再对实际
  goal winner补一次六臂。若correct与wrong/order/no-video同幅上升，仍不构成EMBER达标。

### 35.6 canonical ownership与architecture completion gate

本轮没有新增module、runner、entrypoint、部署分支或并行objective config；旧ECP executable config、旧
single-A40 smoke assembler及其旧runtime状态机已经删除，历史只由Git、retained artifacts和generic
read-only analysis family保留。相对本轮基线，活动`src`在退役清理后净增约204行：

- `effective_objective.py`继续唯一拥有38-target low-rank effective geometry；新增三状态contraction、
  tangent dataclass和双臂output-gradient链都共享既有alignment/ranking primitive，拆到第二owner会让同一
  denominator、gauge和autograd合同跨模块耦合；若该方法被裁决退役，tangent-only surface随退役提交删除；
- `v6_prior_contract.py`仍是唯一config/run/artifact state-machine owner，且本轮净删约284行；删除旧smoke
  assembler后没有兼容执行路径。若出现第二个真实runtime消费者，再提取schema-neutral artifact primitive；
- checkpoint、runtime、step、training和Writer decoder只做各自现有owner内的窄接线；checkpoint仍只由
  checkpoint owner解释，decoder override必须成对传入且保持同一head topology；
- 新增测试直接覆盖dense oracle、same-memory、ownership、trainable-only resume/deployment和family隔离，
  不是第二套实现。

据此接受本轮cohesive exception：当前最小可证伪变量必须同时触及geometry、训练时anchor ownership、
resume和分析identity；人为拆成versioned modules会增加并行活动路径而不减少责任。首次live profile前仍以
全仓回归、`compileall`和`git diff --check`为completion evidence，GPU结果不反向豁免结构门。

### 35.7 六卡gradient/throughput seal（2026-08-09）

clean pushed/frozen`2616773`在live空闲`gpu01:0,1,2|4,5,7`完成唯一一次v3 B10+10 profile；启动前
双节点GPU/UUID/process、3+3 NUMA、CUDA12.8、`/data1` quota和输出非覆盖均闭合，launcher再次实时
fail-close。artifact为train24×B20=`480/480` unique queries、8/8/8 counterfactual、最长105 sampled
frames、0 OOM/nonfinite；结束后六卡自然回到14MiB。

macro0的correct与negative在24/24 tasks上均有`G=G0`：student/anchor norm和projection coefficient逐项
相等，delta norm、directional component、orthogonal norm、tube loss与clamp correction全部exact zero。
unweighted compiler/factor gradients为positive=`.0110556/.105556`、projection=`.402617/1.670787`、
ranking=`.262866/.269814`；projection相对ECP同panel只差约`.27%/.20%`，ranking实质相同，符合CPU
预言而不需要追逐bitwise identity。预注册逐aux逐block`.25`规则唯一给出：

```text
lambda_projection = 0.00686480847114155
lambda_ranking    = 0.010514453175708578
```

加权后compiler两项均`.25`，factor分别`.108659/.026876`。evidence由retained root
`runs/outputs/pi05_v6_tangent_tube_gradient_profile_macro49_r6_lb20_mb10_2616773_20260809`
重新assemble并原样写回config；旧ECP近似权重只作identity参照，没有被复制或扫描。

whole-macro wall/input wait=`21.53076/.60603s`，ECP同图为`20.42496/.17998s`；raw wall增幅约`5.4%`，
扣除input wait后约`3.4%`。peak allocated/reserved=
`43,353,948,672/47,112,519,680` bytes，只比ECP增加约`36/18MiB`。因此同memories在线双decoder是当前
吞吐最优的简单路径：没有OOM、显存增长可忽略、cache构建成本没有摊销证据；不降B10+10、不降六卡、
不启用cache。该seal只解锁严格后继的fresh0→1/exact-resume1→3/contiguous0→3工程门，尚无训练或
closed-loop性能结论。

### 35.8 exact-resume seal与一阶滞后风险（2026-08-10）

strict后继clean pushed/frozen`c1bdcae`在live空闲`gpu01:0,1,2|4,5,7`完成resumed root的fresh0→1与
same-root exact-resume1→3，以及独立contiguous0→3。原自动chain在fresh进程结束后立即进入下一launcher，
inter-phase selected-GPU preflight发现设备不再满足expected-idle合同并fail-close；该阶段没有创建第二次
scientific invocation。重新live检查通过后，resume与contiguous分别在新tmux中完成并exit0。因而三个
科学invocation均有效，但原
chain exit1只作安全门证据，不得改写为训练失败或“整链exit0”。

两个root各有3 metrics、macro1/3完整checkpoints和completion。resumed/contiguous三步step wall=
`62.34061/61.95860s`、input wait=`.09366/.13220s`、macros/s=`.048123/.048419`；peak
allocated/reserved=`43,316,387,840/47,137,685,504` bytes，0 OOM/nonfinite。在线双decoder没有形成新的
显存或data bottleneck，不建立cache、不改workers、不降低logical/physical batch或六卡并行。

artifact assembler重新读取gradient root和两条profile lineage，验证run contracts完全相同、profile
commit是gradient commit的严格后继、dynamic anchor的41 tensors/`3,714,304` parameters不进optimizer、
checkpoint或deployment。macro1/3的cursor、checkpoint contract、6-rank RNG、scheduler/AMP语义相等；
559 frozen Writer tensors exact，41 trainable tensors在macro3的maxabs/relative-L2=
`8.5067e-6/1.14428e-6`。82个Adam moments在macro3的最低cosine与symmetric norm ratio远高于
`.999/.99`；scientific metrics最大tolerance ratio=`.67790`。evidence已原样写入v3 config，profile与
formal同步置为`sealed_from_live_a40_resume_profile_evidence`，formal runtime为`(50,(10,25,50))`；
profile runtime关闭且所有profile checkpoints永久不得作为formal warm-start。

resume seal同时暴露了必须保留的科学预警。macro1是exact anchor；一次更新后macro2有21/24 tasks的
`a_correct`向1移动且expert component上升，但第二次更新后的macro3相对macro1变为0/24，aggregate
`a_correct=.71744`、task median `|a-1|≈.2799`。macro3 correct/negative的
`||Δ⊥||/||G0||` task median=`.03158/.03173`，只有`10/24`与`6/24`低于`.03`；active
`||Δ⊥||/(|d| ||E||+epsilon)`中位约`60.98/61.2`，24/24两臂均未过`≤1`。同一row
`gradient_norm_before_clip≈1.45294`超过clip norm1，不能写成0 clip。

其第一性原理解释是：当前quadratic tube在`G=G0`处loss和一阶gradient均为0，因此第一步的positive、
completion与ranking梯度可以先把共享decoder带入expert-orthogonal方向；偏离后tube才产生随距离增大的
回锚gradient，并可能在warmup末端触发clip。这并不由三步直接证明recipe最终失败，因为video/task panel
每macro轮换且惩罚可能在后续数步回锚；也不能把engineering profile当成mechanism pass。保持原预注册
单变量，clean seal后只fresh到macro10并立即跑strict correct400，同时检查最近三步tube、projection、
clip和task-level pass counts。macro10若仍不满足第35.5节tube门，就算correct落在`130--134`也不得续25；
不在看到该证据前扫tube weight、改LR/WD、换硬retraction或转新架构。

### 35.9 formal0→10、strict non-pass与退役裁决（2026-08-10）

clean pushed/frozen`b308941`在live比较`gpu01/gpu02`与`/data1` quota后，只使用当时空闲的
`gpu01:0,1,2|4,5,7`，按B10+10、logical B20、workers2、3+3 NUMA、Ring/Simple、
`NCCL_P2P_DISABLE=1`和deferred-NCCL完成fresh0→10。formal root为：

```text
runs/outputs/pi05_v6_tangent_tube_formal_r6_lb20_mb10_b308941_20260810
```

10 metrics、macro10 checkpoint和completion完整；总step wall=`207.443583s`、input wait=
`.265486s`、peak allocated/reserved=`43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite，
只有macro3一次gradient clip。macro10的机制证据是：

- correct projection coefficient aggregate=`.7372456`，task median `|a_correct-1|=.2522947`，
  `0/24` tasks通过completion `.05`门；
- correct/negative `||Delta_perp||/||G0||` task median=`.0139001/.0140787`，两臂均`24/24`
  tasks通过`.03`门；
- correct/negative `||Delta_perp||/(|d| ||E||+epsilon)` task median=
  `108.926/126.883`，两臂均`0/24` tasks通过`≤1`门；
- student/anchor task coefficient correlation约`.999`。因此soft tube把总运动压小，却没有把实际运动
  旋进expert方向；绝大多数增量能量仍在目标正交空间。

同一macro10 checkpoint随后完成one-shot correct400：

```text
runs/outputs/pi05_v6_tangent_tube_correct400_noreplacement_seed7_method_macro0010_b308941_20260810
```

72/72 shards、400/400 rows自然exit0，得分`131/400`、correct80=`27/80`、breadth5、per-task=
`0/3/46/31/0/40/11/0`、per-suite=`3/77/40/11`，wall=`858.578s`、吞吐=
`.465887 rollout/s`。与同schedule macro0=`134`的exact state/RNG/language/video pairing保存在：

```text
runs/outputs/pi05_v6_tangent_tube_macro0010_historical_baseline_transition_b308941_20260810/analysis.json
```

gained/lost=`16/19`、churn35、net`-3`、McNemar exact two-sided `p=.735879`，breadth从6降到5。
与native-sealed ECP10=`133`的same-video aggregate比较也是gained/lost=`19/21`、net`-2`。correct80
相对macro0反而`26→27`，再次证明prefix不能替代full400。

按第35.5节预注册门，当前recipe停止：不续macro25、不扫tube/projection weight、LR或WD，不补六臂，
也不把hard retraction作为同方法补丁。该结果淘汰的是“当前soft tangent recipe在0→10窗口内能完成
expert方向写入”这一复合假设；由于completion始终`0/24`，不能扩大成“expert component已完整写入但
policy behavior无效”，更不能否定全部Expert-Manifold。formal config/runtime置为
`retired_after_macro10_strict_closed_loop_nonpass`并fail-closed；历史实现只由Git与retained artifacts保存。

第一失效接口应写为：

```text
LoRA-output cotangent q
  -> shared decoder J^T q
  -> Adam preconditioner P
  -> next-condition output motion J' P J^T q
```

提高soft tube权重只能改变cotangent幅度，并不能保证`J' P J^T`保持expert方向；本次半径成功、方向失败
正是这一结构错位的实证。`G=G0+s(x)E_t`式hard tangent/retraction又会在部署时需要train expert或scalar
route，违反video-only dynamic value，并重复SFT-Anchored Tangent-Basis与online expert-bank的负边界，
因此不进入候选。

## 36. Matched Expert-Flow Teacher Viability Audit

### 36.1 为什么先audit而不直接实现CEFD

task expert参数是从真实task actions得到的privileged train24监督，但“其参数更像SFT LoRA”不等于“它在
当前跨episode B20 states上给出更好的policy flow teacher”。现有positive functional本来就让generated
LoRA预测真实PI05 flow target；若expert velocity没有更低误差，或CEFD gradient只是positive gradient的
近共线重标度，增加一次完整expert PI05 forward只会降低吞吐并重复已有监督。

因此当前唯一动作是零参数更新、零rollout的matched audit。它不授权新Writer、不改变部署图，不读取
validation/test actions，不按task-expert closed-loop outcome筛task；step2000 task9即使direct rollout为0
也必须等权保留。

### 36.2 固定panel与量

使用train24 task-complete panel、每task logical B20跨episodeaction queries、现有B10+10 physical slices。
对每条query复用完全相同的processed observation/action、keyed Gaussian noise和Beta time，计算真实7维：

```text
u_t  = epsilon - action
v_E  = frozen step2000 task-expert policy velocity
v_0  = historical v6 macro0 generated-LoRA policy velocity
v_10 = retired tangent macro10 generated-LoRA policy velocity

L_E  = mean ||v_E  - u_t||^2
L_0  = mean ||v_0  - u_t||^2
L_10 = mean ||v_10 - u_t||^2
```

只比较真实7维action，padding维不得稀释或主导误差。expert forward用`no_grad`且在每个slice后立即释放
activation；macro0 student positive forward同时暴露`v_0`，不得为捕获同一tensor重复student forward。
不做10-step differentiable action distillation、不建长期cache、不扫batch或权重。

在macro0 student处再定义只用于诊断的：

```text
L_CEFD = mean ||v_0 - stopgrad(v_E)||^2
```

分别得到CEFD、positive functional、旧completion和ranking在compiler与factor trainables上的task-complete
global-mean gradient。用小型Gram系统把CEFD gradient投影到现有三项gradient span，记录norm、pairwise
cosine和`||g_CEFD - Projection_span(g_CEFD)|| / ||g_CEFD||`；不保存逐参数gradient dump，不在热路径逐tensor
同步。旧completion/ranking只作历史冗余诊断，不因此恢复为新训练objective。

### 36.3 预注册授权门

只有同时满足以下条件，才授权Cross-Episode Expert Flow Distillation（CEFD）：

1. step2000 expert相对macro0和tangent10都在至少`18/24` tasks降低matched真实flow error；同时在每suite
   六task等权mean上至少3个suite优于两种baseline。报告aggregate/per-task/per-suite，不按outcome删task；
2. CEFD gradient在compiler与factor均finite、nonzero；相对`{positive, completion, ranking}` span的残差
   norm比例两block均`≥.25`，且不能只是cosine近1的常数重标度；
3. audit完整覆盖train24×B20=`480` queries、0 forbidden reads、0 parameter update、0 OOM/nonfinite，
   保持logical B20和六卡吞吐合同。

任一门不通过就以低成本否决CEFD，不做loss weight profile或正式训练；下一候选转向直接改变shared
update parameterization，使condition-specific output motion受结构保证，而不是继续叠加parameter-space
auxiliary。两门都通过时，首版CEFD只做一个科学变量：以stop-gradient expert velocity distillation替换
ECP completion与Tangent tube，保留positive functional和既有bounded video ranking。teacher只存在于
train24训练；部署仍是exact language加恰好一条action-hidden video、一次生成一套完整LoRA。

### 36.4 engineering和执行边界

实现必须复用现有PI05 loss-only、keyed noise/time和logical-microbatch owner，并在唯一
`v6_prior_training.py`/`v6_prior_step.py` vertical path增加`teacher-audit` mode；不新增runner、第二套CLI、
并行functional framework或修改site-packages。初版把全部逻辑继续塞入历史owner后，architecture guard
实证会把`writer/functional.py`从655行推到1008行、`v6_prior_training.py`从731行推到1113行，并产生多个
大于120行的新函数，因此原“不新增module”约束由结构证据修正为以下单一owner map：

- `writer/flow_teacher.py`只拥有matched real-action PI05 velocity捕获和两类LoRA cotangent；复用
  `writer/functional.py`公开的keyed RNG/microbatch primitive，不复制sampler或policy forward；
- `v6_prior_teacher_audit.py`只拥有一次性full24聚合、Gram/span统计和pass/fail evidence，不是训练入口；
- `v6_prior_run_contract.py`从原巨型runtime提取所有v6 launch-contract构造，并唯一拥有audit schema、
  eligibility和tangent10 comparison asset检查；`v6_prior_runtime.py`仍是唯一资产装配/runtime入口；
- canonical CLI仍只有`scripts/train_v6_prior_writer.py -> v6_prior_training.py`，没有第二套可执行方法。

若audit失败，删除`teacher-audit` mode、`v6_prior_teacher_audit.py`、`writer/flow_teacher.py`及其feature tests，
只保留Git、文档和正式evidence；若通过，删除audit gate/orchestration与tangent comparison，只保留并重命名
matched-flow primitive供CEFD使用。`v6_prior_run_contract.py`作为已从853行runtime抽出的canonical contract
owner继续保留。这样新增组件有明确owner和退役触发，不形成历史实现的并行活动版本。

捕获`action_out_proj`的hook必须严格scoped并有CPU oracle证明每次functional call只捕获一次。Tangent
正式runtime保持fail-closed；audit使用独立schema/root，不能resume Tangent optimizer或把macro10当新方法
warm-start。

在代码合同通过后，使用clean pushed commit的frozen worktree；launch前重新live检查双节点GPU、owner、
UUID、telemetry、进程与`/data1` quota。最多6张空闲A40，保持
`NCCL_P2P_DISABLE=1`、NUMA physical/local rank、Ring/Simple和deferred-NCCL。该root是retained scientific
diagnostic，必须保留run contract、逐task误差、gradient Gram/残差、completion和明确pass/fail；但不跑
rollout、不生成SHA/MD5、不做与授权门无关的大量复核。

### 36.5 CPU implementation seal（2026-08-10）

当前实现已经完成但尚未运行GPU audit：

- 每个B10 slice依次执行step2000 expert和tangent10 `no_grad` forward，再执行唯一一次macro0 student
  differentiable forward；两slice共6次PI05 policy forward/task，三臂重放同一seed/logical size/offset；
- 实际`ACTION` width必须等于预注册7，三臂velocity在裁剪到real7后转FP32计算四个loss。这里仅转换
  `B10×50×7`小tensor，消除CEFD与positive的不对称BF16量化而不降低主干吞吐；
- tangent10只从sealed`b308941` macro10 checkpoint加载41个compiler/factor tensors，复用macro0 correct
  memories；不加载optimizer/RNG、不覆盖historical macro0 student；task experts仍由formal inspector按
  step2000和task ordinal加载；
- 四类gradient先rank内4-task等权mean，再一次stacked all-reduce/world6，得到严格full24 global mean。
  completion使用独立correct completion cotangent，negative schedule固定reversed/shuffled/wrong=`8/8/8`；
- span Gram来自FP32 gradients，CPU64 pinv固定`rtol=1e-5`，同时记录Gram eigenvalues、effective rank、
  projection coefficients和显式residual；coefficients/residual必须finite，避免把FP32近奇异噪声误当新方向；
- runtime不实例化optimizer/scheduler，不允许resume/checkpoint/rollout，结束时policy与Writer `.grad`仍为空。

CPU oracle分别覆盖physical B20的3次forward和B10+10的6次forward，测试轨迹合计9次；formal runtime仍固定
6次/task。两种路径都验证三臂noise/time逐tensor相等、same-memory comparison、real-action width、近共线
span、full24 gate和0 update。加载`.env.local`的最新全仓回归为`284 passed in 33.47s`；compileall、
JSON、diff-check通过，architecture guard为`review`且无hard violation/parallel family。`review`仅来自约
1.6k行有退役触发的support/test净增长，不能解释为GPU或方法结果。下一步仍只能clean commit/push、frozen
worktree与live GPU/quota preflight后运行这一次audit。

### 36.6 Formal audit结果与CEFD关闭（2026-08-10）

clean frozen`e8e4728`的正式root=
`runs/outputs/pi05_v6_expert_flow_teacher_audit_r6_lb20_mb10_e8e4728_20260810`自然exit0。完整覆盖
train24×B20=`480/480` unique queries、suite 6×4、reversed/shuffled/wrong=`8/8/8`、每task 6次且总计
144次PI05 policy forward；0 optimizer/scheduler/update/rollout/checkpoint，0 OOM/nonfinite。whole audit
wall/input wait=`39.698123/.684060s`，peak allocated/reserved=
`43,418,974,720/47,133,491,200` bytes，六张A40结束后自然释放。

matched真实7维flow loss为：

```text
step2000 expert  = .09863133045534293
historical macro0 = .09180174038435022
tangent macro10   = .09184316049019496
```

expert相对macro0和tangent分别差`+.006829590`与`+.006788170`，即ratio-of-means约`+7.44%/+7.39%`；
仅global task15、35共`2/24`同时优于两baseline，四个suite mean均失败。即使删去最差global39，expert仍比
macro0差约`6.07%`，所以teacher-quality不是边缘失败、单task outlier或长视频单调效应。

CEFD gradient本身finite且非冗余：compiler/factor相对`{positive,completion,ranking}` span residual=
`.686410/.838727`，existing span rank均为3；但novel gradient来自同一监督度量下整体更差的teacher。
distillation loss最大的tasks又偏向direct expert较弱tasks，继续加weight会优先纠正最不可信teacher。
因此按36.3预注册`authorize_cefd=false`：不做weight profile、训练、换expert step或事后删task。task experts
仍是policy-effective闭环参考，但不再是当前cross-episode pointwise flow teacher。

一次性`teacher-audit` config已切为formal non-pass并fail-closed；按36.4退役mode、
`v6_prior_teacher_audit.py`、`writer/flow_teacher.py`及其feature tests，保留Git、正式root和已抽出的canonical
`v6_prior_run_contract.py`。run contract里`runtime.physical_policy_forwards_per_task=2`实际表示单臂的两个
B10 microbatches，而audit/result的6才是三臂真实forward数；正式总数144无歧义，文档明确该旧字段语义后
不为退役schema重跑GPU。

## 37. Frozen-v6 Counterfactual-Null Condition-Kernel Program Residual

### 37.1 统一根因链与单变量

连续证据不是互相独立的架构失败，而是收敛到同一接口：

1. Program-Credit曾测得24-task原始Program cotangent pair cosine约0，但shared parameter update后的
   condition motion变成mean/median约`.580/.613`的公共方向；隐式condition kernel确实旋转credit。
2. 历史Factorized Condition-Kernel用显式Gram把这种旋转消除，200步Gram全rank、condition约`5--8`，
   predicted/observed Program update relative RMS约`.001--.002`；其strict仅`46/46/45/49`的最早原因不是
   kernel，而是fresh zero-B FactorHeads在macro50冻结时LoRA norm仅`.176`，形成低增益decoder。
3. v6-fast恰好提供已验证的高policy-leverage video→Program→LoRA decoder和`143/400`历史起点。
4. whole-LoRA/ECP/Tangent依次改变parameter target、expert component和local tube，却仍经shared
   `J' P J^T`/Adam更新；strict=`127/105/123`、`133/120`和`131`没有共同改善。
5. matched audit又关闭“换成expert flow cotangent”这一分支：teacher在22/24 tasks比macro0更差。

因此第37节只改变**condition update parameterization**：保留并冻结historical v6的全部600 tensors，复用
其强fused Program和八个FactorHeads；新增一个zero-init、fixed-video-keyed线性Program residual memory，
用显式condition Gram直接写入函数空间。它不是把两个失败架构拼接，而是用一个已验证机制替换旧
Condition-Kernel唯一已定位的cold-decoder失败，同时保持v6 baseline exact identity。

只训factor final layer再做post-Adam 24-task QP作为后备而非首选：它仍修改historical decoder、只保证当前
correct parameter loss的一阶非增，不隔离counterfactual condition，并与SPG/PCGrad/GROUP4的parameter-
conflict路线更接近。第37节则冻结decoder、明确控制induced cross-condition Program motion，证据指向更直接。

### 37.2 部署图和信息墙

对一个合法condition`c=(exact language, one action-hidden video)`：

```text
frozen v6 evidence/memories
  -> frozen compiler -> S0(c) [320,256]

same action-hidden frame evidence
  -> fixed zero-preserving temporal feature phi(c) [256]
  -> mutable Program memory M [256,320,256]
  -> R(c) = phi(c) M

S(c) = S0(c) + cast(R(c), dtype(S0))
  -> frozen historical v6 FactorHeads
  -> one complete 38-target public rank16 LoRA
```

`M`从逐元素零开始，step0的`R=0`，所以同batch/dtype下完整LoRA必须等于historical v6 macro0；没有第二套
LoRA、B-only residual、scalar gate、global scale、expert-bank route或checkpoint融合。residual在唯一factor
decoder之前与base Program融合，A/B两侧均按原v6完整生成。no-video仍走现有template-A/zero-B identity，
不得用`S0`或memory形成language-only fallback。

固定feature只从已经计算的`WriterVideoEvidence`构造，不增加PI05 forward。对每个实际输入帧，把
`frame_evidence`减去同condition的`text_queries`，只在`valid_task_tokens`上取mean得到256维逐帧visual
innovation。以真实sampled frame ordinal归一化`tau∈[-1,1]`，形成四个时间矩：

```text
d(c) = concat_t_basis mean_t [1, tau, cos(pi*tau), sin(pi*tau)] * innovation_t
     in R^1024
phi(c) = zero-preserving L2Norm(W_fixed d(c)) in R^256
```

`W_fixed`是无bias、fixed-seed、row-normalized JL buffer，不训练、不按outcome/held选择。descriptor为零时
`phi`必须精确为零，所以language不能单独写memory value。reversed/shuffled在真实frame content重排后仍用
原位置`tau`重新算feature；wrong video必须在目标exact language下完整重算evidence。same-task其它正确视频
不作negative，也不平均feature或LoRA。

### 37.3 Full48 counterfactual-null显式更新

每个macro仍覆盖24 tasks、每task一条correct video和B20跨episodeaction queries。correct Program leaf只读
真实source-action functional loss，得到：

```text
g_i = d L_functional_i / d S_i,  i=1..24
```

每task同时按固定8/8/8 schedule构造一个wrong/shuffled/reversed feature，但不执行negative PI05 policy
functional forward，也不计算expert completion/ranking/tube/CEFD。将48个condition按
`correct task ordinal 0..23, counterfactual task ordinal 0..23`固定排序：

```text
Phi = [Phi_correct; Phi_negative]          [48,256]
G   = [G_correct; 0]                      [48,320,256]
K   = Phi Phi^T
lambda = .01 * mean(diag(K))
A   = (K + lambda I)^-1 G
Delta M = -1.0 * Phi^T A
Delta R = K (K + lambda I)^-1 [-G_correct; 0]
```

因此当前correct近似沿各自真实functional descent移动，当前counterfactual residual motion近零；方法不主动
让negative policy变差，也不读wrong-task action/expert。若task cotangent本来同向，不强制正交；若不同，
它们不再由shared trainable decoder/Adam state压成公共方向。held condition只经固定video-feature kernel
连续泛化，base v6路径始终保留。

沿用历史已实证的`step_size=1.0`与relative damping`.01`，不做held sweep、global cap、momentum、Adam、
weight decay或逐coordinate preconditioner。只有48×48 Gram/Cholesky使用FP64；inverse operator乘Program
cotangent和约21M-value memory write均用FP32，禁止为底层微差把巨大RHS扩成FP64。每rank只all-gather本地
8个features和4个correct cotangents；六rank按相同排序独立形成同一delta，不all-reduce 80MiB memory。

### 37.4 与历史路线去重

- 不恢复旧Condition-Kernel runtime、RFF authority、fresh FactorHeads或86M随机Program；只移植小矩阵
  kernel数学，并改为zero residual + frozen v6高增益decoder。
- 不恢复SPG/CP-24/PCGrad/SERIAL/GROUP4：这些改变parameter gradient/grouping，没有固定induced
  cross-condition function kernel，也没有counterfactual-null rows。
- 不恢复SFT-Anchored Tangent Basis：后者冻结factor output basis但condition coefficients仍经shared Adam，
  且没有显式video feature Gram。
- 不恢复soft/hard expert bank、Barycentric、Grounded Route或CEFD；训练和部署均不读expert output。
- 不恢复few-shot/K4；本候选继续exactly one video。只有one-shot结构已把真实functional motion正确输运后
  仍被same-task video variance明确限制，才重新讨论固定K。

### 37.5 实现owner、吞吐和retirement

cohesive `writer/condition_update.py`唯一拥有zero-preserving temporal feature、20,971,520-value
FP32 residual memory、48×48 Gram solve和manual apply。`writer/model.py`只暴露frozen fused slots、单一
Program融合和原FactorHeads decode；不复制encoder/compiler/head。`v6_prior_step.py`只构造correct leaf与
negative feature；training只负责full48 gather/order/apply。checkpoint family fresh-incompatible，只保存
zero-residual lineage、memory、cursor、六rank RNG和fixed update contract；没有optimizer/scheduler/scaler
moments。历史600 tensors必须strict load后全冻结且checkpoint不能覆盖。

P256 memory约80MiB，projection约1MiB，full24 correct cotangent约7.5MiB，transient coefficient/delta约
15/80MiB；不增加PI05 forward，不降低logical B20、physical B10+10或六卡并行。profile若relative wall
overhead超过约10%，先融合gather/matmul并去除重复predicted-motion计算，不减batch、不扩FP64、不增加cache。

teacher-audit mode、`v6_prior_teacher_audit.py`、`writer/flow_teacher.py`及其一次性tests在本design实现前删除；
`v6_prior_run_contract.py`作为canonical owner保留并升级新family。旧Tangent/ECP/formal config保持fail-
closed，不能与新memory checkpoint互载。

当前实现已按上述owner落地，并把一次性teacher-audit/effective-objective/flow-teacher执行路径及对应tests
删除；历史只留Git和formal artifact。fresh/profile只允许当前clean HEAD精确等于
`origin/codex/bci-continuation`；formal exact-resume则继续绑定原run contract中的frozen commit，并要求该
commit仍是当前authority的ancestor。移动中的remote authority不写入immutable run contract，因此后续文档
提交不会伪造或阻断同root resume。checkpoint只含单个FP32 Program memory、cursor和六rank RNG；controller
部署检查只读metadata，实际worker load时再做一次finite value scan，避免重复80MiB读取。

artifact状态本身不是证据。mechanism seal必须从`mechanism_profile.json`保存的raw macro重算全部13项门，
并核对同目录run contract中的historical initialization、source identity、train24 task/language/data schedule、
Writer/objective/ownership、A40 rank topology和NCCL runtime；不能只信`passed=true`或预填checks。
`formal_result_sealed`只在同目录completion=`50`、连续50行metrics以及macro10/25/50三个memory-only
checkpoint manifests全部通过metadata/cursor/contract复核时成立；训练中间段保持同一个immutable ready
config，不用可伪造的running字符串改变resume身份。deployment checkpoint的training commit还必须同时位于
当前checkout与`origin/codex/bci-continuation`的共同lineage。评测可在clean detached frozen authority
ancestor上运行，不要求额外创建第二主分支。

### 37.6 CPU、A40和closed-loop证伪门

CPU必须覆盖：zero descriptor/feature/memory、natural/reversed/shuffled真实order、wrong-video target-language、
full48小矩阵predicted/observed equality、negative-null motion、base600 frozen、step0 complete-LoRA identity、
A/B双方响应、checkpoint/resume和0 forbidden reads。代码结构变化后只跑一次聚焦与全仓回归、compileall、
JSON和diff-check。

首次六卡只做macro49 mechanism/throughput profile，不保留权重：

1. 24 correct + 24 negative unique rows、feature rank/regularized Gram finite；记录condition number但不按漂亮
   数值选seed/P；
2. aggregate correct-motion/cotangent RMS ratio必须`≥.25`，negative/correct motion RMS ratio必须`≤.25`；
   同时保留全部24条task-local证据，至少`18/24` correct rows过retention门且至少`18/24` paired negative
   rows过null门，不能靠少数大任务掩盖task-local失败；
3. actual `feature @ applied_delta`必须与Program memory写入后的observed motion在relative RMS`.005`内闭合。
   这是**application closure**，证明生产delta被完整写入；solver本身仍由CPU algebra oracle验证，不能把这项
   写成两个独立求解器的交叉证明；
4. historical 600 tensors的parameter/buffer version不变，A、B两侧LoRA response都非零；另在每个suite
   固定一个task（ordinal `0/6/12/18`）做before/after同observation、同noise的fixed-action probe，要求
   `4/4` task非零。这里共8次PI05 inference forward只属于profile verification，使用observation而不读取
   target action，不进入formal训练热路径，也不计作negative functional policy forward；
5. 生产wall只计算24-task correct functional工作加full48 gather/Gram/manual write，明确排除上述
   task-local/application/LoRA/fixed-action verification。它与sealed A40 v6 macro49 B20/B10+10基线
   `21.095109596s`比较，ratio必须`≤1.10`；不以kernel fraction或底层低位误差决定吞吐门；
6. 生产路径仍是每task两次B10 correct PI05 forward、0 negative policy forward、0 OOM/nonfinite；不得为
   profile过门降低batch、扩宽大RHS到FP64或加入同步型热路径instrumentation。

mechanism profile通过并写回artifact后，还要对**新residual deployment graph**在一张实时空闲A40上用同一
fixed panel实测Writer batch `8/16/32`，选择稳定LoRAs/s最高且有显存余量的点，并完成correct one-GPU smoke；
旧v6/Tangent未改变推理图时的throughput seal不能冒充本family。两类seal均只接受同目录run contract、正确
v8 residual adapter/family、actual clean commit和A40 evidence；其它Writer family必须fail closed。

profile通过后从zero memory fresh到macro10并立即跑完整strict correct400，不用80-row screen替代：

- `≤129`且多task净损失：立即退役；
- `130--134`只有breadth/churn和mechanical evidence同时改善才允许到25；
- `≥135`允许到25；首次`≥144`立即补完整correct/same/wrong/shuffled/reversed/no-video；
- macro25/50仍不超过macro0=`134`或只做suite/task换手：退役，不扫P/lambda/eta、不解冻base、不加expert/
  route/gate；
- 任一点strict`>150`必须由同一checkpoint六臂确认真实视频与时序因果，再继续提高absolute、breadth和稳定性。

若Program motion、LoRA/action传递和counterfactual-null均成立而closed-loop不升，则“ground-truth pointwise
functional cotangent即使被正确输运也不足”获得直接证据；下一步才转同一Program memory上的真实reward
credit，而不是再换rank、能量、expert几何或condition router。

### 37.7 v1 mechanism profile正式non-pass（2026-08-10）

clean pushed/frozen `6903ee6`在`gpu02:0--5`完成唯一一次macro49 profile；root=
`runs/outputs/pi05_v6_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_6903ee6_20260810`。
六卡均为A40且按physical/local rank绑定GPU-local NUMA，`NCCL_P2P_DISABLE=1`、Ring/Simple和deferred
process group均由run contract实证。运行自然exit，0 OOM/nonfinite、0 negative policy forward，退出后六卡
均回到0MiB；root只有contract、invocation、profile和completion，没有checkpoint或retained weight。

13项门中10项通过。强正机制证据为：full48 feature rank=`48`，correct motion/cotangent aggregate=
`.807966`且`24/24` task过retention；predicted/observed application relative RMS=`0`；A/B response RMS=
`1.27385e-5/1.26956e-5`；四suite fixed-action probe=`4/4`且response RMS=`.00121293`。因此frozen-v6
Program→完整LoRA→action路径、显式update符号、full48 gather/order和FP32 memory write都成立，未发现工程bug。

non-pass集中在condition key几何。regularized Gram condition=`1315.33`；aggregate negative/correct motion=
`.264351>.25`，task-local null仅`15/24<18/24`。按negative类型的feature cosine mean、motion leakage mean和
过门数分别为：shuffled=`.98552/.38347/2-of-8`、reversed=`.95645/.20054/6-of-8`、wrong=
`.90627/.12898/7-of-8`。全部9个失败row的paired cosine均`>=.97099`，所有`<.97`的row均通过。单位
pair在ridge `.01`下的解析leakage为`rho*.01/(1.01-rho^2)`；它对24条实测leakage的相关为`.99021`，
所以失败不是finite-rank、Cholesky、sign或分布式误差，而是四时间矩里的DC块让correct与尤其shuffle几乎
重合。最难pair距离仅`.07777`，任何线性memory要分出`[g,0]`都必须放大至少约`12.86x`。

生产wall=`22.493528+1.037176=23.530704s`，相对sealed baseline ratio=`1.115458>1.10`，按原门
保留为non-pass，不事后扣除I/O。它只超允许上限`.326083s`，而本次input wait比跨host baseline高
`.633711s`；去wait诊断约`1.086`，所以不能写成确定的结构性慢化，也不为此单独重跑。verification
`.770155s`已正确排除。v2只清除每condition重复GPU sort/mask同步，并把profile-only version bookkeeping与
15MiB zero tensor移出生产计时；不降B20/B10、不开BF16大memory、不改FP64小矩阵合同。

降低lambda虽可让当前48 rows机械插值，但最难pair要到约`.002`才可能过门，并把差分方向放大从约`3x`
推到约`7.8x`，增加held视频噪声与漂移风险。hard nullspace、SVD或negative reweight同样不能创造缺失的
顺序区分信息。故v1正式退役：不训、不扫lambda/seed/P/threshold；Git和上述artifact保留完整证据。

## 38. Balanced DC–Causal Condition Key v2

### 38.1 单变量与历史依据

显式condition kernel与frozen-v6高增益decoder尚未被否定；被否定的是“未平衡四时间矩JL足以区分正确与
顺序反事实”。v2只替换fixed condition feature，继续冻结historical v6全部600 tensors，保持P256
zero-init Program memory、full48 update、`lambda=.01`、step size1、B20/B10+10、8/8/8 negative schedule、
0 negative policy forward和全部closed-loop门不变。

这个修正直接复用第19节已有证据，不是新猜测：phase16 cache的DC能量中位`.98057`，而phase-centered
sqrt-normalized causal-prefix虽然能量小，same-task template correct/reversed/shuffled cosine为
`.96263/-.94287/-.04463`。纯causal key仍不适合线性memory：若reverse接近`-correct`，同一个线性M也
无法稳健实现`correct→g, reverse→0`。因此必须同时保留一个video-only affine anchor，并把static/dynamic
能量显式平衡。

### 38.2 唯一部署key

对每条视频已经得到的逐帧visual innovation：

```text
x_t = mean_valid_task_tokens(frame_evidence_t - text_queries)
s   = mean_t x_t
z_t = x_t - s
p_t = sum_{u<=t} z_u / sqrt(t+1)
d   = mean_t p_t

u_s = ZeroL2(W_s s)       W_s in R^[128,256]
u_d = ZeroL2(W_d d)       W_d in R^[128,256]
phi = ZeroL2(concat(u_s, u_d)) in R^256
```

`W_s/W_d`由同一fixed seed一次生成，组成一个nonpersistent FP32 buffer`[2,128,256]`，逐row归一、无bias、
不训练、不进checkpoint。两个非零block各自单位化后在最终key中等能；不再让真实DC约50倍能量支配角度，
也不做会把单一高频放大约140倍的逐频归一。causal-prefix固定绑定真实sampled-frame顺序；reverse/shuffle先
重排frame content再重算`z,p,d`，wrong video仍在目标exact language下完整重算evidence。

zero/no-video innovation使`s=d=phi=0`，所以memory read精确零，不能形成language-only LoRA。static block
也不是可独立拟合target的旁路：same-frame-set reverse/shuffle与correct共享完全相同`s`，但full48 RHS分别
为`g/0`，任何只读`s`的memory都无法满足约束，必须利用causal block分解；frozen base `S0(c)`仍保留原v6
完整Core/Procedure与高层task语义。same-task其它正确视频则预期同时共享稳定`s`与相近`d`，不会用纯随机
正交化牺牲鲁棒性。

按历史proxy的等能组合只作预注册机制预测：correct-vs-reverse cosine约
`(1-.94287)/2=.0286`，correct-vs-shuffle约`(1-.04463)/2=.4777`；对应孤立pair在`.01`下的leakage约
`.0003/.0061`，远低于v1 shuffled `.2544`的解析值。真实v6 evidence的判据仍只认一次新的macro49 raw
profile：原13项门全部不变；诊断上预期condition显著低于`1315`、aggregate negative/correct`<=.15`、
null至少`21/24`且每类至少`6/8`，但这些更强预测不替代既有pass门，也不按结果调seed或归一化。

### 38.3 生命周期与下一裁决

canonical config/schema/checkpoint family升级到v2；v1 config和旧feature实现从active tree删除，历史run由
frozen commit读取。`writer/condition_update.py`仍是唯一feature/kernel/memory owner，不增加module、runner、
mode或并行Writer；projection从约1MiB减到约256KiB，PI05 forward数不变。

CPU需新增一个严格反例：两帧具有相同static、相反centered dynamics时，natural/reversed key均unit且内积
为0，证明等能anchor打破`+/-`共线；原zero/order/kernel/application/A/B/checkpoint tests继续通过。之后只在
clean pushed frozen commit、live空闲卡和quota通过时做一次v2 macro49 profile，优先使用sealed baseline同一
`gpu01:0,1,2,4,5,7` panel以减少跨host噪声；若任一卡不空闲则等待或重新选合法空闲panel，不干扰他人。

v2 profile不过则按最早失败接口裁决，不调lambda/seed/P或阈值；全部13门通过后才做新deployment graph的
batch8/16/32吞吐seal、macro0 strict400和fresh0→10。后续strict门与37.6完全相同，最终仍只以同一single
checkpoint的absolute、breadth、低漂移及correct/same/wrong/shuffled/reversed/no-video六臂裁决EMBER性能。

### 38.4 正式mechanism profile结果（2026-08-10）

clean pushed/frozen`5d93434`在live空闲且与sealed baseline相同的`gpu01:0,1,2|4,5,7`完成唯一macro49
profile：root=`runs/outputs/pi05_v6_balanced_causal_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_5d93434_20260810`。
24 tasks、B20、8/8/8 negatives、0 negative policy forward、0 checkpoint/OOM/nonfinite；六卡结束后释放。
functional loss和cotangent RMS仍与v1完全相同，为`.091801740/2.1920664e-6`，因此这是隔离key与热路径
同步的matched比较。

原13项门13/13通过：

- full48 feature rank=`48`，regularized condition=`106.114017`；
- aggregate correct motion/cotangent=`.968254`，negative/correct=`.0218514`；
- task-local correct retention=`24/24`、最小`.942261`；negative null=`24/24`、最大leakage`.048462`；
- predicted/observed application relative RMS=`0`；A/B response RMS=`1.37744e-5/1.38187e-5`；
- 四suite fixed-action=`4/4`、response RMS=`.001210888`；
- production task/kernel=`19.585536/.436306s`，合计`20.021842s`，对sealed `21.095110s` ratio=
  `.949122`；input wait=`.069295s`与baseline`.076318s`匹配，verification另计`.735397s`；
- peak allocated/reserved=`43,261,166,592/46,917,484,544` bytes。

三负臂的feature cosine mean/max与negative leakage mean/max分别为：

| arm | cosine mean/max | leakage mean/max | pass |
| --- | ---: | ---: | ---: |
| shuffled | `.479565/.851083` | `.024184/.048462` | `8/8` |
| reversed | `.013732/.023307` | `.018664/.032562` | `8/8` |
| wrong | `.507178/.762135` | `.025999/.033571` | `8/8` |

这同时满足预注册的更强诊断预测：aggregate leakage`<=.15`、null至少21/24且每臂至少6/8；condition虽为
`106.1`而不是只读审计曾建议的约100诊断值，但它不是正式门，且相对v1降低约12.4倍、全部逐task结果留有
大幅余量，不事后修改阈值或重跑。value-delta RMS从v1`2.12559e-6`降到`1.16318e-6`，correct retention
反而上升，进一步说明移除的是病态差分放大而非有效写出能量。

裁决：v2 mechanism正式seal，v1 key继续退役；不调lambda/seed/P/归一化。该结果尚未测same-task-other、
多步累积或closed-loop absolute，独立block L2放大微小dynamic的示范噪声仍是明确风险。下一步先在新clean
pushed seal上做single-A40 residual deployment graph batch8/16/32 profile与correct vertical smoke；只有该
seal也通过，才运行zero-memory macro0、formal fresh0→10和即时strict400。

### 38.5 Deployment双root seal与formal fail-close修正（2026-08-10）

mechanism artifact写回后复核发现两个执行合同缺口：config/runtime仅凭mechanism seal已经允许formal，且
current v8 evaluation verifier只重读throughput profile root，未引用另行要求的vertical smoke root。这样
虽然权威文档规定“profile+correct smoke后才训练”，机器状态本身仍可能跳过第二个条件。

修正不改变Writer、LoRA、video feature、policy、batch或任何GPU热路径：formal状态改为
`blocked_until_live_deployment_profile_and_smoke_seal`，`runtime_for_mode(formal)`同时要求mechanism seal、
evaluation `sealed_from_live_residual_deployment_profile`和结构化evidence。deployment seal由唯一
`v6_prior_deployment_seal` owner从raw artifacts重建，必须共同验证：

- 同一clean authority commit和同一v8 residual adapter；
- profile为validation8×4-state固定32-request panel，batch8/16/32处理完全相同的entry/frame序列，并按
  stable LoRAs/s选择batch；
- vertical为validation8×state0 correct真实闭环，单A40、单次成功launcher、8 rows、8套新cache LoRA；
- selected batch贯穿vertical run，Writer释放、source policy复用且未reload；
- cache为76 tensors、每entry`2,641,920` bytes、72 BF16+4 F32，0语言旁路、expert-bank与禁止读取。

config evidence只保存三个raw artifact的repo-relative path+bytes、run commit和selected batch，不做SHA/MD5或
逐tensor重校验。任一root/result/cache缺失或改变都fail closed；通过后状态才转换为
`active_deployment_sealed_formal_ready`/`ready_after_live_mechanism_and_deployment_seals`。

最终无GPU合同回归为`283 passed in 26.10s`、compileall/Black/26 JSON/diff-check；architecture guard相对
`5d93434`无hard violation，原1243行contract缩为1101行，新增逻辑集中在一个不进入热路径的deployment-seal
owner，未出现parallel implementation family。

### 38.6 Deployment实证与下一裁决（2026-08-10）

clean pushed/frozen`2af82aa`在live空闲`gpu02:0`完成了第38.5节预注册的两步。throughput profile固定
validation8×4、32 requests、1093 sampled frames和相同longest-first entry序列，仅改变物理forward分组。
batch8/16/32的LoRAs/s=`.911238/.901898/.906482`，两次measured wall分别为
`34.9668/35.2673`、`35.6341/35.3274`、`35.2987/35.3039s`；三点均stable，reserved约12.9GB（约12.0GiB）且
headroom约32.4GiB。因此按原规则选择实测最高的batch8，不因BF16低位差异或可用显存改变结论。

随后的validation8×state0 correct vertical smoke真实执行一条完整部署链：8条action-hidden videos各生成
一套完整38-target rank16 LoRA，写入72 BF16+4 F32的native cache，每entry`2,641,920` bytes；Writer随后
释放，同一个source policy无reload复用并完成8条LIBERO闭环。结果为8/8 rows、`4/8` success、单次launcher
return0、总wall=`336.056s`、rollout window=`199.799s`，0 retry/runtime failure/forbidden reads。结束后GPU回到
0MiB/P8。

唯一deployment assembler共同重读profile、results和cache manifest，验证同commit/v8 adapter、固定panel、
selected batch、8新entries、lifecycle与信息墙后通过。config状态因此切为
`active_deployment_sealed_formal_ready`，formal为`ready_after_live_mechanism_and_deployment_seals`。
这里的`4/8`只证明真实执行，不能作absolute或视频因果成绩。下一步必须从新clean pushed/frozen seal先做
zero-memory macro0 strict correct400，以同schedule建立closed-loop基线；随后才fresh0→10并即时strict400，
按第38.3门裁决停止、继续或补完整因果臂。deployment写回后的CPU门为全仓`284 passed in 26.86s`，并
重新验证raw seal等于config evidence、formal runtime ready和pre-deployment状态必然fail-close。

### 38.7 Frozen-worktree formal prepare路径合同修复（2026-08-10）

deployment evidence由`d228d0d`写回并clean push后，第一次从对应detached frozen worktree执行CPU-only
formal prepare，在0 CUDA worker、0 cache和0 scientific row时被runtime拒绝。三个raw artifacts和seal
内容都未改变；根因是frozen worktree的`runs`是指向canonical仓库`runs`的软链接，旧evaluation verifier
先对artifact调用`.resolve()`，再强制要求resolved path位于worktree自身，因而把合法canonical artifact
误判为越界。相同artifact以canonical repo root检查通过，证明失效接口是deployment evidence path owner，
不是Writer、checkpoint、panel或GPU runtime。

`af7b101`做唯一窄修复：record和load只为词法parts精确以`runs/outputs`开头、且resolved target仍包含在
`(repo_root / runs/outputs).resolve()`内的路径提供canonical映射；绝对路径、`..`、`runs/outputside`、
nested symlink逃逸以及不在vertical root内的manifest继续fail closed。主仓与frozen worktree因此生成同一
稳定repo-relative evidence，不引入fallback family、hash扫描或热路径开销。正向symlink round-trip与两类
逃逸负回归加入现有owner tests；全仓`285 passed in 21.38s`、compileall/Black/JSON/diff-check通过。

clean frozen`af7b101`随后重跑完全相同的CPU-only prepare并exit0。run contract精确确认formal validation
8 tasks×states0--49、correct/without-replacement、seed7、18 rollout workers与18 Writer generators、batch8；adapter v8把
Writer登记为`historical_v6_macro400_load_only`、method macro0、`[256,320,256]` FP32
`fresh_elementwise_zero` residual且checkpoint residual bytes=0。estimated peak new bytes=
`1,064,370,176`。这只解除正式启动的工程阻塞，不是closed-loop成绩；下一证据仍必须是新clean
pushed/frozen authority上的zero-memory macro0 strict correct400。

### 38.8 Zero-memory macro0完整closed-loop identity（2026-08-10）

从包含第38.7节修复和当前authority的clean pushed/frozen`6b5f7a6`，在启动前live确认空闲的
`gpu02:0--5`完成正式validation correct400。root为
`runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_noreplacement_seed7_method_macro0000_6b5f7a6_20260810`。
adapter v8登记historical v6 macro400 load-only、method macro0、`[256,320,256]` FP32
fresh-elementwise-zero residual且bytes=0；8 tasks×states0--49、seed7、correct/without-replacement。

运行自然exit0：72/72 shards、400 rows、18 rollout workers均attempt1/return0；strict correct=`134/400`、
correct80=`26/80`、breadth6。per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
`0/5/48/34/0/35/11/1`，per-suite=`5/82/35/12`。总wall=`867.152s`、shard execution window=
`616.138s`，overall/rollout-only throughput=`.461280/.649205 episodes/s`。400套LoRA由18 generators以
54个batch全部fresh生成，configured/max observed batch均8，max sampled frames69；0 reuse、0 redundant
Writer forward，Writer全释放，source policy全原进程复用且未reload。max per-generator allocated/reserved=
`11,745,421,312/12,895,387,648B`；0 retry/OOM/nonfinite/forbidden reads，六卡结束后0MiB/P8。CPU aggregate
重建与原results完全一致，root实际`1,085,108,227B`。

最关键的裁决不是aggregate恰好等于历史`134`。以`(suite, task, init_state)`匹配历史native v6 macro0的
400行后，language、env seed、policy seed/noise序列、teacher demo/reference demo、video order/selection
seed、video mapping全部0差异；success也逐行完全相同，gained/lost=`0/0`、共同成功/失败=`134/266`。
新旧400 cache entry IDs相同；独立逐tensor CPU直比400/400 files、30,400 tensors和514,867,200 values
全部bit-exact。每task demos0--49各出现一次。唯一非输出差异是一条共同成功episode在新root晚1 env step
终止（`106→107`），其余399 rows steps一致；不改变成功集合或formal结论，也没有理由为此牺牲batch/吞吐。
这把“v2 key/residual adapter即使memory为零仍可能改变部署行为”的工程风险
关闭，并给后续Program learning提供exact native baseline；它不是v2非零memory性能证据。按第38.3门，下一
唯一动作是从新clean pushed/frozen authority formal fresh0→10并立即跑strict correct400。

## 39. Exact Anchored Reconciliation / Recursive Least Squares

### 39.1 v2正式结果与最早未闭合接口

Balanced DC--Causal v2从clean frozen `abd8e08`完成了formal fresh0→10、exact-resume10→25和两个
strict correct400。macro10为`140/400`、breadth6、per-task=`1/2/48/31/0/38/20/0`；相对exact
macro0=`134`为gained/lost=`19/13`、net`+6`。macro25为`139/400`、breadth6、per-task=
`2/4/48/30/0/38/17/0`；相对macro10为gained/lost=`12/13`、net`-1`，相对macro0为`18/13`、net`+5`。
macro0与macro10 success union=`153`，但没有一个single checkpoint保留共同能力。macro25继续产生
episode/task换手而没有提高single score，因此blind `M += Delta M`路线在macro25正式退役，不续macro50、
不补因果臂、不扫scale/damping/step size。

训练工程合同健康：0→25共25个full24 macros、0 OOM/nonfinite/negative policy forward，mean step约
`21.42s`，B20/B10+10、feature rank48和约43.25GB/card peak allocation均稳定。macro10 LoRA健康分析显示
effective BA相对macro0的变化中位仅`1.695e-4`，stable rank仍约`1.00002`、top singular energy约
`.999978`、B-column cosine仍约`.998/.998/.996`。所以`+6`不是LoRA rank/energy健康度修复，继续盲目放大
只会增加阈值翻转与漂移。

更直接的定位来自同task 50条严格配对正确视频：macro0→10 raw LoRA delta方向一致性仅
`.141539--.142175`，等于随机正交参考`1/sqrt(50)=.141421`；真实effective `Delta BA`的same-task
pair cosine mean为`-.001371--.003280`，F32 action-target effective cosine为`-.009579--.014302`。
当前图能看到视频并产生policy-effective变化，但不同正确示范没有形成稳定的same-task修正方向。结合
macro0∪10=`153`，当前最短因果问题是跨macro能力共存/保留，而不是先换decoder、加expert、改rank或
进入few-shot。

### 39.2 唯一单变量：anchored cumulative ridge

部署图完全不变：输入仍是exact language加恰好一条action-hidden teacher video；Balanced v2 key
`phi:[256]`、frozen historical-v6 600 tensors、Program memory `M:[256,320,256]`和完整38-target rank16
LoRA输出全部不变。新增状态只存在于训练期，不能被deployment adapter读取。

对macro `k`定义：

```text
F_k = concat(phi_correct, phi_negative)                    [48,256]
E_k = -step_size * concat(G_correct, exact_zero_negative)  [48,320,256]
T_k = F_k M_(k-1) + E_k
```

negative的语义是“本次incremental motion为零”，不是把历史累计residual强拉回绝对零。唯一累计目标是：

```text
J_k(M) = 1/2 ||M||_F^2
       + sum_(s<=k) 1/(2 lambda_s) ||F_s M - T_s||_F^2
```

维护FP64 feature-space precision `Lambda_0=I_256`，每macro精确执行：

```text
X       = solve(Lambda_(k-1), F_k^T)
S       = lambda_k I_48 + F_k X
gain    = X solve(S, I_48)
Delta M = gain[:, :24] @ (-step_size * G_correct)
M_k     = M_(k-1) + Delta M
Lambda_k = Lambda_(k-1) + F_k^T F_k / lambda_k
```

其中`lambda_k=.01*mean(diag(F_k F_k^T))`，与v2原门相同。首macro严格等价原
`F^T(FF^T+lambda I)^-1 E`；CPU FP64 oracle最大差`6.1e-16`，四批streaming RLS与显式累计ridge最大差
`3.5e-14`。这不是optimizer、checkpoint融合或多模型ensemble；single checkpoint部署只读取最终唯一M。

训练checkpoint新增一个`[256,256]` FP64 precision（524,288 bytes）和`assimilated_rows=48*macro`；
不保存历史feature/cotangent/target。相对约80MiB Program memory新增约`.63%`。大RHS、Delta M和memory
write仍为FP32；利用negative RHS为零只计算`gain[:,:24] @ G`，不增加policy forward、collective或部署
开销，也不为了微小数值误差降低B20/B10+10、BF16 policy或六卡并行。

### 39.3 生命周期、实现owner与fail-close

这是fresh-incompatible family：precision必须从identity与zero Program residual同时开始；v2 macro10/25
缺少历史precision，禁止伪resume或warm-start。旧blind solver不保留runtime fallback，只由Git、正式root和
第38--39节结论保存。

- `writer/condition_update.py`是唯一RLS/precision数学owner；
- `v6_prior_training.py`在完整full48 gather后预验证并共同写入M与precision；
- `v6_prior_runtime.py`创建identity precision并只在exact-resume恢复；
- `v6_prior_checkpoint.py`原子保存deployment Program memory、training-only precision、cursor和六rank RNG；
- `v6_prior_run_contract.py`、config和inspection显式区分deployment state与training-only reconciliation；
- `pi05_eval/analysis.py`登记独立RLS-v3 family并从immutable queue/shards重算formal support gate；
- `v6_prior_step.py`、Balanced feature、historical v6 decoder、Writer generation和official evaluator不改。

结构增长只对应三个当前owner，不是并行版本：`v6_prior_checkpoint_payload.py`隔离Program/precision tensor
codec，`v6_prior_contract_spec.py`隔离sealed static science contract，
`pi05_eval/anchored_reconciliation_gate.py`独占两root strict continuation decision。它们分别避免继续扩张原
checkpoint、contract和通用analysis大文件；runtime只调用gate而不复制paired逻辑。旧blind checkpoint/config/
solver没有兼容fallback；若RLS被正式退役，这三个RLS-specific owner与其tests按同一退役提交删除，历史由
Git/artifact保留。

所有未来launcher文件在启动后不可原位修改。macro25 evaluator的内部completion、72/72 jobs、400 rows、
18 worker return0和`139`结果完整，但外部wrapper `.exit`记录因launcher运行中被修改而缺失；正式记录必须
写成`external_wrapper_exit_status=unobserved_missing_record`，禁止事后合成exit code。该程序性缺口不改变
scientific row，但以后每次launch使用独立immutable script。

### 39.4 最短证据门

CPU必须覆盖首步等价、streaming与显式累计ridge等价、zero cotangent下M不变但precision/rows正常同化、negative zero motion、
重复/共线/rank-deficient feature finite，以及uninterrupted与fresh→checkpoint→resume的M、precision、cursor
完全一致。

唯一discarded A40 profile固定fresh0→3，仍为train24、六rank×4 tasks、B20/B10+10和原policy forward数；
三个macro恰好覆盖每task的reversed/shuffled/wrong。profile只临时保留此前最多48条correct features作解析
对照，不保存cotangent或历史训练状态。相对同一当次cotangent的blind reference必须同时满足：

- macro2/3旧correct-panel drift RMS均不超过blind的`.5x`；
- 至少`75%`旧correct rows的motion小于blind；
- 当前correct motion RMS至少保留blind的`.5x`；
- 原correct-retention、counterfactual-null、application closure、A/B、4-task fixed-action继续通过；
- production wall不超过sealed baseline `1.10x`，0 extra policy forward/OOM/nonfinite。

若旧drift不降，直接退役RLS；若旧drift下降但当前motion低于一半，判为P256稳定--获取冲突，不扫forgetting
factor、window、lambda或step size。profile通过后才允许唯一formal fresh0→10并立即strict correct400。
closed-loop支持门是correct`>=140`、相对macro0 lost`<=6`（保留至少约95%旧成功）且breadth`>=6`；
严格`>140`才是absolute共同提高的强证据。若内部保留门通过但closed-loop仍lost`>=13`且不升分，下一最早
接口转为functional credit/closed-loop alignment；此时才考虑第39.5的reward-credit successor。

fresh0→10启动时必须在结果出现前把唯一macro0与macro10 strict roots写入原run contract。macro0固定复用
已证明bit-exact的`6b5f7a6` 400-row root；macro10 root当时必须不存在。任何10→25 exact-resume都必须使用
同一预注册root，从raw queue/shards重聚合400 rows，核对RLS family、training/evaluation commit、macro10
checkpoint/manifest以及state、RNG、language和actual video identity，再执行上述三门；formal evaluator
只接受预声明macro10/25。macro25因此是支持门后的条件动作，不是config列出25就自动授权。

### 39.5 唯一后备方向

RLS证据完成前不实现reward路径。若D层保留被证伪，后备只允许在同一Balanced v2部署图和full48
preservation solve上，把pointwise source-action cotangent替换为K4真实train24 rollout的binary LOO、Nmc4
executed-prefix on-policy Program cotangent；不恢复旧高维SPSA、shared Adam、第二replay epoch、progress
reward或多video。它是随后单变量的credit测试，不与RLS同时打开。
