# Video-Conditioned Expert-Manifold Topological Writer

状态：2026-08-07建立的当前唯一活动设计authority。它从同一frozen source policy和
functional identity fresh构建，不加载K4 Phase-Aligned、v6或任一历史Writer checkpoint。
K4 Phase-Aligned只由Git、正式artifact和其负裁决文档保留。

## 1. 结论先行

本方法不忽略视频，也不用language-only route直接生成LoRA。唯一dynamic value是
frozen π0.5对“exact task language + action-hidden teacher video”的联合高层内部表示相对
matched text/no-image baseline的视频innovation。该innovation保留时序轴，被直接编译为
一套rank-16 public LoRA；把视频值置零必须精确回到source identity。

根本训练变化是不再让fresh hypernetwork只从高方差的action functional loss猜测哪些
1.29M公开A/B坐标是closed-loop有效方向。先在24 train tasks上用同一source policy、
同一rank-16 topology和各自teacher actions训练24套task experts；再让Writer仅从
language+action-hidden video重建这些已经policy-effective的LoRA。Writer本身不读取
teacher action，validation/test actions在两个阶段都不得读取。

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
- meta训练每macro覆盖train24，每rank固定4 tasks，六rank DDP只在最后local microbatch同步，因而
  DDP平均严格等于24-task等权mean。模型、optimizer、scheduler、每rank RNG和macro cursor原子
  checkpoint；模型与cache完成local CUDA构造后才建立NCCL，BCI仍显式要求
  `NCCL_P2P_DISABLE=1`。
- canonical evaluator新增独立Expert-Manifold adapter：每个rollout按50-state无放回schedule只取一条
  action-hidden video，correct/same/wrong/shuffled/reversed共享state、policy RNG、video ordinal与
  frame-order seed；online frozen encoder和topological Writer先生成episode LoRA cache，随后释放
  Writer并复用同一source policy做cost-balanced rollout。validation/test只开放video，不开放expert
  或action。

以上只完成CPU合同与代码，不构成A40 profile或性能证据。feature extraction与meta训练formal仍由
config阻塞；必须先完成live profile、fresh0→1、exact-resume1→3和原始六rank规模验证。K4旧
executable只在新meta-Writer通过profile后按第12节退役。

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

## 19. Phase-centered dynamic value与no-video裁决

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

据此，在第一次meta GPU profile前原位收紧唯一canonical decoder，不保留旧可执行分支：

```text
routing_memory_t = W(video_innovation_t)
K_t = RMSNorm(routing_memory_t) + phase_key_t
V_t = routing_memory_t - Mean_phase(routing_memory)
LoRA = topological_axial_decode(Q_chunk,rank, K, V)
```

完整joint+Action-Expert特征仍参与phase key/routing，但只有phase-centered video dynamics能提供
content value。任意zero innovation或所有phase完全相同的非零输入都必须精确输出template-A/
zero-B identity；正常ordered输入仍能通过phase keys改变完整LoRA。CPU原型和retained tests验证了
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
