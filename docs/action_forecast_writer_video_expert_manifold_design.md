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

`assemble_v6_prior_evaluation_smoke_evidence`已从profile与vertical retained roots重建seal，config状态
自然转为evaluation sealed、gradient-profile ready。该实证只关闭“当前实现是否能高效完成完整闭环”
的不确定性；冻结上游、expert辅助和temporal ranking能否共同超过143/150仍完全待六卡训练与paired
closed-loop裁决。下一门是结构化gradient/resume verifier和macro49六卡profile，不能直接跳到formal。
