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

视频条件为`16 phase × 2048`。将它投影到512后，与168个chunk identities和16个
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
