# Policy-Lane Coupled Hyperdecoder Writer 设计

状态：2026-08-05 fresh architecture authority；canonical实现、BCI A40六卡最长视频
profile、独立exact-resume与正式fresh0→200均已完成，四点strict correct400待运行。本文在 Policy-Wide Atom Dictionary
完成 fresh0→200、四点 strict correct400 和全部内部分析并正式负裁决后建立。新方法从
functional identity fresh训练，不加载PWAD、v6或任何历史Writer checkpoint；PWAD只由
Git、本文引用的formal artifacts和原design保留。

## 1. 结论先行

下一条架构不再分别生成两张`rank × atom` mixing matrix，也不再把一个A atom和一个B
atom通过所有交叉组合编译成公开LoRA。它把**一个完整public rank lane跨全部38个policy
targets的A/B向量**作为最小条件生成单元：

```text
task language + exactly one action-hidden teacher video
  -> Semantic Core + causal Procedure
  -> 16 condition-dependent lane states Z_r in R^256

for each public lane r:
  H_r(x) = GELU(W_in[r] Z_r(x))                    # width 32
  Delta-A[r, all policy targets] = W_A[r] H_r(x)
  B[r, all policy targets]       = W_B[r] H_r(x)

public A[target, r, :] = template_A[target, r, :] + slice_target(Delta-A[r])
public B[target, :, r] = slice_target(B[r])
```

`W_A[r]`与`W_B[r]`分别输出该lane在所有q0--q17、v0--v17、action-in和action-out
targets上的完整向量，但共享同一个lane hidden state。16个lane拥有独立参数；一个lane
内部的全部policy targets由同一个condition code共同协调。部署输出仍只有一套完整
rank-16 public LoRA。

## 2. PWAD的正式负裁决

PWAD fresh0→200训练本身完整健康：200 macros、96,000 logical queries、4,800 one-video
conditions、8个checkpoint，0 OOM/clip/nonfinite，validation/test action reads均为0。
50/100/150/200 strict correct400为`77/71/80/80`，breadth=`5/6/5/5`；四点success
union/intersection=`115/44`、single envelope gap=`35`，相邻gained/lost=
`19/25,21/12,16/16`。它远低于v6-fast`143`和严格门`>150`，也没有形成单调或共同
累积曲线，因此禁止resume到400。

内部分析排除了“字典根本没打开”：

- 64/64 storage atoms全部active，storage effective count从`63.62`到`63.93`；
- 每个condition的combined effective atom participation从`50.50`增到`54.19`，按storage
  norm加权仍为`47.67→53.49`；
- 38个policy targets全部有能量，target effective count约`12.73→15.26`。

真正失效的是condition到public lane的编译：A/B mixing的mean stable row rank在四点均
只有约`1.000002`，首奇异值能量约`.999997--.999998`；随之effective LoRA stable rank
约`1.0000002`、首奇异值能量约`.9999998`，q/v B-column cosine约`.999998`。这不是参数
本身被初始化成低秩：macro200的16个coordinate-query row stable rank为`10.73`，A/B
有效key projection仍有rank99=`58/59`，但实际condition readout几乎只有一个lane方向。

视频差异同样被共同主写入淹没。same-task video mixing centered/sample variance仅约
`.022--.054%`，effective BA约`.022--.047%`；identity→demo0 fixed-action效应随训练增大，
而demo1/reversed/shuffled的条件差异整体缩小。action projection能量占比又从
`.1197%`降到`.0136%`。PWAD学到的是越来越强的task/common q-dominant adapter，不是
稳定的视频条件policy program。

## 3. 最早失效接口的数学修正

PWAD把下式称为policy-wide rank-one atoms：

```text
A_r = sum_k M_A[r,k] A_k
B_r = sum_j B_j M_B[r,j]
```

但它的真实effective update为：

```text
sum_r B_r A_r
  = sum_r sum_j sum_k M_B[r,j] M_A[r,k] B_j A_k
```

所以A/B atom身份没有保持，`j != k`的全部交叉项都进入policy update。一个完整policy
direction并不是存储单位；public lane的差异完全依赖两张共享mixing matrix。实际训练中
两张matrix的16行又同时塌成近同向，因而宽字典只能生成一个重复的公共方向。

本方法直接让每个public lane产生一对完整A/B向量。有效更新恢复为：

```text
Delta-W[target,x] = sum_r B[target,r,x] A[target,r,x]
```

一个lane内没有第二层atom cross-product，也没有独立A/B coefficient system。lane可以
自然学成同向、低rank或不同方向；架构不强迫谱或正交，但不再硬性要求所有lane通过同一
共享dictionary readout才能彼此区分。

## 4. 为什么采用policy-lane ownership

Target-Owned Factor证明：把每个policy target完全拆开可以把跨layer余弦降到约0，却只得
`99`，因为不同target的变化没有形成协调的policy方向。PWAD相反，policy target由共同
atom index协调，但rank lane没有自己的输出参数。当前设计取两者中已被证据支持的部分：

- 一个lane的hidden state同时生成全部38 targets，保留policy-wide协调；
- 16个lane各自拥有`W_in/W_A/W_B`，避免rank差异只存在于共享attention/mixing行；
- A/B由同一个`H_r(x)`驱动，条件组合在进入policy输出前保持耦合；
- 每个lane hidden width固定32，提供condition-dependent方向而非一个static方向乘scalar。

这里不复制Source-SFT权重、layer profile或谱。direct SFT只给出判别证据：两套SFT的mean
target stable rank也仅`1.505/1.517`，但q/v跨layer方向近零相关、layer-energy profile
高度复现且top-4占`46--59%`。因此目标不是强制高rank，而是让低秩更新仍能形成稳定、
policy-coordinated的跨target组织。

## 5. Fresh identity与梯度阶段

所有`W_A[r]`和`W_B[r]` final projections exact-zero，公开A仍为sealed random template，
公开B为物理零。step0逐tensor严格等于source policy functional identity。

真实BA loss的自然梯度阶段为：

1. 第一次backward只有`W_B`获得非零梯度；
2. B打开后，lane hidden/input、Core/Procedure和`W_A`开始可达；
3. A residual打开后，全部lane A/B与condition frontend共同端到端训练。

不使用手工非零B、B-only residual、static bypass、scalar gate、rank/orthogonality loss或
额外监督head跳过该阶段。

## 6. 参数、训练与A40合同

首版固定16 lanes、hidden width32。输出宽度由真实38-target topology唯一计算，不写死旧
layer数或tensor长度。真实decoder为`41,320,448`参数，完整Writer为`49,041,664`；下文
longest105 profile已确认它可以在BCI单卡约46GB边界内维持logical B20/full24。

训练recipe保持PWAD单变量对照：

- 24 train tasks完整full24 raw mean；
- 每task exactly one video、logical B20 independent action queries；
- six ranks、每rank 4 tasks、policy physical microbatch2；
- AdamW、fast-decay400、每25保存；
- source policy/normalization冻结，validation/test action gradient为0；
- fresh0→200后评测50/100/150/200 strict paired correct400。

不加入reward、policy anchor、多video、contrastive/order loss、SFT distillation、task-ID
supervision、checkpoint融合或新optimizer。AS与RL只是同一最终LoRA的不同credit来源；
架构没有监督专属输出。只有fresh AS同时证明absolute、retention和视频传递健康后，才允许
关闭action入口对同一架构做reward校准。

## 7. 实现与profile门

canonical实现必须原位替换`policy_dictionary.py`及其model wiring；PWAD executable path由
Git保存，不保留双路runtime。新launch/checkpoint/config family与PWAD不兼容。聚焦CPU
合同至少覆盖：

1. 真实38-target flatten/slice和rank16 A/B shape；
2. 16个lane各自拥有完整policy输出，参数无意外共享；
3. A/B读取同一个lane hidden，condition改变可改变完整LoRA；
4. step0 exact identity及真实BA loss的三阶段梯度；
5. source trainable=0、信息墙、full24 ownership与checkpoint拒载不变。

六卡profile必须覆盖最长105-frame、logical B20、三步finite、0 OOM/clip，并完成独立
fresh0→1→exact-resume1→3。46GB不通过时先用activation checkpoint、encoder chunk或
科学等价的microbatch适配；不得减少logical B20、full24或修改objective掩盖OOM。

## 8. 判定

closed-loop仍是唯一性能裁决，不为stable rank或lane差异设人工越高越好的门。内部报告
lane hidden/task/video variance、lane output participation、q/v/action与target energy、
effective BA谱、B-column关系、same-task video与fixed-action传递。

- 若lane输出自然保留低rank但absolute/retention显著恢复，说明完整policy-lane存储单位
  比shared atom mixing更正确，再判断是否需要reward校准；
- 若lane参数确实产生不同policy方向但closed-loop仍低，最早问题上移到AS functional
  credit与closed-loop有效流形，不通过加width、scale或正交去救；
- 若输出仍由task/common condition主导且视频差异继续衰减，下一步重做condition
  composer的语义拥有权，而不是增加lane hidden width；
- 若曲线未超过PWAD且能力继续换手，fresh0→200即负裁决，不续400。

## 9. 当前实现与A40 profile状态（2026-08-05）

canonical PWAD runtime已原位替换：`policy_dictionary.py`删除，新增凝聚的
`policy_lane.py`；model、architecture、config、launch/checkpoint family、task-gradient
ownership与内部analysis均切换到新方法，旧PWAD checkpoint/config不再被活动loader接受。
没有保留双路Writer。

真实38-target topology给出每lane A/B总输出宽度`37,920/42,528`。16 lanes×hidden32的
hyperdecoder为`41,320,448`参数，composer为`660,224`，完整Writer为`49,041,664`；参数增长
全部对应policy-lane condition-to-output容量，没有新增loss、输入或训练分支。

聚焦Writer合同`84 passed`，覆盖完整参数枚举、source freeze、38-target slicing、step0
exact identity、condition写出、真实BA梯度阶段、新config/launch/checkpoint family、task
gradient ownership与lane analysis summary。py_compile与diff check通过；architecture guard
无hard violation、无parallel version/function family。

clean pushed implementation`2aeb22a`在`gpu01`六张空闲A40上完成longest105、logical
B20、full24三步profile。rank/NUMA为`3+3`，显式`NCCL_P2P_DISABLE=1`；step max wall=
`33.457/31.024/31.007s`，峰值allocated/reserved=`36,168,858,624/47,053,799,424`
bytes，0 OOM/clip/nonfinite。三步累计1,440 logical queries与72 one-video conditions；
step1按zero-B阶段只有policy-lane梯度，step2起Semantic Frontend、Core、Program、Composer、
Policy-Lane五个主块全部非零，source policy保持冻结。

独立root又完成fresh0→1，再在不共享他人进程且保持同一`3+3 NUMA`合同的另一组空闲物理卡
上exact-resume1→3；optimizer、scheduler、sampler、RNG、task-cycle与六rank state均由
runtime恢复合同验证，最终仍为1,440 queries/72 conditions/3 scheduler updates。profile与
resume合同SHA为`f0f3ec32...55261`。profile/smoke权重永久不进入正式轨迹；sealed config
已开放clean/pushed代码上的独立functional-identity fresh0→200。

## 10. 正式fresh0→200完成

clean/pushed launch commit`244b677`在`gpu01:0,1,2,4,5,7`以同一未恢复进程完成200
finite macros：96,000 logical action queries、4,800 one-video conditions、每25共8个完整
checkpoint，wall=`6651.965s`。最终峰值allocated/reserved=`36,174,262,272/
42,150,658,048` bytes，200步均0 OOM/clip/nonfinite/collective stall；source policy
trainable=0，validation/test action reads均为0。run contract SHA为`a8ce75f2...00f6`，root为
`runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805`。

训练完整性只开放预注册四点50/100/150/200 strict correct400，不构成性能通过。不得依据
functional loss约`.1504→.0941`选择checkpoint，也不得在rollout前resume400或加入新trick。

## 11. 四点strict correct400正式负结果

50/100/150/200四点correct=`70/63/37/61`、breadth=`6/4/6/6`；逐点均为400 rows、
42 shards、一次launcher、全部worker exit0、每task 50 unique无放回视频。相邻严格配对
gained/lost=`17/24,14/40,40/16`，四点union/intersection=`117/14`、single envelope
gap=`47`，共同policy-noise prefix全部一致。

macro50 single winner仅70，低于PWAD80、v6-fast143和严格门151；macro150降到37后
macro200恢复到61，既没有共同累积，也没有缓解task drift。按第8节预注册门正式禁止
resume400、增加hidden width、调scale或从任一checkpoint warm-start。

当前只开放同一clean canonical owner的四checkpoint内部分析，报告lane hidden、lane
output participation、effective BA谱/视频方差、checkpoint churn和fixed-action传递。
训练ledger的后段same-task相邻CountSketch已给出Policy-Lane输出方向复现偏低的初步信号，
但该信号受block维度与sketch方差影响，不能替代真实LoRA/action证据。完整分析前不裁决
下一architecture。

## 12. 内部分析与最终负裁决

clean`3869d20`六卡formal analysis完整生成96/96 task×checkpoint cells及6/6 rank
payload，wall=`318.446s`、peak reserved=`19,295,895,552` bytes；target action、
validation/test reads均为0。root为
`runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_internal_all4_r6_20260805`。

Policy-Lane没有重演PWAD的假容量：四点storage effective lanes约`15.96--15.97`，
demo0 hidden/output effective lanes约`11.64--12.50/9.57--10.85`，hidden row stable
rank约`4.15--4.41`。effective LoRA stable rank=`1.336/1.409/1.507/1.542`、top
singular energy=`.809/.766/.727/.707`。对四点各400个held LoRA cache的精确
gauge-invariant复核又给出q/v跨layer signed cosine约0、energy CV约
`.75--.83/1.03--1.15`、top4 energy约`47--52%/58--61%`，已经达到direct SFT的层
专门化量级。

但condition ownership朝错误方向发展：cross-task demo0 hidden centered/sample energy
从`.503`升到`.660`、pair cosine从`.488`降到`.313`；same-task video hidden/BA centered
energy却始终只有`.046--.059%/.017--.023%`。macro50的demo1/reversed/shuffled差异到
effective BA为`.0176/.0281/.0133`，到fixed action仅`.00577/.00977/.00597`。
模型能区分train task并产生多lane、跨layer专门化LoRA，却没有从单条视频获得闭环可累积
的内容信用。

因此第8节第二种失败分支成立：lane参数确实产生不同policy方向，但absolute、retention和
task drift全部更差。正式裁决不是继续加width、lane、store、rank或SFT几何约束，而是把
最早失效接口上移到AS functional target对condition的不可辨识性及其与closed-loop有效
方向的错位。Policy-Lane禁止resume400、warm-start或局部修补；下一方法必须直接改变
Writer/LoRA生成层获得闭环相对credit的方式，同时保持one-shot信息墙和IL/RL通用性。
