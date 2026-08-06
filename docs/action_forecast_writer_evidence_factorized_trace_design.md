# K4 Evidence-Factorized Policy-Layer Trace M2P Writer

状态：2026-08-06设计、canonical原位实现与live A40 profile已封存，formal可从identity启动。本文覆盖
`action_forecast_writer_energy_preserving_layer_trace_design.md`的活动地位；旧方法只由Git与
sealed artifacts保存。

## 1. 决策

下一轮从functional identity fresh训练唯一一条新canonical路径：

```text
exact task language + K4 action-hidden same-task videos
  -> frozen PI05 20-group all-layer innovations
  -> raw DCT16 physical coefficients p
  -> factorized direction u + physical p + energy/consensus evidence phi
  -> one shared-evidence, dual-value policy-layer reader
  -> existing four-block layer/parameter-axis M2P
  -> one complete public rank-16 LoRA.
```

本设计不把视频删掉，也不把K4平均成一个静态task embedding。四条视频的方向与物理值仍各自
作为64个unordered tokens进入Reader；跨视频统计只为每个token提供“这个弱方向是否有多条
demonstration支持”的evidence metadata。

## 2. 两个正式反事实已经定位的根因

上一版逐`group × frequency`单位化的macro100五臂为`99/92/57/94/105`。它有强video task
specificity，但把raw high8仅`.359%`的能量放大到约一半，shuffle/reverse产生很强却无益的
写出。其correct→wrong的trace/Reader/BA/action relative-L2中位为
`1.319/.547/.715/.244`。

Energy-Preserving反事实只把V改为global-scale matched raw coefficients，训练期full24
gradient coexistence在前150步明显改善；但四点只有`67/83/74/85`，winner五臂=
`85/85/80/74/87`。correct→wrong的同一路径降为`.310/.297/.478/.146`且行为
gained/lost=`25/30,p=.590`。Reader effective policy groups从约13.97降至10.63。

两者共同排除两个极端：

1. 不能把每个非零方向都当作同等可靠的physical evidence；
2. 也不能让DC和高能policy groups用raw amplitude淹没弱但task-discriminative的方向。

Energy-Preserving LoRA norm/stable-rank/top-energy为`58.71/1.410/.793`，identity action
effect`.581`，因此最早问题不是LoRA增益、rank或Writer容量。shared credit在新表示的前150步
反而改善，所以现在先开sparse experts会给已经丢失的视频方向分配更多参数，接口次序错误。

## 3. 为什么不是power law、频带scale或一个gate

固定`sqrt(energy)`、手选low/mid/high频带或learned scalar mix都仍把“direction是什么”和
“evidence有多强”压成一个幅度。它们只能在两个已失败端点之间插值，无法保留一个弱但K4一致
的方向，同时压低一个同能量但跨video不一致的方向。这类单scalar也会重演项目已禁止的
checkpoint救火式scale/gate。

新设计把三种量作为不同职责：direction是可写入内容，physical coefficient是绝对支持，
energy/consensus只帮助Reader寻址。它们由vector-valued reader/fusion联合学习，不强制任一路
非零或主导。

## 4. Frozen trace与evidence分解

descriptor继续产生当前已经验证的global-total-energy matched raw DCT tensor：

```text
p[c,s,g,f] in R^1024
```

其中`c`是condition、`s in 1..4`是shot、`g in 1..20`是PI05 policy group、`f in 0..15`。
从同一`p`无参数计算：

```text
e[c,s,g,f] = ||p[c,s,g,f]||^2
u[c,s,g,f] = p / sqrt(max(e, eps))
```

exact zero保持`u=0`。再构造三个有界evidence coordinates：

1. `log_group_share`：该group能量占本video总能量；
2. `log_frequency_share`：该frequency能量占本video该group总能量；
3. `cross_shot_consensus`：`u[c,s,g,f]`与其余K-1条video同group/frequency方向和的cosine，
   zero/退化和定义为0。

两个log fraction只使用固定epsilon和固定区间缩放到`[-1,0]`；consensus天然在`[-1,1]`。
它们不读task ID、action、reward或outcome。leave-one-out和只依赖shot集合，因此K4
permutation严格不变，没有shot identity旁路。

## 5. Shared-evidence dual-value Reader

query和group/slot/frequency routes保持上一版。每个token只建立一份attention权重：

```text
q = Wq(LN(group_route + slot_route))
k = Wk(LN(u) + group_route + frequency_route + Wphi(phi))
```

`phi=[log_group_share, log_frequency_share, cross_shot_consensus]`，`Wphi`为bias-free
`3→1024`。energy与consensus只进入K；零视频即使有固定route也不能制造V。

同一attention权重分别读取两种vector value：

```text
v_direction = Wdir(u)
v_physical  = Wphys(p)
r_direction = Attention(q,k,v_direction)
r_physical  = Attention(q,k,v_physical)
r = Wfuse(concat(r_direction, r_physical))
```

三层均bias-free。`Wfuse:2048→1024`是vector-valued fusion，不是scalar gate；它允许每个
feature dimension在direction与physical support之间形成不同组合。随后仍用现有group-specific
zero-initialized output weight产生`20×68×1024`memory，再走四个zero-preserving axis blocks。

当前Writer参数`57,778,176`；新增第二value projection`1,048,576`、fusion`2,097,152`和
evidence projection`3,072`，sealed总数应为`60,926,976`。Writer参数量已无owner上限，新增量
只服务当前双证据职责，不复制完整experts。

## 6. 信息墙、训练和未来RL

- Writer输入仍只有exact task language和K4 action-hidden videos；不读teacher action、state、
  reward、terminal、task ID、filename或normalization。
- K4只联合生成一套LoRA，不逐video生成/平均LoRA，不挑video，不融合checkpoint。
- source policy、normalization、38 targets、rank16、template-A/zero-B identity、full24 B20、
  scheduler、optimizer和200-step四点裁决不变。
- architecture/config/checkpoint family fresh incompatible；不加载任何旧Writer。
- AS functional loss和未来reward/RL看到完全相同的video→LoRA计算图；没有SFT-only auxiliary
  loss、target action label或LIBERO outcome feature。

因此本设计解决的是一般的hypernetwork evidence representation：如何把弱方向与证据可靠度
分开保存和组合，不是针对监督学习loss的训练trick。

## 7. 实现与验证合同

原位替换当前Reader，不保留energy-only或unit-only runtime switch。聚焦合同必须覆盖：

1. nonzero`u`单位norm、zero保持zero，`p`保留raw group/frequency energy fractions；
2. evidence fractions有限、有界，K4 shot permutation后逐token对应等价；
3. 改变raw amplitude而保持direction时，direction V不变、physical V和energy key改变；
4. 改变direction而保持amplitude时，direction/key改变、physical energy metadata不变；
5. evidence/route不进入V，zero video完整回到identity；
6. Reader/axis shape、source freeze、全部新增参数在identity lifecycle后finite可达；
7. fresh checkpoint、exact resume、实际6-rank ownership与A40 longest105 B20 profile闭合。

profile仍在live比较`gpu01/gpu02`后只用最多6张空闲A40，固定3+3 NUMA与显式
`NCCL_P2P_DISABLE=1`。只做fresh0→1、same-root exact-resume1→3；权重弃用。通过后才从
identity formal0→200并固定评50/100/150/200。

## 8. 预注册裁决

行为优先级仍是single-checkpoint strict correct400、breadth、task churn和五臂视频因果性。
内部分析只解释：

- 相对Energy-Preserving，correct→wrong的trace/Reader/BA/action差异是否恢复；
- 相对逐token单位化，shuffle/reverse是否不再形成近正交无效操纵；
- Reader effective groups、direction/physical分支能量和task-gradient coexistence是否共同改善。

若direction恢复视频task specificity、physical/consensus抑制order噪声且absolute上升，继续同一
single checkpoint路线；若Reader仍丢失direction，最早接口仍在evidence read；只有表示与
Reader都闭合而full24 credit再次接近1/24抵消时，才打开condition-specific sparse experts。
最低目标仍严格`>150/400`，达到后继续提高。

## 10. 实现封存

canonical Reader现已原位实现上述factorization和single-attention dual-value read：旧active
config已退休，新唯一config为
`configs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_bci_v1.json`。fresh architecture、
launch与checkpoint family均与Energy-Preserving不兼容；formal在profile完成前fail-close。

实现后Writer精确trainable参数为`60,926,976`。聚焦合同覆盖unit/zero direction、有界energy
evidence、leave-one-out K4 permutation equivariance、amplitude/content分离、zero-video
identity、step1/step2梯度生命周期与hashless size/schema checkpoint；BCI assets下全仓
`192 passed`，compileall、real config load和diff check通过。

live`gpu01:0,1,2|4,5,7`六卡以B20/B2、16-frame chunk严格完成fresh0→1和same-root
exact-resume1→3。三步loss=`.150377/.152820/.148508`、grad norm=
`.000473/.000502/.000489`，0 clip/OOM/nonfinite；step2起evidence key、direction/physical
values、vector fusion、Reader和axis blocks均有finite更新。峰值allocated/reserved=
`18,218,217,984/20,470,300,672` bytes，累计1,440 action queries和288 action-hidden videos；
source trainable=0，validation/test action reads=0。profile权重永久弃用，formal只允许从新的
functional identity root启动。

identity-fresh formal0→200随后已自然完成：200 finite macros、96,000 action queries、19,200
K4 action-hidden video conditions、8 checkpoints、0 clip，wall=`7272.774s`、peak
allocated/reserved=`18,203,289,600/20,304,625,664` bytes；source trainable=0且validation/test
action reads=0。四个50步窗口的full24 gradient retention/cosine/negative-pair中位为
`.10601/.06078/.36957`、`.08578/.05152/.38949`、`.06065/.02493/.44746`、
`.05227/.00727/.47645`。该曲线只作机制证据，正式裁决仍固定为四点strict correct400。

## 9. 禁调项

本轮不改K、DCT16、rank、LR、B20、axis blocks、training objective或checkpoint schedule；
不加power-law sweep、手工band scale、scalar gate、global scale、rank/contrastive/
reconstruction loss、reward、outcome、multi-LoRA或checkpoint融合。失败后只按上述最早接口
决定下一结构。
