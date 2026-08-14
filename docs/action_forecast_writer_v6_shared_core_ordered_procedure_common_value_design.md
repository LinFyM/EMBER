# V6 Shared-Core Ordered-Procedure Common-Value Bridge

状态：2026-08-14 active design authority；formal fresh macro0->25已完成，deployment profile与strict400进行中。
本文授权在唯一canonical Writer path中，以历史v6-fast macro400为冻结底座，恢复已严格得到`139/400`的
Shared-Core边界，并只把跨视频Procedure-Set的Value从centered residual改为raw common ordered Procedure。
不得从Common-Value、Semantic-Core Set或旧Procedure-Set checkpoint续训。

## 1. 为什么是这个接口

最新Semantic-Core Common-Value在macro25 K4 strict只有`133/400`。它不是因为写不进去：Core correction与
current-to-zero effective-BA已经达到`.065856/.053648`，但Long能力换给Spatial/Object。补充train-seen 8-task
严格反事实中，归零该层后为`59/80`，trained为`63/80`，paired=`57 retained / 6 gained / 2 lost`、net`+4`、
McNemar `p=.2891`。所以B20 credit并非在train on-policy上完全无效；更准确的断点是：静态/任务身份占主导的
Semantic Core common mean能获得task-local credit，却没有形成held可组合的任务程序。

历史Shared-Core Procedure-Set提供更强且更接近的起点：K4=`139/400`，相对post-compiler130净增9；其无参数
shared-Core union保留v6的语言任务锚点，而每条video仍独立保留有向Procedure。该轮trained Procedure-Set只造成
`.000918` effective-BA改写，因为近均匀attention下centered跨video Value构造性相消。尚未检验的窄假设是：
**让同一可训练Value只读取有向Procedure的共同部分，能否既打开credit，又避免Semantic Core raw mean的
language/task-identity过拟合。**

## 2. 完整数据流

```text
exact task language + K=1..4 same-task action-hidden ordered videos
    -> 每条video独立运行冻结v6 joint image/language/Action-probe encoder
    -> 每条video独立形成language-aligned Semantic Core与有向Causal Procedure
    -> 对K条Core作无序union，由native Core reader一次形成shared Core
    -> 每条有序Procedure以同一shared Core为query，native reader形成
       320 x 256 per-video policy-aligned Procedure slots
    -> 仅沿K轴聚合raw common ordered Procedure Value
    -> native AdaLN/post-fusion一次
    -> frozen native factor heads一次
    -> one complete 38-target rank-16 task-conditioned LoRA
```

320个slots的真实拓扑是`18 Action Expert layers x 16 rank + action-in 16 + action-out 16`。这里不增加虚构的
layer/token网格，也不把普通latent slots叫作memory tokens。38个LoRA targets仍由已经证明policy-effective的
原生v6 factor heads解码。

## 3. 唯一新增算子

对condition `c`的第`k`条video，原生reader给出`P_ck in R^(320 x 256)`。对每个对应policy/rank slot：

```text
mu       = mean_k(P_ck)
q        = Wq(RMS(mu))
key_k    = Wk(RMS(P_ck))
a_k      = softmax_k(q dot key_k / sqrt(256))
common   = sum_k(a_k * P_ck)
P_shared = mu + Wo(common)
```

`Wq/Wk/Wo`全部bias-free并在320个slots间共享，`Wo` zero-init；总trainable仍为`197120`。K=1走显式native
旁路，始终`P_shared=P_c1`，所以训练后也严格保持原生v6。K>1的step0则严格等于Shared-Core的parameter-free
Procedure mean，不加载其旧optimizer或set权重。

相对matched Shared-Core Procedure-Set唯一改变是`common`不再由`P_ck-mu`构成；位置、query/key、参数预算、
shared-Core union、native compiler remainder、rank16与训练recipe均不变。

## 4. 为什么正确顺序与视频内容不可绕过

原生Procedure reader的query包含shared Core，但Value严格来自video内Causal Procedure，并在时间轴上先做有效帧
中心化。若video没有过程变化，静态语言本身不能凭空产生该Value。reversed/shuffled必须先重排真实frames并重新
运行visual transition、causal encoder和带RoPE的Procedure reader，之后才进入K轴集合层；set不能恢复被破坏的
有向过程。

多video只在K轴做置换不变聚合，不拼接成一条虚假物理轨迹、不做phase alignment、不平均frames/features或最终
LoRA。shared Core负责“任务是什么”，raw ordered Procedure Value负责“视频展示了怎样完成”，两者随后仍由同一
冻结native compiler承诺policy方向。

## 5. 为什么现在不改rank、memory、mapper或直接上RL

- Dynamic-K memory/rank8完整链已得到约`100/101/102/98/96/91`，证明8个真实memory tokens、动态K和结构化
  mapper可运行，但没有保住v6的policy-effective几何；当前不恢复该弱底座。
- SHINE/Doc-to-LoRA的可扩展原则已被吸收，但v6现有320个policy/rank aligned slots与共享factor heads已经提供
  本轮所需的层/参数对应；再叠一套M2P会同时改变多个接口。
- rank14 compression与rank8 Full-Factor分别暴露support损伤和tiny-B/weak-BA；本轮没有证据把rank作为最早
  失效接口。
- train-seen反事实说明现有B20不是处处反向。先用结构把trainable Value限制到有向视频过程，比立即引入更重的
  reward execution graph更窄、更快；若本轮仍只在train seen增益而held失败，才进一步授权on-policy credit。

这些判断不把memory token、rank8、few-shot或reward credit判死；只是在当前单变量实验中冻结它们。

## 6. 训练与效率合同

- 冻结source policy与完整v6-fast Writer，只训练Procedure Common-Value的`197120`参数；
- train24、full24 task-equal、每task B20跨episode action queries与raw-mean optimizer语义不变；
- 每macro K1/K2/K3/K4各6，每task四个macro覆盖全部K；K条video同task、互不重复、action-hidden且与B20 episodes
  错开；
- stride5、final frame、每condition 420-frame cap和long-first负载均衡不变；
- BF16/TF32、真实可用GPU数、deferred NCCL、动态evaluation queue不变；
- 不加negative、expert、consistency、reward、norm/rank、reconstruction或额外预训练数据；
- 不做scale/LR/K/rank/seed sweep，不为低位一致重复forward、batch1、扩dtype或逐tensor scan。

## 7. 快速机制门

1. 任意set参数下K1 Program与76个LoRA tensors逐元素等于native v6；
2. K>1视频集合换位不改变数学输出，video内倒序显著改变Procedure/Program；
3. zero-init时K>1逐tensor等于parameter-free Shared-Core graph；
4. uniform q/k与identity output下，correction严格等于raw Procedure mean，而不是centered zero；
5. 首步output获得有限非零梯度，下一步q/k展开；source policy与v6 base始终无梯度；
6. full24 B20真实profile覆盖K1--K4各6和最长视频，无OOM/nonfinite，吞吐不明显退化。

机制门只证明假设被真实检验，不选择方法。profile通过后从clean pushed commit fresh macro0->25，立即做K4 strict
paired correct400。

## 8. 裁决与淘汰边界

- `<140`或breadth`<7`：终局停止，不resume、不扫参；
- `140..150`且breadth`>=7`：只有相对matched139 gained>lost、至少3 suites不下降，并解锁Goal3或Long2之一，
  才允许一次exact-resume到macro50；
- `>150`且breadth`>=7`：先封存single checkpoint，再补K1--K4 scaling与correct/same/wrong/shuffled/reversed/
  no-video；若正确视频不优于controls，仍不构成有效方法；
- 始终报告per-task/per-suite、breadth、retained/gained/lost、top3 concentration、train-seen/held差异，以及
  Procedure Value -> Program -> effective BA -> rollout传递。

失败只淘汰“shared Core锚点 + raw common ordered Procedure Value + unchanged B20”这一组合。若train-seen仍有
增益而held不增，下一接口明确转向on-policy/generalization credit；若Procedure correction仍近零，则才是本算子
optimization失败。不得把失败改写成“视频没被读取”“rank16太大”或“few-shot无效”。

## 9. Canonical CPU implementation

唯一active `CompleteLoRAWriter`已恢复shared-Core union与per-video ordered Procedure reader，并把旧centered
`PolicyProcedureSetFusion`原位替换为raw `PolicyProcedureCommonValueFusion`。旧Semantic-Core Common-Value
config/schema/checkpoint/evaluator由Git与formal artifacts保存，不留runtime flag或第二实现。新config先以
`unsealed_pending_live_profile`进入实测，且不继承旧profile或deployment batch seal；本节末的当前checkpoint
live profile完成后才切换为`active_formal_ready`。

定向测试证明K1在任意nonzero output下仍与native v6的76 tensors严格相等；K>1集合换位不变、video内倒序改变
Program；zero-init逐tensor等于parameter-free Shared-Core graph；uniform q/k与identity output得到raw Procedure
mean；首步只有output获梯度、output非零后q/k展开，v6 base始终无梯度。正确LIBERO assets下full CPU=
`374 passed`。active source净缩减，architecture guard只有既存大函数review信号、无新增hard violation。

真实source-policy smoke在gpu01物理7通过：K1在zero/nonzero output下均与native v6的76 tensors严格相等，
video倒序Program mean abs=`.217034`，trainable=`197120`、base无梯度、peak allocated/reserved=
`18.749/19.241GB`。K2/K4集合换位Program max abs=`.03986/.03672`、LoRA max abs=`.001953`，属于与历史
相同的BF16 batched-forward低位差异，不为此重复single forward。

clean detached `50a3c36`随后在gpu01物理`0/1/2/4/5/6`完成full24 B20 macro1/2 profile：wall=
`26.1116/22.5427s`，K1--K4各6，functional=`.10118184/.09571959`，gradient=`.00032664/.00036634`；最长
condition 323帧全部保留，peak allocated/reserved=`36.495/40.758GB`，0 OOM/nonfinite。macro1->2 q/k delta=
`.00011578/.00011831`，output norm=`.004367->.009275`，证明raw ordered Procedure Value消除了旧centered
Procedure的credit相消并在首步后展开完整attention。profile root=
`runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_profile_r6_b20_50a3c36_gpu01_20260814`；formal config
现已seal，profile checkpoint不进入formal训练。

## 10. Formal macro25与first4机制证据

clean detached `d316623`在gpu01物理`2/4/5/6/7`以fresh world5完成macro0->25；没有加载profile Writer或
optimizer state。formal root=
`runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_formal_fresh0to25_r5_b20_d316623_gpu01_20260814`。
25/25 metrics、world5 rank states、trainer state、Writer、completion与exit0完整；总elapsed=`745.622s`，纯macro
wall min/mean/max=`27.686/29.790/31.599s`。functional first/last=`.10118184/.09564162`，gradient范围=
`.00025272--.00046269`，K1--K4每macro始终各6，最长condition=`359` frames且全部未截断，peak allocated/reserved=
`36.501/40.758GB`，0 OOM/nonfinite。

macro2->25的Procedure set没有重新关闭：query/key各继续变化`.08636/.08605`，output norm从`.009275`增长到
`.277774`，三个矩阵的全部`65536`元素均发生更新。canonical validation8 first4 K4 output-zero反事实进一步显示：
raw Procedure correction相对per-video Procedure全量mean=`.09601`，attention entropy/log4=`.99443`；但经过
frozen native compiler后，current->zero effective-BA relative-L2 mean/task-mean仅=`.01397/.01392`，action
targets为`.00989`。这比旧centered Procedure-Set的`.000918`明显打开，却仍把约9.6%的Program修正压成约1.4%的
policy方向。artifact=
`runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_formal_fresh0to25_r5_b20_d316623_gpu01_20260814/procedure_common_value_mechanism_first4.json`。
该证据只解释接口，不提前选择方法；最终仍由同一macro25的K4 strict paired400裁决。

同一macro25在gpu01物理2完成固定validation8x4 correct longest-panel的完整deployment profile：B8/B16/B32=
`.2250164/.2247286/.2247036 LoRA/s`，三者stable、0 OOM/nonfinite，peak reserved=
`12.952/12.973/13.011GB`，最长视频226帧；按预注册最高实测吞吐规则锁B8。root=
`runs/outputs/pi05_v6_shared_core_procedure_common_value_bridge_k4_writer_generation_profile_val8x4_correct_gpu01p2_d316623_macro0025_retry1_20260814`。
第一次试图只测B8在任何计时forward前被正式profile合同拒绝，因为合同要求同panel完整覆盖B8/B16/B32和两次实测；
随后未绕过合同而完整重跑。当前evaluator仅seal K4/B8，下一步从clean pushed commit立即做strict paired400。
