# V6 Shared-Core Ordered-Procedure Common-Value Bridge

状态：2026-08-14 active design authority。本文授权在唯一canonical Writer path中，以历史v6-fast macro400为
冻结底座，恢复已严格得到`139/400`的Shared-Core边界，并只把跨视频Procedure-Set的Value从centered residual
改为raw common ordered Procedure。不得从Common-Value、Semantic-Core Set或旧Procedure-Set checkpoint续训。

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
