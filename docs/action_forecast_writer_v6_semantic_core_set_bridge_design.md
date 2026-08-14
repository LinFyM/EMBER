# V6 Semantic-Core Set Bridge

状态：2026-08-14 active single-variable design authority；canonical实现、GPU机制门与full24 B20 profile已完成，
尚无closed-loop结果。前一版
V6 Shared-Core Procedure-Set在macro25 K4 strict=`139/400`、breadth6，按门终局non-pass；其代码由commit
`64c91a4`、终局文档由`fdfc9d8`、训练与评测由formal artifacts保存，不得resume或并行保留runtime。

## 1. 最早失效接口

前两轮形成连续而非互相推翻的证据：

1. post-compiler Dynamic Slot-Set K4=`130`，same-task BA方差降`9.26x`，但task mean几乎不动；
2. 把多视频共享前移到native Core reader之前后，Shared-Core Procedure-Set K4=`139`，相对matched 130净增9；
3. 139相对K1 old134净增5，但breadth仍6，Long1贡献净7，Goal3和Long2仍为0；
4. 在同一macro25与first4×8 K4输入上把trained Procedure-Set output归零，effective-BA只变化`.000918`，
   task mean只变化`.000574`；无参数Core union相对K1则变化`.039674/.016982`。

所以“多视频在compiler前共享”是有效方向，失效的不是v6前端、rank16或few-shot本身；最早未接通接口是：每条
video已经独立形成Semantic Core后，当前只由冻结native reader做无参数union，B20 credit几乎无法通过更晚的
Procedure-Set改写共同任务语义。下一轮只移动这一个可学习边界。

## 2. 单一变量与完整数据流

冻结v6-fast的language-axial evidence、逐video Semantic Core、有向Procedure、native compiler、post-fusion和
FactorHeads全部保留；rank16、B20、full24、动态K和optimizer不变。唯一trainable集合算子仍为`197120`参数，
但从320个Procedure readout slots前移到语言对齐的Semantic Core tokens；后端Procedure只做无参数mean：

```text
exact language + K=1..4 same-task ordered action-hidden videos
    -> each video independently runs frozen v6 evidence
    -> each video independently forms language-aligned Semantic Core C_k
       and internally ordered causal Procedure P_k
    -> one permutation-invariant trainable Semantic-Core set adapter
       computes a shared correction from aligned {C_k}
    -> add the same correction to every video's Core, retaining the native Core union
    -> frozen native Core reader produces one shared 320-slot semantic state
    -> each P_k is independently read against that shared state
    -> parameter-free mean of aligned per-video Procedure readouts
    -> frozen native AdaLN/post-fusion/FactorHeads run once
    -> one complete 38-target rank-16 task LoRA
```

这不是frame、feature或LoRA平均。只有已经由每条video完整v6前端形成、且由exact language自然对齐的Core tokens
参与跨video集合比较；每条不同长度的Procedure始终保持独立物理序列，直到被native policy-slot queries读成同一
320-slot坐标后才做对称mean。非线性fusion与LoRA解码仍只运行一次。

## 3. Semantic-Core set公式

同一condition的第`k`条video给出`C_k in R[L,256]`；`L`是相同task language的token轴，因此跨video可逐token
比较。对每个语言位置`l`：

```text
mu_l       = mean_k C_k,l
center_k,l = C_k,l - mu_l
q_l        = Wq(RMSNorm(mu_l))
k_k,l      = Wk(RMSNorm(C_k,l))
a_k,l      = softmax_k(q_l dot k_k,l / sqrt(256))
delta_l    = Wo(sum_k a_k,l * center_k,l)
C_tilde_k,l = C_k,l + delta_l
```

`Wq/Wk/Wo`无bias，两个RMSNorm与前版相同，总参数仍为`197120`，`Wo` zero-init。`delta`是由整个无序video set
产生并写回每个video Core的共同语义修正；原生reader仍读取`concat_set(C_tilde_1,...,C_tilde_K)`，所以step0
严格保留已得到139的无参数Core-union路径，而不是重新初始化强底座。

K=1时`center=0`，所以任意集合参数下`delta=0`；随后Procedure mean也是singleton identity，完整76个LoRA
tensors严格等于native v6。video集合换位只重排softmax项与Core union，不改变输出。无效language tokens的
correction保持零。

## 4. 顺序、语言与高层知识

- Semantic Core负责对象、关系、目标状态等由语言对齐的高层内容；跨video set在这一层比较同一语言角色，不对齐
  速度、帧数、抓取角度或路径；
- Procedure仍由每条video自己的ordered frames、真实frame ordinals和causal encoder产生。reversed/shuffled会
  改变`P_k`，并经shared Core-conditioned Procedure reader改变最终LoRA；
- exact language既提供每条video内部的task query，也定义跨video Core token对应关系；没有language-only LoRA
  bypass，视频仍是唯一dynamic Value；
- 多video共识不能抹掉有向过程：Core set只聚合语义，Procedure不跨video拼接、不做phase alignment；
- 输出仍是一套完整task LoRA，rollout前生成一次，policy随后不再观看视频。

## 5. 为什么不做其它改变

- 不续训或放大Procedure-Set：matched归零已证明其25 macros后只产生`.000918` BA改写；
- 不把K4失败解释为few-shot失败：无参数Core boundary移动已经带来matched净增9；
- 不改rank16/factor heads：当前方法基本保留old134且接近v6-fast143，rank8 Full-Factor已在91暴露弱BA
  reparameterization；
- 不恢复memory-token mapper：Dynamic-K/Visual-Value/Full-Factor整套分别停在约100/96/91；其动态K、逐video
  保序和集合原则已被当前v6桥接继承，但弱policy geometry不应恢复；
- 不加入negative、expert、reward、consistency或RL：否则无法判断Semantic Core可学习共识是否是因果变量；
- 不生成多个LoRA再平均；Procedure mean发生在native policy-aligned readout，随后仍有一次原生非线性compiler与
  FactorHeads。

## 6. 训练合同

- development initialization仍为历史v6-fast macro400，只作最短机制开发；base与source policy全部冻结；
- train24 full24 task-equal B20 functional AS，AdamW/scheduler/precision保持不变；
- 每macro K1/K2/K3/K4各6，四个macro内每task覆盖全部K；K条video同task、action-hidden、互不重复并与action
  episodes错开；
- 每condition总frame budget420，stride5、逐video有序、K内均分预算；
- optimizer只含Semantic-Core set的5 tensors / 197120参数；K1结构上无set gradient，K2--K4提供credit；
- fresh macro0→25后立即K4 correct strict paired400；不得加载前版optimizer或macro25 checkpoint。

warm start若过门，只证明该架构边界值得保留；最终论文方法仍需在相同train24信息墙下建立从零可复现recipe。

## 7. 快速机制门

1. step0 K>1完整LoRA严格等于前版`Procedure-Set output=0`的数据流；K1在任意set参数下严格等于native v6；
2. K2--K4集合换位不变，video内部倒序显著改变Procedure、Program和LoRA；
3. zero-output首步只有`Wo`获梯度，下一步`Wq/Wk`获非零gradient/update；
4. source/v6 base无梯度，唯一trainable=`197120`；Program→LoRA→functional梯度finite；
5. full24一macro K1--K4各6、B20、最长video稳定，0 OOM/nonfinite；
6. evaluator必须重新做真实K4 B8/B16/B32 generation profile，不能沿用旧schema的B8 seal。

## 8. Strict裁决

最接近基线是matched K4 Shared-Core Procedure-Set=`139`，不是只比较old134：

- `<140`或breadth`<7`：终局停止，不resume、不补controls、不扫K/LR/seed/temperature；
- `140..150`且breadth`>=7`：只有相对139 gained>lost、至少3 suites不下降、并解锁Goal3或Long2之一，才允许
  exact-resume到macro50后再做strict400；
- `>150`且breadth`>=7`：先封存single checkpoint，再补同checkpoint K1--K4 scaling与correct/same/wrong/
  shuffled/reversed/no-video controls；
- 始终报告per-task/per-suite、breadth、retained/gained/lost、churn、top-task concentration及
  Core-set→shared Core→Procedure→effective BA最早传递；内部数值不替代closed-loop。

## 9. Ownership与生命周期

`src/ember/writer/model.py`仍是唯一canonical Writer；`src/ember/writer/slot_set.py`拥有唯一跨video集合算子，
旧`PolicyProcedureSetFusion`被原位替换，不保留flag或第二类。旧schema/config/runtime由commit`64c91a4`和formal
artifacts恢复；新checkpoint fresh-incompatible。`legacy_v6_model.py`与`temporal.py`不再扩展，因为现有分段
native compiler接口足够。实现完成后更新本文件的机制/profile证据，不修改上述预注册裁决。

## 10. Canonical implementation与CPU机制证据

实现已原位替换唯一Writer/config/checkpoint/evaluator schema：`SemanticCoreSetFusion`拥有与前版相同的两个
RMSNorm和三个bias-free `256x256`矩阵，总计5 tensors / `197120`参数；它对每个语言token从K条Core中选择
centered residual，经zero-init output形成一份shared correction并写回各video Core。native Core union reader、
per-video ordered Procedure reader、无参数Procedure mean、AdaLN/post-fusion和FactorHeads依次运行一次。

退役Procedure-Set class、schema、config和evaluator family均已删除，不保留runtime flag。新增CPU精确门证明：

- step0 K>1 Program逐元素等于前版`Procedure-Set output=0`图；
- K1在zero/nonzero Semantic-Core Set output下逐tensor等于native v6；
- K集合换位不变、video内部倒序敏感；
- zero-output首步output gradient非零而q/k gradient为零，nonzero output后q/k gradient非零；
- v6 base全部无gradient，auxiliary严格为零；
- full CPU suite=`372 passed`。

真实gpu01物理`0/1/2/4/5/6` world6 full24 B20 profile在clean detached commit
`7883fa6b71c361a28722ef9ce5047043b2966ebc`完成：macro1/2=`27.214/24.277s`，K1--K4各6，最长condition共323个
stride-5 frames且逐video无截断，peak allocated/reserved=`36.495/40.758GB`，0 OOM/nonfinite。macro1→2
query/key参数delta norm分别为`7.859e-7/7.736e-7`，证明zero-init output打开后完整q/k credit在真实训练图中接通。
GPU机制smoke同时验证K1恒等、source/v6 base冻结、倒序敏感及5个trainable tensors全部获得梯度。profile root：
`runs/outputs/pi05_v6_semantic_core_set_bridge_profile_r6_b20_7883fa6_gpu01_20260814`。

这些只证明实现、机制与吞吐合同成立；不能提前写成有效方法。formal现已seal，下一裁决仍是clean fresh
macro0→25后的K4 strict paired correct400。
