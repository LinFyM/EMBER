# V6 Semantic-Core Common-Value Set Bridge

状态：2026-08-14 预注册，尚无新训练或closed-loop结果。上一轮Semantic-Core Set已在macro25 K4 strict=
`135/400`、breadth7终局non-pass；其代码与artifact由commit`850bd38`保存，不得resume或用scale/LR/seed救援。

## 1. 最早失效接口

上一轮只把同预算set前移到语言对齐Semantic Core，底座、rank16、B20、K1与后端native compiler均保持不变。
formal与strict合同完整：

- correct=`135/400`、breadth7，per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
  `1/2/46/30/0/35/20/1`，per-suite=`3/76/35/21`；
- 相对matched Shared-Core Procedure-Set 139为`120 retained / 15 gained / 19 lost`、net`-4`、churn34；
  suite net=`-2/-4/-1/+3`，Long2从0到1但Goal3仍为0；
- 归零trained Semantic-Core Set output后，first4×8 effective-BA只变化`.001763` relative-L2，task-mean
  只变化`.001472`；相对matched139也只有`.001746/.001439`；
- 更早的原始Core诊断显示，trained correction相对完整Core仅`1.8275e-5`，相对centered Core仅
  `.0003862`；K4 video attention entropy/log(4)=`.9998850`，几乎严格均匀；
- native reader/compiler没有继续压小该信号，而是把约`1.83e-5`的Core变化放大为约`.00176` BA变化。小修正
  已足以造成34条closed-loop churn，但没有形成稳定净增。

因此最早失败不在视频前端、Core/Procedure顺序、rank16容量或compiler增益，而在当前set的Value定义：

```text
M = mean_k C_k
alpha = softmax(<Wq M, Wk C_k>)
R_centered = sum_k alpha_k (C_k - M)
delta_C = Wo R_centered
```

当attention近均匀时，`R_centered`按构造相消为零。跨视频共有的高层语义只通过冻结的无参数mean/union流动；
trainable set只能学习video-specific deviation，不能学习共有内容。这正是项目想解决的变量，而不是compiler之后
再加一个更大residual。

## 2. 唯一架构变量

只把Semantic-Core Set的Value从centered residual改为同一attention下的raw common value：

```text
M_l       = mean_k C_{k,l}
q_l       = Wq RMS(M_l)
key_{k,l} = Wk RMS(C_{k,l})
alpha     = softmax_k(<q_l, key_{k,l}> / sqrt(d))
V_l       = sum_k alpha_{k,l} C_{k,l}
delta_l   = Wo V_l
C'_{k,l}  = C_{k,l} + delta_l                    when K > 1
C'_{1,l}  = C_{1,l}                              when K = 1
```

`V=M+sum alpha(C-M)`：均匀attention时首先暴露跨video共有Semantic Core；若q/k后来学到差异，仍可在共有
backbone上选择video-specific evidence。K1显式结构旁路，任意set参数下都严格等于native v6。

除此之外全部不变：

- exact language + K=1..4 action-hidden ordered videos；每条video独立跑冻结v6 evidence/Semantic Core/Procedure；
- native Core reader读取无序K-video Core union；每条有序Procedure在同一shared Core下解释并作无参数mean；
- native AdaLN/post-fusion/38-target rank16 FactorHeads只运行一次；
- 同样5 tensors / 197120参数、zero-init `Wo`、full24 task-equal B20、AdamW/scheduler/precision与动态K schedule；
- 不加scale、gate、额外loss、negative、expert、reward、RL、rank变化、memory数量变化或新mapper。

## 3. 为什么符合EMBER目标

- 共有Value来自每条真实video经冻结v6形成的language-aligned Core，不新增文本直接写LoRA的路径；exact language
  仍说明关注对象/关系，video Core提供具体正确示范内容；
- 不是平均frames或分别生成LoRA后再平均：每条video先独立保序编码，set只在对齐后的高层语义token上做一次
  permutation-invariant attention，随后有向Procedure继续决定阶段顺序；
- raw common Value能学习same-task videos的交集；attention residual仍允许保留对少数有用差异的选择；
- 完整LoRA仍在rollout前只生成一次，policy执行时不再看视频；
- K1部署保持历史v6几何；K>1才允许从跨video共同证据产生额外知识。

主要风险是raw common Core含有较强静态任务语义，可能放大language/static shortcut。它不是无视频Value，因为
`C_k`只由真实video encoder产生，但若absolute过门，必须用same/wrong/shuffled/reversed/no-video判定增益是否
真正来自正确教学内容和有向Procedure；不能只凭correct高分保留。

## 4. 快速机制与效率门

1. step0 K>1完整LoRA逐tensor等于上一轮output-zero图；K1在zero/nonzero任意set参数下等于native v6；
2. K集合换位不变，video内部倒序仍改变Procedure、Program和LoRA；
3. zero-output第一步`Wo`梯度非零、q/k为零；output打开后q/k非零，base始终无梯度；
4. 直接单元门证明uniform q/k时Value等于raw mean而非零centered residual；
5. full24 B20真实profile保持K1--K4各6、最长视频无截断、0 OOM/nonfinite；
6. 当前只删一次center subtraction，tensor shape、kernel topology和峰值内存不变。deployment沿用matched
   predecessor已实测稳定且最高吞吐的B32，并用新checkpoint做一次真实K4 B32 longest-panel finite/OOM确认，
   不重新花21分钟比较已经由同形图裁决过的B8/B16/B32。

## 5. 训练与strict裁决

从冻结v6-fast macro400 fresh初始化，只训练Common-Value set；不得加载Semantic-Core Set optimizer/checkpoint。
clean profile seal后fresh macro0→25，随后立即K4 correct strict paired400。

- `<140`或breadth`<7`：终局停止，不resume、不扫scale/LR/K/seed/temperature；
- `140..150`且breadth`>=7`：只有相对matched139 gained>lost、至少3 suites不下降，并解锁Goal3或Long2之一，
  才允许一次exact-resume到macro50；
- `>150`且breadth`>=7`：先封存single checkpoint，再补K1--K4 scaling及correct/same/wrong/shuffled/
  reversed/no-video；若高分主要来自static/language shortcut则仍不通过；
- 始终报告per-task/per-suite、breadth、retained/gained/lost、churn、top3 concentration，以及
  Common-Value→Core correction→effective BA→rollout传递。

## 6. 淘汰边界

失败只淘汰“raw common Semantic Core作为同预算trainable Value、仍在native Core reader前注入”这一组合。
它不淘汰dynamic K、few-shot、Semantic Core/Procedure分解、memory-token Hypernetwork或未来reward credit。若
correction已经明显打开但closed-loop仍换手，下一最早接口才是offline B20 credit与held on-policy方向对齐；若
correction仍近零，则应重新设计set的optimization/parameterization，而不是继续搬动注入位置。

## 7. Canonical CPU implementation

唯一`SemanticCoreSetFusion`已原位改为raw attention-pooled Value，并对K1增加显式无运算旁路；旧centered Value、
旧checkpoint/evaluator/config schema与旧formal profile seal均由Git/artifact保存，不保留runtime flag。新增精确门
证明uniform q/k + identity output时K>1 correction逐元素等于raw Core mean，而K1在同样非零output下correction
严格为零；既有step0、K集合换位、video内倒序、gradient staging、base freeze与native compiler等价门全部通过。
首次world6 profile在首个K1 task被训练器拒绝，因为显式K1旁路正确地产生零导数，而通用训练门把“零导数”误判
成“图断裂”；没有完成macro或产生可用checkpoint。修复只允许`K1 + exact_zero_no_auxiliary_loss + weight=0`返回
数学上的零bridge梯度；K1 functional loss仍记录、full24仍除以24，K>1无图仍fail-closed。定向回归与正式环境
full CPU suite=`374 passed`。

clean detached `2eb9da9efae0cead6e0d936172eed7165ea6b8bf`随后在gpu01物理`0/1/2/4/5/6`
完成world6 full24 B20 macro1/2 profile：`25.930/22.530s`，K1--K4各6，最长condition 323个stride-5
frames且逐video无截断，peak allocated/reserved=`36.495/40.758GB`，0 OOM/nonfinite。gradient norm分别为
`.002698/.002795`；macro1→2 query/key delta=`6.552e-6/6.480e-6`，output norm从`.004492`增至
`.009649`。相对上一centered路径约`3.25e-6`的同图profile梯度，raw Common-Value已把可训练credit打开约三阶。
profile root：
`runs/outputs/pi05_v6_semantic_core_common_value_set_bridge_profile_r6_b20_2eb9da9_gpu01_20260814`。

这些只证明设计所针对的Value抵消已消除、真实训练图和吞吐合同成立，不是closed-loop性能证据。formal config现已
seal；下一步必须fresh macro0→25，不加载profile或旧Semantic-Core Set optimizer/checkpoint。
