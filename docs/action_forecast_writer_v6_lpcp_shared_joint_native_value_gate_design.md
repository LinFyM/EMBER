# V6-LPCP Shared Joint Native-Value Gate

状态：2026-08-15 terminal mechanism non-pass，禁止full24、strict、resume或参数小扫。简称 `SJNV-Gate`。本设计建立在 NPVC full24 终局和
PAFS-NV mechanism non-pass 之后。它保留 LPCP143 的完整视频 carrier、K-set、V6 compiler、38-target rank16
LoRA、NPVC 已证明可跨 held compiler 的 ordered native probe Value，以及 matched selected-success reward；只替换
`language/video evidence -> factor hidden residual` 的最后选择接口。

## 1. Decision

NPVC 已经关闭了此前最重要的机制断裂：不同 correct K4 videos 在 validation8 上能够形成
`.4494/.5715` 的共同 native-scale effective-BA，reverse 对 probe/BA 的 relative-L2 为
`1.8408/1.6052`。因此下一轮不应替换 carrier、增加第二次 backbone forward、改 rank 或重写完整 LoRA
decoder。

NPVC 的失败发生在更晚位置：strict 只有 `136/400`，相对 LPCP143 为 `120 retained / 16 gained / 23 lost`；
gained/lost 的 BA 改写不可分，full24 后 train task4 的 four-view 共同方向从 `.5929/.6792` 降到
`.0569/.2951`。PAFS-NV 随后证明，固定四路 language pre-address 加八套 factor-owned selectors 会在 full24 前
就破坏 NPVC 的 held shared geometry，尤其 action family。

本轮因此不再先按 task 或 factor 切开 native Value。它让所有八个 factor families 共享同一个由语言与视频联合
形成的两系数 gate：

```text
exact language + K4 ordered action-hidden videos
  -> frozen LPCP shared Core / causal Procedure / Procedure-set
  -> ordered native Action-probe Value M[320,256]
  -> text slots L[320,256]
  -> one shared joint language-video diagonal gate g[320,2]
  -> the same two coefficients for all eight factor families
  -> frozen V6 W1/GELU policy axes and W2 outputs
  -> one complete 38-target rank16 LoRA
```

## 2. Single causal variable

令 `N(x)` 为无可训练仿射项的 RMS normalization，`S` 为固定交替符号向量。对每个 condition 与 policy slot：

```text
Lhat_s       = N(L_s)
J_s          = M_s elementwise_mul Lhat_s
g_s          = W_gate J_s                         # W_gate in R^(2 x 256)
A_f,0(L_s)   = GELU(W1_f Lhat_s)
A_f,1(L_s)   = GELU(W1_f (S elementwise_mul Lhat_s))
R_f,s        = g_s,0 A_f,0(L_s) + g_s,1 A_f,1(L_s)
```

`W_gate` 无 bias、exact-zero 初始化，且是本轮唯一 trainable tensor，总参数 `512`。`R_f` 加在对应冻结
FactorHead 的 hidden owner，再经原生 W2 生成 public LoRA factors。

这个公式是一个共享的 diagonal bilinear hypernetwork：`M` 提供有向视频 Value，`L` 指定任务相关维度，
`W_gate` 学习哪些 joint components 及符号在 policy 中有用。task/slot 的差异来自输入 `M elementwise_mul L`，
不是 task ID、固定四路地址或八套独立 selector。所有 factor families 使用完全相同的 `g`；q/v/action 的差异只
来自 LPCP 已训练好的冻结 V6 W1/W2 policy geometry。

与 NPVC 相比，改变的是“language-only 四路 route × 固定 dot-product coefficient”；与 PAFS 相比，去掉了低有效
维 fixed address 和 factor ownership。native Value、language slots、direct/signed frozen axes、reward recipe 与
public LoRA topology均不变。

## 3. Why this is the smallest useful Program interface

`J_s=M_s elementwise_mul Lhat_s` 是当前失败接口处的 joint task Program feature。它不是 literal backbone
memory token，也不冒充一个 token。当前没有把新 token 塞入 Action Expert，原因是 LPCP/NPVC 已经用真实图像、
语言和50个 Action probes证明 carrier、顺序与 held native Value 均有内容；此时改变 token sequence 会同时改变
carrier、numerics 与 commitment，不能干净检验当前断点。

memory token 仍是开放的规模化架构候选。如果本轮共享 gate 保住 NPVC held geometry，却仍无法形成稳定多task
credit，后续可以让真实 joint-context memory 提供更丰富的 shared Program；如果本轮连 held gate 都不能通过，
则说明 `M x L` 的当前共同特征不足，届时 literal memory 才直接针对表示接口。无论哪种结果，都不把 memory
本身写成项目目标。

## 4. Information wall and causal order

- `M` 只来自同一次真实 image/language/Action-probe forward、每条 video 内的 causal temporal processing，以及
  video-set 间的置换不变聚合；
- `M=0` 时 `J=g=R=0`，language 不能独立写新增 LoRA；
- language 只选择视频 Value 的任务相关成分，不携带 task ID、expert route、filename 或第二套 LoRA；
- shuffled/reversed 必须重排真实frames并完整重算 `M`；constant frames 必须使新增响应近零；
- video 与 action query 继续同task跨episode；Writer在rollout前运行一次，policy执行期间不看视频；
- K1--K4 使用同一图，当前 matched formal仍以K4裁决；本轮不同时改变dynamic-K schedule。

correct order 的结构作用继承自 LPCP：`M` 来自 ordered native Action-probe deltas，reverse 已被证明产生 material
变化。SJNV-Gate 不能只制造内部 order margin；若进入稳定资格，仍须以 correct 相对 shuffled/reversed 的严格
paired closed-loop优势证明顺序有用。

## 5. Multi-task coexistence hypothesis

PAFS 的每task梯度先落入一个低有效维 fixed address，再被八套 selectors 分裂；NPVC 则让所有task先通过
language-only common router。SJNV-Gate 的单task梯度具有：

```text
gradient(W_gate) = sum_s local_policy_cotangent_s outer J_s
```

不同task和video由其真实 joint feature `J_s` 决定更新坐标，而所有families又共同约束同一个512维 gate。这既允许
task-conditioned分离，也迫使q/v/action只保留可由共享任务程序解释的成分，避免 PAFS 的family-local
overfitting。这是可证伪假设，不是“参数少就一定稳定”的主张。

full24 保留每个active task的512维梯度行，在global mean前只报告 pairwise cosine、task-to-mean cosine/dot 和
positive-descent coverage。该矩阵很小，不新增forward、逐tensor扫描或gradient surgery；它只解释strict结果，
不选择方法。

## 6. Historical inheritance and non-repetition

- **保留 LPCP/NPVC**：共享Core、有向Procedure、K-set、native probe Value、rank16和冻结V6 FactorHeads全部不变；
- **不是 PAFS 小修**：没有basis、temperature、fixed address或factor selectors，不能加载PAFS state；
- **不是 CCT**：不把视频压成固定dot-product后沿language-only route transport；学习的是共享joint bilinear gate；
- **不是 SFMC/GOSC**：没有八个256x256 family maps，不把condition-local arbitrary hidden direction分别写入factor；
- **不是 Dynamic-K rank8 memory 重跑**：不删除V6 Semantic Core、不改rank、不恢复fixed-A/wide mapper；
- **不是 support guard/PCGrad**：不投影、裁剪或保护task gradient；先检验表示本身是否产生可共存raw mean；
- **不是 scale sweep**：gate的512维方向和输入依赖是新的结构变量，global scale、LR、rank与seed保持matched。

## 7. Canonical implementation and lifecycle

1. `src/ember/writer/factor_commitment.py` 原位替换 PAFS commitment；不新增第二个 Writer 或 strategy registry；
2. `model.py` 删除 basis-weight executable semantics，只把同一个 `M,L` 交给共享 gate；LPCP reference path不变；
3. reward runtime只训练 `factor_commitment.gate.weight` 的512参数；PAFS-specific per-family evidence退役为generic
   shared-gate coexistence evidence；
4. 新config/checkpoint/evaluator schema fresh-incompatible；LPCP只作为明确cold start，NPVC/PAFS state不得加载；
5. 历史实现由commit、旧config与formal artifacts保留，active tree只留SJNV-Gate；
6. 不增加backbone forward、baseline forward、dtype扩展、hash或大规模防御性instrumentation。

canonical实现已原位完成：唯一trainable tensor为`gate.weight[2,256]`；定向CPU=`76 passed`、完整CPU=
`399 passed`、compileall通过，architecture guard无hard violation。以上只关闭实现门，不替代真实GPU机制与
closed-loop裁决。

## 8. Fast mechanism and throughput falsifiers

CPU/synthetic先证明：

1. `W_gate=0` 时public LoRA逐tensorexact LPCP；只有512个trainable parameters；
2. `M=0`、constant/no-video时新增response为零或门内近零，language-only不能写LoRA；
3. same-task跨video共享language但gate随video Value变化；K-set换位不变，natural/reverse material；
4. 首次synthetic与真实selected-success update中gate gradient/parameter delta finite且非零，全部base参数0 gradient；
5. post-update q/v/action effective BA与fixed-action response均非零。

随后只做一次matched task4 selected-success update及validation8每task四个disjoint correct K4的video-only gate：

- train task4 pure-SJNV correction cosine至少`.45`、energy至少`.58`；
- validation8 aggregate cosine至少`.40`、energy至少`.52`，至少6/8 tasks过`.20/.40`；
- validation action-family aggregate cosine至少`.20`，避免重复PAFS action collapse；
- held/train effective-BA L2至少`.50x`；
- natural→reversed BA relative-L2至少`.50`，constant/natural BA L2不超过`.005`；
- wall不超过matched NPVC task4的`1.10x`，无第二backbone forward、禁读、OOM或nonfinite。

这些阈值要求新接口大体保住NPVC而不是只比PAFS略好。任一核心held/action/order门失败即终局，不启动full24，
不扫gate scale、hidden width、basis、rank、LR、temperature或seed。

## 9. Formal training and closed-loop adjudication

机制门通过后，从sealed LPCP macro25 fresh执行与NPVC相同的full24 cycle1：24 tasks、48 paired states、96
rollouts、四个disjoint correct K4 credit views、active task等权、AdamW/LR/Nmc4/B8与rollout seeds不变。

cycle1后立即strict paired correct400。只有同时满足才exact-resume cycle2：

- correct至少`142/400`、breadth至少7；
- 相对LPCP143 lost不超过15、gained至少12且没有suite清空；
- post-train held gate仍通过，train task4共同方向不坍到`.10/.30`以下；
- strict优先裁决；task-gradient与LoRA geometry只解释，不可单独否决一个高absolute checkpoint。

稳定约145资格：

- cycle1和cycle2都至少142、两点平均至少145、breadth都至少7；
- 相邻checkpoint churn不超过20、Jaccard至少`.85`；
- final相对LPCP lost不超过10且gained不少于lost，增益覆盖多个tasks/suites；
- 不使用checkpoint union、per-task winner、LoRA平均或融合。

稳定资格通过后，在同一final checkpoint做strict paired correct/same-task-other/wrong/shuffled/reversed/no-video。
same相对correct下降不超过8；correct相对每个negative/no-video至少净高10，并在至少3 suites中paired gained>
lost。单点145或151不能跳过稳定性与视频因果资格。

## 10. Interpretation boundary

- held gate失败：当前shared diagonal `M x L` feature不足以保留NPVC几何；才考虑更丰富的真实joint-context
  memory Program，不调本gate宽度；
- held健康但full24 task4再次坍塌、task gradients强冲突：最早接口推进到显式shared-update coexistence；
- held与coexistence均健康但strict低：selected-success trajectory/occupancy credit不选择有用方向；
- cycle1高、cycle2高churn：共享gate仍未形成稳定积累；
- 稳定约145且six-arm健康：即使未到151，也达到owner认可的有价值结果。

负结果只淘汰“LPCP/NPVC native Value + shared 2x256 joint diagonal gate + frozen V6 axes + matched one-cycle
selected-success credit”组合；不否定memory token、dynamic K、rank8、few-shot、reward learning或生成LoRA。

## 11. Terminal evidence

clean pushed `913d3d3`在gpu02物理6完成task4真实B8 smoke：candidate/reference=`2/1`、1个discordant pair、
4个credit conditions/16条互斥videos，gate delta RMS=`2.99234e-4`，q/v/action与fixed-action响应均非零；cycle=
`135.757s=.99775x` NPVC，0禁读/OOM/nonfinite。

task4 four-view cosine/energy=`.472272/.597814`，刚过train门；validation8只有`.201903/.396448`、2/8 tasks过
`.20/.40`，action cosine=`.042986`，held/train BA L2=`.452509x`，四项held门失败。相对NPVC，held
cosine/energy/L2只保留`.44927x/.69370x/.16601x`；相对PAFS只小幅提高`1.2010x/1.0633x/1.1890x`，action
甚至低于PAFS的`.053486`。

stage localization把断点进一步锁定：validation8 gate与continuous hidden residual cosine分别为
`.940337/.941165`，hidden energy ratio=`.923978`；经过冻结W2并写为native BF16 public factors后，raw factor
delta cosine/energy骤降到`.021353/.265925`，action factor cosine仅`.002672`；effective BA只部分恢复到
`.201903/.396448`。因此最早失败接口不是joint gate没读视频或hidden不共同，而是**coherent continuous hidden
residual -> frozen W2 -> native BF16 public factor delta**。reverse BA relative-L2=`1.24491`且constant/natural=
`0`，顺序/static门仍健康，但不能挽救held commitment。

终局artifact为同一smoke root下的`sjnv_gate_mechanism_gate.json`、`sjnv_gate_stage_localization.json`与
`sjnv_gate_terminal_analysis.json`。该结果说明PAFS失败不只来自factor ownership：学习一个低维joint projection
本身会丢掉NPVC parameter-free native coefficient与video-dependent base-factor几何之间的必要协变。下一轮不得
通过放大本gate、扫LR/scale/rank或直接resume来绕过此结论。
