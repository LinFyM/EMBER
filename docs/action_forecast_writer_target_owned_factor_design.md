# Policy-Target-Owned Factor Writer

状态：2026-08-04 BCI fresh architecture authority。本文在 Semantic Direction Store
完成 `129/107/120/129` 的正式负裁决、winner 内部分析闭合，并重新复核两套
rank-128 Source-SFT 的 effective LoRA 几何后建立。实现必须原位替换 canonical
Direction Store decoder；历史实现由 Git、frozen config 和正式 artifacts 保存，不能
保留第二条可执行 Writer 路径。

## 1. 新证据改变了什么

Direction Store winner macro50 的每个 effective LoRA target 都近 rank-1：stable
rank=`1.000043`、首奇异值能量=`.999957`。但这不能再被解释成 task drift 的根因。
两套直接 Source-SFT step400 也主要沿低秩方向更新：

| geometry | old Source-SFT correct122 | corrected Source-SFT correct109 |
| --- | ---: | ---: |
| total effective norm | 34.4132 | 35.7362 |
| mean target stable rank | 1.5054 | 1.5169 |
| energy-weighted top singular share | .9229 | .9056 |
| q energy share | .9390 | .9249 |
| q mean stable rank | 1.1420 | 1.1571 |
| v mean stable rank | 1.5178 | 1.5564 |

真正稳定且有区分力的是 **policy target 之间怎样组织这些低秩方向**：

| geometry | Source-SFT | Direction Store / SFB |
| --- | ---: | ---: |
| q/v cross-layer BA cosine | 约 `0` | `.93--.97` |
| q/v layer-energy CV | `.71--1.05` | `.03--.14` |
| q/v top-4 layer energy | `46--59%` | `23--27%` |

这不是 SFT 初始化噪声。old 与 corrected SFT 的 q/v layer-energy profile Pearson
相关为`.9931/.9904`，layer norm rank 相关均为`.9835`；q top-4 layer 完全相同，
v 重合3/4。对应 target 的 effective-BA 方向余弦仍为`.8450/.8529`。因此一个
action-trained、closed-loop 有效的 adapter 可以在每个 target 内保持低秩和建设性
增益，同时稳定地让不同 policy layer/target 使用不同方向和能量。

这也修正了此前“下一版必须让16个rank坐标承担多维功能子空间”的表述。强制增加
effective rank、A/B正交或跨层去相关已由 Target-Spectral 负裁决：它把stable rank
提高到`3.3245`、q/v跨层余弦降到`.032/.066`，却令correct跌至`34/400`、norm从
`94.71`降到`25.87`。本版不把rank或正交性作为目标。

## 2. 最早尚未单独检验的接口

Target-Bound 已让38个真实policy targets先读Core，再以16个rank coordinates分别读
private Action/Effect/Change Program。SFB让target Core选择hidden bases；Direction
Store又让不同task拥有完整独立的factor参数存储。两者仍有同一个硬共享边界：

```text
Z[target, rank] = concat(Core, Read_A, Read_E, Read_D)  # 1024
               -> one of only eight factor-family MLPs
               -> public A/B row or column
```

同一factor family的18个q layers或18个v layers共用同一`W_in`和`W_out`。尤其在
factor final projection从zero打开的早期，来自不同policy layers的functional
gradients先在同一输出矩阵中相加；target/rank只改变输入activation，不能拥有自己的
输出参数。Direction Store按task拆参数但每个store内部仍跨全部layers共享factor head，
所以它解决了task address ownership，却没有解决policy-target ownership。这与观察到的
跨层同向、能量均匀几何一致。

本版检验的根因是：

> 条件化Writer必须先把每个真实policy target映射到其自己的policy tangent decoder；
> 相关target可以通过学习自然形成coherent方向，但不能在生成参数层面被硬绑定为同一
> 输出模板。

## 3. 架构：76个完整、独立的target-owned factor heads

严格保留：

- exact task language + exactly one action-hidden teacher video；
- mean-backed Semantic Core；
- 38-target、private A/E/D causal Program；
- target-first、rank-last reader与完整rank-16 public LoRA；
- template A / zero B和factor final zero-init；
- frozen source policy、24/8/8 split及全部信息墙。

删除 frozen language anchor、spherical-kmeans route和八个task Direction Stores。对每个
真实 public tensor `s`（38 targets × A/B，共76个）建立一套独立、bias-free factor
head：

```text
h_s[t,r] = GELU(W_in[s] Z[t,r])       W_in[s]: 1024 -> 256
f_s[t,r] = W_out[s] h_s[t,r]          W_out[s]: 256 -> tensor width
```

`W_out[s]`全部exact-zero，故step0逐tensor等于公开identity template。head选择只由
sealed PI05 LoRA tensor topology决定，不读取task ID、suite、video filename或任何
隐藏输入。不同task和video仍由同一个head对`Z`的连续函数生成不同LoRA；不存在
task-specific adapter bank、static LoRA dictionary或language-only bypass。

这不是Target-Spectral的另一种写法：

- 不约束不同heads的方向、余弦、奇异值或能量；
- 不要求16个rank coordinates不同，也不阻止near-rank1建设性叠加；
- 不分解shared carrier/innovation，不加scale、gate或residual；
- 若多个layers确实需要同向，独立heads可以从数据中学成同向；
- 若它们需要SFT式layer specialization，也不再被共享`W_out`阻止。

## 4. 参数量与通用性

76个heads的参数为：

```text
input projections    76 * 1024 * 256 = 19,922,944
output projections   sum(width_s)*256 = 20,594,688
factor total                             40,517,632
non-factor Writer                         7,340,288
Writer total                             47,857,920
```

owner已解除Writer参数上限。新增参数全部对应当前真实的policy-target decoder ownership，
不扩大Core/Program、不增加public rank，也不建立LIBERO task专家。该机制只依赖
generated LoRA到学习信号的可微路径：当前可用AS functional gradient训练，后续也可
原样用rollout reward/advantage训练。换base policy时需要按其真实LoRA topology重新
实例化heads，这是adapter生成器合理的policy架构依赖，不是LIBERO或SFT特化。

相邻但不选择的方案是只拆`W_out`、继续共享`W_in`。它更省参数，却仍允许共享hidden
transform在输出前把不同target压成同一模板；首个根因实验采用完整ownership，避免负
结果无法区分input与output sharing。也不把Source-SFT权重、层能量profile或方向作为
初始化/teacher，SFT只提供诊断证据。

## 5. 训练合同

首个formal保持Direction Store首段的训练样本和optimizer语义：

- fresh functional identity；
- task-query-keyed independent Beta/Gaussian logical B20；
- physical policy microbatch B2；
- full24 raw equal mean、每macro一次clip/AdamW/scheduler；
- fast cosine decay400、每25保存；
- 6 ranks时每rank 4 tasks，若live空闲卡不足则只允许由代码按actual world size形成完整
  24-task assignment，不能丢task或伪装相同合同；
- 不加入正交/rank/energy loss、SFT distillation、task-ID supervision、gradient
  projection、reward、few-shot或checkpoint融合。

BCI A40正式启动前先做CPU合同、fresh schema、最长105-frame真实video、B20连续宏步、
fresh0→1和exact-resume1→3。profile若OOM，先减`max_frames_per_encoder_call`；再减
physical policy microbatch到B1。logical B20、24 tasks、训练examples与optimizer step
不变。

## 6. 预注册判据

首小时fresh0→200后，严格配对评测50/100/150/200的correct400。single checkpoint
最低目标仍为严格超过150，并同时报告breadth、逐task、相邻gained/lost、union、
intersection和single envelope gap。

内部机制预测：

1. 76个heads的参数与optimizer state完全独立，一个tensor的synthetic loss不能给其他
   head产生gradient；
2. step1只打开各head的`W_out`，step2起semantic frontend、Core、Program、compiler和
   factor均finite/nonzero；
3. effective LoRA norm和每target near-low-rank增益不得发生Target-Spectral式坍缩；
4. q/v跨层余弦不要求接近0，但应显著脱离Direction Store的`.93/.97`硬同向状态，或
   layer-energy CV/top-4 share出现稳定target specialization；
5. same-task video的Program变化到factor/BA不能继续被压到`.019/.032`量级而毫无行为
   贡献；A/E/D、Core-only、Program-only与order反事实继续到达BA/action；
6. 若target-owned heads已形成稳定层专门化而closed-loop仍低且漂移，则policy-target
   sharing不是主要根因，下一步回到condition-to-policy credit/objective，不追加
   target gate、手工层权重、SFT prior或输出正交loss；
7. 若仍跨层同向且uniform，则应检查上游`Z[target,rank]`是否在进入独立heads前已近乎
   相同，而不是继续扩大heads。

## 7. Canonical实现边界

复用一个`CompleteLoRAWriter`、训练入口、checkpoint owner、evaluator和internal
analysis。原位替换：

- `program_compiler.py`：Direction Router/Store Head退役，新增单个完整factor head；
- `model.py`：76 tensor specs各自拥有head，删除task anchor decode；
- `video_program.py`：删除只服务Direction Store的额外frozen text-anchor forward；
- `architecture.py`、`as_config.py`、`checkpoint_schema.py`、`update_contract.py`：fresh
  architecture/config/state family；
- `internal_decode.py`和internal analysis：重放target-owned decode并报告target/layer
  specialization，不保留伪Direction Store字段；
- 旧Direction Store专用focused tests退役，保留并改写identity、ownership、gradient
  staging和internal parity不变量。

不新增第二套model、runner或evaluator。历史Direction Store config可以作为不可由当前
schema加载的provenance保留；需要复现时使用其frozen Git commit。
