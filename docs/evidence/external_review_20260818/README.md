# External Review Paired Evidence

本目录补齐外部专家G.1、G.4和G.9提出的remote-visible证据。`paired_evidence.json`由
`scripts/export_paired_review_evidence.py`从六个本地sealed `results.json`确定性导出；不包含本地主机路径、checkpoint
本体、数据集或secret。

## Pairing audit

六个panel均包含完全相同的8 tasks × 50 initialization rows。以current macro25为reference逐行核验：

- episode key mismatch：0；
- environment seed mismatch：0；
- policy seed root mismatch：0；
- policy noise common-prefix mismatch：0；
- correct K4 teacher-reference demo IDs mismatch：0。

成功episode会提前终止，因此实际记录的replan noise seed数量可少于失败episode；审计比较两panel实际执行范围的公共
前缀，而不是错误要求终止后的未执行seed列表等长。

## Recomputed panels

| panel | strict | breadth@1 | breadth@5 | breadth@10 | suite minimum | top-3 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current macro25 | 123 | 8 | 3 | 3 | 4 | .91057 |
| current macro50 | 84 | 5 | 3 | 2 | 1 | .97619 |
| current macro75 | 89 | 6 | 2 | 2 | 3 | .93258 |
| current macro100 | 87 | 4 | 2 | 2 | 3 | .96552 |
| LPCP143 | 143 | 7 | 4 | 4 | 5 | .84615 |
| GOMQ151 | 151 | 6 | 4 | 4 | 3 | .80132 |

当前macro25的breadth@1=8并不代表广泛强support：只有3 tasks达到5或10 successes，且top-3占91.1%。继续训练
后top-3 concentration升到93--98%，说明漂移伴随能力进一步集中，而不是shared breadth持续积累。

## Recomputed paired transitions

| transition | retained | gained | lost | net | Jaccard | exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current 25→50 | 71 | 13 | 52 | -39 | .52206 | 1.1688e-6 |
| current 50→75 | 59 | 30 | 25 | +5 | .51754 | .59005 |
| current 75→100 | 70 | 17 | 19 | -2 | .66038 | .86794 |
| LPCP143→current25 | 100 | 23 | 43 | -20 | .60241 | .018657 |
| GOMQ151→current25 | 100 | 23 | 51 | -28 | .57471 | .001516 |
| LPCP143→GOMQ151 | 126 | 25 | 17 | +8 | .75000 | .279956 |

最后一行的`.279956`是LPCP143与GOMQ151直接比较；专家报告中的`.019520`对应matched fixed-memory135→GOMQ151，
两者不是同一paired comparison。

## Provenance

- current四点：training commit `aecbce5b4301f98ecaafea650e099b6326c5c98d`，evaluation commit
  `f42edfc0d6504e62146218e8af9d2c2bbbe5959e`，world6，macro25/50/75/100；
- LPCP143：training commit `515f91e9645fabd2fff8faeb01f73f2519241225`，evaluation commit
  `07ec4d8c6d2255ff17817ed8f4561e4c15400d2e`，world6，macro25；
- GOMQ151：training/evaluation commit `8553b613de7791df50e0f3ef85678fcaca1cac0c`，world6，cycle2；
- 六个evaluation均声明clean dirty-path list，完整schema、LoRA contract、video schedule、information wall、training
  contract和checkpoint manifest摘要见JSON的`provenance`字段。

训练commit与evaluation commit不同不被隐藏；评测只读取sealed checkpoint，wrapper/evaluator来自各自登记的clean
evaluation commit。本文只确认可追溯性，不把跨commit差异假设为性能原因。

## Row schema

`paired_evidence.json.rows`逐行公开suite、task、initialization、env/policy seed reference、K4 teacher demo IDs、video
order/selection seeds、每个panel的success bit和实际noise-seed count。完整逐步noise值未复制进Git；首个seed、root、
common-prefix实算结果和sealed evaluation contract共同提供可复核RNG reference，避免为证据复制数十万无新增信息的整数。

## Pre-fix functional gradient audit

`gradient_audit_before_fix.json`来自clean commit `5242ee062884f3fb0c6f95310776c6ffa91dd5c5`，在同一canonical
task39、K3 teacher demos `[37,24,40]`、B20 query和policy RNG上比较fresh state与sealed macro25。它执行真实source-policy
functional loss和Writer backward，而不是只检查`requires_grad`：

- source policy在两点均为0个nonzero gradient tensor；
- fresh点只有q/v/action的B-family末层有非零gradient，A-family与所有upstream group均为0，验证B-first冷启动；
- macro25时Action/Text Meta-LoRA、language projection、Core、visual transition、Procedure、memory tokens、Reader、
  video-set、Core fusion、M2P和八个FactorHead family全部有nonzero finite gradient；
- macro25时`patch_grounding.query/key/output`、两个grounding norm参数和`interaction_projection`仍为
  `gradient_present_tensors=0`；不是小梯度或数值下溢，而是autograd图完全未连接；
- 所有trainable参数均被分组，`unclassified_parameter_names=[]`。

fresh只用来确认初始化信用顺序；不能用其upstream零梯度指控detach。真正区分证据是heads已经打开的macro25：除专家
指出的fresh projection outputs外，后端每组均有信用。

## Macro25 video-causality panel

`video_causality_evidence.json`从同一macro25 Writer的七个strict paired400面板导出。七个面板的8 tasks × 50
initializations、environment seed、policy seed root、实际执行noise common prefix、K4 teacher ordinal/reference
video IDs逐行一致，mismatch全为0。

| video condition | strict | breadth@1 | breadth@5 | breadth@10 | suite minimum | top-3 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| correct | 123 | 8 | 3 | 3 | 4 | .91057 |
| same-task-other | 125 | 7 | 4 | 3 | 5 | .89600 |
| cross-suite-wrong | 81 | 6 | 4 | 3 | 4 | .85185 |
| shuffled | 122 | 4 | 4 | 4 | 0 | .91803 |
| shuffled-keep-first | 131 | 7 | 5 | 3 | 5 | .89313 |
| reversed | 90 | 6 | 4 | 3 | 4 | .87778 |
| no-video | 48 | 3 | 2 | 1 | 0 | 1.00000 |

| correct vs control | both success | correct-only | control-only | correct minus control | Jaccard | exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same-task-other | 105 | 18 | 20 | -2 | .73427 | .87141 |
| cross-suite-wrong | 56 | 67 | 25 | +42 | .37838 | 1.3816e-5 |
| shuffled | 98 | 25 | 24 | +1 | .66667 | 1.0 |
| shuffled-keep-first | 106 | 17 | 25 | -8 | .71622 | .27996 |
| reversed | 68 | 55 | 22 | +33 | .46897 | 2.1680e-4 |
| no-video | 39 | 84 | 9 | +75 | .29545 | 2.1689e-16 |

这组证据支持三个限定后的判断：

1. correct相对no-video和cross-suite-wrong有显著收益，所以当前Writer不是language-only，也不是任意nonzero-video
   carrier即可完成任务；
2. same-task-other与correct总分相当，但仍有38-row churn和`.734` success-set Jaccard，说明具有一定跨demo
   鲁棒性，但未达到专家建议的90% success-retention；
3. correct相对shuffled没有优势，shuffled-keep-first甚至高8分（不显著）。reversed与no-video下降不能掩盖
   这个反例：当前闭环收益需要同task视频内容和某些粗方向线索，但没有证明正确中间阶段顺序是必要的有用
   policy evidence。

correct面板由clean evaluation commit `f42edfc0d6504e62146218e8af9d2c2bbbe5959e`生成，六个control由
clean commit `2606c4973a9f311430e69a56e304b58866057b58`生成。两提交间当前Writer/evaluator有效forward路径未
改变；差异为证据导出/梯度审计、paired metric扩展和退役模块清理。原始提交仍在JSON provenance中分别显示，
没有把跨commit隐藏成同一代码。

## Writer drift and cross-decode evidence

`writer_drift_evidence.json`由`scripts/export_writer_drift_evidence.py`从保留的four-checkpoint drift diagnosis和
Program × FactorHead cross-decode artifact导出，去除本地checkpoint路径，保留全部数值证据。它补齐专家G5：

- macro25→50的native effective-BA relative L2为`1.30573`；
- 旧Program+新heads的heads-only relative L2为`1.31999`，新Program+旧heads的Program-only为`.58234`；
- 同一区间Core/Procedure/parameter-set/core-fused/compiled的relative L2分别为`.60468/.70450/.52819/1.83432/.77010`；
- JSON同时保留macro25/50/75/100的per-module gradient conflict、next-update delta、fixed-B20 task deltas、strict
  transition以及后两个区间的完整cross-decode，避免只引用一个有利数字。

## No-Text / restored-credit gradient attribution

`gradient_credit_evidence.json`把三次clean真实functional backward放在同一remote-safe文件中：

- A（pre-fix、含Text Meta-LoRA）和B（移除Text/VL Meta-LoRA但保留detach）在macro25都显示
  `patch_grounding.query/key/output/norm`与`interaction_projection`完全没有gradient；
- B其余Action Meta-LoRA、Core、Procedure、memory、Reader、K-set、M2P和八个heads均有非零有限gradient，说明移除
  Text没有切断原有主链；
- C相对B只恢复三项返回tensor的autograd连接。在macro1 heads已打开后，grounding四组与interaction全部首次出现非零
  有限gradient，所有其它预期组也保持有信用；
- 三组审计的source-policy nonzero-gradient tensors均为0，unclassified trainable parameters均为空；B/C均不存在
  Text或VL Meta-LoRA，保留native frozen language和Action Meta-LoRA。

C使用macro1是为了记录新路径的最早可观察信用，A/B使用macro25是为了证明旧detach在后端已经打开后仍切断该路径；
因此这里裁决的是graph connectivity，不比较跨schedule的gradient magnitude。稳定regression同时覆盖返回tensor局部
gradient和frozen native memory replay不向source参数回传。

## F2 fixed-occupancy counterfactual

`occupancy_evidence.json`公开原始paired400定义的52 lost、13 gained和71 retained rows在macro25及macro50两套真实
rollout occupancy上的逐replan action disagreement。两次capture均为clean formal run，固定task/state、teacher K4、
policy seed root与noise common prefix；正常并行拓扑低位差异使macro25/50分别有14/17个capture success bit相对原面板
翻转，原始分类保持不变，并另报只保留两边replay-consistent rows的敏感性结果。

- offline B20 functional loss由`.1136444`降到`.1054047`；
- lost rows的`macro50 occupancy - macro25 occupancy`动作RMS均值为`-.006548`，52行中20正/32负，Wilcoxon
  `p=.05466`；replay-consistent 38行同样为`-.007023`；
- gained rows反而为`+.011293`，11正/2负，`p=.00610`；retained为`+.002409`，`p=.1993`；
- lost与retained在共同初态的checkpoint action RMS均值为`.19428`和`.14297`，但Mann--Whitney `p=.1630`。
- 136/136行在初始状态的第一次replan即出现checkpoint行为分歧；初态action RMS范围为`.04284`到`.48279`。

这不支持“macro50只因进入其自身occupancy后两checkpoint分歧变大而丢失support”的简单版本，方向在lost/gained间
反而与该预测相反。它也不能证明macro50动作更正确：validation task expert不可用，且没有读取held teacher action，
所以该审计是matched checkpoint disagreement而非expert-action error。基于这个结果，不授权F2.05用旧occupancy panel
替换B20训练。

## F3 frozen-FactorHead continuation

`f3_headfreeze_evidence.json`比较同一A macro25起点、正常续训macro50和唯一冻结八个FactorHeads的诊断续训macro50：

- normal macro50为84/400；frozen-head macro50为117/400，相对normal是49 gained / 16 lost，exact McNemar
  `p=5.08e-5`；
- macro25 123→frozen-head 117为90 retained / 27 gained / 33 lost，success retention `73.17%`；
- frozen-head满足专家建议的score≥110，但未满足lost≤20、retention≥90%和breadth不降；
- checkpoint张量核验显示16个FactorHead tensor逐元素不变，481个upstream tensor中404个变化。

因此head co-drift是25→50崩落的重要放大器，但不是充分根因；固定heads仍有33个旧support丢失，责任必须继续检查
upstream/objective与fixed-head reachability。
