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
