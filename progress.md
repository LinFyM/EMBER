# EMBER progress

更新时间：2026-09-06 16:06 CST。

## 当前状态与授权

- 最终科学goal持续active；权限内持续自主推进。Active design为`docs/joint_process_policy_writer_design.md`。
  最终资格仍为validation8 strict single-checkpoint >145/400及预登记相邻/跨视频稳定、breadth、四suite与Goal/Long贡献；
  同图random候选和冻结后视频因果controls仍待完成，方法冻结后才做32/8 fresh及Test。没有selected checkpoint。
- 完整LoRA四任务短学习已sealed：32 fit/held50/54、64为64/62（各150），超过旧P/Q及A2；主要收益在Goal，Spatial有损失，Long未稳定。
  正证据支持进入本轮主训练，不等于最终能力或因果资格。原始比较见`docs/research_history.md`§173及四任务analysis。

## 本轮训练与诊断已完成

- Frozen clean pushed detached `041aff5519fa25cf9190bc9cd74da65d3c72af79`，工作目录
  `/data1/user/ymdai/projects/EMBER-worktrees/prw-complete-meta73-20260906`；完整38-target rank16、4,750,208 Writer参数、无carrier。
- 55meta+18target等权1/73，每步各8fresh跨episode action-query；K1/两fit视频、原normalizer，fresh component/AdamW，128updates。
  32/64/96/128 checkpoint及全部恢复状态保存，train launcher exit0；实际9344task executions/74752action rows，每fit视频64 exposures。
- 完整采样审计pass：与短四任务256项、旧A2 meta73 4672项重叠query/video/policy RNG匹配，原normalizers一致；
  正常micro8低位差异不改变任务权重。除以normalizer后的每task梯度系数.069735–.279459，明细已保存。
- 五卡gpu01物理0/3/4/5/6、world5、NUMA0/1/1/1/1，exact resume锁此拓扑。训练2929.82秒、Panel-B397.98秒，
  runtime总3484.74秒（另有启动/环境加载）；peak reserved34.67GiB，临时native cache已自然回收。所有held/Panel-B backward为0。
- 13task功能面板在32/64/96/128分别8/8/9/9项全部fit+held视频高于历史carrier。128时6个gradient meta与诊断meta6均为正；
  target77/83新视频benefit为+.002669/+.004505，72约-.000077，93/94为-.002393/-.004300，诊断target79为-.001222。
  meta与target学习差异明确，不能据此给出闭环分数或确定根因。carrier仅作历史loss参照，不参与部署。
- 同query的33–64步原四task loss均高于短四任务（task1/72/83/93差+.000328/+.008831/+.011236/+.008192）；
  expanded task mean与128-step decay耦合，不把效果拆作单一因果结论。完整诊断见当前analysis的functional_comparison.json。

## 验证结果与下一对照

- 四套primary bank及四组screen80全部完成，所有launcher/worker exit0。32/64/96/128为15/19/19/19；
  Spatial/Object/Goal/Long为0/7/8/0、1/10/8/0、0/9/9/1、1/9/9/0。后三点18/19成功集中Object1/Goal6，能力覆盖仍窄。
- 相邻R/G/L为15/4/0、17/2/2、17/2/2；后两pair Jaccard.810、churn4。128对A2为16/3/1，对SFT为13/6/11。
  同prefix SFT24、A2m64 17。未形成支持扩大评测投入的广泛能力，因此不启动strict400、不续同一混合run；没有final selector或negative/Test使用。
  这只是预登记的screen投入决定，不冒充400-row资格裁决。完整报告与decision.json已保存，历史§174/findings§173登记。
- 下一active arm已在同一design登记：`configs/pi05_ecp_prw_complete_target18_v1.json`，保留相同完整Writer和相同18target，
  移除55meta functional objectives。每步18targets等权1/18、128updates/18432action rows；per-target query/video/normalizer/LR完全匹配。
  原13task诊断全部保留，7meta及target79在此run零梯度，均为复用诊断、不称fresh selector。模型、数据接口、source及训练代码不变。
- 真实CPU config/eval loader、空meta组和完整128步采样检查通过；2304task executions、每target128exposures，primary/other视频ordinal一致。
  复用相同graph下最长87帧/micro8的实测34.67GiB峰值和全部18targets实际timing；六卡实测每步约4.27秒；完整启动/capture/Panel-B成本另计。
  已于15:28左右从clean pushed detached351feb48启动，gpu01物理0/2/3/4/5/6、world6、NUMA0/0/1/1/1/1；
  p2 peer任务在preflight前自然结束，实际六rank task-work估计均约4.01秒，source/code不变。launcher PID3518744，128updates及四点Panel-B已全部结束，launcher exit0，实际34.61GiB peak reserved。
  Run为`runs/outputs/pi05_ecp_prw_complete_target18_s128_351feb48_gpu01p023456_20260906`，analysis为
  `runs/analysis/pi05_ecp_prw_complete_target18_20260906`；精确命令`.codex/tmp/prw_complete_target18/train.sh`，contract/preflight/audit均保存在analysis/launch。
  /data1 quota722611784KiB/1073741824KiB，outputs529GiB、analysis17GiB，shared84TiB可用；缓存含诊断12.24GiB、新增峰值<16GiB。
  全2304项采样审计pass，所有目标task的query/video/RNG与上一轮匹配，两fit视频各64exposures，原normalizer一致；actual topology锁world6 exact resume。
  Train552.64秒、Panel-B390.73秒、runtime996.40秒（另计启动加载），临时cache已自动清理；全部held/Panel-B backward0。
  五个gradient target在32/64/96/128的全fit+held正收益为2/5、5/5、5/5、5/5；128 held benefit为72+.001732、77+.004774、
  83+.007476、93+.004031、94+.022396，均高于meta73；target79+.000962，七个meta诊断仍负。内部loss不代替行为判断。
  四套primary bank生成完毕/exit0；32/64/96/128 screen80分别在gpu02p0、gpu02p1、gpu02p4/6、gpu01p0运行。
  在读取这些screen结果前登记两run固定terminal128的train-side fit/held strict150（原Spatial2/Goal20/Long38）；
  共600 rows，用于分开目标任务行为恢复与未见任务迁移，不选checkpoint、不新增梯度。四个真实eval config loader均通过。
  目标是区分objective取舍，不扫描比例或加入剩余6target。只读train24支持审计保存在target_training_support.json，明确SFT使用全部24targets的支持差异。
- 原始参照仍为配对历史source47/SFT109/A2m64 79；旧Writer143因teacher schedule不同只作count参照。未来强候选仍需同图random、
  预登记strict400相邻/跨视频资格、冻结后视频因果controls和最终32/8合同。

## 证据与资源

- Run：`runs/outputs/pi05_ecp_prw_complete_meta73_s128_041aff55_gpu01p03456_20260906`（约239MiB）。
- 当前analysis：`runs/analysis/pi05_ecp_prw_complete_meta73_20260906`；含launch、四套bank、四组screen、完整schedule与功能报告，
  `summarize.py`按完整raw rows输出per-task/suite/RGL/churn/Jaccard及预登记strict400资格；synthetic边界检查已通过。
- 本阶段live /data1 quota为722600384KiB /1073741824KiB soft，shared84TiB available，整个物化/评测新增峰值<8GiB。
- 等待期间已回收旧A2 random的97.5857GiB可重算cache并移除clean inactive的7f6b1611/b2bb03ce工作目录；其Git、checkpoint及formal证据保留。
  当前保留main、041aff55 inactive frozen tree和351feb48 active frozen tree。旧A2 random仅保存中断证据，不恢复。
- 专家原文：`docs/expert_review_20260905_full_history_joint_process_policy_writer.md`；下一判断须区分训练学习、新视频、未见task与
  真实成功遗忘/occupancy问题。旧路线通过`docs/research_history.md`、`findings.md`和Git查证，不从旧待办恢复。
