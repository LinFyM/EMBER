# EMBER progress

更新时间：2026-09-06 14:59 CST。

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

## 当前验证评测

- 四套primary验证bank全部sealed，物化launcher exit0；首次source准备113.99秒，驻留复用.14–.21秒，每套8task生成约22秒。
- 四个checkpoint的固定8task×10states screen80已并行启动，均使用同一041aff55、同一primary视频及state/env/policy RNG合同。
  32在gpu01p4，64在gpu01p5，96在gpu02p1/2，128在gpu01p3/6；每卡3个persistent workers，单job单节点，总量6物理GPU。
  NUMA分别1/1/0/1。gpu02两卡启动前各162MiB、既有process148MiB且0%util；gpu01四卡空闲，不干扰其它任务。
- launcher PID分别3403145/3405746/3643550/3406242。精确脚本与参数见`.codex/tmp/prw_complete_meta73/screen{32,64,96,128}.sh`、各log首行
  及analysis/launch的live preflight；全部已生成正确80-state合同，workers正在初始化，尚无完整screen结果。
- 四组screen只分配下一步投入，不选最终checkpoint，不线性外推400。若有广泛、有保留的能力且接近强参照，先做一次strict400；
  有希望再补预登记相邻及另一正视频，不能机械扩成多套弱400。Wrong/no-video/language/端点/shuffle/reverse及Test仍未使用。
- 配对历史参照：相同80-state前缀source9、SFT24、A2m64为17；full400为47/109/79。SFT合同与当前state8/action7/replan5、
  normalization和RNG已核对，109为原始轨迹复用；旧Writer143因teacher schedule不同只作count参照，不伪造配对比较。

## 证据与资源

- Run：`runs/outputs/pi05_ecp_prw_complete_meta73_s128_041aff55_gpu01p03456_20260906`（约239MiB）。
- 当前analysis：`runs/analysis/pi05_ecp_prw_complete_meta73_20260906`；含launch、四套bank、四组screen、完整schedule与功能报告，
  `summarize.py`按完整raw rows输出per-task/suite/RGL/churn/Jaccard及预登记strict400资格；synthetic边界检查已通过。
- 本阶段live /data1 quota为722600384KiB /1073741824KiB soft，shared84TiB available，整个物化/评测新增峰值<8GiB。
- 等待期间已回收旧A2 random的97.5857GiB可重算cache并移除clean inactive的7f6b1611/b2bb03ce工作目录；其Git、checkpoint及formal证据保留。
  当前只有main和041aff55 active frozen tree。旧A2 random仅保存中断证据，不恢复。
- 专家原文：`docs/expert_review_20260905_full_history_joint_process_policy_writer.md`；下一判断须区分训练学习、新视频、未见task与
  真实成功遗忘/occupancy问题。旧路线通过`docs/research_history.md`、`findings.md`和Git查证，不从旧待办恢复。
