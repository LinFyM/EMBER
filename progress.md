# EMBER progress

## 当前状态（2026-09-20，论文实验goal启动）

Owner已授权执行冻结1500的论文实验，并明确Test至少领先MT-BC40/400、FT独立演示选点与负结果暂停要求。
Active design：[paper_experiments_design](docs/paper_experiments_design.md)。24任务MT-BC Test入口已实现并推送45a39ba1；13项Source-SFT与45项Writer controls/Test测试通过。
Test能力三组已提交：Source/MT-BC各gpu01两卡，EMBER在gpu02两卡先物化后评测，合计六卡；各组400完整行。
Study为runs/analysis/paper_experiments_20260920；launch_contract.json记录固定资产、阈值和资源，run_panel.sh为实际命令。
Luna completion_watch只监控三个exit标志，完成/异常通知，不读取中间成绩。完整三组后运行study/readout.py核对并裁决。
正式frozen worktree为.codex/tmp/paper-runtime（45a39ba1），运行结束清理；主树不参与当前计算。
现有main干净起点cc320bf4；上午全面清理已完成，保留原件并按本轮用途维护资产，不重复无关全树扫描。
训练/评测按完整阶段完成通知再读结果，不反复轮询；不达预期停止后续任务询问Owner。

## 上轮已封存结果（2026-09-20）

Owner授权的专家最终修订、主候选、唯一匹配消融、后续趋势、消融视频特异性和条件性双相机全部完成。
该轮没有剩余作业；[合同](docs/video_teaching_writer_design.md)已封存。
先读[专家讨论总报告](docs/review_materials/20260919/final_report.md)与[可复制提示词](docs/review_materials/20260919/expert_discussion_prompt.md)。

| 更新 | 同视频 correct /400 | 消融 correct /400 | 双相机 correct /400 |
| ---: | ---: | ---: | ---: |
| 900 | 149 | 125 | 117 |
| 1200 | 174 | 136 | 108 |
| 1500 | 165 | 147 | 108 |
| 1800 | 160 | 136 | 未运行 |
| 2100 | 158 | 159 | 未运行 |

原1500相邻qualification选定主checkpoint后保持冻结；1200的174不作selected结果，追加窗口不重选。
主组1500→2100 correct165→158、other165→156；消融147→159、other155→161。
主组原窗口的正确能力增量未保持到末点，不证明消融更好或两者等价；[续训报告](docs/review_materials/20260919/continuation_report.md)。
固定1500的主组correct/other/wrong/shuffle/reverse/source为165/165/124/113/113/50，消融147/155/122/104/86/50。
两组都有顺序敏感性，但六个匹配DID区间均跨零，尚未证明教学项增强视频特异性；[视频检查](docs/review_materials/20260919/ablation_controls_report.md)。
双相机fixed1500 other105，correct/other相对单相机均下降；train52/64/67及FM下降未转为验证能力；[相机对照](docs/review_materials/20260919/dual_camera_report.md)。
2100和双相机没有额外controls，不能继承旧因果结论。无Test、RL、融合、挑视频或旧A/v5.2重训。

全部40个正式面板、12,048条新闭环、5700实际更新及638,400主＋辅助queries核对通过，共同source与续训父历史只计一次。
保留57个唯一arm/step完整checkpoint、run contract、manifest、raw rows和worker日志；[完成清单](docs/review_materials/20260919/completion.json)。
双相机按真实四卡profile采用world4/frame8/microbatch16，训练与物化／闭环流水并行，末轮三面板在两节点六卡并行。
全部训练/driver/正式workers正常退出；08:50评测收齐，08:51两节点核验无本轮GPU或训练／物化／评测作业。
source始终冻结，完整resume、6000/8400事件、全池teacher映射、逐行state/RNG和真实RGB变换核对通过。
实现阶段CPU检查主教学图累计246项、相机相关222项通过（覆盖有重叠）；后续分析只读取已有证据，不新增模型forward。

正式主/消融/续训/双相机的冻结commit为39c3919c/bd497edc/d1474ce0/35124aa9；临时runtime树已移除，可按commit重建。
本地原件统一在`runs/analysis/video_teaching_20260919/`，远程保留方法、配对CSV/JSON、曲线、输入示例和报告。
findings§121–122与[研究历史](docs/research_history.md)保存完整解释及边界；后继选择供Owner与专家讨论，需新的明确授权。

## 历史证据入口

[A的learned frame-set报告](docs/review_materials/20260918/frameset_report.md)保留上轮匹配诊断；
[46组证据审计](docs/v52_evidence_audit_20260917.md)、[findings](findings.md)§117–122及
[研究历史](docs/research_history.md)索引统一Writer、A900机制、Core/Procedure交叉与更早实验。
历史结果、旧源码与配置均按各自封存口径解释，不恢复已结束路线。
