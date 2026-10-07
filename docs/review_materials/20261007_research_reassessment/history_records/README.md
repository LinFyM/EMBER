# 历史审阅原件的远程补充

2026-10-07。审查远程Git树时发现，9月17日历史审计有一组链接只在服务器运行目录可读。
本目录补充40份已有开发记录（约5.0 MiB），复用6份已在旧材料包中的导出，并在原审计中修正46处入口。
没有运行模型、环境、训练或新统计实验，也没有改变历史结果或当前科学授权。

- [原审计](../../../analyses/v52_evidence_audit_20260917.md)：学习曝光、机制、结果与推断边界；跳过其封存§6.1。
- [机器可读索引](index.json)：每份原件的仓库相对来源、当前导出、复用/新增状态和转换范围。
- [完整历史路线](../HISTORY_MAP.md)：先理解主线，再选择需核对的记录。

新增JSON保留所导出科学字段的数值与顺序，压紧排版；本地路径、私网地址、主机身份、命令、设备等运行现场字段移除或规范化。
shuffled/reversed结果分支及对应封存审计排除，不将Test结果纳入本次材料；记录中的未使用/信息墙字段并非Test结果。
旧包复用文件仍可能含历史封存字段，专家应按本次提示词跳过。原件留在原位置，不宣称这些副本字节相同。

这里导出的是既有合同、汇总、配对/功能读回及其保留行，不是整个历史运行树。模型、完整轨迹、RGB和大张量没有复制；
不能由此声称每条历史结论已经远程独立重算。某些原件内部的来源路径只作provenance，不是可点击的远程文件。

| 原件来源 | 远程记录 | 状态 |
| --- | --- | --- |
| `runs/analysis/source_alignment_20260915/archival_A_B_comparison_boundary.json` | [读取](analysis/source_alignment_20260915/archival_A_B_comparison_boundary.json) | 新增规范化导出 |
| `runs/analysis/source_alignment_20260915/A/node900_reference_comparisons.json` | [读取](analysis/source_alignment_20260915/A/node900_reference_comparisons.json) | 新增规范化导出 |
| `runs/outputs/pi05_as_writer_v52_v6_recipe_matched_exposure_seed7_20260801/analysis.json` | [读取](training/pi05_as_writer_v52_v6_recipe_matched_exposure_seed7_20260801/analysis.json) | 新增规范化导出 |
| `runs/outputs/pi05_as_writer_target_owned_factor_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_34be4a0_20260804T051244Z/run_contract.json` | [读取](training/pi05_as_writer_target_owned_factor_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_34be4a0_20260804T051244Z/run_contract.json) | 新增规范化导出 |
| `runs/outputs/pi05_dynamic_k_backbone_memory_rank8_budget64_formal_fresh0to50_r6_b20_micro8_5319022_20260813/run_contract.json` | [读取](training/pi05_dynamic_k_backbone_memory_rank8_budget64_formal_fresh0to50_r6_b20_micro8_5319022_20260813/run_contract.json) | 新增规范化导出 |
| `runs/outputs/pi05_v6_layerwise_probe_conditioned_procedure_formal_fresh0to25_r6_b20_515f91e_gpu01_20260814/run_contract.json` | [读取](../../20260907/records/historical_lpcp/pi05_v6_layerwise_probe_conditioned_procedure_formal_fresh0to25_r6_b20_515f91e_gpu01_20260814/run_contract.json) | 复用已有导出 |
| `runs/outputs/pi05_v6_lpcp_cfmg_gomq_formal_fresh_cycle0to2_r6_k4_views4_b8_8553b61_gpu02p123467_20260817/run_contract.json` | [读取](../../20260907/records/historical_gomq/pi05_v6_lpcp_cfmg_gomq_formal_fresh_cycle0to2_r6_k4_views4_b8_8553b61_gpu02p123467_20260817/run_contract.json) | 复用已有导出 |
| `runs/outputs/pi05_ecp_prw_complete_shared4_s64_b2bb03ce_gpu02p235_20260906/run_contract.json` | [读取](training/pi05_ecp_prw_complete_shared4_s64_b2bb03ce_gpu02p235_20260906/run_contract.json) | 新增规范化导出 |
| `runs/outputs/pi05_ecp_prw_complete_target18_width256_s128_14bc7605_gpu02p012356_20260906/result.json` | [读取](training/pi05_ecp_prw_complete_target18_width256_s128_14bc7605_gpu02p012356_20260906/result.json) | 新增规范化导出 |
| `runs/outputs/pi05_ecp_shared_compiler_g3_f3_stable_anchor_fold0_m5_20acc33_gpu01p012346_r6_20260827/run_contract.json` | [读取](training/pi05_ecp_shared_compiler_g3_f3_stable_anchor_fold0_m5_20acc33_gpu01p012346_r6_20260827/run_contract.json) | 新增规范化导出 |
| `runs/analysis/pi05_ecp_primal_capacity_p1_v1_c9e8198_gpu01p012345_20260829/report.json` | [读取](analysis/pi05_ecp_primal_capacity_p1_v1_c9e8198_gpu01p012345_20260829/report.json) | 新增规范化导出 |
| `runs/outputs/pi05_ecp_natural_program_g2_behavior_fold0_m10_5cbe76e_gpu01p012345_r6_20260829/run_contract.json` | [读取](training/pi05_ecp_natural_program_g2_behavior_fold0_m10_5cbe76e_gpu01p012345_r6_20260829/run_contract.json) | 新增规范化导出 |
| `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v3_anchor_s110_fd20251_gpu01p012_r3_20260831/run_contract.json` | [读取](training/pi05_ecp_program_bank_candidate_interaction_v3_anchor_s110_fd20251_gpu01p012_r3_20260831/run_contract.json) | 新增规范化导出 |
| `runs/outputs/pi05_ecp_event_bank_set_s2_direct_functional_s110_25477c9_gpu01p013456_r6_20260901/run_contract.json` | [读取](training/pi05_ecp_event_bank_set_s2_direct_functional_s110_25477c9_gpu01p013456_r6_20260901/run_contract.json) | 新增规范化导出 |
| `runs/outputs/pi05_ecp_event_bank_set_s2_functional_polish_gate_s70s110_bb98b81_gpu01p0256_w4_20260901/aggregate.json` | [读取](training/pi05_ecp_event_bank_set_s2_functional_polish_gate_s70s110_bb98b81_gpu01p0256_w4_20260901/aggregate.json) | 新增规范化导出 |
| `runs/outputs/pi05_ecp_event_bank_set_relational_quotient_s0_gate_s110_ad64757_gpu01p34_20260901/aggregate.json` | [读取](training/pi05_ecp_event_bank_set_relational_quotient_s0_gate_s110_ad64757_gpu01p34_20260901/aggregate.json) | 新增规范化导出 |
| `runs/analysis/layered_relation_writer_20260907/train24_shared/decision_after384.json` | [读取](../../20260907/records/current_train24/layered_relation_writer_20260907/train24_shared/decision_after384.json) | 复用已有导出 |
| `runs/outputs/horizon_k1_supervised_v1_seed7_20260908/run_contract.json` | [读取](training/horizon_k1_supervised_v1_seed7_20260908/run_contract.json) | 新增规范化导出 |
| `runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/sharing/completed_matrix.json` | [读取](../../20260911/analysis/sharing_matrix.json) | 复用已有导出 |
| `runs/analysis/video_consumption_20260911/first_round_summary.json` | [读取](../../video_specificity_20260911/analysis/consumption/first_round_summary.json) | 复用已有导出 |
| `runs/analysis/video_change_reference_20260911/step200/paired_summary.json` | [读取](../../video_specificity_20260911/analysis/nochange/step200_summary.json) | 复用已有导出 |
| `runs/analysis/video_functional_20260911/paired_summary.json` | [读取](analysis/video_functional_20260911/paired_summary.json) | 新增规范化导出 |
| `runs/analysis/video_functional_20260911/direct_fm_vl/bounded_200_decision.json` | [读取](analysis/video_functional_20260911/direct_fm_vl/bounded_200_decision.json) | 新增规范化导出 |
| `runs/analysis/local_action_grounded_20260912/bounded_200_decision.json` | [读取](analysis/local_action_grounded_20260912/bounded_200_decision.json) | 新增规范化导出 |
| `runs/analysis/frozen_positive_replication_20260913/paired_replication_summary.json` | [读取](analysis/frozen_positive_replication_20260913/paired_replication_summary.json) | 新增规范化导出 |
| `runs/analysis/execution_aligned_video_20260912/paired_summary.json` | [读取](analysis/execution_aligned_video_20260912/paired_summary.json) | 新增规范化导出 |
| `runs/analysis/native_dual_video_20260913/camera_comparison.json` | [读取](analysis/native_dual_video_20260913/camera_comparison.json) | 新增规范化导出 |
| `runs/analysis/visible_object_grounding_20260913/supervision_comparison.json` | [读取](analysis/visible_object_grounding_20260913/supervision_comparison.json) | 新增规范化导出 |
| `runs/analysis/semantic_path_writer_20260914/paired_readout.json` | [读取](analysis/semantic_path_writer_20260914/paired_readout.json) | 新增规范化导出 |
| `runs/analysis/horizon_relation_writer_20260908/k1_first_query_only/functional_assignment/summary.json` | [读取](analysis/horizon_relation_writer_20260908/k1_first_query_only/functional_assignment/summary.json) | 新增规范化导出 |
| `runs/analysis/video_functional_20260911/native_reader_diagnostic/run_contract.json` | [读取](analysis/video_functional_20260911/native_reader_diagnostic/run_contract.json) | 新增规范化导出 |
| `runs/analysis/video_functional_20260911/native_reader_analysis.json` | [读取](analysis/video_functional_20260911/native_reader_analysis.json) | 新增规范化导出 |
| `runs/analysis/native_corrective_transfer_20260913/paired_summary.json` | [读取](analysis/native_corrective_transfer_20260913/paired_summary.json) | 新增规范化导出 |
| `runs/analysis/native_correction_writer_20260913/oracle_rollout/decision.json` | [读取](analysis/native_correction_writer_20260913/oracle_rollout/decision.json) | 新增规范化导出 |
| `runs/analysis/native_correction_writer_20260913/acquisition_audit/summary.json` | [读取](analysis/native_correction_writer_20260913/acquisition_audit/summary.json) | 新增规范化导出 |
| `runs/analysis/local_correction_field_writer_20260914/paired_readout.json` | [读取](analysis/local_correction_field_writer_20260914/paired_readout.json) | 新增规范化导出 |
| `runs/analysis/process_pullback_writer_20260914/training_audit.json` | [读取](analysis/process_pullback_writer_20260914/training_audit.json) | 新增规范化导出 |
| `runs/analysis/process_pullback_writer_20260914/causal_diagnostics/registration.json` | [读取](analysis/process_pullback_writer_20260914/causal_diagnostics/registration.json) | 新增规范化导出 |
| `runs/analysis/process_pullback_learned_outlet_20260915/paired_readout.json` | [读取](analysis/process_pullback_learned_outlet_20260915/paired_readout.json) | 新增规范化导出 |
| `runs/analysis/source_alignment_20260915/A/training/run_contract.json` | [读取](analysis/source_alignment_20260915/A/training/run_contract.json) | 新增规范化导出 |
| `runs/analysis/source_alignment_20260915/A/continuation_readout.json` | [读取](analysis/source_alignment_20260915/A/continuation_readout.json) | 新增规范化导出 |
| `runs/analysis/source_alignment_20260915/A/node2700_primary_readout.json` | [读取](analysis/source_alignment_20260915/A/node2700_primary_readout.json) | 新增规范化导出 |
| `runs/analysis/source_alignment_20260915/SFT/paired_readout.json` | [读取](analysis/source_alignment_20260915/SFT/paired_readout.json) | 新增规范化导出 |
| `runs/analysis/v52_return_20260915/baseline/paired_readout.json` | [读取](analysis/v52_return_20260915/baseline/paired_readout.json) | 新增规范化导出 |
| `runs/analysis/v52_return_20260915/cooccurrence/paired_readout.json` | [读取](analysis/v52_return_20260915/cooccurrence/paired_readout.json) | 新增规范化导出 |
| `runs/analysis/source_alignment_20260915/archival_B_horizon_weight_bound.json` | [读取](analysis/source_alignment_20260915/archival_B_horizon_weight_bound.json) | 新增规范化导出 |
| `runs/outputs/pi05_as_writer_v52_v6_recipe_video_causality_audit_seed7_20260802/analysis.json` | 本次排除，不新增导出 | 封存控制分析 |
