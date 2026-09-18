# EMBER task plan

## 当前goal（2026-09-19，经最终LoRA的同视频教学）

完成[专家最终修订](docs/review_materials/20260919/expert_proposal.md)，实现、真实验证、训练评测、分析后推送并汇报。
Active design：[方法与有界实验合同](docs/video_teaching_writer_design.md)。主实验单相机agentview，后续双相机仅据证据裁决。

1. **已完成：合同与实现。** 保留A完整参数骨架，加入重复完整H/相邻视觉读取；同视频教学与跨episode主FM共同训练全部Writer/Meta。
2. **已完成：CPU与真实profile。** 数据对齐、标签信息墙、两组归一与重放、1–6卡逻辑不变、最长视频及完整resume；profile权重不入formal。
3. **进行中：fresh主实验。** 1500硬上限，900/1200/1500各correct400/train96，真实资源由profile登记。
4. **待完成：有界后续与分析。** 按预注册相邻能力决定other/冻结controls及唯一监督配对消融；双相机默认不补，不开展无依据扫描。
5. **待完成：远程交付。** 更新结果、findings与研究历史，推送代码/合同/精简证据，核对main后完成goal。

上轮frame-set已结束1200，结果79/109/142/118，对A99/88/140/135；[报告](docs/review_materials/20260918/frameset_report.md)。
不重训旧A/v5.2，不恢复旧路线，不用controls选点，不访问Test或RL。负结果也须完整分析交付。
