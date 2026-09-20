# EMBER task plan

## 本轮goal已完成（2026-09-20，经最终LoRA的同视频教学）

已完成[专家最终修订](docs/review_materials/20260919/expert_proposal.md)及Owner追加的续训、消融视频检查与双相机对照。
当前无active design或自动后继实验；[方法与有界合同](docs/video_teaching_writer_design.md)已封存。

1. **已完成：合同与实现。** 保留A完整参数生成骨架，加入完整H/相邻视觉重复读取，两项功能信用共同训练Writer/三Meta，source冻结。
2. **已完成：实际验证与资源提速。** CPU图/信息墙检查、最长视频、完整恢复通过；固定逻辑四task，按合法空闲卡并行训练、物化和闭环。
3. **已完成：主组与匹配消融。** 各fresh1500并按追加合同完整续至2100；原主selected1500保持冻结，所有相邻及换视频面板收齐。
4. **已完成：视频检查和双相机。** 两组固定1500 controls及匹配DID完成；唯一双相机fresh1500及7个面板完成，不追加相机扫描、controls或续训。
5. **已完成：分析与交付。** 40面板/12,048条新闭环和三臂5700更新核对通过；精简原件、总报告、专家提示词、findings与历史已归档，集成main并推送。

[总报告](docs/review_materials/20260919/final_report.md) · [可复制专家提示词](docs/review_materials/20260919/expert_discussion_prompt.md) · [完成清单](docs/review_materials/20260919/completion.json)

主组2100 correct158、消融159，原窗口增量未保持；匹配controls未证明教学项增强视频特异性。
双相机900/1200/1500 correct117/108/108、fixed other105，未改善当前配方。负结果和不确定性已完整报告。
单seed、八个validation task簇、未校正任务bootstrap限制外推；source不是learned language-only/static prior。
无Test、RL、checkpoint融合、挑视频或旧A/v5.2重训；本轮完成不构成新实验启动授权。
