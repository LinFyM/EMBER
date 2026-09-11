# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner 2026-09-11已授权自主高效实施。以[active design](docs/video_functional_writer_design.md)为唯一运行合同，
结合全部历史重新设计，不恢复v5.2底座；暂不要求145/400，但闭环真实收益、跨视频/初态/相邻保持及validation迁移不可替代。

## 计划与完成口径

1. **已完成：具体设计与实现。** 保留T×L到长程过程表示，明确完整H首次读取；实现辅助真实FM和分组cotangent。
   复用官方source/data/完整LoRA出口/evaluator，退役旧活动Writer路径；旧结果留在Git与formal artifacts。
2. **已完成：**验证梯度分配、identity启动、teacher墙、query/noise配对和checkpoint身份；真实长视频profile决定物理batch与段长。
3. **进行中：**fresh主方案与纯FM首段100更新及3968条correct/other闭环已完成；与主方案同配方的无序视觉参照已fresh启动，验收动态收益。先不同时改任务覆盖/rank/K或加入RL。
4. 已登记有界200/300窗口：主方案原拓扑从100 exact-resume，frame_set fresh至300并保留50/100/200/300，最终均76,800queries。报告训练获取与validation strict400，区分辅助诊断与可部署LoRA；相邻保持与最早未成功接口决定后续，不登记300以后。
5. 达到设计登记的正确视频收益、换视频/相邻保持及迁移后，冻结方法和选点，完成视频内容/顺序因果确认，才完成当前goal。
6. 下一阶段另以保持视频收益并提升绝对性能为goal，恢复长期145/400及完整稳定/breadth资格。

每轮只根据真正检验的因素修正；不以辅助loss、wrong退化或单点峰值完成goal。不启动95-task、不恢复旧候选；
新实验live资源与精确命令写run contract，阶段状态写progress，跨轮结论写findings，历史结果写research_history。
