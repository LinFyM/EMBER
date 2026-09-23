# EMBER concept

EMBER研究能否把exact task language与action-hidden正确教学视频，在rollout前一次编译为冻结π0.5 source的一套完整task-conditioned LoRA，
使机器人从未见初始化闭环完成任务。语言说明目标与关注对象，视频中的操作内容和顺序应提供必要条件信息。
人从他人教学迁移到自己身体的能力是科学动机；LIBERO结果本身不证明跨身体泛化。

本文保留科学对象、信息流与证据标准。稳定要求以[Owner要求](current_owner_requirements.md)为准；
当前授权及active design只看[progress](../progress.md)，已完成实例的配置、曲线与结论由封存设计和报告保存。

## 可达能力、学习过程与部署效果是三层问题

令Writer输出为`G_phi(language, video)`。Owner提出的固定LoRA退化解是
`G_phi(language, video) = theta_shared`：如果模型类能表示这一常量映射，同rank共享LoRA的可达策略就在其解空间中。
这是理想可达能力的下界参照，不是对有限训练轨迹的保证，也不推出先训练公共底座的课程。
研究问题是端到端学习怎样实现并超过这一解；不能把rank差异或当前实现限制当作研究结论。

性能是正确条件下的闭环能力；特异性用于解释教学内容的有益作用；稳定性描述能力如何在训练和条件变化中保持。
视频输入引起Proxy、LoRA或动作变化，最多证明影响存在。正确视频和另一正确视频都获得能力，且相对于合法语言/static参照
有增量，才更接近“Writer利用了教学内容”的主张；仅wrong下降或时间重排敏感不够。

下面三点是2026-09-23讨论形成的解释框架，**不是已经验证的根因或新实验授权**：

1. **生成参数的学习几何。** 写`theta_i=G_phi(z_i)`，`J_i=dG_phi(z_i)/dphi`，
   `g_j=dL_j/dtheta_j`；普通SGD的一阶输出位移为`delta theta_i ≈ -eta sum_j alpha_j J_i J_j^T g_j`。
   因而任务间功能迁移依赖生成器建立的条件联系；Adam、裁剪和有限步幅还须按真实更新核验。
   这个公式不证明梯度冲突、某个头受限或改归一化有效。当前私有Writer与自由A/B的局部比较未显示后者明显占优，不能沿用旧模型的容量诊断。
2. **辅助目标改变了什么。** 实际flow输入`x_tau=(1-tau)*a+tau*epsilon`，目标`epsilon-a`。
   tau=1时速度MSE等于`epsilon-v(epsilon,1,condition)`对真实动作的MSE；它从纯噪声端点强调条件预测，
   但条件也含执行观测和语言，不保证教学视频被利用。第二组从mean50改成mean5，使其中每个执行前缀位置的名义系数增大10倍；
   这不是实测梯度10倍，也不是10步flow最终动作误差。端点、horizon和episode配对是不同变量，不混称标准FM或单一辅助机制。
3. **学习条件映射需要任务关系。** 同task跨episode查询有助于阻断逐帧复制，但同task固定语言和多条视频也允许已见任务识别。
   更多episode不等于更多独立任务映射；训练支持、视觉表征、共享参数和损失一起决定是否学出可迁移操作关系。
   这不等于视频无信息或理论上必然无用。检验数据原因应固定诊断目标，再控制训练支持与总监督预算。

固定四臂只能分辨直接参数优化、语言条件生成、完整视频路径和目标组合的整体差异；不能单独定位具体模块或数据根因。
后继须预先区分竞争解释，再以针对性干预检验功能预测及闭环收益，而非把四条曲线本身称为根因答案。

## 信息如何成为执行能力

```text
exact task language + action-hidden、内部有序的正确教学RGB视频
  → stride5及真实末帧；每条视频独立编码，保留时序与视角身份
  → 冻结source的真实图文prefix与原生Action Expert读取
  → 任务相关视觉内容、动作计算响应及跨帧操作关系
  → 共享Writer把这些条件编译为唯一38-target完整A/B LoRA
  → rollout前固定参数；source根据机器人自身观测与state闭环执行
```

语言负责目标与关注对象，视频提供实际操作及其后果，执行观测确定机器人目前应做什么。
共享Text/VL/Action读取Meta属于Writer；source基础权重冻结。原生动作知识必须在视频理解中有明确计算作用，
不能以无真实prefix、假动作query或无意义forward代替。H是条件化动作计算响应，并非恢复出的teacher action或待执行标签；
完整50个horizon位置应保留到实际learned read。Video time、action horizon、flow time和layer depth分别解释。

操作内容、顺序与条件作用可以由不同结构表达；Core、Procedure、memory token和具体decoder都是候选实现。
模块须职责清楚、能复制加深并自然扩参；选择由完整相关历史及实际行为决定，不把某次局部诊断变成永久结构限制。
共享参数和结构依赖提供学习偏置，不保证视频必要性、迁移或能力保持。

## 信息墙与共同学习

Teacher action、state/proprio、reward、terminal、task ID、filename、pose和policy outcome不进入部署Writer。
执行policy使用机器人自己的观测/state，与teacher信息墙不同。一套生成LoRA覆盖所有38个执行目标，
不挑视频、不平均最终LoRA、不融合checkpoint，也不部署第二expert adapter。
两相机是同一时点的同步观测，不是K=2；如声称dynamic K，必须训练覆盖各cardinality，并只在集合阶段做置换不变聚合。

主监督来自同task、严格跨episode的真实执行FM。授权的训练期辅助标签须登记来源、梯度消费者和权重，
只能在生成之后用于监督，不得进入条件表示。Writer与读取Meta从fresh状态共同学习，source冻结；
合法identity可使首步部分上游信用为零，须核实后续实际学习过程，不能由单次梯度或机制smoke宣称方法成功。
适配的Z/KV/H不能跨参数更新缓存，validation/test不产生梯度。部署无teacher标签、loss、optimizer或环境试错。

## 怎样判断

首先看single-checkpoint strict paired400的正确视频绝对能力，再看相邻成功保持、任务广度、四suite、获取/丢失与churn。
性能接受标准服从Owner最新要求，历史门槛只约束对应封存实验；loss、参数几何和内部时序margin不替代闭环。
正式K1每task每轮使用全部50条teacher各一次，state–video映射与policy RNG跨checkpoint固定。

同task换视频检验条件覆盖；选定并冻结checkpoint后，以严格配对的wrong／no-video／shuffled／reversed controls检查视频作用。
顺序干预须重排真实frames后完整forward，不能用hidden扰动代替。输入敏感、顺序敏感和正确教学带来有益增量分别判断；
source不是learned language-only/static prior。开发诊断及其反馈范围须另有明确登记，Test默认封闭。
连续有信息量节点没有改善时，应综合证据裁决实际检验的假设，不靠无依据的小扫或无限续训延长。

## 已完成方法与证据入口

- [同视频教学设计](designs/video_teaching_writer_design.md)与[最终报告](review_materials/20260919/final_report.md)：完整A/B生成、H与相邻视觉重复读取、跨episode主FM和同视频短程功能信用，以及匹配消融、续训和双相机比较。
- [learned frame-set合同](designs/learned_frameset_reference_design.md)与[匹配诊断报告](review_materials/20260918/frameset_report.md)：给定播放顺序的学习作用及相邻能力边界。
- [统一Writer封存设计](designs/v52_evidence_based_writer_design.md)与[46组历史证据审计](analyses/v52_evidence_audit_20260917.md)：统一表示实例、v5.2及后继正负证据与比较条件。
- [研究历史](research_history.md)与[findings](../findings.md)：旧专家论证、源码commit、formal原件和跨轮结论。保留历史不构成恢复执行授权。
