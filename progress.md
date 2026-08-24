# EMBER progress

更新时间：2026-08-24。分支：`main`。本轮起点：`7ab5a04`。

## 当前状态

owner已经把自然LIBERO-only的ECP核心复核prompt发送给专家，回复尚未提供。本轮只做仓库瘦身、状态固化和跨session交接，
没有启动GPU训练，也没有向专家发送任何消息。

当前active design：`docs/event_conditioned_policy_compiler_design.md`，状态为“骨架有效、开放点等待专家裁决”，不可启动formal run。

## 本轮已完成

### 活动代码收敛

- 删除旧Writer网络、训练、cache、live adapter、generation profile与parallel evaluation实现；
- 删除旧functional fingerprint/code/phase decoder、outer credit和shared residual实现；
- 删除ECP Stage 1 v1--v24后继、MDCO、PECS、effect calibration、fixed realizer、two-sided coordinate和人工process模块；
- 删除projected historical adapter、旧Writer family registry和跨历史benchmark分析路径；
- 删除已冻结后不再使用且接口失效的旧source-base trainer；source authority与Source-SFT路径保留；
- evaluator只保留source SFT与task-local expert静态adapter、动态队列、恢复、聚合和occupancy diagnostics；
- 将仍被多个活动模块复用的基础能力重命名并提升到稳定位置：
  - `meta_protocol.py`：non-held meta split；
  - `privileged_actions.py`：训练期action对齐；
  - `video_conditions.py`：视频control变换。

### 配置、脚本与测试

- 删除退役Writer/ECP/人工process配置与冗余checksum sidecars；
- canonical脚本缩减为source封存/Source-SFT、task experts、ECP Stage 0和PI0.5评测入口；
- 删除只覆盖退役实现的测试；保留当前source、LoRA、experts、Stage 0、functional、reward和evaluation合同测试；
- Python compile及全部126项活动测试通过；所有保留脚本的`--help`入口可加载。

### 文档与证据

- 删除41份旧Markdown设计/审查文件和87份分散证据JSON；
- 重写`README.md`、owner要求、concept、ECP设计、research history、findings、plan和progress；
- 历史精确事实改由一份`research_history.md`、Git提交和formal artifacts恢复，不再让旧文档宣称自己是active；
- 新增临时`HANDOFF.md`，等待专家回复并供下一session消费。

### 精简规模

- tracked/pending tracked文件：456 -> 146；
- Python文件：251 -> 105，代码与测试总行数：75,039 -> 26,118；
- Markdown文件：50 -> 10；
- Git历史与唯一formal artifacts继续承担精确历史恢复，不在活动树保留平行实现。

### 本地运行资产

- 删除两个人工复合数据集（约6.2GB）；
- 删除人工composite/process teacher、recovery和separate-plates运行目录（约5.4GB）；
- 保留原始LIBERO数据、tokenizer、`.venv`、现有task experts、唯一formal checkpoints和非人工历史结果；
- 这些被删人工资产不在Git中，不能直接恢复，但可从对应历史提交重新生成；当前路线明确不再需要。

## 当前可复用资产

- 固定24/8/8 split与过滤后的71-task source corpus；
- 已冻结source PI0.5 authority，以及source SFT训练、resume、validation；
- rank16 LoRA topology和batched/materialized执行；
- train24、non-held meta、validation diagnostic和独立particle task-expert contracts；
- Stage 0 native observer、event binding、Stage 0 training/meta-training/checkpoint；
- Stage 0跨episode video/action pair schedule；
- functional flow loss和detached LoRA gradient bridge；
- privileged reward rollout、occupancy capture；
- cost-balanced PI0.5 evaluator与strict aggregation。

## 当前阻塞

唯一有意阻塞是专家回复。等待其再次明确：ECP最终架构、Program schema、realizer、`q_pi/q_V`训练关系、只用现成LIBERO的
阶段计划、每个Gate和最终全Writer训练方式。

回复前禁止：

- 新建ECP版本、训练新checkpoint或恢复人工任务；
- 把当前设计文档中的开放候选当成专家已确认方案；
- 联系专家或替owner发送补充信息；
- 因等待而转去重跑GOMQ、PECS、v24或无关历史实验。

## 下一动作

1. 等owner粘贴专家回复；
2. 按`task_plan.md`的四步处理回复，提交最终handoff；
3. owner开启新session后，新session完整阅读mandatory docs，消费并删除`HANDOFF.md`，再按冻结计划推进。
