# EMBER 独立审查补充材料

本包补充第一轮专家无法从远程读取的现存实验记录。下文保留2026-09-07补证据时的讨论边界；当前方法与授权见
[progress](../../../progress.md)及[后续讨论与Owner裁决](../20260908/README.md)，不能把本包历史暂停或建议覆盖最新接续安排。
当时科研实现与实验执行暂停，上传材料不等于采纳专家的实验建议。
当前 layered Writer、旧 P/Q width256 与未合并 native-heads 草稿均不因本包而自动恢复执行。

科学代码参考提交为 ec02710b169b3dc624b6dfca998a4ad9bdc8dd14。历史实验各自的代码版本、配置、数据与恢复身份由其原始合同记录。
本包没有新训练、新 rollout、新 checkpoint 选择，也没有把不同实验的最好结果合并。
共覆盖511份现存源文件，导出843个记录文件；其中95个rollout面板包含24,100条结果，另有17张诊断行表。
导出记录约96 MiB，已按原件复核保留的科学字段与行序；总量包含policy RNG等配对证据，不应作为一段上下文全文阅读。

## 阅读入口

1. [第一轮专家意见](expert_review_round1.md)：Owner提供意见的路径规范化副本；保留原论证、数字和建议，作为待核验观点。
2. [第二轮讨论 prompt](ROUND2_PROMPT.md)：先校准证据，再从需求、信息与监督逐步推导架构和训练，不进入实现。
3. [原件与导出文件索引](index.json)：每个条目给出原始仓库相对路径、所属组、导出文件、记录数和转换说明。
4. [逐条结果复算汇总](rollout_summary.json)：从本包所有 outcome rows 重新计算总成功数、per-task、per-suite与breadth；不是新评测。
5. [项目分层历史](../../research_history.md)：寻找各实验原设计、原代码、以前专家意见、正负结果与后续修正。

不要把文件夹名称、曾经的“下一步”或历史建议当作当前授权；也不要把“材料已上传”理解为所有因果解释都已经核实。

## 证据分组

| 分组 | 内容 |
| --- | --- |
| [current_train24](records/current_train24/) | 当前384步run合同、metrics、exposures、192/384资格评测、熟悉/held视频诊断、登记与裁决 |
| [current_source_reference](records/current_source_reference/) | 当前严格400行比较使用的47/400 source原始参照 |
| [current_mechanism](records/current_mechanism/) | 已有真实GPU机制/profile记录；只作工程与成本证据 |
| [current_short4_original](records/current_short4_original/)、[current_coordinate_control](records/current_coordinate_control/)、[current_readout_control](records/current_readout_control/) | 原始short4及两次受控修正的合同、完整已选运行指标、配对行为与诊断 |
| [historical_v52](records/historical_v52/)、[historical_v52_taskcomplete](records/historical_v52_taskcomplete/)、[historical_v6_fast](records/historical_v6_fast/) | 早期强Writer的训练合同、学习曲线、correct及同checkpoint视频controls |
| [historical_source_sft](records/historical_source_sft/) | rank128 train24 SFT合同、metrics、400/425结果及已有口径兼容性核查 |
| [historical_dynamic_slot](records/historical_dynamic_slot/)、[historical_shared_core](records/historical_shared_core/)、[historical_as139](records/historical_as139/)、[historical_lpcp](records/historical_lpcp/) | 后续多视频方法的初始化继承、冻结范围、训练与结果；两个AS139相关方法不能混同 |
| [historical_gomq](records/historical_gomq/)、[historical_gomq_rank16](records/historical_gomq_rank16/) | GOMQ已有周期与视频controls、训练credit记录、等效rank16重物化；不把151解释成稳定selected模型 |
| [pq_shared](records/pq_shared/)、[pq_single_task](records/pq_single_task/) | 完整P/Q共享与同图clone的已有功能/行为比较 |
| [pq_random128](records/pq_random128/)、[pq_random256](records/pq_random256/) | 匹配曝光的fully-random宽度对照合同、指标和checkpoint元数据；width256没有闭环结果 |

精确覆盖以index.json为准。这是针对第一轮审查缺口的补充包，不是整个runs目录，也不代表所有历史分支的原始记录都已公开。

## 如何还原与使用记录

- 普通JSON保留科学内容并压紧排版。
- JSONL在完整record边界拆成part001、part002等文件，按索引exports顺序连接即可还原完整导出记录。
- 含rows数组的JSON拆为同名metadata.json与rows.part*.jsonl；metadata中的review_export_rows指向对应记录。
  把这些行依次读入数组并放回rows字段，即得到路径规范化后的原始结果对象。
- index的result_rows表示逐条rollout结果；rows_table表示functional、几何或其它诊断表，不为后者计算success分数。
- task/state、视频条件与帧序号、policy/env RNG、success、steps、loss、权重、曝光等科学字段保留。
- 主机、进程、CPU affinity、GPU UUID、认证、启动命令及资源preflight/snapshot等非科学现场字段移除；机器绝对路径、私网IP与邮箱改为占位符。
  原始文件仍保存在canonical本地位置，因此这些导出文件不声称与原件字节相同。
- 原件已经存在的身份标识/哈希在适用字段保留；本包没有新增逐文件或逐tensor的hash sidecars。
- index中的expected_rows_from_panel_name与declared_panel_complete仅帮助识别未完成面板；最终是否可比较仍须核对具体合同。
- 不应把几百个JSON全文一次性塞进上下文。先用索引定位相应实验，再用程序复算需要的统计，并检查关键具体行。

例如，复算某个结果时应连接该条目列出的所有rows分片，按success计数，并按suite/task_id/init_state_id对齐。
成功集合比较还要核对语言、source、normalization、视频条件及RNG；仅总行数相同不构成strict pairing。

## 仍缺少或明确不提供的内容

- width256尚未产生的闭环结果：不能通过整理材料补出，不为此启动评测。
- 模型、optimizer、数据集和大视频/张量；本包提供必要checkpoint元数据，不复制checkpoint实体。
- 当前失败轨迹的新录像或完整逐步观测/动作记录：没有在本轮重新运行环境生成这些内容。
- 用户私有聊天、凭据、主机配置和实时资源状态。
- 未合并、未验证的native-heads草稿没有变成当前实现；其存在也不构成下一路线的决定。

如果某个判断依赖上述缺口，请明确指出它影响哪一结论，以及最少还需要哪些已有记录或新实验。
缺口本身不证明某一假设成立，不要求自动重训、恢复旧任务或调整评测标准。

## 第二轮讨论的边界

架构、训练方式和数据监督必须一起推导。旧方法既有真实能力，也有未完成的问题；当前图既有实现资产，也有尚未证明的职责。
数学表达的自洽、局部可达性、内部差异或梯度非零都不能替代共享闭环能力。

请先根据本包修正第一轮意见的证据等级和事实口径，再解释每一个建议如何来自已有证据、如何面对历史反证，
以及什么结果会推翻它。讨论完成、Owner明确决定后，才重新登记科研执行。
