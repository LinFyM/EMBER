# 单次覆盖重训与论文实验合同

2026-09-20 Owner明确授权，采用专家最后修订；替代旧固定1500的训练边界，后继实验仍沿用paper_experiments_design。

## 问题与唯一变量范围

训练支持内的对象和基本操作，迁移到新关系、目标配对和组合。改变任务分布、采样覆盖与停止/选点方式；不声称分离各项因果贡献。
仅一条fresh EMBER轨迹和一条fresh MT-BC轨迹。Source aligned1000及source-only normalization复用；不暖启动旧Writer/MT-BC。
单相机K1、完整38-target rank16 A/B、三Meta、跨episode主21＋同视频教学7、两loss权重和全Writer梯度保持。
MT-BC共享rank128，训练使用同36个任务；其50 demos和Writer46梯度demos差异继续披露。
无新保持loss、k折、第二seed、Unpaired重训、双相机或合并Validation重训。

## 数据与权限

唯一新协议：configs/libero_24_8_8_coverage_v1/protocol.json；manifest统一52个任务身份（目标40＋辅助12），训练36/验证8/测试8。
目标suite内Train/Validation/Test依次为：
- Spatial: 0,1,2,4,5,7 / 3,6 / 8,9。
- Object: 2,3,4,5,7,9 / 1,6 / 0,8。
- Goal: 0,1,2,5,8,9 / 3,6 / 4,7。
- Long: 2,4,5,6,7,8 / 1,9 / 0,3。
辅助libero_90: 2,3,11,15,16,22,24,33,55,56,57,61；辅助global ID=40+suite内ID，不进入部署输入。
数据不复制，所有HDF5复用canonical revision。完整held等价任务不得作为辅助任务；原语/子任务包含关系允许并公开。
覆盖审计见本协议coverage.md；依据canonical BDDL规格与36个训练HDF5 metadata，不声称逐轨迹重认证。
held动作/状态/reward不进入共享训练。Writer每task demo0..45训练、46..49诊断；MT-BC demo0..49训练。
训练、物化和eval都绑定新协议，不修改旧manifest角色，不通过关闭role检查实现新划分。

## 采样、时钟与完整恢复

Writer每4个不同task一次update；36task随机轮转，每9updates覆盖每task一次，轮次与GPU分片独立。
事件由固定seed与task/occurrence决定，保存稳定算法合同、逐条件实际曝光和sampler cursor；扩展区段保留全部历史前缀。
MT-BC每update逻辑576查询，36task各16；4个逻辑rank各144，物理卡数/microbatch只负责分片。
本轮预注册LR时钟按36/24伸展：Writer warmup150、原cosine时钟18000、tail1350..2250后保持0.1倍tail起点LR；
MT-BC warmup150、decay1200，衰减后保持1e-5非零floor。Writer每步只抽4task，因此该伸展近似保持每task曝光相位；
MT-BC每步已覆盖全部task各16query，该伸展是显式延长其每task学习时钟，不能称作与旧配方曝光等价。
时钟与训练终止解耦，后续不改LR、不重置optimizer。
配置maximum_updates/total_steps=null表示没有科学硬终点；每段明确stop_after_step，原topology/optimizer/scheduler/rank RNG完整恢复。

## 完整Validation与停止规则

Writer每100保存、每200完成correct400；MT-BC每25保存、每50完成validation400。种子7、视频映射seed20260911、全50teacher无放回，跨节点严格配对。
每段训练与完整400评测结束后统一读结果，不读取部分成功流，不用monitor代理；错误退出立即处理。
两方法独立按绝对验证曲线停止，不能按相互差距提前停止某方：
1. 最近3节点均低于此前历史最高，均值低至少8条/400，且最近三点线性斜率<=0：持续下降停止。
2. 最近5节点均未刷新历史最高、最高到最低跨度<=8条且线性斜率<=0：平台停止。
任一节点刷新最高则清除该段未改善窗口；正在恢复的趋势不触发上述停止。一次下降不停止，loss下降不否决闭环规则。
非finite、资源错误或预算中断不称科学收敛，保留可恢复完整状态；明显异常按Owner要求报告。
每种方法最终选已评估correct成功最多的单checkpoint；MT-BC同分取更早；EMBER同分候选统一补other400后取other最高，再同分取更早。
最佳、最新恢复点及邻近正式观察点保留完整训练状态；验证后明确不再会使用的非最佳中间ckpt可按资产合同清理。
保留全部登记节点的指标和raw rows，报告邻近曲线、suite/task、breadth、R/G/L、churn与任务簇bootstrap。

## 冻结后的判定与后继实验

新Validation不与旧165做绝对分数比较。冻结后检查other400与视频controls，各400；wrong/shuffle/reverse只做冻结诊断，不用于选点。
正确/换视频的收益相对wrong/顺序干预消失或区间证据不明确、other明显退化时，停止询问，不重选checkpoint。
新Test先Source/MT-BC/EMBER correct各400，EMBER−MT-BC至少40成功才继续Test other/wrong/shuffle/reverse。
相同原语支持不保证EMBER优于MT-BC。任何失败按实际结果停止；不通过换split、降低门槛或追加fresh训练补救。
通过后直接接paper_experiments_design登记的单support FT、必要三support复核、直接A/B task-local RL和外部比较。
旧78/74/82与全部旧协议保留；新划分受旧Test启发，不称研究过程从未接触的盲测。

## 资源与运行

两节点合计卡数遵守AGENTS现行上限；低负载他人GPU可安全共驻，按实际峰值及吞吐选择，不抢占。
优先Writer多卡与MT-BC异节点并行；释放/重用区段之间的卡供完整评测。每次launch检查live卡况，formal来自clean pushed detached runtime。
2026-09-20启动准备quota：data0约90GiB使用、data1约860GiB，各soft1TiB；新输出选data0，初始预留50GiB并控制checkpoint/cache保留。
大产物使用前重新核对peak预算；不复制source/dataset。每个400-LoRA bank约2GiB，完成正式原件与报告后清理无后继依赖载荷。
确切GPU、profile、命令、commit及launch路径在实际启动前登记一次，未launch不得写成运行中。

## 实现边界与验证

继续使用现有Writer和Source-SFT训练器、materializer及official evaluator；没有第二套训练实现。
共享task_protocol负责训练/物化/eval三处身份与role权威，early_stopping是两方法共用的纯完整面板裁决。
Source-SFT control内保留原有配置/恢复边界，新增动态终点和非反弹LR。旧固定配置仅服务历史证据恢复，不是本轮默认。
架构检查两处新增长复杂度已按辅助任务解析与动态时钟校验职责抽取；剩余review项集中在原有checkpoint/run合同函数，
继续由原模块负责，避免为行数机械拆分。新源增长主要为完整resume及权限检查；无平行fallback。
CPU集成原94项中的5个fixture问题修复后视频测试45项通过；新增协议/早停8项通过；最终相关52项通过。
