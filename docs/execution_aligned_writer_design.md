# Execution-Aligned Video Writer

2026-09-12登记。当前阶段先修正并验证执行监督的时间对应，再完成有界有序／无序比较。
Owner授权内推进；整体目标仍是正确action-hidden内容和顺序在唯一完整LoRA中的可重复闭环收益，
跨同task视频、初始化、相邻checkpoint及固定validation迁移。暂不要求145/400。

## 1. 综合判断与本次选择

冻结视频先验的train200有序52对静态40是实际局部正例；validation100差额下界0、200差额为负，
相邻成功集合漂移，未建立迁移。此前辅助功能读出、局部动作反演、VL适配、覆盖与输出共享干预的边界见findings§62–70。
早期v5.2/v6普通FM有较强能力，且架构与更新组织有反向交互；因此不能把FM、共享参数或单probe普遍判为不可能。
当前已具有VL及Action Meta，不存在一个尚未打开的同名模块。显式prior、再次改D共享或再加辅助头均缺少新的修复依据。

另一项已核实、但此前为隔离局部监督变量而保留的事实是：当前LIBERO保存的是post-action RGB/state，
主FM却用相同数组索引取动作。生产证据与本地时序核对见[原分析§9](video_process_acquisition_analysis.md#9-当前数据的实际时间对应post-action-rgb)。
在train24的action16–41共624条episode中，相邻原始动作差异的任务／episode等权7维MSE为.00946481，
前6维MSE为.00162719，夹爪切换率为.01412262。这是原始控制量，不能与source归一化FM数值直接相比；
它只量化标签改变，不能证明行为改善、优化冲突或视频过程的根因。

本次先把学习目标与执行时刻对齐。它不新增表示模块、不扩大任务池、不替换知识源，
也不预言一位偏移能解释全部历史。保留本轮已出现训练有序收益的完整模型，改变一项主要科学变量：
执行观测对应的未来动作序列。新旧采样支持不同，旧分数仅作历史参照，不能称逐query匹配的因果对照。

## 2. 时间与数据合同

固定数据生产关系为`states[i]=s_i, actions[i]=a_i, obs[i]=o_(i+1)`。
执行policy在`obs[i]`及其同一时刻8维proprio下，应预测从`a_(i+1)`开始的动作，
因此长度N的episode只允许query frame `0..N-2`，50-horizon目标为`actions[i+1:i+51]`。
不足50个真实未来动作时沿既有official末动作padding；不补造末帧之后的新真实动作。
最后的post-action观测没有下一条动作标签，不进入执行监督，但完整teacher视频仍保留真实末帧。

train24／teacher0–15／action16–41／动作诊断42–45／held teacher46–49、K1、stride5均保持。
每次仍四suite各一task、一条video、64queries，task权重1/4；action episode均匀后在合法query frames均匀采样。
两臂使用同一任务／video／query种子、合法frame支持、实际动作／noise／time与初始化，实际曝光逐项核验。
query trace保留观察frame与动作起点，二者只用于训练审计，不进入teacher模型输入。
Source与normalization冻结，不重训source、不重算held统计；已有SFT／expert独立消费者的同索引历史合同显式保留，
本轮Writer唯一入口强制offset1，不提供旧Writer兼容或resume路径。

## 3. 完整模型与学习

exact language＋完整RGB → 冻结vision／Gemma与teacher VL Meta、Action Meta的完整50-H响应，
以及冻结V-JEPA2.1过去四帧dense先验 → task-token视觉与native H读取 → 过去单向T×L表示
→ 两层target/rank Compiler → 完整native A/B → 唯一38-target rank16 LoRA。
冻结policy读取自己当前观测／state、该未来动作chunk的加噪输入与flow time，执行原生完整主FM。
保持[前轮模型§2–4](pretrained_video_grounded_writer_design.md)的精确encoder、Value路径与无序参照定义。

只有主FM；Writer、两组Meta、Compiler与D全部fresh共同学习，合法identity LoRA、fresh AdamW及scheduler。
不使用辅助预测、专家蒸馏、RL、task-local优化或训练中的错视频／乱序／逆序标签。
训练schema、data version及run stage更新，旧checkpoint只在冻结运行树中复现，不能exact-resume到新标签合同。

## 4. 验证与有界行为裁决

先以真实post-action时序的最小数据回归检查：当前RGB／state保持同一时刻，首动作是下一次控制，
末尾不足horizon仅重复最后真实动作，最后无未来标签的观测不被采样；训练与动作留出共用该对应。
保留source/SFT消费者既有显式offset0行为，检查新Writer拒绝offset0及旧训练身份。
随后对真实train episode经实际dataset取样核对动作起点和尾端，不重放环境或建立全量数据副本。

模型图、每条件queries、物理batch和prior计算没有增加；沿用前轮最长93帧的真实profile量级，
在看到新分数前固定100/200两个checkpoint、每臂200updates／800条件／51,200queries。
新label domain的0/100/200留出诊断单独报告，不与旧同索引loss直接比较。
两臂先完成两个节点train96与validation strict400 correct，共8面板／1,984rows。

行为资格保持前轮的精确口径：validation400 correct对匹配充分训练frame_set的task-cluster bootstrap95%下界>0，
至少两个suite净收益，相邻增量同向，correct相对source有实际收益；报告全部task/suite、breadth、R/G/L、churn与Jaccard。
仅出现相邻正向候选才补same-task-other，并按原定义补条件性原生image静态参照；两者资格通过后才能选定single checkpoint。
选点冻结后才运行sealed seed20260912的correct/other/cross-suite-wrong/shuffled/reversed，内容与顺序主要差额下界>0。
开发seed20260911、env/policy seed7、validation每task50视频各一次及train held46–49/states32–35固定映射保持。

若只有两臂共同能力改善而无有序差额，只支持时间对应对能力有帮助，不支持过程目标；
若训练有序增量仍不迁移，停止把该对应视为迁移修复；若正确条件后段退化或资格失败，结束有界学习，
不追加节点或扫描offset、LR、rank、encoder窗口。时间正确性由生产合同确定，不因分数低而恢复错误时间解释。
若获得相邻和换视频有益过程，才完成上述资格与最终controls；辅助损失或错误条件退化不能代替目标。

## 5. 执行与保留

复用唯一trainer／materializer／dynamic persistent evaluator，不新增第二模型或数据副本。
当前实现所有权为writer/data.py的显式动作偏移、learning_data.py的唯一Writer对应，training/materialization负责新身份。
小型CPU审计只读取train action池，不读取validation/test动作或执行新rollout。
formal启动前从clean pushed detached commit冻结，并刷新两节点GPU、独立quota与整轮峰值预算；
Owner两节点合计最多8卡，空闲合计不超过10时最多6卡，单节点至多6卡。exact命令与资源写launch记录。
本设计登记不表示新训练已经启动；当前进度只看progress.md。


## 6. 实施检查

唯一Writer训练／留出强制offset1；共享dataset以必填offset让既有source-SFT／expert消费者显式保持offset0，
不是可切回的Writer运行模式。新run／training schema、stage与data version拒绝旧合同；三个当前配置复用同一图。
378项测试通过，包含下一观察转移的动作oracle、末尾padding、采样支持及新身份拒绝旧标签；
四suite训练任务0/12/20/34的demo16首／末合法query共8例通过真实dataset检查，使用既有FP32数据转换。
旧模型checkpoint与配置保留在611770d1冻结运行面；本实现不修改其分数或重训source。
