# EMBER concept

EMBER研究能否把exact task language与action-hidden正确教学视频，在rollout前一次编译为冻结π0.5 source的一套完整task-conditioned LoRA，
使机器人从未见初始化闭环完成任务。语言说明目标与关注对象，视频中的操作变化应提供必要条件信息；执行由机器人自己的观测触发。
人从他人教学迁移到自己身体的能力是科学动机，LIBERO实验本身不证明跨身体泛化。

当前方法为[Process Pullback Writer](process_pullback_writer_design.md)。本文解释其完整工作原理与边界；
接口、训练和证据节点由该设计规定，实际状态只看[progress](../progress.md)，计划见[task_plan](../task_plan.md)。

## 从原始视频到一次性策略参数

```text
exact language + 一条同步agentview／eye_in_hand教学视频（K=1）
  → 真实双相机prefix，stride5完整视频及真实末帧
  → 冻结vision／Gemma基础 + 共享读取Meta：逐帧图文Z与完整50-horizon动作响应H
  → language引导语义角色对齐，形成逐帧语义状态e_t
  → 真实相邻变化d_t，正向变化递推与反向上下文形成过程Value
  → 过程Value产生每帧、每horizon的7维动作作用码q_t
  → 同视频裸冻结source的输出导数 + 全视频原生X的PCA rank16投影
  → 唯一38-target完整A/B LoRA
  → source按自身当前观测闭环执行，LoRA在rollout期间固定
```

同一演示的两路相机共同形成原生prefix，不能把双视角当成K=2。Teacher的动作、state、reward、terminal或任务身份不进入Writer。
原生动作horizon表示模型动作计算的相对位置；它与teacher-video time、flow time和网络层深分别处理，不能互相替代。

## 语义、动作知识与过程各自的作用

图文语义负责解释画面中的对象、关系和任务关注点。Action Expert的完整50-horizon响应提供原生动作知识，
帮助共享网络在不同画面中形成具有一致角色对应的语义状态。读取侧Action Meta和VL Meta属于Writer，基础权重始终冻结。
完整H保留到实际learned read；它不是已经恢复的教师未来动作，也不能由hidden差异直接宣称理解了过程。

相邻语义变化作为过程内容进入递推，静态语义和裸source动作估计只条件化查询与门。
前后文可以共同解释操作，但正向与反向保留不同的前态／后态角色，以表达抓取前需接近、运输前需抓住等前置关系，
以及语言中“先A后B”的要求。架构对重复相同合法画面施加零变化、零过程作用码的性质；这只排除该静态输入的非零输出，
不证明真实视频的顺序已被有益消费。

## 用冻结source把动作作用转为参数作用

作用码q保留每个真实视频状态的全部50个horizon位置，使用7维实际动作输出坐标；原生32维输出的其余位置补零。
它是作用于裸source输出的余切，不是teacher action标签或一条待播放的轨迹。网络仅从过程Value产生q。

令F0为同一合法双路RGB／language条件下裸source的原生flow输出。对每个LoRA目标层，固定编译关系为：

```text
G_l = (1/T) sum_t J_(W_l) F0(V_t)^T q_t
A_l = 全视频裸source X_l的top16右奇异向量（正交行）
B_l = G_l A_l^T
DeltaW_l = B_l A_l = G_l P_l，P_l = A_l^T A_l
```

这把学习网络的输出约束在有实际动作含义的坐标中，再由冻结source导数确定参数作用；共享网络不再同时学习任意的高维参数译码。
裸source坐标在两组Meta作用域之外计算。所有真实frame／horizon参与投影，完整A/B组成唯一执行LoRA，无第二adapter或task-local候选。
编译在一次Writer调用内部允许固定、只读、多阶段重放，明确使用冻结模型导数；部署没有teacher标签、loss或optimizer。

这一映射只描述F0处的局部关系，有限LoRA的实际行为仍由完整非线性policy决定。PCA保留输入能量，未保证保留有用功能方向，
因此完整学习前须对本次G P出口完成有界的训练侧功能核验；旧G或旧局部场的正例不能代替这项前提。

## 共同学习、迁移与保持

Writer、Action Meta与VL Meta从合法identity开始fresh共同学习，q末端投影初始为零。唯一loss是同task跨episode的真实主执行FM：
教学视频给出任务条件，动作queries来自同task其它episode，避免逐帧复制该教师轨迹。Teacher video本身不需要动作标注或专门q监督。

真实FM先产生完整LoRA余切，再经固定q到LoRA映射的精确伴随回到过程网络及两组Meta；各部分在同一参数版本完成信用后统一更新。
基础source始终冻结，已适配Z/KV/H不跨更新缓存。本轮不采用分段冻结课程、辅助q标签、RL或部署优化。

共享语义与动作坐标、跨episode监督和固定编译关系，是本方法尝试获得跨视频／初始化／任务复用的理由。
固定参数映射可能减少共享译码漂移，但读取器仍可能遗忘、走捷径或不泛化；普通FM与结构性质都不能保证能力获取或保持。
参数在rollout期间固定却作用于随自身观测变化的激活，因此可以形成状态条件化行为，无需按教师视频时钟播放动作。

## 怎样判断方法是否成立

当前优先检验正确视频的实际闭环能力、有益的视频特异性及相邻保持，>145/400仍为长期目标和参照，本阶段不强制。
正式判断使用single-checkpoint strict paired400，结合task／suite、breadth、retained/gained/lost、churn及相邻success-set重合。
能力与相邻资格成立后补same-task-other，冻结选定单checkpoint，再测试wrong／no-video／shuffled／reversed；不另训frame_set作为硬要求。
最终controls不参与训练、选点或架构修正，错误条件退化不能替代正确条件获益。

若有信息量学习后仍缺少正向信号，须降低对实际检验组合的支持并作有限原因分析，不以loss下降无限续训。
是否结束坚持或回到v5.2由owner决定。完整相关正负证据、旧方法及专家论证统一从[研究历史](research_history.md)追溯，
跨轮判断见[findings](../findings.md)；新图不继承历史模型的分数或资格。
