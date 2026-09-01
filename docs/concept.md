# EMBER concept

## 问题定义

人看过一段没有动作标注的教学视频，通常会先理解目标，再把视频中的条件、过程和结果迁移到自己的身体与当前场景。EMBER
研究PI0.5能否做同一件事：只看task language和`K`条action-hidden正确视频，在rollout前把观察到的知识编译成Action
Expert的一套LoRA，随后零交互完成任务。

这不是视频检索、task-ID分类、行为克隆或运行时视频条件策略。部署时没有teacher action、state、reward和第二个expert；
Writer只运行一次，输出的参数必须直接成为闭环策略的一部分。

## 为什么问题困难

原生PI0.5中，Gemma处理当前language和静态图像prefix，Action Expert把50个未来horizon位置上的noise tokens通过flow
matching推进为动作chunk。教学视频则是一串跨时间的静态帧，而且没有teacher actions。EMBER必须同时解决三个接口：

1. 从帧级PI0.5表示中提取与动作过程相关、而非只识别物体或task模板的动态证据；
2. 把可变长度、可变`K`的视频压缩成保留event顺序和Action Expert层对应关系的固定结构；
3. 让这个结构直接从PI0.5各LoRA目标的原生input/output空间选择低秩因子，而不是从低维latent凭空生成高维参数或把held更新
   投影回fit-task固定span。

训练task数量有限还会造成欠识别：language、video和task identity可能高度相关，模型即使完全忽略过程也能降低训练loss。因此
方法必须靠task-disjoint评测、视频controls、多个独立策略lineages和真实closed-loop结果证明因果路径。

## ECP假设

当前方法称为ECP Native-Factor Compiler。核心假设是：教学视频中的可迁移知识可以表示为一个有序、event-conditioned、与
Action Expert target对齐的Program；同一condition在冻结PI0.5各目标层中产生的原生input/output activations提供task-specific
参数基底，Program只学习选择、组合与缩放这些向量。

```text
exact language + ordered action-hidden videos
              -> Pass A: q_V(owner-specific Program)
              -> Program queries read the real native bank
              -> Pass B0: scope-matched target/rank/event/type bank responses
              -> Program + current-bank content jointly form the primal direction
              -> Pass B1: current-bank dual and exact signed candidate pooling
              -> one exact signed measure over real X/Y
              -> rank4 task residual + frozen rank12 carrier
              -> one complete rank16 LoRA
              -> frozen PI0.5 closed loop

training only: successful policies + verified occupancies/actions/effects/reward
              -> nonparametric set-valued functional critic
              -> supervise generated policy response, never produce Program
```

canonical不再包含神经`q_pi`或privileged Program teacher。privileged evidence仍重要，但只在训练时以global-member set loss和
functional/effect critic约束最终生成的policy，不进入deployment forward，也不要求video预测不可观察的recovery信息。

## Program候选结构

每个视频帧使用原生PI0.5 prefix和一组固定Gaussian action probes。flow时刻`t=1`表示denoising的噪声端点：输入仍是50个
按未来horizon排列的noise tokens；它们的中间hidden是当前language/image条件下的时间索引policy response，不是已经预测好的
50步动作，也不包含teacher action。

当前Stage 0候选保留38个LoRA target owners、50个horizon位置和各层hidden，再将帧序列分段为最多`E=8`个有序event slots。
`E=8`是固定最大容量；每个任务实际激活多少slot、哪个视频段落写入哪个slot均由模型学习。跨视频聚合只在event对齐后进行。

当前候选Program为：

```text
P_lang    [38, 128]
P_scene   [38, 128]
P_process [8, 38, 128]
rho       [8]            # event presence
tau       [8, 2]         # event center and duration
sigma     [8, 38, 128]   # cross-video uncertainty
```

这是专家复核后固定的schema。Pass B另外读取每个q/v/action-in/action-out目标的真实input/output以及output的adjacent、init、goal
differences；这些量不是Program字段，也不在内存中整段物化。稳定结构原则是Program只能以query等方式读取当前真实bank，不能用
absolute code绕过bank直接决定LoRA；Program与current-bank content必须共同形成方向，再由当前bank上的唯一signed measure对真实X/Y做
exact pooling并产生rank4 outer products。内部两阶段读取仍属于rollout前一次Writer调用。2026-09-01的Program-through-bank实验链已经
停止一种具体实现：高相似summary经family-scalar gate调制共享event-additive anchor不能提供足够wrong-bank功能分离。2026-09-02
owner采纳全局专家主选A，当前active realization为PNBTT：Program只形成低维query，真实candidate形成低维key且自身继续作为唯一native
value；B0只建立current-bank key covariance/whitening，B1用同一bank上的antithetic signed measure直接pool真实X/Y。这样wrong bank会
改变矩阵值transport，而不是只改变一个scalar或公共anchor幅度。完整合同见
`docs/program_conditioned_native_bank_tangent_transport_design.md`。

## 训练原则

- 只使用现成且授权的LIBERO tasks，不制作人工process数据集。
- train24与审计后的non-held LIBERO-90 meta tasks产生梯度；validation/test不产生梯度。
- video与action query跨episode；多个successful policies用独立优化lineages构成分布，不把同一轨迹的checkpoint当独立任务知识。
- task-local free-code已经证明native factor bank与pooling有容量；P0/P1进一步证明full-inverse primitive能跨video保留共享primal，
  但wrong-bank反事实证明它会消除bank specificity。Program-through-bank的scope-matched free-summary正控通过，真实Program read却未保留
  correct/held；随后bank-conditioned-primal恢复correct，但原query、free query和充分行使的full-native free anchor都无法同时压低wrong。
  后继若获授权必须直接修复Program与当前bank共同决定功能方向的接口，不能继续用scalar gate、anchor步长或普通超参修补同一函数类。
- staged gates用于定位接口，不是Final必须重演的训练课程。Final既保留从已验证组件初始化的fresh joint run，也保留整套Writer
  完全随机初始化并直接端到端fresh训练的正式选项，由同一closed-loop合同选择。
- shuffled/reversed只在最终selected checkpoint已选定并冻结后评测时序特异性，不进入训练、loss、
  checkpoint选择、G1--G5 Gate或架构修正依据。

## 目前知道与不知道的

已经知道：task-local rank16 LoRA有闭环容量；Action Expert内部能捕获任务相关动态结构；rank12 carrier有有限支持；mobile rank4
解析投影在held5具有5/5容量；policy-effect objective对known-success paths有用；fit-span realizer会丢失held低能量创新。12+4因此
是首版最合理的参数分配，但不是不可由capacity evidence推翻的永久结论。

G1已经证明自然视频产生的target-native banks与exact signed pooling可形成强task-local rank4 residual；原G2动态Gate证明Natural
Program保留了可用视频动态；P0/P1证明full-inverse primitive有同任务跨video容量。R10又证明真实functional credit能把Natural Program
推到中等效用，而R5 cross-bank、R12/R13和最新Program-through-bank链共同证明：现有共享坐标和scalar-gated additive anchor尚不能把
bank中已经存在的差异转为足够强的功能分离。当前知道的最早缺口是Program与current-bank content的联合方向形成；不知道的是哪种共享、
可泛化且不绕过video的结构能解决它。该负结果不淘汰Program schema、Stage0、真实native X/Y、signed pooling、rank4或ECP整体。
