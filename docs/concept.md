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

## 当前假设：Policy-Response Event-to-Factor Writer

ECP两周的正证据已经分别证明：

1. 教学视频在冻结PI0.5内部产生可利用的Action Expert policy-response dynamics；
2. 当前视频的真实target-native X/Y与exact signed pooling具有强task-local rank4闭环容量。

尚未解决的是如何学得可泛化的Program--bank功能映射。旧固定Natural Program到summary、solver、transport、anchor和gate的连续接口
长期不能同时保持correct容量与bank specificity。当前方法因此保留通过验证的原生证据与factor几何，但取消固定Program tuple作为唯一
deployment瓶颈，改用两个可复制扩展的learned模块。

```text
exact language + K ordered action-hidden videos
              -> frozen per-frame PI0.5 evidence capture
                    layer x horizon x probe x owner response
                    current-video true native X/Y banks
              -> learned Policy-Response Video Process Encoder
                    frame-local full policy-response states
                    real-time process innovations + ordered events
              -> learned Current-Video Native Factor Composer
                    repeated same-frame bank -> ordered-event -> rank blocks
                    38 targets x 4 mobile ranks + exact signed raw X/Y pooling
              -> rank4 video residual + frozen rank12 carrier
              -> one complete 38-target rank16 LoRA
              -> frozen PI0.5 closed loop
```

主要learned模块内部只使用可重复attention/MLP blocks。三个不可复制的固定边界是PI0.5原生证据捕获、视频单调顺序与LoRA
materialization；不再为每轮问题增加新的latent坐标。

## PI0.5时序与ordered events

每个视频帧使用原生PI0.5 image-language prefix和固定antithetic action probes。flow时刻`s=1`是denoising噪声端点；50个
horizon positions表示当前静态观测下的future-action policy response field，不是teacher未来50帧或teacher actions。

teacher-video frame time、Action Expert horizon、flow time、layer depth和probe是五个不同轴。新Writer在learned attention前保留
19个layer boundaries、50 horizons、正负probe、layer state、residual increment与flow velocity；禁止用`t+h`把不同frames映射到
共享机器人绝对时钟，也禁止horizon mean或抽样。

owner-matched target-rank-side queries在每个frame用两个并行cross-attention分别读取native prefix + 完整`2 x 50` policy response与
side-matched真实native bank；两个来源各自softmax并直接相加，然后同一种factor block沿真实teacher-video time与rank/side轴建模。
最终只做一次content centering；位置只进入attention Q/K，不作为动态value，静态重复视频不能仅靠位置产生mobile update。每条视频
独立保序，多视频只在集合阶段置换不变聚合。

G2已通过的`P_lang/P_scene/P_process/rho/tau/sigma`仍是初始化、诊断和历史机制证据，但不再是下游唯一固定schema。active图不再
构造HMM、relation marginal、occupancy、C/D分解或其它连续解析链。

## Current-video native factor path

每个q/v/action-in/action-out target继续读取当前视频产生的真实input X、absolute output Y以及adjacent、initial和goal-relative
output differences。输入端只把冻结PI0.5的native prefix、完整2-probe x 50-horizon response与真实bank投影为typed memory；显式
`frame x target x rank x X/Y-side` factor latent在每个同构Unified Policy-Native Factor block内以同一query并行读取policy evidence和
side-native bank，两者独立softmax后相加，并沿真实teacher-frame time以及rank/side轴交互。最终centered逐帧states直接产生two-branch logits，对raw X/Y做exact signed pooling。时序不再
先压成独立Process/event坐标再由Composer解释，静态重复视频也不能由language、owner或位置凭空产生mobile update。

language和静态context只负责grounding query，不能作为factor value或独立输出mobile residual。首版不使用task-expert dictionary、
held retrieval、free learned residual、独立Process/event/gain网络或末端base/contrast变换链。q按8个native query-head groups、
action-in按32个native-width blocks处理；这来自真实output tensor布局，不是四条不同compiler。learned主干只有一种可复制block，
扩展深度不引入新的数学阶段。

rank4 factors只做一次small-core canonicalization，再与frozen rank12 carrier拼成唯一rank16。PNBTT、EBSRI、Program-through-bank及
旧G3实现保留为历史和kernel来源，不构成active fallback。完整合同见
`docs/unified_policy_native_factor_writer_design.md`。

## 训练原则

- 只使用现成且授权的LIBERO tasks，不制作人工process数据集。
- train24与审计后的non-held LIBERO-90 meta tasks产生梯度；validation/test不产生梯度。
- video与action query跨episode；多个successful policies用独立优化lineages构成分布，不把同一轨迹的checkpoint当独立任务知识。
- task-local free-code已经证明native factor bank与pooling有容量；P0/P1进一步证明full-inverse primitive能跨video保留共享primal，
  但wrong-bank反事实证明它会消除bank specificity。Program-through-bank的scope-matched free-summary正控通过，真实Program read却未保留
  correct/held；随后bank-conditioned-primal恢复correct，但原query、free query和充分行使的full-native free anchor都无法同时压低wrong。
  当前Native-Temporal Axial Writer已整体替换反复失败的shared utility接口；后继负结果必须定位并替换责任模块，不能继续用scalar gate、anchor步长、
  normalization或普通超参修补同一函数类。
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
