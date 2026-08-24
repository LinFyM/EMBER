# EMBER findings

只记录跨session仍影响决策的结论。分数与提交的完整历史见`docs/research_history.md`。

## 科学结论

### 1. 输出形式可行，Writer映射未解决

validation8 task-local rank16 oracle为250/400，四suite均有明显收益。因此“冻结PI0.5，只给Action Expert生成一套完整LoRA”
不是根本错误。source只有48/400，说明通用policy不能替代task-conditioned更新。

失败集中在共享Writer：如何让language+video在source-unseen task上生成正确更新。内部hidden、LoRA重建、retrieval或低loss不保证
闭环有效。

### 2. Action Expert内部有可利用的动态结构

成功task experts的跨层、跨horizon response能形成task geometry，完整成功轨迹在held5五个任务上都有可捕获policy effects。
Stage 0 native v3的owner/layer/horizon observer通过基本非退化门。这支持继续利用Action Expert原生时序结构，但不证明当前
event Program已经理解视频过程。

固定`t_flow=1`probe的50个token是按未来horizon排列的Gaussian noise输入。中间hidden是当前language/image条件下的
time-indexed policy response，不是teacher action，也不是已生成动作。

### 3. 视频因果性仍未建立

多个历史Writer的full-video结果接近language-only、video-only或first+final，Goal/Long为0。当前不能声称模型已理解视频过程。
最终方法必须证明full video的必要增量、same-task其它视频鲁棒性、wrong/static controls差异，并在冻结checkpoint上表现出对
shuffled/reversed的时序特异性。

shuffled/reversed不应进入训练或选模，否则会把“对负样本敏感”误当成闭环有用的视频理解。

### 4. task数量与映射多样性是识别问题

train24中language、scene、video和task identity高度耦合，Writer可能记住task特征。可使用审计后的non-held LIBERO-90扩充独立
meta mappings，但不能用更多同task episodes冒充更多任务，也不能泄漏validation/test或形成task dictionary。

当前owner明确不制作人工process数据。因此新的ECP必须诚实评估现成LIBERO是否足以识别Program；若自然数据不能支持强
same-endpoint/opposite-process claim，应收窄claim，而不是暗中补人工任务。

### 5. shared prior有价值，错误residual会伤害

held5 source/shared为21/43；某些task residual反而从43降到37/33。稳定共享底座可以提供支持，但错误的条件更新比不更新更危险。
这要求realizer有no-worse/retention机制和早期closed-loop Gate，而不是只追求更大更新norm。

rank12 carrier+rank4 residual只是历史方案。fixed-A投影与mobile solver结果表明其具体坐标受限；它不再是ECP硬架构。

### 6. policy effects有效，但历史realizer失败

独立successful members达到113/250，说明privileged policy evidence包含真实策略。历史structured solver到78/250但breadth3/5、
Goal/Long0；fixed-A members 49/41/35；mobile raw solver 49；centered coordinate 80且仍breadth3/5。

所以已淘汰的是这些coordinate/solver/realizer组合，不是“Program绝不能生成LoRA”。下一条路线必须用跨任务共享、部署兼容、
坐标固定的realizer，并在训练video posterior前完成held closed-loop Gate。

### 7. `q_pi`的含义与证明责任

`q_pi`是一个训练期共享网络，从multiple successful policies、occupancies和policy responses推断Program posterior。它不是现成
teacher、手工标签或凭空正确的中间state。其合理性只能由task-disjoint、冻结schema/realizer、无held optimizer的闭环结果证明。

`q_V`从language+video预测同构posterior。二者同构的目的，是让privileged supervision指向部署可推断的结构，避免任意latent
和decoder共同旋转。rollout-only recovery信息不能被要求由video预测。

### 8. event slot是固定容量、动态激活

当前Program候选最多`E=8`个有序slots。简单任务可以激活少量，复杂任务可以激活更多；slot presence、视频边界与段落到slot
的对应均由学习得到。固定`E`只是为了让可变长度视频最终进入固定形状Program和LoRA realizer。

### 9. staged Gate之后必须联合训练

阶段冻结用于确定最早失效接口，不是最终模型形态。Program/realizer和`q_V`分别通过后，必须有冻结PI0.5 backbone、解冻所有被
允许Writer参数的联合训练阶段；随后才考虑structured outer credit。

## 已关闭路线

- 旧action-memory、LOOM、CVADR、LMMPC/LPCP及其梯度/credit小变体；
- ECP Stage 1 v1--v24、MDCO和deterministic privileged codes；
- PECS式直接effect solver；
- fixed-A、rank12+rank4惯性分解和raw-factor 12-step solver；
- 人工opposite-order process tasks、primitive/recovery expert acquisition与distillation；
- 把GOMQ重跑或归入ECP阶段。

这些路线可作为历史启发，不再保留活动代码或自动恢复。

## 当前开放问题

等待专家回复明确：

1. 当前Program schema是否保持`P_lang/P_scene/P_process/rho/sigma`，slot和owner/layer坐标是否需要修改；
2. realizer应直接生成LoRA，还是经policy-effect distribution与固定可微operator；
3. 如何固定LoRA因子坐标并进行posterior marginalization；
4. `q_pi`与realizer的冻结/联合训练顺序；
5. 只用现成LIBERO时，哪些natural task mappings足以形成有效task-disjoint Gate；
6. 每阶段明确的数据、模型、目标、通过条件、失败分支与最终全Writer训练方式。

## 工程结论

- source、task expert、Stage 0、functional loss、video data/control、reward occupancy和dynamic evaluation queue是当前可复用基础。
- 旧Writer/functional decoder/ECP Stage 1实现已经造成版本惯性，现已从活动树删除；后续只允许一个canonical Writer路径。
- formal checkpoint与raw rows保留在ignored `runs/`；精确旧代码用Git恢复，不在源码树存archive或fallback。
- 人工process datasets和对应约12GB运行产物已删除；它们可由旧提交重建，但不是当前资产。
- 不新增checksum sidecar、重复证据JSON或一实验一文档；关键结果只更新本文件、`progress.md`和`research_history.md`。
