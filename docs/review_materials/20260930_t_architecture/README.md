# EMBER T 架构数学审阅入口

本材料供只能读取远程 EMBER 仓库的专家，从实际计算图和已有正反证据独立分析 T 的主要限制及改进原理。
研究目标是让语言和 action-hidden 教学视频在 rollout 前一次生成有效的任务 LoRA；不是证明当前架构正确，
也不预先指定 Transformer、删除 B₀、增加回读或扩大 rank。数学上可表示、有限训练中可学、跨初态和任务可迁移须分别判断。

材料日期为 2026-09-30；计算图核对至源码 `5313257c8070a792bbec0c6345c98376ed5cd1ea`。
这只是审阅源码快照，不是全部历史实验的训练或评测 Git 身份。当前授权与进度以 [progress](../../../progress.md) 顶部为准。
成熟原 T 的完整训练曲线已到 2790，选定点为 2340；change_clock 候选只到 450。
design §33 学习限制诊断已完成：有限train面板父17/32、S20、P20、D23；结果与范围已纳入材料，当前暂停新增实验。

可直接转交 [专家提示词](EXPERT_PROMPT.md)。统计与证据范围见 [EVIDENCE](EVIDENCE.md)。

## 先读哪些材料

1. [Owner 要求](../../current_owner_requirements.md)及 [Concept](../../concept.md)：目标、合法信息、理论与证据的关系。
2. 本页的完整计算图，再读 [设计 §1–4](../../designs/operator_read_write_learning_design.md)和实际源码：
   [model.py](../../../src/ember/operator_writer/model.py)、[native.py](../../../src/ember/operator_writer/native.py)、
   [credit.py](../../../src/ember/operator_writer/credit.py)、[data.py](../../../src/ember/operator_writer/data.py)。
3. [EVIDENCE](EVIDENCE.md)：先看完整学习阶段、强参照和机制正反例，再形成主要解释。
4. 按问题读 [机制分析](../../analyses/feature_to_operator_mechanism_20260926.md)的 §42–44、§50–56、§63–76，
   以及 [研究历史](../../research_history.md)索引到的最近似方案。旧论证是可批评的解释，不是审阅结论。

不要求读完所有历史。若建议与 LocalField、ProcessPullback、条件速度、公共 prior 或回读相近，应完整核对那一段原论证和反例。
旧文件中的“当前”“下一步”和执行许可只对应当时，不构成新实验授权。

## 实际输入和完整计算图

source 是从 generic `lerobot/pi05_base` 建立的冻结 π0.5-LIBERO policy；没有使用读过目标 40 tasks actions 的 `pi05_libero`。
当前方法 K=1，输入是 exact task language 和完整、有序、同步双 RGB，stride=5。教学输入没有 action、state、reward、terminal、
task ID、文件名、物体 pose 或 policy outcome。自身执行仍使用合法的当前图像、语言与 8 维 state。

每个 target ℓ 有跨任务共享的 `Aℓ[128,d_in]`、`B0ℓ[d_out,128]`。38 个 target 为 18 层 action-expert Q/V 投影及 action_in/out；
各 target 有自己的参数，不是全网络共用一对矩阵。rank=alpha=128，source 基础权重始终冻结。

公共策略 `βℓ=B0ℓ Aℓ` 参与教学读取和最终执行。A、B₀及 Writer fresh 共同学习；B₀从零初始化，未先训练一套 MT 底座。
M=0 时模型类能表示自由公共 rank128 LoRA，但实际训练不单独优化其独立策略风险，不能据可表示性推断公共分支已学到强 MT。

教学读取对每个采样帧使用真实双相机 patch 和语言 prefix，以及固定 Gaussian probe `[50,32]`（seed1729）、flow time=1。
通过包含公共 β 的真实动作网络取得各层输入 `Xℓ,t[d_in,50]` 和最终投影前 `H_t[1024,50]`。
不同帧在 native reader 中独立处理；每帧内部的原生 attention 已有跨位置作用，不能把 50 列称为相互独立的图像 patch。
公共 β→X/H 在正常训练中有梯度，不能跨 β 更新缓存。图文 embedding 冻结。

令 `h̄_t=RMSNorm(H_t)`、`Δh̄_t=h̄_(t+1)−h̄_t`，每个 target 的原 T 为：

```text
K_t[:,j] = normalize(A X_t[:,j])
V_t = O { GELU(P K_t + C h̄_t) ⊙ D Δh̄_t }
M_0 = 0
M_(t+1) = M_t + (V_t − M_t K_t) K_tᵀ / 50
A_final = A
B_final = B0 + M_end
δyℓ(qℓ) = (B0ℓ + Mℓ,end) Aℓ qℓ
```

P:128→256，C/D:1024→256，O:256→d_out，全部无 bias，O 零初始化。Value 依赖相邻帧变化，没有独立静态 Value shortcut。
M 是一次编译前向中的矩阵状态，不是部署 optimizer 参数；rollout 开始后不再看教学视频，也不更新 M/A/B。
最终只安装一套 38-target LoRA。上式 q 是已经装入完整 LoRA 后的真实层输入，依赖上游参数与自身状态；
不能把固定 q 的线性式当成整个多层、十步生成策略的总效果。

几个会改变推导的边界：

- 教学 key 单位化，执行 `Aq` 不单位化；不是对称 cosine 检索。
- 教学使用公共 B₀，执行使用 B₀+M；共享 A 没有令全部教学/执行特征或参数一致。
- 删除 B₀仍可保留 `K=normalize(AX)` 和 `MAq` 的共享 A 联系，但失去公共教学适配及直接公共执行项。
- action_in 的教学 X 是固定 probe；部分浅层 X 尚未读图像，不能先验赋予对象角色语义。
- action_out 的实际投影用于完成捕获，但输出被丢弃；自身 B₀/A 不经这次投影改变返回的 X/H，不能假设所有 target 都有同样的三路梯度。
- 50 是动作 horizon 的隐位置数，不是视频帧数或 50 段已识别操作。H 的变化也不是教师动作标签。

## 为什么有 T U 和 change_clock

T 的教学地址与执行读出使用同一 A。U 仅把教学 key 中 A 改为独立 S，S 初值 clone A，其余图与事件保持；
U 参数更多，故比较是共享约束干预，不是严格等参数消融。A 的共同坐标和动作梯度提供学习偏置，不能证明语义已对齐。

change_clock 保留 T 的 A/B₀/Value、初始化、训练数据和目标，只改变原递推的覆盖项：

```text
g_t,j = 1 − exp(−RMS(Δh̄_t[:,j]))
M_(t+1) = M_t + [V_t − (M_t K_t) diag(g_t)] K_tᵀ / 50
```

g 不乘新 Value。这个 fresh 候选修正了“变化型 Value 与逐帧固定覆盖”的一个结构不一致；
270 的真实信用传播改变成立，但 450 的绝对能力未改善到强参照水平。停止其本轮追加投入不等于证明更晚训练不可能提高。

## 训练和评测怎样约束这些算子

T/U 使用 24 train tasks 加 12 个经审计的 non-held 辅助任务，共 36 个独立任务映射。
每宏步 4 tasks×28 queries，主监督是同 task、跨 teacher episode 的真实完整 50-position FM；已经存在多 episode 监督。
source、normalization 固定。自身 query 有 RGB/state/actions；这些标签不成为 deployment Writer 的输入。
真实 FM 对生成 LoRA 求余切，再沿同版本 native/Writer 图回放 VJP；A/B₀/P/C/D/O 接受同一完整策略风险的信用。
正常训练是共同学习，不是只训末层 O。没有 RL、部署局部优化或按 teacher action 拟合 M。

固定 Validation8，single checkpoint、每 task50 init、每轮50条不同 teacher、同 scene/RNG 配对，共400条。
执行是 render256/model224、双相机180度旋转、8维 state/7维 action、10步 flow、前5 actions 后 replan、dummy10、成功即停。
训练 FM 含全50及既定尾部 padding，生成策略经过十步与闭环反馈；不能直接用某个 FM 余弦或 M 范数代表控制能力。
MT300 为强能力参照，但早期 T270/450 与它的训练 query 总量不同，不能称全程等曝光比较。

## 远程能核对什么

- 源码和公式、设计、历史论证，以及本目录 CSV 的逐 task/init 成败、学习曲线和得失数，可在仓库内检查。
- 小型机制 JSON 给出本地原件读回后的数值、来源和范围；原始 checkpoint、X/H/余切大张量、RGB、完整连续轨迹不在远程仓库。
  因此专家可以重推数学关系，不能声称重新执行了模型或独立核验了全部特征数值。
- 本目录只导出已有完成结果，没有新增模型计算、训练、环境、数据标签或额外选点。
- Test 以及 shuffled/reversed 的封存结果不进入这次改进设计的论据或材料；它们即使能在历史中找到，也不得反哺方法。
- design §33 的 S/P/D 是已完成的有限训练诊断；逐例结果、动作风险与保存Z的代数分析均有入口，
  不能把四个train tasks上的特权诊断当成正式模型选择、未见任务泛化或已证实根因。

审阅允许撤换 A 绑定、公共 B₀、Value 参数化、矩阵递推、读取方式或输出解码器，只需满足最终任务及信息墙。
把 M 换成 Transformer 须明确是替换内容编码、矩阵更新还是直接生成最终 B；M 作为实际参数这一点本身可以保留。
从现有图证明的缺点，只有在实际特征和学习条件下足以产生已观察不足，才构成主要性能解释。
