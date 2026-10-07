# 从教学特征到实际控制：已执行的两条主链

本文件说明实际计算，特征具有何种操作语义仍需证据。完整边界见[Owner 要求](../../current_owner_requirements.md)。
原 T 的更多实现细节见[既有机制说明](../20260930_t_architecture/README.md)；C 见[完整设计](../../designs/conditional_read_write_architecture.md)。

## 共同的数据、原生读取和部署

source 从 generic `lerobot/pi05_base` 在经 specification 审计保留的 LIBERO-90 中 **71 tasks×50 成功 episodes**
上建立后冻结；不是读过目标40 actions的 `pi05_libero`。normalization 随 source 冻结。
目标固定24 train / 8 validation / 8 test；T/C通常使用train24+12 audited support，共36个独立meta-task映射。
source见过71tasks不等于Writer已在71个video→task-adaptation映射上学习。

一条教学视频包含同步双相机 RGB，stride5并保留末帧，附exact language；teacher state/action/reward/terminal不进Writer。
每帧走真实视觉语言prefix：两相机共512视觉tokens、既定200语言位置及mask。teacher prompt省略State段，
并非state填零。固定seed1729的Gaussian probe为50×32，flow time τ=1；每帧不重新抽probe。
由公共LoRA β 参与的18层action expert读出：

- `H[t]∈R^(1024×50)`：末端norm之后、action_out之前的native hidden。
- `X_l[t]∈R^(d_in,l×50)`：38个target各自真实projection输入，不是38份相同H。
- 50是预测horizon槽，t才是视频时间；τ是flow time，l是层。H差分不是teacher action标签。

每层LoRA rank=alpha=128，scale=1。18个Q为1024→2048，18个V为1024→256，action_in为32→1024，
action_out为1024→32。最终仅安装一套38-target LoRA，Writer退出。
自身执行输入是当前双图/state/language/自身action noise；完整policy做10步flow，预测50 actions，实际执行前5后重新观测。
真实层输入q依赖自身状态、flow步骤和上游全部LoRA，不能把某个固定q的线性公式当成整个闭环作用。

源码入口：[native.py](https://github.com/LinFyM/EMBER/blob/85919994/src/ember/operator_writer/native.py)、
[model.py](https://github.com/LinFyM/EMBER/blob/85919994/src/ember/operator_writer/model.py)。
这些链接固定到实际C900来源，历史T入口与公式相同处由[旧材料](../20260930_t_architecture/README.md)说明。

## 原 T：公共读取坐标与视频写入

β含各层A/B₀，共同参与教学native；T的教学key与执行读取绑定同一A。省略target下标，列为horizon槽：

```text
h̄_t = RMSNorm(H_t)                    Δh̄_t = h̄_(t+1) − h̄_t
K_t = column_normalize(A X_t)
V_t = O { GELU(P K_t + C h̄_t) ⊙ D Δh̄_t }
M_0 = 0
M_(t+1) = M_t + (V_t − M_t K_t) K_tᵀ / 50
A_final = A                           B_final = B₀ + M_end
δy_l(q_l) = (B₀,l + M_l,end) A_l q_l
```

P:128→256，C/D:1024→256，O:256→d_out，均无bias；O零初始化。
局部变化提供Value内容，静态H与key选择/调制内容。M是一次编译前向的矩阵状态，不是部署optimizer。
递推中的`−MK`会按当前地址覆盖旧内容；即使新Value接近零，覆盖仍可发生。

教学key单位化，而执行Aq不单位化；公共教学β与实际执行β+M也不同。共享A给出受同一功能风险约束的坐标偏置，
不保证“teacher动态对应什么控制”和“自身何时调用”自动对齐。action_in的X是固定probe，浅层X未必已读到视觉；
action_out投影的velocity被丢弃，其自身β不经这次投影改变返回H/X。各target的间接信用不能一概而论。

U只把教学地址A换成独立S（初始化clone A），其余保持。U参数更多，因此T/U检验共享约束，不是严格等参数比较。
它们早晚优劣翻转与成熟T的有限优势都需解释，不能从一次T>U推导绑定普遍必要。

## 条件读写 C：让视频同时改变读和写

C基于公共β={A₀,B₀}读一遍真实native，保存H和38份X；解释器及两侧生成头fresh，与β共同接受完整FM信用。
4个独立层、宽1024、16 heads；完整N×50网格不先平均。上下文c始于token内归一化H及时间/槽位编码，
动态d始于后向差分，首帧为零。帧块因果mask允许同帧/过去帧间传播。
每层c的attention权重同时运输动态Value；d的线性/输出无bias、零保持，c仅门控。
局部差分为零时仍可收到过去的动态；整个视频差分全零才保证S=M=0。

转移(t−1,t)在t可见，两次编译均用起点`X[t−1]`寻址、到达时的c[t]/d[t]/H[t]生成Value。
令Ξ为X按列单位化，内部生成头宽256：

```text
U_A = O_A { GELU(P_X Ξ + P_z A₀Ξ + C_A[c;H]) ⊙ D_A d }
S ← S + (U_A − SΞ) Ξᵀ / 50            S_init=0
A_V = A₀ + S_end

# 片尾A_V固定，重放保存的X与c/d；不重新运行native
K = column_normalize(A_V X)            δz = S_end X
V_B = O_B { GELU(P_K K + P_δ δz + C_B[c;H]) ⊙ D_B d }
M ← M + (V_B − MK) Kᵀ / 50             M_init=0
B_V = B₀ + M_end
δy_l(q_l) = B_V,l A_V,l q_l
```

S为128×d_in，M为d_out×128；不是左乘A₀的128×128变换，也不是部署两套LoRA。
最终A_V不detach，它同时改变执行读取、M的key、覆盖与Value。S行空间受本视频X张成空间限制；
M是共享低秩读出的最终参数，算子可表示什么不等于这些算子已学出所需映射。

实际源码：[conditional_read_write.py](https://github.com/LinFyM/EMBER/blob/85919994/src/ember/operator_writer/conditional_read_write.py)。
实际实现保留Owner先前的帧块因果约束，而[上次专家](../20260930_t_architecture/EXPERT_RESPONSE.md)允许离线双向解释。
因此C是对建议中“条件读写共同学习”的实测实例，不能把它的阴性称为完整否定专家所有替代。

## 真实训练如何给予信用

每宏步4tasks×28queries；同task、跨teacher episode查询，排除自身teacher episode。
自身query有RGB/state/动作标签；这些只用于训练功能监督，不进入部署Writer的条件输入。
随机τ下`a_τ=τ ε+(1−τ)a`，完整策略预测velocity，target为`ε−a`；真实FM覆盖全部50×7及既定尾部padding。
训练得到生成LoRA的余切，再沿**同参数版本**编译和native重放VJP。共同更新β、解释器、S/M各头；source基础权重冻结。
公共β既有直接执行梯度，也有经X/H、寻址和条件生成的间接信用；不存在统一把所有target间接路径当相同的前提。

源码：[credit.py](https://github.com/LinFyM/EMBER/blob/85919994/src/ember/operator_writer/credit.py)、
[训练数据](https://github.com/LinFyM/EMBER/blob/85919994/src/ember/operator_writer/data.py)。
后来的辅助动作/关系监督、固定B头学习、独立共享RL与在线Reader是各自已登记干预，不应倒写成T/C从未端到端训练。

## 从计算可推导与仍待判断的事

动态图依赖、真实FM路径与非零梯度证明条件能影响策略并可被优化；T闭环说明某些作用已经学成。
它们不证明H表达了正确操作变量、关联地址可跨自身状态复用、训练任务足以识别可迁移规则，或小的FM降低足以改善闭环。
反过来，held失败也不能单独定位到其中某一环：标签分布、特征/地址、共享更新与闭环状态覆盖共同变化。
专家需要从实际算子和观测效应建立相互竞争、可失败的解释，并决定哪种完整方法最值得继续。
