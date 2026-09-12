# Pretrained Video Grounded Writer

2026-09-12登记，当前阶段为实现与真实profile，尚无学习或闭环结果。
Owner授权内的新候选；目标仍是正确action-hidden内容和顺序在唯一完整LoRA中的可重复执行收益，
跨同task视频、初始化、相邻checkpoint保持并迁移到固定validation8。暂不要求145/400。

## 1. 机制判断与本次投入

综合依据见findings§62–69及[历史索引](research_history.md)。Video Functional的功能信用、蒸馏去除、
VL适配、纯FM和不同读出没有获得稳定有序增量；Local Action Grounded的局部FM小幅优势没有成为主FM或闭环优势。
任务覆盖扩展也已有有界负证据，不能把更多独立任务自动指定为解法。v5.2普通FM的视频正例说明
外部预训练不是理论必要条件，也不能据此宣布现有FM普遍不可能利用过程。

本次检验另一项知识来源：冻结通用视频预训练编码器提供已有的跨帧视觉特征，再由真实任务条件和完整native H
选择与执行有关的内容。新增的是训练前可用的视频表示，而非再增大局部辅助头、改变信用路由或扩大LIBERO梯度任务集。
现有证据没有唯一定位根因；采用本候选是对可检验的竞争解释投入，不是将预训练缺失宣称为已证实根因。

若有序先验相对充分训练静态参照，在两个节点和不同正确视频均产生validation收益，支持这项知识经当前编译路径可用。
若两臂只共同提高而没有有序差额，支持一般视觉表征改善，不支持过程机制；若训练有益但validation无益，说明迁移仍未成立；
若主FM／闭环均无差额，则停止该冻结先验与当前编译方案的组合，不启动层位、窗口、rank、LR或辅助loss连续扫描。
局部损失、特征差异或非零梯度均不能改变此裁决。正负结果不自动导向再改另一个局部模块。

## 2. 输入与冻结先验

- 固定train24及既有episode角色：teacher0–15，跨episode action queries16–41，动作留出42–45，held teacher46–49。
  validation/test不产生梯度，不新增meta tasks；source、normalization及执行合同保持。
- K1，stride5并保留真实末帧。沿用当前teacher的agentview单相机RGB及180度旋转；不同时新增wrist视角。
  执行policy仍按官方双相机处理。先验只读取与native observer相同的授权RGB，不能读取actions/state/reward/ID。
- 采用冻结V-JEPA2.1 ViT-L/16 384最终norm后的dense encoder tokens，width1024；不使用predictor、action模型、规划或online memory。
  编码器保持eval，无预训练权重梯度；只新增共享可训练投影与task-conditioned attention。
- 官方代码固定为`facebookresearch/vjepa2` commit `204698b45b3712590f06245fbfba32d3be539812`。
  官方权重为`https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt`，使用`ema_encoder`。
  只构建／加载encoder，strict state load；不调用会额外加载predictor的pretrained hub入口。
  已检查上游入口存在localhost下载地址，显式本地资产加载绕过该入口，不修改服务或引入fallback。
- 图像空间预处理采用官方单view确定性eval口径：bilinear短边resize至438（int(384×256/224)）、
  center crop384、RGB除255、mean=(.485,.456,.406)/std=(.229,.224,.225)。不做随机crop、flip或时间增强。
  使用官方eval transform，不能把source的224输入归一化直接当作V-JEPA输入。

对采样后第t帧，ordered先验读取四个真实过去位置
`[max(t-3,0), max(t-2,0), max(t-1,0), t]`。开头重复第一张真实图，末尾实际短gap不补假帧。
每个窗口独立完整forward；tubelet2产生两个时间patch组，只取最后一组24×24 tokens作为M_t。
该组已经在窗口内与前组交互，不是对历史帧取平均；所有t都形成自己的M_t，完整视频无帧截断。
V-JEPA窗口内双向attention只见到当前及过去，因此不会把未来偷偷写入M_t；长程层继续过去单向。
视频时间、窗口内tubelet位置、native action horizon和flow time各司其职。

## 3. 完整数据流与学习

```text
同一完整teacher RGB + exact task language
 ├─ frozen vision/Gemma + teacher VL Meta → 每帧contextual Z/task span/KV
 ├─ 上述真实prefix + frozen Action Expert/reading Meta → R[T,50,1024]
 └─ frozen V-JEPA2.1过去四帧窗口 → M[T,576,1024]
 Z的任务token读取真实image patches → z[T,L,256]
 z查询project(M)，dense M作为Value → m[T,L,256]，g=z+m
 g的相邻端点形成query → 首次读取两端完整H → 两轮语言／过去时间交互
 E[T,L,256] → 两层Compiler → native D → 一套38-target rank16完整A/B
 冻结source读取自己当前RGB/state、加噪动作与flow time → 真实完整50-horizon主FM
```

任务条件选择预训练patch evidence；M直接进入Value并改变完整H读取query及E，不只用作attention位置或辅助标签。
新投影1024→256，query/key标准LayerNorm和现有8-head Attention，输出残差加到z；不加gate、summary、
协方差、手工motion差分或最终LoRA平均。原生38-target full A/B和两层Compiler保留，未假定它们已掌握任务过程。

仅使用跨episode主FM。执行状态条件化由真实source policy的当前观测／state及flow query实现，
其LoRA梯度通过Compiler、E、task grounding和两组teacher Meta回传；不新增独立执行辅助头或局部动作loss。
冻结source权重与视频先验始终无梯度；Writer、两组Meta、新投影／attention、Compiler/D共同fresh，
fresh optimizer/scheduler/sampler/RNG，LoRA合法identity初始化。每update四suite各一task、一K1条件及64queries，
每task权重1/4，共256queries。沿用已登记主动作索引与FM噪声／时间口径，不改变目标时间对齐。

新增先验改变表示与归纳偏置，并不从统计上排除language/task捷径。
成功标准仍由实际正向行为与最后视频controls决定，不将结构或预训练论文的成绩当作该机制已经成立。
旧局部目标退役后，与历史成绩变化不能单因归给先验；本次有序效应仅从下节共同配方的匹配比较判断。

## 4. 匹配静态参照与强度检验

主比较为`ordered`和`frame_set`，均使用同一个冻结视频编码器、video tokenizer、modality embedding及四帧形状。
frame_set对每个真实I_t输入`[I_t,I_t,I_t,I_t]`，同样取最后时间patch组；该M_t只依赖I_t。
随后native pair的before=self，无端点身份／gap路由，长程无时间PE且无因果mask，Compiler时间Key为零。
所有真实帧均进入同一集合，不固定首末帧、不携带原始t或历史窗口身份，不打散已含顺序的ordered tokens充当无序输入。
故整个静态输入路径对帧置换不变。重复真实图是静态读取合同，不对source构造zero-image或缺失prefix的forward。

两臂共同可训练参数以同一seed初始化；任务、teacher、action query、noise/time、逻辑曝光、优化器和LR匹配。
物理batch可按实测吞吐调整，不改变逻辑任务权重。差别是有序过程信息及消费，不能把两臂共同的视觉预训练贡献当作时序收益。

四帧重复输入可能弱于编码器原生image模式，不能只凭数值正常断言参照足够强。
因此预登记条件性的第三臂`frame_set_image`：**仅当主两臂在两个节点满足§5的候选资格时**，
用同一V-JEPA2.1的原生单图tokenizer/modality branch逐帧读取全部I_t，保持其余无序Writer、fresh共有初始化及完整曝光。
在相同节点补train96与validation400 correct/other。该臂与主静态臂的冻结模态处理不同，不能作纯顺序归因；
它是强静态能力参照。最终候选必须也满足相对该臂的§5资格，否则不得以重复输入参照的弱势宣称过程成功。
若主比较失败，不追加该臂挽救结果。没有预注册的无先验重训、第三个encoder或多seed搜索。

## 5. 学习窗口与裁决

最长完整teacher真实profile后，在学习分数产生前登记约一小时的总updates和两个等间隔附近节点，采用50或100的倍数。
100/200仅为既有速度参照，尚未指定本次正式步数；profile不能作为fresh学习或性能结果。
每节点train96与validation400先完成correct；主比较无正向资格则不为挽救方法扩大对照。
主比较出现两节点正向候选时补两臂相同节点other，用于完整资格；这些条件在看到分数前固定。
主动作留出在0及两个节点执行，train24每task128queries、seed20260908、原held视频池；只作定位，无checkpoint选择。

候选资格沿用上一设计：single checkpoint validation400 correct相对匹配训练的全帧无序参照，
task-cluster paired bootstrap95%下界>0；other同向，至少两个suite有净收益，相邻节点correct增量同向。
correct必须有相对冻结source的实际收益；保留所有绝对分数、source对照、per-task/suite、breadth、
R/G/L、churn、相邻及换视频success-set Jaccard，不以参照退化或单点峰值掩盖正确条件自身退化。
两节点主参照资格成立后按§4检查原生image静态参照；不能在该检查前宣称selected已冻结。
两个节点皆满足全部参照资格时选后一个；仅一个节点满足单点区间条件时，仍须另一节点增量同向及稳定证据，
不作checkpoint union／融合。若明显坏或正确条件后段退化，本窗口关闭，不用无限续训追峰值。

评测沿用task/state/video seed20260911、env/policy seed7、官方preprocess与动态长任务优先persistent queue。
validation每task50合法视频整轮各一次、两个正确视频臂逐行不同，跨checkpoint固定映射。
train96为held46–49与states32–35的已登记有限池，记录复用范围，不声称全训练面板无重复。
paired bootstrap20,000次、seed20260911、task-cluster、双侧percentile95%。

方法及single checkpoint冻结后，sealed seed20260912的correct/other/cross-suite-wrong/shuffled/reversed完成最终确认。
correct−wrong及correct−shuffled的task-cluster95%下界>0；reverse作方向支持。
shuffle/reverse先改真实帧再完整重跑V-JEPA窗口、native和Writer，不重排已有features。
这些最终controls不进入训练、checkpoint选择或架构修正。Test继续封存，不混入RL、K>1或长期145目标。

## 6. 实现所有权、性能和资源

现有`native.py/runtime.py`拥有RGB准备和冻结输入生命周期；新增一个凝聚的`video_prior.py`拥有官方encoder加载、
确定性空间预处理、窗口批处理和dense token提取，冻结模型不登记到Writer optimizer或重复保存到每个checkpoint。
`video.py`拥有learned prior read并复用完整H／Compiler／D；`supervised.py/function_credit.py`仍唯一主FM及信用重放。
同条件M只计算一次并复用于同版本重放；native Z/R继续按当前Meta参数重算，不允许缓存跨update的learned激活。
冻结prefix缓存如保留M须计入同一byte上限；默认不建立磁盘dense库或整训练池GPU缓存。

先验窗口按真实吞吐分batch，M可存CPU再按condition读取，所有帧仍参与learned attention及梯度。
93帧单相机BF16 dense M约105MiB（93×576×1024×2 bytes）；原生输入／Z/R及执行FM峰值另计，
不能凭模型参数大小声称A40必定容纳。以最长视频两次完整更新与物化profile验证实际峰值、吞吐和重放成本。

实现时退役局部动作头、局部数据读取／loss／diagnostic／配置／专属测试；历史由冻结commit与formal evidence保留。
同步training/materialization schema及资产身份；不添加旧checkpoint兼容fallback或第二trainer。
共享不变量测试继续保留，覆盖prefix依赖、静态置换、全H、唯一LoRA、冻结模型／Meta梯度和严格checkpoint身份。

主写在隔离`codex/`工作树，验证后合main并push，formal从clean pushed detached commit执行。
代码依赖和权重只存一处canonical资产根，权重HTTP长度已知为5,151,198,524 bytes；尚未下载。
下载／新run前刷新strg01独立quota及相关目录用量，预算包括下载临时文件、模型、checkpoint、LoRA库和评测输出。
先验原始checkpoint含非encoder对象的可能空间须计入，未检查内容前不删原件或承诺精简大小。
launch前两节点GPU和合计额度、NUMA／NCCL依现有合同核查，具体路径／命令／增长预算写launch contract。

## 7. 外部证据边界

[V-JEPA2.1论文](https://arxiv.org/html/2603.14482v1)与[官方实现](https://github.com/facebookresearch/vjepa2)
支持dense视频表示及image/video模态实现的选择；其机器人规划结果使用另外的action-conditioned模型和online方法，
并不证明教学视频一次编译LoRA。[JEPA-VLA](https://arxiv.org/html/2602.11832v1)也不证明本部署合同。
本方案的四帧前缀使用和任务条件化native消费是待验证设计，预训练任务成绩不代替EMBER行为证据。

## 8. 实施检查记录

2026-09-12：唯一运行面已接入冻结prior准备、统一容量缓存、learned Value读取和主FM／native重放，
training及materialization使用新schema，局部动作头／loss／片段读取／专属诊断与两份专属测试已退役。
三份配置明确ordered、重复帧静态和条件触发的image静态；正式启动仍等待profile后的节点／曝光登记。
370项共享源码测试通过；其中直接autograd检查覆盖主LoRA及两组Meta信用，Value干预覆盖全部76输出tensor，
前缀、静态置换、缓存容量、严格checkpoint与配对调度合同保留。官方代码导入和384预处理检查通过。
这些检查尚不包含下载完成后的真实encoder加载、GPU峰值／吞吐或闭环行为；不能据此宣称目标已达。

结构检查为review：active source/test合计+271/−621，净减350行，仅新增一个冻结先验owner，未复制trainer或evaluator。
保留的复杂度提示来自既有采样／selection／checkpoint合同函数，相关分支未扩展；
materialization长函数仅更新身份与元数据，695行评测测试仅调整同一checkpoint所有权fixture。
Writer目录26个直接源码文件的既有规模未增长，不为这些提示拆散凝聚的合同或进行无关重构。

### 真实profile后的正式节点登记

实际93帧teacher、完整H／76-tensor LoRA、64queries的两次单条件反向均完成，profile不保存正式checkpoint或初始化。
第二次19.5573秒，allocated39.0794GiB、reserved42.0840GiB；第二次Writer及两组Meta／prior投影梯度均非零，
source与外部encoder无梯度，实际官方EMA权重strict加载成功。保留frame chunk8、policy microbatch8、prior window batch4。
据此在学习前固定100/200两节点：每臂200updates、800个K1条件曝光、51,200主FM queries；四suite每update各一task、
每task64queries，权重1/4。profile单条件耗时给出约65分钟的最长条件上界估计，实际多卡段长另据日志报告。
两臂采用相同逻辑曝光与fresh共有初始化；每个arm的source／prior保持冻结，不继承profile参数或optimizer。
主比较先按§5执行correct，达到预登记候选才补other及条件性的image静态参照。具体GPU、环境、frozen commit与命令写launch record。
