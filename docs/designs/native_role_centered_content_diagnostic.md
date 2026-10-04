# 教学角色内容的物体参照重编码：匹配学习辨识

2026-10-04，机制§137。此处登记分析阶段的一次匹配学习，不是formal fresh、旧R续训或新的最终Writer。
实际接手与启动只看progress；固定数据规模、源策略、38-target输出和所有旧阴性保持。

## 1. 要区分的具体解释

修复后的自身首计划中，G/R的butter角色rho在原/换布局差异显著；R八对margin差主要来自已学R对自身key差的投影。
教学中后段选到butter并未保证自己的原布局选择，原生K又在最后一次RoPE之前就含位置和上下文。
同时G换布局8条虽由butter占优却全败，故本项必须检验完整获取与完成，不把读取指标当修复。

竞争解释：教学角色内容在原图位置/背景坐标中被编码，给跨布局Q学习带来不利先验；或这一坐标处理并非主要限制，
自身key、状态调用和后续动作仍不能利用所取内容。二者可同时成立。本项不声称已确诊唯一根因。
主要干预是在像素端把同一个已学角色区域移到统一局部视野，再由同一冻结source的真实图文prefix重新编码。
不是删除最后一次RoPE、改rho loss、复制ROI标签作条件或直接把teacher位置搬给机器人。

两臂为raw_key与centered_rgb。二者使用完全相同且冻结的已学RGB角色选择器、原完整C900公共/native/c/d、
同一组可学习P/D及原38处A/B生成头；只改变P所消费的角色内容p。新的控制臂必须训练，旧R不能代替它：
旧R的选择器会随学习变化，本项两臂共享同一固定选择器，学习条件不同。

## 2. 共同来源与冻结边界

父C900仍为conditional_read_write_continuation900_20261002中真实macro_00000900，训练85919994、450父a0e0248d。
source1000训练b8ea00e9；不重训、复制source或改normalization/tokenizer/数据划分。
旧角色批根为/data1/user/ymdai/ember_runs/native_role_binding_compilation_20261004。
只从其中R/checkpoint64/binding.safetensors读取Pq/Pk，作为两臂共同冻结的区域选择器；
它仅由原8个non-held task训练，没有held梯度。不得继承R64的P/D、Writer heads、optimizer或其更差控制参数。
两臂的原ConditionalTarget十组头均从同一C900初始化；公共A0/B0、四层解释器及native冻结。
P_l为18个256→256无bias矩阵，初始identity；D_l为18个1024→1024无bias矩阵，初始零。
两臂结构/初始化/可训练参数相同，新增可训练P/D共20,054,016参数；Pq/Pk的327,680参数冻结。

复用旧25条合法视频的X/H/c/d/原生K和frame indices，以及旧query_manifest与训练/B20可见标签。
25条为16条train teacher加task16已有9条teacher；禁止增加数据、teacher、held特权标签或重跑原native获取。
Pq/Pk在原固定c/K上计算完整alpha[t,j,p]，而非仅用旧保存的mean50近似恢复个别槽。
原选择器计算及teacher prefix沿旧12cd293a/ef5eeb68的实际autocast/原生混合精度合同；新编码只改变RGB视野。
不以新读回消费者a019232a的精度合同倒改历史teacher训练定义。

## 3. 从真实RGB到两个匹配的内容字段

所有t指转移到达帧，内容从origin t−1取；j=0…49完整保留，c/d和D来自到达时刻，stride5和末帧保持。
两臂共用旧R的真实公式：

    alpha[t,j,p] = softmax_p(Pq rms0(c[t,j]) · Pk rms0(K0[t−1,p]) / 16)
    raw_key: p_l[t,j] = sum_p alpha[t,j,p] rms0(K_l[t−1,p])

centered_rgb按以下固定算子生成p，所有框均由alpha产生，绝不使用真mask/位置/body名称决定框或前向内容。

1. 每帧先对已经学习产生的alpha在50槽求均值得a，再按两camera各256格分开。
   每camera内归一为二维密度。在与model224一致的归一化坐标上计算中心mu与逐轴方差var；
   16格的实际中心为−1+(2i+1)/16，i=0…15，不用包含端点的linspace。
   这只是把同一帧已形成的角色选择变成一个共有局部视野，不平均原frames/H/不同video。
2. 每轴半宽固定为min(1, max(2/16, 1.25*sqrt(3*var)+1/16))；中心夹在[-1+半宽,1−半宽]。
   sqrt(3*var)对应均匀矩形的半宽，1.25与每侧半patch余量用于保留边缘上下文；最小整框宽度相当于原图两个patch。
   这是预先固定的一种视野，不声称最优；不得扫描宽度、阈值、框、相机或按结果选不同规则。
3. 在原native实际输入的双224 RGB浮点图上，用该仿射网格双线性采样回224，align_corners=False、padding_mode=border。
   保持同一双相机顺序、颜色/数值范围、exact language、无State、padding/mask及真实prefix位置合同。
   不再rotate、不量化成uint8再resize、不填假图；框在原图内，border只承接边缘插值的半像素，不加入背景图或新图像来源。
4. 只重编码这两张真实局部图的冻结PaliGemma prefix，捕获18层RoPE前K_crop；
   不为取得prefix再跑一个没有消费者的action suffix。原完整native H/c/d仍提供真实动作知识和动态Value。
   源prefix保持原bidirectional有效token mask，不误用默认causal mask，不插第三个dummy camera或新的prompt token。
5. 对各j、各camera的原16×16 alpha密度用同一框的16格中心网格变换至16×16，沿同一双线性/border/align_corners=False规则，再在camera内归一，
   乘回该j原有camera概率质量；保留原相机选择，不因裁切面积或不可见而切换/丢弃相机。
   记结果为alpha_crop，取p_l[t,j]=sum_p alpha_crop[t,j,p] rms0(K_crop,l[t−1,p])。

alpha与框固定后，两臂p均可缓存；依赖可学习P/D或A/B头的量不得缓存。
source一次resident，分合法视频/帧chunk编码并立即归约到18×(N−1)×50×256；
不落盘每帧完整18层512-token K_crop、不复制旧X/H/c/d/K、source或dataset。
保留alpha、框、camera质量、变换前后密度及实际局部RGB作可复核条件；另报逐video框面积/全图比例，判断本次干预实际改变了多少视野。
它们不是新的GT或模型成绩，不能因框太宽、目标不可见或效果弱而选择帧/换规则。
逐行/逐video保存前保证逻辑tensor不拖带未引用批storage，并以首个合法产物大小更新本批峰值估计。
生成一次即可，失败只恢复未完成输入，不为选batch重复成功输入或追加teacher填满显存。

在理想平移且区域选择同样平移时，局部采样图可保持不变，所以重新计算的所有后续位置嵌入/前层attention见到相同局部图。
这比从已混合的最后一层K减去一个位置项更强；但实际裁切还改变尺度、背景及双视角上下文，不能把结果唯一归因平移。
透视、遮挡、错误alpha以及自身K中的位置影响均未消除；原c/d也仍依赖全图，所以只对p这一内容字段取得上述性质，
不宣称整个R或LoRA已具有平移不变性。所有25视频原样进入，不据质量筛选或替换。

## 4. 完整参数消费者、学习与配对

两臂其余计算完全一致：

    u_l[t,j] = P_l p_l[t,j]
    r_l,h[t,j] = reshape_heads(D_l rms0(d[t,j]))
    R_l,h = sum_(t,j) u_l[t,j] r_l,h[t,j]^T / ((N−1)*50)
    A_Q,l = A_C,l ; B_Q,l = B_C,l + concat_heads(R_l,h)

其余20个target仍由可共同适应的原C900头生成，最终只生成一套rank128、scale1、完整38-target LoRA。
P/D接收真实FM及自身角色余切；A/h与上游LoRA的导数保持。d=0时新增R=0，初始化D=0时两臂都是原完整C900策略。
不训练crop选择器是本分析的共同条件，用于隔离内容坐标；不冒称这已满足未来完整fresh Writer的端到端训练资格。
部署只读RGB/L和冻结选择器一次；自身policy既不裁图，也不使用teacher mask/GT、视频再读或额外adapter。

完全复用旧§132的八task12/13/14/15/17/19/43/96、teacher0/1、query demo2…29、64更新及全部query帧/noise/tau。
每步四task、各两teacher，八condition各1/8、各28query各1/28；两臂同一事件流，不扩数据或改变task权重。
fresh AdamW lr1e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、clip1，无scheduler，seed7。
每condition目标均为完整50×7主FM+.1自身L_Q；L_Q定义、tau1、前5槽及所有18层/8头沿旧合同。
冻结选择器的L_T不回传、不重新拟合，作为条件背景可读原既有数值，不能因此重复监督或增加标签。
同query tau1消费者仍复用FM epsilon。每臂14,336主FM及14,336自身角色消费者，保存实际全部训练项/有效分母。
64完整checkpoint保存Writer heads、P/D、冻结locator来源、optimizer、sampler/cursor、RNG/topology/schema。
本项是匹配的暖父点辨识，不能称C900 exact resume，也不加载旧R优化状态。

## 5. 一次终点读回

仅物化64，各25个条件一次，每condition得到一套完整76因子；同teacher跨场景复用，不按轨迹调参。
两臂各32 train闭环（8task×两teacher×init32/33）及23 task16原八格/原换位去重面板，共110条新环境行。
全部scene、env/policy RNG、noise绝对时钟、官方预处理/十flow/前5、各原任务goal/horizon保持旧§132。
只使用原butter成功谓词，orange入篮被动记录；43保留Close，不用目标物抬高或夹爪命令代替官方成功。
各臂train teacher0/init32的8条full、held23全full，其余24compact：共62full/48compact。
所有真实action chunks、物理动作、T+1 body/EEF/state/quat/gripper/goal及已有RGB层级保留。

在上述31条full的第一次真实规划上，被动记录全部10tau/18层/8head/前5槽的rho原scores及实体/image实际mass，
复用已核25自身mask，不增加首次规划、环境或precision replay。测量FP32，完整生成无外层autocast，原混合精度/inference/TF32保持。
无需重做八套R代数分解、晚期帧/层扫描或再制造101输入面板；本次问题由匹配内容干预和实际行为裁决。

另各读原16 teacher×20分离query的B20，共640次官方十步生成。全部动作/有效future/unmasked/通道误差、
固定角色scores与真实mass保留，不做新的query/teacher选择。B20统一使用原闭环生成上下文，无整体autocast；
两新臂可直接配对，旧F/G/R B20的精度上下文不同，只作带边界背景，不直接相减归因本干预。
标签仍仅来自原训练八task；无held动作/reward梯度、无Test、RL或正式400。

## 6. 判读与停止线

主要比较是centered_rgb对raw_key：同一冻结选择器和全部学习条件下，教学内容重编码是否改善原布局的正确获取与完整完成，
并能跨teacher/init保持，而非只提高已训练task的rho。报告per-task/teacher/layout、R/G/L/churn及全部不利例。
旧父/F/G/R、T/MT有的行为原件直接复用；缺少交叉条件的强参照不补造、不另重跑。

- 若重编码在多个原/交换情境减少错误对象并取得完整成功、保留训练任务及旧正例，支持该特征接口的有限修正价值；
  不从局部面板宣布完整架构、相邻稳定性、必要视频增量或超过强MT。
- 若只增加角色rho/mass或抬起、完整控制无收益，进一步降低这一角色内容修复作为主线的优先级，不能接新utility/tau loss保护它。
- 若两臂共同改善而无明确配对差，收益不能归给裁切；冻结选择器和重新学习的共同影响保留为解释。
- 若未出现预期改善或出现明显能力丢失，关闭本固定内容重编码构造；不扫描crop大小/相机/层/归一化、追加64或新head。

一次整批交回后结束，不自动接fresh、400、数据扩充或其它诊断。该停止线约束像素端角色规范化作为本失败主修复的假设，
不只关闭一个版本名；仍保留当前自身K/调用/完整控制的其它未知，不以阴性声称所有对象表示不可能。

## 7. 预算、工程与交付

唯一新root为/data1/user/ymdai/ember_runs/native_role_centered_content_20261004；新增只写data1，source/原数据/原run只读复用。
预计含工程2–4小时；新增硬限8完整GPUh、24GiB新增峰，含两臂所有失败/恢复、加载/缓存/学习/物化/110闭环/640 B20与临时工程产物。
依据前批三臂64及197/960总2.316GPUh；本批减少学习/环境量，新增25视频局部prefix编码，保守预计2–4GPUh计算。
预期新产物约10–18GiB，source/K/数据不复制；profile最多每臂两次可丢弃真实更新，随后重置共同初值/optimizer/RNG再计64。
旧角色2.316264及自身读取.045252GPUh单列，不动用旧余额或抹掉旧资源违规。
预计超过4小时或累计6GPUh仍不能在8内完成时，按实际剩余量报告；不静默缩面板/改框或扩大预算。

创建新root/缓存/启动前，执行者查strg01独立data1 quota、相关实占/共享容量及含所有旧引用资产的新峰值余量。
每次launch现场同时查两GPU节点并沿8/6总卡、单节点6和真实A40吞吐安排；独立两臂及已就绪物化/评测可并行。
缓存只保留实际需要p/框/alpha/局部RGB，不长驻未消费的大K；采用合法帧/查询packing，不重复成功输入作profile。
资产路由显式指向canonical data1并强制离线，避免LIBERO默认下载；本项标签全复用，没有新的环境标签生成阶段。

既有实验session独占工程及canonical tracked/Git，按code-architecture-gate复用实际owner；main不重复工程审查/测试。
原binding/self-call/precision入口继续封闭，新临时入口只服务本批，结束后退役并封闭；原C900正式合同仍支持。
验证实际crop坐标/两camera/RoPE前K、完整因子/余切、冻结边界、首次合法storage及真实消费者后push并clean detached执行。
已定位工程问题在原合同内修复/new push/freeze，原件/失败/费用保留；科学语义/预算边界才交main裁决。
运行只用一次持续退出等待，不轮询、自Queue或阶段性回报。完整产物/退出/费用/Git集成/清理后一次回报并交回窗口。
