# Native Correction Writer

> 文档角色：封存设计合同。原文中的“当前／active／下一步”仅指当时阶段；当前授权与暂停状态以[progress](../../progress.md)为准。保留原科学定义和结果，不据此恢复执行。

2026-09-13登记。Owner授权继续推导、实施与依结果调整；完整有益视频特异性goal未完成。
本项依据已通过的[原生纠正传递](../analyses/native_corrective_transfer_audit.md)开发合法前向生成器，
不把训练侧oracle的分数当作Writer结果。当前实现、资源和运行状态只看progress。

**本有界候选已完成关闭，完整裁决见§10；没有selected checkpoint。以下§1–9保留登记与实施历史，不恢复执行。**

## 1. 新证据改变了什么

裸source在state-free教学观测上，由四个真实动作纠正构造的一次38-target rank16 LoRA，
在另一批episode的t1/full10均改善实际动作输出，两个区间严格正、四suite净正。
这支持继续检验这个条件与纠正联合出口的摊销获取。它没有证明视频读取已充分、LoRA具有闭环收益，
或改善必然来自特定视频而非共享偏置；完整负条件仍在findings§88。

历史近邻限定了新增机制：

- Target-Owned及nativeD早已有自由B、纯前向与完整A/B；它们的99/76/86/68等结果不能被抹去。
  新方法的区别不能仅是另一个自由B head。精确参照为`3a6f801d:docs/action_forecast_writer_target_owned_factor_design.md`
  §3/5/8和`7a71ca14:src/ember/writer/native_factor.py`。
- EBSRI/PNBTT的真实FM LoRA叶VJP只给共享生成器训练信用，forward的B仍在native Y signed span内；
  没有把预测纠正量与同位置X的乘积定义为前向出口。原始代码为`25477c9`的
  `bank_set_shared_training.py`与`e65c6388`的`pnbtt_training.py`；相关负结果继续约束本项。
- 旧behavior authority的几何恢复不是当前完整source参数更新后的行为验证；当前新oracle补的是实际函数联系，
  不能替它补出不存在的RGB共享学习证据。

本项把主要改变收敛到**原生输入与预测纠正共同生成参数，并以实际source纠正更新提供训练目标**。
保留当前有实测后段绝对收益的视觉／运动落点监督和读取前端，不再重做一个不进入LoRA的动作头。
新监督与出口联合改变，不能在结果后声称唯一分离了两者的各自贡献。

## 2. 完整合法信息路径

```text
exact language + 单条完整双相机RGB，stride5及真实末帧
  ├─ 原vision/Gemma + VL Meta → Z / task spans / KV
  │  原Action Expert + Action Meta，public noise1729、t=1 → 完整H[T,50,1024]
  │  原冻结agentview V-JEPA过去四帧 + task条件视觉读取
  │  原两端完整H读取 + 过去单向T×L组织 → E
  │  原两块集合Compiler、38×16个target/rank queries → Q
  └─ 同组真实RGB、同language、同public noise与t=1
     裸冻结source，不安装任一Meta → 全38目标的实际输入X[T,50,d_in]

Q与每个位置的X → signed系数R；Q → 自由输出因子B
  A_raw = R X / N；A逐行单位范数，B以共享训练目标单位表达；ΔW = B A
  → 唯一38-target rank16完整LoRA → source依据自己的观测／state闭环

仅训练：同教学episode的四个真动作片段 → 固定source纠正oracle标签ΔW*
        → 实际参数更新误差；另一episode动作 → 原主FM
        OOI／存储state → 原实际注意力空间loss
```

两次原生读取都在同一次Writer调用内部，针对同一组授权RGB，只读、固定计算。
原生X分支没有teacher action/state、loss、VJP、optimizer、参数更新、额外预训练模型或运行时迭代。
Meta仍是共同训练的读取模块，其参数不进入执行；参数地址X固定在裸source坐标，不被Meta更新移动。
全部真实frames和50个H均参与读取；不选择几个帧代替完整部署输入。

X对每帧独立读取，完整捕获全部38个真实线性层输入。不能将观察Meta的X、native Y或最后一层hidden冒充这些输入。
它们可以随冻结pre-Gemma输入一起作有预算的CPU缓存；Z/KV/H/E仍按现有同版本重放合同处理。
缓存身份只属于调度，不能进入Writer。数据、source权重与已有V-JEPA均复用。

## 3. 不在部署求梯度的低秩构造

对目标l，把该视频的每个真实frame与完整horizon位置记为i，N=T×50。
现有Compiler给出q_lr；共享路由投影U、目标专属输入投影P_l和输出投影O_l定义：

```text
k_li = LN(P_l x_li)
u_lr = LN(U q_lr)
r_lri = tanh(u_lrᵀ k_li / sqrt(d))
a_raw_lrᵀ = (1/N) Σ_i r_lri x_liᵀ
a_lr = a_raw_lr / ||a_raw_lr||_2
s_l = sqrt(E_train ||ΔW*_l||_F² / (rank × d_out_l))
b_lr = s_l O_l GELU(q_lr)
ΔW_l = B_l A_l
      = (1/N) Σ_i [B_l diag(1/||a_raw_lr||) r_li] x_liᵀ.
```

A的精确零行保持零，数值实现使用标准L2 normalize的零范数保护。s_l来自全部train24×26条的624个固定label，
逐task／episode等权，只有38个全局共享常数；写入config与checkpoint buffer，部署不打开label文件，不进行task条件归一。
这是B的物理参数单位，原主FM、空间及全层原生ΔW误差的权重和度量保持原定义。

模型预测的输出纠正向量可写作c_hat_li=B_l diag(1/||a_raw_lr||) r_li/N。它与同一位置的x_li配对后形成实际参数，
不是先各自平均纠正和X再外积；后一写法会引入未获oracle支持的跨位置项。
B不受native Y span限制；A保留实际source输入的坐标。固定tanh和1/N分别定义有界系数与集合贡献。A单位行固定因子幅度规范，B共享单位依据初始profile的实际放大与
解析复核修正（§9）；它们不证明最优。理想任意因子的表示空间保留，有限共享网络的参数化／优化轨迹改变，必须fresh验证。

该分解一次输出rank16，不在部署运行SVD或任何梯度更新。O_l为零初始化，故B=0而A可非零，
输出为合法identity；第一步B可从实际FM与更新目标得到梯度，随后其它模块共同学习。
不对零矩阵的SVD求导，不增加任务内优化或第二个adapter来处理初始化。

一个有限容量事实：oracle的G_l=C_l X_l，故其右奇异向量属于X_l的行空间。
包含四个oracle采样位置的完整X，也包含rank16投影ΔW*_l的行空间；若允许任意R，则存在
R=A* X†使B* R X=B* A*。对非零行把A*归一、把其范数及正s_l的逆吸收进B*即可保留该空间。
这是因子空间的存在性，**不证明上述有限神经网络能学出该R，也不证明RGB足够**。
本监督只约束预测纠正与X形成的参数作用，不声称恢复了每个位置唯一的真实cotangent；X的零空间中仍有不可识别分量。

学习误差信号有[synthetic gradients的原始研究先例](https://proceedings.mlr.press/v70/jaderberg17a.html)。
这里仅借鉴“纠正信号可成为共享预测对象”的概念；该文的模块更新机制及DAML的部署梯度更新都不是本部署合同，
也不为本方法的效果背书。

多video只在集合Compiler后合并各自的条件×纠正贡献，使用各video的1/N与集合等权；不平均原始frames或最终LoRA。
本轮只训练与声称K=1，保留集合置换不变接口，不开展K2/4。

## 4. 训练数据与真实纠正目标

固定train24、所有meta-task梯度仍来自这24task，不添加95-task、validation/test动作、RL或新仿真任务。
Owner已允许action训练池自身RGB—动作监督；据此本次教学RGB从既有action池demo16–41取样。
同一池产生主FM query，但**每个condition必须排除其教学episode**；不能在全局池重合时丢掉逐条件跨episode检查。
教学episode与允许的query episode均按登记口径均匀，task仍四suite各一个、每task64query与权重1/4。
0–15不产生新增动作标签。42–45继续只作无梯度诊断，46–49继续是train96的未用教学池。
这是显式的新episode合同；不声称与旧0–15教学实验的逐行训练曝光相同。

每个train24/demo16–41生成一个训练目标，共624套。复用原诊断中16–19的96套state-free原件，
仅为20–41新增528套，不复制已存在因子。目标严格复用f39d594f的算子：

- 由所有有效stride5起点序列的1/5、2/5、3/5、4/5附近确定四个p；真标签为actions[p+1:p+16]。
- 裸source state-free双相机prefix，public probe1729、t=1，真实15×7 MSE对全部38个weight求导。
- 固定q24/niter2/seed20260923的rank16投影，teacher JVP给出半步线性幅度；不看query定幅，不线搜索。
- 完整ΔW*的76个因子及构造记录只作训练label；teacher loss与幅度仅核对构造，不能据它选视频或过滤差条件。

构建不执行query FM或闭环，不训练Writer，不读取42–45动作。source物理参数无梯度登记／写入。
原96套曾经同时另做query评分；本项只引用其已固定state-free参数，旧query结果不参与标签权重或筛选。
部署load/forward没有这个label参数、文件路径或loss对象；模型只收到RGB派生的合法输入。

空间监督扩展到同一train24/demo16–41，复用已验证的OOI/rigid-body/双相机及prior坐标规则；
624条标签只进原两处实际注意力loss。无需重新下载数据，也不改变mask定义来追分。

## 5. 共同学习目标与曝光

保留同一组Meta、encoder与Compiler参数共同fresh训练；只替换native因子出口。
每个condition同时获得原跨episode主FM、原空间loss及真实更新目标：

```text
L_update = Σ_l ||B_l A_l − B*_l A*_l||_F² / Σ_l ||B*_l A*_l||_F²
L = L_FM + .1 L_spatial + .1 L_update.
```

L_spatial沿旧合同为object/motion两KL的平均。更新loss使用所有目标的共同原生参数尺度，
不是逐因子cosine、任意SVD符号／rank排列的回归或逐层人为重加权。分子可通过rank16 Gram矩阵精确计算，
不物化每个dense ΔW；浮点近零截断只处理平方范数舍入。若完整目标恰为identity，分母取1并保留该条件。
identity输出的非零目标L_update=1，固定系数.1让初始项与既有source FM约.1处于同一数量级；
它是预先指定的尺度默认，不是最优性结论，不做λ扫描。

source动作归一、full50主FM、noise/t、256queries/update、AdamW lr3e-5/betas(.9,.95)、8步warmup与clip1保持。
新采样流及排除当前teacher后的query/RNG口径共同登记；ordered/frame_set须逐行匹配全部采样与动作随机数。
监督期没有RL、rollout学习、trust回滚、分段冻结或旧checkpoint继承。

先完成最长新教学池视频的真实profile，确认一次完整条件前向、主FM、更新／空间信用与Meta重放；
同时核对identity后第二步的共同梯度、source冻结、38-target shapes及推理无autograd/loss依赖。
profile不选科学参数、不继承权重。f962feb6的真实最长profile已通过，正式学习前固定100/200两个节点与每臂800条件／51,200主query；
不能看到分数后延长或缩短。

## 6. 有界行为比较与停止

ordered与frame_set均使用相同新出口、纠正标签、空间信用及fresh曝光。frame_set仍看全部真实frames、
独立重复四帧的视频prior、无teacher顺序；X分支也只按帧独立读取并作集合聚合，不给它额外的时间配对或顺序标签输入。
只有有序臂的原encoder/Compiler时序部分不同；完整X的共同置换不会改变frame_set输出。

两个已固定节点各做train96与validation400 correct，共8个面板／1,984rows。
固定source、state-video schedule、预处理和single-checkpoint规则完全沿现行合同；validation每task50teacher整轮各一次，
train96明确复用46–49/states32–35有限池。部署官方10 flow steps，不能用t1替代闭环。
当前先保留既有资格：validation有序−frame_set的task-cluster95%下界严格正、至少两个suite净正、
相邻节点增量同向、正确条件相对source有收益且后段不明显退化；完整报告per-task/suite、breadth、R/G/L、churn与相邻集合。
frame_set资格的Owner问题尚未改变合同，本项不利用新身份放宽它。

资格通过后补same-task-other、原强静态参照及跨初始化保持，再选定并冻结single checkpoint。
最后才运行sealed内容／shuffled/reversed controls；Test、held梯度、RL及checkpoint融合保持关闭。
若更新目标拟合改善而闭环／有序资格不通过，停止这个具体共享获取与原生因子组合；不把LoRA重建当成功，
不以rank/λ/seed/LR/层位扫描、冻结Meta或新增动作头继续维护它。
若主FM下降但更新目标未获获取，完整有界结果也不自动触发更大读取器；先综合两个损失与真实行为的竞争解释。

## 7. 工程、存储与生命周期

保留video encoder/Compiler、原生Meta重放、采样持久状态、checkpoint、cost-balanced调度与evaluator的既有owner。
`native_factor.py`由输入配对的因子构造替换，旧自由A/B出口退役；不保留另一默认或fallback。
原生X读取与更新label/loss分别由小的明确模块拥有。两个一次性数据入口在数据完成核验后退役，Git保留实现。
新schema/architecture/config/run身份拒绝旧checkpoint；fresh optimizer、sampler及完整rank RNG/topology照常保存。

2026-09-13 strg01现场data0为52.2/1024GiB、data1为1017.6/1024GiB。
大新输出放data0的独立canonical run根，经当前workspace的ignored runs入口引用；data1只新增源码worktree，峰值预算768MiB。
完整新增峰值先按18GiB预留，覆盖528套新纠正labels、624空间labels、两臂四完整checkpoint、8个LoRA bank、原始rows与临时输出；
实际checkpoint大小和profile峰值在正式launch前刷新。已有source、prior、数据及96个oracle标签不复制。
不因data1接近额度删除formal evidence；每个filesystem独立准入，launch仍须live复核资源。

数据构建、正式train/eval使用clean pushed detached frozen树。GPU launch前同时查两节点，
按Owner总量／节点限制选择实际提高吞吐的卡；DDP保留NUMA、NCCL_P2P_DISABLE=1及deferred NCCL。
主写在隔离codex分支完成后验证、及时集成并推送main；本轮frozen/工作树在消费完且证据保留后移除。


## 8. 数据完成与实施核验

ca87f05c clean pushed detached构建新528套完整纠正标签，旧96套只引用原始路径；合计624套已seal。
六GPU worker全部exit0，墙钟95.29–99.55秒、峰值10.225GiB；smoke的task0/demo20为10.06秒含加载，
teacher MSE .11535222→.08484499，只作算子运行检查。原始teacher速度／真实动作／JVP幅度与MSE、76个因子核验通过。
CPU空间标签624条、22,319frames，438.79秒exit0；21,695支持frames、17,552有motion mass，
独立重算真实位置／末帧／512与576布局／非支持位置清零及aggregate通过。没有query读取、环境步或Writer学习。

当前物理原件根`/data0/user/ymdai/ember_runs/native_correction_writer_20260913`，由workspace的同名analysis根引用，
约2.6GiB；两个一次性builder退役，源码在ca87f05c中保留。原件含build/launch合同、worker metadata、
registration/completion和`label_data_audit.json`。旧96套没有复制，也不把旧query评分引入新label权重。

实现复用原生读取／Meta replay、监督引擎、采样和checkpoint owner。`native_inputs.py`只负责裸source真实输入读取，
`correction_supervision.py`只拥有训练label加载及实际权重更新loss；`native_factor.py`替换旧自由A出口，
无第二decoder。新增两模块分别71/77行，避免把部署只读运算与privileged训练目标放入同一接口。
结构检查计入tests为+417/−112净305行、两个新source文件，无新增hard violation；既有materialization合同／调度复杂度
只做身份与参数接线的窄改，不借本轮拆分。两个helper虽使writer目录peer数增加，职责和部署边界清晰，保留此有界分工。

135项现有／新增相关检查通过，补充联合FM与更新loss的回放—直接autograd对照也通过；
覆盖完整identity、第二步共享梯度、dense ΔW误差／梯度与因子gauge不变、实际source输入、cache预算、
frame_set联合置换、主FM teacher排除、采样恢复及checkpoint拒绝旧身份。
这些不构成机制或性能正结论。最长新教学条件已定为task38/demo36、105frames；下一步真实profile后再固定formal准入。


## 9. 正式学习前的因子单位修正

9c14f476的最长105帧profile已完成，两次完整反传43.12／32.84秒、峰值35.77GiB，纯推理亦正常exit0。
identity的B首先获得梯度，第二步Action/VL Meta与所有读取图均收到有限信用；但L_update从1升为2910.36，
主FM为.117707→.120403，不能把图接通直接当作数值准入。

一次不执行optimizer的固定初始化读取核对发现：原始A行范数为.412–10.408，目标完整ΔW能量为.00386317。
只用纠正loss、固定Q/A并代入原第一步Adam闭式式，得到更新能量11.24792、相对误差2906.34，复现实际放大；
它支持参数输出坐标尺度这一解释，没有把有限loss上升宣称为软件bug。单独单位化A的解析误差仍162.19，故不采用这一不充分修正。
原件在analysis/profile的`scale_registration.json`、`scale_audit.py/json`；该诊断无模型学习、held读取或rollout。

在看到修正后的学习表现之前，登记并采用§3的单位行A＋共享s_l输出单位；公式由
`profile/output_units_registration.json`固定，没有扫描倍率、LR、lambda或rank。
全部624训练label得到38个s_l，范围3.91385e-8–.00305696，平均总ΔW能量.00547348；
统计文件`native_output_units.json`保留完整目标顺序、来源与原始能量统计。没有按条件筛选或按task/held重标定。
这些常数是共享模型的一部分，不是部署输入、task字典或损失逐层重加权；B仍可通过学习产生超过参照单位的幅度。

architecture更新为`native_input_corrective_factors_v2`，旧profile不成为初始化。保持同组视频、主query采样、优化器和科学目标，
从fresh identity重做两次最长条件联合反传与纯推理。两次均需finite、source/prior冻结、第二步完整共同梯度成立；
首步后L_update不得再次放大到identity的10倍以上，作为本次已识别尺度问题的运行准入，不作为科学资格或调参分数。
正式100/200节点仍须在这次真实profile之后、正式学习之前冻结；
该修正是被实际尺度证据支持的参数化变更，不是视频有益性或闭环结果。

修正后的f962feb6 profile完成exit0：两次完整条件33.264／33.090秒，峰值allocated35.770GiB／reserved37.182GiB；
主FM .11770744→.11770465、L_update 1→.99981159。第二步实际Q/K、native路由／key和Action/VL Meta均有有限梯度，
source/prior保持冻结。纯前向一次生成全38目标／76因子，未调用loss或autograd。它只通过数值与运行准入。
据此在正式学习前固定100/200、每臂800条件／51,200主query及§6的8个配对面板；microbatch8、frame_chunk4、
prior_window_batch4不变。两臂均fresh seed7，全部profile状态丢弃，不继承学习或优化器。

实际trainable参数30,659,328，四份完整参数／Adam状态约1.4GiB；1,984套完整LoRA bank按既有同shape文件约9.55GiB，
加现存2.6GiB标签、单份checkpoint临时写入和日志／rows，18GiB总新增峰值仍覆盖。data0为唯一物理输出根，
data1只保留源码及轻量入口；正式launch与后续物化／评测各按实时独立quota和GPU证据准入。

## 10. 完整有界结果与关闭裁决

09c1a0d6 clean pushed frozen完成ordered/frame_set各fresh200、每臂800教学条件／51,200主FM query，
四完整checkpoint、8个sealed bank和8个闭环面板／1,984条记录全部保留。两臂18个曝光字段、
0/100/200独立诊断10个配对字段相同，主query逐条件排除teacher episode；所有阶段及96个评测worker均exit0。

| 节点 | train ordered / frame_set | validation ordered / frame_set | validation O−F 95%CI | 净正suite |
| --- | --- | --- | --- | --- |
| 100 | 15/96 / 15/96 | 50/400 / 51/400 | [−.75,0]pp | 0 |
| 200 | 19/96 / 20/96 | 50/400 / 49/400 | [−.75,+1.5]pp | 1 |

两个task-cluster区间下界均未严格正，有序增量−1→+1不满足相邻同向、净正suite数不足2。
依据§6关闭本共享获取／原生因子组合：不追加训练、节点或rank／λ／seed／LR／scale扫描；
不触发未获资格的same-task-other、强静态／跨初始化或最终内容／shuffled/reversed controls、Test、RL。
没有selected checkpoint，完整有益视频特异性goal未完成。

有序validation两节点S/O/G/L均0/6/41/3，frame_set为0/6/41/4、0/6/42/1，breadth均3/8。
source47/400为0/5/41/1，有序各相对source保留45、新增5、丢失2、churn7、J=.86538，净率区间[0,+1.75]pp。
有序相邻50→50保留46、新增4、丢失4、churn8、J=.85185；无序51→49为45/4/6、churn10、J=.81818。
有序对无序两节点R/G/L为45/5/6与46/4/3，churn11/7、J=.80357/.86792。
保留source名义+3、Goal总数保持和低churn；没有新增成功task或有益有序增量，不能以稳定低分代替资格。

训练侧无序15→20保留全部15次成功并新增5次，净率区间[+1.042,+9.375]pp，相对source15亦有正区间；
有序15→19的相邻区间跨零，四模型train breadth均7/24。固定独立FM初始.15328534，
100有序／无序.15307564/.15304530，200为.15281900/.15280163；末25条件L_update约.90642/.90657。
这些事实保留有限学习，不证明合法RGB已获取oracle，也不能唯一识别读取、有限R/A空间、B预测／学习信用的限制。
相比上一空间监督有序200，本轮Goal41对9、Object6对59、总数50对71；teacher池和监督／出口共同改变，不能作单变量归因。

完整逐task／suite、source与相邻成功集合及全部区间见同物理run根的`OVERNIGHT_READOUT.md`、
`paired_summary.json`；原始raw、completion、四checkpoint及8个bank在`training/`。
`evidence_audit.json`核对真实stride5／末帧、validation50视频无放回、train96的46–49/states32–35有限池、
task/state与RNG配对、一次Writer调用、76因子和checkpoint身份、raw／aggregate与worker退出；全项通过。
累计评测launcher墙钟4,051.92秒，controller exit0且已确认无本研究在途Python进程。
`bounded_200_decision.json`保存最终关闭裁决，自动初步decision单独保留；launch contract登记完成证据。

§88 oracle功能前提与§90标签共同结构仍作为限定证据保留（见findings对应章节）。
下一步须先用已有证据形成可失败的竞争解释，优先区分有限原生输入空间与纠正获取，登记最小冻结诊断后再执行；
不从本次non-pass自动启动新Writer，不用参数空间存在性证明有限RGB获取。
