# 夹爪执行等价类中的随机幅度是否污染共享信用

2026-09-25。是否active只看progress；机器合同`configs/return_score_conditioning_v1/experiment_spec.json`。
主讨论01a0cd94-65da-7b22-8ca9-7ba35f454632；执行者仍为Sol 01a0cd90-ebb7-77a1-a20b-a858825d2f66。
本批仅一次冻结梯度分解，无参数更新、候选、物化、额外采集或闭环。

本批已完成并独立复核，裁决见findings§148；本文保留原合同，不再自动执行。

## 1. 为什么只做这一步

§147的P/R/NEG/FM为35/26/26/29，R未兑现事前收益；只6/32组有信用，偶/奇梯度cosine−.02686。
不能因此继续长RL，也不能把所有失败归为梯度方差或架构上限。
§83已知Panda夹爪使用sign；原局部物理Jacobian方案未过门，不恢复该方案或增加夹爪loss。
§145说明几何方向不是成功充分统计量，§146也有单个夹爪query主导代理差额的反例。
本批问的是**这次回报估计器的可去除随机分量**，不把它当两个月监督/视频利用不足的统一原因。

主讨论已在旧study的`coordination/main_gripper_score_audit.{py,json}`及512行读数中完成CPU检查：
96个非零A decision的全部480个夹爪均值，离物理零指令边界最少9.5478847个标准差；
512保存decision均无采样导致的夹爪sign改变。这只覆盖封存decision，不外推未保存的全部控制步。
加权夹爪score平方和占全部动作端score的3.7131%；条件化后该块平方和比原块约1.34e−43。
这不是Writer梯度占比、总体方差估计或闭环因果比例。必须通过完整Jacobian检验其实际影响，
不能因为夹爪输出存在随机信用就直接声称它主导了R失败。

## 2. 数学机制和反例边界

既有pipeline为`u~N(mu_phi(q,xi),Sigma)`，前五步35维，`Sigma=C(.8)⊗diag(.05²×6,.10²)`。
Source normalization使夹爪环境指令为`T(u)=(u+1)*(2+1e−6)/2−1`，零点
`tau=−1+2/(2+1e−6)`。canonical环境默认Panda/default gripper，`format_action`只通过`sign(T(u))`
累加固定speed，再clip内部夹爪状态；普通success与后续observation不读取同sign区间内的幅度。
实际依据：`pi05_eval/environment_pool.py`、安装的LIBERO `env_wrapper.py`和robosuite `panda_gripper.py`。
执行者需只读核对相同安装路径/实际配置，不新增环境或物理探针。

令夹爪五维块`u_g,mu_g`，`Sigma_g=.10² C(.8)`，观察到的sign序列为b。
原score `s_g=Sigma_g^-1(u_g−mu_g)`；替换为

`s_g^RB = E[s_g | b,mu_g] = grad_mu_g log Pr(sign(T(U_g))=b)`。

其它30个分量完全保持。时间相关系数.8，**不能逐步独立正态CDF相乘**。
令`S=diag(b), x=S(mu_g−tau)/.10, K=S C S`，则概率为五维`Phi_5(x;K)`；第j导数是
`b_j/.10 * phi(x_j) * Phi_4(x_-j−K_-j,j*x_j; K_-j,-j−K_-j,j*K_j,-j) / Phi_5(x;K)`。
使用实际联合分布，不采用直通sign梯度、监督夹爪标签、任意置零权重或改探索尺度。

保留同一成功R、LOO的A、Q/M、1/128、采集mu_old、flow噪声、真实输入和全部Writer参数。
在动作幅度只通过上述sign影响环境、未来policy不读latent幅度、原探索子流/采样独立的理想分布下，
这是原score估计的条件期望：期望相同，条件协方差移除一个半正定项。
`Cov(g_raw)=Cov(g_RB)+E[Cov(g_raw | executed equivalence class)]`是理论总体恒等式，
不能由单批的梯度范数变小宣称验证了总体方差下降；普通BF16/重建Writer Jacobian仍有限制。
有限seed、不同偶/奇初态的真实异质性、J_Sigma/J0差异和一步非线性也不会因此自动消失。

本批非零A数据远离边界，联合翻转概率的union bound≤3.31e−21，条件score各坐标绝对值≤6.39e−20。
这给出独立于CDF舍入的数值边界；不要把浮点概率输出1误写为数学上严格1。
该特殊数据可使用稳定正态边界公式加有界数值积分，不必为零A或本批未覆盖的条件建设通用分布库。
保留实际RB小数值，不能把近边界的一般情形都改成零。
Fujita/Maeda的[Clipped Action Policy Gradient](https://proceedings.mlr.press/v80/fujita18a.html)
提供按执行等价动作积分score的相关先例；它研究clipping，本批sign及五步相关块以上述推导为准。

## 3. 固定输入、计算量与唯一干预

只读复用`/data0/user/ymdai/ember_runs/return_credit_direction_causality_20260925`的
128采集/32组/512decision、dd2e00bc八个实际collection bank、父C_S00@1155和E2已保存梯度。
梯度任务及teacher/state/replica/replan/RNG身份全部沿原机器合同，不更换任务或reward权重。
96个非零A decision全部参与；其余416个贡献数学上为零，CPU完整核对后可跳过模型forward。
仍以128个episode归一，不能按96个decision或四个非零task重新归一。
不读取新expert query/action、held输入、官方Val/Test、teacher46/47或环境回报。

RAW与RB两种cotangent在同一冻结父上计算完整十步flow与38-target A/B VJP、完整Writer和三Meta VJP。
策略VJP使用实际采集bank；Writer重编译Jacobian，沿原design§6的局部锚定解释，不将偏移写入部署。
复用canonical flow/VJP所有者；可以共享前向图，也可用现有函数分别重放，最多192次十步forward、96个唯一输入。
首个非零decision作为工程pilot，计入上限；不新增smoke query/episode。
任务/初态偶奇累积RAW和RB（八task完整列出，四task零向量允许用显式零记录，不重复保存大文件）。
同一condition尽量共享Writer图以减少重复读数，但不为bitwise一致更换dtype、kernel、批大小或反复重算。
记录实际bank重放误差，保持各活动task及总RMS≤.01；再超限先停止并报告具体错误，不开启新精度矩阵。
记录RAW对E2旧g_R的相对差/夹角；若该差足以与RAW−RB修正混淆，则判未识别，不靠更多重试求理想数。
两rank TF32=True、Source冻结、完整活动梯度、同一参数布局；world2优先，也可用有吞吐依据的单卡，
不接父Adam、不创建optimizer、不改变任何权重。所有新计算来自同一clean pushed detached提交。

## 4. 输出与裁决

保存512行身份/A/Q/M/原均值/sign/RAW与RB输出cotangent，CDF及解析边界验证；
保留总梯度、偶/奇梯度和每个非零task×parity的两种完整向量，零组明确登记；
报告各层/模块norm、余项、`cos(g_raw,g_RB)`、`||g_raw−g_RB||/||g_raw||`、
被移除分量与原g的内积份额、task37贡献及偶/奇cosine的前后变化。
这些是冻结方向分解，不是能力、闭环成功或视频增量，禁止用梯度范数排名采纳模型。

预定竞争解释：
- H_amplitude：执行无关幅度显著旋转共享梯度，尤其影响旧task37占比/两半方向；结果支持估计器存在重要污染分量。
  它还不证明该分量造成旧256闭环的损失；后继是否值得独立闭环由主讨论裁决。
- H_small：RAW和RB差很小或局限于低影响块，无法解释旧R/NEG结果。停止这一修正的后继，不因理论方差性质硬开训练。
- H_other：去掉该分量后偶/奇仍不一致，保留稀疏回报、不同state和其它动作信用噪声；不自动扩采集或换task。
- H_numeric：RAW重算差与干预差同量级，或数值/身份验收失败；如实未识别，不将差异命名为学习机制。

预定义一个“接近”描述区：相对修正<.05且cosine>.99只支持本批共享方向变化较小；
超过该区不自动构成重要根因，更不自动授权下一阶段。所有数值都完整公开，不能挑某个模块作为主成功标准。
CPU积分验证只覆盖实际非零条件及少量解析1D/独立块检查；不用大蒙特卡洛验证已知公式。
该固定批次完结后主动Queue主讨论完成信号、原件及缺项，停止新增实验。

## 5. 资源、产物与停止

新study `/data0/user/ymdai/ember_runs/return_score_conditioning_20260925`。
data0新增≤2GiB（梯度按非零task/parity保留，旧decision/bank只引用），data1开发+formal代码≤768MiB；
预计.2–.4 GPU-hours，全部工程/正式/初始化/失败累计硬上限1 GPU-hour，最多同节点两卡、项目总量仍≤6。
启动前由Sol核对strg01两文件系统独立quota及共享余量、两节点live GPU身份/占用；不固定等某张卡。
launch记录一份精确命令/环境/来源/预算，退出计时覆盖CUDA初始化/加载至进程结束，未计时项如实单列，
不拿任意reserve冒充实测上界。正常运行持续等退出事件，不轮询日志/cache，不发例行心跳。
新产物至少含registration/launch/退出回执、逐decision读数、实际梯度、分析脚本/summary/completion。
无需bank、checkpoint、视频、额外geometry或full case；不能为了复用旧分析器偷偷跑旧评测矩阵。
预算/数据身份/梯度语义或资源准入失败即保存已完成部分并回报，不自动重采集/抬阈值/追加试验。
