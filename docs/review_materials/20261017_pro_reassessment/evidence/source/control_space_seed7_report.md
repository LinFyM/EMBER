# Source/r8参考末四项与E/D144整批交付

36参考已全部冻结、独立report齐，E/D144及唯一36×16函数保留读出真实结束并验收。参考显著强于source；本窗E/D重构接近source，未保存强参考能力。

|范围|source|参考|E/D144|相对source保持/得/失|参考新得能力保存|
|---|---:|---:|---:|---|---|
|全部36|218/576|502/576|216/576|206/10/12|10/288|
|target24|48/384|315/384|45/384|39/6/9|6/270|
|source-seen support12|170/192|187/192|171/192|167/4/3|4/18|
|去除五identity的31|138/496|422/496|136/496|126/10/12|10/288|

相对参考全体R213/G3/L289、churn292；相对source R206/G10/L12、churn22。4/36 task净改善；task21由14降10，不能只报小幅总净差。完整每task/suite和成功集合在results.json；原selection初态0..15/root202610151，含参考选点面板偏差，无held、视频因果或相邻学习稳定资格。

五identity [42,55,64,96,101]的80逻辑行复用真实source，属于恒等输出。另31套均真实生成、B非零并执行496请求；它们参数有变化，控制大体仍停留source水平。B norm只作定位，不构成学习成功。

## 参考库与科学边界

§9仅5/25/38从末1920各接2160，43接2160/2400；selected按原规则分别1680/1680/1680/1920、最高点后两完整节点。独立source→selected为1→18、0→11、1→3、8→17 /18，38失去source state46。原32 selected及983事务原件逐字段保持。全部36独立report source248→545/648，R241/G304/L7；target60→337/432，support188→208/216。五identity90逻辑复用；31非identity selected实际成功RGB455/558、失败103，全部selected逻辑成功545/648。
完整bank：runs/source_policy_space_compiler_20261015/reference_seed7/analysis/engineering_final_four_20261016/reference_bank.json；全部候选/相邻RGL/selection/report/真实producer在同目录report/results及reference原case/NPZ。新第三窗144 raw report视频header通过（selected49成功/23失败，source10成功/62失败），成功和失败实际像素亦读核。

E只在标签侧编码完整参考因子，D只读64维码；2层E/6层D、38-target/r8、76单位、两个live anchor、source冻结、唯一真实native10前5×7动作MSE。144有效更新/每task16访问不是充分拟合。旧native负证据继续有效；真实图接通不会把本窗阴性变成软件bug，也不能推出全部低维方法或自身交互学习不可能。没有新q/Reader/RL/held/Test或自动288等续训。

## 实际学习与消费者

4032唯一观测/noise query、8064逻辑reference/student配对均核对。原生flow执行行计230720=8064×10 + (8064−560)×20，含实际student forward及checkpoint replay；source identity合法复用560学生响应，半权不重分配。实际prefix queries4508含物理分片重复，不增加逻辑监督query。36/72/108/144四CP完整E/D/220 active Adam状态/constant scheduler/S/identity/bank/流/各rank RNG及topology已验收。36为world6，72起world4，完整36边界迁移，不称bitwise exact。
平均MSE=0.045855132、selected=0.048578894、other=0.043131369；同query忽略代码下界均值0.002869352，均值比15.981，0/144更新低于下界。高于下界不能反推完全忽略代码。首末MSE用不同query/noise，不能当配对学习曲线。首步E梯度0/decoder head非零，后续E实际非零，训练未被detach。

首次80878 torchrun子--start被父解析器误识别，0更新/0环境；2.538855s×6卡计0.004231426GPUh并保留原件。纯launcher分隔符修复后从eb78完成全部144。CPU物化误把新造FP32模板当native mixed-dtype identity，并采用过严bit guard；实际CP中心含36 BF16模板因子/40 FP32（含38零B），与新造模板最大差6.1035e-5。仅修复消费者读取保存identity，并对已登记五源端点使用E(0)=0/D(0)=identity的常量，保留双锚真实训练图，不关闭高效kernel。24已生成nonidentity文件mtime/大小保持；144、S、Adam、RNG和流未重算。只补12物化与末点读出，0训练/有效行重跑。diagnosis与两失败原件在inputs/owner_attempts/jobs。

## 吞吐、资源、时间与费用

首6有用更新6rank/7、4rank/14、2rank/28整步中位5.732/2.943/5.409s；峰allocated11.194/13.175/17.305GiB，0 OOM。14合并同query双参考将prefix读从56减28；28仍有余量但全局吞吐更差83.8%，选择4卡/14。整144更新中位2.849687s；两个成功train job含加载总510.854s约8.51min，GPUh0.643399942。

readout按node预算为其它就绪GPU保留8/4槽后再给当前16，继承同一long-first队列/惰性pool/idle和tail完整close；31个物理shard为16槽4个、8槽22个、4槽5个。它们task/cost不同，非配对microbenchmark。真实端到端plans/s中位分别7.754/6.825/4.754，native9.459/8.464/6.157。node slot cap60，实际最大预留52；没有把两主机内存合池。
单worker采样PSS峰56.675GiB/18process/52threads；host MemAvailable最低276.371GiB。异步worker PSS不称同时node总峰；训练CPU峰未单独采样。8完整close回执无残留，全部记录readout PID消失，最终双节点本用户CUDA0、control module PID0，foreign未动。

§9 owner19:46:04.560640→20:10:19.443483Z，24.248min；新1200更新/33600query/5 CP，304 starts，81053 control+3040 dummy=84093步，0失败/未知/额外尝试。reference ROOT累计46560更新/1303680query/266 CP、10870 starts、2472414总步，原16额外未用。原首次09:05:48.038不重置。

§10接收19:51:01.953Z、首次CPU准备19:52:32.143Z、首启动20:16:25.735154Z；恢复20:19:13.859893Z、readout-only恢复20:40:10.102290Z→20:56:20.011405Z exit0。接收到最终退出65.301min；读出恢复owner16.165min，低于20–40min粗估。两工程失败/恢复时间与费用全部保留，未增加科学量。
§10实际496 starts、142481 control+4960 dummy=147441步，0环境失败/未知，独立16额外未用。GPUh精确2.001268105308（含首次0.004231426失败）；其中readout1.353636738。§9新增2.371412874791，本两窗合计4.372680980100。reference ROOT累计区间[52.014113090,52.121858595]保留旧六loader未知区间；加本CS ROOT为[54.015381195,54.123126700]。旧N16及其它依赖费用独立，不作为上述carry-in；无累计GPUh硬限。
§10退出ROOT实耗0.617GiB，reference约10.404GiB；未声称持续采得精确磁盘峰。预估新增12GiB内，未复制source/数据。storage-node data1 used1747619008KiB/soft2147483648KiB，data0独立未新增产物；shared容量另核。

## 代码与整批交付

原首8 selected权重训练来源44c0ee3a、新28中23个nonzero selected来源0f8e0023，五identity为raw source；§9仅新5个末端CP恢复/新selection及首次report读取代码355396b0，四项最终selected权重仍来自原0f8。E/D144训练eb78edd7；剩余CPU物化/全部新readout39d21bf0，三者均clean pushed detached。初次参数失败80878保留。最终canonical main身份见Git_delivery.json。main科学合同e254b6a7已集成为c4ca1220并push；报告docs/designs/source_policy_space_compiler_20261015.md§10及既有analysis稿，不是第二active设计。

source净增893行/10新文件，guard无hard，职责与复用见progress。三个task-owned clean dev worktree/branch在完整集成确认后已清理；child两提交均gitcherry等价已在main，其它main-owned docs worktree未动，实际formal冻结保留。实验执行记录/Git窗口在最终commit/push和一次整批main送达后释放，后续科学判断归主讨论。

参照analysis/results.json、acceptance.json、training_acceptance.json、functional_readout_details.json、resource_exit.json、reference_bank及全部raw case/CP/job/owner退出原件。只此一次整批回报，无阶段Queue/ACK/第二waiter。

旧固定MT128/67原件完整保留，未新增MT请求。144唯一读出结束后的相同八任务既有配对子集为E/D18/128对MT67，R15/G3/L52（case RNG/proprio逐task核配）；不是36任务或held的MT对照。原首8 selected105/128仍保留。详见existing_MT128_pairing.json。
