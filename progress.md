# EMBER progress

更新时间：2026-08-26。当前G2 formal authority：clean pushed `main@c1493a1`的`macro_00000020`；最新G3 formal
evidence为clean pushed `main@435cb4a`的50-task/98-condition bank-operator F1 pass，最新结构裁决基于全新专家对
`main@ed2883b`及完整可达历史的复核。

## 当前状态

两轮专家回复均已收到、完整阅读并固化。active design现为`docs/event_conditioned_policy_compiler_design.md`中的
**ECP Native-Factor Compiler**，核心架构、数据角色、阶段Gate、最终controls和停止条件均已明确。

专家1416行原始回复已完整保存为`docs/expert_review_20260824_native_factor.md`，逐行内容与附件一致，仅换行从CRLF标准化为LF；
第二位专家1538行原始回复已逐字保存为`docs/expert_review_20260826_bank_conditioned_native_factor.md`并与owner提供附件byte-identical；
active design明确是解释/执行层，不能替代任一原文。

全仓库orientation和两轮owner复核已完成，owner已正式许可推进。G1 task-local free-code capacity oracle与G2 Natural Program
均已通过；当前处于G3 shared compiler，v1 macro10、v2 macro5、低维dual-basis与跨视频score acquisition已形成连续non-pass及
根因证据。固定target-specific raw-dual basis不具备首版所需的compact functional capacity；进一步证据否定了把逐video解析
dual/score直接作为candidate-local shared mapping标签。新专家确认主因是stable functional target被表达成随current-bank covariance
旋转的脆弱inverse coordinates；当前活动修正为两阶段流式bank-conditioned Pass B：B0累计每视频单位质量的statistics/native anchors，
regularized solve后由B1重放同一bank做exact signed pooling。没有恢复旧Writer/realizer/GOMQ/PECS/人工process路线。

owner接受该G3修正，同时明确覆盖专家的Final初始化偏好：整套Writer完全随机初始化、从头端到端fresh联合训练必须保留为Final
正式可选项；G1--G3是组件因果验证而非Final强制课程。Final仍不预设目标LoRA，具体初始化与最小监督由matched closed-loop证据裁决。

F1隔离operator已从clean pushed detached `435cb4a`完成formal Gate：新增的`ember.ecp.bank_conditioning`只拥有B0 sufficient
statistics、截断谱query solve与B1 exact signed pooling，随后由同一模块直接供canonical compiler复用；analytic anchor、teacher
lookup和四family对照只存在于单一formal analyzer，不进入deployment模块或checkpoint。固定50 tasks/98 conditions/536
member-family rows上，q/v/action-in/action-out的task-mean median分别为`0.999871/0.999824/0.999960/0.999884`，minimum为
`0.999757/0.999544/0.999951/0.999743`；streaming-to-materialized row minimum为`0.99999988`，全部远高于预注册门。
Action Meta实际未加载、held reads为0；全仓`177 passed`。ridge在约`1e6`条件数bank上会收缩有效方向，故首版采用与G1稳定span
一致的FP64谱截断，并保留q/action-in真实output-group相对gain，未做width/LR/seed扫。该pass只证明operator capacity与数值合同，
不证明shared Program-to-anchor mapping。analytic-only analyzer只保留为formal evidence入口，旧G3 v2 compiler将在
bank-conditioned canonical实现通过对应Gate后删除而不是长期并行。

bank-conditioned canonical实现面现已接通但尚未冒充formal Gate：`SharedNativeFactorCompiler`执行B0流式单位measure统计、
Program/native anchor compatibility、可开关的FP64谱solve和B1 exact signed replay；输入候选不含output type，四类Y bank保持
各video的adj/init/goal边界。F2的off模式精确定义为`C=I`：保留centered first-moment native anchor与B1，不是固定query或第二套
deployment Writer。50-task/451-condition split预注册为329 fit、40 held-video、82 held-task；每macro用固定5次`3 target + 3 meta`
更新覆盖全部15个target-fit并轮换15/25个meta-fit，world size只做cost-balanced吞吐分片。mapping训练只解冻anchor scorer，
Program/source/scale/Action Meta均冻结；相邻checkpoint稳定性和held/fit口径已进入Gate。全仓在当前实现上为`180 passed`。

实现ownership保持单一：`shared_compiler.py`是唯一deployment Writer的B0/B1 orchestration；`bank_conditioning/anchor.py`与
`operator.py`分别拥有content mapping和数值operator；`mapping*.py`只拥有F2/F3 acquisition/evaluation/Gate，`f0.py`只拥有一次
formal prelaunch qualification，三个scripts均为薄入口。旧v1/v2 config与旧training/evaluation模块只用于读取既有formal history，
active train入口已唯一指向v3 mapping，不是fallback。若F2失败且F3通过，删除`C=I` off执行模式；F3结束后mapping-only训练面随
F4 joint owner演化或退役，不另建平行Writer。

真实吞吐profile没有用dummy占卡：同一固定六任务、同一梯度结果的F3 optimizer step，gpu01单卡为`181.21s`、峰值allocated/reserved
约`25.14/25.42GB`，计算段SM/UTL为`70--100%`；物理1/2/4/5/6五卡弹性分片为`44.96s`，约`4.03x`加速，各卡约
`19.5--25.1GB`且计算段大多`100%`。物理3因他人约`78--92%` UTL而未使用，物理0继续遵守prohibited。正式launch仍重新live检查；
每卡以真实step time、LoRA/s和持续UTL为准，显存安全余量不是必须填满的配额。

吞吐profile同样按真实稳态而非卡数解释：单worker峰值allocated约`10.1GB`但启动主要是CPU runtime装载；同一A40共驻两个长寿命
worker时实测约`37.5/46.1GB`、稳定GPU UTL `94--100%`、memory UTL约`50--73%`，两条件总墙钟相对串行提升约`66%`。
第三worker无安全显存余量；formal F1实际使用gpu01 p1--p6每卡两个长寿命worker，六卡稳态均约`37.5--37.8GB`且UTL `100%`，
12个worker全部完成，最长总时长`228.44s`。这只优化分片与吞吐，不改变50-task/98-condition authority或Gate。

G2实现面现已接通并通过最小真实检查。Program严格输出`P_lang/P_scene/P_process/rho/tau/sigma`固定schema；每条video分别运行
两条fixed antithetic native probes，再做monotonic canonical alignment与`beta_k=1/K`集合聚合。首轮真实held检查发现把多视频frames
先扁平后按全局chunk分批会使K4集合置换最大误差达`0.132`；按阶段合同改为每条video完全独立native forward后，同一检查降为
`2.38e-7`，K1为bitwise exact identity。该修正针对真实失效接口，不引入learned video reliability或task/frame route。

95-task training-only dynamic label authority已升级并在
`runs/outputs/pi05_ecp_natural_program_labels_g2_v2_cpu_20260825`封存完成：meta56/held15、target-fit19/held5共735,519 frames，
BDDL goal predicates为80个单predicate、14个双predicate、1个三predicate task；`obs[i]`按真实`state[i+1]`恢复，缺失terminal
post-action contact显式mask，`rising[0]`明确比较`states[0] -> states[1]`。全量复核发现4750个demo均无首action完成goal，故v2相对v1
rising数值不变，schema升级防止旧语义被静默复用。四个旧LIBERO-90 scene4任务的HDF5 XML使用pre-rename `salad_dressing_1`，当前BDD model使用
`new_salad_dressing_1`；只在内存中对模型identifier做已验证alias后补齐，未修改原始数据。

formal前代码复核发现并修正两项会污染解释的问题：旧rank-local negative queue与`local_index % 4` robustness使辅助loss随rank分配
不等权；现改为每task一次robustness及固定8个、target/meta fit各4个且与rank/world-size无关的content negatives。旧action target
先按video长度取整再映射action episode，而其它动态标签直接按action episode取整；现统一使用唯一action-episode query grid。修正后
architecture guard无hard violation，全仓库149 tests通过。修正后的真实K4 profile在v2 label authority上完成：macro time 23.53s、
peak allocated 18,853,217,280 bytes、84/84 trainable tensors均进入optimizer state、loss/gradient finite；run contract实际记录固定
role-balanced negatives、每task robustness、Action Meta module/parameter 0及source trainable 0。

clean pushed `main@141a110`的首轮G2 macro10 formal及meta-held15+target-held5 Gate已经完成。same-task nearer为`1.0`、probe
margin为`0.9`、one-event为`0`、median active events为`6`、K1 exact identity与K4 permutation均通过；唯一non-pass是full相对
endpoints的action/progress改善仅`0.0226%`，低于`10%` Gate，因此没有进入G3。

read-only held20机制诊断显示，full相对endpoints的`P_process/rho/tau`差异分别是same-task不同视频差异的约`2.20x/13.77x/60.00x`，
说明最早失效接口不是native动态捕获。相反，decoder action时序标准差仅`0.00060`而target为`0.33789`，清零`P_process`的静态路径
combined loss反而由`0.39574`降到`0.39088`。当前唯一修正因此移除`P_lang/P_scene`对`P_process` fusion及action/progress等时序heads的
直接加性旁路；静态scene-only head保持独立。修正后的task92 K4真实forward/backward已完成，84/84 trainable tensors均进入optimizer
state，gradient有限、peak allocated 18,851,367,936 bytes，Action Meta module/parameter 0且source trainable 0；下一步是集成后fresh
训练并复评同一held20 Gate。

该静态旁路修正已由clean pushed `main@30b98ef`完成，fresh macro10 formal与同一held20 Gate也已完成，但仍为non-pass：
same-task nearer `1.0`、K1 identity、K4 permutation与median active events `3`通过；full相对endpoints改善为`-0.0570%`，
one-event fraction `0.30`，probe margin `0.65`。无梯度temporal诊断显示event weights虽随时间变化（std `0.04398`），最终pooled
state std仅`0.00111`，action预测std仅`0.00093`而target为`0.32725`；hard-nearest、uniform event measure和mean-repeated
process消融几乎不改变loss，最早接口已推进到event/owner内容而不是query measure。

进一步在target-held5对比初始frozen Stage 0与macro10 encoder状态：Stage 0的event/owner relative RMS为`0.06069/0.36992`，
训练后的raw encoder降为`0.02601/0.22824`，fusion后owner进一步降至`0.14837`。这证明G2梯度在Program读出成立前先侵蚀了已有
Stage 0 event grounding。当前唯一机制修正因此冻结已验证的Stage 0 v3 observer，仅训练新的language/scene/process readers、alignment
与training-only heads；不同时修改owner readout、slot、width、seed、LR或Gate。run contract将实际强制Action Meta 0、source trainable 0
及native observer trainable 0。

该修正的task92真实K4 forward/backward smoke已经完成：46/46个新增Program parameter tensors进入optimizer state，loss与gradient
有限，peak allocated 10,016,652,800 bytes；实际run contract记录native observer trainable 0且training mode为eval、source trainable 0、
Action Meta module/parameter 0。checkpoint中39个encoder tensors与Stage 0 v3 authority逐tensor相等，证明optimizer/forward没有改写
frozen observer。profile输出核对后将删除，不冒充formal evidence。

clean pushed `main@db84a50`的frozen-observer formal已从fresh训练到macro10，并按原world5 topology exact-resume到macro20。macro10
held20 Gate中same-task nearer `1.0`、K1/K4、median active events `5`与one-event `0`通过，但full相对endpoints仅`+0.0051%`、
probe margin `0/40`；macro20仍为non-pass，full相对endpoints为`-0.0207%`、probe margin `0/40`、one-event `0.025`，其它上述
资格项保持通过。fit total从macro10的`1.17260`降至macro20的`0.97637`，因此内部loss下降没有转化为视频动态因果增量。

macro20无梯度诊断确认冻结修正确实保存了native结构：raw full event/owner relative RMS为`0.06252/0.36771`，fused为
`0.05590/0.26447`，full与endpoints的fused Program RMS差异仍有`0.00618`。最早失效接口现为temporal owner readout：当前共享
`Linear(128,1)`对38-owner轴严格置换不变，owner entropy为`0.99898`，action prediction temporal std仅`0.00173`，而target为
`0.32725`；从macro10继续到20没有改善该比例。两个raw antithetic branches仍不稳定，但把零均值residual只在辅助Gate路径缩小、
而不改变部署`P_process/rho/tau/sigma`，会成为Gate-only旋钮，已明确不采用。

当前隔离实现只把training-only temporal readout改为38个固定LoRA owner各自的linear query；38条query从旧共享Linear完全相同的
向量初始化，保持其余head的旧RNG序列，之后只能由owner-specific梯度分化。queries跨task共享且只读取
`P_process` content，不是task-ID route。scene head、probe、Stage 0、Program schema、数据、loss、seed/LR/slot/width与Gate均不变。
task92真实K4 profile已通过：owner-query gradient norm `0.01827`，一步后query row centered RMS为`3.23e-5`，证明共享初始化已由
owner-specific梯度分化；46个Program parameter tensors/915,554 parameters trainable，native observer/source policy/Action Meta
trainable均为0，39个observer tensors逐tensor不变，peak allocated 10,016,671,744 bytes；profile只作机制smoke，核对后删除。

clean pushed `main@407340b`的owner-specific scalar-query formal已从fresh macro10按同一world2 topology exact-resume到macro20。
macro10/macro20 held20 Gate的full相对endpoints改善分别为`+0.0158%/-0.0340%`，probe margin均为`0/40`；same-task、K1/K4、
active-event范围继续通过，所以仍未进入G3。owner queries的row-centered RMS从自身RMS的`1.58%`增长到`2.94%`，但macro20
actual与强制shared-query的held combined loss只差约`4.9e-5`，action prediction temporal std仍只有`0.00171`，target为
`0.33589`；hard-owner readout同样不改善。该结果淘汰owner-specific scalar selection作为充分修正，也排除继续到macro40只等待
query分化的解释。

进一步无梯度反事实把frozen Stage 0 raw process与其已训练action head重新配对：held action absolute loss从当前fused/current的
`0.25511`降至`0.20767`，说明Stage 0坐标与head包含可复用信息；但full相对endpoints仍只改善`0.2467%`，prediction temporal std
仅`0.00298`，所以只复用旧head不足以满足G2动态门；该反事实也没有提供直接增加owner value map的充分证据。当前最早接口是
absolute cross-episode MSE被trajectory mean解主导，未单独约束query-time action/progress residual。当前机制修正保留absolute losses并新增等权query-centered action/progress
residual MSE，不改模型、Program schema、数据、K、seed/LR或Gate。task92真实K4 profile得到`action_temporal=0.14324`、
`progress_temporal=0.08779`、owner-query gradient norm `0.01839`，39个observer tensors不变，Action Meta/observer/source trainable均为0，
peak allocated 10,016,671,744 bytes；profile已删除。

clean pushed `main@68f8705`的temporal-residual objective已从fresh训练到macro10。held20 Gate中same-task nearer `1.0`、K1/K4、
median active events `5`与one-event `0`继续通过，但full相对endpoints只改善`0.0381%`，probe margin为`0/40`，因此明确non-pass，
没有进入G3，也没有用fit loss下降或继续同一低更新数run冒充进展。

该轮结果冻结后的read-only根因分析排除了新的表示架构猜测：固定Program的full-owner temporal readout相对endpoints可改善
`15.17%`；tied-query与independent-query初始化曲线近乎相同；cross-episode监督可识别。旧trainer每macro访问38个task却只执行
一次Adam更新，所以macro10仅10次更新。同一frozen readout temporal loss从`0.311873`开始，10/60步仅为
`0.311827/0.311164`，到200/500步才降至`0.294034/0.257824`。最早接口由此定位为optimizer cadence，而不是再次增加Program
slot、width或readout结构。

当前隔离实现保持模型、数据、loss、K、seed/LR峰值和Gate不变，把每macro拆为10个role-balanced optimizer steps：常规
2 target-fit+2 meta-fit，尾部1+1并随macro轮换；scheduler与exact-resume cursor按真实optimizer step计数。单卡及gpu02 world4
真实profile均完成：world4实际聚合4个互异task、role为2+2、finite owner-query/全局gradient，46/46 Program tensors进入Adam，
四个rank checkpoint齐全；run contract记录source/observer trainable 0、Action Meta module/parameter 0。profile只作执行证据，
核对后删除，不冒充formal。

该cadence修正已由clean pushed `main@49e7769`完成，并从fresh训练至macro10/100 optimizer steps。held20 Gate仍为non-pass：
full相对endpoints改善`0.3080%`、probe margin `13/40=0.325`，低于`10%/0.75`门；same-task、K1/K4、event范围与tau资格项均通过。
相对旧10-update checkpoint，动态增量由`0.0381%`提高约`8.1x`，20个held task中17个方向改善，meta/target-held分别为
`0.2781%/0.3891%`，所以这是宽泛但幅度不足的真实信号，不是偶然峰值，也不能进入G3。

冻结macro10后仅用12个fit task（target/meta各半，K=1/2/4等量）做gradient diagnostic；held gradient为0。full与endpoints的
`P_process`差异仍有`0.07296 RMS`，但full action/progress prediction temporal std仅`0.00379/0.00160`，而target为
`0.35248/0.32500`。temporal梯度相对non-temporal在Program process参数上为`0.01031/0.10345`，在temporal decoder上为
`0.00885/0.18567`；cosine仅`-0.065/-0.071`，说明问题不是方向性强抵消，而是近常数读出使temporal梯度小约`10--21x`。
结合frozen readout在100--500步才开始明显展开的既有曲线，下一步按同一commit/world4 topology exact-resume到预注册macro20；
这是对“有效但尚未跨过学习时标”的可证伪检验。若held增量和prediction temporal std没有实质继续增长，该解释即被否定，下一修正
必须直接针对Program-to-temporal-readout的梯度饥饿/近常数结构，而不能靠继续训练或超参小扫。

macro20首次resume在训练前被exact-contract拒绝：旧v2 contract把当时`origin/main` tip记录为`authority_commit`，后续纯文档提交使该
浮动字段变化，尽管detached formal code仍是同一clean pushed `49e7769`。失败attempt没有追加invocation、metrics或checkpoint；本轮
用可逆local ref pin通过旧contract后立即恢复`origin/main=e952823`。主线窄修复现让formal contract固定记录自身detached commit，
profile仍记录当前authority tip，并以定向回归保护；它不改变模型、数据、优化或Gate。

同一run随后已成功exact-resume到macro20/200 updates并完成held20 Gate。full相对endpoints改善从macro10的`0.3080%`跃升到
`8.6878%`，probe margin由`13/40`升到`36/40`，same-task与K1/K4 invariance继续通过；fit-only prediction temporal std从
`0.00379/0.00160`升到`0.03393/0.04789`。这验证了readout学习时标，但Gate仍non-pass：median active events `1`、one-event
fraction `1.0`，且动态增量尚未严格超过`10%`。

按K分解已把最早接口定位到canonical alignment：K1在macro20仍平均`6.42` active events，而全部K2/K4训练条件均为one-event；
每条video的local presence仍约7--8个有效槽，DP却把约`6/8` path mass集中到同一canonical slot。fit-only、held-gradient 0的
counterfactual中，identity会产生5--8 events而过强；仅给现有DP首尾加canonical 0/7边界锚点就恢复为稳定3 events，同一frozen
decoder的full增量从`15.82%`略升到`16.47%`。当前隔离实现只做这个boundary修正，保留中间stay/skip、content/time score、
readout、loss、数据、K、seed/LR和Gate。全量合同测试为`155 passed`；真实macro0 K4 profile读取4条视频、102个采样帧并完成
forward/backward/optimizer step，gradient norm与owner-query gradient均finite/nonzero，active events为2、one-event为0，峰值显存约
`9.97 GB`。run contract实测Action Meta module/parameter均为0，source policy与native observer trainable parameters均为0。
clean pushed `main@c1493a1`随后从fresh训练到macro10/100 updates并按同一world4 topology exact-resume到macro20/200 updates；两段均
exit 0，metrics/invocations严格为20/2。macro10已把event Gate修复为median 2、one-event 0，但动态增量仅`0.8268%`；macro20 held20
Gate全部通过：full/endpoints action+progress loss为`0.28167/0.36207`，相对改善`22.2047%`，median active events 4、one-event 0、
probe `38/40`、same-task 1.0、K1 identity 1.0、K4 permutation 1.0（max abs `4.77e-7`）、tau violation `0.00357`。因此G2的
最早失效接口确为无边界K>1 alignment，而不是readout容量；当前冻结`macro_00000020` Program并进入G3。

G3首个共享编译器实现面已接通但尚未形成formal科学结果：frozen G2 Program现暴露每条video的canonical event assignment；Pass B
按真实native content与Program query计算正负两支softmax，输入候选严格为`(video,frame,probe,horizon)`，输出候选严格为额外含
`type={abs,adj,init,goal}`的集合。每条video先以event assignment和时间quadrature构成单位质量measure并独立chunked pooling，再由
uniform初始化、最大修正0.5的置换不变bounded beta合并K=1/2/4；K=1严格identity。实现不含task/video/frame selection参数，最终只
输出一套rank4 residual并复用唯一rank12+4 rank16 materialization。

同时已接通95-task/118-member authority、G2 checkpoint冻结加载、member相对carrier的small-core最佳rank4投影、set-valued
functional effect losses，以及target-fit successful-member occupancy的窄 evaluator capture合同。全仓`158 passed`；覆盖
chunk/non-chunk边界等价、gradient、K1 identity、K4 permutation、bounded beta及无free logits。target-fit occupancy和75-task/93-member
fit effect authority现已完成，真实GPU forward/backward/materialization也已通过；这些仍只是formal launch资格，不是G3闭环Gate结果。

该实现面随后补齐了canonical训练runtime：每个optimizer step严格一项target-fit与一项meta-fit，member identity只拥有training-only
critic/sampler；deployment forward只接收Program与native candidates。loss由single-global-member log-sum-exp、四family等权functional、
cross-episode flow、sensitivity-normalized mobile update、carrier preservation与定期same-task不同video functional consistency组成；
source、Native observer、G2 Program、carrier及experts全部冻结，只更新shared query/key/signed pooling/scales/bounded beta。由于每step
只有两个独立task，允许world size收紧为1--2，避免多GPU空转。复核发现旧实现只拟合member flow response而未实际使用预留action demos；
现已改为fit75独立action episodes上的真实PI0.5 flow loss，held actions读取为0，member flow只保留在global set-valued effect中。

target-fit verified occupancy首次clean formal capture中，step1000为`19/19`，step2000为`17/18`。唯一失败是global38 moka-pots
state4；新旧结果的adapter、init state与全部policy-noise seeds一致，但旧evaluator在step434成功，新clean evaluator到520未成功，说明
该晚完成轨迹对允许的BF16/kernel低位差异缺少裕量，而不是authority错配。修正不试seed：只用旧sealed fixed50结果预先选择每member
完成步数最短的成功state（再以state ID破同分）；global38因此改为state36、旧证据step401成功。clean重采后step2000为`18/18`，
与step1000的`19/19`共同进入critic；失败run作为formal capture evidence保留，不混入effect authority。

一项meta-fit真实GPU effect-bank smoke已完成：复用的旧meta occupancy schema为`ember_writer_occupancy_trajectory_v1`，新target schema为
`ember_pi05_occupancy_trajectory_v1`；sealer现在同时接受并逐项核对suite/task/state/success/adapter，不做宽松fallback。task1输出
4个trajectory states、1个global member、38-owner response、flow/action response及76个rank4 projection tensors，全部finite；实际对象图
Action Meta为空。约33.7MB smoke artifact核对后已删除，不冒充formal evidence。

完整effect root为75个fit tasks、93个successful members、约3.0GB，manifest状态`complete`；meta56与target-fit19角色及全部task manifest
均核对。三步真实训练profile覆盖普通K1/K4、触发same-task-other的K2/K4，以及target93共332个采样帧的长K4：三条compiler gradient
probe均finite/nonzero，真实cross-episode flow分别进入loss，唯一rank12+4 rank16被policy实际消费；Action Meta module/parameter为0，
source/Program trainable为0。最初双条件同时驻留在约44.39GiB OOM，已按最早接口改为primary主loss先backward、other对detached primary
response做轮换consistency，并对每video signed pooling做activation checkpoint；最终三步峰值分别约16.68/17.39/29.28GB，均exit0。

G3 held5 Gate执行面已补齐：同一冻结compiler checkpoint可一次性分别物化`correct_full`、`first_final`和disjoint
`same_task_other`三套评测条件，每个条件仍只有一套完整rank12+4 rank16；另有fit75 frozen-`P_lang` linear-kernel ridge到verified
rank4 effect的learned language-only control，held video/action/reward读取均为0。paired Gate report强制核对250行source、task、normalization、
tokenizer与RNG身份、三条video arm的唯一compiler checkpoint、carrier retention、breadth、Goal/Long、full相对language/endpoints及
same-task retention；shuffled/reversed未进入该Gate。

G3 v1随后从同一clean detached `5140362` fresh训练到macro10/190 updates，并完成同一五臂strict250。正式Gate仍为non-pass：
carrier/language/full/endpoints/same-task=`43/42/38/39/40`；full逐task为Spatial0 `32`、Spatial9 `2`、Object8 `4`、Goal5 `0`、
Long6 `0`，breadth`3/5`、carrier retention `32/43`、相对language/endpoints `-4/-1`。仅same-task retention
`32/38=84.2%`与adapter/checkpoint/配对authority通过；报告为
`runs/analysis/pi05_ecp_shared_compiler_g3_gate_m10fresh_5140362_4770c5e_20260826/report.json`。shuffled/reversed未使用。

macro10把“只是warmup后更新不足”证伪：total/global-member/effective-update虽从约`2.381/1.015/0.929`降至
`2.135/0.926/0.894`，190个optimizer steps却全部触发同一全局gradient clip；macro10 pre-clip median约`10.87`，scale path
gradient均值约`13.88`，而input/output query约`0.754/1.057`。macro5到10的input/output query-key相对参数变化只有约
`2.14%/1.70%`，scale组约`14.4%`，且held闭环没有跨suite或方向性跃升。因此不能再靠续到macro20或调LR/seed解释。

独立fit-only K1几何同时排除了“真实native bank不能承载member效果”：6个fit tasks、9个verified members的full-to-native
update cosine median为`0.7029`，native named/global functional retention median为`0.7855/0.7981`，9个member均有正action benefit；
full-to-mobile update cosine为`0.9978`。最早失效接口由此确定为shared Program-to-content selection的监督与优化，而不是rank4压缩、
K1 native span或多视频鲁棒性。当前活动G3修正为fit-only、K1-only的离线native-feasible teacher：对formal40 schedule实际覆盖的
50个fit tasks/451个task-video条件，把verified member投影到对应真实K1 bank，训练时以detached set-valued responsibilities监督
共享query/key产生的input/output subspace、paired update direction与独立small-core spectrum。K2/K4严格不读teacher；teacher键和
factors不进入compiler forward、checkpoint model state或deployment。selection与scale/video使用分离clip预算，首版保持原Program、
sampler、K、LR、rank、bounded beta和无confidence gate。

该G3 v2修正已在clean pushed `main@93dffc7`实现并通过全仓`168 passed`。formal teacher authority已由同一detached commit在
gpu02 p4/p5/p6封存：50个K1-covered fit tasks（meta31/target19）、451个唯一task-video、covered tasks内68个verified members、
662个teacher states、828MiB；三个worker、aggregate和master均exit0。root明确登记full fit authority 75 tasks、missing25、held reads 0、
Action Meta 0、deployment use false，且只含38-target rank4 pre-scale directions、scales与provenance。

随后单GPU真实三步profile覆盖K1/K2/K4、same-task consistency和target93长K4：两个K1条件分别精确读取1个task tensor shard并找到
`2/1`个members，所有K2/K4条件teacher reads与lookups均为0；input/output query和scale gradients全部finite/nonzero，selection与
scale/video pre-clip norms被分别记录，scale/video heads不反传shared context。所有条件均物化76 tensors的唯一rank16，K>1 beta
从uniform的最大偏差低于`1e-6`；Action Meta module/parameter、source与Program trainable均0，峰值allocated
`29,320,510,976` bytes。该profile只证明v2训练面和信息墙接通，不是G3 Gate；下一步从fresh到macro5并复评同一五臂strict250。

G3 v2随后由clean detached `2a7f760`从fresh训练至macro5/95 optimizer updates，并以同一checkpoint分别物化full、first+final和
same-task K4 banks。五臂strict250的carrier/language/full/endpoints/same-task为`43/42/41/38/37`；full逐task为Spatial0 `34`、
Spatial9 `5`、Object8 `2`、Goal5 `0`、Long6 `0`，breadth`3/5`、carrier retention`33/43`、相对language/endpoints为`-1/+3`、
same-task retention`30/41=73.2%`。只有carrier retention和全部authority检查通过，因此明确non-pass；shuffled/reversed未使用。

同一fit K1 `meta9/video40`真实bank上的固定条件审计排除了loader或梯度墙故障。step0到macro5的input/output subspace从
`0.9298/0.9292`轻微降至`0.9070/0.9083`，但paired update cosine由`0.00409`降至`0.00299`，spectrum loss由`3.7536`升至
`4.2118`。macro5梯度分解显示teacher-selection与其它selection梯度范数为`0.3235/21.8015`（约`67x`），teacher spectrum与其它
scale梯度cosine为`-0.989657`；两条显式gradient wall泄漏均为0。teacher-only反事实能同时改善selection/update/spectrum，证明分支
可优化，但v2把几乎正交且量级悬殊的selection credit以及近乎反向的scale credit放在同一步，旧functional职责实际覆盖了direct
mapping监督。当前不续训v2，也不通过seed/LR/loss系数小扫修饰结果。

隔离credit之后又完成了更早selection接口的固定bank反事实。target20的free full-native query到500步仍只有`0.4313` factor、`0.1624`
update cosine；相同真实X/Y、teacher和冻结G2 `rho` measure下的FP64 inverse-covariance dual达到input/output
`0.99628/0.99997`、update`0.99750`，retained scatter condition约`1e6`；将dual缩成最大logit`0.1`并继续使用现有
`softmax(+s)-softmax(-s)`仍为`0.99749`。最早失效接口由此定位为shared query隐式获取高条件数dual，而不是native banks或pooling表达力。

clean pushed detached `main@e7d86b0`已完成上述50-task、98-condition、四family formal probe。full-dual reference的task-mean
update cosine median/p10/min为`0.996949/0.995468/0.993884`，worst-video为`0.996487/0.994944/0.991649`，证明capture、dual、真实
X/Y回放与signed pooling合同有效。相同LOTO basis压到最大128维后，overall task-mean median/p10仅`0.288444/0.249615`，50 tasks中
没有一个达到`0.95`；family median分别为action-in `0.999983`、action-out `0.146885`、q `0.000490`、v `-0.000586`，故Gate明确
non-pass，不能扩到38 targets或把compact raw-dual code接入compiler。

同一formal artifact上的后续fit-only机制反事实把失败再分解了一层：对最差q/v/action-out条件，不再欧氏投影dual，而是在当前bank的
functional geometry中求固定basis内最小二乘，128维update cosine分别由`-0.0119/-0.0288/-0.1178`升至约
`0.684/0.507/0.805`，证明旧realizer几何错误；但仍远低于门。把同一LOTO构造延伸到实际稳定秩后，q在384维为`0.981`、
action-out在384维约`0.978`，v到其稳定span仍只有约`0.933`，说明128维raw query span本身也不够。selected-task effect-basis
反事实同样需要约384--512维，因而不恢复fixed effect-code/fit-span realizer。

直接对现有content-key做小型functional-span acquisition进一步发现：当前按width共享的64维近线性key在两条同task视频训练后，第三条
视频的三family update cosine均值只由`0.380`升到峰值约`0.521`；改为owner-specific keys改善已见拟合但没有消除未见视频缺口。
按解析曲线指定的512维owner-specific key在相同train/third-video条件上，其解析functional span从step0已达
`0.9907/0.9904`，50步为`0.9997/0.9932`，但cross-covariance最小/最大奇异值比仍约`1e-8--1e-6`。因此下一接口不是继续宽度扫，
exact bounded-score反事实确认该问题不可忽略：随机512-key必须动用约`1e7`条件数才使q/v/action-out达到约
`0.993/0.984/0.994`，在`1e6`内只有约`0.956/0.936/0.966`，且同task跨video query cosine很低甚至为负。相反，直接以真实native
X/Y作为content keys时，`1e6`截断配合不读取完整bank的固定`0.01` score scale，三family跨三video update cosine均值为
`0.99886/0.99551/0.99788`、minimum为`0.99810/0.99447/0.99703`。q的八个output groups若各自单位化会降至约
`0.967--0.985`，保留归一到`[0,1]`的相对group gain后恢复约`0.999`。因此下一canonical候选是owner-native direct content score、
非线性高容量Program query与显式bounded group gain；先做隔离的K1 mapping acquisition，不能把这些selected-condition内部值冒充
G3 Gate或shared mapping成功。

该候选随后经过同task跨video反事实复核，结论已被收紧。q/v/action-out的verified teacher effective update在三条video间仍较稳定，
mean cosine约`0.873/0.866/0.884`；冻结G2 Program更稳定，same-task flattened cosine均值约`0.9971`。但逐video minimum-norm
native dual明显旋转：直接把一个video的raw query用于另一条video时q/v几乎为零，action-out仅约`0.086`；同一条raw query对三条
video联合求解的update upper bound也只有q/v/action-out约`0.736/0.381/0.823`。保留8个event query虽把两条训练video拟合到约
`0.965/0.525/0.986`，第三条held video仍为`-0.004/0.012/0.049`；稀疏event anchor同样不迁移。candidate-local nonlinear
512D key以factor loss训练时，train/held update仅为q `0.177/0.105`、v `0.244/0.175`、action-out `0.593/0.487`。

最后以逐video FP64 analytic dual产生直接score标签，并把同一candidate-local nonlinear scorer固定训练到2000步。训练score已持续升至
q input/output `0.887/0.699`、v `0.897/0.722`、action-out `0.912/0.979`，排除了500步欠拟合解释；但held-video q为
`0.133/0.111`、v为`-0.246/-0.232`，action-out虽为`0.491/0.961`，完整held paired update仍分别只有
`-0.001/-0.003/0.114`。结合50-task/98-condition frozen-Program decoder的task-holdout与held-video dual decodability低值，当前
最早失效接口是：解析dual/score依赖整条bank的高条件数协方差，既不是稳定Program的确定标签，也不是单candidate内容可唯一决定的量。
因此不再把direct score supervision或owner-native raw key写成已确定的canonical修正；现存代码仍是已记录non-pass的G3 v2实现，尚无
新架构被保留。后续必须在两项有区别的假设间做机制裁决：一是保持one-pass合同、用跨task/video factor监督学习真正的functional
canonicalizer；二是让Pass B利用bank-global sufficient statistics/preconditioning，后者可能需要修订“query预先确定且只流式一遍”
的当前合同。该分叉先交由全新专家基于完整远程历史复核，不能用width/LR/seed或更多逐video score拟合替代。

owner明确formal训练实现不得固定world2：保持固定全局task group、role权重、loss归一化和optimizer cadence，launch时按1--6张有效GPU
弹性分片；exact-resume锁定该run首次launch topology。该要求同样适用于后续G3/G4/Final训练，不能让卡数改变科学batch定义。

owner再次明确G1--G3的分段冻结是组件因果验证，不是Final默认训练模板。组件Gate通过后，G4/Final优先直接联合优化完整Writer并使用
最小充分loss集合；只有后续机制证据要求时才采用有退出条件的warmup或分段。该建议与当前joint Writer目标一致，具体loss删留仍由
闭环和最早失效接口决定。

G1 canonical实现面已接通。首轮formal held5 free-code优化与strict250已完成：唯一rank16 candidate为`88/250`，relative recovery
`45/67=0.6716`、breadth`3/5`、高于carrier`2/5`、carrier retention`30/43`，逐task为`33/18/37/0/0`；因此Gate为
`non_pass`。全部250 paired rows、47 shards与15 workers完整，Action Meta关闭，失败是科学结果而非运行故障。

按Gate合同完成的read-only span/response分析定位到当前scalar output pooling的结构性上限：无bias q/v组合位于base weight列空间；
q只能覆盖`1024/2048`输出维。action-in带bias且可跨output types相减，精确上限为`span(column_space(W),bias)`、至多`33/1024`。
15个known-success mobile-rank4
reference整体只保留约55--56% update energy。将independent member正交投影到该上限后的paired strict250为`109/250`，逐task
`34/30/45/0/0`；原independent mobile authority为`120/250`且Goal/Long为`11/8`，投影单独抹掉了两个process-sensitive suite。

q-head修正的formal optimization与strict250已经完成：唯一rank16 candidate为`84/250`，逐task`28/21/35/0/0`、relative recovery
`0.6119`、breadth3/5、高于carrier2/5、carrier retention`24/43`，Gate non-pass。step500 generated update与known-success references
整体cosine仅约`0.06`，所以增加的q自由度没有被随机近均匀dense logits实际利用。

随后稳定native-bank投影诊断把latest member materialize为同一唯一rank16，在strict250达到`94/250`、逐task`24/24/44/1/1`；
relative recovery、breadth、Goal/Long和四task高于carrier均成立，但retention仅`22/43`，故不是Gate pass。该结果证明稳定bank内存在
process-sensitive闭环方向，并把最早失效接口推进到free-logit可达优化与retention。

latest-only解析free logits的精确step0 strict250已完成：`100/250`、逐task`24/28/45/3/0`，relative recovery`0.851`；breadth4/5、
Long 0、仅3/5高于carrier且retention`22/43`，Gate仍non-pass。step0与解析投影residual cosine为`0.952--0.964`，第一次Adam更新后即
降至`0.039--0.070`；五task 500-step formal也未在预注册内部effect/update证据上恢复step0，因此没有用held reward选择被扰动checkpoint。

set-valued formal与strict250已完成：每task按fixed50 count选择carrier/independent/latest/independent/independent，结果`111/250`、
逐task`35/29/45/2/0`，relative recovery`1.015`且retention`34/43`；breadth4/5、Long 0、仅3/5高于carrier，Gate non-pass。

最早接口现为signed-measure闭式初始化的数值稳定性：`1e-3` span threshold使scatter inverse condition number可达约`1e6`，task94
FP32实际direction cosine最低为input `0.978`、output `0.883`。只把小型初始化solve的sufficient statistics改为FP64后，真实task94
forward/materialization两侧minimum cosine均恢复到`>=0.99999988`；candidate、rank、pooling、loss、38 hooks、唯一rank16和Action Meta 0
均不变。clean pushed formal与同一strict250已经完成：`116/250`、逐task`35/34/44/3/0`、relative recovery`1.090`、
retention`35/43`；但breadth4/5、Long0且仅3/5高于carrier，Gate仍non-pass。

FP64已排除数值失真后，最早剩余结构接口是action-in whole-vector output pooling：`32 -> 1024`真实Y共享一个scalar signed measure
时必然受限于`span(column_space(W),bias)`、至多`33/1024`。paired response只把task94的action-in target恢复为known-success
independent mobile，其它37 targets保持当前native candidate，Long从`0/50`变为`1/50`。当前canonical修正因此按native input width
把action-in真实Y切成32个32D blocks，各block独立signed pooling；完整response counterfactual为`118/250`、逐task
`35/35/44/3/1`、breadth5/5、4/5高于carrier、retention`35/43`，数值上满足全部G1门，但因task94 action-in来自privileged
reference而不是native pooling，不能冒充G1 pass。候选索引、四类bank、rank、scale、唯一rank16和G1/G3边界不变。

action-in native-block修正由clean pushed `main@31f0053`完成，142项CPU回归及task94真实forward/gradient/materialization smoke通过。
从detached frozen worktree生成的五task step0 bank完成同一four-arm strict250：`114/250`、逐task`35/31/45/2/1`，relative
recovery为`71/67=1.060`，breadth5/5、四suite非零、Goal2、Long1、4/5 task高于carrier、carrier retention`35/43`，全部Gate
checks通过。54/54 shards、250/250 rows与18/18 workers完整，Action Meta module/parameter为0，adapter为唯一完整rank12+4 rank16，
没有使用shuffled/reversed。该pass只回答native X/Y banks与signed pooling形式的capacity问题；shared Program query-key attention仍由G3验证。

专家复核锁定的是远程`main@7ab5a04`。其后`6fdaeb8`只删除退役代码/人工资产并整合文档，没有新增实验结果；专家指出的当前
Stage 0实现缺口已在瘦身后的代码中复核：q/v owner仍来自layer input/residual，尚无真实38-target input/output hooks。因此该科学
裁决可直接应用于当前活动树。

## 专家裁决已固化

- ECP继续推进，名称细化为ECP Native-Factor Compiler；
- 取消neural `q_pi -> fixed effect-code realizer -> LoRA`前置链；
- privileged experts/effects只作nonparametric set-valued training critic；
- Video Program固定为owner-specific language/scene/ordered events及`rho/tau/sigma`；
- 第二pass读取38个target的真实native inputs/outputs与动态differences；
- Program通过signed pooling产生mobile rank4，与frozen rank12 carrier拼成唯一rank16；
- 当时唯一下一步是fold0 held5 task-local free-code strict250；该Gate现已通过；
- 通过后依次进行Natural Program、frozen-Program shared compiler、joint Writer、conditional outer credit和final fresh；
- validation8与完整video controls的资格门、Test8 sealed规则及ECP根本失败条件均已固定。

owner已接受专家的Action Meta门槛：只有base Writer先产生明确闭环增量，matched Action Meta又有明确净收益且无breadth/retention
损害时才加入，否则保持关闭。rank12 carrier + mobile rank4是首版有证据配置，不是永久锁死；active design保留了rank-ceiling
诊断通过后重开分配的正式分支。

owner最新取消所有人为阶段工期、固定修正次数、结构版本和训练轮数上限。Gate与失败定位仍保留；有新机制证据可继续修正，
无信息超参小扫不算推进。执行应积极复用、并行和提升吞吐，顺利时力争数天内完成整体架构实现并推进关键Gate。

owner最新进一步明确：唯一正式性能目标线是validation8 strict paired correct严格`>145/400`，且必须同时满足
相邻稳定性、breadth、四suite非零、Goal/Long贡献、same-task鲁棒性和视频因果controls，不能用偶然峰值通过。
shuffled/reversed只在最终selected checkpoint选定并冻结后测试时序特异性，不进入训练、loss、checkpoint选择、
G1--G5 Gate或架构修正依据。

## 本轮仓库整理结果

- 退役Writer、functional decoder、ECP v1--v24后继、MDCO/PECS、fixed/two-sided realizers与人工process模块已删除；
- evaluator保留source/task-expert adapter、dynamic queue、occupancy diagnostics和strict aggregation；
- canonical基础模块为source/corpus/SFT、LoRA、task experts、Stage 0、policy effects、functional loss、reward/occupancy与evaluation；
- 旧41份Markdown、87份分散证据JSON、退役配置/测试及约11.6GB可重建人工datasets/runs已清除；recovery Gate A残留
  作为历史formal evidence保留，不删除也不恢复为当前路线；
- 瘦身提交`6fdaeb8`的126项活动CPU测试、compile、脚本入口与引用审计均通过；
- orientation清理节点当时只有`main`一个worktree、无task-owned branch或GPU job；后续G1按合同使用隔离实现面与detached formal
  worktree，动态状态以本节“当前下一步”和live检查为准。

## 当前可复用资产

- 固定24/8/8 split、71-task source corpus、五fold meta manifests与target fold0 manifests；target其余folds在G4多fold验证前补齐，
  不阻塞G1；
- frozen source PI0.5 authority、rank16 LoRA topology/materialization；
- task-expert bank、independent successful members、mobile-rank4解析容量与effect calibration；
- Stage 0 v3 full-layer/horizon observer、transition matcher、event binding/segmenter；
- cross-episode video/action schedule、functional flow loss与detached LoRA gradient bridge；
- natural reward rollout、occupancy capture、BDDL progress与cost-balanced strict evaluator；
- ignored `runs/`中的唯一formal checkpoints、raw rows和aggregate。

## G1真实smoke证据

- 使用纯`load_frozen_native_observer`路径，`action_meta_lora=None`、`install_action_meta_lora=False`；实际对象图中无
  `MetaLoRAStack/MetaLoRAProjection`，policy与Stage 0 trainable列表均为空；
- 38个target均从identity LoRA wrapper的真实`base_layer`捕获X/Y；输入候选不含output type，输出候选含四类bank；
- task 90的一步真实优化中`rank_queries/event_logits/input_logits/output_logits/scale_logits`均有有限非零梯度；
- 峰值allocated约27.24GB A40显存；输出checkpoint为single complete rank16、76 tensors、carrier slots`[0,12]`与task slots`[12,16]`；
- profile输出已核对后删除，不作为formal evidence。
- q-head修正后task93一步真实profile中`output_logits`的16,793,600个元素全部获得非零梯度，其余四类free variables也全部非零；
  peak allocated为28,332,442,624 bytes，single complete rank16与纯Native/Action-Meta-off合同保持不变。
- reference-projected初始化后task93的pre-update latest loss从旧随机路径约`1.32`降至`0.817`，global-member effect为`0.107`；
  全部五类free variables仍有非零梯度，峰值28,676,537,344 bytes，真实chunk cache为521,625,600 bytes。
- action-in 32×32D修正的task94真实profile中，32个output blocks均为stable rank32，input/output minimum direction cosine仍为
  `>=0.99999988`；一步真实loss backward使全部26,208,000个`output_logits`及其余四类free variables获得有限非零梯度，
  peak allocated为29,771,734,528 bytes。纯Native loader、Action Meta module/parameter 0、38 hooks和76-tensor唯一rank16均保持。

## 当前下一步与延期漂移

1. 从当前bank-conditioned实现完成clean pushed detached F0真实K1/K4 forward/gradient/materialization与信息墙资格；
2. F0通过后先fresh运行一次`C=I`的F2消融并在预注册451 conditions上评估；若充分训练后non-pass，不以LR/seed小扫挽救；
3. 随后fresh开启B0 covariance/preconditioning进入F3，要求held-video median/p10、held/fit与相邻checkpoint稳定同时通过；通过后才
   恢复scale/functional、K2/K4和held5 strict250。逐video dual/score、task/video键、解析系数不得进入deployment；
4. G2没有引入learned video reliability；G3只在F5根据mapping证据恢复从uniform初始化的bounded K correction，并必须防止单条
   video覆盖其余videos；
5. target当前只有fold0 manifests；在G4需要至少两个train24 folds前补齐，不阻塞G2/G3；
6. 32-task fresh refit与71 meta+train24 development recipe的精确顺序延迟到Final前解决，不阻塞G2--G5。
