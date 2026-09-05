# Unified Policy-Native Factor Writer

状态：2026-09-05 active design。该文档取代
`docs/axial_policy_response_native_factor_writer_design.md`作为唯一active架构合同；旧文档、旧配置和Git快照只保留历史证据，
不构成fallback。

## 1. 要回答的问题

Native-Temporal Axial Writer已经证明完整PI0.5 response、真实native X/Y和signed factor readout能形成task-local增量，
但73-task shared训练在seen functional继续改善时仍让unseen task和held5闭环恶化。整模块替换诊断进一步显示：learned Process
只在task1产生正增量、在task6/79/93产生主要负增量；learned Composer单独也没有跨task形成稳定正映射；两者联合后继续按task
放大正负。最早失效接口因此是Process输出的learned坐标再由Composer解释，而不是full horizon、native bank、rank4或训练步数。

本设计只改变这一接口：不再先生成一个独立视频表示再交给另一个网络翻译，而让最终factor latent在每一层直接读取冻结证据。首版
把policy evidence和native bank放入同一个softmax，task1/task93容量控制与attention-mass诊断发现其概率质量受token cardinality支配；
第二版因此把policy与side-native拆成并行cross-attention。第三版恢复`base(context,current-bank)+delta(dynamic)`共同定位，并取得明确
task-local正控，但73-task shared的held5在optimizer100/200只有`35/31`，低于carrier43，且Goal/Long均为0。

进一步的whole-module和block-sublayer替换诊断排除了几个诱人的误判：learned evidence projection在两个true-held task上都是正增量；
近rank1也不是根因，因为成功的task-local解同样收敛到近rank1。共同失败集中在重复factor block学成seen-task expert，而不是某一个
temporal、factor、MLP或signed子层单独损坏。信息流审计同时发现，当前policy read把约15--24个language token、256个patch token和
400个response token放进同一个softmax，language多数层只得到约2.2%的概率质量，接近纯token-count占比。active v4因此只修正这一个
task-grounding接口：同一个标准policy cross-attention分别读取language、patch和完整response，各自softmax后与side-native read相加。
它不增加参数、阶段、module type、gate或手工校正器。

## 2. 一句话结构

```text
exact language + ordered teacher frames
                  |
                  v
       frozen PI0.5 full response + native X/Y
                  |
                  v
  [Unified Policy-Native Factor Block] x depth
                  |
                  v
       direct signed raw-X/Y pooling
                  |
                  v
     rank12 carrier + rank4 mobile = one rank16 LoRA
```

learned主干只有一种可复制block。增加深度就是复制该block，不增加新的summary、gate、solver或校准阶段。

## 3. 输入、轴和信息墙

每条视频先独立、保序处理。部署输入只有exact task language和action-hidden正确视频；Writer不读取teacher action、state/proprio、
reward、terminal、task ID、filename、object pose或policy outcome。

冻结PI0.5对每个teacher frame产生：

- 原生image patch与language prefix token；
- 两个固定probe下完整的19 layer x 50 Action Expert horizon状态；
- 相邻layer residual、probe noise和flow velocity通道；
- 与38个LoRA target逐层对应的真实native input X与output Y候选。

teacher-frame time、50-step action horizon、flow time、layer depth、probe、target、rank和X/Y side始终是不同轴。禁止coarse、horizon
mean、horizon sampling或其它等价抹平。

## 4. 仅作tokenization的入口

入口只做线性投影、类型/位置embedding、mask和LayerNorm，不生成learned video code：

1. prefix tokenizer保留全部patch和有效language token；
2. response tokenizer把state、layer residual、noise、velocity各分成probe-even与probe-odd，共8种通道；
3. 每个通道保留完整50 horizon，并带owner、family、layer、horizon和channel标识；
4. native X/Y只投影成attention key；未经替换的raw X/Y同时保留到最终value readout。

这些投影可复用G2已经验证的Stage0 native projection与结构embedding，但不冻结为不可学习接口。fully-random候选使用完全相同拓扑。

## 5. 显式factor latent

主状态为每条视频的
`frame x target x residual-rank(4) x factor-side(X/Y) x width`。
初始状态只由共享rank、factor side、target owner和family embedding组成；task-local正控可额外使用训练期task query，但shared/deployment
图没有task ID或task query。

显式X/Y side从主干第一层开始存在，避免在末端才从一个共同向量线性分叉两侧。

## 6. 唯一可复制block

每个`UnifiedPolicyNativeFactorBlock`重复完全相同的四个标准子层：

1. **同frame并行evidence read**：同一个X/Y factor query用同一套policy-attention权重分别读取该frame的exact-language token、全部
   image patch和对应target完整PI0.5 response；三者各自softmax。另一个标准cross-attention只看本side完整native bank（X只读native
   input，Y只读native output）。四个读出直接相加后进入同一个residual state；没有串行handoff、gate、标量权重、token-count修正或
   新增参数；
2. **teacher-time attention**：同一target/rank/side沿真实frame顺序交互，相对frame位置只进入attention query/key；
3. **rank/side attention**：同一frame内四个rank与X/Y side协调，使低秩两侧联合决定更新而非独立漂移；
4. **标准GatedMLP**：提供逐token非线性容量。

所有子层都是pre-norm residual attention/MLP。当前width 128、4 heads、4个block；4不是理论常数，只是与前代2 Frame + 2
NativeTemporal block保持近似深度的首个matched点。未来扩大模型只改变width、heads或复制block，不创造新的模块类型。

## 7. 唯一readout

末端只保留三项职责：

1. 每条视频把最终factor state分为共同context `C=mean_t z_t`与frame-relative innovation `D_t=z_t-C`；
2. X/Y各用一个linear signed-query head形成`q+/-_t=b(C)+delta+/-(D_t)`，再对完整raw native X/Y候选做two-branch exact
   online-softmax，正负分支之差直接形成rank4 A/B；共同base在两分支完全相同，故`D=0`时两branch查询和pooling严格相同，完整mobile为0；
3. 对每个target的完整`B @ A`只做一次`s_ref`安全cap，再做small-core canonicalization并与冻结rank12 carrier拼接。

`C/D`不是新的representation网络或串联阶段，只是同一个最终state的均值与残差两种视图；linear head直接消费二者。exact streaming
reduction、cap和canonicalization分别实现完整候选归约、已知安全幅度边界和唯一LoRA物化。不得在它们前后加入gain、temperature、
whitening、transport、reconstruction solver或post-hoc calibration。

多条视频时，每条视频独立编码；只在最终候选measure中以等video、等frame base mass作置换不变集合聚合。不得平均frames、raw features
或最终LoRA。

## 8. 吸收的正证据

- G1：真实native X/Y、two-branch signed pooling、rank4 task-local mobile和rank12+4物化继续原样保留；
- G2：冻结PI0.5内部响应包含有用视频动态，因此保留真实prefix、两个probe、完整layer和50-horizon，而不是重新训练一个独立视频编码器；
- Policy-Response Writer：exact language负责grounding，Action Expert response承担视频时序路径；correct cross-episode functional把Writer
  直接连接到最终policy行为；
- Frame-Aligned与Frame-Bank：candidate所属frame必须在方向形成前参与，故每层都做same-frame native read，而不是把global query广播到
  全视频；
- Native-Temporal：X/Y必须从factor trunk入口显式分侧，并同时建模真实teacher time和rank/side；
- task-local与shared负证据：局部可学不代表shared映射成立；learned evidence本身可以跨task泛化，真正需要修复的是共享block如何从
  exact language建立task-grounded读取，而不是冻结上游或添加末端数学补丁。

## 9. 明确没有重复的旧路线

- 不同于v5/v6类多网络Writer：没有自由video latent、独立decoder网络或仅凭language/static feature生成LoRA；最终latent每层都受当前
  视频的PI0.5 response与native bank约束；
- 不同于G3 Program路线：没有固定Program tuple、event summary、relation marginal或Program-to-bank solver；
- 不同于PNBTT：没有covariance、whitening、transport和anchor链；
- 不同于旧Axial/FrameBank/NativeTemporal：没有Process先学坐标、Composer再解释坐标的边界；factor latent直接消费冻结证据；
- 不同于手工margin路线：wrong、shuffled、reversed或no-video不进入训练loss，正确视频优势必须由正确视频functional监督自发产生。

## 10. 训练与裁决

首轮使用component initialization、K1、correct cross-episode functional positive-only。combined-softmax v1、innovation-only
parallel-read v2和common-base v3均已完成并封存。v3的task1/task93 optimizer25/50正控证明完整图有容量，但73-task shared与两个相邻
held5闭环共同否定了现有task grounding。active v4保持evidence、native bank、block depth、readout、loss、rank和训练任务不变，只把
language、patch、response从一个policy softmax拆成同权重的三个独立标准attention调用。task6/79已被消费，只作为zero-gradient重复诊断，
不再伪装成fresh checkpoint selector。每step任务数与meta/target比例只是本次短实验的配置选择，不是架构约束。

执行顺序：

1. synthetic合同与真实最长视频full forward/VJP/materialization smoke；
2. task1/task93各自optimizer25/50 task-local容量控制；
3. fresh 73-gradient whole-Writer只先跑optimizer25/50短资格，不在架构尚未显示shared信号前投入长跑；
4. 只有内部held functional与相邻学习轨迹支持继续时，才物化single checkpoint并做held5 correct-only strict250；
5. 只有出现可信shared闭环增量后，才扩训练长度、mixed-K、fully-random Final和validation8 paired400；
6. checkpoint由correct-only闭环、breadth和相邻稳定性选择并冻结后，才运行same-task、wrong、no-video、first+final、shuffled、reversed
   因果controls。

内部functional或task-local recovery不设成人为性能门。明确负结果也不靠延长训练、seed/LR小扫或末端数学补丁挽救；应定位统一block中
最早失效的标准职责，优先修正evidence layout、attention ownership、优化目标或数据识别性。任何后继结构仍必须维持少数标准可复制模块。

最终正式目标不变：validation8 strict paired correct稳定严格大于145/400，并有相邻checkpoint、低churn、高breadth、四suite非零、
Goal/Long、same-task鲁棒性及冻结后视频因果controls共同支持。
