# Unified Policy-Native Factor Writer

状态：2026-09-05 active design。该文档取代
`docs/axial_policy_response_native_factor_writer_design.md`作为唯一active架构合同；旧文档、旧配置和Git快照只保留历史证据，
不构成fallback。

## 1. 要回答的问题

Native-Temporal Axial Writer已经证明完整PI0.5 response、真实native X/Y和signed factor readout能形成task-local增量，
但73-task shared训练在seen functional继续改善时仍让unseen task和held5闭环恶化。整模块替换诊断进一步显示：learned Process
只在task1产生正增量、在task6/79/93产生主要负增量；learned Composer单独也没有跨task形成稳定正映射；两者联合后继续按task
放大正负。当前最早失效接口因此是Process输出的learned坐标再由Composer解释，而不是full horizon、native bank、rank4或训练步数。

本设计只改变这一接口：不再先生成一个独立视频表示再交给另一个网络翻译，而让最终factor latent在每一层直接读取同一份冻结证据。

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

1. **同frame evidence attention**：每个X/Y factor token直接读取该frame的prefix、对应target的完整PI0.5 response，以及本side的
   完整native bank；X只读native input bank，Y只读native output bank，共享prefix/response；
2. **teacher-time attention**：同一target/rank/side沿真实frame顺序交互，相对frame位置只进入attention query/key；
3. **rank/side attention**：同一frame内四个rank与X/Y side协调，使低秩两侧联合决定更新而非独立漂移；
4. **标准GatedMLP**：提供逐token非线性容量。

所有子层都是pre-norm residual attention/MLP。当前首版width 128、4 heads、4个block；4不是理论常数，只是与前代2 Frame + 2
NativeTemporal block保持近似深度的首个matched点。未来扩大模型只改变width、heads或复制block，不创造新的模块类型。

## 7. 唯一readout

最后只保留三个有明确科学职责的固定操作：

1. 每条视频沿frame做一次中心化，使静态重复视频结构性地产生零mobile value；
2. X/Y factor states分别对完整raw native X/Y候选做two-branch exact online-softmax，正负分支之差直接形成rank4 A/B；
3. 对每个target的完整`B @ A`只做一次`s_ref`安全cap，再做small-core canonicalization并与冻结rank12 carrier拼接。

中心化、exact streaming reduction、cap和canonicalization不是额外learned阶段：它们分别实现视频动态必要性、完整候选归约、已知安全幅度
边界和唯一LoRA物化。不得在它们前后加入gain、temperature、whitening、transport、reconstruction solver或post-hoc calibration。

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
- task-local与shared负证据：局部可学不代表shared映射成立；因此首个shared实验保持数据、loss、初始化和训练规模matched，只裁决新接口。

## 9. 明确没有重复的旧路线

- 不同于v5/v6类多网络Writer：没有自由video latent、独立decoder网络或仅凭language/static feature生成LoRA；最终latent每层都受当前
  视频的PI0.5 response与native bank约束；
- 不同于G3 Program路线：没有固定Program tuple、event summary、relation marginal、C/D或Program-to-bank solver；
- 不同于PNBTT：没有covariance、whitening、transport和anchor链；
- 不同于旧Axial/FrameBank/NativeTemporal：没有Process先学坐标、Composer再解释坐标的边界；factor latent直接消费冻结证据；
- 不同于手工margin路线：wrong、shuffled、reversed或no-video不进入训练loss，正确视频优势必须由正确视频functional监督自发产生。

## 10. 训练与裁决

首轮使用component initialization、K1、correct cross-episode functional positive-only。为保持单一因果变量，shared split、task采样、rows、
optimizer100/200和functional panels与刚完成的73-gradient Native-Temporal run完全matched；task6/79已被消费，只作为zero-gradient重复诊断，
不再伪装成fresh checkpoint selector。每step任务数与meta/target比例只是配置选择，不是架构约束。

执行顺序：

1. synthetic合同与真实最长视频full forward/VJP/materialization smoke；
2. task1/task93各自optimizer25/50 task-local容量控制；
3. fresh 73-gradient whole-Writer optimizer100/200；
4. 两个single checkpoint分别完成held5 correct-only strict250；
5. 只有出现可信shared闭环增量后，才扩到mixed-K、fully-random Final和validation8 paired400；
6. checkpoint由correct-only闭环、breadth和相邻稳定性选择并冻结后，才运行same-task、wrong、no-video、first+final、shuffled、reversed
   因果controls。

内部functional或task-local recovery不设成人为性能门。明确负结果也不靠延长训练、seed/LR小扫或末端数学补丁挽救；应定位统一block中
最早失效的标准职责，优先修正evidence layout、attention ownership、优化目标或数据识别性。任何后继结构仍必须维持少数标准可复制模块。

最终正式目标不变：validation8 strict paired correct稳定严格大于145/400，并有相邻checkpoint、低churn、高breadth、四suite非零、
Goal/Long、same-task鲁棒性及冻结后视频因果controls共同支持。
