# K4 Energy-Preserving Policy-Layer Trace M2P Writer

状态：2026-08-06设计authority已封存，clean`22234c4`已原位实现并push；
live A40 fresh/exact-resume profile已通过，config seal=`3b7eb4a`，等待formal fresh0→200。本文覆盖
`action_forecast_writer_k4_layer_trace_m2p_design.md`的活动地位；旧方法由Git和sealed artifacts保存。

## 1. 决策

下一轮只修复已经实证的最早表示接口：

> exact task language + four action-hidden same-task videos
> -> frozen PI05 all-layer video innovations
> -> energy-preserving temporal spectrum
> -> existing layer/parameter-axis M2P
> -> one complete rank-16 LoRA.

保留K4、20个policy groups、16项DCT-II、68 slots、1024 width、四个axis blocks、full24 B20、
source freeze、38 targets与single-LoRA部署合同。fresh训练，不加载上一版checkpoint。

## 2. 上一版已经排除什么

macro100五臂为`correct/same/wrong/shuffled/reversed=99/92/57/94/105`。correct相对wrong的
paired gained/lost=`61/19,p=2.73e-6`，same-task alternate set与correct同档；因此视频中的
task identity确实进入了LoRA和closed loop，不能退回language-only或删除视频。

8-task内部probe进一步显示：

- same/wrong的normalized trace差异中位约`.995/1.319`，到reader为`.135/.547`，到effective
  BA为`.167/.715`，到fixed action为`.040/.244`；video→LoRA→action链路存在；
- LoRA norm中位`48.28`，stable rank中位`1.34`，top singular energy中位`.836`，q/v B-column
  cosine约`.279/.306`；它比旧K4的near-rank1几何更丰富且有足够policy leverage；
- reader和axis memory仍覆盖约12--14个effective policy groups，没有layer/group坍缩；
- 最后50步reader/axis task-gradient retention中位`.04666/.04371`，pair cosine
  `.00418/.00225`，shared credit cancellation仍存在，但它不是当前最早故障。

## 3. 新发现的最早故障：逐频率单位化破坏能量语义

冻结PI05原始DCT trace并不是16个等强语义分量。8个validation tasks的真实分布为：

- DC能量占比中位`.95664`；
- 高频8项总能量占比中位仅`.003592`；
- effective frequencies中位`1.0923`；
- 最弱/最强频率能量比中位仅`.0001868`。

上一版却对每个`group × frequency`向量独立做`L2 normalize`。所有非零频率因此都变成单位
norm：原始仅约`.36%`能量的高频8项被提升到约一半token energy，约放大140倍；最弱频率相对
最强频率的幅度被提升约70倍。reversal只改变奇数DCT项符号，因逐项单位化使normalized trace
几乎正交（中位relative-L2约`1.414`），并形成显著BA/action扰动；但closed loop反而
`reversed=105 > correct=99`且不显著，证明该order敏感性没有对齐真实任务程序。

这比“共享参数不够多”更早：如果先加experts，只会给被放大的低能量轨迹噪声更多独立容量。

## 4. Energy-preserving temporal spectrum

每条视频仍计算同一个orthonormal DCT-II pooled tensor`[20,16,1024]`，但不再逐token单位化。
改为每视频一个全局scalar：

1. 计算raw pooled tensor总能量；
2. 计算旧逐token normalize在同一tensor上的总能量，只作为固定scale target；
3. 用一个全局比例缩放raw tensor，使新旧每视频总trace能量完全matched；
4. 保留所有group/frequency之间的原始相对能量、零group与符号。

因此本实验只改变能量在频率/层之间的分配，不改变总输入scale、DCT basis、token数、Reader/M2P
参数量或optimizer exposure。zero video仍严格zero，K4 set permutation仍成立。

Reader的Q/K仍可对每个token做content normalization并结合temporal route决定关注位置；关键变化
是V保留真实raw amplitude，所以低能量高频即使被Q/K选中也不能冒充强证据。该机制对functional
AS与未来reward credit完全相同，不是监督专用loss或LIBERO outcome trick。

## 5. Fresh身份与retirement

- 原位替换`temporal_trace_tokens`，不保留可执行的per-frequency normalization旁路；
- 新architecture、launch schema、config schema与checkpoint family，严格拒载旧checkpoint；
- 旧config从活动树退休，历史由Git与sealed run保存；
- source policy、Writer参数拓扑和初始化seed不变，step0仍为template-A/zero-B identity；
- 不从macro100或任何历史Writer warm-start。

## 6. 验证、profile与正式裁决

CPU合同新增：raw frequency energy ratio在输出中保持、每视频总trace energy与旧scale target匹配、
zero/order/K4 permutation不变；其余shape/freeze/gradient/checkpoint合同沿用。

clean commit后在live最多6张空闲A40上做longest105、K4/B20/B2、16-frame chunk的
fresh0→1与exact-resume1→3。通过后才从identity fresh0→200，并严格评50/100/150/200。

预注册机制判断：

1. 若task-gradient retention相对上一版有持续改善且correct/breadth共同上升，说明被放大的
   temporal noise是credit不稳定的重要来源；
2. 若频谱修复后task gradients仍接近1/24抵消且行为不升，才打开由冻结语义地址驱动的
   condition-specific sparse value experts；
3. 不用functional loss、漂亮rank、单个control或checkpoint envelope替代single-checkpoint
   strict correct400；最低仍严格`>150`，达到后继续提高。

## 7. 禁调项

本轮不改K、DCT项数、LR、warmup、B20、rank、global scale、axis blocks或训练objective；不加
rank/contrastive/reconstruction auxiliary loss，不做多LoRA/多checkpoint平均，不用held actions、
reward或outcome选择频谱。若失败，只根据fresh行为与task-gradient证据决定是否进入sparse experts。

## 8. 实现封存

- clean`22234c4`将`temporal_trace_tokens`的per-token L2 normalize替换为每视频一个
  total-energy matched scalar；旧normalization不保留可执行旁路。
- 新唯一config为
  `configs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_bci_v1.json`；新architecture、
  config/launch/checkpoint schemas与family均严格拒载旧layer-trace checkpoint。
- 聚焦测试、全仓`191 passed`、compileall、real config load和diff check通过。CPU合同直接
  检查输出frequency-energy fractions与raw DCT一致，同时每视频总能量与旧输入一致。
- `gpu01:0,1,2|4,5,7`六卡profile已严格fresh0→1、exact-resume1→3通过；三步
  loss=`.150377/.152822/.148504`，grad norm=`.000589/.000636/.000639`，0 clip/OOM/
  nonfinite，step2起reader和axis均有finite update。peak reserved=`20,375,928,832` bytes，
  三步约`36.98/36.91/36.70s`，权重永久弃用。
