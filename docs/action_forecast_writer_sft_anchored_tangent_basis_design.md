# SFT-Anchored Policy Tangent-Basis Writer 设计

状态：**2026-08-05 diagnostic与独立one-cycle A40 profile已通过；formal0→1前authority。**

## 1. 目标与根因

目标仍是同一single checkpoint的strict correct严格超过`150/400`，同时不牺牲task
breadth与视频因果性。本方法不把RL当成绕过LoRA质量问题，而把IL与RL分工：

- IL负责从target action supervision学出闭环有效的policy tangent dictionary；
- RL在关闭teacher action后，只学习task language + one hidden-action video应如何组合
  dictionary中的方向。

Task-Grounded Progress Credit cycle1/2证明reward可以到达Writer，但AS125→cycle2的
参数hybrid又证明：upstream贡献更多effective BA变化，8个共享factor-output矩阵却贡献
更多fixed-action变化，而且这种policy leverage随task/suite变化。继续让reward同时旋转
dictionary与系数会把不同task credit写进同一个高杠杆共享接口，形成能力换手。

## 2. 选择已有basis/coefficients边界

每个v6 `FactorHead`已经是：

```text
conditioned rank-slot state
-> network.0 + GELU: hidden coefficients
-> network.2: public A/B row in policy parameter coordinates
```

`network.2.weight`的列天然是该policy target的输出方向字典；8个heads覆盖q/v/action的
A/B输出。因而本轮不新建store、router、target-owned heads或正交loss，而把已有分界变成
显式训练合同：

- 冻结`semantic_encoder`，既保持Writer条件表征也作为只读progress observer；
- 冻结恰好8个`factor_heads.*.network.2.weight`；
- 训练`semantic_core`、`visual_transition`、`procedure`、`compiler`和8个
  `factor_heads.*.network.0.weight`；
- public LoRA仍是完整rank16、38 targets，template A/zero B、source policy和
  normalization全部不变。

这不是监督专用regularizer。任何policy-gradient、actor-critic或离线reward方法都可在
一个先学稳定action basis、后学condition coefficients的actor上使用；未来若完全不用
SFT，也可由其他policy-aware预训练获得同一dictionary。

## 3. IL起点

使用唯一历史强v6-fast macro400：

`runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729/checkpoints/step_00000400`

其strict correct=`143/400`、breadth6，五臂=`143/135/125/128/129`；writer SHA为
`6a42642b...c7011f`。不从AS125=`97`继续，因为新方法的目的就是保护已由IL获得的强
policy basis，而不是要求reward从弱起点重新学五十多个成功。

原run contract保留A100绝对路径，只作provenance。BCI warm-start只用现有
`source_reference_matches`按source optimizer step、training commit与已由current source
authority验证的model layout重绑定；authorities、Writer config、LoRA contract和checkpoint
manifest仍须一致，不改写历史contract、不复制或派生假checkpoint。

历史v6 schema已退役，不恢复为活动loader。canonical RL config引用当前受支持的v6
cold-start config，并显式封存non-parameter runtime override，把encoder chunk从该config的
16恢复为checkpoint原始32；解析后的authorities与完整Writer字段逐项等于macro400 run
contract。A40不能假定32可用；正式前必须以最长105-frame真实任务完成diagnostic与
one-cycle writer-update profile。若46GB OOM，才允许把该显式runtime override改成16并
重新profile，不能暗改科学权重或伪造checkpoint authority。

历史macro400 manifest/launch schema只在`initialize_writer_phase`的load-only warm-start
入口接受；仍逐文件验证size/hash、manifest canonical payload、owning run contract、完整
Writer字段和LoRA contract。exact-resume、正式AS evaluator及新checkpoint写入不接受该旧
schema，因此这不是恢复历史训练路径。

## 4. Reward与信息墙

保持现有Task-Grounded Progress Credit：

- official random-reset K4；mixed task只用binary LOO；
- all-success零梯度；all-failure才用冻结semantic progress LOO；
- positive PPO、negative SPO、Nmc4、executed prefix、full24等权、two epochs；
- Writer输入仍只有task language + exactly one action-hidden video；
- teacher/rollout action不进入observer，validation/test action/reward均为0。

progress observer随IL checkpoint改为macro400冻结semantic encoder，因此不能照抄AS125的
`50 successes/14 mixed/5+5`身份常量。新预注册门只约束机制而不按outcome选task：

- 24×K4完整，success在`[8,88]`且至少6个mixed tasks；
- mixed task中success utility更高比例至少`.60`，pair AUC至少`.60`；
- 所有teacher change finite/nonzero且重复误差不超过`1e-5`；
- 若存在all-failure tasks，至少一半utility range≥`.05`且range中位≥`.05`；若不存在，
  semantic tie-break不需要启用，该门自然通过；
- successful rollouts上correct对wrong/shuffled/reversed胜率均≥`.65`且margin中位>0；
- all-failure utility对pixel change的Spearman绝对值<`.8`。

门不过则不profile、不训练；不得为匹配AS125旧outcome重跑或改seed。

## 5. 单变量实验与训练合同

canonical config仍原位为`configs/pi05_rl_writer_development_v1.json`，但schema与method升级
为SFT-Anchored Tangent-Basis；旧cycle1/2由各自run contract和Git保存，不保留parallel
config。与上一正式RL相比，科学变量只有：

1. cold start从AS125换为v6-fast macro400；
2. 同时把该checkpoint的8个factor-output矩阵作为冻结basis。

这两项是同一个不可分割设计：强IL起点提供要保护的basis，冻结basis定义IL→RL边界。
reward、K4/Nmc4、LR`1e-5`、two epochs、task/video schedule、full24 aggregation与随机种子
不变。不得同时加入policy-distance loss、额外head、router、scale、rank/energy约束或新
reward。

正式顺序：

1. 六卡只读diagnostic；
2. 独立fresh one-cycle profile，验证8个basis参数grad/optimizer ownership为0、其余五个
   主block可达、finite/OOM、完整checkpoint与fresh→resume边界；profile权重丢弃；
3. 新formal root从macro400 fresh做cycle0→1；
4. 立即做与macro400 strict panel配对的correct400。

只有cycle1相对143出现多task共同净增、breadth不降，或已经严格超过150，才允许续cycle2；
若只是单task换手、aggregate下降或Long/Spatial coverage继续丢失，则停止。functional
loss、训练reward与几何不选checkpoint。

## 6. 结果解释

- 若cycle1提高且basis严格不变，支持“稳定policy dictionary + reward学习coefficients”；
- 若fixed basis仍发生大规模能力换手，说明共享dictionary本身覆盖不足或credit在系数空间
  仍冲突，下一步才考虑显式多basis但必须保留policy-aware初始化；
- 若训练几乎不能移动action，说明macro400的可用方向需要decoder适配，此时再比较软
  policy anchor，而不是立即解冻全部basis；
- near-rank1、raw A/B cosine或更漂亮layer energy都不能覆盖closed-loop结果。

## 7. 实现ownership与退役

唯一训练owner继续是`src/ember/rl_writer/`与`scripts/train_rl_writer.py`；不新增第二套
runner。`runtime.py`负责freeze与optimizer ownership，`contract.py`负责schema，现有
checkpoint/evaluator原位升级。参数hybrid一次性分析入口已删除，artifact与commit
`67b245a`保留历史复现。

## 8. Macro400只读diagnostic结果

clean`303e714`在live空闲`gpu01:1,2,3,4,5,7`完成24×K4全量诊断，正式root为
`runs/outputs/pi05_sft_anchored_tangent_basis_diagnostic_macro400_r6_303e714_20260805`。
96/96 rows、61 successes、11 mixed、8 all-success、5 all-failure；suite successes为
Spatial/Object/Goal/Long=`17/17/16/11`。wall max`388.80s`，peak CUDA reserved
`19,289,604,096` bytes；0 optimizer update、Writer backward、checkpoint和teacher/
validation/test action read，六卡结束后全部释放。

全部预注册机制门通过：mixed success utility更高`11/11`，pair AUC`.91429`；all-failure
range有`4/5`≥`.05`且中位`.27273`；pixel Spearman`.48421`。successful rollouts上correct
胜wrong/shuffled/reversed比例=`1.0/.90164/1.0`，margin中位=`.55919/.37889/1.53747`。
因此当前progress observer不仅对binary outcome有序，而且明显读取正确视频内容与时序，
允许进入第5节独立one-cycle profile；这仍不证明一次RL更新会改善closed-loop。

首次launch在旧raw v6 config schema处、第二次在旧manifest schema处分别fail-fast，均为
0 rollout/checkpoint并释放GPU。根修不恢复历史训练loader：当前v6 config加显式32-frame
runtime override重建effective Writer；旧manifest/launch schema只在IL→RL load-only
warm-start入口接受，exact-resume和AS evaluator继续拒绝。真实macro400 manifest、owning
contract与逐文件身份已一次核验通过。

## 9. One-cycle A40 profile结果

clean`2f934bd`在live空闲`gpu01:1,2,3,4,5,7`完成独立fresh profile，root为
`runs/outputs/pi05_sft_anchored_tangent_basis_profile_macro400_r6_2f934bd_20260805`。
24 tasks×K4全覆盖，61/96 successes、11 mixed、5 all-failure semantic；两轮学习均为
finite，ratio范围为`.98433--1.01762`/`.98100--1.05652`，clip fraction为
`0/.000143`，grad norm为`.004373/.003961`。五个主block在两轮均有非零
gradient，5/5 all-failure tasks都有非零generated-LoRA gradient，progress observer
gradient tensor始终为0。

wall max`2033.38s`，peak CUDA reserved`19,478,347,776` bytes，0 OOM/watchdog；
teacher/validation/test action/reward reads全为0。两轮都先收齐6/6 CUDA-complete
rank marker再对称进入NCCL，原子cycle1 checkpoint包含6份实rank state、trainer state、
Writer及完整24-task/K4 consumed schedule。

对macro400与profile cycle1的Writer做逐张量比较：8个basis张量和440个
semantic-encoder张量逐元素完全不变；恰好只有76个预注册系数侧张量改变，
且semantic core/visual/procedure/compiler/factor-input分别`22/5/16/25/8`张量
全部改变。这同时封存了optimizer ownership、真冻结和可达性，profile权重永久弃用。

该profile contract封存`total_cycles=1`，不得为了形式resume将已完run改成
`total=2`，也不新增只服务验证的resume-load旁路。正式合同从一开始就封存
`total=8`并先停cycle1；若held结果过续训门，同checkpoint cycle1→2将在原
96-rollout/two-epoch规模提供真实exact-resume证据。因此当前唯一下一步是
clean/pushed commit上的fresh formal0→1，随后立即做与macro400配对的strict correct400。
