# V6-LPCP CFMG Unit-Secant Endpoint Preference

状态：2026-08-16 active preformal authority。简称 **USEP**。本轮从sealed LPCP fresh开始，完整保留
CFMG的输入、backbone memory、content-first grid、rank32 LoRA和global commitment；唯一改变每个matched
winner/loser action pair如何定义endpoint preference。

## 1. Decision

CFMG已经把视频读取、same-task跨video方向、held写出与native LoRA链路接通，但full24第一次public update仍
被task38支配。六个active tasks各有相同的8个selected pairs和16个replay rows，故差异不是task权重或样本数：

| task | matched winner/loser action RMS | Writer gradient norm |
|---:|---:|---:|
| 4 | .001659 | .0003374 |
| 19 | .001330 | .0000319 |
| 20 | .001743 | .0001172 |
| 25 | .001595 | .0000841 |
| 34 | .001375 | .0001934 |
| 38 | .007816 | .0198151 |

现有每行loss为`softplus(D_w-D_l)`。它对生成动作的梯度正比于`a_l-a_w`，所以动作对比更大的task会在进入
Writer之前先获得更大的cotangent；task38最终达到次大task的`58.73x`，task4和task19反而与shared mean方向相反。
CFMG的content-first改动只把六task梯度统一放大`1.78--1.92x`，证明继续移动gate或增加Program容量不针对这
个更早的尺度接口。

USEP把每个真实matched state的action secant只归一到单位方向，不按task gradient、suite或结果重权。

## 2. Exact objective

对一个selected matched state，令`a+`和`a-`分别为同一observation、noise和physical batch下重询得到的成功臂
与失败臂动作，`a_hat(x)`为当前video-conditioned LoRA下的完整10步deployed endpoint action。只在实际执行的
前缀和7个action维上计算：

```text
D+ = mean_valid (a_hat - a+)^2
D- = mean_valid (a_hat - a-)^2
s  = sqrt(mean_valid (a+ - a-)^2)
z  = (D+ - D-) / s
J  = softplus(z)
```

`s`由detached on-policy arm actions计算，逐selected state形成；先得到每行`J`，再完全沿用原来的trajectory、
view和task等权平均。每个secant必须finite且严格非零；CFMG full24中六个active tasks的最小selected RMS均至少
`.0009867`，因此不加可调epsilon、floor或temperature。

在winner/loser中点，若有效标量数为`N`，则`||grad_a_hat z||=2/sqrt(N)`，不再随winner/loser动作间距线性增大，
也不会像除以MSE那样让小secant获得反比更大的action gradient。raw winner/loser MSE继续写入metrics作诊断；
`preference_margin`和commitment acceptance改为同一normalized `z`，避免训练和裁决使用两套目标。

## 3. What remains unchanged

```text
exact language + K=4 ordered action-hidden same-task videos
 -> one native PI0.5 joint context forward
 -> 37 layer-matched one-way memory tokens
 -> per-video ordered temporal program
 -> permutation-invariant K-set aggregation
 -> layer/token M2P content grid
 -> zero payload gate -> native rank16 B residual
 -> concatenate frozen LPCP rank16 carrier
 -> one complete rank32 LoRA -> frozen policy rollout
```

CFMG的2,828,928 trainable parameters、step0 exact LPCP、four disjoint correct K4 credit views、两paired states、
reward labels、selected occupancy states、AdamW、`j=0..10`全局all-task/all-view commitment、frame stride5、
信息墙与one-forward合同全部不变。没有task ID、task-gradient norm、PCGrad、task weight、第二套adapter、expert route、
额外rollout/forward或生成后RL。

## 4. Historical boundary

这不是TCEC后被禁止的task-gradient normalization：USEP在任何Writer backward之前、对每个物理action pair使用
同一个无task公式，无法观察task梯度范数，也不改变task/view权重。它也不是CGIK-JC否决的视频feature
common/secant单位化：不配对或归一化teacher-video representation，不会在duplicate videos处产生不连续；
winner/loser actions本来就是当前reward合同中必须存在且已经验证非相同的训练期targets。

本轮也不声称动作距离越小证据越可靠。选择RMS而非MSE分母正是为了只移除原始cotangent的一阶幅度因子，不把
低能pair反向放大。若单位secant仍不能形成共同方向，结果会否决本objective calibration，不能再用epsilon、
temperature、clamp、task weights或gradient solver补救。

## 5. Implementation ownership

`src/ember/writer/reward_preference.py`继续是唯一loss owner；在现有gradient和no-grad acceptance两处复用同一个
unit-secant helper。CFMG model/parameter grid不改，不新增Writer或runtime分支。fresh config、checkpoint、launch、
completion和evaluation identity统一改为USEP；CFMG checkpoint不得resume，历史由Git和formal artifacts保留。

## 6. Preformal falsifiers

CPU与真实GPU按以下顺序裁决：

1. 公式测试证明mask正确、中点margin为0、交换winner/loser翻转符号、统一缩放action secant不改变中点action
   gradient，以及targets/denominator不获梯度；现有信息墙、one-forward、step0和全链测试继续通过；
2. fixed world3使用full24中预先观测到的三个机制witness task4/34/38，沿用formal RNG并必须复现
   candidate/reference=`0/1,1/2,1/0`、每task 8 selected pairs；它们分别代表shared反向、four-view冲突和幅度支配，
   不是performance checkpoint selection；
3. 三task每个four-view gradient均finite nonzero；task38/次大task mean gradient norm ratio必须从`58.73x`降到
   `<=15x`，三task相对raw shared mean descent coverage必须为`3/3`，task34四个views对其task mean的descent
   coverage必须为`4/4`；
4. 原有11个actual Adam backtracking candidates至少一个使三个tasks的全部12个normalized deployed margins
   严格下降，并产生非零q/v/action native BA与fixed-action response；否则终局；
5. 没有新增policy/video forward、禁读、OOM或nonfinite；RMS运算的wall影响只作吞吐诊断，不为低位逐元素一致
   改batch、kernel或dtype。

world3通过后才做同checkpoint validation8 four-view held gate，沿用CFMG的至少6/8 tasks、aggregate BA
cosine/energy至少`.40/.50`、held/train L2在`.30--4x`、reverse material与constant近零合同。该门只证明训练所得
Program可迁移，不选择closed-loop模型。

### 6.1 Canonical implementation evidence

唯一`reward_preference.py`已原位复用同一个unit-secant helper完成gradient与no-grad commitment evaluation；
CFMG grid、参数量和forward图未改。fresh config/checkpoint/launch/completion/eval identity已统一切换为USEP，旧CFMG
config从active tree退休且checkpoint不能resume。新增公式测试覆盖mask、winner/loser几何、detached targets与统一
action缩放下的中点softplus gradient不变；相关合同=`143 passed`，完整CPU=`413 passed`，compileall与diff check
通过。architecture guard为0 hard violation、无新module/entrypoint/并行version；review项仅为既存大文件/函数。
这些结果只关闭实现门，不提供真实task共存、视频held或closed-loop证据。

## 7. Full24 and strict decision

所有preformal门通过后，只允许一次从sealed LPCP fresh开始的full24 cycle1。cycle1必须保持每task等权，并报告
active tasks、每task/view gradient、task38 dominance、global acceptance、parameter delta和q/v/action response。
若global commitment no-op、task38 dominance仍`>15x`或任何active task/view不能沿同一candidate下降，USEP终局，
不补MSE分母、epsilon、temperature、task weights、PCGrad、rank/scale/seed或其它小扫。

若产生nonzero checkpoint，立即做single-checkpoint K4 strict paired400并与LPCP143、v6-fast143、SFMC144和最近
direct baselines逐task比较。cycle1至少correct142、breadth7、相对LPCP lost<=15且gained>=lost才允许cycle2。
相邻checkpoint约145或更高且低churn/high Jaccard后才具稳定资格；首次达到该门立即补same-task-other、wrong、
shuffled、reversed和no-video，correct必须沿有用policy direction显著更好。

## 8. Negative boundary

失败只淘汰“CFMG memory/content grid + per-state unit-RMS action secant endpoint preference + 一轮global commitment”。
它不否定memory token、dynamic K/few-shot、rank8、完整A/B、LoRA生成或后续task-local RL；也不能把内部梯度更均衡
冒充closed-loop提升。
