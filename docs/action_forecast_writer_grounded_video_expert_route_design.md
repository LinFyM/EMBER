# K4 Grounded-Video Semantic-Expert Route Writer

状态：2026-08-07当前设计与实验authority。Sparse Semantic-Expert的identity-fresh训练、四点、
winner五臂与内部分析已全部完成并负裁决；本方法必须从functional identity建立fresh
architecture/config/checkpoint family，不加载任何历史Writer权重。

## 1. 决策

下一轮保留当前已经实证可工作的K4 video-to-LoRA主体，只把**parameter owner的地址来源**从
task-language-only改成冻结source policy内部的task-grounded video semantics：

```text
exact task language + K4 action-hidden same-task videos
  -> frozen PI05 multimodal task-token video innovations
  -> K4 semantic address -> frozen train24 video-derived top1 route
  -> one complete independently-owned Trace Reader + axis-M2P expert

same exact inputs
  -> frozen PI05 20-group DCT16 direction/physical/evidence traces
  -> routed experts read video values and combine one memory
  -> one complete public rank-16 LoRA.
```

Router和Reader读取的是同一组K4视频的不同冻结视图。route只决定哪个完整parameter map接收
credit；LoRA的全部动态value仍来自20-group traces。不是language直接生成LoRA、raw task ID分桶、
逐视频LoRA挑选或多LoRA ensemble。

## 2. Sparse Semantic-Expert的决定性负结果

routefix formal从clean`3820f27`和identity完成macro0→200：200 finite macros、96,000 action
queries、19,200 K4 videos、0 clip/OOM/nonfinite。四点correct400=`74/74/78/75`，breadth=
`6/5/5/5`；single winner macro150=`78`。expert-local Reader retention四窗=
`.2847/.2542/.2187/.2053`，axis=`.2419/.2146/.2034/.1959`，明显高于上一版shared owner最后
约`.05`，所以参数隔离确实减少了局部task-gradient抵消。

但winner五臂为：

| correct | same | wrong | shuffled | reversed |
| ---: | ---: | ---: | ---: | ---: |
| 78 | 85 | 90 | 83 | 92 |

correct是五臂最低。wrong相对correct gained/lost=`20/8`，reversed=`26/12`；五臂success
union/intersection=`123/55`。这不是继续训练或挑checkpoint能修复的弱margin，而是当前条件地址
与视频语义的方向相反。

内部production-batch replay进一步排除了“视频被忽略”：same/wrong/shuffle/reverse从
physical trace到Reader/program/effective-BA/fixed-action的relative-L2中位分别为
`.135/.053/.051/.065/.010`、`.309/.194/.209/.279/.050`、
`.251/.191/.183/.254/.035`、`.335/.197/.205/.278/.044`。正确LoRA norm/stable-rank/
top-singular-energy=`44.79/1.412/.791`，top4 target energy`.489`。wrong/order视频真实改变
高增益LoRA并传入action；问题是这些变化没有被组织成“正确视频更适合当前task”的policy方向。

## 3. 当前最早失败接口

language-only route对同一target language的correct/same/wrong/shuffle/reverse永远选同两个experts。
因此训练只教会某个language-owned map读取matched videos；部署时任何非零视频trace都进入同一map，
而router本身无法表达“这段视频的高层任务语义是否与目标语言相符”。wrong与reversed造成更大的
LoRA扰动时，恰好把source policy推入更通用的成功方向，于是behavior反而上升。

所以不能把结果重新概括成shared credit、LoRA rank、能量或视频数不足。完整expert隔离已改善
gradient coexistence却没有改善absolute；最早结构缺口是：

```text
high-level video semantics -> policy-parameter ownership
```

下一轮必须让视频内容参与寻址，而不是继续增加language-routed expert数、放大trace、改rank、
增加训练时长或先上reward。

## 4. Frozen task-grounded video address

地址直接复用冻结PI05对每个teacher frame已经执行的multimodal prefix：

1. 用target exact language与一帧teacher image运行冻结PaliGemma+action-expert joint forward；
2. 在最后一层对PaliGemma prefix作final norm，只平均authoritative task-span tokens；
3. 减去同language、同position、zero-image prefix的task-token hidden，得到2048维video innovation；
4. 每条视频对全部stride-5 frames取mean并L2 normalize；
5. K4四条video address等权mean，形成一个task-grounded video semantic address。

这个address不是单独的视觉scene embedding：task token经过冻结multimodal backbone，因而表示“在这条
task描述下，这段视频对高层语义造成什么改变”。zero-image subtraction去掉language-only公共量；
视频值全零时address即使产生任意route，Reader仍因zero trace输出zero memory，最终严格identity。

route address故意对frame permutation取mean，因此同一组frames的correct/shuffled/reversed route
完全一致；时序变化只能经DCT value路径影响LoRA，不能通过换expert冒充时序理解。same-task另一K4
set应保持高route overlap，cross-suite wrong则应由视频语义改变owner。

## 5. Train24-only route artifact

只读取24 train tasks各50条action-hidden teacher videos，不读任何action、state、reward、terminal、
outcome或validation/test input：

1. 对每条train video生成上述2048维unit address；
2. 每task平均50条address得到task prototype；
3. 对24 prototypes取global mean，中心化并L2 normalize；
4. seed7 deterministic farthest initialization + spherical k-means生成8 centers；
5. runtime对K4 address作同一mean-center-normalize，固定取top1 center，权重`1.0`。

初始top2预门在不读action/outcome/validation input的前提下得到primary稳定率`1.0`、top2 exact
`.984833`、overlap`.992417`，8 experts均有primary owner；但batch4/singleton只有`23/24` exact，
task35的secondary owner在两个近邻间跳变。secondary仍占`.5`完整Writer参数，不能把这个差异当作
数值噪声放过。由于primary在全部6,000个随机K4 set及24个batch/singleton对上都严格稳定，本authority
在正式训练前把route收敛为top1 one-hot；这是input-only gate作出的结构裁决，不使用rollout结果调参。

最终训练前gate要求：8 experts均有primary task；train task随机K4 route稳定率不低于`.90`；
batch4/singleton top1严格一致。若该门失败，不得用rollout、action、outcome或validation performance
调center、expert数或route。

## 6. Reader、LoRA与信息墙

- 八套完整independent Trace Reader+四轴M2P、20 groups、DCT16、direction/physical/evidence、
  memory68×1024、38 targets与public rank16保持不变；
- 每condition只执行route选中的一个完整expert，再decode一次；
- Writer输入仍只有exact task language + exactly four action-hidden videos；不得读取teacher action、
  proprio、reward、terminal、task ID、filename、object pose、normalization或policy outcome；
- 每task每episode只产生一套LoRA，不能逐video生成后平均、挑选或ensemble；
- source policy、normalization、descriptor与route全部冻结，step0与zero-video严格identity。

## 7. 为什么同时适用于AS和未来RL

route由部署时同样可获得的language+video输入固定产生，不依赖functional loss、teacher action或
LIBERO outcome。AS cotangent和未来rollout reward cotangent都沿同一selected experts反向；改变的
只是condition如何选择可累积credit的parameter owner。没有SFT reconstruction、contrastive label、
rank/order auxiliary、success gate或环境heuristic，因此不是监督学习专用trick。

## 8. 实现与聚焦验证

保持一个canonical runtime，原位替换language router：

1. `video_program.py`在现有冻结joint forward中同时返回per-video grounded task-token address，
   不重复执行source policy；
2. route generator改为六卡流式提取train24×50 input-only addresses并封存小型artifact；
3. `model.py/fewshot_m2p.py`只把router输入从language anchor换成K4 grounded address；完整experts、
   decode与optimizer语义不变；
4. architecture/config/checkpoint schema fresh incompatible，旧sparse checkpoint拒载；
5. 聚焦合同覆盖address baseline、K4 permutation、order-invariant route、wrong-video route变化、
   top1 stability、zero identity、unselected gradient、source freeze、actual-world-size ownership和resume。

不得增加learned router、language residual route、task-ID fallback、load-balance/contrastive loss或
outcome-selected centers。真实vertical path足以给证据时不扩展大而泛的test harness。

## 9. A40与正式裁决

route artifact input-only gate通过后，live比较`gpu01/gpu02`并只使用最多6张空闲A40。先以现有
logical B20、policy B2、K4、16-frame chunk、longest105做fresh0→1与same-root exact-resume1→3；
保持显式`NCCL_P2P_DISABLE=1`和3+3 NUMA。route address复用现有forward，预期显存不高于上一
sparse profile45.59GB，但必须实测。

formal从identity fresh0→200、every25 checkpoint，严格评50/100/150/200 correct400；只由single
checkpoint absolute、breadth、churn选择winner，再做五臂和内部分析。核心机制门是correct必须高于
wrong且video-derived route在same/wrong/order arms符合预注册语义；functional loss与内部几何不选点。
长期最低目标仍为同一single checkpoint strict correct`>150/400`，达到后继续提高。

## 10. 禁调项

本轮不改K4、DCT/evidence、expert数/top1/one-hot weight、rank、LR、B20、full24、AS objective、
optimizer、checkpoint schedule或source policy；不加reward、SFT-only auxiliary、scalar/global
scale、multi-LoRA、checkpoint融合、挑video、延长同一失败schedule或从任何历史best warm-start。

## 11. Formal训练完成与当前裁决边界

clean`a758bba`已按第9节合同从functional identity自然完成macro0→200；root为
`runs/outputs/pi05_as_writer_k4_grounded_video_expert_trace_m2p_formal_fresh0_200_r6_a758bba_20260807`。
200行metrics均finite，共96,000 action queries、19,200 K4 action-hidden video conditions和8个
every25 checkpoints；0 clip/OOM/nonfinite，source trainable=0，validation/test action和video
value reads均为0。wall=`8828.911s`，peak allocated/reserved=
`36,708,964,864/42,727,374,848` bytes，未降低B20或改变任何scientific contract。

按expert实际owner task数消除互斥block的zero padding后，Reader与axis-M2P在八个25步窗口的
gradient-energy retention中位始终约`.45--.60`与`.49--.53`；grounded top1 ownership确实避免了
shared map的1/24抵消。但该机制证据不选择checkpoint，functional loss也不作行为结论。下一步仍只
严格评macro50/100/150/200 correct400，再由single-checkpoint absolute、breadth、churn选择winner
并做五臂与内部route/path分析。

本轮rollout启用canonical evaluator hashless launch v2：不生成或复核checkpoint、authority、raw
shard、aggregate与completion内容hash；用path/schema/size、真实解析/加载、显式UUID run reference
与direct paired-control字段保持身份和配对。policy-noise RNG算法保持原sealed schedule，以便和历史
fixed panel严格配对。

## 12. Formal rollout、内部机制与最终负裁决

四点strict paired correct400已经全部完成：macro50/100/150/200分别为
`76/88/77/82`，breadth>=5为`3/4/3/3`，success union/intersection=`125/40`；相邻
gained/lost=`27/15,17/28,25/20`。single winner是macro100=`88`，明显低于v6-fast`143`
和严格门`>150`，并且100→150发生净退化，task轮换没有解决。

winner五臂为`correct/same/wrong/shuffled/reversed=88/87/82/86/86`，五臂
union/intersection=`129/48`。相对correct的gained/lost分别为`15/16,16/22,17/19,17/19`，
全部direct paired-control字段为0 mismatch，任一视频对照都没有形成material closed-loop margin。

8-task refs1内部分析证明失败并非“视频完全被忽略”：wrong从grounded address到
`physical/direction/Reader/effective-BA/fixed-action`的relative-L2中位为
`.168/.310/1.319/.293/.433/.099`，并使2/8 tasks切换完整expert；shuffled与reversed保持route
不变但到BA仍分别为`.426/.435`，说明时序输入能实质改变LoRA。same-task另一K4 set到BA/action
仍有`.087/.0217`差异，leave-one-video-out到BA/action为`.0379/.00642`，四条视频都有非零影响。

生成LoRA也不再是严格rank1坍缩：stable-rank均值`1.463`、首奇异能量`.773`、90%/99%能量
平均需要`4.78/12.20`维。然而这些更丰富且视频敏感的LoRA没有形成正确视频的行为优势。结合
expert-local约`.45--.60`的gradient retention，正式拒绝“只要让视频地址选择完整独立parameter
owner即可解决漂移”的假设。hard top1把优化冲突隔离掉，也切断了v6曾有用的跨task共享语义迁移；
当前最早缺口是共享高层video program的形成与组合，而不是继续调route、expert数、rank、scale或
训练时长。本方法不得resume、warm-start或恢复为活动路径。
