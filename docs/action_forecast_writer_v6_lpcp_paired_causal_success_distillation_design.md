# V6-LPCP Paired Causal Success Distillation

状态：2026-08-15 design authority。简称`V6-LPCP-PCSD`。本轮从已经封存的V6-LPCP macro25权重开始，
冻结视频carrier、AS139底座和完整rank16 compiler，只用train24严格配对的AS139-reference/LPCP-candidate
闭环结果校准现有`query_delta`。canonical实现已完成、全量CPU=`387 passed`且architecture guard无hard
violation；尚无GPU或closed-loop结果，本文任何预期都不能写成性能结论。

实现封存前状态（2026-08-15）：

- `model.py`把一次joint context read拆为可复用conditioning state与query-on/query-off recompile；
- `rollout.py`只保留canonical K2 trajectory replay和selected-success collate；
- `reward_preference.py`只保留positive selected-success CFM LoRA cotangent；
- `reward_cycle.py`实现同reset/RNG两臂、ties zero、active-task等权及query-only AdamW；
- fresh config/checkpoint/evaluator schema与ADSP不兼容；尚未做真实GPU coverage/profile。

## 1. Decision

V6-LPCP strict=`143/400`只追平历史v6-fast，不是突破。相对同teacher/state/RNG schedule的AS139为：

```text
120 both succeed / 23 LPCP-only / 19 AS139-only / 238 both fail
```

因此两套严格成功集合的union是`162/400`。这不是可部署成绩，也不允许在validation时做oracle选择；它只提供
一个新的训练事实：**AS139强底座与LPCP新增分层视频路径合起来已经包含超过150的闭环support，但blind B20
把19条旧support换掉了。**

LPCP机制证据同时排除了若干更早断点：

- 同一次真实image/language/50 Action probes joint forward中的18层carrier有material顺序证据；
- reader/controller持续获得gradient，same-task correction coherence没有重新近正交；
- 完整LoRA能改变closed-loop，但effective-BA只在AS139邻域移动约`.002653` relative-L2；
- gained与lost的BA幅度高度重叠，functional loss也不能区分哪种移动闭环有益。

所以下一轮不把literal memory token、rank8、更多factor heads或更健康LoRA几何当作先验答案。唯一主要变量是：

> 用同初态、同policy RNG的AS139 reference与当前LPCP candidate两臂真实成败，选择两臂中唯一成功的
> policy-generated trajectory，并只训练LPCP最后的layerwise Procedure query commitment map。

## 2. Why this is not an old reward or guard experiment

历史V6 Ordered-Procedure Reward使用同一个candidate的四条rollout做LOO：成功轨迹正模仿、失败轨迹反模仿；
cycle1得到`138`，因为一个shared update仍产生18 gains/19 losses。ADSP再约束成功prefix一阶不增，仍为`138`。
PCUG/NPCG/CVEG则用base/candidate paired outcome决定是否把blind Program write投影掉，不从成功arm学习新的
continuous direction。

PCSD与它们的区别是：

1. reference arm是同一V6-LPCP Writer把新增`query_delta`严格置零后的AS139，不是另一个checkpoint selector；
2. reference和candidate在相同reset、env seed与policy-noise prefix上执行；
3. 只有两臂结果不同时才产生credit；
4. candidate成功时蒸馏candidate自己的成功轨迹，保留真实新增能力；
5. reference成功时，在candidate LoRA下蒸馏reference的成功轨迹，恢复被新增路径破坏的support；
6. 不反向“去模仿”失败轨迹，不用reward scale、margin、candidate reject或guard projection；
7. reference冻结，训练不可能靠把reference推坏制造相对优势。

该objective仍可能失败：train24 paired success不一定外推validation8，executed-prefix flow matching也不等于
return gradient。它的价值在于直接检验union-support是否能经一个共享、视频条件化的commitment map共存。

## 3. Unchanged deployment architecture

部署图完全保持V6-LPCP macro25：

```text
exact task language + K same-task action-hidden ordered videos
  -> one native joint image/language/50-action-probe forward per sampled frame
  -> frozen V6 Semantic Core + directed Procedure
  -> 18-layer Action-probe carrier
  -> frozen layer/rank reader + frozen causal conditioner
  -> trainable shared query_delta[256 x 256]
  -> native 320 policy-layer/rank Procedure queries
  -> frozen AS139 set/fusion/compiler/factor heads
  -> one complete 38-target rank16 LoRA
  -> frozen source policy
```

训练结束后只保留一个Writer checkpoint和一套生成LoRA。reference arm只存在于train24 ephemeral credit；
validation/test deployment不运行reference、不选择arm、不平均LoRA，也不进行在线更新。

本轮固定K4，因为现有严格union证据来自K4，且owner不要求人为做K1/K4公平竞赛。底层AS stage已经均衡覆盖
K=1..4，架构仍能接受dynamic K；PCSD是否保持其它K只在方法absolute过门后检查。本轮不同时改变video数量。

## 4. Exact paired training graph

初始化：

```text
V6-LPCP macro25 complete Writer weights
  freeze base_writer, procedure_set, layer_probe_reader, probe_conditioner
  train only query_delta.weight (65,536 FP32 values)
  fresh AdamW state
```

每cycle、每个train task读取exact language和同一K4 correct videos一次。Writer共享一次backbone evidence计算，
然后从同一detached conditioning state生成两套ephemeral LoRA：

```text
reference: query_delta = exact zero -> exact AS139 LoRA
candidate: current query_delta       -> current LPCP LoRA
```

每task固定两个paired unseen random-reset initializations。reference K2与candidate K2严格共享：

- rollout cursor、reset state和environment seed；
- policy-noise root及每次replan noise prefix；
- language、K4 videos和video order；
- official preprocessing、dummy settling、horizon、5-action execution和10 flow steps。

两臂按相同K2 batch shape执行，接受正常BF16/TF32低位差异。每个arm只运行一次，不做batch1复现或重复forward。
完整train24共`24 x 2 states x 2 arms = 96`条rollouts，与旧K4单arm reward的总episode数相同。

## 5. Paired causal target

对task `i`、paired state `j`，令`r_ij`与`c_ij`分别为reference/candidate binary success。目标轨迹为：

```text
c_ij=1, r_ij=0 : candidate successful executed trajectory
c_ij=0, r_ij=1 : reference successful executed trajectory
c_ij=r_ij      : no gradient
```

两臂都成功时不重复强化已有support；两臂都失败时不伪造一个reward方向。每个被选target只保留当前policy实际
执行过的observation、normalized action chunk与前1--5个有效动作；它不是teacher action，也不进入checkpoint。

在当前candidate LoRA下定义成功轨迹executed-prefix flow loss。每条trajectory先对自己的replan chunks等权，
同task的1--2个discordant pairs再等权：

```text
L_i = mean_{discordant pair j}
        mean_{executed replans in winning trajectory}
        mean_{m=1..4} CFM(candidate LoRA; observation, successful action prefix)
```

使用sealed exact-Beta(1.5,1) time、task/cycle keyed Gaussian noise、Nmc4和最大安全physical batch。先对detached
candidate LoRA求cotangent；随后从缓存的shared Core、ordered Procedure memory和layerwise conditioners只重解
`query_delta -> Procedure read -> frozen compiler`一次，把cotangent传到`query_delta.weight`。不重读视频、
不重复joint backbone forward，也不跨simulator保留大autograd graph。

有discordant evidence的tasks各自先task内mean，再对active tasks等权形成一次shared AdamW update。无discordant
task严格贡献zero，不因轨迹更长或success更多获得权重。task ordinal只用于队列和RNG，不能进入Writer。

## 6. Why high-level video knowledge and order remain necessary

PCSD不把reward或reference policy变成部署输入。candidate的唯一可训练路径仍由每条视频的18层Action-probe变化、
video内causal delta与layer/rank conditioner决定；language只提供V6原有的task query/context。不同初始化共享同一
K4-video LoRA，而蒸馏target来自与teacher错开的closed-loop trajectories，因此不能逐帧复制teacher运动。

correct顺序不是negative loss制造的margin：训练从不把wrong/shuffled/reversed LoRA推坏。正确视频的有向carrier
只有在它生成的candidate改写真实增加或保留闭环成功时才得到正credit。若query_delta学习的是language/static
shortcut，最终wrong/shuffled/reversed/no-video会同步改善，方法按controls判为科学non-pass。

## 7. Why memory token is not the next variable

memory token仍是合法后继机制，特别是在证据无法按policy layer读取、或共享decoder缺少layer-addressed state时。
但当前V6已经拥有320个显式module/layer/rank routing slots，LPCP又证明真实18层carrier在correct/reverse/static
之前有内容。历史Target-Owned、Policy-Lane、Target-Spectral和Dynamic-K memory/rank8还分别证明：增加target
ownership、lane容量、高rank健康度或literal memory并不会自动给出policy-effective direction。

因此本轮保留memory原则中的有效部分——真实context、layer alignment、共享可扩展mapper——但不增加token。
若PCSD证明reward credit有内容而`query_delta -> effective BA/action`仍物理过小，才有证据改变commitment位置或
以layer-aligned memory替换它；不能在同一轮混入carrier、rank和decoder变化。

## 8. Coexistence hypothesis

PCSD不是per-task bank。所有task仍更新同一个`256 x 256`map；不同task/video的conditioner states提供不同输入，
共享map必须学到“什么样的分层有向证据应保留或修正哪些policy-slot query”。candidate-only success与
reference-only success同时进入同一update，分别提供acquisition与retention方向；这比全任务blind source-action
mean或只做局部guard更直接。

但`162`只是validation retrospective oracle upper envelope，不是该shared map的可达下界。若train paired gradient
仍互相冲突，或train success behavior不覆盖held occupancy，PCSD会失败。届时不能靠LR、pair数、seed或cycle小扫
解释，而应把最早接口推进到更强的Procedure-to-LoRA commitment或显式多task shared program factorization。

## 9. Implementation ownership and lifecycle

- `writer/model.py`继续唯一拥有V6-LPCP生成图，新增可复用的detached conditioning-state/recompile接口；
- `reward/rollout.py`复用同一persistent env与K2 paired arm执行，只增加successful replay返回，不复制simulator；
- `writer/reward_preference.py`原位替换退役ADSP的credit owner，拥有selected-success CFM LoRA cotangent；
- `writer/reward_cycle.py`原位替换旧support projection orchestration，拥有reference/candidate pairing、task-equal
  gradient和single AdamW step；
- `writer/reward_training.py`与`reward_checkpoint.py`保留一个CLI/checkpoint owner，schema fresh-incompatible；
- 旧Ordered-Procedure Reward/ADSP由Git、sealed configs、formal roots和research history保存，不保留active
  strategy flag、fallback或第二trainer；
- evaluator只扩展同一个V6-LPCP checkpoint family识别新的reward-stage kind，deployment model不分叉。

## 10. Fast falsification

### 10.1 CPU and synthetic

1. zero-query reference对K=1..4与AS139逐tensor等价，candidate仍等于LPCP；
2. reference/candidate共享一次joint evidence，reward recompile不重复backbone；
3. K2两臂state/env/policy seed和noise prefix严格配对，arm顺序不进入seed；
4. candidate-only/reference-only各选择正确成功trajectory；ties严格zero；
5. trajectory内、pair内与task间权重不受horizon、chunk数或rank ownership影响；
6. Nmc physical slicing不改变logical flow samples与LoRA cotangent；
7. gradient只进入`query_delta.weight`，65,536以外Writer/source policy trainable=0；
8. one optimizer step后query delta、effective BA和fixed-action response finite/nonzero；
9. checkpoint保存完整Writer、fresh optimizer、cycle、每rank RNG与world topology，不能误载ADSP；
10. teacher/target dataset/validation/test action或reward reads为0。

### 10.2 First full24 cycle

CPU门通过后，从clean pushed commit运行一个full24 cycle并同时作为真实coverage/profile。动态long-first queue、
同节点至多6张有益A40、deferred NCCL和P2P off保持。必须满足：

- 24 tasks、48 exact pairs、96 rollouts、四suite完整；
- discordant pairs至少4，candidate gains与reference gains均至少1；
- active tasks至少3并覆盖至少2 suites；
- selected replay、LoRA cotangent、query-delta gradient和AdamW delta finite/nonzero；
- reference/candidate initial BA/action有非零差异，更新后candidate BA/action再次有非零变化；
- 0 forbidden read、OOM、nonfinite、watchdog；
- cycle wall不超过旧96-rollout reward matched topology的`1.35x`，因为不应重复backbone或保留大图。

任一机制门失败即终止，不扫paired states、K、Nmc、LR、rank、dtype、seed或gradient scale。

## 11. Closed-loop and stability decision

机制门通过后立即用cycle1单一checkpoint运行与AS139/LPCP相同K4 correct strict paired400，逐episode同时比较
AS139、LPCP143、历史v6-fast143、old134/compiler138/online128：

- `correct <=143`、breadth<7、相对LPCP lost>10或gained<=lost：终局non-pass，不做cycle2；
- 首次`>=144`且breadth>=7、lost<=10、gained>lost、至少3 suites不降：立即补same/wrong/shuffled/reversed/
  no-video与same-task-other controls，并以新train pairs exact-resume cycle2后再次strict correct400；
- 稳定资格要求cycle1/cycle2都`>=144`、两点mean`>=145`、breadth都>=7、相邻success-set churn<=20且
  Jaccard>=`.85`，没有suite或持续task能力坍塌；
- 视频资格要求最终checkpoint same-task-other/correct>=`.9`，correct对wrong/shuffled/reversed/no-video每臂
  aggregate margin至少8，并在严格paired rows上correct-wins多于negative-wins；
- `>150`仍是更高性能目标，但也必须通过上述相邻checkpoint与视频资格；不能把一个波动151当成完成。

最终结果仍要求每个被报告点都是单一checkpoint，而非AS139/LPCP oracle union。若PCSD失败，只淘汰“LPCP carrier + query-only
commitment + paired selected-success CFM”的当前组合，不否定所有on-policy credit、memory token、dynamic K、
fresh rank8或生成LoRA后的未来task-local RL。

## 12. External hypernetwork context

本设计选择性继承两类成熟原则，而不引入其额外数据：SHINE把context中的少量memory states按layer/token双轴
变换后映射到结构化LoRA payload；Doc-to-LoRA用layer/module/rank indexed output queries和共享heads生成A/B。
V6-LPCP已有module/layer/rank routing与共享factor heads，本轮因此不再重复验证“增加输出地址容量”，而用真实
closed-loop paired credit校准最后的shared layer-aligned commitment。参考：

- https://arxiv.org/abs/2602.06358
- https://github.com/MuLabPKU/SHINE
- https://arxiv.org/abs/2602.15902
- https://github.com/SakanaAI/doc-to-lora
