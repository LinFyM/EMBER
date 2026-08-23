# ECP Gate A2后续数据路线决策审计

日期：2026-08-24。状态：**task65/68 Gate A3 formal已完成并以`28/50、9/50` non-pass；旧shared-realizer与phase-composed teacher均保持关闭。**

## 1. 现在真正要决定的事

Gate A2把当前scene3 soup/butter pair的失败缩小到了一个具体数据问题：在soup已进tray的
occupancy上，butter task expert没有恢复能力。phase experts将另一向从19提高到44，证明切换机制有效；
`soup -> butter=0/50`证明继续在当前pair上推进已不是“训练一个expert”，而是首先要创造目前不存在的
soup-first successful action data。

因此实质选择是：

1. 为soup/butter开发新的privileged planner、人工demonstration或task-local RL，获得soup-first labels；
2. 换到scene/init一致、primitive experts可靠、但composite goal与两个orders都source-unseen的process family，
   用phase switching重新采集两个方向的canonical demonstrations。

## 2. 为什么不推荐先救soup/butter

- A2只产生44条butter-first successful actions，soup-first为0；失败轨迹不能作为正确SFT label。
- 现有task-expert trainer可高效训练rank16 LoRA，但它读取的前提是50条successful action HDF5；它不能创造缺失的actions。
- 已有outer-credit实现作用于Writer的低维code，不是PI0.5 task-local policy LoRA RL。PI0.5 flow action chunk也没有
  可直接复用的离散action log-prob policy-gradient路径。为了取得teacher先发明一套RL方法，会把数据Gate变成新的
  主研究项目。
- robosuite有human demonstration通路，但仓库没有可直接复用的自动pick-place planner。若专家坚持保留该family，
  需要的新authority应明确写成“至少20条通过temporal wrapper的soup-first真实single-episode demonstrations”，
  而不是再试step2000、更长horizon或新LoRA组合。

## 3. 现有source composite demonstrations证明了什么

本轮没有执行actions或加载模型；只在stored simulator states上重算每个BDDL goal predicate的首次rising step。
结果如下：

| source task | 50 demos的事件顺序 | step1000 expert | 用途 |
| --- | --- | ---: | --- |
| task1 drawer-close + bowl-on-top | 50/50 close→place | 50/50 | 独立fixture/object候选 |
| task21 stove-on + pan-on-stove | 50/50 turnon→place | 43/50 | 后续异构动作family候选 |
| task23 bottom-close + top-open | 50/50 close→open | 28/50 | 两个fixture control候选，expert较弱 |
| task45 stove-on + pan-on-stove | 50/50 turnon→place | 46/50 | task21的跨scene复现 |
| task63 left-on-right + stack-in-tray | 50/50 stack→tray | 44/50 | 物理耦合suite候选 |
| task64 right-on-left + stack-in-tray | 50/50 stack→tray | 43/50 | 物理耦合suite候选 |

task38的50条demos中stove在step0已全部turn-on，不构成两事件order pair。完整结构证据见
`docs/evidence/ecp_20260824/ecp_process_candidate_demo_order_audit_20260824.json`。

## 4. 推荐的下一个最小family

首选`LIVING_ROOM_SCENE5` task65/task68 separate-plate pair：

- task65和task68的fixtures、objects和init specification逐字相同；
- 两个variants的终点都是`On(red mug,left plate) AND On(yellow-white mug,right plate)`；
- variant A要求`red mug -> yellow-white mug`，variant B要求反向；
- 两个纯primitive step1000 experts分别为`43/50、47/50`，它们的goal objects和goal plates都互不共享；
- source中没有这两个primitives的composite task，因此两种composite orders都是source-unseen，比stove/pan“一种顺序
  已在source composite中出现”更干净；
- 可直接用task65/68 experts做phase switching，复用现有temporal wrapper/collector，无需先训新expert；
- 它保留原soup/butter Gate想检验的video-specified order，却去掉shared tray occupancy这个已知teacher confound。

同scene的task66/task67具有第二组互不共享的mug/plate goals，experts为`42/50、49/50`，可作为同机制复现。
这仍只是最小“视频指定顺序”可行性，不应被写成general physical process understanding。过Gate A/B后，
再加task21/45 stove/pan异构动作family，之后加task64的`stack -> tray`与`tray -> stack`物理耦合family，
才扩成family-disjoint process suite。

现有source HDF5的RGB是128×128，而process-meta canonical collector是render256且stride5。所以source videos只作顺序和expert
证据，不直接重标为public teacher videos。选定family后，两个variants都必须由phase experts在同一temporal wrapper、
配对states/noise下重新rollout，再产生统一goal-only exact language的256分辨率action-hidden videos。

## 5. realizer侧的同步裁决

fit90 learned realizer和fixed two-sided coordinate已经分别在strict250得到`33/37`与`80`，且Goal/Long均为0。
专家原合同规定两种principled coordinates都失败后应停止当前mobile-rank4 shared-realizer family。
因此新process mappings不启动任何旧realizer新版，也不把“数据更多”自动解释为复活当前coordinate。task65/68只先补足
process-identifying teacher与observer证据；Gate通过后需要按专家合同重新建立deployment-compatible Program-to-effect桥，
而不是恢复已经失败的balanced-SVD learned map或fixed two-sided coordinate。

## 6. 已冻结的执行裁决

重新对照专家完整合同并由owner确认后，三个问题已有一致答案：

1. 不再为soup/butter先开发新的teacher algorithm；task65/68作为下一个最小双顺序feasibility；
2. source见过两个primitive不影响其conjunctive goal与required orders属于source-unseen mappings；该Gate只声称
   video-specified order feasibility，不声称object skill本身未见；
3. 新process mappings是后续Program识别的必要数据，不是旧shared-realizer family的复活条件。两种principled coordinates
   已按专家停止合同关闭；Gate通过后另建deployment-compatible bridge。

预注册执行合同见`docs/ecp_process_separate_plates_teacher_gate_20260824.md`。
