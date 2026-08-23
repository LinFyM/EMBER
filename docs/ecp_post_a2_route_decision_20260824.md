# ECP Gate A2后续数据路线决策审计

日期：2026-08-24。状态：**只读审计完成，已形成推荐；未选定或实现successor family，未启动GPU训练。**

## 1. 现在真正要决定的事

Gate A2把当前scene3 soup/butter pair的失败缩小到了一个具体数据问题：在soup已进tray的
occupancy上，butter task expert没有恢复能力。phase experts将另一向从19提高到44，证明切换机制有效；
`soup -> butter=0/50`证明继续在当前pair上推进已不是“训练一个expert”，而是首先要创造目前不存在的
soup-first successful action data。

因此实质选择是：

1. 为soup/butter开发新的privileged planner、人工demonstration或task-local RL，获得soup-first labels；
2. 换到已有可靠composite demonstrations和task experts、但反向procedure仍source-unseen的process family。

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
| task21 stove-on + pan-on-stove | 50/50 turnon→place | 43/50 | **最佳首个替代pair** |
| task23 bottom-close + top-open | 50/50 close→open | 28/50 | 两个fixture control候选，expert较弱 |
| task45 stove-on + pan-on-stove | 50/50 turnon→place | 46/50 | task21的跨scene复现 |
| task63 left-on-right + stack-in-tray | 50/50 stack→tray | 44/50 | 物理耦合suite候选 |
| task64 right-on-left + stack-in-tray | 50/50 stack→tray | 43/50 | 物理耦合suite候选 |

task38的50条demos中stove在step0已全部turn-on，不构成两事件order pair。完整结构证据见
`docs/evidence/ecp_20260824/ecp_process_candidate_demo_order_audit_20260824.json`。

## 4. 推荐的下一个最小family

首选`KITCHEN_SCENE3` stove/pan pair：

- 统一public language只陈述goal，不用含明确顺序的句式；
- 两个variants的终点都是`Turnon(stove) AND On(pan,stove)`；
- variant A要求`turnon -> pan`，variant B要求`pan -> turnon`；
- task21的50条source demonstrations都是A顺序，step1000 expert为43/50；task18同scene pan primitive expert为50/50；
- 可用task21 expert执行stove phase、task18 expert执行pan phase，复用现有temporal wrapper/collector，无需先训新expert；
- 两事event不共享一个容器内的放置空间，不会重复soup先改变tray occupancy的已知失败。

这仍是最小“视频指定顺序”可行性，不应被写成general physical process understanding。若Scene3双向teacher过Gate A，
再用task45的Scene9复现；之后把task64的`stack -> tray`与`tray -> stack`作为物理耦合family，才扩成
family-disjoint process suite。

## 5. realizer侧的同步裁决

fit90 learned realizer和fixed two-sided coordinate已经分别在strict250得到`33/37`与`80`，且Goal/Long均为0。
专家原合同规定两种principled coordinates都失败后应停止当前mobile-rank4 shared-realizer family。
因此新process mappings在获得前不启动任何realizer新版；即使获得，是否足以重开shared realizer，也应由专家明确
裁决，不把“数据更多”自动解释为复活当前coordinate。

## 6. 需要专家回答的三个核心问题

1. A2之后，是否同意不再为soup/butter先开发新teacher algorithm，而把stove/pan反向顺序pair作为下一个
   minimal feasibility？
2. 现有source composite demonstrations可否在不改video内容的前提下，用统一goal-only neutral language作为process-meta
   teacher videos；还是两个variants都必须由新privileged teacher从头rollout？
3. 在两种principled realizer coordinates均non-pass后，新process mappings是否构成重新定义Program-to-effect识别问题的
   充分新证据，还是应直接放弃当前shared-realizer family并重设部署桥？

## 7. 可以给专家的最短prompt

> 我们按你的Gate补做了phase task-local experts：soup→butter仍为0/50，butter→soup从19/50提高到44/50，
> 所以切换有效，但当前pair的soup-first successful action data完全不存在。另外只读重放发现，LIBERO-90
> task21/45共100条stove+pan demonstrations全部严格为turnon→place，两个experts为43/50和46/50，同scene
> pan primitive expert为50/50。我们因此倾向停止为soup/butter发明新teacher，改用统一neutral language、同终点的
> stove/pan双顺序pair做下一个feasibility，通过后再加stack/tray这类物理耦合family。请你核心裁决：
> 这个数据路线是否正确；既有composite videos能否重标为neutral goal language；以及在learned和two-sided realizer都失败后，
> 新process mappings是否足以重开shared realizer，还是应重设Program-to-effect部署桥。
