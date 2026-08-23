# ECP separate-plates process teacher Gate A3

日期：2026-08-24。状态：**formal完成，Gate A3 non-pass。**

## 问题与唯一变量

scene3 soup/butter在source-phase Gate A和phase-expert Gate A2中分别得到`0/19`与`0/44`。A2证明phase切换真实有效，
但soup占据共享tray后，butter primitive expert没有可靠恢复支持。当前family因此关闭，不再延长horizon、挑state、续训expert
或修改predicate。

本Gate只把process family换成`LIVING_ROOM_SCENE5`的两个互不干扰plate goals：

- `red_left`：LIBERO-90 task65，`put the red mug on the left plate`，step1000 expert为`43/50`；
- `yellow_white_right`：LIBERO-90 task68，`put the yellow and white mug on the right plate`，step1000 expert为`47/50`。

两者scene、fixtures、objects与init specification相同，目标mug和plate均不共享；source policy只见过两个primitive mappings，
没有见过二者的conjunctive goal或任一required order。因而本Gate检验的是source-unseen composite/order mapping，不把
source-seen primitive误写成source-unseen object skill。

authority：`configs/pi05_ecp_process_meta_separate_plates_v1/manifest.json`。

## 两个variants

两个variants严格共享：

- exact goal-only language：`put the red mug on the left plate and the yellow and white mug on the right plate`；
- 同一个custom BDDL、task65的50个固定init states、environment seed与policy-noise schedule；
- 最终谓词：`On(red_coffee_mug_1, plate_1) AND On(white_yellow_mug_1, plate_2)`；
- render256、frame stride5、10 flow steps、执行前5 actions后replan、strict horizon400。

唯一差别是temporal wrapper的required order：

1. `red_left_then_yellow_white_right`；
2. `yellow_white_right_then_red_left`。

每条轨迹必须是phase experts在该composite环境中重新rollout得到的真实、完整、单episode轨迹；不得拼接或重标已有128分辨率
primitive videos。phase改变时丢弃旧action chunk、切换既有rank16 LoRA并立即重新规划。

## 信息墙

task ID、phase language、required order、predicate transitions、teacher actions、expert weights、reward与success只进入privileged
ledger。公开video只保存统一exact language、双相机RGB、有序帧、source step index与固定stride；两个variants不能通过filename、
task ID、language或metadata被Writer区分。phase expert是数据采集teacher，不是deployment adapter。

## Smoke与formal面板

先对两个variants各运行固定state0一次，只验证：

- composite BDDL和两个predicate可执行；
- 两个LoRA都在同一episode中被实际安装；
- phase切换后重新规划；
- privileged ledger与公开video字段符合信息墙。

smoke不用于选择state、checkpoint、horizon或Gate阈值。若两向均完成工程通路，再从clean pushed detached commit一次运行
100行formal面板：两个variants各state`0..49`。同state两向使用同一task65 noise key，因此environment与policy-noise
common prefix配对。单节点使用当时真正提高吞吐的最多六张GPU，禁止gpu01 physical0。

双向state0 smoke已在gpu02 physical0/1并行完成：

- red→yellow-white在step`144/291`完成两个events；
- yellow-white→red在step`94/374`完成两个events；
- 两行均success、invalid=false，并实际安装task65与68两个experts；
- 两条公开video只有预注册六个字段，语言完全相同，分辨率为256；
- 两向policy-noise的59个共同replans逐项一致。

这只证明链路与双向可执行性，没有改变100行面板或Gate。

## Gate A3

同时满足才通过teacher acquisition：

1. 两个variants各至少`20/50` strict temporal successes；
2. 总成功至少`50/100`；
3. 每个成功episode按required order产生两个唯一rising transitions，最终两个predicate同时为真，wrong-first invalid为0；
4. 100份privileged ledgers唯一完整，公开video数量等于success数且没有privileged字段；
5. 两方向50个state IDs完整，配对state的environment seed与policy-noise common prefix一致；
6. 成功轨迹中两个phase experts均被实际调用，不能由单一expert或预先成立的predicate冒充双事件teacher。

通过只授权Gate B：用冻结observer检查correct video相对sibling wrong-order、language/no-video与first+final的process信息增量；
不自动重开已经失败的balanced-SVD learned map或fixed two-sided shared-realizer coordinates，也不直接启动`q_pi/q_V`。

失败则关闭本task65/68 phase-composed acquisition，不做step2000、延长horizon、挑state、改seed或predicate救援。先按raw ledgers
定位是某一primitive在另一目标已完成后的恢复失败，还是两方向普遍不支持；之后才在已登记的task66/67同机制复现、真正
composite privileged data或异构process family之间作下一次因果选择。

## Formal结果与裁决

clean pushed detached `4bf50394f75307568339143d17a39c0bfe2c2829`在gpu01 physical`1,2,3,4,5,7`以六个固定shards
完成100行，六个worker返回码均为0；physical0为Prohibited且未使用。

- red→yellow-white：`28/50`；第一event完成`43/50`，第二event完成`30/50`；
- yellow-white→red：`9/50`；第一event完成`46/50`，第二event完成`9/50`；
- 总计`37/100`，低于50；反向也低于每方向20的floor；
- invalid为0，phase/expert错配为0，100份ledgers与50对state/noise完整；
- 37条公开video恰好对应37个success，字段和统一language均无泄漏。

这不是horizon-only失败：yellow-white→red的37个second-phase失败在第一event后仍剩余`229..322`步，median`297`。因此最早
失效接口是primitive task expert在sibling goal已经改变场景后的恢复支持；目标物体和plates物理分离仍未让phase composition
双向可靠。Gate A3判为non-pass，task65/68 phase-composed acquisition关闭，不运行Gate B，也不以task66/67同机制full formal、
step2000或合同调整盲目续试。下一数据机制必须提供真正order-specific composite privileged policy/data；本轮37条完整成功轨迹
只作bootstrap evidence，不能冒充通过的teacher bank。

正式证据：`docs/evidence/ecp_20260824/ecp_process_separate_plates_teacher_gate_20260824.json`。
