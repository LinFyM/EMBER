# ECP process pair phase-expert teacher Gate A2

日期：2026-08-24。状态：**formal完成，Gate A2 non-pass。**

## 问题与唯一变量

首个process pair的Gate A中，冻结source PI0.5在predicate phase切换后分别读取task55/56 primitive language，得到
`soup -> butter = 0/50`、`butter -> soup = 19/50`。两个方向第一事件均为`50/50`，失败集中在第二事件支持。

本卡只把每个phase的执行policy从shared source换成已经formal固定、各自在原primitive任务上`50/50`的step1000
rank16 task-local expert：

- alphabet soup phase：LIBERO-90 task55 expert；
- butter phase：LIBERO-90 task56 expert。

同一composite episode仍由同一个privileged phase scheduler执行；phase改变时丢弃旧action chunk、切换既有LoRA并重新规划。
不训练新expert，不选择checkpoint，不改变BDDL、exact language、init states、predicate、horizon、seed、noise、replan或成功定义。
旧Gate A结果保持有效，不重命名也不覆盖。

authority：`configs/pi05_ecp_process_meta_expert_teacher_v1/manifest.json`。

## 信息墙

task-local expert weights、task ID、phase language、required order、predicate、actions、reward与success全部是teacher acquisition的
privileged信息。公开video仍只有统一exact language、双相机RGB、原始有序帧与固定stride；每条video来自一个真实single episode，
不拼接primitive视频。该teacher不是deployment Writer或第二adapter。

## Smoke与formal面板

先用两个variant各一个固定state做工程smoke，只验证LoRA切换、环境、ledger与public artifact通路，不据此改合同或选state。

smoke已在gpu02 physical0/1并行完成，两个进程均正常退出。state0的`butter -> soup`在step296完成，`soup -> butter`
在step97完成第一事件后于strict400失败；两份ledger都记录到task55与56两个expert ID，成功video仍只含六个公开字段。这个
结果只证明切换链有效，不改变下面100行Gate或state面板。

smoke有效后，从clean pushed detached authority一次运行原`100`行面板：两个variant各state `0..49`，strict horizon `400`；
同state两方向的environment seed与policy-noise schedule严格配对。单节点最多六个persistent workers，按当时live GPU选择，不使用
gpu01 physical0；每个worker只加载一次source policy和两个小LoRA states。

## Gate

同时满足才通过teacher acquisition：

1. 两个variant各至少`20/50` strict temporal successes；
2. 总成功至少`50/100`；
3. 每个成功episode的两个transition顺序正确、最终两个predicate同时为真、无wrong-first invalid；
4. 100个privileged ledgers唯一完整，公开video数量等于success数且不含privileged字段；
5. 两方向50个state IDs以及配对seed/noise common prefix完整一致。

通过只授权原Gate B的视频顺序可识别性与闭环必要性诊断；不自动扩process suite，也不启动`q_pi/q_V`。失败则关闭当前
`scene3 soup/butter + phase-composed primitive experts` acquisition，不用step2000、延长horizon、挑state或改predicate救援；下一步
回到专家判断是训练真正composite privileged expert，还是选择物理机制不同的替代family。

## Formal结果与裁决

clean pushed detached `24c5bdc`在gpu01 physical`1,2,3,4,5,7`完成100行；六个worker全部返回0，100份ledger、50/50
paired state IDs和policy-noise common prefix均完整。公开video共44条，只含预注册六个字段，成功episode的event order与最终
predicate均正确。

- `alphabet_soup -> butter`：`0/50`；50行均完成soup（step97--245），没有一行完成butter；
- `butter -> alphabet_soup`：`44/50`；50行均完成butter，44行在step245--400完成soup；
- 总计`44/100`，低于50；第一方向也低于20。

相对原source-phase teacher的`0/19`，task-local experts把butter-first提高到44，却没有改变soup-first的0。由此排除adapter
未加载、phase route错误、共享source容量和第一事件执行；当前family的剩余失败是soup已占据tray后task56 primitive不能恢复。
Gate A2因此non-pass，Gate B、suite扩展、`q_pi/q_V`均不启动。当前family对phase-composed source/task-local primitive teacher
关闭；下一决策回到专家选择true composite privileged expert还是不同物理机制的source-unseen family。

正式证据：`docs/evidence/ecp_20260824/ecp_process_phase_expert_teacher_gate_20260824.json`。
