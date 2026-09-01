# EMBER temporary handoff

本文件是一次真实跨session交接的临时索引。新session完整消费并确认持久文档足以接手后，应删除本文件；任何长期要求不得只保存在这里。

## 1. 当前状态

- canonical仓库：`/data1/user/ymdai/projects/EMBER`
- canonical branch：`main`
- 科学前驱：`a185fe223d1ef77635d83696c3e164a48520edbf`
- 本次整理后的准确HEAD必须用`git rev-parse HEAD`和远程`main`现场核对，不要从本文件猜测。
- 当前无active goal、无active实现分支、无训练/评测/GPU任务。
- 全新专家正在审查锁定的`main@a185fe2`及完整历史；owner尚未返回专家意见。
- 在owner给出专家原文并明确授权前，只做阅读、事实核对和解释，不创建新架构或启动GPU。

## 2. 必读顺序

先完整阅读`AGENTS.md`，再严格按其mandatory reading顺序读到EOF：

1. `docs/current_owner_requirements.md`
2. `task_plan.md`
3. `findings.md`
4. `progress.md`
5. `docs/concept.md`
6. `docs/research_history.md`
7. `docs/expert_review_20260824_native_factor.md`
8. `docs/event_conditioned_policy_compiler_design.md`

随后完整阅读2026-08-26、08-28、08-29、08-30、08-31、09-01的六份后续`docs/expert_review_*.md`。以
`docs/research_history.md`为旧架构索引，只在核对精确实现/公式/结果时查看对应Git提交；不得恢复旧提交的“active/current/next”。

## 3. 一句话科学状态

G1证明真实native X/Y与signed pooling存在强task-local rank4解，G2证明Natural Program保留视频动态；G3尚未学得可泛化的
Program--bank功能映射。最新链已排除free summary容量、query/anchor under-travel和若干坐标混杂，最终只停止
`summary -> family-scalar gate -> shared event-additive anchor`这一parameterization，未根本否定ECP。

精确结果、formal roots和停止边界见`progress.md`顶部及`docs/research_history.md`第91--96节。

## 4. 等待中的专家裁决

owner已要求全新专家从最早EMBER到当前ECP逐架构复核，并明确判断：

- 是否继续ECP并更换Program--bank联合方向接口；
- 是否保留部分Native-Factor证据但开启新Writer分支；
- 是否直接做整体端到端Writer训练；
- 现有证据是否接近根本停止条件。

专家prompt同时要求给出唯一推荐路线、张量与梯度接口、最小决定性实验、失败边界、Final fully-random Writer、最小loss、Action Meta和吞吐建议。
收到回复后必须先逐字落盘，再核对其引用与当前authority；专家意见不是自动生效的active design。

## 5. owner口头要求的持久入口

以下要求已写入`docs/current_owner_requirements.md`第6--8节和`AGENTS.md`，这里只给索引：

- 根因优先，non-pass后先做机制分析，不盲目迭代或小扫；
- 代码通过最小真实smoke后立即推进科学实验，文档/清理/非必要合同移到等待期间；
- gpu01与gpu02都可用；每次live检查，卡数1--6弹性，不固定两卡；大量空闲时EMBER总量最多8；
- 优先空闲卡，必要时可与低显存/低util进程安全共驻，但不得干扰、抢占、kill或reset；
- 单卡也要提高真实SM/UTL、memory UTL与吞吐，不设35GB等人为显存上限，只保留真实最长样本和共驻安全余量；
- 自设throughput阈值不合理时应修订/删除，但训练规模与墙钟明显不相称时必须先优化执行结构；
- subagent只在真正可并行并显著缩短关键路径时使用；
- G1--G3分段只为机制验证，Final保留component-init和fully-random Writer的matched fresh端到端候选，loss保持最小充分；
- Action Meta默认关闭，仅在base Writer已有闭环增量且剩余误差稳定集中于action-in/out后做一次matched on/off；
- 未经owner当次允许不得联系专家；只有owner明确要求才创建goal。

## 6. workspace状态

- 只保留canonical主worktree；没有local `codex/*` branch。
- `.codex/tmp`为空；约5.1GB旧临时diagnostics/cache已清理。
- formal evidence、checkpoints、raw rows、aggregate、dataset、models和authoritative caches均保留。
- tracked旧configs/代码尚未因等待专家而提前删除；它们是审计与复用表面，不代表active路线。
- 历史vector-interaction分支仍可从`origin/codex/g3-vector-interaction@2295f48`读取。

## 7. 新session第一组动作

1. 完成上述阅读并核对`git status`、HEAD、`origin/main`和远程实际main；
2. 确认没有运行中的EMBER进程，不依据旧GPU记录做调度；
3. 等待owner粘贴专家原始回复；不要要求owner重新解释已有历史；
4. 收到后保存原文、核对事实、用通俗语言汇报核心裁决与分歧；
5. 只有owner确认路线后才更新active design、建立唯一`codex/*` worktree并进入实现；
6. 完成交接后删除`HANDOFF.md`，确认其中没有独占长期信息。
