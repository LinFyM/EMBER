# EMBER Progress

更新时间：2026-08-17。本文只记录当前工作进度和可执行状态；完成的科研结果进入`docs/research_history.md`，跨实验
结论进入`findings.md`。

## Current state

- 本轮“全面整理仓库并重建EMBER认知”goal已完成；本轮没有启动新架构或GPU实验。
- canonical workspace为`/data1/user/ymdai/projects/EMBER`，主写分支为`codex/bci-continuation`。
- 整理起点HEAD为`8553b613de7791df50e0f3ef85678fcaca1cac0c`；交付状态以包含本文的当前分支tip为准。
- 当前没有active design、GPU run或可resume successor checkpoint。
- 当前协作边界：暂时不使用subagents。

## Completed in this cleanup

- 完整检查Git、两节点进程和独立存储quota；未发现EMBER GPU进程。
- 删除103个历史worktree checkout，仅保留主worktree；五个Dynamic-K scratch branches经patch-equivalence确认内容
  已进入当前HEAD后删除，本地只保留`main`和主工作分支。
- 清除两节点已死EMBER tmux sessions，未触碰他人进程。
- 删除退役Expert-Manifold Writer、v6-prior、policy-innovation、condition-update和reward-credit-gate可执行路径。
- 将task-expert package收缩为task-local expert训练/评测；历史package/schema名因artifact兼容保留。
- evaluator收缩为当前Dynamic-K/LPCP/GOMQ运行面及静态历史分析。
- 删除退役rank8/rank128、旧reward successor configs和对应测试；保留固定数据、source、rank16、task expert、
  LPCP和terminal GOMQ authorities。
- GOMQ config标记为`terminal_nonpass`：checkpoint仍可评测，训练入口拒绝继续执行。
- 相关定向CPU测试：`119 passed`。
- `AGENTS.md`已收缩为稳定项目总览和原则，不再存放动态实验记录。
- 删除持久handoff与重复execution brief，恢复`task_plan.md`、`findings.md`、`progress.md`三文件工作状态结构。
- README、owner requirements、concept、research history和findings已重写；新增非authority的逐步设计推理文档。
- 退役per-design文档已合并进统一ledger，只保留六份高价值结构锚点；精确旧内容仍可由Git检索。
- 已清除项目本地uv/HF cache、bytecode、pytest cache、egg-info和`.codex/tmp`，约释放9.5 GiB；formal
  runs/data/evidence/models未触碰。
- 清理后fresh全量CPU测试为`311 passed`；canonical entrypoints `--help`、compile、20份JSON parse、内部import、
  17份Markdown本地链接、retained sealed checksums、删除路径反向引用与Git diff检查均通过。

## Cleanup result

- active文档从约19,389行收敛到3,569行；历史精确语料仍由整理起点Git commit可恢复。
- active tree只保留六份深结构设计锚点和一个统一research ledger，不再维护几十份互相竞争的“active”设计。
- 共删除112个退役或重复tracked paths；其中111个代码/config/doc/hash路径均无当前反向引用。正式runs、checkpoints、
  data、evidence和models未删。
- 验证重新生成的bytecode和pytest cache已再次清除；项目根不存在持久handoff。

## Next decision boundary

当前停在“无active design、无active run、无可resume successor”的清晰边界。下一科研阶段先使用现有formal artifacts
回答`docs/architecture_reasoning.md`登记的stage-wise证据问题，判断最早缺口究竟在Program形成、native compiler、
reward direction还是shared retention；证据指向单一接口后，才建立新的单变量design authority。memory token、
rank16、V6 compiler和MCPS都仍是候选部件，不是预设答案。

## Scientific boundary carried into cleanup

- v6-fast仍是有完整五臂的历史最好：`143/135/125/128/129`。
- LPCP是当前最强可复用carrier baseline：K4 correct=`143/400`、breadth7。
- SFMC单点`144/400`但lost15/churn31且无六臂，不具稳定资格。
- GOMQ第一步真实learned-memory update=`151/400`，相邻updates=`135/400`、`131/400`；它证明memory query有
  closed-loop价值，但当前shared direct-B tail不能稳定保留held support，且六臂未测。
- 当前最早未解决的宏观接口是：跨video可复现、跨task可分的高层Program，如何通过policy-effective compiler和
  shared credit在同一checkpoint中稳定共存。

## Storage snapshot

本轮清理前观测到`/data0`个人quota约21.1 GiB / 1 TiB，`/data1`约542.2 GiB / 1 TiB；主要保留量为`runs/`约
410 GiB、`data/`约94 GiB和`.venv/`约593 MiB，项目本地约9.5 GiB cache已清除。该观测会漂移；任何新大copy/cache/
training前必须重新检查。
