# ECP process-identifying 最小顺序对 Gate

## 目的

本 Gate 在任何新 `q_pi`、`q_V` 或 joint Writer 训练前，检验一个真正的
same-language/same-scene/same-init/same-final-predicate、opposite-required-order 任务对是否成立。
它是 process-identifying 数据资格与 privileged upper bound，不是最终部署方法。

首个 family 固定为 `LIVING_ROOM_SCENE3` 中 alphabet soup 与 butter 放入 wooden tray：

- exact language：`put the alphabet soup and butter in the tray`；
- 两个 variant 共用同一 tracked BDDL、task55 的同一 50 个 fixed init states、同一 environment seed 和 policy-noise schedule；
- 最终谓词均为 `In(soup,tray) AND In(butter,tray)`；
- 唯一差异是环境内部要求 `soup -> butter` 或 `butter -> soup`；
- wrong-first 永久 invalid，之后即使达到相同终点也不能成功。

authority：`configs/pi05_ecp_process_meta_v1/manifest.json`。

## 信息墙

privileged teacher 可根据当前 predicate phase，让冻结 source PI0.5 分别读取已有 task55/56 的 primitive language；phase 切换后
立即丢弃旧 action chunk 的剩余动作并重新规划。这个 phase language、variant、predicate、actions、reward 与 success 只进入
privileged ledger。

公开 teacher video 只含：

- 统一 exact language；
- 一条真实闭环 episode 的双相机 RGB 帧；
- 原始逐步帧序列，后续模型输入固定 stride 5。

不把两条 primitive 视频拼接成 composite video，不向 Writer 传 filename、variant/task ID、predicate、phase、action、reward、
success 或 init state。teacher video 与后续执行 episode必须跨 episode。

## 已完成工程 smoke

`ff1e1a9`上只用state0各跑一个方向，不计入科学 Gate：

- `butter -> soup`：成功，两个事件分别在step104与350完成；
- `soup -> butter`：step110完成soup，horizon400内未完成butter；
- 两个进程都完成custom BDDL加载、predicate wrapper、source PI0.5 phase switch和privileged/public artifact写出；
- 首轮大写`In`在rollout前被LIBERO runtime拒绝，已按其真实注册键修为小写`in`，没有产生科学结果或改变任务。

## Gate A：teacher acquisition

从下一clean pushed commit的detached frozen worktree运行一次完整100-episode panel：两个variant各用state `0..49`，horizon400，
不挑state、不延长失败episode、不重跑科学失败。六个persistent workers各自只加载一次policy，三张GPU处理一个方向，另外三张处理
反方向；physical0禁止使用。

同时满足以下条件才进入视频可识别性 Gate：

1. 两个variant均至少`20/50` strict temporal success；
2. 总成功至少`50/100`，每个方向至少保留20条完整action-hidden teacher videos；
3. 成功episode的两个predicate transition顺序与variant完全一致，最终两个predicate同时为真；
4. 两方向的50行init-state IDs、environment seed和policy-noise schedule严格同面板；
5. 公开video文件不含privileged字段，privileged ledger与公开video一一对应。

若失败，只定位是哪个方向、哪一阶段或final-state保持失败；不改horizon、seed、phase language、predicate阈值或挑state救本family。
它不会否定process数据路线，但在进入新模型训练前必须换成一个有明确依据的候选family或重新取得更强privileged expert。

## Gate B：视频可识别性与闭环必要性

Gate A通过后，固定state `0..29`为task-local diagnostic train、`30..49`为held；同一state两variant只处于同一role。训练一个
冻结PI0.5视觉特征上的小型有序视频probe，仅预测两个required orders，不生成LoRA，也不进入后续方法。

- full video held order accuracy至少`32/40`；
- first+final held accuracy不高于`26/40`，且full至少高20个百分点；
- language-only/no-video输入在两个variant间完全相同，只允许固定balanced prior，不训练task/filename route；
- probe只读取统一language与公开RGB帧，不能读取文件名、state、variant或privileged ledger字段。

随后用held video预测出的order驱动同一个privileged phase-switched source expert，在跨episode执行states上做correct/sibling-wrong/
first+final配对闭环。full-correct必须两方向均非零、总计至少`16/40`，sibling-wrong至多`2/40`，且correct相对wrong形成
显著配对优势；first+final不能达到full的闭环水平。环境wrapper、seed和policy noise严格配对。

Gate B只证明“action-hidden有序视频包含必要顺序，且该顺序能指导同一个privileged执行器”；它不证明最终Stage 0 Program、
distributional `q_pi`、deployment realizer或`q_V`已经成立。

## 裁决后顺序

- Gate A/B都通过：扩展family-disjoint process-meta suite，同时继续Phase 2 realizer coordinate calibration；
- Gate A通过、B失败：先定位endpoint leakage或video observer不可识别，不启动`q_pi/q_V`；
- Gate A失败：停止本pair的数据扩展，不以Writer或solver调参掩盖teacher/data acquisition失败。
