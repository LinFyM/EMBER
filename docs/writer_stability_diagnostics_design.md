# Writer 训练稳定性修订诊断

状态：2026-09-21 Owner 已授权按专家最后修订执行；这是当前 active design。已完成的 E0/E1 直接复用，
原 E2 已取消且不得恢复。剩余实验只比较父状态与固定学习率的短窗响应，不产生新模型资格、checkpoint
选择或 Test 结论。

## 1. 固定边界

- 不 fresh 训练，不运行新 Test，不改变已冻结的 Writer1000 与 MT-BC300 选择，不启动 FT、RL 或外部比较。
- 复用 E0/E1 六个只读父资产和已有完整结果。短窗只从旧 Writer1200（O1200）与新 Writer1800
  （N1800）恢复完整 Writer、AdamW moments 和 step。
- 数据只使用已授权当前36个训练任务及固定诊断 demo46..49。Validation闭环只读状态、RGB、
  reward/terminal用于评测，不产生梯度；不读取Test动作、reward或状态。
- 四个短窗节点均为诊断资产，不得用于重选、延长训练、改配方或进入下游实验。

## 2. 已完成并冻结的 E0/E1

E0已验证O1200/N1800完整Writer、三组Meta、AdamW参数名、shape、finite、optimizer state和原生frame
chunk。E1已完成1728条六模型冻结动作探针及280条固定闭环，正式原始结果继续作为父节点基线。
本轮不重新运行E0/E1，也不从其局部结果选择新模型。

## 3. 原 E2 取消

取消逐任务梯度、Gram矩阵、21个独立单步与norm-matched位移副本。此前Owner暂停时留下的部分文件是
无效中断记录，不合并、不补算、不作科学结论；执行入口随修订实现一并退役。

## 4. 四臂36步短窗

| 分支 | 父状态 | 固定学习率 |
| --- | --- | ---: |
| O-H | O1200完整Writer＋AdamW | `1.6279650115e-4` |
| O-L | O1200完整Writer＋AdamW | `2.959936e-5` |
| N-H | N1800完整Writer＋AdamW | `1.6279650115e-4` |
| N-L | N1800完整Writer＋AdamW | `2.959936e-5` |

四臂各执行当前36任务采样事件1801..1836，共36次更新、四个完整平衡周期。每更新4 tasks；每task
主监督21个跨episode queries、同视频教学7个queries，教学权重1/3；一次全局clip和一次AdamW更新。
父step和moments保留，固定LR不被原scheduler覆盖。

只保存local36。不得保存或评测local0/local18，也没有local72。四臂相互独立；GPU不足时只降低并发，
不得改事件、拆分一个分支或重算已取消的E2梯度。

## 5. local36终点动作探针

每臂只运行36条compact probe：36个训练任务各一条，teacher46→query47、query episode 25%位置、
第一个固定noise seed。筛选在model forward前完成。父基线从E1完整表中按相同key只读抽取，不重新forward。
每条保存主FM、tau=1 endpoint、真实10步flow采样误差与相对父节点的前5步动作差；四臂共144条。

## 6. local36终点闭环与轨迹

每臂只评以下四个Validation任务，每任务state0..3、teacher demo0..3，并复用既定env/policy RNG：

- Spatial/3（global 3）
- Object/1（global 11）
- Goal/6（global 26）
- Long/1（global 31）

每臂16条，四臂共64条。Goal/3（global 23）只从本终点面板移除，不改E1历史结果。每条保存成功、
控制步数、目标谓词、每次replan的8维原始state与50×7归一化action chunk。预登记每个分支仅
Spatial/3-state0和Long/1-state0额外保存完整双相机processed observations，因此总计8条带图像轨迹；
其余56条不保存图像张量。

另从既有E1轨迹只读导出16条轻量记录，不做新rollout或model forward：O1200和N1800 × Spatial/3和
Long/1 × state0..3。采用固定控制步采样，保存小尺寸压缩视频/图像与对应action记录。

## 7. 产物、裁决与生命周期

正式根为`/data0/user/ymdai/ember_runs/writer_stability_diagnostics_20260921`。新增轻量结果包括四臂
training steps、实际task events、local36 compact probes、local36闭环rows、轨迹索引、16条既有轨迹轻量包
和completion。checkpoint与完整轨迹仅本地保留；上传包不含大权重。

执行agent只验收合同、exit和完整行数，不撰写因果归因，也不决定哪一分支继续。四臂结束后不自动训练、
重选模型或启动Test/FT/RL，由Owner与专家依据完整证据裁决。

唯一closed-loop实现仍是official `rollout_shard`。诊断捕获只在其现有occupancy接口增加向后兼容的
compact/full条件：旧调用默认full行为不变，修订终点面板由contract指定少量full conditions。诊断结束后，
detached runtime在确认无进程引用时删除；显式诊断入口与轻量复算材料保留到最终研究历史吸收结论后再清理。
