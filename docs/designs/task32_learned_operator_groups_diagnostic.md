# task32已学控制修正的执行投影分组诊断

2026-10-02登记。仅冻结train案例的有限机制分析，不训练、不选checkpoint、不产生EMBER正式分数。
活动状态看progress；本合同不恢复旧T训练、当前条件900、prefix候选或任何历史未完成清单。

## 1. 解释对象与判别作用

机制§99直接确认旧T2340经过64步真实FM后，D17在init0/2/3保留开炉并新增搬壶成功；
同task的S43在init3新增成功，却在init2把父的“开炉而未放置”换成“放置而未开炉”。
因此本次有两种真实学得的完整控制变化，不能把私有D当作唯一可学上界。

main另读已有B20十步预测：task32在S/P/D下的功能修正，teacher差异能量分别只有.346%/.479%/.789%；
两D修正余弦.9844。几乎全部平均风险改善来自两teacher共有部分，但真实闭环收益仍不同。
这不证明某个投影是根因，也不将共有函数分量归给公共B0；它要求进一步弄清真实修正在哪项执行计算上发挥作用。

本项只区分：已学修正能否主要由执行Q变化承载、能否由非Q变化承载，还是必须依赖它们共同适配。
这是理解视频最终应写入什么控制作用的一项有限辨识；不据参数量、attention图、平均MSE或某个局部norm选模型。
只有真实搬壶及完整目标保留/丢失改变结论，不要求MT在这些场景先成功。

最近近邻是旧设计§28/机制§66–§67：当时删去成熟父的整组条件M，主要消费专家query的十步风险，未形成通用删减修复。
本次保留父已学功能，只拆解已经观察到真实收益的64步学习增量，并消费其原物理初态；
不是重跑原删除实验，也不是声称旧负证据已失效。结果仍不构成某组应从新架构删除或独立训练的依据。

## 2. 数学干预及其实际特征含义

旧T所有38处A在这64步学习中固定。令parent和learned的B分别为Bᵖ、Bᵃ，学习增量为DeltaB=Bᵃ−Bᵖ。
将18处action-expert `q_proj`定义为Q组；其余18处`v_proj`及`action_in_proj/action_out_proj`为R组。
按完整target名称分类，保存显式18/20清单；不是挑选某一层、head、rank或幅度。

两套新增完整LoRA只选择原有因子，不作数值求解或缩放：

```
A_Q = A_R = A_parent
B_Q[target] = B_learned[target] if target in Q else B_parent[target]
B_R[target] = B_parent[target]  if target in Q else B_learned[target]
```

每套始终是完整38-target单LoRA，source/public来源、rank/alpha和normalization保持旧合同。
直接选取已有B张量，避免无必要的相减再相加；检查固定A/shape/finite及实际消费身份即可，不加hash/大范围逐tensor审计。
不运行Writer/native，不读取teacher actions/state，不拟合新参数，也不改变源policy计算。

在真实attention中，`Q=(Wq+Bq A_q)h`经RoPE和softmax改变对真实prefix及suffix的读取；
`V_suffix=(Wv+Bv A_v)h`改变action-expert suffix内容。prefix图像/语言/state的K/V不属于这38个LoRA targets。
非Q组还改变动作嵌入和末端速度投影；早层V也会改变后层h/Q。
因此Q与R不能分别叫作纯感知/纯运动，也不能用组间得分差估计“多少能力来自注意力”。
干预测的是从父函数出发，哪一组已学改变在其余父计算条件下足以保留具体收益。

## 3. 唯一面板：16条新增行

只有train global32，同旧run的init0–3及scene/noise、环境seed7、policy root7。
两份学习来源固定：teacher17使用D64相对自身parent；teacher43使用S64相对自身parent。
每份各Q/R两臂×四初态，共16条；不增加teacher、初态、seed、切换点或其它task。
这个两来源面板在结果前固定，不能删掉teacher43的能力交换反例来只保留D17正例。

原parent与对应完整D/S共16条直接复用，来源均为：

`/data1/user/ymdai/ember_runs/operator_learning_limit_diagnosis_20260930/`

- `parent/bank/task032_teacher17.safetensors`及`D/bank/task032_teacher17.safetensors`；
- `parent/bank/task032_teacher43.safetensors`及`S/bank/task032_teacher43.safetensors`；
- 原`parent/D/S/bank/panel_teacher0|1.json`保存base/spec/shared/实际因子身份；
- `scenes/manifest.json`及对应原scene，原results/continuous/trajectory提供配对与行为。

原父训练e2afbfd7、学习092a0ae8、读取5313257c分别保留，新生成bank/读取代码身份另记。
旧trainer或可恢复checkpoint存在与否不影响使用已保存完整bank；不可伪造已删除的恢复载荷。
完整来源索引见旧root `analysis/task32_absolute_learning_readback_20261002/rows.jsonl`。
旧T与当前条件900分开，不混用后者的step180、1/8背景或对应成功初态。

复用canonical official rollout：render256/model224、双相机180度、8维state/7动作、十步flow、执行前5、
settling10、成功即终止、Long horizon520。使用原封存scene、policy噪声时钟及同task/init配对。
原有限面板本就复用两个teacher，不声称正式整轮视频无放回、400资格或未见task泛化。
不为了正常BF16/batch/reduction低位差异重跑旧有效面板、限制batch1或扩dtype；若实际消费者/配对合同无法承接，
先修复已理解的读取问题；若原件有效性或科学含义无法确定，报告具体边界，不自行扩大科学范围。

## 4. 必须读取的行为与结果分支

16条都保留真实physical actions、T+1 body/EEF/夹爪/goal与result。
teacher17/init0及teacher43/init3的Q/R四条保存full双RGB，其余12条compact；可直接复用现有capture能力。
报告首次/最终开炉、首次/最终放置、最大抬高及首次3cm（仅描述），完整成功/结束步，以及对父和完整学习臂的得失。
真实闭合命令和RGB只辅助解释，不能由body中心或3cm阈值宣布抓持/接触机制。

- 若Q保留已登记的新增成功而R不保留，支持这些修正的执行读取作用值得优先解释；
  不等于视频编码缺陷已定位，不据此删V或把Q映射直接命名为操作语义。
- 若R保留收益而Q不保留，降低“必须改变执行Q才学成这些修正”的支持，优先解释非Q实际作用；
  R包含多种计算，本项不自动继续拆V/输入/输出或逐层扫描。
- 若两者都不保留完整收益，支持已学作用须共同适配的解释，关闭用这一分组单独指定修复的路线。
  若两者都保留，登记有限冗余/多种足够路径，不发明唯一瓶颈。
- 两teacher/初态结果混合时，保留具体适用条件，不能用净分挑一个全局赢家。
  特别报告S43/init2是否保留放置、是否恢复开炉；部分能力变化不能冒充完整成功。

任何分支都不自动触发formal训练、当前900重训、MT补测、更多投影/层位/scale探针、视频controls、Test或RL。
这项结果应更新完整理论中“真实学习通过哪些执行算子获得/交换能力”，而非产生新的局部Gate课程。

## 5. 成本、工程与停止线

唯一新root：`/data1/user/ymdai/ember_runs/task32_learned_operator_groups_20261002/`。
预计含工程与交付30–60分钟；16条Long有限行及一次source加载，依据最近四格90秒有效消费者和现有bank/评测入口，
GPU阶段预计数分钟至十余分钟。硬限1完整GPUh（含失败、加载、退出）与4GiB新增峰值（含代码/冻结树/缓存/原件）。
不通过更多有效环境case做profile；在本16行内用真实余量验证吞吐，优先共用source、批处理和persistent workers。
不把单卡作为额外永久上限，按现场资源与真实吞吐选择；遵守仓库双节点GPU总量及非干扰要求。
启动前执行者实时核查两节点、data1独立quota/个人用量/共享容量，并给出峰值估计；旧大资产只读引用。

执行者从最新clean pushed main在独占分支/worktree完成必要的最小bank/读取接入、针对消费者验证、Git集成推送和clean detached冻结。
复用canonical owner，不为本分析另造长期并行evaluator；task-only入口在交付后退役，Git/冻结源码及原件保留。
科学记录/代码集成串行：派发后canonical tracked/Git窗口交执行者，main继续只读科学工作。
已定位且不改计算语义/配对/数据/预算的工程问题由执行者自行修复并完整计费，不等待main工程验收。
只有实际科学边界、原件有效性未明或预算不足时回报；否则完成16条后统一交付一次并停止。
不得额外派subagent、监控chat或自通知链；已有直接等待者不再自Queue。
