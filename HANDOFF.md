# 新 session 接续入口

准备日期：2026-09-08。Owner已结束本轮架构讨论，指定后续由新session推进；本session只做启动前准备。
本文件是临时导航，长期目标、最终设计、授权和计划均已在正式文档。接班者确认消费后删除本文件并修正所有引用即可。

## 先读与立即做什么

1. **按Owner明确要求创建覆盖整个科研推进过程的goal，不设置token预算。**随后全面阅读仓库，覆盖源码、测试、脚本、配置、项目文档、
   历史索引与相关原始证据，弄清EMBER目标、已做工作、正负结果、当前实现和资产。不能只读README或局部模块就开始修改。
   先以 `AGENTS.md`、`docs/current_owner_requirements.md`、`task_plan.md`、`progress.md` 区分authority和旧执行历史；大批raw rows用程序读取核对，
   不把checkpoint/数据二进制全文dump进上下文，也不以读摘要代替理解其证据。
2. 完整读 `docs/concept.md` 与 **`docs/horizon_relation_video_writer_design.md`**。
3. 读 `docs/review_materials/20260908/README.md` 的Owner裁决，再读最后架构原文 `expert_past_interleaved.md` 和完整RL原文
   `expert_fm_rl_protocol.md`。相关历史先读 `findings.md` 与 `docs/research_history.md`，按问题展开原件。
4. 核对最新main/remote和dirty工作树。简述最终图与实施缺口后，直接开始唯一canonical实现；无需再次请求整体方案批准。
   新session中的goal应覆盖完整实现、正式实验、依据结果迭代与性能/稳定性达标；不要只把建模或smoke设为最终目标。

## 最容易恢复错的决定

- **长程四组全为过去单向past+self。**专家原文推荐双向，已被Owner最新选择覆盖；不要按原文未来反馈的段落实现。
- **完整H-query沿H双向**，与视频t的长程因果mask分开；每组仍保留完整50个H。
- 只取实际 `action_out_proj` 输入的post-norm末层hidden。旧 `ActionLayerStateCapture` 最后项是norm输入，直接slice旧stack不正确。
- 最终Z和KV来自同一次真实冻结prefix；现有缓存尚未提供最终Z，不能用静态language embedding冒充。
- 过去4帧 → 软对应 → H-query → 两端视觉核实 → 按历史u短GRU → 临时H-read → 单向long → 前三组逐H回写。
  第四组P直接进608-query集合compiler与native完整A/B；不增加第四次回写、旧前置S或18层融合。
- 首版FM辅助共享Writer RL；同版本采集/FM/RL梯度后一次更新，真实10step flow，LOO和Q/M。
  旧reward credit是antithetic坐标扰动，不等于新action-Gaussian RL；拒绝更新不能倒退已消费sampler/RNG而重复同一批。

## 实施与资源状态

目前新方案只有正式设计，**没有新源码、checkpoint或分数**。现有main科学代码是旧layered Writer。
`configs/pi05_layered_writer_v1.json`仍是旧16-query、LR1e-4、672步协议，不得直接用它启动新run。
新session负责同步修改canonical模型、采样、训练、物化、配置和checkpoint schema，拒绝旧模型resume；不会留下平行fallback。
**首个正式实现必须完整忠实于最终设计，不能缩水、静默省略模块或以近似替代关键依赖。**真正的设计矛盾须说明并在合同内解决；
后续方法迭代要有真实结果与因果依据，并更新正式设计，不能把“迭代”当作实施前打折扣的理由。
Owner进一步明确：核心架构思想保持，但具体方法不是永久冻结；有充分已有证据或实际问题时，可以先修订合同再完整实现，
不必强行运行已知错误的配置。允许合同内实质重构，不限于小修小补，也不需要逐项再获批准。

代码地图和现有资产入口见design §9–10与README。关键source/tokenizer/40目标及71source HDF5/LIBERO assets已做存在性核对，
不是全树完整性扫描。环境复用`.venv`，不要重新复制大资产或安装重复环境。
本session没有刷新GPU与quota，也没有启动GPU作业；launch和大输出前必须现场核验两节点/strg01独立配额及预计峰值。
formal train/eval来自clean pushed commit的detached frozen worktree，canonical实现集成main并push。

## 必须记住的基线与重复失败

- 已存validation8 source参照47/400（11.75%；另一历史面板48/400）。train24 rank128 SFT相邻step400/425为109/107（27.25%/26.75%）。
  这些是已存正式结果，非新后端重跑；SFT不读取teacher视频，也不是rank16 Writer同参数量对照。比较前核对配对与执行口径。
- 有信息量学习后仍不及或仅略超source/SFT，是严重能力缺口。旧67/64超过source47却明显低于SFT，不能再用这种局部提升或loss/几何改善开脱。
  不用初始identity/smoke低分立即推翻全图，也不以“还需训练”无限推迟判断。目标仍是>145及稳定/广度/因果。
- 改动前必须沿research_history查最近等价尝试：完整H或梯度非零不等于过程理解；参数几何/低方差不等于行为；
  clone强而shared弱、熟悉视频也弱不能只解释为迁移；多K或单峰不等于稳定；堆summary/gate/校准及反复小扫不能替代机制定位。
- 根据证据可以自主修改读取、关系、时序、回写、读出、FM/RL和优化/工程实现，必要时实质重构；保持总体思想与硬合同，更新正式设计并验证。
  先找最早失效接口和竞争解释，不能只消掉报错或让内部指标变好就宣布解决。详见design §1.1和长期要求。

## 旧工作必须保留边界

- 旧train24永久止于384：correct69→67、other72→64、seen/held21/18。原件位于
  `runs/analysis/layered_relation_writer_20260907/train24_shared/`；不恢复576/624/672。
- 旧P/Q width256只有训练/checkpoint、无闭环，未授权为接续自动补评。
- `.codex/worktrees/native-factor-readout`有dirty草稿，分支`codex/native-factor-readout`基于ec02710b；
  修改layered及测试、删除coordinate、新增native_factor，仅换旧图末端。保护其未提交内容，可只读审视后选择性迁入，不能整支直接合并。
- 7个历史detached worktree及唯一checkpoint保留；没有必要为当前实现全部恢复或清理。

## 推进与完成条件

按task_plan完成：实现→真实机制与最长K1/K4成本→首个结果前冻结run合同→完整train24共享学习/训练侧诊断
→有信息量strict400与相邻稳定→冻结后因果controls→方法冻结后的32/8 fresh/Test。
首段以24个接受联合更新及训练侧Sigma/0配对诊断为起点；batch、拓扑、真实预算和后续节点由实测与结果前登记落实。
不把smoke/梯度接通/单次高分当完成，不把每个模块变成冻结课程，不无限小扫或遇到局部问题就推翻全图。

最终科学线为strict paired single-checkpoint correct>145/400及AGENTS/design的全部稳定与因果合同。
只在真正的权限、数据、资源或无法自主裁决的重大科学边界回到Owner；普通实施与验证不再请求重复批准。

## 本次交接验证

交接准备前main为e868de525fda0a20c597dee2c7bffe5717f5e2fd且干净；完成后的精确提交以Git最新文档提交和最终交付消息为准。
本次仅保存原文、更新正式设计/状态/入口、核对资产存在性并检查文档引用与diff；不声称新架构已经通过CPU或GPU测试。
