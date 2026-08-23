# ECP process-identifying meta-task feasibility audit

日期：2026-08-23。状态：**只读可行性结论；尚未获得扩展数据合同的owner授权，未实现、未生成数据、未启动GPU。**

## 1. 为什么下一步不能再是realization solver变体

effective-update profile已经在事前固定的最小回溯尺度前停止。该结果只关闭一个确定性operator，却与此前
fixed-A、raw-factor mobile-rank4的失败共同暴露出更上游的问题：现有训练任务不能让“教学视频决定必要过程”成为可识别变量。
继续改step、rank、初始化、预条件或objective，会把尚未补齐的数据识别问题错误归因给数值求解器。

现有审计已经确认：source71与target40的BDDL成功条件都是当前最终状态谓词的合取，没有已证明的
same-endpoint/different-required-procedure任务对。它们可以支持scene、object、relation和有限order表征研究，但不能单独建立
“相同语言和终点下，只有视频能说明哪个过程才算正确”的因果训练信号。

## 2. 最小且可证伪的任务合同

一个process-identifying pair必须同时满足：

- 两个variant使用完全相同的exact language、场景、初始状态分布和最终状态谓词；
- variant A与B要求相反的必需事件顺序，且错误顺序不能在事后通过达到同一终点被计为成功；
- policy与deployment Writer都不接收variant ID、task ID、reward、predicate state或阶段标记；
- 正确教学视频是区分两个variant的唯一deployment信息；
- teaching video与执行episode跨episode，仍为action-hidden、内部有序视频；
- sibling variant的视频天然构成wrong-video control；language-only、no-video和静态端点对两个variant都不应有可利用的route。

首个候选family可复用LIBERO-90 `LIVING_ROOM_SCENE3`：场景中alphabet soup、butter等多个物体与wooden tray共存，现有
单物体任务共享同一场景和初始谓词。新pair可固定相同语言“put the alphabet soup and butter in the tray”和相同终点
`In(soup,tray) AND In(butter,tray)`，只让两个variant分别要求`soup -> butter`与`butter -> soup`。这只是可行性候选，不是
已批准的数据集设计；正式suite还必须跨scene/family切分，不能靠同一场景中的物体组合形成虚假的task数量。

只读资产核查支持先选这个family：task55的source policy在两个已保留rows分别以97/99步成功，task56分别以85/87步成功。
相比之下，target train24中现成的双物体task37虽有三条保留成功轨迹，但目视复核三条都是alphabet soup先进入basket，未提供
反向teacher。精确资产记录见`docs/evidence/ecp_20260823/process_meta_asset_feasibility_20260823.json`。因此首轮不应先训练新
task expert；应先检验把两个已有source primitives在同一episode中按privileged phase切换能否得到两个方向的可靠teacher。

## 3. 最小工程接入点

LIBERO的`BDDLBaseDomain.step()`最终把`_check_success()`的当前谓词合取直接作为`done`；EMBER canonical evaluator又把
`done`直接记为success。因此仅新增一个包含两个goal predicates的BDDL仍然无法表达顺序。

最小接入是一个repo-owned、有状态但不向policy暴露状态的environment wrapper：

```text
OffScreenRenderEnv
  -> reset/set_init_state时清空temporal state
  -> 每个step后用底层_eval_predicate读取两个授权谓词
  -> 首个新满足谓词若符合required_order：推进phase
  -> 首个新满足谓词若违反required_order：标记episode permanently invalid
  -> 只有顺序正确且最终两个谓词同时成立时返回success=True
```

wrong-first不应返回`done=True`，因为当前evaluator会把任何done解释为成功；wrapper应把该episode保持为invalid并运行到horizon，
同时只在privileged meta-training ledger中记录失败原因。

现有`PersistentTaskEnvironmentPool._task_assets()`还会强制查询installed benchmark task并核对其language；自定义组合language
不能伪装成已有task从这里进入。实现时不应放松正式target40的asset检查，而应让显式meta-task manifest登记custom BDDL、
复用的base task/init-state authority、两个goal predicates与required order；专用meta collector创建底层
`OffScreenRenderEnv`并包上上述wrapper。训练reward路径再通过共享的environment factory复用同一wrapper语义，避免canonical
evaluator与meta task各自实现一套成功判定。π0.5输入、action chunk、replan、preprocessing和正式target40 evaluator均不改变。

## 4. 教学与oracle数据如何取得

不能把两条独立单物体视频直接拼接后冒充同一闭环episode。优先顺序应为：

1. 已保留的三条多物体成功轨迹只观察到一种顺序；完整raw 50-demo corpus当前不在本地，不能把“未见反向”夸大为数据集证明；
2. 首轮直接复用已有source policy成功的task55/task56，在同一episode中以内部predicate phase切换primitive language；
3. 所有收集结果必须用temporal wrapper重新rollout验证，只有顺序与最终状态都通过的episode进入teacher bank；
4. deployment输入仍只保留渲染视频和统一language，teacher phase language、actions、predicate transition与reward全部留在信息墙外。

这符合现有“non-held meta tasks训练时可用action、privileged expert与reward”的原则，但**构造新的temporal task semantics本身超出
当前枚举的LIBERO任务合同**，因此必须先由owner明确授权。

## 5. 授权后的第一道Gate，而不是直接扩成大suite

先只做一个pair family的低成本feasibility，不训练shared Writer：

- 两个variant各有跨episode的成功teacher videos与执行轨迹；
- 同一个task-local privileged oracle或expert能在两个variant上分别取得非偶然闭环成功；
- correct-video相对sibling-wrong-video形成配对优势；
- language-only/no-video不能稳定区分两个variant；
- wrapper的顺序判定、最终状态判定与episode ledger一致。

只有这五项成立，才扩展为跨scene的process-identifying meta suite，按family而不是episode做train/held拆分，然后重新建立专家要求的
multiple-policy、occupancy-complete effect distribution与fixed realization oracle。若最小pair连privileged upper bound都不能成立，
应停止数据扩展并定位expert/data acquisition，而不是回头调Writer。

## 6. 当前决策边界

推荐授权上述最小pair feasibility。未授权前允许的动作只有只读资产审计与方案记录；不创建custom BDDL/wrapper、不给现有任务
改success语义、不解封validation/test、不生成demonstration、不启动GPU，也不建立新的solver或Writer版本。
