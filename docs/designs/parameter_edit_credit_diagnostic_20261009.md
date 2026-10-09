# 固定真实incoming的一步经验贡献与控制变化（2026-10-09）

## 1. 科学决定与问题

本设计是否active只看progress；主讨论已接回χ360完整补充交付的tracked/Git窗口。
本批是冻结模型的定向机制辨识，目标是决定下一种共享学习怎样处理经验信用和能力保持；没有登记新训练方法。
固定原χ360和已保存的实际参数，比较一次真实编辑前后及同incoming屏蔽经验的结果。
完整130/400对MT153为R104/G26/L49，Long1的真实改善不能补偿其它任务损失。
两个train48仍MT/end/null27/25/25，已有null按不同中间参数自行递推，不能隔离一次编辑中E的作用。

main直接从完整400原始行和成功集合复算：

| 原实际条件分组 | 条件数 | MT | end | R/G/L |
|---|---:|---:|---:|---|
| 实践成功后结束 | 218 | 130 | 118 | 97/21/33 |
| 预算耗尽 | 182 | 23 | 12 | 7/5/16 |
| J=1（均MT实践成功） | 163 | 103 | 99 | 83/16/20 |
| 实践成功且J≥2 | 55 | 27 | 19 | 14/5/13 |

这些是事后分组，不是不同J的因果效果。J1净损4只占全局净损23的一部分；
不能把“成功后不编辑”或一项保持loss当作充分修复。多次编辑的最终MT/end比较又混合了前面各次编辑，
尚不知道最后一个真实incoming已丢失什么、最后一步增加/破坏什么，以及E相对相同incoming是否提供有益定向变化。
这是当前参数编辑机制的具体缺口，不另立通用FM/flow路径问题。

最近似历史必须保留：ADSP的自身成功动作FM一阶保护实际满足22约束，139→138；
SKNC严格保持15个训练条件完整LoRA，134→137仍丢13；CV-CSD/DJNFR成功动作信用143→134/136。
它们不是当前E/真实incoming条件的完整十步函数比较，但已降低“保护有限成功支持自然迁移”的预期。
详见[findings§386](../../findings.md#386-成功功能保持并非没有历史负担的新解法本轮不再自动接续学习补丁2026-10-07)。
旧T161/other150保留有限视频正证据；phase learner-state、SEOD和shared-SDE/experience阴性也不被重新命名。

## 2. 冻结范围和面板

父ROOT为`/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009`。
只用其`evaluation/train360/results.json`中的下表8个原condition与原state32/33/34，共24个训练诊断条件。
已核这些实际文件和`record.json.events`存在；incoming由最后一个实际event决定，不能挑中间最高分版本。

| global task | 两个原teacher | 原实际J | 选择理由 |
|---|---|---|---|
| 0 / Spatial0 | 43、29 | 5、1 | 预算失败和单次成功编辑均有原能力损失 |
| 13 / Object3 | 16、4 | 1、1 | 保留已有能力且有新增成功，不只选择坏例 |
| 20 / Goal0 | 15、6 | 2、2 | 非MT实际成功incoming；end与旧null存在差别 |
| 32 / Long2 | 16、44 | 1、2 | 多阶段任务中的获取/损失及成功/预算条件 |

面板按已见结果定向选择，用于区分机制，不是盲测、总体能力估计或strict400资格。
不添加teacher/初态/其它checkpoint，原所有正反结果不改。不计算新的validation或Test条件。
χ固定`training/checkpoints/macro_00000360`；source、MT、normalization、tokenizer、38-target/rank128和prefix冻结，
资产沿原contract引用，不复制source或数据，不把完整恢复点改成仅权重存档。

对每个原condition固定最后事件的实际输入：

- `I`：该事件`incoming`（J1为原MT，其余为对应`incoming_00j.safetensors`）。
- `E+`：原`end.safetensors`，即χ360消费真实累计经验后的实际结果；直接复用，不重新选择或重采实践。
- `E0`：同一χ360、相同指定V/L、**相同实际I**，只把经验输入替换为训练中已有的empty分支`{}`，前向一次生成完整LoRA。

因此`E+`与`E0`固定了当前参数和教学，只删当前编辑的显式经验路径；
I可能已承载前次经验，链长度/编辑时点由旧真实链固定。E0不是独立无交互算法，也不是“删去全部经验历史”。
本E0不能直接复用旧`null.safetensors`：旧null在J≥2时使用自己的中间Λ。
只物化8套新E0，保留它们作为诊断原件；不部署第二adapter、不做LoRA平均/插值/择优或condition内optimizer。
teacher始终action-hidden；合法train任务非teacher query labels只供下述冻结分析，不进入Compiler。

## 3. 三方配对闭环与实际动作

每个原condition×state32/33/34分别运行I/E+/E0，共72条；三臂使用原train360正式消费者的完整scene恢复、
exact language、env/policy RNG和source处理。原end/MT行保留作背景，本批新行单独保存，不据重算差异替换旧分数。
目的是获取此前没有保存的行为及一次编辑的反事实，不按复现旧success筛选或反复运行。
训练面板原scene来自其`evaluation_contract.json.environment_contract`；不得改成普通reset后假称相同scene。

官方render256/model224、双相机180度rotate、state8/action7、真实10-flow、执行前5后重规划、dummy10、
原220/280/300/520 horizon及成功即停保持。这里不再适应、不再读取经验或修订参数。
复用canonical动态queue、persistent env和不同LoRA原生batch；不得另写平行策略/模拟器。

保存每条真实replan前的双RGB/proprio、完整noise seed、生成动作50×7和实际执行prefix/mask、
原生终止/steps，以及既有被动capture可提供的T+1对象/夹爪/fixture/predicate。
后者只服务行为辨认，不输入Writer；无法由图像确认的接触、抓持和唯一几何原因写unknown。
不新增全层hidden大载荷、逐tensor一致检查或哈希。所有72条保留紧凑raw；生成足以比较教学、实践与新初态的图/短片索引。

## 4. 同一观测上的功能交叉读出

令`F_P(o,xi)`为actor P在实际观测o和同一Gaussian xi上十步积分后送环境的前5×7命令。
直接比较各自轨迹动作会混合参数与已到达状态，故在每个配对state中按I/E+共同replan前缀长度m，
取`unique(floor(linspace(0,m-1,min(6,m))))`的索引；不按动作差或成功挑query。
实际同索引noise须按原完整种子流配对，不能用不同噪声解释为经验/参数效应。

在每个索引使用原始o_I、o_E+与已保存的本臂动作，只增加三次完整十步预测：
`F_E+(o_I,xi)`、`F_E0(o_I,xi)`、`F_I(o_E+,xi)`。
前两项隔离相同自身观测处E路径引起的真实动作变化；第三项允许检查反向状态分布上的变化。
例如完整动作差精确分为

`F_E+(o_E+,xi)-F_I(o_I,xi)`
`= [F_E+(o_I,xi)-F_I(o_I,xi)] + [F_E+(o_E+,xi)-F_E+(o_I,xi)]`。

第一项是旧actor所到状态上的编辑作用，第二项是新actor对状态变化的响应；另用I参考作对称分解。
这是函数分解，不是两项对应失败率的因果分摊，不用范数自动标记好坏。报告每通道/夹爪实际开合与行为阶段，
未执行尾部只作模型输出，不称真实行为。两轨迹后期相同时间编号不保证相同操作阶段，必须结合实际RGB解释。
最多24×6×3=432次新增十步预测，所有原本实际预测直接复用。

另在每个原condition最后实践episode的全部decision索引中，按同样首末均含的6点规则取样，最多48个自身经验query。
用实际I、E+、E0在**同一原RGB/proprio/记录noise**上各作一次十步前向，最多144次。
实际经验动作及执行mask保留；旧完整50动作未保存时，不伪造为已保存原件，也不做低位精度重跑。
成功实践的I确是其行为actor；预算失败实践的动作不被当成正确标签。合计离线新增十步上界576次。
这里在当前condition已真实观察过的状态检查功能变化，再与未用于适应的新初态联系；不把局部保持外推为泛化保证。

## 5. 与实际FM目标的联系

对每condition固定7个非teacher成功episode，每episode四等分区间各取1个合法query，共28。
使用canonical offset1、50×32 latent、前7维和原loss的实际padding约定。采样完全独立于本批动作/成绩：
`rng=default_rng(SeedSequence([20261009,0xED17,global_task_id,teacher_demo]))`，
从`[0..49]`排除teacher后`choice(7,replace=False)`，区间取样按现有`data.event_for_update`的端点约定；
FM seed使用`condition_seed(task_id,teacher_demo,domain=0xED17)`，三臂共用同一28组noise/time。
只读授权train动作；无gradient、optimizer或参数更新。总8×28×3=672次单time native预测。

报告逐condition I/E+/E0的原full50 FM、配对差，以及前5/动作通道诊断，保留不利query和取样不确定性。
当前keep导数只是`[1+.2*1(l_out>l_in)]*grad(l_out)`，没有独立旧函数保持方向。
本分析检查当前编辑是否确实出现“专家FM更优但实际自身控制更差”；若没有，不用该口号保护信用根因假说。
FM差值和同观测函数差只能约束解释，不能单独选择模型或推出视频/E有效。

## 6. 预期分支与下一判断

- **经验有益但编辑总收益不足：** E+相对固定I的E0有重复出现的新增/保持，相关真实动作与教学任务相容；
  I→E+仍净损或部分失效。增强保留经验入口、改变其信用/能力保持组织的依据，但不证明任意保持项可学或足够。
- **当前经验路径有害：** E0相对E+恢复实际成功，且同incoming/观测比较显示E改变了相关动作；
  加强当前经验编码/条件信用组织有缺口的解释，削弱只增加递推次数/经验量的充分性。仍不由此区分唯一编码层或loss原因。
- **主要是共同编辑：** E+/E0相近而都相对I退步，且相同观测上的相关行为都被改写；
  降低“新增E自然带来定向修正”支持，应研究共享教学编辑怎样取得新控制并控制漂移，不能把失败全推给E。
- **损失早已存在：** 多次编辑condition的I已失败、最后E+没有进一步损失或有改善；
  削弱最后一次成功编辑是主因的解释，不据旧MT/end差就加成功门控。J1和多次编辑分开解释。
- **局部成功函数保持而新初态失败：** 在已成功自身实践观测上动作大体保留、相同I在新初态仍无优势；
  削弱“擦除该段成功功能”解释，关注支持覆盖和纠偏能力。细小夹爪/接触时序仍可能重要，不用平均MSE排除。
- **动作差无方向性或样本不足：** gains/losses/retained中的变化相近或相反，不能把偏离MT大小当质量；
  如实降低统一keep修复的优先级，不因存在差值就自动开启学习。

任何分支都不自动转入keep训练、RL、架构更换或更多panel。main须把实际效应放回完整130/MT153、
Long1正例和历史结果，选择能获得新教学能力而非仅保护旧策略的下一方法。单个极端例不承担全部总体缺口解释。
本批不检验视频必要性；teacher仍相同。后继若涉及视频因果证据须另登记，shuffle/reverse不提前使用。

## 7. 工程、预算与交付

新ROOT：`/data1/user/ymdai/ember_runs/parameter_edit_credit_diagnostic_20261009`。
代码实现/实际消费者检查/运行/原件消费/Git由原实验session闭环；main负责科学取舍，不重复工程验收。
按现有拥有者增加窄诊断入口，遵循code-architecture-gate；不恢复旧trainer，不扩大旧正式400入口的科学准入。
从最新main建立独占codex分支/worktree，targeted consumer通过后push，实际计算用clean pushed detached代码。
普通接口/capture故障在同科学合同内自行修复并新冻结继续，保留失败/有效产物/费用；未知原因或科学语义变化交main裁决。

预计实现/接通1–3小时，科学计算20–60分钟、1–3GPUh；GPU观察线6GPUh/科学elapsed3h。
依据上一批不同LoRA B16实测单worker约17–30环境steps/s，本批仅72条且没有新实践链，
总环境上界24,480steps（72次settling已含），加8次E0完整教学读取、576次离线十步及672次FM。
加载/工程pilot/失败均计费；确需额外接口pilot时最多16次完整十步、0额外环境条件，计入上述观察线。
这仍是预测，执行者用实际消费者成本更新预计；main自主调整预算，不向Owner增加审批。

预计新增峰24GiB，包含代码、完整RGB/动作/compact、E0权重、临时输出；复用原资产和经验，旧原件只读。
不建立新的64GiB全片cache；本批没有学习/经验重采，必要小缓存纳入24GiB并优先复用现有合法来源。
开root/launch前现场查strg01独立data1 quota、相关个人用量与共享容量；不足按已授权生命周期清理，含糊/关键原件不删。
每次launch现场检查两节点GPU。按实测总吞吐配置2–4个有用worker，资源充分且确实有收益时可增加，
遵循最新全局6/8和单节点≤6合同；不把建议worker数当科学上限，不dummy占卡、不操作其他用户任务。
优先沿用已验证B16，再按实际空余/吞吐调整microbatch或离线batch，不额外扩大科学query来profile。

一次完整交付：8个原condition/I/E+/E0身份，72原行及所有paired success sets/R/G/L/churn/breadth、
8×28 FM、所有交叉预测/原动作/索引、教学和实践/新初态图像的行为说明、竞争解释支持度、全部实际成本/失败/退出、
clean pushed Git与代码退役记录。新行与历史重算不同时保留两者，不追逐一致或优选分数。
所有科学原件/必要权重保留；无用途临时产物与合并worktree按生命周期退役，不把旧路径继续称为可直接运行命令。
执行者只在整批完成或具体阻塞时向main `01a11b05-3460-7eb0-aca8-177d6d86ef48`可靠回报一次，
沿既有active Steer/idle resume-start机制验证实际处理；不按阶段自Queue、不发心跳，不叠加退出等待者。
从派发被实际接受至整批移交，实验session独占tracked/Git；main继续独立科学分析。批次完成后main主动裁决并接续。
