# task32已学修正的视频来源诊断

2026-10-02。仅冻结模型的功能来源分析，不训练、不选择checkpoint、不产生EMBER正式性能。
活动状态看progress；本项不恢复旧T/当前900/prefix/MT/RL或其它历史路线。

## 1. 具体问题与对方法判断的作用

旧T的同一份共享Writer S64把teacher43/init2从“开炉而未放壶”变为“放壶而从未开炉”，
又让init3获得完整成功；teacher17的同一共享更新只获得部分完整/部分抬壶能力。
最近Q/R干预表明：43/init2的Q部分能完成两目标，完整更新反而不行，43/init3则只有完整更新成功。
按投影名称不能区分技能，继续拆V/IO/层位已经关闭。

现在需要区分的是视频到LoRA的具体联系，而不是再挑一个参数组：

- 学习在教学后段产生的作用，可能帮助放壶，却也在自身尚未开炉的状态产生不合适的控制；
- 损害也可能来自教学早段的映射改变，或者有益/有害作用本来就跨整段视频分布，不能按视频事件局部分开。

第一种解释只有“后段来源的已学作用实际保留放置、同时导致丢失开炉”的功能证据才得到加强。
内部Value范数、后段参数能量大或两个bank不同都不够。
若效果无法按该可见事件定位，关闭以阶段局部写入/隔离为主修复的理由，不继续细分帧窗或扫描切点。
若可以定位，后续重点是该来源的作用怎样在自身状态被调用，不能简单删除有益后段，也不能直接宣布某个新编码器正确。
这项辨识不要求MT先成功，不把提高一个train案例当作超过MT或新任务迁移。

最近近邻包括§67成熟条件M分组删除、§69–75变化时钟及§100学习增量Q/R分组。
前者不能证明重新共同学习无效；变化时钟已未兑现广泛收益，不能以本项重开；Q/R已经没有全局赢家。
本项保持原帧、地址、覆盖、公共参数和全部38处执行目标，只改变一段真实视频转移由哪个已存映射产生Value。
它针对已有“获得后目标、失去前目标”的实际反例，不能事后把任何结果改写成记忆遗忘证据。

## 2. 同一学习臂、同一数据及可见事件边界

只使用旧T2340与**同一份共享S64**的task32、teacher17/43；不再让两个teacher对应不同学习方式。
父训练e2afbfd7、S64学习092a0ae8、旧读取5313257c，source/public A/B0/probe/normalization不变。
实际S只更新P/C/D/O，native与A固定；S的`recovery_64.pt`只读取末点参数，不创建optimizer或恢复训练。

原件根：`/data1/user/ymdai/ember_runs/operator_learning_limit_diagnosis_20260930/`。
父及S完整bank各用`bank/task032_teacher17.safetensors`与`task032_teacher43.safetensors`；
scene沿用该root的`scenes/manifest.json`，原parent/S的两teacher×四init共16参照直接复用。
完整索引为该root `analysis/task32_absolute_learning_readback_20261002/rows.jsonl`。

main只读两条教学的真实RGB，在stride5画面上检查炉面首次由灰变红：

| teacher | 最后尚未红的采样帧 | 首次已红的采样帧/固定cut | 原帧/采样数 |
| --- | ---: | ---: | ---: |
| 17 | 75 | 80 | 250 / 51 |
| 43 | 90 | 95 | 240 / 49 |

图源为canonical task32 HDF5 `obs/agentview_rgb`，双相机仍按原native合同读取；只为定义本分析的边界查看这些RGB。
原图联系图见本树`.codex/tmp/task32_teacher_stove_transition_samples.jpg`，执行者将其作为小型来源图复制到本批analysis并引用原HDF5路径。
没有读取teacher action/state/reward/terminal来定义cut；也不把视觉红色称为新增的精确teacher goal记录。
以转移的到达帧`frame_indices[t+1] <= cut`定义E，其余为L。cut在新功能结果之前固定，不按成功率调整。
“E/L”只指视频来源时间，不预设两组分别是开炉/放壶的神经语义；后段仍包括离开炉子、接近、取得和搬运。

## 3. 数学干预：保留38处联合更新，隔离写入来源

对每个target，父和S的native X/H、公共A及keys完全共享。
按旧T真实算子，令hbar为RMS归一化H、K_t为50个`normalize(A X_tj)`列，

```
V_t^r = O_r [GELU(P_r K_t + C_r hbar_t) * D_r(hbar_(t+1)-hbar_t)]
R_t   = I - K_t K_t^T / 50
M_(t+1)^r = M_t^r R_t + V_t^r K_t^T / 50,       r in {parent,S}.
```

式中逐horizon计算后按列排列，实际shape沿原owner；矩阵乘积按真实时间顺序，不能交换R。
由于这里仅Value映射改变，精确的有限学习分解为

```
DeltaV_t = V_t^S - V_t^parent
DeltaM_(t+1)^E = DeltaM_t^E R_t + 1_E(t) DeltaV_t K_t^T / 50
DeltaM_(t+1)^L = DeltaM_t^L R_t + 1_L(t) DeltaV_t K_t^T / 50
DeltaM_0^E = DeltaM_0^L = 0
DeltaM_T^E + DeltaM_T^L = M_T^S - M_T^parent.
```

每条video产生两套**诊断用**完整38-target bank：

`A_E=A_L=A_parent; B_E=B_parent+DeltaM_T^E; B_L=B_parent+DeltaM_T^L`。

已有B_parent包含B0，不再另加公共项。所有target同时按同一视频边界干预；不拆Q/V，不使用额外scale。
也可沿原recurrence逐帧选择parent/S的V实现相同干预，但必须保留完整父贡献和同一原覆盖。
这种跨已存映射的有限干预只用于来源归因，不是合法新Writer或部署checkpoint融合方案，不报告为新方法成绩。
它不是视频截断、重排、frame平均、LoRA平均、参数求逆或stage-wise训练；没有删掉教学早段或破坏真实顺序。

原学习的38处native只在RAM保留，现存`native_reader_transport`仅有三target X，不能冒称完整可重用。
准许对这两条视频用原父公共native各重读一次，随后的parent/S Value与两段递推只消费同一份固定特征。
复用旧source/tokenizer/assets及原生owner；记录完整训练/读取身份，不改变source权重或输入合同。
为复现保留H、全部38处归一化K、frame indices及参数来源即可；不复制大policy，不保留重复全X或所有逐帧V。
记录一次全组重构及E+L合计closure；shape/finite/来源/模型语义保持。正常BF16/TF32/reduction差异不触发dtype/batch扫描、
重复reference环境或逐tensor校验；结构性不符、来源错误或无法解释的大偏差按科学边界回报。

## 4. 唯一16行功能面板及读回

teacher17/43 × E/L ×原init0–3，恰好16条新增，四个不同物理初态。
不增加其它teacher、task、seed、cut或环境smoke/profile；父和完整S的16既存行复用。
原scene、env seed7、policy root7及完整绝对噪声时钟保持；official256→224、双rotate、state8/action7、十步flow、
前5执行、settling10、成功停、Long520由canonical consumer承接。

全部保存result、T+1连续body/EEF/夹爪/predicates、实际physical actions；
teacher43的init2与init3在E/L各保存full双RGB，共4 full，其余12 compact。
原S43参照没有RGB，不能编造；新RGB也只按实际replan保存帧解释。
读回首次/最终stove及placement、完整success/结束步、对父与S的R/G/L、真实目标交换和全部不利例。
抬高3cm/最大高度/闭合命令只描述，不作为抓持门槛；无contact/force不推断具体接触根因。

事前结果解释：

- 若L单独保留S43/init2放置并丢失开炉，支持后段来源的学习作用具有这项跨自身状态的损害；
  结合同S17及其它初态判断适用范围，不据单例宣布普遍晚期干扰。
- 若L完成两目标而加入E后丢失开炉，或E单独已导致开炉丢失，则削弱“只是后段知识覆盖前目标”，
  损害涉及早段映射或其联合调用。若E得到放壶，更直接反对按视频时间给内部作用贴技能标签。
- 若E/L均不能保留完整新增能力，或两teacher/初态混合，则登记分布式共同适配；
  关闭以此次时间划分支持事件独立化/阶段门控的方向，不进一步细分切点、加scale或开逐层探针。

任何分支都不自动触发fresh、续训、controls、Test、RL或部署该混合bank。
本批只补“哪段实际教学特征对应了有用/有害的学习作用”这一缺口，后继方法仍须完整因果与历史论证。

## 5. 时间、资源、工程与停止

唯一新root：`/data1/user/ymdai/ember_runs/task32_learned_video_segment_20261002/`。
预计含工程/读回35–75分钟：最近16行约284秒完整GPU占用，本项另加两条native及四bank编译，GPU预计5–15分钟。
硬限1完整GPUh、4GiB新增峰值（包括冻结代码、H/K、bank、轨迹和失败），不因失败自动扩预算。
重用大资产、source一次加载，按实际授权case/frame作最大有益packing；不固定额外卡数上限或用新增工作填显存。
执行者launch前检查两节点、data1独立quota/个人实占/shared及启动后总GPU数量；预计新增峰值先按3GiB核算。

派发后canonical tracked/Git窗口交唯一实验session，main只读独立科学工作。
执行者负责独占分支实现、实际消费者检查、main集成推送和clean detached冻结；源码和运行问题在权限内自行修复。
复用canonical owner，临时entry在结束后退役，Git/frozen及所有有效/失败原件保留；不留第二套长期consumer。
只在整批完成或科学语义/原件有效性/预算边界确需裁决时回报；完成16行和读回后停止。
不派新subagent/thread/monitor，不排自通知链，直接等待者不再自Queue。

## 6. 完成与科学裁决（2026-10-02）

16新行及全部原件已交付，main直接消费continuous、四份RGB及构造源码，详见机制§101/findings§280。
父/S/E/L成功集合17为{1}/{1,2}/{0,1,2}/{0,1}，43为{0,1}/{0,1,3}/{0,1,2}/{1}。
43/init2的E完成开炉与放置，L只开炉；后段单独重现完整S能力交换的事前预测未通过。
43/init3两部分均未保留完整S的成功，17/init0两部分各自成功而历史S失败；保留全部七条失败。
E是完整父LoRA上的早段Value增量，其传播仍经过后段K，不能当作仅看早段教学。
相对历史S-parent的合计增量重构误差1.22918%/.82360%，未重跑新重构完整S；数值边界限制精确四角归因。
关闭由这次时间分区支持事件独立化/阶段门控、选择E部署及继续切点/层位/scale扫描，不触发新训练。
实际.082021884GPUh、留存约1.025GiB，GPU释放，专用入口退役，主讨论已接回记录窗口；本合同结束。
