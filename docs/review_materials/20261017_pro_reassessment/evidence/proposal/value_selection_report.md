# §16独立价值学习：完整工程验收与一次整批交付

Phi180的唯一strict correct400为 **167/400**，较旧完整流程163得13失9、净+4；相对MT153净+14、首G165净+2，但低于D72的175和D144的168。14个pair MT优于G反例只正确选MT 2个，6个RL末前缀MT胜全部G反例为0；396个held多候选条件全部未选MT。实际使用模块的梯度及冻结/信息墙/配对已通过，不能将这些负结果当工程失败或仅归held迁移。主讨论按既定§16裁决，本验收不自动续训。

ROOT：`runs/video_guided_proposal_writer_20261014/independent_value_fixed_paths`。整批唯一owner02于2026-10-14T02:26:00.449780Z exit0，已按task/ROOT/attempt验收。原owner01发布失败及448已完成native查询保留、原样复用；没有第二等待者、阶段通知或重启已消费旧批。

## 完整配对结果

| 参照 | 原成功 | 保持 | 获得 | 丢失 | 净变化 | Churn | 成功集合Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT | 153 | 123 | 44 | 30 | +14 | 74 | 0.624365 |
| 旧完整流程 | 163 | 154 | 13 | 9 | +4 | 22 | 0.875000 |
| 首G | 165 | 139 | 28 | 26 | +2 | 54 | 0.720207 |
| D72 | 175 | 145 | 22 | 30 | -8 | 52 | 0.736041 |
| D144 | 168 | 141 | 26 | 27 | -1 | 53 | 0.726804 |

各task均50个不同教学视频与固定初态配对；breadth7/8。保持、获得、丢失的全部初态列表见results.json，不拼success union，不逐task选点。

| Task | Phi180 | MT | 旧完整 | 首G | D72 | D144 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 43 | 41 | 44 | 43 | 45 | 43 |
| 6 | 10 | 7 | 9 | 8 | 6 | 7 |
| 11 | 39 | 36 | 39 | 37 | 37 | 37 |
| 16 | 19 | 10 | 17 | 21 | 24 | 20 |
| 23 | 1 | 0 | 0 | 0 | 1 | 1 |
| 26 | 34 | 35 | 37 | 38 | 37 | 33 |
| 31 | 21 | 24 | 17 | 18 | 25 | 27 |
| 39 | 0 | 0 | 0 | 0 | 0 | 0 |

Spatial/Object/Goal/Long各100行，Phi为53/58/35/21；MT48/46/35/24、旧完整53/56/37/17、首G51/58/38/18、D72 51/61/38/25、D144 50/57/34/27。

## Non-held固定输入读回

144真实路径、576prefix、1286候选项，不是新样本或未见task泛化；全部原query16标签/类别保留。按task→prefix→candidate等权，beta只作同输入排序参照，其softmax不充作成功概率。

| 子集 | BCE | Brier | Phi选中query回报 | beta选中query回报 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 0.522808683 | 0.046751195 | 0.715856481 | 0.715639468 | 0.018590856 |
| G_seen | 0.536402269 | 0.050856333 | 0.688657407 | 0.687789352 | 0.009693287 |
| G_novel | 0.509215098 | 0.042646057 | 0.743055556 | 0.743489583 | 0.027488426 |
| pair | 0.518598849 | 0.045303618 | 0.706163194 | 0.706163194 | 0.013020833 |
| RL | 0.527018518 | 0.048198773 | 0.725549769 | 0.725115741 | 0.024160880 |

14个pair反例仅task64/101正确选MT（2/14），其它12选G；6个RL末prefix反例（111/66/121/108/41/0）全选G（0/6）。正反原行都在nonheld_readout.json；Phi0全0.5及同分MT是定义，不是额外forward，选中query回报.662326389。不能用BCE/Brier或query均值替代最终控制，也不把训练域拟合称外推。

## 工程与信息墙

- 固定G38349/beta18/source，Phi复制原final有效编码、独立fresh AdamW、零head/counts。180有效logicalB4、720路径曝光、72task各pair5/RL5，576/1286唯一训练面板；单候选prefix仍训练，实际重复曝光1010次。counts只来自当前prefix已结束自身episode并按实际参数键匹配，query和未来H不进条件。
- 全720 rank记录的query标签、真实prefix/H/候选/counts及BCE消费者核对通过；首步实际logits0且仅head有信用，第二步全部实际使用Phi模块信用为正。G/beta/source无更新，不添加fake loss。学习后的Phi Reader/static/memory重算，缓存仅冻结输入。
- 60/120完整恢复点、180唯一末点。完整G/beta/Phi、sampler、4rank RNG/topology齐全；209个Phi参数中198个有Adam状态，全部step180。11个共同结构参数未参与本实际计算：两层static memory_attention的8参数、末static broadcast的2参数、空H分支reader.empty。没有无消费者STOP/parent/practice头，不虚称209个都有优化器信用。
- 缺响应346项实际5530 native10查询（上限5536），原RL940/held1573完整复用；首次失败已保存448未重做，恢复仅5082。274条补输入ledger全部完成、finite、原noise及70维响应通过；这不是新训练query/环境采集。
- 全400 Phi选择于02:17:03Z封存后才读取原final/MT结果；每行完整候选、原path及参数源绑定，不能按同名G跨path冒认同参数。162同完整候选原行复用、4单MT原行复用、234变化G真实paired执行；真正比较后MT0/396。
- 新234完整episode/234physical starts/66,783环境步（含2,340 dummy），失败partial/未知/在途0，低于400/416/144480上限。原scene/env-policy noise、50video/task、官方horizon和自身适应/最终初态分离通过。4条旧官方MT原行没有initial_proprio、settling_steps及environment_steps字段，保留缺项，不补采或伪造。
- 新方法仍依赖旧1988适应episode/683,330环境步；引用旧§23整ROOT的34.272177007GPUh（含其它阶段和失败），不将其冒称仅1988 held实践费用，不重复计入本批新费用。更早G训练等上游费用不在该ROOT合计内，不能宣称这是完整端到端总费用。

## 真实资源、时间与费用

- 全28 GPU收据合计6.776979351 GPUh，含首次6个发布失败0.063266839；没有GPUh硬限或遗漏失败。00:26:16Z首次观测派发→整批退出119.741min（实际派发可更早），恢复owner84.769min；原90–180min时钟未重置。
- 六卡GPU-node-A:1/3/6/2/7/0持久缺输入消费者；随后四rank Phi(1/3/6/2)实际46.011min。端点3+3队列交错、释放的nonheld消费者承接held剩余工作；3个空nonheld消费者CPU省略/0GPU launch。选择封存后六个真实final消费者，未静态锁一worker/面板。
- 全180 B4的最大rank耗时平均15.128s、最长36.932s，最大H624；allocated15.326GiB/reserved24.484GiB/host RSS17.454GiB。按完成全长实际消费者计峰值，首B4不当全长保证。native物理query batch16已达实际support cap；frame64实际使用，不宣称已证明frame128或当前物理配置最优。
- 每个launch独立storage-node data1 quota、个人目录用量/共享容量和双节点GPU现场均有原件；合计6/node6，不按util阈值或通用固定显存过滤，不驱逐外部任务。ROOT约0.994GiB，64GiB是初估而非通用显存硬限；大H/模型/权重只读引用，未存146.71GiB逐prefix重复特征。

## 代码与交付边界

首次448查询源b77eccc64935914600f1738a4f5b7dbb9b9d1f40；成功准备/学习/读出源c683ff03aef0a8424e4d5980f5896d06fe981535。窄修仅四role不可变shards发布，CPU真实原输入消费通过，模型/数据/损失/更新/400科学合同不变。两正式clean pushed detached冻结保留，开发branch/worktree已确认完整ancestor后清理。

本次验收CPU8.489s、0模型forward/环境，verify.py与acceptance.json同目录可回读。原失败、完整checkpoint、720实际rank记录、144 nonheld原行、全400选择及最终原行、资源/费用回执保留。根状态/设计/历史串行交付main后，以同目录delivery/complete.json记录最终Git身份和唯一main回执。

此为单Phi末点：无相邻稳定性、视频因果controls或Test证据。完整EMBER方法目标未完成；§16有限计算已结束，0自动后继、第二同目标块或新训练授权。科学消费和后继由主讨论自主接续。
