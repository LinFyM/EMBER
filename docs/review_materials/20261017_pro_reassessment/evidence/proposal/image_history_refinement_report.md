# §18 完整优化量批工程交付（2026-10-14）

ROOT：runs/video_guided_proposal_writer_20261014/image_gemma_H_refinement_from8640
唯一 attempt：refinement_from8640_01；2026-10-14T13:14:02.831776+00:00 exit0。
计算来源：778ff8eae613f0a3c9a03cc63a0553b2604266a8，clean pushed detached；源8640及旧失败/费用只读保留。

10800 全144 mean BA相对原8640改善 V8.846%/VL6.024%，按事前规则共同继续至25920。每臂新增17280有效B4，恢复点不参与选择。25920全144 mean BA为V .185180383448、VL .170818593591。
V seen405/MT retained305，未通过306保持门槛；VL421/retained315，通过全部四suite净收益规则。按至少一臂通过则两臂都做held的原合同，完整执行两个400，未放宽门槛。

| 面板 | V25920 | VL25920 | 各自8640 | 强参照 |
|---|---:|---:|---|---|
| seen576 | 405 | 421 | V372 / VL353 | MT340 / 首G427 / D72与D144414 |
| strict held400 | 148 | 141 | V149 / VL151 | MT153 / 首G165 / D72 175 / D144168 / Phi167 |

held对MT：V R127/G21/L26/net−5/churn47；VL R118/G23/L35/net−12/churn58。
对各自8640：seen V R331/G74/L41/net+33，VL R327/G94/L26/net+68；held V R120/G28/L29/net−1，VL R118/G23/L33/net−10。
V对VL：seen R380/G25/L41/net−16；held R122/G26/L19/net+7。全部其它强参照配对与重合率在results.json，未融合/union/择臂或选恢复点。

| held task | V | VL |
|---|---:|---:|
| 3 | 48/50 | 41/50 |
| 6 | 4/50 | 5/50 |
| 11 | 37/50 | 37/50 |
| 16 | 11/50 | 7/50 |
| 23 | 0/50 | 0/50 |
| 26 | 32/50 | 36/50 |
| 31 | 16/50 | 15/50 |
| 39 | 0/50 | 0/50 |

两臂seen breadth35/36（task29为0），held breadth6/8（task23与39为0）；全部失败初态仍为有效完整结果。seen提升未转化为held增量，本次完整结果交main科学消费；不据内部fit或局部正例认定动态教学理解，也不修补科学阴性。

完整25920 compiler/Adam/sampler/三rank RNG保留：V284/VL290 Adam entries及25920 step；新增流global8640..25919、每逻辑B4四condition/各.25权重/rank SUM与原144重复流保持。真实Reader/history/decoder/output信用、learned memory重算、冻结source/MT/E与无execution hidden保持。
两臂各36 seen完整编译+400 held编译，共872完整38-target LoRA；各576+400原行、400不同task-video、原state-video/env与完整policy noise、真实MT-H1、原场景/初始proprio及全配对集合保留。无需重跑参照或模型。
一个整批owner、19个GPU job已全部exit0，原六卡GPU-node-A:0/4/5及1/3/6的world3+3学习后按ready接六个持久共享消费者，slots3/3/3/3/2/2总16、frame chunk64。实际所有128读出shard资源另见actual_readout_resources.json；GPU-node-A按精确ROOT/冻结路径查询无残余consumer。
新ledger1952 finished/1952 submitted upper/0 failed/0在途/0未知步，480317环境步含19520 settling，均在1952/1968/16/1043040合同内。新增practice/training-query/controls/Test为0；实际最终闭环交互已按原合同计入。
新ROOT 16.951125519 GPUh（19jobs/19attempts，0失败，carry-in0）。父H批12.571101360GPUh及47失败/522600步上界，旧no-H中止2.371076951GPUh另列；原MT36/8131步+400/114908步共436唯一episode/123039步共享只计一次，其旧专属GPU费用未知。不能称整个方法零交互。
09:41:00.214Z派发→13:14:02.831776Z完成为213.043630min，在原150–240min内。detached launch10:17:46.900832Z、owner实际main10:18:27.800841Z与10535.031秒另列，没有重置派发时钟。新ROOT实际约4.559GiB（4895346688bytes），低于16GiB增长初估；大cache只引用旧资产。

验收原件：verify.py / acceptance.json / process_release.json / actual_readout_resources.json；results.json、两个节点、seen_gate_decision.json、complete.json、原19job exit/预算事件/完整checkpoint保留。费用与wall采用job start/end和实际物理卡数；shard.seconds沿旧接口为monotonic时间点，未当作elapsed费用。
当前本批执行结束，无自动第三同条件窗口、重试/后继或额外等待者。最终Git与交付回执见本目录delivery记录。

最终main=b1a4c4d79e0a0d196f1eec629ab1f4f11f647516，clean/pushed；唯一main整批交付status=processing_verified，核实时间2026-10-14T13:32:49.290364+00:00，派发至已核实交付231.818min。
