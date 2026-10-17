# §14同T BA/native两臂完整工程交付

仅BA的D为seen414/576、strict firstG175/400；BA+native的C为393/576、158/400。原G38349为427/165，强MT为340/153。所有结果是唯一72末点，完整原行保留；本文件交付工程事实，科学取舍由主讨论按事前合同消费。

| 面板 | MT | 原G38349 | D：仅BA | C：BA+native |
|---|---:|---:|---:|---:|
| seen576 |340|427|414|393|
| strict firstG400 |153|165|175|158|

D的held相对原G得29失19、净+10，对MT得46失24、净+22；C相对原G得27失34、净−7，对MT得33失28、净+5。C对D held得22失39/净−17，seen得36失57/净−21。D seen对原G得35失48/净−13；C得32失66/净−34。validation覆盖D7/8、C6/8，D task23仅1/50、task39仍0；单点不能认作相邻稳定或视频/H必要性已证实。

| validation suite | MT | D | C |
|---|---:|---:|---:|
| libero_spatial | 48 | 51 | 46 |
| libero_object | 46 | 61 | 49 |
| libero_goal | 35 | 38 | 33 |
| libero_10 | 24 | 25 | 30 |

| validation task | MT | D | C |
|---|---:|---:|---:|
| 3 | 41 | 45 | 41 |
| 6 | 7 | 6 | 5 |
| 11 | 36 | 37 | 38 |
| 16 | 10 | 24 | 11 |
| 23 | 0 | 1 | 0 |
| 26 | 35 | 37 | 33 |
| 31 | 24 | 25 | 30 |
| 39 | 0 | 0 | 0 |

固定72点BA误差均值S/D/C=.152745613/.323312608/.481264834，两臂各5改善67恶化；native到T均值.002205425/.003837930/.005041636，D12改善60恶化、C13改善59恶化。共同诊断BA+lambda*native为.305491226/.589123902/.830443661，两臂各6改善66恶化。完整H1/H2、T来源、T强弱、所有空类别及每query权重见results.json.fixed_input_readback；72子集不称完整288曲线，D没有native训练forward，不用内部误差替代闭环。

两臂同完整G38349/bank9、fresh同AdamW/原288条件/36task各8遍/72有效B4；BA每condition一次、/4和rank SUM、C固定lambda69.25903014646993。原query/noise/目标/缺类别、全部4组前端信用及meta双支通过实际原记录核验，source/Actor/T冻结。CP18/36/54仅恢复，72唯一末点，完整Adam/sampler/rank RNG/schema/topology保留；485实际Adam状态步均72，11个旧Reader.empty/static未使用参数无状态，两臂一致。

D/C训练native学生查询分别为0/5376，T相同；末点各1344新student native10、各72函数输入，全无梯度。944个唯一完整38-target LoRA的全部76因子元信息核验，seen复用其中72个H1编译，0新编译。有效训练576次编译另计；原OOM部分训练native调用未落盘，每student/T未知上界96仍保留，未写0。原S/T响应、BA校准、模型及旧MT-H/成功支撑只读引用，旧实践/父批费用不重复计入本批，完整编译流程仍有真实交互前置成本。

总账1952提交/1952完整返回、477689环境步（1952×10 settling含在内）、0未返回/0失败partial/未知额外0，符合1952/1968/16/1043040。GPU费用14.577632926h，共29GPU收据、2原OOM，另保留原CPU误拦/恢复及全部原件。owner03 seen恢复40.753min，原18:30:39Z派发→21:41:38Z退出190.992min，超过初估90–165；恢复后的30–60min估计内完成。eval reserved峰值14.514GiB，ROOT约4.720GiB，data1每launch独立quota已留档。

资源为D world2/C world4先4+2学习，全部同节点DDP；依赖就绪函数/400队列及时接卡。seen恢复GPU-node-A五/GPU-node-B一、六独立world1初始D/C各3worker，D释放接C剩余队列，实际9个seen GPU作业；三个无未领取shard的D消费者CPU省略，没有空占卡。全部GPU子消费者已退出并结算，唯一整批回报已消费；没有另一等待者或阶段通知。

GPU模型/训练/函数/validation计算源92d0a234；seen和计账读取源045ee602，仅src/ember/proposal_writer/analysis.py发生源码窄修。两棵source worktree均clean pushed detached保留，开发分支/树已退休。最终文档main commit在complete.json登记。0 Test/新实践/新T补录/中点选择/科学重跑/自动后继；本有限批完成，完整EMBER目标未完成。

原件：runs/video_guided_proposal_writer_20261014/operator_native_joint_learning/results.json；runs/video_guided_proposal_writer_20261014/operator_native_joint_learning/analysis/completion_20261014/acceptance.json及verify.py；coordinator/operator_native_seen_resume_03/exit.json与whole_batch_exit_delivery.json。旧01/02 exit1原样保留，不改写为成功。
