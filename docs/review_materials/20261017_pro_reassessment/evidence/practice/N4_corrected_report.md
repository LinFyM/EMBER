# 10条件joint schema纠正：完整工程交付

唯一有效方法读出为本子根 results.json / aggregate.json；原父根 results、corrected_results、LoRA、H、sealed与费用全部保留。仅内部8任务×16新初态，不是strict held400或方法资格。

CPU原件验收 0.686s；没有模型/环境/DTW评分/GT或sealed重读。10个完整38-target/r8因子、Adam、logical cursor与rank RNG齐；26个native更新18接受，停止为5 all_four_success、2 two_no_acceptable_proposals、3 budget。

原20唯一MT bootstrap与160首次MT snapshot分支只通过180个明确旧producer路径的schema视图复用；原文件只读，新的canonical H不二次转换。first draw保留原point/u/v/四格override/执行mask与后续noise身份，baseline由原预测重新计算；fresh MT300/空Adam，不导入旧proposal/Adam/接受。原9458逻辑parent control及20336首次分支control照占编译预算，新增物理成本0。Phi1120/240patch与原预测只读引用，0重训/GT重算。

512逻辑行=352未改原行+144新增final+16同condition原/新均明确MT300的原行复用（task20/video0）。全局task/init、env/policy root、first noise、初始proprio、official dummy10与完整final参数身份已核。累计实际final执行784，不能只报512成本。

| task | MT | video0 | video1 | no_video |
|---|---:|---:|---:|---:|
| 0 | 12 | 13 | 13 | 13 |
| 1 | 4 | 2 | 4 | 4 |
| 12 | 8 | 8 | 7 | 8 |
| 13 | 7 | 8 | 7 | 9 |
| 20 | 10 | 10 | 12 | 10 |
| 21 | 15 | 16 | 16 | 16 |
| 32 | 2 | 1 | 5 | 4 |
| 34 | 9 | 6 | 6 | 10 |

| suite | MT | video0 | video1 | no_video |
|---|---:|---:|---:|---:|
| libero_spatial | 16 | 15 | 17 | 17 |
| libero_object | 15 | 16 | 14 | 17 |
| libero_goal | 25 | 26 | 28 | 26 |
| libero_10 | 11 | 7 | 11 | 14 |

合计 MT67 / video0 64 / video1 70 / no_video74；breadth均8/8。no_video原14更新0接受、全部MT300；其相对MT的+7属于同参数实际执行差异，不能称学到能力。原受错误影响video0 60/video1 58保留；新旧同臂原行交换V0得8失4、V1得15失3，仅为实际两批对照，不据此将全部变化归因schema。

| arm | reference | retained | gained | lost | net | churn | Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|
| video0 | MT | 55 | 9 | 12 | -3 | 21 | 0.723684 |
| video1 | MT | 58 | 12 | 9 | 3 | 21 | 0.734177 |
| no_video | MT | 60 | 14 | 7 | 7 | 21 | 0.740741 |
| video0 | no_video | 56 | 8 | 18 | -10 | 26 | 0.682927 |
| video1 | no_video | 59 | 11 | 15 | -4 | 26 | 0.694118 |
| video1 | video0 | 53 | 17 | 11 | 6 | 28 | 0.654321 |

| task/arm | updates | accepted | stop | final |
|---|---:|---:|---|---|
| 0/video0 | 1 | 1 | all_four_success | joint_schema_corrected_task_0000_video0_proposal_000 |
| 0/video1 | 1 | 1 | all_four_success | joint_schema_corrected_task_0000_video1_proposal_000 |
| 1/video0 | 3 | 1 | two_no_acceptable_proposals | joint_schema_corrected_task_0001_video0_proposal_000 |
| 1/video1 | 4 | 2 | budget | joint_schema_corrected_task_0001_video1_proposal_002 |
| 20/video0 | 2 | 0 | two_no_acceptable_proposals | MT300 |
| 20/video1 | 2 | 1 | all_four_success | joint_schema_corrected_task_0020_video1_proposal_001 |
| 21/video0 | 3 | 3 | all_four_success | joint_schema_corrected_task_0021_video0_proposal_002 |
| 21/video1 | 2 | 2 | all_four_success | joint_schema_corrected_task_0021_video1_proposal_001 |
| 32/video0 | 4 | 3 | budget | joint_schema_corrected_task_0032_video0_proposal_002 |
| 32/video1 | 4 | 4 | budget | joint_schema_corrected_task_0032_video1_proposal_003 |

新增500提交/500完整返回=256新snapshot分支+100候选+144final，0失败partial/未知步；control98748+dummy2440=101188，满足936/228126/3360/231486。新增 1.873172839151GPUh/carry0。父批（含无效MT、额外8320dummy、旧工程失败/所有已发生计算）5.131869526307GPUh保持；累计2184实际提交/返回、433706control+18000dummy=451706步，7.005042365458GPUh，仍在2620/573838/18920/592758联合边界内。16失败余量未消费。

GPU正式22:25:51.829Z→22:44:51.958Z，共19.002min；六持久worker使用GPU-node-B:1/7/2/0/4/6，每波16slots，20个动态就绪项全complete。最大allocated 11.809GiB/reserved 15.539GiB；16为本批实际最大原子分支/final wave，不能为填显存增加科学请求。0新profile；本子根实际约5.767GiB<24GiB新增预估。双节点live证据 process_release.json确认六job exit0且本用户CUDA为0。原首次18:16:48.556Z/3–6h及本次45–90min不重置；GPU启动/退出与原派发分别记录。

训练Phi来源8801e701；旧执行8801e701/fe034f78与MT参照a986e7b2；本次纠正编译/读取clean pushed detached afe6acc5。原参数21:01:01.939804Z冻结→原sealed21:21:15.600653Z→MT修订→本次纠正冻结22:44:51.941135Z，明确sealed顺序例外。未重开GT、未将GT用于新参数/输入/选择；旧sealed受旧own schema影响的边界保留。

验收代码与原件：verify.py、acceptance.json、condition_records.json、paired_metrics.json、resources.json、process_release.json。当前无active计算、自动后继、第二waiter或Test。完整科学消费与接续由main01a13399负责；旧schema一次性导入桥只在本批科学消费完成后按登记生命周期退休，本回报不自行提前删除共享源码或formal冻结。

Git交付：main 77b1390f clean pushed，仅五份现有记录/设计增31行，无源码变更。唯一整批已通过Steer发main01a13399；实际回执processing_verified=true表明接收方开始处理，不代称科学裁决完成。receipt为main_joint_schema_corrected_complete_receipt.json，ROOT/attempt的consumed.json完成判重登记。
