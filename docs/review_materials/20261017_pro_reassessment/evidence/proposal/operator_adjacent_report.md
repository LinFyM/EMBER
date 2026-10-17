# §15 D相邻稳定性整批工程交付

D144：seen414/576、strict firstG validation168/400；原D72=414/175、S38349=427/165、MT=340/153。D144 held相对D72净−7/得19失26，相对S净+3/得29失26，相对MT净+15/得44失29。

相邻held保留149/175，churn45、Jaccard0.768041；seen保留379/414、得35失35、churn70、Jaccard0.844098。总分相同不能代替成功条件稳定。D72相对S的29新增held成功保留18/29；19原保持损失有8/19被144恢复，全部原行保留。

held task16从24→20（S21/MT10），task31从25→27（S18/MT24），task26从37→33（S38/MT35）；task23仍1/50、task39仍0，breadth7/8。原G增益由D72净+10收窄到D144净+3；不选择每task较好点、不将两点并集当性能，也不从工程通过宣称稳定完整Writer或唯一机制。结果交主讨论按事前§15分支消费；0自动第三块/BCE/Actor/controls/Test。

## 完整学习和输入身份

从PARENT D72完整psi/485 Adam/sampler/两rank RNG到D144，world2→4完整边界迁移，不称bitwise exact。rank0/1恢复原RNG；rank2/3新seed771091015/2330581364来源保存，最终4份RNG及原2份RNG均在完整checkpoint。global72…143对应v0…71，全部原288condition各一次、36task各8次、H1/H2各144；原T/类别/查询来源/完整noise和权重逐条件与D72 receipt一致。

每condition原归一化38层BA一次、/4与rank SUM、AdamW lr1e−4/betas(.9,.95)/eps1e−8/wd1e−4/clip1、native10训练查询0；完整meta两支/Reader/static/F全72更新实际信用与source/T/Actor冻结。90/108/126恢复点、144唯一新科学端点完整；11旧闲置参数仍不造fake使用，485 Adam均step144，scheduler/scaler=None。

## 末点评测与账目

36 seen条件共576原行和400不同held task-video条件，436套新LoRA各76因子/38target完整；seen每condition一套共享16states，held50videos/task无放回。原MT-H1、最终state-video mapping/scene、env/policy noise与D72一致；全部未进入H、无Actor/新实践/测试/固定函数读回，官方horizon/10 dummy settling/10flow/前5动作/成功即停。原MT400缺initial_proprio的旧限制保持，不补采。

976 submitted=976 finished、237361实际环境步（含9760 settling）、0failed partial/在途/未知环境步，符合976/992/525760。13 GPU收据含3失败：01四rank配置拒绝、02frame64 OOM、03frame32 OOM；失败共.349143253GPUh，48已完成失败编译/12未保存更新与≤8失败未落盘编译保留原attempt。72保留流/288编译未被这些物理重算扩为新逻辑训练。所有成本保留，未将失败写零或当discarded profile。

总GPUh=5.428371164，源模型/训练/读取51924861（与6c7c数学代码相同）clean pushed detached；前三源及原件保留。从首次本session观测派发22:22:32Z到实际00:04:44.632788Z退出102.211min，实际派发可略早，时钟未重置。owner04 58.717min，学习19.065min，学习结束至整批退出39.313min；原60–110估计与恢复40–70余wall范围内完成。

四卡frame16完成全72 B4 mean/max=15.404/31.532s，reserved最大26.504GiB。末点六GPU持久队列初始seen/held各3，seen释放接held，实际9读出job；三空seen消费者CPU省略0 GPU starts。各launch保存双节点live/data1独立quota/数量上限；ROOT实占约2.279GiB（48GiB预估内）。

## Held per-task

| task | MT | S38349 | D72 | D144 |
|---|---:|---:|---:|---:|
| 3 | 41 | 43 | 45 | 43 |
| 6 | 7 | 8 | 6 | 7 |
| 11 | 36 | 37 | 37 | 37 |
| 16 | 10 | 21 | 24 | 20 |
| 23 | 0 | 0 | 1 | 1 |
| 26 | 35 | 38 | 37 | 33 |
| 31 | 24 | 18 | 25 | 27 |
| 39 | 0 | 0 | 0 | 0 |

## Held per-suite

| suite | MT | S38349 | D72 | D144 |
|---|---:|---:|---:|---:|
| libero_10 | 24 | 18 | 25 | 27 |
| libero_goal | 35 | 38 | 38 | 34 |
| libero_object | 46 | 58 | 61 | 57 |
| libero_spatial | 48 | 51 | 51 | 50 |

## Seen per-task

| task | MT | S38349 | D72 | D144 |
|---|---:|---:|---:|---:|
| 0 | 14 | 14 | 13 | 16 |
| 1 | 4 | 14 | 13 | 12 |
| 2 | 14 | 16 | 16 | 16 |
| 4 | 8 | 9 | 9 | 6 |
| 5 | 6 | 12 | 3 | 9 |
| 7 | 7 | 12 | 11 | 11 |
| 12 | 7 | 10 | 9 | 8 |
| 13 | 7 | 10 | 7 | 7 |
| 14 | 11 | 13 | 13 | 14 |
| 15 | 3 | 4 | 6 | 7 |
| 17 | 0 | 6 | 9 | 10 |
| 19 | 6 | 11 | 9 | 9 |
| 20 | 8 | 9 | 9 | 11 |
| 21 | 14 | 16 | 16 | 14 |
| 22 | 10 | 15 | 15 | 15 |
| 25 | 2 | 4 | 5 | 4 |
| 28 | 15 | 14 | 16 | 15 |
| 29 | 0 | 3 | 3 | 0 |
| 32 | 2 | 13 | 11 | 12 |
| 34 | 11 | 10 | 10 | 9 |
| 35 | 12 | 12 | 12 | 12 |
| 36 | 1 | 8 | 8 | 8 |
| 37 | 7 | 13 | 13 | 10 |
| 38 | 0 | 1 | 2 | 2 |
| 42 | 16 | 16 | 16 | 16 |
| 43 | 7 | 14 | 9 | 12 |
| 51 | 14 | 16 | 16 | 16 |
| 55 | 16 | 16 | 16 | 15 |
| 56 | 13 | 14 | 14 | 14 |
| 62 | 14 | 16 | 14 | 16 |
| 64 | 16 | 15 | 16 | 15 |
| 73 | 14 | 11 | 13 | 13 |
| 95 | 13 | 13 | 14 | 12 |
| 96 | 16 | 16 | 16 | 16 |
| 97 | 16 | 16 | 16 | 16 |
| 101 | 16 | 15 | 16 | 16 |

## Seen per-suite

| suite | MT | S38349 | D72 | D144 |
|---|---:|---:|---:|---:|
| libero_10 | 33 | 57 | 56 | 53 |
| libero_90 | 171 | 178 | 176 | 177 |
| libero_goal | 49 | 61 | 64 | 59 |
| libero_object | 34 | 54 | 53 | 55 |
| libero_spatial | 53 | 77 | 65 | 70 |

## 原件和边界

唯一ROOT/results.json保存全部原行/配对/per-task/per-suite/相邻得失和两个分组；analysis/completion_20261014/verify.py与acceptance.json为一次CPU实物验收，0新模型或环境，CPU2.54s。checkpoints/90/108/126/144、288完整有效记录、436LoRA+generation、976rows、全失败/exit/费用保留。科学解释由主讨论继续，不把本交付作为最终EMBER目标完成。
