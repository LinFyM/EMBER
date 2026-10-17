# §15既有控制器同教学独立query：整批交付

原124新增有效query及720逻辑配对已完成。Reader90089/144，first-best73，full79，source58，reference127。当前完整探索/选择流程高于source，却低于已存在的同教学初始控制器；后续探索/选择保留有限独立增益，不能据此认证已学会H纠错。

|group|source|reference|Reader900|first-best|full|
|---|---:|---:|---:|---:|---:|
|all144|58|127|89|73|79|
|source_seen_support12|44|47|46|44|44|
|target24|14|80|43|29|35|
|libero_10|3|15|5|2|2|
|libero_90|44|47|46|44|44|
|libero_goal|8|20|14|14|14|
|libero_object|0|21|9|1|3|
|libero_spatial|3|24|15|12|16|
|identity5|20|20|20|20|20|
|changed7/28|4|27|17|6|12|
|new4/16|3|16|12|5|8|
|canonical3/12|1|11|5|1|4|

Reader900→full：R75/G4/L14，churn18、Jaccard.80645；Reader900→first-best：R70/G3/L19。first-best→full：R72/G7/L1，churn8、Jaccard.9；全部变化都在7task/28case，其余29task116行严格保留full原执行。
新16行first-best5→full8（R4/G4/L1）；source复用12为1→4（R1/G3/L0）。task4 +2、12 +1、5 +2、19 +1，task14交换1成功、task1/43无净变；没有按query改选点。参考对source新增70个成功，Reader900保29、first-best18、full23。
breadth为source20/ref35/Reader29/first23/full25任务；全部per-task成功集合、十对RGL、churn/交并及target/support/identity分层见results.json。

实际首提案前缀为244practice episodes/69702步/25提案与25教学读取/889frame-equivalent；full为632/170804/122/122/4008。追加真实97提案与388episode/101102步只在当前有限query净增6；前缀截断是反事实描述，不声称本批实际节省了旧费用，也不推反事实GPUh。Reader900本批只加载已有文件、0实践/Reader forward。
该有限nonheld面板源于变化theta，不是meta9 single-checkpoint或held资格；无H/video causal controls，不能唯一归因H/共享PG。source早停11task的44case未读取当前教学/未提案，source42/Reader44/full42不能归新增反馈；19个最终source任务的原source45/full44同参数差异不作学习归因；强MT153/G165/D175/168及此前所有正负结果保持原口径。

## 恢复、费用和真实退出

首次派发05:27:06.038Z、首次带时钟准备05:27:32.260Z；原owner05:42:29.601265→05:46:33.197387Z exit1，恢复授权05:49:56.934Z，新owner05:56:36.703370→06:01:29.252261Z exit0。收件至实际退出34.387分钟，原预计45–90分钟；故障/修复与等待全部保留，未重置旧参考09:05:48.038。
新费用精确0.364542476932GPUh（8job、7exit0/1exit1，旧故障6job0.238637107611计入）；0未回收GPU退出区间。124有效新行72成功/52正常科学失败，control28016+dummy1240=29256步；32旧unknown完整保留15280上界，观察1955仅lower bound，故新批总步区间[31211,44536]，实际账本保守44536。
物理提交156（80原valid＋32旧unknown＋32一次恢复＋12首次），累计control42976/dummy1560/total44536，均在§15.5的172/66160/1720/67880内。原32 status不改、费用不回填，具名各一次；原16通用额度实用0/16，不借未用恢复额度、不滚动重置。原80有效行均仅一个attempt，原12仅首次；0正常科学失败重跑。
原§14的9.325162564582294GPUh/776提交/207328步与所有更早独立区间均不重计；本批不是重新训练或新实践。

## 吞吐、资源与代码

有用native plan/s：8槽5.297；16槽6.924/7.373/8.100；20槽8.232；24槽9.388/9.803。24最高allocated12591MiB、reserved32922MiB；32槽在MjrContext framebuffer创建失败，未把低allocated误当完整EGL容量，也不把错误说成CUDA OOM。恢复仅44行用两worker24+20，保留official分辨率/双相机/RNG/数值合同；不额外重跑32或增加profile行。
完整runner.close共8次，释放156个环境子进程；两节点GPU/本批owner/环境均0存活、0zombie，退出UID PSS分别0.348/1.741GiB。每launch live双节点/6或8/node6与nodeRAM/PSS及storage-node独立quota准入；foreign未动。资源原件见resource_exit/useful_throughput/GPU_exit。
新增ROOT实耗约24.4MiB，开发树/双冻结/必要CPU小夹具也远低2GiB峰；全部新增data1，无模型/数据/旧148GiB特征复制，无新H/视频/训练。
源身份：首冻结e81fa5d419f3f566c50f30fe990b30b5cdc1b82c；恢复冻结1ff30c7e159e5116ac30e5a0f4a743f7f51c5301。原run_contract/manifest/attempt001保留，revision另档；CPU消费者/结构guard与原子账本接入均有证据。最终Git集成/push/clean与记录写窗口释放见Git_delivery.json及main交付回执。
当前124目标已完成，停止新计算；0自动训练/新实践/held/Test/追加探针，由main消费后科学裁决。
