# §17 H V/VL完整工程交付（2026-10-14）

ROOT：runs/video_guided_proposal_writer_20261014/image_only_gemma_with_history
唯一最后attempt：image_gemma_H_paired_publication_resume_03，08:56:12Z exit0；complete.json/results.json及全部原行齐全。实际工程验收通过，科学取舍交主讨论01a13399；0自动后继/Test。

| 方法 | seen /576 | strict held /400 |
|---|---:|---:|
| V：无直接L、真实MT-H1 |372|149|
| VL：匹配直接L、真实MT-H1 |353|151|
| 强MT |340|153|
| 首G38349 |427|165|
| D72 |414|175|
| D144 |414|168|

held其它强参照：旧完整流程163、Phi167。两臂各breadth6/8；V对VL seen R325/G47/L28/net+19，held R121/G28/L30/net−2/churn58。本次held未超过MT；不由seen或内部BA声称视频动态知识、held收益或相对旧G的单一语言因果。

held任务順序3/6/11/16/23/26/31/39：V=40/2/36/12/0/35/24/0，VL=40/6/40/5/0/37/23/0。suite按spatial/object/goal/long：V=42/48/35/24，VL=46/45/37/23。全部seen任务和其它分布及保持得失见results.json。

held对MT：V R122/G27/L31/net−4/churn58；VL R125/G26/L28/net−2/churn54。对首G：V R124/G25/L41/net−16，VL R124/G27/L41/net−14；对D72：两臂R128，V G21/L47/net−26、VL G23/L47/net−24；对D144：两臂R125，V G24/L43/net−19、VL G26/L43/net−17。

两个2160全144节点BA=.874476357/.866080762，按原共同规则都接到唯一8640，0中点选择；8640全144 BA=.526683348/.532420738是定位读数，不替代闭环性能。

一次CPU原件验收4.974s：每臂8640有效逻辑B4、25920 rank更新原收据、284/290完整Adam均step8640、sampler及三rank完整RNG/共同fresh种子2026101417/18/19齐全；144注册任务/教学流与两臂实际流一致。Reader/history/decoder/output真实非零梯度、每更新trainable-dependent memory重算，冻结source/MT/E，无旧no-H权重Adam复用。

每臂seen固定36条教学编译供576最终query，held8task×50各50不同教学编译供400最终query，共每臂436完整38-target/76因子LoRA。144是训练教学条件，不是seen编译数；原面板/全部scene、state/video/环境policy噪声、初态与精确MT-H1前缀保持，policy有L，V Compiler无直接L、VL仅多Reader直接L，H无execution hidden。教学习得特征重算，旧pure cache144/4679帧只读复用，0再算Gemma；H冻结vision/projector共享36cache，held有界流式。

原失败47提交上界及22990未知步保持。完整1952final=496150环境步（含dummy），确认失败partial3460步，known499610；保守总522600。累计提交上界1999、failed上界47、0在途；已证reset范围1961–1997，保守提交不是精确reset。恢复新增failed0，有效2015/63/1043040内完成；原1968/16已经越界的历史和唯一06:42修订保留，0自动额度扩大。

新ROOT费用12.571101360GPUh/32 GPU jobs，含4失败job；carry-in=0。旧no-H按范围纠正中止的2.371076951GPUh/4320 CP/未保存范围及全部原件另存，不作科学阴性。共享MT依赖：train36/8131步及held400/114908步，共436次/123039步，两个臂不算两份；其专属历史GPU费用未知，不能借整§23费用代替。新增practice/训练query/Test均0，方法具有旧适应依赖，非零交互。

训练7f05e3e6；首16 held实际compile/execute b9c8f824；余784 held8ecddd64。三者均为clean pushed detached且已集成main。旧02原16行由CPU双臂publisher结算，未重复forward/episode；14既有队列测试及真实原行CPU消费者已通过。原失败/旧源码完整保留；只修render/prefix共驻与双臂发布计数，不改数学、标签、选择或官方控制。

actual六卡：GPU-node-A:3/2/0/4/5+GPU-node-B:6，独立持久共享long-first队列/slots3/3/3/3/2/2总16保护仍余16新增失败额度。原H104/video56与H104/held最长video91消费者皆计入有效固定final；后者compile allocated12.883/reserved13.838GiB、execution9.416/reserved10.133GiB，4有效final/160.065s。32job已结算/owner及消费者已释放。ROOT约7.257GiB<48GiB增长初估，引用大资产不复制。

从H04:19:48.975Z到完成276.385min（原100–190）；06:42:01.831Z裁决到完成134.171min（原35–70）；03从07:59:09到完成57.043min（修正45–75）。从最初no-H02:54:45.949Z到完成361.436min，范围停止、准备及验证/发布遗漏的延误和2.371GPUh不清零。

原件索引：results.json；analysis/engineering_completion_20261014/acceptance.json、verify.py；history_manifest.json；input_manifest.json；endpoint_decision.json；arms/{V,VL}/checkpoints/update_00008640/state.pt；arms/*/readouts/{seen,validation}/rows与loras；jobs/*/attempt_*/{launch,exit,episode_events}.json*；三attempt exit.json；inputs/H_readout_engineering_amendment_20261014.md/json与run_contract原snapshot；analysis/failure_interaction_audit_01.json；coordinator/03/longest_registered_consumer.json。

Git：main 05866b2d82a81781bb487cdeb9b173acf374341e，五份文档相关diff与diff --check通过，clean pushed；本次没有科研源码改动，未重复GPU/模型/环境/测试。

整批交付：主讨论01a13399已processing_verified，turn [operational-id]；03 consumed去重完成，仅确认消息处理开始，不冒称科学裁决已经完成。
