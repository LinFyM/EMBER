# EMBER progress

更新时间：2026-09-06 17:33 CST。

## 当前目标与授权

- 最终科学goal持续active；权限内持续自主推进。Active design为`docs/joint_process_policy_writer_design.md`。
  最终资格仍为validation8 strict single-checkpoint >145/400及预登记相邻/跨视频稳定、breadth、四suite与Goal/Long贡献；
  同图random候选和冻结后视频因果controls仍待完成，方法冻结后才做32/8 fresh及Test。没有selected checkpoint。
- 完整LoRA四任务短学习32 fit/held50/54、64为64/62（各150）已sealed，主要改善Goal；meta73四点validation screen15/19/19/19亦已sealed。
  历史§173/174保留完整边界，弱混合run不续训。本轮继续同图，检查目标学习与迁移的区别。

## Target18对照已完成

- Frozen clean pushed detached351feb48684c019130d5d5d1b44c900abbaeddb9，world6、gpu01物理0/2/3/4/5/6、NUMA0/0/1/1/1/1。
  完整38-target rank16、4,750,208 Writer参数、无carrier、source/observer冻结；fresh component/AdamW。
- 仅移除55meta objectives，保留原18target IDs、K1/两fit视频、micro8、source/normalizer、128-step LR与全部per-target query/video/RNG。
  128updates、2304task executions、18432action rows，两个fit视频各64exposures；全部2304项与meta73匹配，短四任务192项重叠目标记录匹配。
- Train552.64秒、Panel-B390.73秒、runtime996.40秒（另计启动加载），peak reserved34.61GiB；run232MiB，临时cache自然回收。
  四枚checkpoint及全恢复状态保留，所有held/Panel-B backward0，launcher exit0。Run为
  `runs/outputs/pi05_ecp_prw_complete_target18_s128_351feb48_gpu01p023456_20260906`。
- 五个受监督target在32/64/96/128的全fit+held正功能收益为2/5、5/5、5/5、5/5。128 held benefit为72+.001732、77+.004774、
  83+.007476、93+.004031、94+.022396，均高于meta73；target79+.000962，七个meta诊断全负。内部功能收益不替代行为判断。
- 四套primary bank及四组screen80全部完成，所有launcher/worker exit0。32/64/96/128为17/17/20/16；逐Spatial/Object/Goal/Long为
  0/10/7/0、0/7/6/4、1/10/8/1、0/9/7/0，nonzero-task breadth2/5/6/3。64新增Long4次，后续只保留1次再全部丢失。
- 相邻R/G/L为11/6/6、14/6/3、16/0/4，churn12/9/4，Jaccard.478/.609/.800；final相对meta73为14/2/5，相对SFT为12/4/12。
  96的6-task breadth含4个singleton；四点均低于同prefix SFT24，没有稳定累积。目前不扩strict400、不续同run、不扫role比例；
  这只是评测投入决定，不是400-row资格裁决。未使用negative/Test，历史§175/findings§174登记。

## 训练任务行为诊断已完成

- 在读取上述screen结果前，已登记两run固定terminal128、原Spatial2/Goal20/Long38（global[2,20,38]）的fit/held strict150。
  原first-fit demos[3,5,2]、held[49,49,48]及50states固定，共四套single-checkpoint/600新rows；不用于最终checkpoint选择。
  目的是区分受监督任务行为是否恢复与未见task迁移；这些结果仍不能唯一断定occupancy、覆盖或初始化为根因。
- Clean pushed detached9998f204247c0b3d4bd33762e18e895318d0f991，工作目录
  `/data1/user/ymdai/projects/EMBER-worktrees/prw-complete-training-diagnostic-20260906`；只增加4个eval配置和文档，训练配置/source/scripts/tests不变。
  4个真实eval loader通过，四套3-task LoRA banks都已生成/exit0；复用每run驻留runtime，不重新训练或更新任何checkpoint。
- 四组评测均已完成/launcher和workers exit0：meta73-fit gpu02p4/NUMA1（PID1117）、meta73-held gpu02p6/NUMA1（PID1604）、
  target18-fit gpu01p0/NUMA0（PID3708470）、target18-held gpu02p0/1/3/NUMA0（PID69214）；每GPU3workers，执行期最多6张实际工作卡，现已全部自然退出。
  使用live显存/util/process准入，保留低负载peer，不做抢占。launchers在`.codex/tmp/prw_complete_training_diagnostic/`。
- 原18与SFT全部train24的支持差异已在`target_training_support.json`记录。进一步核验SFT历史400步实际230400queries、每task9600、
  50episodes、rank128；当前每task1024、16个Panel-A episodes、rank16。闭环可配对，训练剂量/支持/容量/优化未匹配，不能由差异直接推出根因。
  完整事实见`sft_training_recipe_context.json`，不据此擅自长续训。原source47/SFT109/A2m64 79为配对参照，旧Writer143只作count参照。

## 当前证据与资源

- Canonical analysis：`runs/analysis/pi05_ecp_prw_complete_target18_20260906`。`comparison.md`/`closed_loop_comparison.json`为完整validation screen；
  `decision.json`为投入决定；`functional_comparison.*`、`actual_training_schedule.json`为训练侧证据；
  `local_behavior_comparison.*`为本轮完整600-row诊断：meta73 fit/held32/36，target18均53；逐Spatial/Goal/Long为
  27/5/0、27/6/3、39/14/0、37/15/1。两run fit R/G/L27/26/5、held26/27/10；target18跨视频45/8/8、Jaccard.738。
  恢复主要在Spatial与Goal，Long仍弱，且低于short4的64/62。历史§176保留完整边界。
- /data1最近live quota722850436KiB/1073741824KiB soft，shared84TiB available；全部本轮物化/验证/诊断新增峰值<8GiB。
  精确runtime合同、两节点GPU证据、quota、launch命令与日志保存在analysis/launch和各run launch目录。
- main已集成并推送。351feb48训练/screen frozen tree、041aff55历史mixed tree保留inactive，9998f204诊断tree当前复用于下述只读轨迹诊断；旧A2 random不恢复。
- 下一步定位实际动作阶段失败，再确定训练或数据干预；继续参考专家原文§6/§8及历史SEOD/GOMQ/guard边界，不以functional或一次screen波动推翻全图。

## 动作阶段定位

- 固定terminal128 held视频、原三train tasks与states0/1/2的18条轨迹已完成，两launcher exit0；复用frozen9998f204 canonical rollout，
  batch3仅作illustration，不改strict150/80、不选择checkpoint。两arm均3/9，双相机与BDDL阶段证据在analysis下`behavior_replay/`。
- target18 Long三条均曾放好一个壶，但均未完成第二个，包含碰倒/放置后丢失与后续抓取失败；Goal有错误抽屉与接触失败，Spatial也有放置失败。
  不能统一归因为时序或occupancy。历史直接rank16专家task93仅1–5/50；实际1024训练queries有530位于恰一壶已放置阶段，后半阶段并未漏采。
- 同Long任务的teacher-state continuation已完成：原Panel-A demos12/7/38，固定one-placed-released/later-contact两阶段，两terminal arms共12条，
  不训练、不选择、不读negative/Test。恢复原demo XML/state，zero settling避免破坏抓持，额外完整520 horizon作为明确的接续机会，不冒充官方初始化。
  gpu01p2/NUMA0 meta73 PID3817477、p3/NUMA1 target18 PID3817826，总2卡；preflight为724831152KiB/1073741824soft、shared84TiB，额外峰值<2GiB。
  具体states、labels、命令和边界在analysis下`teacher_state_continuation/registration.json`；driver复用原source，两个checkpoint不变。
  两arms第一阶段均0/3，后段接触均2/3；初始predicates匹配labels，两个launcher exit0、wall261.13/256.99秒，GPU自然释放。
- 已登记剩余15个训练targets的terminal128 held-video screen10，并复用原3tasks的10-state prefixes，组成两组完整18-task/180-row训练breadth。
  不训练、不选择、不改validation；两套15-task bank已完成且exit0，各runtime prepare114秒、物化41秒。
  新frozen detached1be11184，当前meta73 screen在gpu01p0/2/3（launcher允许NUMA0,1，workers按GPU绑定0/0/1）PID3862038，
  target18 screen在gpu02p4/5 NUMA1 PID506029；各每卡3workers，总5卡。实时低负载peer保留，精确preflight与命令在analysis/launch。
  quota726340528KiB/1073741824soft、shared84TiB，现analysis3.4GiB、新增峰值<2GiB；三配置与active design登记，完整探针见历史§177。
