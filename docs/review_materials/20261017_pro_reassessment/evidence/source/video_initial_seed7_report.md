> 既有报告摘存；来源：`runs/source_policy_space_compiler_20261015/video_initial_seed7/analysis/engineering_completion.md`。历史执行状态不构成当前授权。下文未入Git的原目录链接改为文字引用；可读结果投影见[证据地图](../../EVIDENCE_MAP.md)。

# §13真实视频初始编译：完整工程交付

**本批已完整执行，但未达到held性能目标。** 两点validation均低于MT153/G165/D175/168；按预登记规则冻结900不表示方法合格。

本批只检验H为空的初始提案：source-only/full38r8，固定已有linear64，fresh纯图像Reader；没有实践/RL、Test或视频因果controls。
按两个strict correct400高者/平局早冻结900。完整两点、全部原始case/生成LoRA均保留；训练域收益和held分别报告。

|update|seen /576|validation /400|参考新增288保留|source218损失|
|---|---|---|---|---|
|900|334|117|116|13|
|1800|413|111|193|12|

强参照保持：source218/ref502/native144216/operator1170224/linear484（seen576）；held MT153/G165/D175/168（各400）。同八task MT67/128仅旧子集，0新增MT。
全部1952行均真实执行，五identity也未复用80源行；全部相邻R/G/L/churn、success交并/Jaccard、breadth、各task/suite/target/support/identity见results.json。

|task|suite|source|reference|linear64|900|1800|
|---|---|---|---|---|---|---|
|0|libero_spatial|5|16|13|8|12|
|1|libero_spatial|0|16|16|2|12|
|2|libero_spatial|7|16|16|13|15|
|4|libero_spatial|0|14|11|8|11|
|5|libero_spatial|2|15|15|12|16|
|7|libero_spatial|0|16|15|14|15|
|12|libero_object|0|14|14|7|8|
|13|libero_object|0|13|11|1|0|
|14|libero_object|0|15|15|13|12|
|15|libero_object|0|14|14|0|12|
|17|libero_object|0|14|12|0|0|
|19|libero_object|2|14|14|8|3|
|20|libero_goal|0|9|7|7|6|
|21|libero_goal|14|16|16|13|15|
|22|libero_goal|1|16|16|13|15|
|25|libero_goal|0|10|10|0|6|
|28|libero_goal|11|16|16|16|16|
|29|libero_goal|0|7|7|0|5|
|32|libero_10|0|14|15|1|13|
|34|libero_10|4|7|6|3|5|
|35|libero_10|1|12|10|7|11|
|36|libero_10|0|10|11|3|12|
|37|libero_10|1|15|15|6|9|
|38|libero_10|0|6|5|4|3|
|42|libero_90|16|16|16|16|16|
|43|libero_90|9|15|16|9|13|
|51|libero_90|15|16|15|15|16|
|55|libero_90|16|16|16|16|16|
|56|libero_90|14|16|15|15|14|
|62|libero_90|16|16|16|16|14|
|64|libero_90|16|16|16|16|16|
|73|libero_90|7|13|11|9|12|
|95|libero_90|14|15|15|15|16|
|96|libero_90|16|16|16|16|16|
|97|libero_90|15|16|16|16|16|
|101|libero_90|16|16|16|16|16|

实际分组、held与保持：

|update|target24 /384|support12 /192|seen breadth /36|held breadth /8|
|---|---|---|---|---|
|900|159|175|32|5|
|1800|232|181|34|5|

|held task /50|MT|G165|D72|D144|900|1800|
|---|---|---|---|---|---|---|
|3|41|43|45|43|21|29|
|6|7|8|6|7|16|35|
|11|36|37|37|37|42|23|
|16|10|21|24|20|0|0|
|23|0|0|1|1|0|0|
|26|35|38|37|33|20|16|
|31|24|18|25|27|18|8|
|39|0|0|0|0|0|0|

|suite /100|MT|900|1800|
|---|---|---|---|
|libero_10|24|18|8|
|libero_goal|35|20|16|
|libero_object|46|42|23|
|libero_spatial|48|37|64|

|配对 before→after|retained|gained|lost|churn|Jaccard|
|---|---|---|---|---|---|
|seen source→900|205|129|13|142|0.590778|
|validation MT→900|84|33|69|102|0.451613|
|seen source→1800|206|207|12|219|0.484706|
|validation MT→1800|71|40|82|122|0.367876|
|seen 900→1800|294|119|40|159|0.649007|
|validation 900→1800|61|50|56|106|0.365269|

训练域表现提升、held总数下降且相邻case交换明显；不能由seen提升认定教学内容已泛化。五identity两点均80/80。此批不包含controls，不能由这些结果单独认证视频动态知识的因果作用，也不将有限负结果扩大为全部视频或自身交互学习无效。

运行与优化：

全2345视频/75873完整帧；原始stride5+terminal，producer已采样不再stride；兼容旧纯图像prefix只读引用，0held离线labels。
唯一1800有效更新/201600 FM query/7200原始+7200辅助读取；67 Reader Adam状态/constant scheduler/scalerNone/rank RNG/完整cursor/topology均已验收。
实测1/2/4/6卡中位13.159/6.633/3.901/3.329秒；修正提前留卡，117完整状态续6卡到900、4卡与两读出并行。只取消尚未完成初始化的4卡阶段，0有效更新丢弃/重算；98.420s/0.109355895GPUh仍计入。
FM实际测试7/14/28/56/112，最大112实际allocated约38.5GiB、没有比56更快；逻辑112/aux4不变。特征4..128有用测试，读取复用node实际16/64；环境8/16按live余量及node CPU准入，无固定host上限。
本批GPUh 13.033086503（含加载/保存/取消/失败）；与四个旧source家族ROOT合计区间[69.830619989,69.938365494]，旧六loader无exit不确定区间保留，非整个项目总费。
实际交互账本：{"complete": {"control": 486758, "dummy": 19520, "submissions": 1952, "total": 506278}}。
最终ROOT实耗144.038GiB；原峰192GiB预算，实际临时/atomic额由资源记录说明。
全部实际子进程已退出；14份完整close回执、双节点无本批CUDA/模块进程，未触碰foreign。CPU记录是各node实际快照/保守准入，不伪称连续总PSS峰。

原派发22:44:59.573Z、准备22:45:07.991Z、首次owner23:07:52.528946Z、原参考09:05:48.038Z均保留；attempt001 exit1是物理调度纠正，attempt002实际退出见owner_exit.json。
代码：0..117及特征93e27530；117后训练/两点读出5568f80c，旧冻结不热改。canonical最终记录commit/Git clean状态见Git_delivery.json。
不从这两点自动开后继H/q/RL、controls、Test或加量；主讨论消费后科学裁决。

费用、资源与验收原件：

实际GPU费分项：features 1.387482945、train（含初始化取消）6.975695974、readout 4.669907585 GPUh，合计13.033086503；21份退出结算、20成功+1已登记取消初始化。环境0失败/未知/有效行重跑，独立16额外环境尝试全部未用。实际486758 control+19520 dummy=506278步，1952提交；上限1968/679360control/19680dummy/699040total不变。0任务内适应与1952次实际评测交互须分开理解。

最终退出2026-10-16T01:49:06.528824Z、exit0；从首次派发22:44:59.573Z至退出184.116分钟，位于原2.5–4.5h估计内。ROOT最终144.038GiB，data1独立quota最终Blocks1899920384KiB、Quota2147483648KiB（2TiB），shared容量与launch前quota均另存；旧data0未新增产物。

资源快照中GPU01/02最低观察MemAvailable约261.86/232.29GiB；最大单worker进程树PSS约80.78/73.26GiB，分别80/78个快照，不能相加当作同时峰值或连续node总PSS。读出最大Torch allocated13128.69MiB/reserved31206MiB；训练最大112物理批allocated39443.48MiB。14份完整close回执、136完成shard，终态双节点无本批CUDA/模块/环境子进程。所有foreign未发信号。

`完整数值与逐case引用`（原运行目录：`results.json`）、`科学产物/完整checkpoint验收`（原运行目录：`acceptance.json`）、`资源/费用/退出审计`（原运行目录：`resource_exit/acceptance.json`）、`真实owner退出`（原运行目录：`../owner_exit.json`）。CPU真实消费者包括完整逻辑流/等权/teacher-query排除、固定D与FM真实梯度链、零head首步及后续body梯度、67 Adam状态、恢复cursor/RNG/topology、held配对和相邻集合；结构guard无hard项。新checkpoint1800/900与1952套实际LoRA均保留，0未运行部分冒充完成。
