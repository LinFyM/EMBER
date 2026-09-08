# Horizon K1 冻结接口与行为诊断

日期：2026-09-09。承接[已有证据审计](horizon_k1_evidence_review_20260909.md)，本轮按Owner新授权实际运行冻结分析实验。对象固定为原fresh K1 checkpoint200/600；所有Writer、Meta和source数值保持，无optimizer更新、无Test、无最终shuffle/reverse controls。A/A2/A3与两点行为分析均已完成；本报告区分已确认依赖、实际行为和仍未识别的根因。

## 1. 已有结论的变化

视频P4和两个compiler cross分支都有真实功能作用，不能继续使用“视频路径断开/完全被忽略”作为当前解释。原参数对compiler语言初始项的依赖明显大于对单个cross分支的依赖，但A3已把首层语言检索与残差内容分开，但后续层和两个cross之间仍可能补偿，不能把差值直接命名为语言捷径。

行为图像已定位到不同失败阶段：Spatial两个任务在200的固定四状态中都操作了错误物体或错误实例，具有实际抓取搬运动作；Long炉灶任务则同时存在错误物体、正确moka放置但未开炉、以及倾倒后抓取失败。这排除了“所有失败都是同一种motor失效”的简单解释，没有单独定责某一个Writer模块。

A3支持把首层残差内容作为下一项受控学习变量：保留原首层检索，把其任务条件内容改为从视频memory读取；其它前端、decoder与4×64训练口径保持。此项测试“直接语言内容是否影响共享迁移”，尚未证明删除必然改善。原24-task组织和rank共享后置，未执行草稿已撤回。

## 2. A：实际接口导数

固定24个train tasks，每task正确teacher46、同task其它episode42–45中固定16个action queries，跨200/600共享真实视频/query/time/noise，共48条件/768queries。实际完整四组视频编码后，以原functional FM对真实LoRA的cotangent反传到compiler中间分量；仅输入激活求导，模型不更新。

- FM均值 `.115832 → .107259`，20/24 tasks下降，四suite均值均下降。这是新16-query面板，不与旧128-query诊断拼接。
- P4 cotangent全部非零；两层cross/self的维数校正局部敏感性同数量级，不能认为视频路径未进入动作拟合。
- 初始query的语言项范数占比约`.9998 → .9993`只表示几何大小；P4全局缩放导数接近零受LayerNorm尺度不变性影响，不能解释成视频被忽略。
- 314.89秒，峰值23.411GiB；原件 `runs/analysis/horizon_relation_writer_20260908/k1_fresh/internal_diagnostic_20260909/{summary,conclusions}.json`。

## 3. A2：单支路activation lesion

相同48条件，每条件normal、query_language输出零、首cross输出零、次cross输出零四臂，3072queries。每臂保留真实P4、前端exact language和至少一个视频cross，不改其它输入或模型参数。下表为任务等权FM；括号是相对normal变化及FM变差task数。

| checkpoint | normal | query_language零 | cross1零 | cross2零 |
|---|---:|---:|---:|---:|
| 200 | .115832 | .144306（+.028474；23/24） | .117044（+.001211；14/24） | .117232（+.001400；14/24） |
| 600 | .107259 | .136691（+.029431；24/24） | .109963（+.002704；18/24） | .110241（+.002982；18/24） |

移除query_language使实际50×7速度预测的RMS变化约`.1651/.1715`，单cross移除约`.0397–.0583`。这些是同一FM时刻的预测变化；`x_t−t*v`的clean估计也不是10-step部署轨迹。不能用LoRA因子变化比例或FM变化量分配完整闭环“贡献百分比”。

48个normal与A的FM相同，实际time/noise/query/teacher与hook检查通过，没有为逐元素重合重复forward。445.71秒，峰值10.680GiB；原件 `internal_diagnostic_20260909/branch_lesion/{registration,summary,conclusions}.json`。

## 4. A3：完成首次路由/内容分离

A2的query_language零同时改变首次视频检索与残差。A3固定`q=e_target+e_rank+l`与实际`c(q)=Cross(LN(q),LN(P)+time,LN(P))`，比较原`q+c(q)`、只删残差语言的`q-l+c(q)`、只删检索语言的`q+c(q-l)`。后续self/FFN和第二block沿真实中间状态继续；48条件×3臂×16queries，无梯度和模型修改。登记见[active design §8.2.4](horizon_relation_video_writer_design.md)。

| checkpoint | normal FM | 只移除language残差 | 只移除首层language路由 |
|---|---:|---:|---:|
| 200 | .115832 | .142614（+.026782；23/24变差） | .116020（+.000188；13变差/11改善） |
| 600 | .107259 | .136067（+.028808；24/24变差） | .107152（−.000107；12变差/12改善） |

当前训练侧动作拟合明显依赖首层language残差内容；仅该层检索query的language边际影响小且方向混合。600速度预测RMS扰动分别`.169834/.016665`。content臂保留首cross实际输入和原输出，route臂保留原残差q；48 normal与A一致，全部实际输入配对通过，原FP32代数重排最大误差5.96e-8。353.70秒，峰值10.684GiB，原件 `internal_diagnostic_20260909/route_content_lesion/{summary,conclusions}.json`。

**适用边界：** route臂仍允许language经原残差进入后续self/FFN与第二cross query；content臂会自然改变后续query。因此不能升格为“所有语言检索无用”，也不能把当前训练任务对语言残差的依赖直接证明为held错误对象的根因。大幅移除出现OOD中间状态；fresh训练能否让视频内容承担该职责，只有后续受控学习与strict400能回答。

## 5. B：真实闭环回放与历史差异

在看新轨迹前固定validation8全部任务与states`0/12/25/37`，每checkpoint32条。复用原correct400的相同LoRA、teacher、state、source/normalization、environment与policy RNG，记录双相机、action chunks和逐step BDDL目标谓词。谓词不是完整人工阶段标签；捕获的state在policy实际token化输入内，不额外声称保留了原始浮点proprio。

200已完成：9/32，历史对应子集7/32，保留6、新增3、丢失1；两worker exit0，完整墙钟464.89秒。实际LoRA/teacher/state/env与policy noise公共前缀全部核对通过。正常数值和执行分组差异下闭环轨迹可以分叉，本次图像只解释本次行为，不能冒充历史轨迹。原200正式成绩仍110/400，小面板不重选checkpoint。

以下是已目视检查的实际200图像事实，原件 `behavior_replay_20260909/contact_step200/`，逐case记录 `visual_observations.json`：

| task/检查范围 | 最早可观察缺口或行为 | 限制 |
|---|---|---|
| Spatial1，四state | 4/4均操作ramekin并移向盘，目标黑碗留在原处 | 当前输入条件下的错误目标获取；不是视觉模块单独责任证明 |
| Spatial3，四state | 4/4均操作cabinet上的碗，cookie box目标碗留在原处 | 错误实例选择；不由此推断所有Spatial初始化 |
| Object11/13，state25 | 正确目标抓取搬运并入篮，分别237/138步成功 | 只说明所见案例实际具备这些动作 |
| Goal23，state25 | 已打开上抽屉，之后操作碗，最终未入抽屉 | 失败发生在已有开抽屉行为之后，不是全程目标不明 |
| Goal26，state25 | cream cheese与碗附近发生多次操作，后期碗被移动，最终in谓词为假 | 稀疏帧不足以唯一还原接触顺序，不猜测精确碰撞原因 |
| Long31，state25 | 在cream cheese附近反复交互，后期向篮移动，但两目标从未满足in | 不把零终局谓词等同于没有接近或抓取 |
| Long32，四state | state0/12曾满足moka-on-stove但未开炉；25操作平底锅；37将moka碰倒后反复抓取 | 同task含多种缺口，不能统一称为后期组合失败或初始目标错误 |

600同样完整exit0：7/32、583.61秒；历史对应也是7/32，但保留6、新增1、丢1。两点64个trajectory大小与行记录一致、最终谓词合取与success一致；teacher/LoRA/实际noise配对通过。轨迹约5.56GiB，在原8GiB总预算内。

| 回放checkpoint | S/O/G/L（各8条） | per-task（1/3/11/13/23/26/31/32，各4条） | breadth |
|---|---|---|---:|
| 200 | 0/5/4/0 | 0/0/3/2/1/3/0/0 | 4/8 |
| 600 | 0/2/4/1 | 0/0/2/0/0/4/1/0 | 3/8 |

200→600保留5、新增2、丢4，churn6/32、J=.4545。小面板不估计真实success rate、不替代原110→82/400；它的价值在于给相应行为提供可观察解释。

600新增图像检查覆盖同state25全部8task，以及Spatial1/3、Object13、Long32各自其余三state；另外比较Long31的固定state37，两点共42份9时刻双相机图：

- **Object13目标获取变化最明确。** 200在state0/25正确BBQ成功，37正确BBQ晚到篮边但超时，12操作了绿色瓶干扰物。600四条都先操作干扰物：0把棕色长方包装物放入篮；25/37把绿色瓶放入篮；12反复操作绿色瓶。不能把该任务退化只归为“抓到了但放不准”。
- **Spatial并非所有细节不变。** 600 task3四条仍首先拿cabinet上的错误碗，37后期转向cookie-box目标但未成功。task1的12/37仍搬运错误ramekin，0/25长时间在碗与ramekin附近降低夹爪而无可靠搬运，稀疏图不判定精确接触意图。
- **Long32含独立子目标与干扰物问题。** 600的0/37曾放好moka但没开炉，后续还继续操作；12开炉后将moka碰倒并改操作平底锅；25先操作平底锅。不能由某个子目标曾成立推断完整组合已学会。
- **Long31保留正能力边界。** 固定37在200于381步才放入cream cheese，随后搬butter到篮边却未在520前满足目标；600按相同任务输入依次放入二者，296步成功。25则先放入错误milk，再放对butter，之后cream cheese未完成。相同图既有正确组合，也有错误对象与慢/失败抓取，不能一概判无多步能力。

64条完整原件保留在 `behavior_replay_20260909/`；逐checkpoint `analysis_step*.json`包含所有case与历史差异，`comparison.json`保存完整配对、per-task/suite和相邻统计，`visual_observations.json`保存实际目视范围及限制。
