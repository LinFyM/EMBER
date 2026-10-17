# §12匹配原生控制学习整批交付

ROOT: `runs/video_guided_proposal_writer_20261013/full_g_native_function_identification`。27既定补录、两臂72有效更新、四个576/400末点读出全部保存并验收。原owner06在GPU全exit0后因旧MT400缺global字段而CPU汇总exit1；原件不改写。clean pushed detached db1bff33仅改为官方suite/local task/state唯一匹配并缓存原件，实际全四消费者配对通过，0新forward/GPU/episode/重训。

| 面板 | MT | 原G38349 | A F-only72 | B full-G72 | B−A |
|---|---:|---:|---:|---:|---:|
| seen36×16 | 340 | 427 | 404 | 339 | −65 |
| strict firstG Val400 | 153 | 165 | 159 | 150 | −9 |

Val对MT：A R131/G28/L22/net+6，B R123/G27/L30/net−3。对原首G165：A R126/G33/L39/net−6/churn72，B R122/G28/L43/net−15/churn71；B对A R123/G27/L36/net−9/churn63。seen对原G427：A R367/G37/L60/net−23，B R314/G25/L113/net−88；B对A R313/G26/L91/net−65。

| validation task | MT | A | B |
|---|---:|---:|---:|
| 3 | 41 | 39 | 41 |
| 6 | 7 | 2 | 6 |
| 11 | 36 | 37 | 35 |
| 16 | 10 | 16 | 10 |
| 23 | 0 | 0 | 0 |
| 26 | 35 | 37 | 36 |
| 31 | 24 | 28 | 22 |
| 39 | 0 | 0 | 0 |

Val coverage两臂6/8、task23/39仍0；seen coverage A36/36、B34/36。Spatial/Object/Goal/Long MT48/46/35/24、A41/53/37/28、B47/45/36/22。A的Long增益/Spatial损失及B的Spatial相对A改善/其它损失均保留。两臂均未提升强原G；负结果只约束本次固定288/72窗口，不推全链永不可学，不借BCE遮盖G缺口，主讨论按原§12.6裁决。没有新完整Compiler/H修订/视频必要性/稳定/Test资格，不自动第二块、phi/beta或选点。

实际CPU载入两臂完整checkpoint/Adam72/source38349/fresh同seed、sampler/一rank RNG/topology；36task各8次，288同条件/T/query/完整noise/类别权重逐行相同。A仅endpoint166有梯度；B meta双支/Reader/static/endpoint均有原生10步前5x7信用，496 owned/485 Adam state如实保存。可训练前端依赖随更新重算，source/Actor/T冻结。工程梯度通过不代替控制改善。训练源A8dd68689/B1c424cd8、GPU读出41bd9587，摘要修复db1bff33不改变这些计算身份。

27既定最早成功query单次补录21成6败、5582步，不换失败病例；已有支撑51/57T、64/72PH、256/288conditions沿main_support_coverage.json，32缺失条件仍有正常跨episode查询。T成功与自身保持可复用同一经历，类别等权但不计作独立证据两份。72 PH角色E38/Gparent20/MTparent14不等于全部专家纠错；1/2episode训练不声称完整五轮覆盖。

872套实际生成的38-target/r8完整LoRA（每臂36seen+400held），每condition一套固定执行；held各task50不同教学、state-video/scene/env-policy/完整noise前缀配对，own-MT H1与最终state分离，0新held实践/梯度/Test/Actor。所有新行5x7 finite动作及A/B初態比较通过；旧MT400缺initial_proprio限制保留，引用MT-H有历史交互成本，不称zero interaction。

全量事件ledger1979提交/1979返回=27补录+1952正式query、497912环境步含dummy，未返回0/额外未知上界0，符合2024完整/2040physical/1081200步上界。profile两臂各2完整discarded B4共4已消费；原profile/authority/GPU probe/CPU摘要失败0额外episode，但原件和费用全保留。GPU总10.459490537h，12收据含3失败.073219029h，CPU恢复不重计GPU。

实际训练消费A59.30/B81.23min、reserved15.563/18.467GiB，micro24与原frame16/8。四原读出一worker/world1/8slot并行，Aseen frame32，其余16；seen约79.85–84.92min、held148.01–151.0min。原30–60min估计显著低估；原首GPU11:42:53Z至17:00:25Z=5.292h，不含之前CPU准备且不重置14:29恢复时钟。owner未将已释放seen消费者续接held队列，不能追认最大吞吐；全部完成后不为调度改进重跑。各面板group throughput/峰值和全收据在acceptance.json。

canonical原件ROOT/results.json；各arms/<arm>/readouts/{seen,validation}/results.json及query/generation/queue；唯一arms/<arm>/checkpoints/update_00000072。完整验收/配对/账目在analysis/completion_20261014/acceptance.json，原失败coordinator/matched_native_readout_resume_06/exit.json与error.txt保留。§23已科学消费，不新增计算/等待者；一次交主讨论01a13399自主科学接续。

Git交付：main `3fb2fad874676bca1b2fd3e56b6278d10552b0a9`已push且clean；摘要代码`db1bff339245d3b6a7d111cb6516617c0b681180`在main保留，临时开发worktree/本地远端分支在clean ancestor核验后退休，原训练/读出冻结树与全部原件不删除。§23科学记录be8abab4亦随main继承，未单发ACK。
