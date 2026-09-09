# 历史 v6 的 matched-current-protocol 探索参照

## 授权、识别对象与边界

本实现仅服务 2026-09-09 学习过程因果 goal。Owner 最新授权允许隔离中的架构修改、实验训练与闭环验证；本臂忠实恢复历史模型作为架构包参照，不采纳为当前方法，不修改 main 的 Horizon/fullH 合同，不启动下一轮正式训练。

四个模型文件来自 `bad9a961ff864d593cf0eb5b05d93930cd2280c1:src/ember/writer/`，迁入 `src/ember/v6_reference/`，仅调整相互 import namespace。`4efa7371ba1659a2b70c619cb148e43fa0768023` 的这四个文件与该版本一致。`3a6f801d` 的设计文档用于历史解释，其较晚扩展的 model 接口不进入此实现。

被测架构保留：Text/VL/Action 三组 rank4 Meta；逐帧原生 50×1024 suffix hidden 在任何 learned horizon read 前直接 mean；task-token Semantic Core、相邻 grounded patch transition、两层 causal Procedure；320 slots 与八个跨 layer/rank 共享的 family/side heads。完整 rank16 LoRA 为 38 targets / 76 tensors。原固定 probe seed=`7+0x5A17=23070`，与当前 Horizon 的1729不同。这是显式保留的架构包差异，不能把本臂称作当前 fullH 前端，也不能单独归因某个模块。

共同不变的硬边界：同一 frozen source、固定 train24/validation8/test8、stride5、exact language + action-hidden 有序视频、一次完整 LoRA、teacher/query 严格跨 episode、held 无梯度、官方执行。Test、wrong/shuffled/reversed/no-video 与最终视频 controls 不开放。旧121/143 checkpoint仅为历史证据，不加载为初始化，不根据旧 controls 选择本次架构。

## 共同学习与证据协议

- 复用当前 `WriterTrainingData`：teacher0–15、query16–41、独立动作诊断42–45、held teacher46–49；seed7，各 suite 随机一个 task，每条件64 queries，总256，权重各1/4。
- Fresh Writer，包括三组 Meta、optimizer/scheduler、sampler/RNG。AdamW lr3e-5、warmup8后恒定、betas(.9,.95)、eps1e-8、weight_decay1e-4、clip1。
- 正常重采样独立 Beta flow time / Gaussian noise；调用当前 keyed logical-batch RNG 与 microbatched FM，task/episode/frame/query events 与当前共同基线可配对。旧 no-replacement 50-episode sampler、20-query 配方与旧LR不恢复。
- 200/400更新分别为800/1600条件和51,200/102,400 queries。可分两段运行，400须从本臂完整200 exact-resume；不能加载其它架构或拓扑的学习状态。
- 200/400固定 train24×128 的独立动作诊断，每节点3072 queries；共同的 identity LoRA 起点不重复诊断，不推进训练 sampler。200/400各做当前 canonical validation8 correct400和held-video train96（states32–35，videos46–49），保留逐task/suite、breadth、R/G/L、churn和Jaccard。
- 全部新schema包含`exploratory_v6`，每个bank/checkpoint/episode保留真实方法身份。官方 evaluator 的`--mode formal --state-count 50`只指定既有 strict400 执行模式；run contract额外写入`evidence_purpose=exploratory_v6_matched_causal_comparison`、`scientific_qualification=false`，不宣称正式资格。

## 代码职责与生命周期

| Owner | 保留职责 |
|---|---|
| `v6_reference/{architecture,model,temporal,video_program}.py` | 历史2018行模型，import之外原样保留，约10.775M trainable parameters |
| `v6_reference/contract.py` | 精确历史/当前匹配配置、探索身份、200/400与正确视频面板 |
| `v6_reference/runtime.py` | source与原模型装配，完整v6 graph调用和cotangent replay，无新采样器 |
| `v6_reference/artifacts.py` | 探索checkpoint inspector、独立冻结动作诊断CLI |
| 当前 `writer/training.py` | 唯一更新循环、任务调度、逻辑SUM、clip/optimizer/scheduler、ECP保存恢复；显式schema分派 |
| 当前 `writer/materialization.py`、`writer/evaluation.py` 与官方eval接口 | 唯一条件规划、bank读写、官方rollout worker和动态队列；按独立schema派发，不复制orchestrator |

三组 Meta 已属于原 Writer，不注册第二套 Meta 参数；state dict只保存一次。运行方式为完整 Writer no-grad→当前分块 FM cotangent→完整 Writer replay→联合反向。不得调用 Horizon `FrozenVideoPrefixCache`；Text/VL Meta变化使任何经Meta的text/prefix/KV缓存跨步失效。纯RGB/固定token允许复用，当前实现每条件直接加载。

原日志峰值 allocated约71.7GiB，不能照搬原frame chunk32到A40。现配置frame chunk8、FM physical microbatch8仅是待实测起点；64-query logical batch和模型图保持。没有本机GPU吞吐/显存或实际source forward证据，启动前须完成最长视频与完整forward/replay、identity/finite/三Meta梯度和短exact-resume验证。

结构例外：原模型文件含613/690行及若干长函数，保留完整历史单元比拆分更能核对忠实性。新增配置、运行、证据三个owner负责实际第二种模型接入，共享trainer和evaluator只增加有界schema/runtime分派，不增加第二条执行系统。guard已消除新增复杂度>25的函数；其余review主要来自这些历史单元与原有共享模块。整个分支独占 `codex/v6-causal-reference`，不得合入main。goal后模型、脚本和实验测试只由Git分支与原件保留，删除task-owned执行worktree；若Owner另行决定方法采纳，须重新决定唯一canonical实现，而非保留fallback。

## 可执行接口

以下在clean、pushed、detached的本探索worktree内运行。`EMBER_ASSET_ROOT`指向现有canonical仓库，`EMBER_PYTHON`和`EMBER_TORCHRUN`指向现有环境，`EMBER_V6_RUN`为预登记的新探索输出路径。GPU列表、world和microbatch只在双节点live检查与独立quota/峰值预算后填写；此文不预留固定设备。源码不复制任何大资产。

```bash
export PYTHONPATH="$PWD/src"
export NCCL_P2P_DISABLE=1

# 小型GPU profile由主agent安排；profile checkpoint不能进入探索证据bank。
CUDA_VISIBLE_DEVICES="$EMBER_GPU_LIST" "$EMBER_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$EMBER_WORLD" \
  scripts/train_v6_reference.py --asset-root "$EMBER_ASSET_ROOT" --output "$EMBER_PROFILE_ROOT" \
  --mode profile --stop-after-step 2 --policy-microbatches "$EMBER_MICROBATCHES" --cpu-threads 4

# Fresh到200；分段登记保持config中的共同200/400节点。
CUDA_VISIBLE_DEVICES="$EMBER_GPU_LIST" "$EMBER_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$EMBER_WORLD" \
  scripts/train_v6_reference.py --asset-root "$EMBER_ASSET_ROOT" --output "$EMBER_V6_RUN" \
  --mode exploratory --checkpoint-updates 200 --stop-after-step 200 \
  --policy-microbatches "$EMBER_MICROBATCHES" --cpu-threads 4

# 同拓扑完整200恢复至400；不继承旧历史checkpoint。
CUDA_VISIBLE_DEVICES="$EMBER_GPU_LIST" "$EMBER_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$EMBER_WORLD" \
  scripts/train_v6_reference.py --asset-root "$EMBER_ASSET_ROOT" --output "$EMBER_V6_RUN" \
  --mode exploratory --resume "$EMBER_V6_RUN/checkpoints/macro_00000200" \
  --checkpoint-updates 400 --stop-after-step 400 --policy-microbatches "$EMBER_MICROBATCHES" --cpu-threads 4

"$EMBER_PYTHON" -m ember.v6_reference.artifacts inspect --checkpoint "$EMBER_V6_CHECKPOINT"

# 独立冻结动作诊断：单个已调度GPU，完整24×128，同训练侧固定面板。
CUDA_VISIBLE_DEVICES="$EMBER_DIAGNOSTIC_GPU" "$EMBER_PYTHON" -m ember.v6_reference.artifacts diagnose \
  --checkpoint "$EMBER_V6_CHECKPOINT" --asset-root "$EMBER_ASSET_ROOT" \
  --output "$EMBER_DIAGNOSTIC_JSON" --device cuda:0 --policy-microbatch 8

# 所有50 teacher视频各一次；不支持本轮未登记的其它arms。
CUDA_VISIBLE_DEVICES="$EMBER_MATERIALIZATION_GPU" "$EMBER_PYTHON" scripts/materialize_v6_reference.py \
  --checkpoint "$EMBER_V6_CHECKPOINT" --asset-root "$EMBER_ASSET_ROOT" --output "$EMBER_V6_BANK" \
  --role validation --task-ids 1,3,11,13,23,26,31,32 --k 1 --arm correct --state-count 50

"$EMBER_PYTHON" scripts/evaluate_pi05.py run --source-run "$EMBER_SOURCE_RUN" \
  --checkpoint "$EMBER_SOURCE_CHECKPOINT" --tokenizer-path "$EMBER_TOKENIZER" \
  --output-dir "$EMBER_V6_EVAL" --role validation --mode formal --state-count 50 \
  --replicas-per-gpu "$EMBER_EVAL_REPLICAS" --gpu-indices "$EMBER_EVAL_GPUS" \
  --static-task-lora-manifest "$EMBER_V6_BANK/manifest.json"
```

train96物化使用同一入口，`--role development_train --task-ids 0,2,4,5,7,9,12,14,15,16,18,19,20,21,22,25,28,29,34,35,36,37,38,39 --k 1 --arm correct --init-state-ids 32,33,34,35 --video-pool 46,47,48,49`；官方评测对应`--role development_train --mode screen --state-count 4 --init-state-ids 32,33,34,35`。不同节点bank必须独立物化。批量请求可用共享materializer的`--requests-json`，runtime只在同source/model/observer/schema时复用，完整checkpoint逐次重新加载。

此分支的Git authority限定为`origin/codex/v6-causal-reference`；原source provenance仍接受`origin/main`。不接受dirty或未push的代码，不恢复到main authority冒充正式方法。
