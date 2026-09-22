# 36任务 Writer 辅助 episode 配对 fresh 对照：轻量正式复核报告

本页是 sealed 正式产物的阅读入口；逐行数据、完整性和资产索引位于同目录的 [README](README.md)。没有追加训练、物化、评测或新诊断。

## 合同与结论边界

唯一科学变量是辅助 teaching query 的 `data.teaching_episode: same_video → cross_episode`。Source-71、normalization、36任务、K=1、完整 38-target rank16 A/B、fresh Writer/三 Meta、每更新四任务、每任务主21＋辅助7、固定 `tau=1` 前五步和权重 `1/3`、seed、采样、优化器和 LR 时钟均保持不变。

所以本结果只支持：在该合同下，改变**辅助 episode 配对**会改变学习结果。它不证明辅助端点/前缀目标整体有效，也不分别证明 `tau=1`、前五步或 `1/3` 必要；未执行 shuffle/reverse，不能判断顺序特异性。

## 六个 correct400 节点

| step | same-video | cross-episode auxiliary | 差值 | 相对 Writer 历史最佳117 | 相对 MT-BC300=155 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 110 | 151 | +41 | +34 | -4 |
| 400 | 92 | 120 | +28 | +3 | -35 |
| 600 | 115 | **154** | **+39** | **+37** | **-1** |
| 800 | 87 | 137 | +50 | +20 | -18 |
| 1000 | 117 | 129 | +12 | +12 | -26 |
| 1200 | 80 | 124 | +44 | +7 | -31 |

1200 的完整 correct400 节点触发 `sustained_decline`；原 early-stop 和 extended-run 的选点均为 C600=154，post-stop extension 为空。相邻节点的 retained/gained/lost 为：200→400 `84/36/67`，400→600 `80/74/40`，600→800 `101/36/53`，800→1000 `107/22/30`，1000→1200 `98/26/31`。因此不能把曲线的同节点整体提升写成稳定保持。

## C600 的关键配对

以下 `C600-only` / `reference-only` 均以逐行相同 suite、task、init/state 为单位；所有行的 language、environment seed、policy seed root 一致。因不同终止时点造成的 policy-noise 序列长度差不视为配对失配：共同前缀在全部 400 行均相同。

| 参照臂 | C600 | 参照 | 共同成功 | C600-only | reference-only | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical same-video@600 | 154 | 115 | 78 | 76 | 37 | .4084 |
| frozen MT-BC300 | 154 | 155 | 111 | 43 | 44 | .5606 |
| Source validation | 154 | 51 | 35 | 119 | 16 | .2059 |
| same-task-other control | 154 | 160 | 134 | 20 | 26 | .7444 |
| cross-suite-wrong control | 154 | 153 | 111 | 43 | 42 | .5663 |

C600 相比 MT-BC300 的总分仅低1，但并非相同成功集合；这正是 `c600_vs_mtbc300_rows.csv` 和 `c600_pairing_summary.json` 供后续判断的核心。correct 相比 wrong 仅 +1，且 other 比 correct 高6，故当前 controls 不支持“正确视频内容恢复”的主张。

## C600 的 suite/task 组成

| suite | same-video@600 | C600 | MT-BC300 | C600−same | C600−MT-BC |
| --- | ---: | ---: | ---: | ---: | ---: |
| libero_10 | 13 | 22 | 20 | +9 | +2 |
| libero_goal | 42 | 34 | 36 | -8 | -2 |
| libero_object | 42 | 35 | 47 | -7 | -12 |
| libero_spatial | 18 | 63 | 52 | +45 | +11 |

| suite/task | same-video@600 | C600 | MT-BC300 |
| --- | ---: | ---: | ---: |
| libero_10 / 1 | 13 | 22 | 20 |
| libero_10 / 9 | 0 | 0 | 0 |
| libero_goal / 3 | 0 | 0 | 0 |
| libero_goal / 6 | 42 | 34 | 36 |
| libero_object / 1 | 41 | 33 | 35 |
| libero_object / 6 | 1 | 2 | 12 |
| libero_spatial / 3 | 14 | 41 | 43 |
| libero_spatial / 6 | 4 | 22 | 9 |

Spatial 的大幅净增与 Goal/Object 的净损失同时存在；不能用 154 的总分遮蔽能力交换。

## 完整性、训练与成本

- 已核验 11 个面板：6 个 cross-episode correct、C600 other/wrong，以及 same-video600、MT-BC300、Source。每个都有 400 条唯一 task/state 行、结果/contract/completion 一致；8 个本 study 面板还核验了 400 条 materialized conditions 与全部 worker 零退出。详见 `panel_integrity.json`。
- 训练共 1200 updates；主/辅助/总 queries 为 `100800/33600/134400`。update loop 为 11,720.8 秒，均值 9.767 秒/update，P95 15.104 秒，四卡峰值 reserved 22.803 GiB/GPU。所有每步 loss、LR、既有梯度和时延在 `training_metrics.csv`；36任务 exposure 汇总在 `task_exposure_summary.csv`。
- correct400 六面板评测墙钟 5,228.9 秒；含选点 controls 共 6,994.0 秒。物化 3,200 条条件，但 formal manifest 未记录 materialization elapsed，因此明确为 `not_recorded`。
- C600 checkpoint 的 `ecp.safetensors`、`trainer_state.pt` 和四份 rank state 均存在且大小与 checkpoint manifest 一致。它是“完整 resume state 仍在”的资产事实，不是恢复或启动新分支的授权。

实际运行时 effective config、停止/选点记录、C600 resume 文件清单和所有原始本地证据路径在 `effective_training_config.json` 与 `cost_and_assets.json`。若需要任务簇统计推断，应从本包逐行表计算；本包不自行增加新的置信区间或诊断。
