# ECP order-specific composite privileged teacher：bootstrap数据合同

## 1. 目的与边界

Gate A3的phase-composed primitive teachers只有`28/50`与`9/50`，未通过teacher门；但37条成功episode均是在统一
composite language、完整两事件required order和未见初始化下真实闭环完成的轨迹。本阶段只把这些成功轨迹转换为两个
order-specific composite policy的privileged SFT数据集，补齐后续训练所需的逐步policy state。

这不是Gate A3翻案，也不是Writer训练或部署输入。它只授权后续训练两套privileged composite experts；任何LoRA仍须在原
50×2状态面板上fresh闭环复评，不能把训练轨迹成功率当teacher通过。

## 2. 固定输入

- process manifest：`configs/pi05_ecp_process_meta_separate_plates_v1/manifest.json`；
- formal authority：
  `runs/outputs/pi05_ecp_process_separate_plates_gate_a3_4bf5039_gpu01p123457_20260824/`；
- variants与成功state IDs完全由formal privileged ledgers解析，不人工挑选：
  - `red_left_then_yellow_white_right`：28条；
  - `yellow_white_right_then_red_left`：9条；
- 每条轨迹复用formal保存的teacher actions与public render256双相机RGB；不重新运行PI0.5、不重新生成action、不改变
  language、required order、init state、dummy settling或环境seed。

## 3. 构建方法

对每条成功轨迹在同一custom BDDL和原init state中确定性重放teacher actions。每个action执行前读取原生PI0.5使用的8维
state：`eef_pos[3] + quat2axisangle(eef_quat)[3] + gripper_qpos[2]`。replay render仅用于满足模拟器观察合同，固定为32；
训练RGB直接使用formal public video，不使用replay画面。

每个variant输出一个标准LIBERO风格HDF5：

- `data/demo_i/actions`：`[T,7]` teacher actions；
- `obs/agentview_rgb`、`obs/eye_in_hand_rgb`：public video的前`T`帧，`[T,256,256,3] uint8`；
- `obs/ee_states`：`[T,6] float32`；
- `obs/gripper_states`：`[T,2] float32`；
- demo索引按source state ID升序重新连续编号，原state ID、init state、completion steps和source ledger/video只作
  privileged provenance attrs，不进入policy input。

canonical output root：
`data/datasets/ecp_composite_teacher_bootstrap_a3_v1/`。两个进程各写一个variant，不能并发写同一HDF5。

## 4. 信息墙

- 数据来自经授权的non-held LIBERO-90 task65/68 process acquisition；target40 action reads为0；
- composite expert训练时只消费exact unified language、RGB、8维当前state与teacher actions；variant name、state ID、
  predicate、completion step、reward和success不进入policy forward；
- 这些privileged experts不部署，也不成为Writer的task-ID route或第二adapter；它们只用于生成更可靠的action-hidden
  process videos与建立teacher ceiling；
- deployment Writer输入仍严格只有exact language与action-hidden ordered videos。

## 5. 构建门与正式命令

代码入口为`python -m ember.ecp.composite_teacher_data`。正式构建从clean pushed commit的detached frozen worktree执行；
两variant可分别使用同一节点的两个空闲GPU作EGL。固定命令模板：

```bash
PYTHONPATH=src .venv/bin/python -m ember.ecp.composite_teacher_data \
  --collection-root runs/outputs/pi05_ecp_process_separate_plates_gate_a3_4bf5039_gpu01p123457_20260824 \
  --variant <variant> \
  --physical-gpu-id <gpu> \
  --output-dir data/datasets/ecp_composite_teacher_bootstrap_a3_v1
```

构建通过要求：

1. 28/28与9/9 replay全部仍success、invalid为0，completion steps逐条精确匹配formal ledger；
2. 两份HDF5分别有连续28与9个demo，action/RGB/state长度逐episode一致；
3. `FunctionalQueryDataset`可直接读取两个authority并产生`[3,256,256]`双相机、8维state与`[50,7]`action chunk；
4. 输出manifest记录0 divergence、source state IDs、episode lengths与bytes；不新增hash校验。

任一replay divergence都停止构建并定位模拟器/authority差异，不静默丢episode。

## 6. 后继训练与裁决（本阶段不执行）

构建门通过后，两个variant分别从同一冻结source PI0.5和fresh zero rank16 LoRA训练一个order-specific composite expert。
首轮固定step1000、batch16、相同优化器和统一composite language；不按variant、loss或训练状态选择不同checkpoint。

训练后在原50×2 init-state面板上运行一次strict400：

- 分别报告全部50、bootstrap-success states与A3原失败states；
- 既要求保留大部分bootstrap support，也必须在原失败states获得足够新成功；
- 只有两个方向均达到预注册teacher门，才进入Gate B；否则定位为composite data coverage或policy learning问题，不返回
  primitive phase composition、step sweep或新pair盲跑。
