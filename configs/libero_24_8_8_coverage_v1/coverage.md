# 训练操作覆盖审计

2026-09-20，基于canonical overlap_audit中的完整fixtures/objects/init/goal、source71与target40 manifest，
现场只读核对36个训练HDF5的存在、每task50 demos、目标语言及辅助BDDL身份。未读held动作/state/reward或模型结果。
这证明任务规格和示范身份的覆盖，不声称逐演示重放认证所有操作时序。

S/O/G/L为Spatial/Object/Goal/Long，A为libero_90，均suite内0-based。

| Held | 训练支持 | 留出变化或限制 |
|---|---|---|
| Val S3 | S5高处取碗；S0/1/2/4/7碗到盘 | 饼干盒来源几何 |
| Val S6 | S0/1/2定位与碗搬运 | 饼干盒旁选择 |
| Test S8 | S0/1/2、G8碗到盘 | 盘旁选择 |
| Test S9 | S5/7高处取碗、G8碗到盘 | 柜顶取出未直接训练；A15放柜顶不算取出 |
| Val O1 | L7 cream cheese入篮、A57入托盘 | 完整任务留出，primitive已见 |
| Val O6 | A3 butter入抽屉、A56入托盘；Train Object入篮 | butter—篮新配对 |
| Test O0 | L7 soup入篮、A55入托盘 | 完整任务留出，primitive已见 |
| Test O8 | L6/A61搬pudding；Train Object入篮 | pudding—篮新配对 |
| Val G3 | A11开顶抽屉、A2碗入已开抽屉 | 组合留出；A2不提供开操作 |
| Val G6 | L7/A57搬cream cheese、A16叠碗 | cream—碗配对；目标是On而非In |
| Test G4 | A15中间碗到柜顶、G2酒瓶到柜顶 | 场景与选择变化；A15已审计非完整等价 |
| Test G7 | L2真正执行Turnon | L8初始已Turnon不计此支持 |
| Val L1 | L7双物体入篮、A3/A56搬butter | 新双物体组合 |
| Val L9 | L4搬白黄杯、A33关微波炉、A3放入后关闭 | 无训练物体入微波炉；明确留出新装置几何 |
| Test L0 | L7双物体入篮含soup、O5 tomato入篮 | 新双物体组合 |
| Test L3 | A24碗入底抽屉、A22关闭、A3放入后关闭 | 完整组合；初始底抽屉已开 |

12辅助均属于原source71 allowlist，不恢复原审计排除的19个完整等价任务。完整held任务未被辅助别名绕过。
训练支持是对象与基本操作层的支持；不保证每个对象—受体组合、容器几何已见，更不保证模型已学会。
数据位置复用configs/pi05_target_data_v1/manifest.json与configs/pi05_source_corpus_v1/source_manifest.json的同一revision。
