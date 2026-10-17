# 历史代码身份与直接阅读入口

main已经多次退役旧运行面，不能用当前目录缺失推断旧机制未实施。下列链接固定到实际历史提交；训练来源、修复、物化和读取可能不同，表内顺序与角色应结合对应原报告解释。它们全部可从本次科学基点`b55226ed`及当时`origin/main`的历史到达。

这些是科研代码身份，不是恢复运行授权。原run contract的科学字段见[evidence/contracts](evidence/contracts)，精确文件来源见[evidence_index.json](evidence_index.json)。

| 方法/阶段 | 角色说明（与右栏顺序对应） | 固定代码目录 |
| --- | --- | --- |
| T2340 | 训练 e2afbfd7；物化 cb535c1e | [e2afbfd7](https://github.com/LinFyM/EMBER/tree/e2afbfd7c997e3f792921600608efa2fa3c1b25a/src/ember/operator_writer), [cb535c1e](https://github.com/LinFyM/EMBER/tree/cb535c1e477dc48ca90fc8d2a11225c0f8847173/src/ember/operator_writer) |
| Native Correction / oracle | Writer；真实纠正构造；oracle读取 | [09c1a0d6](https://github.com/LinFyM/EMBER/tree/09c1a0d6c15d48ea103b8bd5bac807d3251c088d/src/ember/writer), [f39d594f](https://github.com/LinFyM/EMBER/tree/f39d594fa6e1e014288dbcef2d91495400af38c3/src/ember/writer), [2c0a8dde](https://github.com/LinFyM/EMBER/tree/2c0a8dde5b3f123d67343c97ce2dc2f131f5be0a/src/ember/writer) |
| LocalField | 学习；物化/评估 | [50dafb05](https://github.com/LinFyM/EMBER/tree/50dafb05dfd2f7a395be7dfd8a7212b92e775beb/src/ember/writer), [46aaf22d](https://github.com/LinFyM/EMBER/tree/46aaf22d2ed5badf8db2e66475875a9608d70c2f/src/ember/writer) |
| ProcessPullback | 学习；读取；free-q/free-AB；learned outlet | [d6defa69](https://github.com/LinFyM/EMBER/tree/d6defa69fa851a5d6c5e1092139e5b008f0a09dd/src/ember/writer), [f5d9db78](https://github.com/LinFyM/EMBER/tree/f5d9db78fb7b36fd6d1b4afd8118dc32f1811c2e/src/ember/writer), [adc31a15](https://github.com/LinFyM/EMBER/tree/adc31a15bfc63f4494b92baab789991c2782261d/src/ember/writer), [85ecfd18](https://github.com/LinFyM/EMBER/tree/85ecfd18c13becbafab4d3d3a7e674955e20070f/src/ember/writer) |
| G/F | 关系监督、真实功能信用 | [85614d9c](https://github.com/LinFyM/EMBER/tree/85614d9c1a146d27dbb36fb833b700037fe4b63e/src/ember/relation_writer) |
| P/Q | 匹配预测/GT关系学习；读取 | [84ef79c1](https://github.com/LinFyM/EMBER/blob/84ef79c14d644acda28554d9d365b4ed3b603207/src/ember/relation_input_compilation.py), [e7950406](https://github.com/LinFyM/EMBER/blob/e79504062abc433bb9daedcfef52c7355640adfc/src/ember/relation_input_compilation.py) |
| Experience | 共享学习；formal恢复 | [76c136a3](https://github.com/LinFyM/EMBER/tree/76c136a308d3a1814483ad3d3ac7b6a820b5cb2a/src/ember/experience_compiler), [517e0c20](https://github.com/LinFyM/EMBER/tree/517e0c20024dd52bc85598621b35e135457cc49c/src/ember/experience_compiler) |
| Parameter | 共享学习/原formal；360补读 | [db53422e](https://github.com/LinFyM/EMBER/tree/db53422e3a2e56048b606dc62f5558045de71221/src/ember/experience_compiler), [f0c49811](https://github.com/LinFyM/EMBER/tree/f0c49811cda3fcc1a55190d7b46164daef101935/src/ember/experience_compiler) |
| 同incoming诊断 | 闭环；函数读取 | [c381f178](https://github.com/LinFyM/EMBER/tree/c381f17855aef3d364dfd8a4e0aa70937461d93b/src/ember/experience_compiler), [6204e027](https://github.com/LinFyM/EMBER/tree/6204e02774204e702e9899cdb3a6e692befad94f/src/ember/experience_compiler) |
| FR | 实施；实施修正；正式学习；formal恢复 | [0791d711](https://github.com/LinFyM/EMBER/tree/0791d711a13736140dd67f1a1157379f4b70375d/src/ember/experience_compiler), [6e4aac1d](https://github.com/LinFyM/EMBER/tree/6e4aac1db34ac82aa6e2c2473f2c710a178b44b3/src/ember/experience_compiler), [6b2b4ffd](https://github.com/LinFyM/EMBER/tree/6b2b4ffd3a8bdd37515bf1cc1d4301e9e090eb67/src/ember/experience_compiler), [8d7cab05](https://github.com/LinFyM/EMBER/tree/8d7cab05218ba45788a198fe207ff1633ef7f37c/src/ember/experience_compiler) |
| G24849/视频证据 | endpoint学习；视频对照读取 | [7179d581](https://github.com/LinFyM/EMBER/tree/7179d581299f3b3be069847e3084989b874431df/src/ember/proposal_writer), [c60bd494](https://github.com/LinFyM/EMBER/tree/c60bd4941394ffaef3946962800fb22ba21f10c1/src/ember/proposal_writer) |
| 完整Writer RL9 | 初始学习；恢复/正式读取 | [1bcb2601](https://github.com/LinFyM/EMBER/tree/1bcb2601a682505fcb8557db5d2d81fdd79a623c/src/ember/proposal_writer), [6f6a8aa6](https://github.com/LinFyM/EMBER/tree/6f6a8aa60f604f19c966d45ef632d63276d2cc02/src/ember/proposal_writer) |
| selector B18 | 初始合同；实际物理调度修正 | [060485ba](https://github.com/LinFyM/EMBER/tree/060485bab50126c167f0e60d39c7adf49e9f9791/src/ember/proposal_writer), [bf2c88b0](https://github.com/LinFyM/EMBER/tree/bf2c88b0d6412ea7660a4e5674fc02c667abb5f0/src/ember/proposal_writer) |
| G功能、D/C | 同目标operator/native学习；实际交互预算修复 | [92d0a234](https://github.com/LinFyM/EMBER/tree/92d0a234f484623a4cb3ae23981f499f3a5d20cd/src/ember/proposal_writer), [045ee602](https://github.com/LinFyM/EMBER/tree/045ee6028de0308213db5364c90f3dcef9385374/src/ember/proposal_writer) |
| D相邻144 | 保留失败前缀后的正式恢复/读取 | [51924861](https://github.com/LinFyM/EMBER/tree/51924861b1a8b8ac71915c5f9654d7ed54bc49b5/src/ember/proposal_writer) |
| 图像/H与独立价值 | H编译训练；H继续；价值结果发布修复 | [7f05e3e6](https://github.com/LinFyM/EMBER/tree/7f05e3e6982444a339f052929ee589f0a916706c/src/ember/proposal_writer), [778ff8ea](https://github.com/LinFyM/EMBER/tree/778ff8eae613f0a3c9a03cc63a0553b2604266a8/src/ember/proposal_writer), [c683ff03](https://github.com/LinFyM/EMBER/tree/c683ff03aef0a8424e4d5980f5896d06fe981535/src/ember/proposal_writer) |
| 接管可行性 | 原始合同与接管实验 | [8e718be2](https://github.com/LinFyM/EMBER/tree/8e718be23d60b5b9f4dbf99f0c62940e7b57d47b/src/ember/proposal_writer) |
| 更早phase消费者 | 实际learner-state aggregation，不与10/13接管实现混同 | [966353e3](https://github.com/LinFyM/EMBER/tree/966353e309def2ba7b3c71f291a471c3a87fd0c2/src/ember/functional_adaptation) |
| N4/N16 | 原信用路径；schema修正；官方MT读取；N16正式；N16恢复 | [8801e701](https://github.com/LinFyM/EMBER/tree/8801e701bd642ebd33eec534d8ff3e6901ef1b0d/src/ember/video_effect), [afe6acc5](https://github.com/LinFyM/EMBER/tree/afe6acc5901da7e69860192794247f2f4b6e6f08/src/ember/video_effect), [a986e7b2](https://github.com/LinFyM/EMBER/tree/a986e7b2371e6e1eb1a90456a217bf3ecf56c469/src/ember/video_effect), [7a146fc0](https://github.com/LinFyM/EMBER/tree/7a146fc090a5e2373f322c5764d84b34779995a7/src/ember/video_effect), [78d4f73e](https://github.com/LinFyM/EMBER/tree/78d4f73e4ddd162f9f074b456145ea0041a5a302/src/ember/video_effect) |
| source/r8 reference | 原窗口；后继训练；最终四项继续/读取 | [44c0ee3a](https://github.com/LinFyM/EMBER/tree/44c0ee3a5de19313aafc77f8236377a61fa81fb5/src/ember/reference), [0f8e0023](https://github.com/LinFyM/EMBER/tree/0f8e00230968920e2b2bf544008b22d076b04d00/src/ember/reference), [355396b0](https://github.com/LinFyM/EMBER/tree/355396b016a2132688288980388e81951af774e6/src/ember/reference) |
| source E/D144 | 训练；物化/读取 | [eb78edd7](https://github.com/LinFyM/EMBER/tree/eb78edd7e68a6fd518906011df0f994541896037/src/ember/control_space), [39d21bf0](https://github.com/LinFyM/EMBER/tree/39d21bf0f831f03e41a920c1061f776e49ae943d/src/ember/control_space) |
| source operator1170 / linear64 | operator训练/读取；linear fit/读取 | [528a57b1](https://github.com/LinFyM/EMBER/tree/528a57b19d57c6889f2cce13c582e12057fffc21/src/ember/control_space), [c99502cf](https://github.com/LinFyM/EMBER/tree/c99502cf0478dd047f7e845dc49166b48d25c520/src/ember/control_space) |
| video initial Reader | 初始特征/117前；117后训练及两点读取 | [93e27530](https://github.com/LinFyM/EMBER/tree/93e27530a2cbd4f75ebabd95e229a938b8189be8/src/ember/control_space), [5568f80c](https://github.com/LinFyM/EMBER/tree/5568f80ce23cd4ecccff352dba3c9622c309da8b/src/ember/control_space) |
| source actor / fixed Reader | Actor及恢复；固定Reader编译 | [81c859e6](https://github.com/LinFyM/EMBER/tree/81c859e689f281fc2beec5accc5b9a7b3af81a51/src/ember/control_space), [7a149c3e](https://github.com/LinFyM/EMBER/tree/7a149c3e979abfcadfe645f78721eb75e2bd618a/src/ember/control_space), [56c4dc99](https://github.com/LinFyM/EMBER/tree/56c4dc99b1be4f6d3517f49dc8f8dbd3a20d26aa/src/ember/control_space) |
| functional反馈 | 共享学习；formal最终读取 | [0b9b8080](https://github.com/LinFyM/EMBER/tree/0b9b8080421a05ec32fd979d591f76707c91b4df/src/ember/control_space), [26c284ee](https://github.com/LinFyM/EMBER/tree/26c284eeb7075bd5e7a0f2551f5abbdd735569df/src/ember/control_space) |
| task-exclusion | fit；首次/最终读取 | [537c6763](https://github.com/LinFyM/EMBER/tree/537c676387cb4bde68de017772c48d5c32009c59/src/ember/control_space), [56491f82](https://github.com/LinFyM/EMBER/tree/56491f827d979a6f8b92274b2d4e4622e257b979/src/ember/control_space), [08437d23](https://github.com/LinFyM/EMBER/tree/08437d2331c3a00dbc60e160a89ec8e70a0e9c72/src/ember/control_space) |
| source-W生成 | 静态输入；训练；物化/读取 | [c7ef1372](https://github.com/LinFyM/EMBER/tree/c7ef1372838250614544703e60f00fe1f2c351ca/src/ember/control_space), [8395a9e1](https://github.com/LinFyM/EMBER/tree/8395a9e1224cf93912de62ca276d6bf7733e5025/src/ember/control_space), [fb3eb80e](https://github.com/LinFyM/EMBER/tree/fb3eb80e9f3bfb22a00da53a87fdc4bd21c1efb7/src/ember/control_space), [bd0e37b0](https://github.com/LinFyM/EMBER/tree/bd0e37b060f6f5b55413a31556a0599d3a7888ed/src/ember/control_space) |
| source-W factor credit | pair学习；相邻1090读取 | [eadfe958](https://github.com/LinFyM/EMBER/tree/eadfe958ff17b0d1b4e1ff10265c234883b243b0/src/ember/control_space), [d963e8c3](https://github.com/LinFyM/EMBER/tree/d963e8c34199bd71a90b00ecf1843853c15cb086/src/ember/control_space) |
| native首写180 | 完整因子collective修复后的正式来源 | [cbd445a6](https://github.com/LinFyM/EMBER/tree/cbd445a6313661b07ae6fed00d271a9b2e67b4ef/src/ember/control_space) |

## 先检查哪些实际消费者

- **T：** `operator_writer/model.py`中TargetWrite、实际共享A、递归M与最终A/B；公共路径和教学prefix也要读，不能只看末层输出。
- **Native Correction / LocalField / PPW：** 历史`writer`中的候选构造、真实原生导数、功能loss与materialization消费者；设计描述和实际实现相互核对。
- **G/F：** `relation_writer`内feedback、credit、原生query和关系监督；区分GT辅助老师F与合法视频生成G。
- **Experience / Parameter / FR：** `experience_compiler`内episode/compilation、实际incoming参数编码、native_credit、loss与采样；FR实施小PG消费者和正式FM训练分别解释。
- **G/π：** `proposal_writer/runtime.py`核执行基底，Reader/Actor/selector和实际终点消费者核共同E、H与参数监督；不能因source权重冻结就误称raw-source-only。
- **N4/N16：** `video_effect`内scoring、compiler、真实分支续接与joint-distribution信用；schema修正前后分开。
- **source系列：** `control_space`内阶段相应的fit/feedback/learning、weights/native以及完整LoRA物化与原生执行；不同commit的同名文件不是同一种方法。

## 资产状态与不可恢复范围

Git保存代码，不保存大模型或训练状态。Experience/Parameter的原退休记录已附在本包，当前保留weights-only而非exact-resume；functional validation的392条raw H已退役，仅8条init0完整链仍在。不要把历史报告中的“完整checkpoint”解释为今天全部状态仍可用，也不要把代码历史可读当成大资产可从Git下载。

后继若需要执行，应另依当时有效合同核实际资产和环境；本次不恢复、不运行。
