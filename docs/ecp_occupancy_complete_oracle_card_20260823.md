# ECP Occupancy-Complete Privileged Realization Oracle

状态：2026-08-23 **preregistered before retained implementation and GPU launch**。

## Scientific question

在fold0 held5上，若用多个独立成功task policies在真实`initial/successful/candidate/recovery` occupancy上定义分布式
policy-effect等价类，一个从stable shared carrier出发、固定A且只求Delta-B的统一12-step solver，能否生成显著高于carrier、
覆盖Goal/Long并恢复相当比例task-local upper bound的一套rank16 LoRA？

## Why this question

PECS的59/250表明exact policy effects有真实增量，但其支持只有教学视频选出的稀疏帧。它没有检验policy执行时的双相机
观测、candidate drift或recovery states。MDCO及v24则已充分否定当前learned hyperdecoder family。这个oracle只补最早未测
接口，不改变video encoder、learned compiler width或训练recipe。

## Inputs and information boundary

- task-local privileged inputs：train-task actions、三个successful policy particles、rollout observations、success与BDDL progress；
- frozen shared inputs：source PI0.5、native observer projections、fit19 stable carrier；
- held5之间不共享梯度、参数、optimizer、task table或checkpoint；
- validation/test actions、reward和checkpoint selection读取均为0；
- 产物只作Stage 1B mechanism oracle，不是deployment Writer或第二adapter。

## Fixed data panel

每task四类先等权、类内等权，共48个anchors：

- initial：fixed init IDs `0,7,14,21,28,35,42,49`；
- successful：现有latest、新独立seed step2000、现有earliest各一条strict-success trajectory，每条8 strata；
- candidate：prior PECS trajectory candidate在fixed init ID 0上的8 strata；
- recovery：source按`1,26,2,27`顺序取得第一条failed trajectory的8 strata。

所有policy-effect查询使用每anchor固定noise seed及其antithetic pair、官方10 denoising steps、双相机+state-conditioned language
prefix。缓存source、carrier和三个member的owner DCT4、full flow trajectory与integrated action trajectory。

## Fixed realization

每target固定carrier `A_c`，只优化`Delta B`：

```text
W = (B_c + Delta B) A_c = W_carrier + Delta B A_c
```

objective固定包括stage-consistent member-particle soft minimum、carrier no-worse barrier、source/shared preservation、effective
trust和category/stage balance。solver沿用12 steps、inverse-sqrt decay与owner-normalized gradient；不early-stop、不挑step、
不保存task-local optimizer。一次fit-task profile只拥有数值与资源修正权，不拥有科学调参权。

## Direct closed-loop decision

final step12直接跑原held5 fixed250 strict paired panel。通过条件同时为：

- candidate至少74/250且相对carrier43净增至少20；
- 5/5 task非零，至少4/5严格高于carrier；
- Goal和Long各自非零；
- carrier success retention至少33/43；
- overall oracle-normalized recovery至少0.35，且至少4/5 task为正；
- pairing、single-LoRA与information-wall合同无误。

若final12接近或通过，只补step10/11相邻稳定性，不改变final选择。own cosine、retrieval、loss下降与exact-row retention均不能
代替closed loop。

## Allowed outcomes

- **Pass**：授权轮换fold、补source-unseen/process-identifying meta data并进入Stage 1C shared video-to-effect learning；
- **Teacher-bank non-pass**：若独立member或四类occupancy不完整，先补Stage 1A，不运行solver；
- **Realization non-pass**：暂停分析，不自动生成successor或小扫；该结果关闭本卡组合并继续阻止video predictor；
- **Engineering invalidation**：只在可复现实现、数据、OOM或合同错误时修复并重跑同一科学卡，不增加版本号。
