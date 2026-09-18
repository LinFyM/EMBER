# A与learned frame-set：配对结果表

本表由完成的raw rows生成；差值为frame-set减A，区间按task-cluster bootstrap（20,000次，seed20260915）计算。
同一个seed、8个validation任务的有限诊断；总分接近或区间含零不证明等价。主解释预注册为900和1200。

## validation

| 更新 | A | frame-set | 差值 | 差值95%CI（百分点） | breadth A→set | R/G/L | churn | Jaccard |
|---:|---:|---:|---:|---|---|---|---:|---:|
| 300 | 99/400 | 79/400 | -20 | [-12.50, 1.25] | 5→5 | 54/25/45 | 70 | 0.435 |
| 600 | 88/400 | 109/400 | +21 | [-4.00, 16.50] | 6→5 | 54/55/34 | 89 | 0.378 |
| 900 | 140/400 | 142/400 | +2 | [-4.00, 5.75] | 6→7 | 102/40/38 | 78 | 0.567 |
| 1200 | 135/400 | 118/400 | -17 | [-12.00, 1.75] | 6→5 | 90/28/45 | 73 | 0.552 |

### Per-task

| 更新 | Suite / task | A | frame-set | R/G/L |
|---:|---|---:|---:|---|
| 300 | libero_10 / 1 | 20 | 25 | 10/15/10 |
| 300 | libero_10 / 2 | 0 | 0 | 0/0/0 |
| 300 | libero_goal / 3 | 0 | 0 | 0/0/0 |
| 300 | libero_goal / 6 | 31 | 21 | 16/5/15 |
| 300 | libero_object / 1 | 35 | 25 | 22/3/13 |
| 300 | libero_object / 3 | 3 | 3 | 2/1/1 |
| 300 | libero_spatial / 1 | 0 | 0 | 0/0/0 |
| 300 | libero_spatial / 3 | 10 | 5 | 4/1/6 |
| 600 | libero_10 / 1 | 4 | 23 | 0/23/4 |
| 600 | libero_10 / 2 | 0 | 0 | 0/0/0 |
| 600 | libero_goal / 3 | 1 | 0 | 0/0/1 |
| 600 | libero_goal / 6 | 35 | 34 | 27/7/8 |
| 600 | libero_object / 1 | 26 | 31 | 20/11/6 |
| 600 | libero_object / 3 | 9 | 16 | 3/13/6 |
| 600 | libero_spatial / 1 | 0 | 0 | 0/0/0 |
| 600 | libero_spatial / 3 | 13 | 5 | 4/1/9 |
| 900 | libero_10 / 1 | 33 | 28 | 21/7/12 |
| 900 | libero_10 / 2 | 0 | 1 | 0/1/0 |
| 900 | libero_goal / 3 | 0 | 0 | 0/0/0 |
| 900 | libero_goal / 6 | 30 | 38 | 25/13/5 |
| 900 | libero_object / 1 | 44 | 44 | 42/2/2 |
| 900 | libero_object / 3 | 16 | 17 | 5/12/11 |
| 900 | libero_spatial / 1 | 1 | 1 | 0/1/1 |
| 900 | libero_spatial / 3 | 16 | 13 | 9/4/7 |
| 1200 | libero_10 / 1 | 24 | 10 | 7/3/17 |
| 1200 | libero_10 / 2 | 0 | 0 | 0/0/0 |
| 1200 | libero_goal / 3 | 2 | 0 | 0/0/2 |
| 1200 | libero_goal / 6 | 40 | 41 | 34/7/6 |
| 1200 | libero_object / 1 | 37 | 38 | 33/5/4 |
| 1200 | libero_object / 3 | 18 | 21 | 11/10/7 |
| 1200 | libero_spatial / 1 | 0 | 0 | 0/0/0 |
| 1200 | libero_spatial / 3 | 14 | 8 | 5/3/9 |

### Per-suite

| 更新 | Suite | A | frame-set | R/G/L |
|---:|---|---:|---:|---|
| 300 | libero_10 | 20 | 25 | 10/15/10 |
| 300 | libero_goal | 31 | 21 | 16/5/15 |
| 300 | libero_object | 38 | 28 | 24/4/14 |
| 300 | libero_spatial | 10 | 5 | 4/1/6 |
| 600 | libero_10 | 4 | 23 | 0/23/4 |
| 600 | libero_goal | 36 | 34 | 27/7/9 |
| 600 | libero_object | 35 | 47 | 23/24/12 |
| 600 | libero_spatial | 13 | 5 | 4/1/9 |
| 900 | libero_10 | 33 | 29 | 21/8/12 |
| 900 | libero_goal | 30 | 38 | 25/13/5 |
| 900 | libero_object | 60 | 61 | 47/14/13 |
| 900 | libero_spatial | 17 | 14 | 9/5/8 |
| 1200 | libero_10 | 24 | 10 | 7/3/17 |
| 1200 | libero_goal | 42 | 41 | 34/7/8 |
| 1200 | libero_object | 55 | 59 | 44/15/11 |
| 1200 | libero_spatial | 14 | 8 | 5/3/9 |

### 相邻保持

| 模型 | 节点 | 前→后 | R/G/L | churn | Jaccard |
|---|---|---|---|---:|---:|
| A | 300_to_600 | 99→88 | 57/31/42 | 73 | 0.438 |
| A | 600_to_900 | 88→140 | 68/72/20 | 92 | 0.425 |
| A | 900_to_1200 | 140→135 | 98/37/42 | 79 | 0.554 |
| frame_set | 300_to_600 | 79→109 | 57/52/22 | 74 | 0.435 |
| frame_set | 600_to_900 | 109→142 | 86/56/23 | 79 | 0.521 |
| frame_set | 900_to_1200 | 142→118 | 91/27/51 | 78 | 0.538 |

## train

| 更新 | A | frame-set | 差值 | 差值95%CI（百分点） | breadth A→set | R/G/L | churn | Jaccard |
|---:|---:|---:|---:|---|---|---|---:|---:|
| 300 | 36/96 | 33/96 | -3 | [-12.50, 6.25] | 18→15 | 23/10/13 | 23 | 0.500 |
| 600 | 47/96 | 40/96 | -7 | [-16.67, 1.04] | 17→16 | 34/6/13 | 19 | 0.642 |
| 900 | 54/96 | 49/96 | -5 | [-15.62, 6.25] | 19→18 | 40/9/14 | 23 | 0.635 |
| 1200 | 62/96 | 57/96 | -5 | [-12.50, 2.08] | 20→19 | 52/5/10 | 15 | 0.776 |

### Per-task

| 更新 | Suite / task | A | frame-set | R/G/L |
|---:|---|---:|---:|---|
| 300 | libero_10 / 4 | 2 | 2 | 1/1/1 |
| 300 | libero_10 / 5 | 1 | 2 | 0/2/1 |
| 300 | libero_10 / 6 | 1 | 0 | 0/0/1 |
| 300 | libero_10 / 7 | 0 | 0 | 0/0/0 |
| 300 | libero_10 / 8 | 0 | 0 | 0/0/0 |
| 300 | libero_10 / 9 | 0 | 0 | 0/0/0 |
| 300 | libero_goal / 0 | 0 | 1 | 0/1/0 |
| 300 | libero_goal / 1 | 4 | 4 | 4/0/0 |
| 300 | libero_goal / 2 | 3 | 1 | 1/0/2 |
| 300 | libero_goal / 5 | 0 | 0 | 0/0/0 |
| 300 | libero_goal / 8 | 3 | 4 | 3/1/0 |
| 300 | libero_goal / 9 | 0 | 0 | 0/0/0 |
| 300 | libero_object / 2 | 1 | 1 | 0/1/1 |
| 300 | libero_object / 4 | 3 | 2 | 1/1/2 |
| 300 | libero_object / 5 | 1 | 1 | 1/0/0 |
| 300 | libero_object / 6 | 2 | 2 | 2/0/0 |
| 300 | libero_object / 8 | 3 | 4 | 3/1/0 |
| 300 | libero_object / 9 | 1 | 0 | 0/0/1 |
| 300 | libero_spatial / 0 | 2 | 4 | 2/2/0 |
| 300 | libero_spatial / 2 | 4 | 3 | 3/0/1 |
| 300 | libero_spatial / 4 | 1 | 1 | 1/0/0 |
| 300 | libero_spatial / 5 | 1 | 1 | 1/0/0 |
| 300 | libero_spatial / 7 | 1 | 0 | 0/0/1 |
| 300 | libero_spatial / 9 | 2 | 0 | 0/0/2 |
| 600 | libero_10 / 4 | 2 | 2 | 2/0/0 |
| 600 | libero_10 / 5 | 2 | 3 | 2/1/0 |
| 600 | libero_10 / 6 | 0 | 0 | 0/0/0 |
| 600 | libero_10 / 7 | 0 | 0 | 0/0/0 |
| 600 | libero_10 / 8 | 0 | 1 | 0/1/0 |
| 600 | libero_10 / 9 | 0 | 0 | 0/0/0 |
| 600 | libero_goal / 0 | 2 | 1 | 1/0/1 |
| 600 | libero_goal / 1 | 4 | 3 | 3/0/1 |
| 600 | libero_goal / 2 | 3 | 0 | 0/0/3 |
| 600 | libero_goal / 5 | 0 | 0 | 0/0/0 |
| 600 | libero_goal / 8 | 4 | 4 | 4/0/0 |
| 600 | libero_goal / 9 | 0 | 0 | 0/0/0 |
| 600 | libero_object / 2 | 4 | 2 | 2/0/2 |
| 600 | libero_object / 4 | 4 | 4 | 4/0/0 |
| 600 | libero_object / 5 | 1 | 0 | 0/0/1 |
| 600 | libero_object / 6 | 3 | 3 | 2/1/1 |
| 600 | libero_object / 8 | 4 | 4 | 4/0/0 |
| 600 | libero_object / 9 | 3 | 2 | 1/1/2 |
| 600 | libero_spatial / 0 | 3 | 3 | 3/0/0 |
| 600 | libero_spatial / 2 | 3 | 4 | 3/1/0 |
| 600 | libero_spatial / 4 | 1 | 0 | 0/0/1 |
| 600 | libero_spatial / 5 | 1 | 1 | 1/0/0 |
| 600 | libero_spatial / 7 | 3 | 2 | 2/0/1 |
| 600 | libero_spatial / 9 | 0 | 1 | 0/1/0 |
| 900 | libero_10 / 4 | 2 | 4 | 2/2/0 |
| 900 | libero_10 / 5 | 4 | 2 | 2/0/2 |
| 900 | libero_10 / 6 | 0 | 2 | 0/2/0 |
| 900 | libero_10 / 7 | 0 | 0 | 0/0/0 |
| 900 | libero_10 / 8 | 1 | 0 | 0/0/1 |
| 900 | libero_10 / 9 | 0 | 0 | 0/0/0 |
| 900 | libero_goal / 0 | 2 | 0 | 0/0/2 |
| 900 | libero_goal / 1 | 4 | 4 | 4/0/0 |
| 900 | libero_goal / 2 | 3 | 4 | 3/1/0 |
| 900 | libero_goal / 5 | 0 | 0 | 0/0/0 |
| 900 | libero_goal / 8 | 4 | 4 | 4/0/0 |
| 900 | libero_goal / 9 | 0 | 0 | 0/0/0 |
| 900 | libero_object / 2 | 3 | 3 | 3/0/0 |
| 900 | libero_object / 4 | 4 | 4 | 4/0/0 |
| 900 | libero_object / 5 | 2 | 1 | 0/1/2 |
| 900 | libero_object / 6 | 2 | 3 | 2/1/0 |
| 900 | libero_object / 8 | 4 | 4 | 4/0/0 |
| 900 | libero_object / 9 | 3 | 4 | 3/1/0 |
| 900 | libero_spatial / 0 | 4 | 2 | 2/0/2 |
| 900 | libero_spatial / 2 | 3 | 3 | 3/0/0 |
| 900 | libero_spatial / 4 | 3 | 2 | 2/0/1 |
| 900 | libero_spatial / 5 | 2 | 1 | 1/0/1 |
| 900 | libero_spatial / 7 | 2 | 1 | 0/1/2 |
| 900 | libero_spatial / 9 | 2 | 1 | 1/0/1 |
| 1200 | libero_10 / 4 | 3 | 3 | 3/0/0 |
| 1200 | libero_10 / 5 | 4 | 2 | 2/0/2 |
| 1200 | libero_10 / 6 | 2 | 1 | 1/0/1 |
| 1200 | libero_10 / 7 | 0 | 0 | 0/0/0 |
| 1200 | libero_10 / 8 | 1 | 0 | 0/0/1 |
| 1200 | libero_10 / 9 | 0 | 0 | 0/0/0 |
| 1200 | libero_goal / 0 | 1 | 1 | 0/1/1 |
| 1200 | libero_goal / 1 | 4 | 4 | 4/0/0 |
| 1200 | libero_goal / 2 | 4 | 2 | 2/0/2 |
| 1200 | libero_goal / 5 | 0 | 0 | 0/0/0 |
| 1200 | libero_goal / 8 | 4 | 4 | 4/0/0 |
| 1200 | libero_goal / 9 | 0 | 0 | 0/0/0 |
| 1200 | libero_object / 2 | 3 | 3 | 3/0/0 |
| 1200 | libero_object / 4 | 4 | 4 | 4/0/0 |
| 1200 | libero_object / 5 | 1 | 3 | 1/2/0 |
| 1200 | libero_object / 6 | 3 | 3 | 2/1/1 |
| 1200 | libero_object / 8 | 4 | 4 | 4/0/0 |
| 1200 | libero_object / 9 | 4 | 4 | 4/0/0 |
| 1200 | libero_spatial / 0 | 3 | 3 | 2/1/1 |
| 1200 | libero_spatial / 2 | 4 | 4 | 4/0/0 |
| 1200 | libero_spatial / 4 | 3 | 2 | 2/0/1 |
| 1200 | libero_spatial / 5 | 2 | 2 | 2/0/0 |
| 1200 | libero_spatial / 7 | 4 | 4 | 4/0/0 |
| 1200 | libero_spatial / 9 | 4 | 4 | 4/0/0 |

### Per-suite

| 更新 | Suite | A | frame-set | R/G/L |
|---:|---|---:|---:|---|
| 300 | libero_10 | 4 | 4 | 1/3/3 |
| 300 | libero_goal | 10 | 10 | 8/2/2 |
| 300 | libero_object | 11 | 10 | 7/3/4 |
| 300 | libero_spatial | 11 | 9 | 7/2/4 |
| 600 | libero_10 | 4 | 6 | 4/2/0 |
| 600 | libero_goal | 13 | 8 | 8/0/5 |
| 600 | libero_object | 19 | 15 | 13/2/6 |
| 600 | libero_spatial | 11 | 11 | 9/2/2 |
| 900 | libero_10 | 7 | 8 | 4/4/3 |
| 900 | libero_goal | 13 | 12 | 11/1/2 |
| 900 | libero_object | 18 | 19 | 16/3/2 |
| 900 | libero_spatial | 16 | 10 | 9/1/7 |
| 1200 | libero_10 | 10 | 6 | 6/0/4 |
| 1200 | libero_goal | 13 | 11 | 10/1/3 |
| 1200 | libero_object | 19 | 21 | 18/3/1 |
| 1200 | libero_spatial | 20 | 19 | 18/1/2 |

### 相邻保持

| 模型 | 节点 | 前→后 | R/G/L | churn | Jaccard |
|---|---|---|---|---:|---:|
| A | 300_to_600 | 36→47 | 29/18/7 | 25 | 0.537 |
| A | 600_to_900 | 47→54 | 39/15/8 | 23 | 0.629 |
| A | 900_to_1200 | 54→62 | 48/14/6 | 20 | 0.706 |
| frame_set | 300_to_600 | 33→40 | 27/13/6 | 19 | 0.587 |
| frame_set | 600_to_900 | 40→49 | 33/16/7 | 23 | 0.589 |
| frame_set | 900_to_1200 | 49→57 | 42/15/7 | 22 | 0.656 |
