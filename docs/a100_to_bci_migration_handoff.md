# A100 → BCI Migration Handoff

状态：2026-08-03最终迁移authority。本文交给后续迁移智能体；本session不执行跨机
迁移，也不在BCI创建、覆盖或删除文件。

## 0. 2026-08-03 post-seal增量：必须做第二次同步

2026-08-02原迁移封存完成后，owner临时重新开放了一个A100研究窗口。因此迁移智能体
即使已经同步过原封存集，也必须再做一次本节规定的增量同步。当前所有训练、评测、
profile和内部GPU分析均已结束，没有需要继承的进程；不要迁移A100 Codex、venv、
cache或worktree。

唯一机器可读增量authority为：

```text
/data/ymdai/migration_manifests/ember_postseal_20260802/assets.tsv
/data/ymdai/migration_manifests/ember_postseal_20260802/README.md
```

`assets.tsv`当前有34行`must-transfer`，源端合计`16,483,938,529` bytes。它们包括：

- Target-Bound fresh0→200正式训练、4个paired correct400、winner内部分析及日志；
- Semantic Factor-Basis fresh0→400正式训练、8个paired correct400、macro200内部分析、
  training-dynamics分析及日志。

Semantic Factor-Basis完整correct400为
`69/91/118/127/117/81/126/120`，winner仍是macro200。Target-Bound为
`75/120/90/110`，winner为macro100。两者均是需要保留的漂移/架构证据，不能只迁
winner。variance-reduced estimator仅完成longest105 B20与fresh/exact-resume垂直
验证，没有0→200或rollout；其profile、smoke和日志标为`do-not-transfer`或
`debug-only`，只通过Git迁移实现、配置和结论。

本窗口没有新增MemLLM代码、模型、数据或结果；MemLLM仍只按原2026-08-02封存清单
迁移，不需要第二批科研artifact同步。

### 0.1 Git增量

在BCI执行`git fetch origin`后，以实时`origin/main`为canonical；它必须包含本文、
Semantic Factor-Basis和VR estimator，并满足：

```bash
git merge-base --is-ancestor 50662a842cfa5c6e0a4356587ea73ea95e1ff521 origin/main
```

需要逐commit审计时再fetch
`origin/codex/variance-reduced-functional-estimator`。不要复制A100 worktree或`.git`，
也不要恢复所有Codex refs。

### 0.2 正式artifact增量

先把整个`ember_postseal_20260802/`小目录同步到BCI manifest root，再由BCI从
`assets.tsv`生成精确文件列表。以下命令只展示安全形状；`A100_HOST`、staging和BCI
目标路径必须由迁移智能体按现场填写：

```bash
export A100_HOST='REPLACE_ME'
export BCI_STAGE='/absolute/path/.incoming-20260803'
export POSTSEAL_LEDGER="$BCI_MANIFEST_ROOT/ember_postseal_20260802/assets.tsv"
export POSTSEAL_LIST="$BCI_STAGE/ember-postseal-must-transfer.txt"

mkdir -p "$BCI_STAGE/postseal-data-ymdai"
awk -F '\t' 'NR > 1 && $7 == "must-transfer" {
  path = $1
  sub("^/data/ymdai/", "", path)
  print path
}' "$POSTSEAL_LEDGER" > "$POSTSEAL_LIST"

rsync -aH -r --partial --info=progress2 --protect-args \
  --files-from="$POSTSEAL_LIST" \
  "$A100_HOST:/data/ymdai/" \
  "$BCI_STAGE/postseal-data-ymdai/"
```

然后只把staging中的`outputs/ember/`合并到`$EMBER_OUTPUT_ROOT/`，把
`logs/ember/`合并到`$EMBER_LOG_ROOT/`；不用`--delete`，不覆盖来源不明的BCI文件。
同一staging和命令可安全重跑。完成后逐行确认34个`must-transfer`源对象在BCI映射后
存在，正式correct roots各有400 rows，训练root的checkpoint/manifest可读。无需重新
复制原封存的整个122GB树，也不需要为本增量重算全量数据hash。

## 1. 完成定义与安全边界

迁移完成必须同时满足：

1. EMBER与MemLLM的Git源码、远端分支/标签和两份recovery bundle可读；
2. EMBER frozen source policy、正式训练checkpoint、raw evaluation rows、aggregate、
   internal analysis、logs、tokenizer、LIBERO simulation assets及必要feature cache完整；
3. MemLLM保留的data/model/result树和科学hash完整；
4. 新环境从锁文件或pinned freeze重建，不复制A100 venv；
5. BCI上的路径由新host-local映射管理，历史sealed artifacts不被改写；
6. 新Codex不依赖A100会话、memory、auth或worktree即可准确接手；
7. 未经owner重新授权，不启动Target-Bound或任何GPU实验。

禁止把整个`/data/ymdai`盲目rsync到BCI，也禁止对既有BCI目录使用`--delete`。所有
传输先进入明确的`.incoming-20260802` staging目录，验证后再切换。不得覆盖来源不明、
dirty或仍被消费的旧服务器资产。

## 2. A100最终现场

迁移准备前：

```text
EMBER main: f0b123f20f531baf4bfc5c6f75eb96af27f33ac1
Target-Bound remote branch: b260a57a94dc21bd3446b212bfa42f71b037ce13
MemLLM scientific main before migration docs: edc549d4e4ad9cc36584aa9bcc1f84b55e8f40af
```

迁移文档/路径可移植性提交会在上述EMBER main之上形成；迁移智能体应以实时
`origin/main`为准，并核验本文件存在。当前没有EMBER/MemLLM训练、评测、torchrun或
tmux。不要把A100的GPU4–7约束自动复制到BCI；目标机设备边界必须重新获取authority。

原迁移封存、post-seal研究重启前的主要占用（allocated bytes）：

| 路径 | bytes | 处置 |
| --- | ---: | --- |
| `/data/ymdai` | 229,312,688,128 | EMBER提交前最终快照；Git/Codex活动会带来小幅变化 |
| `/data/ymdai/outputs/ember` | 102,850,465,792 | SSH迁移；正式证据 |
| `/data/ymdai/ember_data` | 100,448,129,024 | 主要是LIBERO数据；先查BCI是否已有 |
| `/data/ymdai/ember_assets` | 426,569,728 | 当前LIBERO simulation assets；SSH或精确重下 |
| `/data/ymdai/logs/ember` | 439,922,688 | SSH迁移 |
| `/data/ymdai/memllm_migration_20260708` | 18,970,701,824 | SSH迁移 |
| EMBER `.venv` | 0 | 已删除；从`uv.lock`重建 |
| MemLLM venv | 0 | owner关闭消费者后已删除；重建 |
| `/data/ymdai/migration_manifests` | 151,117,824 | SSH迁移；提交前快照 |

`/data/ymdai`由约430.78GB降到约229.31GB，净释放201,471,401,984 bytes。删除内容
及理由：

- 52个Writer rollout LoRA caches：55,392,653,312 bytes；
- 138个profile/smoke/resume/WIP/aborted/debug roots：27,977,928,704 bytes；
- 退役SmolVLA outputs/numeric dataset及旧asset cache：50,902,618,112 bytes；
- source-base rejected EMA和optimizer/DDP/scheduler resume state：约24.48GB；
- superseded feature cache v1和endpoint LoRA tensor cache：约2.83GB；
- reseal/capacity/mechanics operation-only roots：约1.00GB；
- 可按精确revision重下的generic `lerobot/pi05_base`：14,467,239,936 bytes；
- owner明确不迁移的Codex archived sessions：5,653,884,928 bytes；
- 已完成验证的EMBER A100 venv、可重建uv/Hugging Face caches，以及owner关闭消费者
  后的MemLLM A100 venv；
- 55个clean辅助worktree、本地实验branches和obsolete stash。

清理后的原始contract测试发现`hf-libero`仍通过绝对symlink消费旧asset root。已只按
`lerobot/libero-assets`精确revision`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`
恢复586文件/426,569,728 bytes；四个原始失败测试通过，旧额外cache/revision不恢复。
correction ledger和586-file hash list在cleanup manifest目录。

精确manifest、各manifest SHA、环境freeze和Git bundles位于：

```text
/data/ymdai/migration_manifests/a100_cleanup_20260802
/data/ymdai/migration_manifests/git_bundles
```

cleanup总清单为`a100_cleanup_20260802/SHA256SUMS`，自身SHA256为
`338f8a0b4d7e9a1a16c788fb307cff15c02ab3baa81aa1838059ecb25a87173a`。

历史文档若引用已删除profile/resume/cache root，不表示正式evidence损坏。原封存的
60个checkpoint roots和406个complete evaluation roots，再加本节post-seal的2个
正式训练root与12个正式correct400 root，是task漂移、checkpoint轮换及架构×recipe
复核证据，故有意没有“只留winner”。

## 3. 资产分流总表

机器可读版本：
[`a100_retained_assets_20260802.tsv`](a100_retained_assets_20260802.tsv)。

### 3.1 通过GitHub

| 项目 | GitHub动作 | 必须核验 |
| --- | --- | --- |
| EMBER | clone `LinFyM/EMBER`，取实时`origin/main` | 本迁移文档、clean status |
| Target-Bound | fetch `origin/codex/target-bound-role-program` | commit `b260a57a...` |
| MemLLM | clone `LinFyM/Episodic-Memory-Chatbot` | live`origin/main`含迁移handoff；`edc549d4...`仅为迁移文档前科研点 |
| MemLLM历史 | fetch 7个`archive/*` tags | tags与repo文档一致 |

不要复制两个repo的`.git`目录或A100 worktree树。EMBER迁移前全refs bundle包含138
refs、旧local branches、worktree heads和stash；它用于灾难恢复，不是默认工作区：

```text
/data/ymdai/migration_manifests/git_bundles/ember_all_refs_pre_migration_cleanup_20260802.bundle
SHA256 c78fb94d0c2bafe460b4d269925c619250d72fdcda1018548b6a3a458aa44ec7
```

MemLLM bundle：

```text
/data/ymdai/migration_manifests/git_bundles/memllm_all_refs_pre_cleanup_20260722.bundle
SHA256 feef0e906842bb6f2a092a4bfbdb3cddb0653fbd346cb510715fec8b43ca7656
```

在BCI只需先执行`git bundle verify BUNDLE_PATH`。不要批量恢复`refs/codex/*`；需要某
个历史commit时再从bundle按SHA或明确ref fetch到`refs/archive/a100/*`，避免污染
canonical branches。

### 3.2 通过SSH/rsync

默认必迁，不含LIBERO HDF5约122.8GB：

| A100源 | 约大小 | 目标逻辑路径 | 原因 |
| --- | ---: | --- | --- |
| `/data/ymdai/outputs/ember/` | 102.85GB | `$EMBER_OUTPUT_ROOT/` | 正式weights/results/analysis |
| `/data/ymdai/logs/ember/` | 0.44GB | `$EMBER_LOG_ROOT/` | 正式运行日志 |
| `/data/ymdai/ember_data/openpi/` | 4.3MB | `$EMBER_TOKENIZER_ROOT/` | 精确tokenizer |
| `/data/ymdai/ember_assets/` | 0.43GB | `$EMBER_ASSET_ROOT/` | active LIBERO simulation assets |
| `/data/ymdai/migration_manifests/` | 0.15GB | `$BCI_MANIFEST_ROOT/` | bundles、deletion ledger、freeze |
| `/data/ymdai/memllm_migration_20260708/` | 18.97GB | 见MemLLM反向映射 | 唯一data/model/result树 |

EMBER outputs中两个最关键资产：

```text
frozen source run:
  pi05_source_base_v1_seed7_1k_e2cc238_20260722
  allocated bytes: 9,354,506,240
  selected raw policy SHA256:
  60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36

canonical feature cache v2:
  pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722
  allocated bytes: 17,994,743,808
```

source checkpoint已精简为frozen inference asset：raw `policy/`、trainer_state和原
manifest保留；rejected EMA及训练resume state已删除。它能继续做下游Writer训练与
评测，但不能exact-resume source-SFT step1000。

推荐从BCI拉取，变量值必须由迁移智能体先核验：

```bash
export A100_HOST='REPLACE_ME'
export BCI_STAGE='/absolute/path/.incoming-20260802'

mkdir -p "$BCI_STAGE/outputs/ember"
rsync -aH --partial --info=progress2 --protect-args \
  "$A100_HOST:/data/ymdai/outputs/ember/" \
  "$BCI_STAGE/outputs/ember/"
```

对logs、tokenizer、simulation assets、manifests和MemLLM使用独立显式命令。不要把
变量设为空；不要把
`/`、home或整个磁盘作为目标；不要使用`--delete`。传输中断时对同一staging目录
重跑即可。

### 3.3 先查重，再SSH或按revision下载

LIBERO HDF5：

```text
A100 path:
/data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a
allocated bytes: 100,443,860,992
Hugging Face: yifengzhu-hf/LIBERO-datasets
revision: f13aa24a3da8c43c7225569f28c562979fa0e35a
```

BCI很可能已有LIBERO资产。先核对revision目录、sealed manifest、文件大小和HDF5
schema；一致则不传。若没有，网络下载和SSH二选一，但必须锁定上述revision。不要
通过文件名相似就复用，也不要运行结果驱动的数据修补。

LIBERO simulation assets：

```text
A100 path:
/data/ymdai/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
Hugging Face dataset: lerobot/libero-assets
revision: 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
files: 586
file-list SHA256: 721aa2484de396be5267e936f115ddd5f03ffd12e0849cc1cd05bb17454996b9
```

它是当前`hf-libero` runtime必需资产，不是退役SmolVLA cache。体积仅426.57MB，默认
可SSH迁移；也可按上述revision重下并用file list校验。BCI安装后不要保留指向A100
`/data/ymdai/ember_assets`的绝对symlink；设置`EMBER_LIBERO_ASSETS_ROOT`指向BCI
实际snapshot。代码仍兼容package-local assets作为默认。

MemLLM的Qwen/BGE理论上可重下，但revision未被足够精确地锁定；默认随19GB
MemLLM物理树SSH迁移并校验checkpoint hashes，比重下更稳。

### 3.4 在BCI重建/下载，不迁移

- EMBER `.venv`、MemLLM venv；
- `.cache`、uv/pip/Hugging Face cache、`.nvm`、`.codex-cli`、VS Code server、
  GitHub CLI和临时downloads；
- A100 Codex sessions、archived sessions、auth、credentials、logs、plugin cache、
  worktrees；
- generic `lerobot/pi05_base`本地快照；需要source重训时按以下identity重下：

```text
repo: lerobot/pi05_base
revision: 7de663972b7817d2c4cf2d84c821153dfea772e9
model.safetensors bytes: 14,467,165,872
model SHA256: 0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f
config SHA256: 367869712a2847c27e95c431ecb03f17bec4eee01a63995e2fb2d91940752b53
```

已退役的SmolVLA额外cache、numeric dataset、旧`ember_outputs`和删除的cache不得
迁回；上节列出的426.57MB LIBERO simulation snapshot必须保留或精确重下。

### 3.5 未纳入默认迁移

- `/data/ymdai/cat-pr`约764MB：来源与两个项目关系不明确，未删除，也不应由EMBER
  迁移智能体擅自处理；交owner单独决定。
- 当前`.codex/sessions`用于本次会话收尾，不是科研authority；不迁移。
- `.codex/tmp`内少量旧prereg/scratch未因容量而冒险删除；不默认迁移。科研决定已在
  Git文档和正式output中封存。

## 4. BCI路径策略

### 4.1 EMBER

不要假设BCI有`/data/ymdai`。迁移智能体应先检查`/data0/user/ymdai`和
`/data1/user/ymdai`的容量、现有资产和权限，再选一个逻辑根，例如：

```bash
export EMBER_STORAGE_ROOT='/data1/user/ymdai'
export EMBER_STORAGE_CAP_BYTES='REPLACE_WITH_OWNER_CAP'
export EMBER_REPO_ROOT="$EMBER_STORAGE_ROOT/projects/EMBER"
export EMBER_OUTPUT_ROOT="$EMBER_STORAGE_ROOT/outputs/ember"
export EMBER_LOG_ROOT="$EMBER_STORAGE_ROOT/logs/ember"
export EMBER_DATA_ROOT="$EMBER_STORAGE_ROOT/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a"
export EMBER_TOKENIZER="$EMBER_STORAGE_ROOT/ember_data/openpi/paligemma_tokenizer.model"
export EMBER_ASSET_ROOT="$EMBER_STORAGE_ROOT/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
export EMBER_LIBERO_ASSETS_ROOT="$EMBER_ASSET_ROOT"
export EMBER_SOURCE_RUN="$EMBER_OUTPUT_ROOT/pi05_source_base_v1_seed7_1k_e2cc238_20260722"
export EMBER_SOURCE_CHECKPOINT="$EMBER_SOURCE_RUN/checkpoints/step_00001000"
```

示例路径和cap不是authority；实际选盘并取得owner容量边界后，把host-local变量放在
未提交的shell/env文件中。评测preflight现在读取`EMBER_STORAGE_ROOT`与
`EMBER_STORAGE_CAP_BYTES`而非固定A100值。训练/评测资产仍由
CLI显式传入：

```text
--source-run
--checkpoint / --source-checkpoint
--tokenizer-path
--data-root / --writer-video-data-root
--output-dir
```

不要改写历史`run_contract.json`、analysis或sealed config中的A100绝对路径；那些是
provenance。新命令使用BCI实际路径。若某个旧analyzer错误地把provenance路径当活动
输入，应在新Git提交中增加显式CLI override，而不是批量sed artifacts或创建假的
顶层`/data/ymdai`。

### 4.2 MemLLM反向映射

A100物理树有意保留了旧BCI相对布局。应按两个明确子树分别迁回，而不是把外层
`memllm_migration_20260708`原样嵌套：

```text
A100:
/data/ymdai/memllm_migration_20260708/data0/user/ymdai/LLM_memory/...
BCI:
/data0/user/ymdai/LLM_memory/...

A100:
/data/ymdai/memllm_migration_20260708/data1/user/ymdai/memllm_dense_ttt/...
BCI:
/data1/user/ymdai/memllm_dense_ttt/...
```

先传到同盘`.incoming-20260802`，核验后再与目标合并。不要覆盖旧BCI上未知的
`LLM_memory`或`memllm_dense_ttt`内容。clone MemLLM repo后重建ignored symlinks：

| workspace link | BCI target |
| --- | --- |
| `assets` | `/data0/user/ymdai/LLM_memory/memllm_mainline_ignored_artifact_archive_20260620_1415/data/mlp_memory` |
| `data/benchmarks` | `/data0/user/ymdai/LLM_memory/memllm_mainline_ignored_artifact_archive_20260620_1415/data/benchmarks` |
| `models` | `/data0/user/ymdai/LLM_memory/memllm_mainline_ignored_artifact_archive_20260620_1415/models` |
| `data/clean_large_100k_v1` | `/data1/user/ymdai/memllm_dense_ttt/clean_large_100k_v1` |
| `data/dsom_runs` | `/data1/user/ymdai/memllm_dense_ttt/dsom_runs` |
| `data/lora_runs` | `/data1/user/ymdai/memllm_dense_ttt/lora_runs` |
| `data/migration_manifests` | 迁移智能体选定的manifest root |
| `venv` | 新建BCI venv；不能指向A100路径 |

在创建symlink前逐个执行`test ! -e LINK && test ! -L LINK`或人工审查现有target；不要
使用会无条件覆盖的`ln -sfn`。

## 5. 环境重建

环境版本和完整freeze：

```text
/data/ymdai/migration_manifests/a100_cleanup_20260802/environment_versions_20260802.json
/data/ymdai/migration_manifests/a100_cleanup_20260802/ember_pip_freeze_20260802.txt
/data/ymdai/migration_manifests/a100_cleanup_20260802/memllm_pip_freeze_20260802.txt
```

EMBER：Python`3.12.3`、torch`2.11.0+cu128`、transformers`5.5.4`、
peft`0.19.1`。以tracked `uv.lock`为authority：

```bash
cd "$EMBER_REPO_ROOT"
uv sync --frozen
```

`ember_pip_freeze`只作现场审计；其中editable EMBER路径仍是A100绝对路径，不应原样
安装。`lerobot`锁定commit
`30da8e687a6dfc617fcd94afc367ac7071c376ce`。

MemLLM：Python`3.10.20`、torch`2.11.0+cu128`、transformers`5.6.2`、
peft`0.19.1`、faiss-cpu`1.14.3`、sentence-transformers`5.5.0`。其
`requirements.txt`只有宽松范围，优先根据pinned freeze新建环境，再以repo tests
校验。不要复制7.6GB venv；其shebang和解释器路径绑定A100。

BCI NVIDIA driver必须支持PyTorch CUDA 12.8 runtime。A100系统toolkit 12.4不是要
复制的环境组件。

## 6. 验证顺序

### 6.1 A100源端收尾

1. `git status --short --branch`；EMBER/MemLLM均clean且main对齐origin；
2. `git bundle verify`两份bundle并核对SHA256；
3. 核验无训练、评测、torchrun和tmux；
4. 记录最终`du -s -B1 /data/ymdai`及各传输根大小；
5. 验证cleanup目录`SHA256SUMS`；该文件自身SHA见第2节。

### 6.2 BCI落地

1. 核验destination free space和existing assets，先去重；
2. clone repos，核验Git commits/branches/tags；
3. 传manifests和bundles，先验hash；
4. 分根rsync到staging，重跑直到无增量；
5. 验证frozen source policy SHA、tokenizer SHA、关键MemLLM hashes和内部checkpoint
   manifests；
6. 重建路径变量和MemLLM symlinks；
7. 重建环境，运行CPU compile/tests；
8. 只做read-only asset inspector，不做GPU smoke；
9. owner确认迁移成功并重新授权后，才恢复科研实验。

关键EMBER hashes：

```text
tokenizer SHA256:
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6

source raw policy SHA256:
60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36

source original checkpoint manifest SHA256:
c236cb2d92e5e9b859a6266c059a685d715d4bc129fb8cb5f36f60b6351cd6bf
```

关键MemLLM hashes继续以`data/README.md`为authority，包括selected Wiki
`c0f339d3...1a7e`、FAISS`3b206bdd...6acc`、historical LoRA adapter
`595137ac...e96f3`及保留Writer/B checkpoints。

## 7. 新Codex如何接手

不迁移A100 Codex。BCI上重新安装、重新认证，然后：

1. clone EMBER与MemLLM，不恢复A100 session或auth文件；
2. EMBER先读`AGENTS.md`、本文、`docs/active_session_handoff.md`、
   `docs/execution_brief.md`；改变实验状态前完整读完`AGENTS.md`要求的authority；
3. MemLLM先读`AGENTS.md`、`docs/current/a100_to_bci_migration.md`、
   `docs/current/restart_status.md`和`data/README.md`；MemLLM仍暂停；
4. 核验main已包含`50662a8`及本文；Target-Bound分支仅作历史审计，不自动merge或
   launch；
5. 用BCI现场替换host-local路径变量，绝不依赖本文件中的示例盘符；
6. 向owner报告Git、assets、hash、环境和无活动GPU作业状态，等待恢复实验authority。

旧Codex对话、sessions、memories和Codex-generated refs都不是科学authority；本次已把
必要状态压缩到Git文档、正式artifacts和migration manifests。

## 8. 迁移后的科研起点

迁移成功且owner重新授权后，首个候选不再是Target-Bound。Target-Bound已正式完成并
以`75/120/90/110`拒绝第二小时；Semantic Factor-Basis也已完成400 macros并以
`69/91/118/127/117/81/126/120`证明task routing有效但漂移未解。下一项最小可证伪
实验是在不改SFB拓扑、目标期望和full24 operator的前提下正式测试variance-reduced
functional estimator：

```text
pull latest origin/main
-> CPU focused regression
-> BCI live GPU/storage preflight
-> longest-105-frame B20 profile
-> fresh0→1→exact-resume1→3
-> fresh macro0→200
-> paired correct400 at 50/100/150/200
-> root-cause gate
```

已有A100 profile/smoke只能证明可运行，不得warm-start BCI正式训练，也不能冒充性能
证据。该实验必须同时检查single-checkpoint absolute、task breadth、状态换手和梯度
方向稳定性；只有形成强single winner后才进入视频五臂。本迁移handoff本身不授予任何
实验或GPU启动权限。
