# EMBER Durable Findings

## Research framing

- The broad information-to-parameter idea is meaningful only over a structured
  task family with executable source-task bridge supervision. It is not a
  universal optimizer claim.
- A static task-specification-to-adapter Writer is best described as an
  amortized task-conditioned parameter-update generator or hypernetwork.
- The candidate novelty lies in the complete controlled combination, not in
  language-conditioned weights, action-free video, LoRA, subspaces, or RL alone.

## Adopted scientific lessons

- Separate useful-update existence, representation, and amortized Writer
  acquisition into different gates.
- Functional behavior and closed-loop return take priority over raw parameter or
  LoRA-factor distance.
- Update diversity does not prove task specificity; average, retrieval,
  wrong-spec, scene-only, and shuffled-video controls are mandatory.
- Immediate gain, control harm, adaptation speed, final performance, and
  one-time meta-training cost must be reported separately.

## Adopted execution decisions

- Active compute cap: four A100 80GB GPUs, normally one or two for pilots.
- After a minimal correctness baseline, useful batching, simulator concurrency,
  and caching should target about 70GB used per allocated A100 with about 10GB
  average headroom. Unused memory is not filled with dummy allocations.
- Completed evaluations retain a compact local HTML/video gallery. Metrics,
  manifests, failure packets, `latest`, and designated evidence are durable;
  only verified-regenerable, unpinned older media is eligible for recorded
  cleanup when visual artifacts accumulate.
- SmolVLA plus LIBERO is the primary development surface. OpenVLA-OFT is scale
  confirmation after lower-cost gates survive.
- Neutral-prompt parameter compilation is a co-primary mechanism test.
- A language-only HyPoGen/DISC-style parameter generator is a required strong
  baseline.
- The geometry receives a real training signal through a differentiable
  low-dimensional source support/query loop before reward-based refinement.
- The default adaptation representation is a canonical center and soft geometry
  with residual escape, not an inescapable hard subspace.

## Verified design risks

- A successful same-embodiment action-hidden robot video proves an
  information/supervision conversion mechanism, not lower data-collection cost
  or human-to-robot transfer.
- LIBERO task scenes, layouts, language templates, filenames, episode length,
  and normalization statistics can leak task identity.
- The proposed LIBERO-90 task split remains a hypothesis until a task-factor and
  initialization audit is generated from pinned files.
- A geometry emitted by the Writer is meaningless unless its training objective
  and matched unit/global-metric comparisons are explicit.
- Joint Writer/base optimization creates a moving parameter coordinate system;
  shared base adaptation is optional and comes only after a frozen-base result.

## Unknowns requiring evidence

- Whether language leaves measurable incremental information for video on the
  selected task subset.
- Whether the intended action-policy matrices admit useful, safe local updates.
- Whether task oracles share a compact canonical functional representation.
- Whether a predicted geometry transfers from offline support/query learning to
  sparse-reward local adaptation.
- Useful-batch simulator throughput, memory scaling, and end-to-end iteration
  cost under the four-GPU ceiling; batch one is now measured but intentionally
  underutilizes the GPU.

## Verified Phase 0 substrate facts

- The compatibility set is pinned to LeRobot `v0.6.0` commit
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`, official LIBERO commit
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`, and the `hf-libero==0.1.4`
  runtime fork commit `8561c60eea2fb93096146f240194649df73d8b1e`.
- LIBERO semantic authority is recorded separately from the runtime package:
  BDDL tree `8ac44efab7572ed2a050fc606dda1b3585348523`, init-state tree
  `135bba718a31923846e197e9e459d63670dddde1`, and scenes/assets tree
  `81c6f610667545eb3990e5ca3edb19c3e5e7ee19`.
- The locked Linux environment uses Python 3.12, PyTorch 2.11.0+cu128,
  torchvision 0.26.0+cu128, Transformers 5.5.4, MuJoCo 3.8.1,
  robosuite 1.4.0, and `hf-libero` 0.1.4. The live driver satisfies the CUDA
  12.8 runtime floor; GPU availability still must be rechecked at every launch.
- The host lacks a system C++ frontend. A user-space Zig 0.16.0 C++ wrapper and
  CMake 3.31.6 reproduce the native EGL probe builds without `sudo` or system
  changes.
- The `bddl==1.0.1` wheel contains a duplicate top-level `.egg-info` payload,
  which makes environment consistency tools report two installations. The
  bootstrap removes it only when it byte-matches the installed `METADATA`.
- robosuite 1.4.0 hard-codes file logging to the shared path
  `/tmp/robosuite.log`. The version-guarded private macro disables only that
  handler; experiment stdout/stderr logging remains active. This avoids
  cross-user collisions without changing simulator behavior.
- The SmolVLA base revision is
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`. The official LIBERO smoke
  checkpoint revision is `31d453f7edd78c839a8bbc39744a292686daf0de` and
  is mechanics-only because its training tasks overlap LIBERO evaluation. It
  must never become the EMBER shared base or scientific evidence.
- The official smoke checkpoint declares a six-dimensional state feature while
  its stored state statistics are eight-dimensional. The official LeRobot CI
  path is reported to run, but this metadata/runtime discrepancy is a required
  probe rather than an assumption.
- SmolVLA construction and tokenization reference an unrevisioned SmolVLM name.
  EMBER pins `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467` and will use a generated local
  runtime view so both model and tokenizer resolve offline to that snapshot.
- Canonical demonstrations are the 90 HDF5 files under `libero_90` in
  `yifengzhu-hf/LIBERO-datasets` revision
  `f13aa24a3da8c43c7225569f28c562979fa0e35a`: 66,658,085,995 bytes total,
  50 demonstrations per task, with expected HDF5 tag `libero-v1`. The Hub cards
  disagree on licensing; the original LIBERO dataset release's CC-BY-4.0 is the
  recorded authority.
- The official downloader names `libero_100`, but the pinned Hub surface is
  `libero_90`; Phase 0 must download the exact pinned subdirectory directly.
- For `lerobot/libero-assets` at the pinned revision, the current snapshot is
  exactly 586 files and 422,320,936 bytes. Hub `usedStorage` reports
  492,798,408 bytes because it is repository-level storage accounting, not the
  current snapshot payload. Storage budgets and completeness tests use the
  former; the latter is retained only as provenance.

## Phase 0 official mechanics smoke

- The first launch attempt failed before GPU allocation because LeRobot 0.6.0's
  `LiberoEnv` does not accept `env.seed`. Removing only that invalid field and
  retaining the global `seed=1000` fixed the configuration layer. The failed
  command, traceback, and 0 MiB GPU telemetry remain in the local long-run
  record; no scientific threshold or task surface changed.
- The recovered offline run used the official overlap-trained mechanics-only
  checkpoint on `libero_spatial` task 0, one episode, seed 1000, one synchronous
  environment, and one A100. The task was “pick up the black bowl between the
  plate and the ramekin and place it on the plate.” It succeeded 1/1 at roughly
  step 78. This validates mechanics only and is not Gate -1 or policy-quality
  evidence.
- Core evaluation time was 19.168 seconds and total process wall time was 40.74
  seconds, including imports, environment construction, and model load. Peak
  GPU memory was 2,029 MiB; across the full sampled process window average GPU
  utilization was 1.60%, maximum utilization was 21%, and maximum host RSS was
  4,604,404 KiB. The retained metrics, telemetry, HTML, manifest, and 80-frame
  H.264 video total 58,817 bytes.
- The runtime LIBERO processor constructs an eight-dimensional state
  `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`, matching the checkpoint's
  eight-dimensional normalization statistics. The checkpoint config's declared
  six-dimensional state feature is stale metadata rather than the runtime
  observation contract; manifests and later oracle code must use the verified
  eight-dimensional authority.
- `torchcodec==0.11.1` is import-discoverable but cannot load because this host
  currently has no compatible shared FFmpeg libraries. LeRobot would therefore
  incorrectly select a broken default video backend by package presence alone.
  The bundled `imageio-ffmpeg` binary can inspect/render current artifacts, but
  a pinned working decoder backend remains required before video-backed data
  loading is declared reproducible.
