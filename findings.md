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
- Whether closed-loop evaluation identity can be made invariant to policy batch
  size. The pinned upstream maps explicit per-environment seeds and fixed
  init-state indices, but larger-batch calibration produced some different
  outcomes for the same seed/index prefix; reset, first-action, and trajectory
  probes must localize the divergence before scientific comparisons.

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
  incorrectly select it by package presence alone. EMBER now explicitly pins
  `av==15.1.0` and `video_decode_backend="pyav"`; a generated H.264 round-trip
  test and LeRobot timestamp-selection test pass. TorchCodec is not selected
  unless a later pinned shared-FFmpeg path beats this validated backend.

## Phase 0 throughput calibration

- The first matched load rung used the same task 0 with batch 8, episodes 8,
  seeds 1000–1007, synchronous vector environments, and one A100. Seven of
  eight episodes succeeded; by LeRobot's preserved batch order, seed 1002 was
  the failure and ran all 280 steps, while successful videos contained 69–83
  frames.
- Evaluation time was 119.131 seconds, or 14.891 seconds per retained episode,
  versus 19.168 seconds for the batch-one baseline. Peak GPU memory was 6,003
  MiB, active-window mean memory was 3,476.6 MiB, active-window mean GPU
  utilization was 5.60%, peak utilization was 61%, and maximum host RSS was
  11,922,208 KiB. The complete metrics/telemetry/gallery/video surface was
  524,142 bytes, so accumulated media remains far below cleanup pressure.
- Batch 8 amortizes model inference, but synchronous simulator stepping and the
  longest unfinished environment dominate the wall clock. A matched
  asynchronous batch-8 run preserved the exact eight-outcome vector while
  reducing evaluation time from 119.131 to 63.403 seconds, a 1.879x speedup.
  It used 6,167 MiB peak GPU memory, 5,009.2 MiB active-window mean memory,
  11.71% active-window mean utilization, and 82.08 seconds total process time.
  Asynchronous vector environments are therefore the selected high-throughput
  path; synchronous mode remains the low-complexity diagnostic control.
- Scaling the asynchronous one-batch run to 32 episodes produced 26 successes
  in 120.004 evaluation seconds (0.2667 episodes/s), with 20,477 MiB peak and
  16,295.7 MiB active-window mean GPU memory, 24.35% active-window mean GPU
  utilization, and 138.72 seconds total process time. Scaling to 96 episodes
  produced 78 successes in 249.347 seconds (0.3850 episodes/s), with 58,560 MiB
  peak and 44,174.2 MiB active-window mean memory, 34.92% active-window mean
  utilization, and 268.30 seconds total process time.
- The final measured rung used 112 asynchronous environments and one A100. It
  produced 84/112 successes in 282.943 evaluation seconds, or 0.3958 episodes/s:
  2.8% faster than batch 96 despite a longer failure tail. Peak memory was
  68,080 MiB, leaving 13,076 MiB reported free; active-window mean memory was
  50,664.6 MiB, active-window mean utilization was 35.49%, peak utilization was
  100%, and total process time was 302.21 seconds. Batch 112 is the measured
  resource-rich single-GPU throughput limit; batch 96 is the conservative rung
  when host CPU/RAM contention matters. Larger batches are not justified by the
  marginal gain and would erode the requested OOM headroom.
- These calibration success rates are mechanics/throughput observations, not
  benchmark evidence. Source inspection confirms that LeRobot supplies seeds
  `1000+i`, while `LiberoEnv` binds each sub-environment to init-state index
  `i`; nevertheless, outcome prefixes differed at 2/32 positions between
  batches 32 and 96 and at 10/96 positions between batches 96 and 112. The
  current evidence does not yet distinguish batch-dependent policy numerics,
  simulator sensitivity, or another implementation effect. Gate -1 must record
  reset observations and initial actions before interpreting any return change.
- PyAV decoded all 833 retained frames from the batch-8 videos in 0.123 seconds
  (about 6.8k frames/s) and LeRobot selected four requested timestamps from the
  280-frame failure video as `uint8 [4, 3, 360, 360]` in 0.289 seconds. Decoder
  throughput is not the current evaluation bottleneck.

## Phase 0 ten-task mechanics sweep

- One sequential process loaded the pinned overlap-trained checkpoint once and
  exercised `libero_spatial` task IDs 0–9 with one synchronous environment,
  seed 1000, init-state index 0, the official two-camera mapping, and the
  relative controller. All ten task/BDDL/init-state paths completed without a
  configuration, simulator, camera, normalization, or artifact error.
- Nine of ten single episodes succeeded. Task 5, “pick up the black bowl on the
  ramekin and place it on the plate,” returned zero reward and ran all 280
  steps; the other task videos contained 81, 109, 96, 94, 124, 175, 122, 93,
  and 120 frames. This one sample is retained as a mechanics observation only.
  It does not justify policy-quality interpretation, task-specific tuning, or a
  Gate decision because the checkpoint overlaps the evaluated suite.
- Core evaluation time was 97.362 seconds for ten episodes and total process
  wall time was 116.77 seconds. Peak GPU memory was 2,029 MiB, active-window
  mean memory was 1,849.6 MiB, active-window mean GPU utilization was 8.38%,
  peak utilization was 44%, and maximum host RSS was 6,084,464 KiB. Low GPU
  occupancy is intentional for this one-episode-per-task compatibility sweep;
  duplicating mechanics episodes solely to fill memory would add no evidence.
- The retained artifact is 784,268 bytes and contains aggregate/per-task JSON,
  resource telemetry, a hash manifest, an HTML gallery, and one valid H.264
  video for every task. A decoded first/last-frame review showed the ten
  expected spatial bowl layouts and non-empty robot trajectories. The canonical
  local review page is `$EMBER_OUTPUT_ROOT/phase0/latest/index.html`.

## Gate -1 evaluation-identity mechanics diagnostic

- The first frozen identity probe ran from clean commit `07baaca` on the
  official-overlap `libero_spatial` task 0 only. It explicitly bound seeds 1000
  and 1001 to init-state indices 0 and 1 and their canonical hashes, plus the
  BDDL/init-state-file hashes, camera mapping, relative controller, reset
  observation, and five dummy-action steps. Six predeclared conditions cover
  sync/async batches 1 and 2 plus same-mode batch-1 repeats. No policy was
  loaded after the mechanics stop condition fired.
- Strict bitwise observation identity did not pass: only 1/7 reset comparisons
  and 0/7 fixed-trajectory comparisons were exact. Every mismatched leaf was a
  `uint8` camera image. The maximum absolute channel delta was 1, and the
  largest leaf changed 28 of 388,800 values (0.0000720); the same sparse
  variation also appeared between repeated runs of an unchanged mode and batch
  size. All compared non-pixel state, reward, termination, and truncation leaves
  were exact.
- This evidence excludes seed, init-state selection/hash, BDDL authority,
  camera-name mapping, controller, and simulated-state divergence as the source
  of the initial mismatch. It instead classifies the strict Gate as a
  benchmark/specification ambiguity with renderer-level implementation
  nondeterminism. It does not yet establish the exact EGL/context cause or the
  effect on initial actions and closed-loop trajectories, and it does not
  authorize a nonzero pixel tolerance.
- The canonical long-run record is
  `.codex/longrun/gate_minus1_identity_mechanics_20260717_160231`; the retained
  artifact is
  `$EMBER_OUTPUT_ROOT/gate_minus1/evaluation_identity_mechanics_20260717T160231Z`.
  Its result, failure packet, resource summary, telemetry, and checksum manifest
  total 285,741 bytes. The probe took 35.83 seconds, peaked at 1,104 MiB GPU
  memory and 25% sampled GPU utilization, and reached 4,191,292 KiB host RSS.
  Low GPU occupancy is inherent to this minimal simulator-identity diagnostic;
  duplicating conditions solely to fill memory would change the comparison.
- The bounded next diagnostic keeps this strict failure intact. It will first
  isolate policy batch numerics by repeating one identical reset observation
  across batches 1, 2, 8, 32, 96, and 112, then compare actual initial actions
  and five-step trajectories for the matched environment conditions. Changing
  upstream evaluator semantics, accepting tolerant RGB identity, or selecting
  a deterministic-render workaround requires a separate recorded decision.

## Gate -1 policy batch and mechanism diagnosis

- The frozen action-layer recovery ran from clean commit `0c60c20` without
  relaxing the mechanics stop reason. For a byte-identical reset observation
  and RNG seed 20260717, the first SmolVLA action differed between policy batch
  1 and batch 2 in all seven dimensions: mean absolute delta 0.001014 and
  maximum 0.002254, outside the predeclared `atol=rtol=1e-6`. The batch ladder
  therefore stopped at 2 rather than spending compute on batches 8–112.
- Across the seven matched sync/async, batch-1/batch-2, and same-mode-repeat
  environment comparisons, no initial action was within tolerance. The maximum
  initial-action delta was 0.004378 and the maximum delta over five policy steps
  was 0.009512. The resulting observations diverged in both pixels and robot
  state, although all seven five-step reward/termination/truncation records
  remained exact. This short outcome agreement is not trajectory identity or
  policy-quality evidence.
- A follow-up mechanism probe used the same raw reset observation, explicitly
  matched first-sample flow-matching noise, and the pinned preprocessing/model
  path. The first sample was bitwise identical after raw replication,
  observation preprocessing, environment preprocessing, and policy
  preprocessing; the `50 x 32` noise sample was also bitwise identical. A
  repeated batch-1 forward was bitwise identical, while batch 1 versus batch 2
  still changed all seven action dimensions (mean absolute delta 0.001263,
  maximum 0.004668). The cause is therefore localized to SmolVLA model-forward
  numerics under a changed batch shape, not preprocessing, RNG/noise, or
  same-shape repeat nondeterminism.
- The pinned model contains 474 bfloat16 and 26 float32 parameter tensors; the
  official path enables TF32 matmul and cuDNN benchmarking. These facts make
  mixed-precision/kernel shape effects plausible, but the first responsible
  attention or ten-step denoising operation has not been proven and should not
  be guessed. Exact cross-shape equality is not an established property of this
  inference stack.
- Three scientifically distinct recoveries remain. The lowest semantic and
  engineering risk is to freeze one async batch/mode for every compared method,
  evaluate task-level repeated evidence with confidence intervals, and retain
  a small batch-1 audit; this changes Gate -1 from cross-batch bitwise identity
  to fixed-contract statistical/functional reproducibility without patching the
  upstream evaluator. A per-environment batch-1 policy wrapper changes evaluator
  semantics and sharply reduces throughput, while a deterministic-render or
  precision fork changes inference semantics/resources and may still not yield
  cross-shape equality. Selecting among them affects the paper contract and is
  intentionally left for an explicit decision.
- The canonical records are
  `.codex/longrun/gate_minus1_identity_policy_recovery_20260717_161316` and
  `.codex/longrun/gate_minus1_policy_batch_mechanism_recovery_20260717_162212`;
  their checksummed artifacts are under
  `$EMBER_OUTPUT_ROOT/gate_minus1/evaluation_identity_policy_recovery_20260717T161316Z`
  and
  `$EMBER_OUTPUT_ROOT/gate_minus1/policy_batch_mechanism_recovery_20260717T162212Z`.
  They total 716,617 and 69,355 bytes. The policy recovery used 2,631 MiB peak
  GPU memory and 76.10 seconds; the mechanism probe used 1,540 MiB and 18.90
  seconds. A first mechanism launch failed before model load because the
  one-off script skipped the existing LIBERO runtime binding; its long-run
  traceback is retained and the unchanged script succeeded after that binding
  was restored.
