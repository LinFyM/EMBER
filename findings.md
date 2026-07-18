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

## Evaluation recovery decision

- The selected recovery is fixed-contract statistical/functional
  reproducibility. Primary scientific evaluation keeps the upstream evaluator
  unchanged, uses asynchronous environments, and predeclares one measured-safe
  batch size for the episode budget before reading results. Every compared arm
  uses the same mode, batch, seeds, init-state mapping, hardware/math path, and
  interaction budget; results from different batch shapes are not pooled as if
  they came from the same policy realization.
- Task is the primary independent unit. Gate decisions use repeated functional
  estimates and confidence intervals rather than cross-batch bitwise equality.
  A small batch-1/sync audit retains exact seed/init/action diagnostics, while
  sparse one-level RGB differences and milliscale action deltas are recorded but
  do not block progress unless a matched test shows material functional bias.
- This decision favors efficient mechanism development while preserving the
  core held-out and matched-control contract. It does not erase the strict
  identity failures, lower a useful-update threshold, expose held information,
  or authorize comparing methods evaluated with different batch semantics.

## Canonical LIBERO-90 authority probe

- The pinned Hub tree contains exactly 90 `libero_90/*.hdf5` LFS objects and
  66,658,085,995 bytes. The resumable, zero-GPU download completed under
  `.codex/longrun/libero90_canonical_download_20260717_163546` at immutable
  revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`; its launch contract kept the
  personal storage surface below the 500GB cap and verified file count and total
  bytes on completion.
- The retained manifest implementation validates each local file against the
  Hub LFS byte count and SHA256, checks all 50 contiguous demos, required HDF5
  dtypes/shapes, frame totals, task-map/BDDL basename, camera pair, OSC_POSE
  controller, robot, and control frequency. It never reads or serializes the
  producer `model_file` XML or its embedded private paths.
- Numeric HDF5 values are permitted only for the 60 source tasks and episodes
  8–27 (`source_base_fit`). The derived eight-dimensional runtime state is
  `[ee_states(6), gripper_states(2)]`; the action is seven-dimensional. Mean,
  population standard deviation, min/max, and q01/q10/q50/q90/q99 are written
  to a separate normalization artifact. Validation and held HDF5 files receive
  schema/metadata and raw-file integrity audits only; no decoded numeric values
  are returned.
- A live three-surface probe passed on downloaded files. Source task 0 yielded
  1,525 normalization rows with state shape `[1525,8]` and action shape
  `[1525,7]`; validation task 3 and held task 1 both returned no numeric sample
  arrays. All three exposed the expected two cameras and `OSC_POSE` controller.
  Held BDDL files are hash-checked but not semantically parsed, so goal/object
  labels cannot enter the manifest or a later Writer path.
- A read-only authority pass confirms that the coarse validation scene, object,
  fixture, and goal-predicate sets represented in parsed BDDL are subsets of
  the source sets. This is not yet the required atom-multiplicity or
  verb/object/receptacle/relation/order split proof. Held semantic BDDL remains
  unparsed: the manifest now serializes semantic held coverage as `null` with
  `not_evaluated_due_to_access_policy`, rather than the misleading empty list.
  Scene coverage remains independently evaluable from the task-name prefix.
- Task-map and HDF5 instructions agree, but parsed BDDL wording differs on the
  non-held tasks 14, 84, and 85. These are semantic paraphrase/provenance notes,
  not filename or task-index mismatches: the manifest preserves both strings
  and does not remap tasks, alter the split, or treat the difference as a Gate
  failure. The producer `env_args` also names legacy `libero_100` paths while
  canonical BDDL basenames match; this is retained as a non-blocking provenance
  note rather than copied as a local path.
- The resumable download completed with long-run main rc 0 at exactly 90 HDF5
  files and 66,658,085,995 bytes, with no remaining `.incomplete` files. The
  first canonical full audit (`libero90_canonical_manifest_20260717_190204`)
  stopped after 12.01 seconds at task 14 because the producer's legacy
  `env_args.bddl_file` basename differed from the canonical name. It used zero
  GPUs, peaked at 834,120 KiB host RSS, generated no output artifact, and retains
  its traceback as a data-authority failure packet.
- A metadata-only scan found six such legacy basename differences: four source,
  one validation, and one held metadata record. In all 90 tasks, the canonical
  HDF5 `bddl_file_name` matches the pinned task map and official BDDL filename,
  and the HDF5 instruction matches the task map. Therefore the legacy producer
  path is non-authoritative provenance, not a remapping source. The bounded
  repair converts only this field to a visible
  `legacy_env_bddl_basename_mismatch` note; canonical HDF5 basename mismatch
  remains fatal. The real task-14 audit now passes with the expected two legacy
  provenance notes and 2,087 permitted source-normalization rows.
- The fresh recovery
  (`libero90_canonical_manifest_recovery1_20260717_190939`) completed from clean
  commit `d6cdac7` with main rc 0 and checksum follow-up rc 0. Eight CPU workers
  audited all 90 tasks in 12.45 seconds at 783% aggregate CPU, zero GPUs, and
  864,788 KiB peak host RSS. The canonical summary is 4,500 demonstrations,
  669,043 frames, 60 source tasks, 15 validation tasks, 15 held tasks, and the
  exact 66,658,085,995-byte HDF5 surface.
- Source-only episodes 8–27 contributed 166,475 finite eight-dimensional state
  rows and the same number of seven-dimensional action rows. Every source task
  records exactly those 20 normalization episodes; all 30 validation/held tasks
  are metadata-only with empty normalization indices, and held BDDL records
  contain only filename, hash, and access policy. Episode lengths cover 4,500
  demos, range from 58 to 373, have median 141, and sum to the recorded 669,043
  frames.
- The final quality status is `pass_with_documented_notes`, not an unqualified
  pass: all 90 producer paths name the legacy `libero_100` suite, six have a
  legacy basename difference, three non-held parsed BDDL instructions are
  paraphrases of the task-map/HDF5 instruction, and the Hub card license field
  conflicts with the recorded original-release license authority. None changes
  canonical task identity, split, or numeric access, and each remains visible
  in `quality_report.json`.
- The generated report is a dependency-free, filterable local HTML page linked
  to checksummed manifest, normalization, and quality JSON. It has an atomic
  `latest` symlink and contains no dataset payload, raw model XML, local paths,
  host identity, actions, or held labels. The canonical 1,307,643-byte artifact
  is `$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/recovery1_20260717T190800Z`,
  and the durable review page is
  `$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/latest/index.html`.

## Gate -1 official-overlap specification pilot

- The first launch attempt stopped in 0.13 seconds before policy, simulator, or
  GPU allocation because the pilot TOML's checkpoint role contained an extra
  `overlap_` token and failed exact identity against `phase0.toml`. This is an
  implementation/configuration failure, not benchmark evidence. Its long-run
  record `gate_minus1_specification_pilot_20260717_174551` and 1.6KB checksummed
  failure/telemetry packet are retained.
- The bounded repair changes only that role string to the already locked
  `official_mechanics_only_never_ember_shared_base` value and adds a regression
  test binding both role and Gate thresholds to the Phase 0 contract. No task,
  prompt arm, seed, init state, threshold, batch shape, or resource budget
  changed.
- The recovered launch then reached model load and async-vector construction but
  stopped before the first simulator reset: pinned LeRobot's
  `_LazyAsyncVectorEnv` forwards `call/get_attr` but not the underlying
  Gymnasium `set_attr`. The earlier sync-only unit probe therefore overstated
  prompt-override compatibility. This is a verified runtime adapter defect, not
  a prompt or benchmark result; the 6,867-byte failure/telemetry packet and
  long-run `gate_minus1_specification_pilot_recovery_20260717_175010` remain
  checksummed.
- During that launch window GPU 0 had become occupied by a separate four-GPU
  MemLLM job. The EMBER process failed and released without touching that job,
  but the launch shell should have rejected the device instead of relying on a
  human-readable telemetry line. The canonical wrapper now hard-fails before
  output/model load when any compute PID exists or preexisting memory is at
  least 1,000 MiB; a live negative test correctly rejected the occupied GPU.
- The bounded async repair calls the pinned lazy wrapper's `_ensure`, then uses
  the verified underlying Gymnasium `AsyncVectorEnv.set_attr`; it does not alter
  reset, step, success, or policy evaluation semantics. A regression double
  reproduces the missing-forwarder shape and now passes.
- The fresh recovery2 scientific command completed all 64 predeclared episodes
  on GPU 4. Correct prompts succeeded 6/8 on task 0 and 6/8 on task 1; no-spec
  succeeded 0/8 and 0/8; scene-only succeeded 2/8 and 0/8; swapped succeeded
  0/8 and 0/8. The paired task-level descriptive gaps were 75 percentage points
  for correct versus no-spec, 62.5 for correct versus scene-only, and 75 for
  correct versus swapped, with the predeclared bootstrap intervals retained in
  `probe_result.json`. Both correct arms exceeded the predeclared nonzero
  competence stop condition, so the result is a scale candidate rather than an
  early stop.
- This result authorizes only a prompt-path and benchmark-specification
  diagnostic for an overlap-trained official checkpoint. It is not a
  LIBERO-90 Gate -1 pass and says nothing about Writer utility, video utility,
  general policy quality, or same-init counterfactual goal switching; the
  environment goal stayed fixed while only the policy-visible prompt changed.
- The canonical evidence is
  `$EMBER_OUTPUT_ROOT/gate_minus1/specification/pilot_recovery2_20260717T180100Z`.
  `checksums.sha256` verifies both result JSON files, the gallery manifest and
  HTML, GPU telemetry, and all eight videos. The artifact is 1,166,960 bytes;
  the videos decode at 80 fps, and a temporary first/last-frame contact sheet
  visually confirmed consistent initial scenes and plausible rollout motion
  before being deleted. The durable local review surface is the atomic
  `$EMBER_OUTPUT_ROOT/gate_minus1/specification/latest/index.html` gallery.
- Measured wall time was 5:41.57. The single A100 peaked at 6,167 MiB with at
  least 74,988 MiB free; over telemetry samples with at least 1,000 MiB active,
  mean memory was 4,950.59 MiB, mean GPU utilization 14.89%, and peak
  utilization 100%. This deliberately small, one-batch-per-arm diagnostic was
  simulator/episode bounded; allocating dummy memory or changing its fixed
  batch after seeing results would not improve scientific throughput.
- GNU `time` records `Exit status: 0` after the evaluator prints
  `pilot_completed_scale_candidate`. Only afterward, the one-off outer
  telemetry `EXIT` trap emitted `pop_var_context: head of shell_variables not a
  function context`, causing the long-run shell to record rc=1. The long-run
  state is intentionally not rewritten: it is a wrapper-failure packet whose
  inner scientific command succeeded and whose complete post-run artifacts
  checksum correctly.
- The host's Bash 5.2.21 reproduces the exact signature with the documented
  `errexit` plus function/evaluation minimal case; GNU Bash maintainers describe
  this 5.2 function-context message as an `errexit` unwinding defect
  (https://lists.gnu.org/archive/html/bug-bash/2024-02/msg00092.html). Because
  the brittle construct lived only in the temporary long-run launch command,
  the canonical repository launcher is unchanged. Future launch templates save
  the main rc explicitly, finalize telemetry in the normal path, use only
  `INT`/`TERM` handlers for interruption, and exit with the saved rc. A shell
  smoke preserved rc 0, 7, and 124 for success, failure, and timeout while
  stopping the sampler in all three cases.

## Gate -1 role-aware split validity

- The coarse BDDL-category coverage was insufficient because objects can appear
  as scene distractors without ever being task-relevant source supervision. A
  conservative audit over only the permitted language specification therefore
  counts role-specific verbs, moved objects, target relations, and target
  receptacles at the task grain.
- The rejected original split has zero source tasks for `stack` (validation
  16/63, held 17/64), moving `tomato_sauce` (validation 49/59, held 54), target relation
  `under` (validation 42/89), target relation `front_of` (held 34), and target
  receptacle `wine_rack` (held 27). It has only one source task for moving
  `moka_pot` (source 38, validation 19), `wine_bottle` (source 26, held 27), and
  `white_bowl` (source 43, validation 36, held 37).
- These are exact task-specification matches, not held executable labels or
  policy results. They are sufficient to falsify the predeclared requirement
  that every validation/held atom occur in at least two source tasks. Gate -1
  original split validity therefore failed before any LIBERO-90 policy result;
  this preserved failure motivated the permanent reseal below.
- The durable technical report is `docs/benchmark_validity_report.md`. The
  recommended bounded recovery is a one-time specification-only split redesign
  and permanent reseal before LIBERO-90 training/evaluation. Preserving the
  original split instead would have required narrowing the source-covered
  compositional claim. Counting distractors, using held privileged labels, or
  changing factor semantics after policy results is disallowed.

## Gate -1 one-time specification-only split reseal

- Owner authorization selected the recommended bounded recovery before any
  LIBERO-90 policy training/evaluation outcome. The rejected split and its
  failure packet remain intact in `configs/libero90_split_reseal.json`,
  `docs/benchmark_validity_report.md`, and Git commit `5897406`; recovery does
  not relabel the original split as valid.
- The pinned task-map Git blob is the sole redesign authority. Its 90 ordered
  task names deterministically yield only `task_index`, scene identity, and
  English instruction. The code has no BDDL, HDF5, simulator, model, reward, or
  normalization input, and the sealed record declares an empty privileged-field
  access list. The allowed specification surface hash is
  `9ec40758b7b5c2a6c3c0aacb5e41c2a0bd30a21e702e9b0f1187c1adeeb8ea39`.
- `libero90_role_factors_v1` parses all 90 instructions exactly once and fails
  closed on unknown or ambiguous templates. It distinguishes verbs, moved
  grammatical patients, destination receptacles/relations, source and target
  selectors, actuated fixtures/subregions, and explicit operation order.
  Distractor presence is never role exposure; `pick_up > place` and
  `stack > place` remain ordered two-step compositions.
- The frozen search is
  `sha256_multistart_greedy_plus_steepest_swap_v1`, seed `20260718`, 16,384
  candidates, and a hard minimum of two source-task occurrences for every
  validation/held primitive role. Its predeclared priorities then maximize
  source-unseen full compositions and same-scene controls before scene/difficulty
  balance and prior-assignment retention.
- The permanent result has 41 evaluation primitive roles, zero coverage
  violations, and a minimum observed source count of two. All 30 evaluation
  tasks have an exact source-unseen full composition and a different same-scene
  source task; 28 also have a role-sharing same-scene hard negative. Two-step
  counts are 29/60 source, 7/15 validation, and 8/15 held. These are benchmark
  design mechanics, not policy competence or causal-use evidence.
- The canonical reseal record is 144KB with SHA256
  `9f5bc62e15e2cb07887e97bc98630a3f527ac6b5e253f41c203cf37459568428`.
  It regenerates byte-for-byte after installing the new active split because
  the old split is explicitly retained as `prior_split`. Gate -1 thresholds did
  not change, and the active split is now permanent.
- The previous canonical HDF5 audit remains valid as historical data-integrity
  evidence, but its normalization belongs to the rejected split. The fresh
  clean-commit audit `libero90_resealed_manifest_20260718_044358` replaces that
  active normalization: all 90 factor records and split labels match the seal;
  source episodes 8–27 contribute 183,555 finite state/action rows; all 30
  validation/held tasks remain metadata-only; evaluation numeric access is
  exactly zero; and held BDDL stays identity-hash-only.
- The fresh artifact at
  `$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/reseal1_20260718T044331Z`
  is 1,537,733 bytes with four valid checksums, parseable JSON, no private paths
  or producer model paths, and a dependency-free role-aware HTML view. It was
  generated from clean commit `23f3301` in 12.34 seconds using eight CPU workers,
  840,572 KiB peak RSS, and no GPU.
- Quality remains `pass_with_documented_notes`: all 90 files carry the legacy
  producer-suite note, six carry a legacy producer-basename note, and the two
  currently legal non-held BDDL parses at tasks 17 and 84 preserve wording
  differences while task-map/HDF5 instructions agree. Newly held BDDL semantics
  were not opened to reproduce prior wording notes.

## Gate -1 source same-state executable-goal mechanics

- The official-overlap `libero_spatial` suite cannot establish executable-goal
  switching: all ten tasks have the same native success predicate,
  `on(akita_black_bowl_1, plate_1)`, and differ only in initial spatial
  specification. Its prior prompt-swap result therefore remains prompt-path
  evidence and is not relabeled as a goal counterfactual.
- The smallest legal paired-goal surface is resealed LIBERO-90 source tasks 3
  and 4 in `KITCHEN_SCENE10`. Their declared object/fixture state schema is
  identical, while the native goals differ only in the relevant butter:
  task 3 requires `butter_2` in the top drawer and the drawer closed; task 4
  requires `butter_1` with the same receptacle/closure predicate.
- The checked-in probe freezes the seal and data hashes, task IDs, first eight
  init states, first eight source demonstrations per task, final recorded-state
  selector, exact flattened-MuJoCo-state identity, unmodified native BDDL
  evaluator, and the existing 0.80 counterfactual threshold. It reads source
  states only and serializes hashes and boolean goal matrices, not actions,
  model XML, raw states, local paths, or any validation/held numeric surface.
- Both environments produced the exact same model-layout hash. All 8 shared
  task-3 initial states reproduced byte-identical post-injection state in both
  evaluators with maximum absolute delta zero and no premature success. All 8
  task-3 terminal states evaluated `[true, false]`, and all 8 task-4 terminal
  states evaluated `[false, true]`; bidirectional specificity is therefore
  16/16 and the minimum direction fraction is 1.0.
- This is a mechanics pass for a feasible same-state executable-goal evaluator,
  not evidence that a learned policy follows a same-init instruction switch.
  It does not pass Gate -1, say anything about video utility, or authorize
  Writer training. The next causal probe must hold the observation/state fixed
  while measuring policy action or behavior under matched language conditions.
- The canonical run is
  `.codex/longrun/gate_minus1_source_same_init_goal_20260718_050511` from clean
  commit `25b1276`; main rc is 0, wall time 4.73 seconds, peak RSS 2,748,868
  KiB, and GPU count zero. The 21,482-byte checksummed JSON/HTML artifact is
  `$EMBER_OUTPUT_ROOT/gate_minus1/specification/source_same_init_goal_20260718T050511Z`,
  with the atomic review link `source_same_init_goal_latest/index.html`.
- A preceding longrun ending `050456` is retained as a launch-argument failure:
  shell expansion left `--output-dir` empty, the canonical launcher rejected it
  with rc 2 before runtime setup, data read, simulator creation, or GPU use. It
  is not scientific evidence and no output directory was created.

## Gate -1 same-observation language-to-action diagnostic

- The prior full-rollout prompt pilot held task IDs, seed/init mapping, batch
  shape, mode, and policy RNG fixed, but each prompt arm performed a separate
  simulator reset. Sparse one-level renderer variation meant a residual visual
  confound remained even though the success gaps were large.
- The follow-up freezes and reuses that exact pilot authority and its
  checksummed result. For each of tasks 0/1 it performs one async batch-8 reset
  at seeds 5100–5107, caches the resulting two-camera/state observation, and
  feeds that same object to correct, no-spec, scene-only, and swapped prompts.
  Policy RNG seed 20260718 and batch shape stay fixed; the policy queue is reset
  before each condition. A second correct plan is a deterministic control.
- Each condition uses one SmolVLA forward to generate the 50-step action chunk;
  the probe retains the first 10 postprocessed actions rather than recomputing
  the model ten times. A per-episode plan is called substantive only when its
  maximum absolute delta reaches 0.01, more than four times the previously
  observed 0.002254 cross-batch numerical artifact.
- Correct-repeat plans are exactly equal across all 16 samples (maximum delta
  zero). Correct versus swapped, no-spec, and scene-only plans are all
  substantive for 16/16 samples. Their overall maximum absolute deltas are
  0.452167, 0.342599, and 0.318189 respectively; even the smallest per-episode
  maxima are 0.230196, 0.098777, and 0.159854. All 16 first actions in every
  comparison also exceed 0.01.
- Combined with the linked prior 6/8 and 6/8 correct-arm competence and 0/8
  swapped arms, this demonstrates a same-observation language-to-action causal
  path on an overlap-trained checkpoint and removes reset-render variation as
  the explanation for prompt sensitivity. It still cannot show that behavior
  switches to the correct counterfactual goal because the overlap suite's
  native goal is unchanged. Gate -1 and Writer authorization remain false.
- Canonical evidence is
  `.codex/longrun/gate_minus1_language_action_20260718_052249` from clean commit
  `2038129`, plus
  `$EMBER_OUTPUT_ROOT/gate_minus1/specification/language_action_20260718T052249Z`.
  The run completed with rc 0 in 41.72 seconds, peak host RSS 3,980,224 KiB,
  Torch peak reserved memory 1,232 MiB on one A100, and a 268,038-byte
  checksummed JSON/HTML artifact. The GPU was released and no new rollout video
  was duplicated; the linked prior pilot gallery retains the matched videos.

## Gate -1 action-hidden video protocol frozen before outcomes

- The canonical LIBERO-90 source HDF5 files already contain the pinned
  third-person `obs/agentview_rgb` stream. State replay is unnecessary: the
  probe can extract action-hidden RGB directly from the same hashed authority,
  avoiding simulator drift and redundant rendering while never passing action,
  proprioception, reward, or terminal fields to the encoder.
- Resealed source tasks 3 and 4 are the smallest current same-scene content
  pair: both show the same `KITCHEN_SCENE10` layout and drawer interaction, but
  successful demonstrations move the back versus front butter. The support
  and query demonstrations are disjoint, and first-frame/static/last-frame
  controls explicitly test initial-scene and endpoint shortcuts.
- The frozen information probe is deliberately weaker than Writer evidence.
  It asks whether an RGB-only frozen representation plus a fixed source-trained
  linear readout can recover the task on independent source videos, and whether
  reversed or shuffled time degrades that readout. It neither generates policy
  parameters nor demonstrates zero-interaction utility, held generalization,
  or correct paired-goal behavior.
- Predeclared point thresholds are 0.80 ordered balanced accuracy, 0.80
  bidirectional query-pair correctness (without a same-init claim), 0.80
  wrong-video specificity, 20-point ordered gaps over
  first-frame and static-median controls, and a 10-point gap over both reversed
  and shuffled videos. A last frame reaching 95% of full accuracy narrows the
  interpretation to endpoint conditioning; drop-last-20% must retain 90% of
  full accuracy. Confidence intervals use 10,000 source-stratified bootstrap
  samples but do not replace the fixed point thresholds.
- A technical duplicate-input sweep selected one-GPU batch 48 because it was
  faster end-to-end than batches 32 and 64 while remaining far above the 10 GiB
  headroom floor. These measurements select execution efficiency only and
  contain no task-classification outcome.

### First canonical result: current causal-context representation fails

- The run is mechanically complete and leakage-safe: every checksum, JSON,
  cache shape/dtype/finite check, MP4 decode, source/model/split authority, and
  atomic link passes. The encoder was exactly repeatable at the same batch
  shape. This excludes implementation failure as an explanation for the
  scientific result.
- Ordered query balanced accuracy is 0.625 (95% bootstrap [0.50, 0.75]) versus
  the frozen 0.80 minimum. Only 9/24 bidirectional query pairs are both correct,
  and same-scene wrong-video specificity is 0.625. Ordered exceeds first-frame
  and static-median by only 12.5 points, not 20.
- Temporal and endpoint controls contradict a motion-use claim: reversed
  accuracy is 0.729 and shuffled is 0.708, both above ordered; last-frame is
  0.729, also above full video; drop-last-20% is 0.542. Consequently temporal
  order, endpoint independence, and action-hidden video content utility are not
  established on this reader.
- Readout-only underfitting is not the leading cause. Ordered support accuracy
  is 0.8125, query is 0.625, and nearest-centroid query accuracy is identically
  0.625. Class centroids in the final causal-context space have cosines
  0.999601 on support and 0.999775 on query, showing that this single pooled
  token largely collapses the subtle task difference. RGB insufficiency remains
  possible because the two identical-color butters are small and often
  occluded, but representative videos visibly differ in which object moves.
- The prioritized bounded remedy is therefore representation-specific: keep
  the same pinned model, cached source RGB, support/query rows, conditions,
  ridge readout, batch, metrics, and thresholds; extract per-frame frozen visual
  connector features and use one fixed temporal-moment descriptor. This is the
  least invasive test of whether temporal detail was lost by causal-context
  pooling. It is not a Writer, learned temporal encoder, or threshold repair.

### Fixed framewise recovery: temporal signal appears, content gate still fails

- The single predeclared recovery reused the exact RGB cache at SHA256
  `26c29bc69c2bd6ed633aef3ab3f9de3357ece1118c89bd17177cde2c316edfe8`
  and changed only the frozen feature to per-frame visual-connector temporal
  moments. The source tasks, 24/24 support/query rows, seven controls, ridge
  readout, model weights, batch 48, metrics, and thresholds are unchanged.
  Validation and held numeric access remain zero.
- Ordered balanced accuracy improves from 0.625 to 0.7917 (95% bootstrap
  [0.6667, 0.8958]), with symmetric 19/24 accuracy on each source task.
  Wrong-video specificity is also 0.7917. Both miss the frozen 0.80 thresholds,
  and only 15/24 demo-index pairs are jointly correct versus the 0.80 pair
  requirement. Nine pair indices have at least one error, including one where
  both task videos are wrong; errors are not a one-class collapse.
- The temporal diagnosis changes materially without changing the decision
  standard. Ordered exceeds first-frame by 0.2708, static median by 0.25, and
  the better of reversed/shuffled by 0.2292; last/full is 0.6053. Thus the
  frozen temporal-order criterion passes and the causal final-token collapse
  is repaired. Drop-last-20% retains only 0.8158 of full accuracy, below 0.90,
  so late trajectory evidence remains important on this pair.
- Mechanics are valid: all listed checksums pass, the 384x4800 float32 cache
  loads without pickle and is finite, exact repeated-batch delta is zero, and
  all fourteen reused videos retain their hashes and decode as 16 frames at
  128x128. The remaining failure is classified as source RGB
  information/content acquisition on this fixed reader and pair, not an
  implementation, leakage, temporal-pooling, or optimizer failure.
- No second reader, new task pair, threshold change, or held inspection is
  authorized after this outcome. The result establishes only a useful temporal
  representation diagnostic on one resealed source pair; it does not pass
  Gate -1, establish held-video sufficiency, or authorize Writer training.
