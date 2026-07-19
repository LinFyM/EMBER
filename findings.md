# EMBER Durable Findings

## Research framing

- The broad information-to-parameter idea is meaningful only over a structured
  task family with executable source-task bridge supervision. It is not a
  universal optimizer claim.
- A static task-specification-to-adapter Writer is best described as an
  amortized task-conditioned parameter-update generator or hypernetwork.
- The candidate novelty lies in the complete controlled combination, not in
  language-conditioned weights, action-free video, LoRA, or RL alone.
- Owner authority on 2026-07-18 confirms that current EMBER is direct
  language/action-hidden-video to complete task-specific LoRA, then ordinary
  task-local LoRA RL and source-only Writer reward/meta learning. Mandatory
  canonical bank/shared subspace/soft geometry/residual escape was a later
  assistant/expert addition and is superseded.
- The persistent Goal record cannot be text-edited by the available Goal API.
  It remains active for continuity, while its mandatory Gate-1/geometry wording
  is superseded by the dated owner contract; no fake Gate 1 result or replacement
  Goal is scientifically valid.

## Adopted scientific lessons

- Separate useful-update existence from amortized Writer acquisition. Gate 0
  proves the common target-layer/rank LoRA can contain useful updates; it does
  not require or authorize a second canonical representation Gate.
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
- Writer emits all factors for the full predeclared task-local LoRA space.
  Independent source-query functional loss is primary; raw factor MSE is
  diagnostic/auxiliary only.
- Ordinary task-local RL updates the same generated LoRA in place. Writer emits
  no bank, basis, mask, metric, radius, learning rate, or other search
  constraint.
- Reward learning has two distinct questions and must retain two distinct
  stages. Immediately after supervised Writer cold start, source-only
  Writer-only RL freezes the base and treats the generated LoRA only as the
  functional output through which rollout reward updates Writer; it neither
  optimizes that LoRA in place nor updates shared policy state. Later ordinary
  task-local LoRA RL freezes both base and Writer and updates only the generated
  or matched-zero LoRA. The former tests reward acquisition by the generator;
  the latter tests whether its initialization adapts faster or better.
- The shared base stays frozen during direct Writer and default source
  reward/meta learning. Inner adaptation updates task-local LoRA and the outer
  source objective updates Writer parameters.
- Historically, the deleted bank supplied a shared span, geometry
  scaled/preconditioned its coordinates, and residual escape could leave the
  span. Their removal is a scope decision, not a pending experiment. Shared
  base/shared LoRA source outer training is also a future separate matched
  ablation, not current completion evidence.

## Verified design risks

- A successful same-embodiment action-hidden robot video proves an
  information/supervision conversion mechanism, not lower data-collection cost
  or human-to-robot transfer.
- LIBERO task scenes, layouts, language templates, filenames, episode length,
  and normalization statistics can leak task identity.
- The proposed LIBERO-90 task split remains a hypothesis until a task-factor and
  initialization audit is generated from pinned files.
- A generated adapter can appear task-specific in parameter space while having
  no matched functional utility; all Writer claims require behavioral controls.
- Joint Writer/base optimization creates a moving parameter coordinate system.
  The current mainline avoids it by freezing the base; any shared-base update
  is a separate future ablation.

## Unknowns requiring evidence

- Whether language leaves measurable incremental information for video on the
  selected task subset.
- Whether the intended action-policy matrices admit useful, safe local updates.
- Whether legal language/action-hidden video can acquire the useful full-LoRA
  updates demonstrated by independent source oracles.
- Whether Writer initialization improves ordinary matched-budget local LoRA RL
  and whether source-only reward can improve future Writer initializations.
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

## Gate 0 pre-outcome mechanics and access findings

- The pinned base snapshot exposes the intended task-local mechanism without
  borrowing an OpenVLA layer name. The last action self-attention and
  VLM-to-expert cross-attention q/v weights have shapes `(960,720)`,
  `(320,720)`, `(960,720)`, and `(320,320)`. Rank 8 therefore gives exactly
  40,320 trainable scalars. PEFT 0.19.1 supports the frozen full target list and
  orthogonal initialization and is now part of the project lock.
- HDF5 demonstration init states and official `.pruned_init` rows are distinct
  authorities: zero of 50 rows match for either source task 3 or 4, with
  differences much larger than floating-point noise. Offline episode numbers
  must not be reused as simulator init-state indices. Locked offline loss and
  official fresh-rollout success are separate evidence surfaces.
- Replaying source task 3 demo 8 state 0 through the pinned simulator shows the
  HDF5 RGB is already aligned with the raw simulator observation. Identity
  orientation has correlation 0.973/0.909 for agent/wrist cameras, whereas a
  180-degree H/W flip has -0.578/0.087. Because the canonical LIBERO runtime
  performs exactly one H/W flip before SmolVLA, streaming training must also
  apply exactly one flip to HDF5 images.
- Gate -1 previously accessed RGB only for demonstrations 40--47. It did not
  access their actions, rewards, terminals, or policy outcomes. Consequently
  46--47 are modality-scoped action/reward/policy-outcome locks, while 48--49
  alone are fully pristine. The Gate 0 source contract records that distinction
  before any policy result.
- A real task-3 support batch passes the pinned SmolVLA preprocessing path with
  two finite float32 `128x128` cameras, finite 8D normalized state, finite
  `50x7` normalized action chunks, boolean padding, and `48` language tokens.
  The streaming path repeats the terminal action under the padding mask to
  match LeRobot dataset semantics and does not materialize a duplicate video
  dataset.
- The first tracked adapter mechanics invocation failed before any policy
  forward/backward because the SmolVLA preprocessor intentionally removes
  complementary `task_id/demo_index/frame_index` fields before the fixed-noise
  row key was constructed. Capturing those keys from the raw batch is the
  narrow implementation fix; the failed long-run record ending `072633` is
  retained and has no scientific outcome.
- The recovered rank-8 probe resolved exactly the four declared targets and
  40,320 trainable scalars, produced finite gradients and updates, and round-
  tripped every physical delta bit-exactly. It also falsified one mechanics
  assumption: PEFT 0.19.1 `orthogonal` initialization produced nonzero physical
  delta norms up to `0.025996` and changed the fixed support-batch loss by
  `8.08e-5`. This contradicts the predeclared functional-zero wording even
  though it is small, so it is not accepted as the canonical initializer.
- The unique repair, chosen from the stated no-op requirement rather than the
  observed loss sign or magnitude, is PEFT's default seeded nonzero-`A` and
  zero-`B` initialization. It is now contractually required to give exactly
  zero physical deltas and exactly the frozen-base fixed loss. The amendment
  occurred after only one fixed task-3 support mechanics batch and before any
  multi-step training, query/report access, or rollout result; the prior
  artifact remains provenance rather than being overwritten.
- The clean-commit exact-zero recovery validates the repair on the real SmolVLA
  target set: all four initial physical-delta norms and maxima are exactly zero;
  enabled and disabled adapter losses are both `2.4018678665161133`; and one
  AdamW step creates finite nonzero updates in all four matrices. The saved
  adapter reloads every physical delta bit-exactly. Both scientific and wrapper
  return codes are zero, artifact checksums pass, Trackio contains the run, peak
  sampled device memory is 2,331 MiB with 78,823 MiB minimum free, and GPU 4 is
  released. This closes mechanics only.
- The shared-base one-step mechanics path independently trains 99,880,992
  declared parameters with finite loss and gradient from the same task-3
  support batch. It completed from clean commit `51b9405`, with outer/scientific
  rc 0, 2,345 MiB peak sampled device memory, and full GPU release. This is
  implementation evidence only, not base competence or Gate 0 evidence.
- The first one-load all-source batch calibration completed all four effective-
  batch-64 candidates without OOM. Its resource measurements remain valid:
  microbatches 8/16/32/64 achieved 49.03/76.25/86.30/92.19 samples/s, and the
  largest retained 61,712 MiB free with 19,441 MiB sampled device peak. It used
  one A100, four persistent HDF5 workers, recorded no loss/policy outcome,
  finished in 71.44 seconds, passed checksums/Trackio, and released the GPU.
- An independent code-path audit found the four candidates were not strict
  matched controls: they successively inherited updated model and AdamW state,
  while the sampler seed and row key depended on the microbatch/accumulation
  partition. This is an optimization/data-authority defect, not a scientific
  Gate result. The resource telemetry is preserved, but the microbatch-64
  selection is superseded and training authorization is false.
- The unique bounded recovery is predeclared before source training: restore
  one identical trainable snapshot, reset the same RNG, create a fresh empty
  AdamW per candidate, draw by absolute optimizer step/effective-batch slot,
  and derive fixed flow noise/time from matching row keys. Candidate row-key
  digests must compare equal before the fastest headroom-safe shape can become
  authority. No threshold, task surface, effective batch, or policy outcome is
  changed.
- The matched recovery from clean commit `e3a653a` satisfies that repair. All
  four candidates repeat the exact three row-key SHA256 values, share fixed
  flow seed 2026071806, and start from the restored trainable snapshot plus a
  fresh AdamW. Microbatches 8/16/32/64 measure 48.79/77.42/88.89/92.17
  samples/s; microbatch 64 / accumulation 1 wins again, keeps 61,750 MiB free,
  and has 19,403 MiB sampled peak device memory. Main/wrapper rc is zero,
  checksums and `latest` pass, Trackio is present, and GPU 4 is fully released.
  This freezes a resource shape only, not source competence, Gate 0, or Writer.
- Optimizer inspection shows a mixed parameter/state reality hidden by the old
  shorthand: 96,607,440 trainable values and both moment tensors are bf16;
  3,273,552 values and both moments are fp32; 155 scalar step tensors are also
  fp32. The precision label is corrected before any formal fit. This means the
  eventual full-state resume probe must compare these native mixed-dtype AdamW
  tensors exactly, rather than assuming or converting them to all-fp32 state.
- The exact stochastic resume probe passes from clean commit `a4689b7`. The
  uninterrupted two-step branch and the step-1 checkpoint/resume branch have
  identical full-model, mixed-dtype AdamW, upstream scheduler, global RNG,
  next-full-raw-batch, and next-row-key SHA256 values. Checkpoint saving itself
  leaves RNG unchanged. This closes the resume-mechanics claim without fixed
  flow inputs or policy outcomes.
- Its atomic checkpoint contains 12 retained payload/manifest files totaling
  1,319,431,002 bytes and has manifest SHA256
  `d06ad8907b6abf707fe30028fe74e5745ef3b29734e14816bf1b56dfdb5a528a`.
  Whole-tree validation passed before cleanup; the final checksummed result is
  only 48KB at SHA256
  `bc7a17cd3ddb0b8c3f6daf5f529b0357ff65fa426c85d90d90b6592ecbe5d3ed`.
  Peak device memory was 19,467 MiB, minimum free 61,687 MiB, main rc was zero,
  Trackio is complete, and GPU 4 was released. This authorizes only the formal
  source-base fit, not source competence, Gate 0, or Writer.
- The durable logging repair changed the trainer file hash after that first
  probe, so the implementation-binding rule required one revalidation before
  formal launch. Clean commit `d71c9ce` again gives identical branch records;
  all scientific state/data digests exactly equal both each other and the first
  probe. The active result SHA256 is
  `fab1eb111b5b2edf32d9103a51ff0e4ec6783ead1a1d2a09c372ca4a6e3ceab1`,
  its checkpoint manifest is
  `21afdce35869dcaa5ec944d01e4f8c5db7472adb2d58330222399d5936cc79ff`,
  and the old exact pass remains provenance rather than execution authority.

## Multi-GPU efficiency amendment frozen before topology outcomes

- The clean 10k source-base run at commit `8ff06f2` completed untouched with
  main rc 0 as the one-GPU reference and recovery record. Its final result SHA256
  is `0db5485707711657ecaad2806019c0d28d3a2ec9b94973a5c4aa7b327dc2a1b2`;
  the schema-2 step-10000 manifest SHA256 is
  `ca0c83abd8d4b46cf59e8f0a01bd267f7f0e019d3e2bfea8c8baeb2e851d4d00`
  with role `source_base_candidate_pending_competence`. The 7036.82-second run
  averaged 96.31% GPU utilization while active, peaked at 19,311 MiB device
  memory, retained validated recovery steps 8000/9000 plus final step 10000,
  reached Trackio step 10000, and released the GPU. This is base acquisition
  and systems evidence only; no rollout outcome has been read. Stability alone
  does not establish an efficient long-term topology. No topology probe or
  source-policy outcome was read when the 1/2/4-GPU contract was written.
- The sole trainer now accepts world sizes 1, 2, or 4 through the same
  `torchrun` entrypoint. Every topology keeps global effective batch 64, one
  optimizer step per 64 unique absolute slots, the same total four loader
  workers, rank-0-native global flow draws followed by contiguous scatter, and
  DDP mean aggregation over equal local batches. Rank 0 alone publishes
  Trackio and the atomic checkpoint; schema 3 binds topology and stores every
  rank's RNG, while legacy schema 2 remains readable only for the uninterrupted
  one-GPU reference.
- CPU/gloo integration tests prove that 2- and 4-rank global slots are neither
  duplicated nor omitted, global flow inputs reconstruct the single-rank draw
  bit-exactly, one-step DDP gradients match the single-global-batch reference
  within the declared numerical tolerance, and same-topology interruption/
  resume reproduces model, optimizer, scheduler, RNG, and next-batch state.
  Injected checkpoint failures publish neither a directory, sidecar, nor
  `last` link, including injected failures after the checkpoint directory has
  already been renamed: the new directory and sidecar are removed and the
  previous `last` target is restored. Topology selection additionally rejects
  telemetry whose bytes no longer match the run-finalized checksum, so the
  memory-headroom evidence is provenance-bound. The complete repository suite
  is 161 passing tests before live throughput probes.
- Cross-topology comparison binds the actual rank-0 global noise/time tensor
  digest plus the exact initial model, optimizer, and scheduler digests. It does
  not compare an ambient serialized CUDA-RNG bundle whose shape changes merely
  because more devices are visible. Shared base weights and all source HDF5
  hashes are verified once per process and then reused by the uninterrupted,
  checkpoint, and resumed branches; every loader still enforces manifest paths,
  byte sizes, schema, and demo allowlists.
- The live selection remains explicitly pending. The fixed probe is five warmup
  plus 25 measured steps, one step-30 full checkpoint, and one exact resumed
  step. It chooses the fastest headroom-safe topology only when speedup is at
  least 1.20 and parallel efficiency at least 0.55; a result within 2% prefers
  fewer GPUs. If DDP scaling fails those rules, the recorded fallback is not a
  second trainer: subsequent independent baseline/arm/task/seed jobs occupy up
  to four GPUs under the same per-job contract.
- Source competence is separately frozen before the final checkpoint exists:
  resealed source tasks 3/4, correct and same-scene swapped prompts, official
  init indices 8--15, async batch 8, and one video per arm. Four ranks assign
  the four arms exactly once and rank 0 builds the checksummed local gallery.
  Passing only permits task-local oracle fitting; failure permits exactly one
  otherwise identical extension to 20k. No validation/held numeric surface and
  no Gate/Writer authorization is exposed.
- Architecture ownership is intentionally nonparallel despite the larger test
  surface: `gate_zero_base_train.py` is the 226-line sole CLI and formal-fit
  owner; `gate_zero_base_session.py` owns shared model/data/optimizer/RNG and
  result/checkpoint session state; `gate_zero_distributed.py` owns global-slot,
  collective, flow-input, and topology-selection invariants;
  `gate_zero_base_probe.py` owns only resume/throughput probes;
  `gate_zero_base_competence.py` owns only the source rollout prerequisite; and
  `gate_zero_topology_report.py` owns the local selection report. There is no
  versioned trainer or fallback executable. The architecture guard has no hard
  violation; the three 600--800-line review modules stay cohesive because
  splitting their fail-closed contracts from their sole runtime would add
  cross-owner state without reuse. Retire topology-probe/report launch surfaces
  after selection and evidence export, and retire the competence launcher after
  Gate 0 evidence freezes; retain invariant tests and the selected trainer path.

## Four-rank resume failure diagnosed before topology selection

- Under pre-amendment topology contract SHA256
  `e334d4e8c8a3e6f904b0798177da3c82d2366c0dd80c13b7a4b1401643ecd68a`,
  matched world-size 1 and 2 probes completed at 93.353 and 170.822 global
  samples/s. Both had exact same-topology resume, complete global slots and
  flow authority, valid telemetry/checksums, safe headroom, and cleaned their
  transient checkpoints. These are retained engineering evidence but cannot
  participate in the final selection after the execution contract changes.
- The world-size-4 command completed all 30 fixed steps and atomically wrote a
  valid schema-3 checkpoint, then failed closed because the continuous step-31
  model hash differed from a newly constructed world-size-4 resume. It was not
  an OOM, data, sampler, flow, optimizer, scheduler, or checkpoint-corruption
  failure: two independent fresh resumes from that same checkpoint matched on
  every state surface with zero differing tensors. The installed PyTorch DDP
  implementation explicitly rebuilds gradient buckets after its first
  iteration, so the continuous reducer and a fresh resumed reducer used
  different bucket lifecycles; world size 4 exposed the resulting bitwise NCCL
  reduction-layout difference.
- The bounded mechanical repair declares the already fixed SmolVLA computation
  graph as DDP `static_graph` and binds that choice into topology and checkpoint
  manifests. It changes no data, sample, flow, model, optimizer, learning-rate,
  or global-batch budget. The failed run remains a checksummed 96KB failure
  packet plus telemetry and durable log; its validated 1.32GB transient
  checkpoint was removed after the compact fresh-resume diagnostic froze.
  Final topology selection now requires a live four-rank resume pass and fresh
  matched 1/2/4 probes under the amended contract.

## Static-graph four-rank RNG-boundary diagnosis

- The clean live static-graph resume run
  `gate_zero_world4_static_resume_20260718_112741` used contract SHA256
  `84c5bcf702dea4c322c282b45394c5d00bab15db3178386f17c6ffae698c0896`.
  Static graph removed the prior model-state failure: completed step, every
  model tensor, AdamW, scheduler, next raw batch, and next row keys were exact.
  The run still failed closed at `rng_state_sha256`; peak memory was 12,162 MiB
  with 68,993 MiB minimum free, so this was not OOM or resource pressure.
- A four-GPU read-only diagnostic reused the validated schema-3 step-1
  checkpoint and compared every serialized RNG component. Checkpoint RNG and
  resumed-before-step RNG matched on all four ranks. After step 2, NumPy, CPU
  Torch, and CUDA RNG were exact; only Python `random` differed. Flow tensors,
  loss, gradient norm, model, optimizer, scheduler, and data authority remained
  exact, with zero differing model tensors. The compact result SHA256 is
  `9d0bf693891d7549bf40d586f9e89a12f622ad927994da0bbb9d1d4a6081bac7`.
- The mechanism is a misplaced stochastic boundary: the fresh seed was set
  before one-time model/DDP/loader/authority setup. The first fresh runtime
  consumed Python RNG in that setup path, while a later checkpoint branch began
  after the process-level setup and retained the seed state. This is ambient
  setup contamination, not training/checkpoint divergence.
- The bounded fix retains deterministic seeding before model construction and
  reseeds a fresh training runtime only after DDP, loader iterator, authority,
  and setup complete. A resume restores its checkpoint RNG at the same boundary.
  A red-then-green regression asserts the ordering; no sample, flow, batch,
  optimizer, model, or threshold contract changes. Live four-rank exact-resume
  and fresh matched 1/2/4 probes remain required. The amended topology contract
  was frozen before those outcomes at SHA256
  `04bf00a4326f62119b32ca22ef9836980d5743e61eb2f1366e85ae4feae25e9d`.
- The failed directory retains a checksummed 96KB failure packet and telemetry.
  After whole-tree validation and the read-only diagnostic, its 1.32GB transient
  checkpoint was removed. The failed long-run state remains unchanged.
- The corrected research-scope and RNG-boundary milestone passes all 165
  repository tests, Python compilation, shell syntax, and diff-whitespace
  checks. A scope regression fixes the historical expert-plan SHA, rejects
  active Gate-1/geometry milestones, and documents the sealed phase-0 legacy
  field as non-executable provenance.

## Full-lifecycle direct-LoRA causal contract

- The current project uses LoRA only. Bottleneck adapters, IA3, prefix tuning,
  a shared base adapter/shared LoRA, and any other parallel trainable policy
  state are excluded. Gate 0's independently fitted task-local LoRA establishes
  useful-update existence and a behavioral upper bound; its factors are not a
  unique teacher that Writer must imitate.
- Because LoRA can itself be the limiting mechanism, Gate 0/source-validation
  includes a bounded capacity audit. Zero/base versus independent task-local
  LoRA is the matched comparison. Action-expert partial/full updates or full
  fine-tuning are explicitly non-matched capability upper bounds. If only an
  upper bound succeeds, the target/rank contract is too narrow and enters Gate
  recovery; Writer acquisition is not blamed for missing capacity.
- Writer consumes language plus action-hidden teaching video and emits the full
  task-specific LoRA under the fixed target/rank contract. Its primary signal is
  independent-query action/flow/behavioral loss through functional adapter
  application. Raw factor MSE cannot be primary and oracle-delta imitation is
  optional auxiliary supervision only.
- The ordinary-RL causal test is fixed as A) zero-LoRA initialization plus
  ordinary RL, B) cold-start Writer LoRA initialization plus identical RL, and
  C) reward-outer-trained Writer LoRA initialization plus identical RL. Targets,
  rank/count, optimizer/algorithm, hyperparameters, seeds, reward, and
  interactions match. Primary reports include J0, AUC, time-to-threshold, J_K,
  and J_K-J0; claims about learning dynamics require a
  matched-initial-performance or equivalent control.
- The default source outer stage keeps the shared base frozen, updates only the
  inner task-local LoRA on the model side, and updates Writer with the source
  outer objective. Held evaluation freezes base, Writer, encoders, and all
  shared state; only the predeclared task-local LoRA may use held reward.
  Accordingly the supported claim is an improved task-conditioned LoRA
  initialization without restricting later ordinary RL, not a learned
  optimizer, direction, geometry, or subspace.
- The active Goal spans the entire mechanism through frozen-held comparisons
  against zero/base, average, retrieval, capacity-matched language-only direct
  generation, causal language/video controls, cold-versus-outer Writer, and
  OpenVLA-OFT scale confirmation. No environment, resume, throughput, Gate -1,
  Gate 0, one-run, or authorization milestone can complete it.

## World-size-4 recovery is operationally sufficient

- The amended clean world-size-4 resume run at commit `9a8a8f5` completed rc 0.
  Result SHA256
  `e996dd9f5290bbb8302c1a051d81783a3353ad433d15530742ae436f6ef60a12`
  records exact completed step, model, optimizer, scheduler, RNG, next raw batch,
  and next row-key identity under topology contract SHA256
  `04bf00a4326f62119b32ca22ef9836980d5743e61eb2f1366e85ae4feae25e9d`.
  The checkpoint was schema 3/world size 4, validated before its 1.32GB payload
  was cleaned, and the resumed step completed without functional divergence.
- GPUs 4--7 peaked at 12,162 MiB with at least 68,993 MiB free; measured probe
  time was 93.19 seconds. This is safe but far below the desired steady-state
  memory envelope, so later useful batch/task/seed parallelism should consume
  the available capacity rather than reserve it idle.
- This closes the bounded resume investigation. Exact or ambient RNG digests,
  bitwise telemetry/log identity, and new identity surfaces cannot block a
  scientific Gate absent evidence that they alter checkpoint recovery, sampled
  data, closed-loop success, matched fairness, or held isolation. The full
  planned 1/2/4 curve is superseded by at most one necessary short four-GPU
  throughput/stability window before source competence and Gate 0.

## Four-card science path selected and source competence passed

- The single permitted world-size-4 short window completed its 30 training
  steps at about 294 global samples/s with 12,368 MiB maximum sampled device
  memory, 68,787 MiB minimum free, and 100% peak utilization. Its schema-3
  step-30 checkpoint validated at manifest SHA256
  `bb99f2c41e89c32709ea724d2c523d35d20a11c9f666b6a477d1c38b8e0e89c4`,
  loaded on four ranks, and executed the resumed step. The probe then failed
  only its stronger continuous-versus-resumed `model_state_sha256` equality.
- Under the owner stop rule this is not another diagnosis program. The 1.32GB
  validated transient checkpoint was removed, leaving an 88KB telemetry/log
  failure packet. World-size-4 DDP is not selected for resumable long training;
  subsequent work fills four GPUs with independent arm/task/seed jobs unless a
  future scientific workload separately demonstrates that one DDP job is the
  efficient safe choice.
- Canonical source competence long-run
  `gate_zero_source_competence_base10k_20260718_122116` completed rc 0 from clean
  commit `e386925`. On the permanently frozen source tasks 3/4 and official init
  indices 8--15, correct prompts achieved 8/8 and 5/8 successes; the matched
  same-scene swapped prompts achieved 0/8 and 0/8. All four arms are
  mechanics-valid. The task-level correct-minus-swapped gap is 81.25 percentage
  points with the predeclared bootstrap 95% interval [50, 100].
- Result SHA256
  `c9697c4cf71d452c431424be4cd12fd6a869ac4fd58755d07152ee6928da83cc`
  binds the step-10000 checkpoint manifest, four arms, seeds/init states,
  metrics, resources, and interpretation. All result/gallery/telemetry/video
  checksums pass; the run used world size 4 arm parallelism, took 83.71 seconds
  for the evaluation body, peaked at 17,546 MiB, and released every GPU.
- This is legal source-base competence and a language/specification causal
  prerequisite. It authorizes fitting the frozen task-local LoRA oracle. It is
  not a Gate -1 pass, Gate 0 pass, Writer authorization, video-utility result,
  or validation/held evidence; the action-hidden-video content threshold miss
  remains preserved for Gate -1 interpretation.

## Gate 0 pilot support is not the final Writer support

- Owner authority now explicitly classifies the current last-two
  action-expert q/v rank-8 LoRA as a low-cost Gate 0 pilot. Its 40,320
  parameters can establish useful-update existence cheaply, but ease of
  generation is not evidence that this is sufficient final Writer capacity.
- SmolVLA v0.6.0's own default PEFT target regex spans q/v projections across
  every action-expert layer and the state, action-input/output, and action-time
  projections. OpenVLA and OpenVLA-OFT use rank-32 `all-linear` LoRA in their
  official fine-tuning scripts. These are evidence that mature VLA PEFT support
  is commonly broader than a last-layer probe, not mandates to copy one scope.
- Before Writer acquisition, one bounded source/validation-only audit will
  compare the current pilot, all action-expert q/v, and near-official SmolVLA
  support, with rank adjustment only if needed. Selection is the minimum
  support with robust closed-loop oracle utility; held numeric access stays
  zero and exact targets/rank/alpha/dropout/count are then permanently resealed.
- Wider direct generation is an architecture problem. SHINE's layer-aware
  multi-state generation and Doc-to-LoRA's per-layer/Perceiver-style output
  organization motivate structured layer/module embeddings, chunked generation,
  or type-specific heads. They do not justify a shared update bank, subspace,
  geometry, or any constraint on subsequent ordinary LoRA RL.

## Gate 0 fit selection exposes support-to-query generalization failure

- All four source-only fit commands completed rc 0 from clean commit
  `96cd0f9`; every result, selected state, candidate record, and telemetry hash
  validates. No locked-report, validation, or held numeric surface was opened.
- Task 3's last-two q/v rank-8 LoRA reduced fixed query flow MSE by 3.27% at
  step 250, but its action-drift proxy was 0.02646, above the frozen 0.02 cap.
  Later candidates had larger drift and worse query loss, so the immutable
  query-only rule selected exact step 0. Task 4 selected step 250: query flow
  MSE fell only 1.12%, with drift 0.01995 just inside the cap; later candidates
  again worsened query loss and increased drift.
- The non-matched 99,880,992-parameter partial-update diagnostics both selected
  step 0. Although support flow losses fell substantially, step-250 query loss
  was already 54.46% worse on task 3 and 54.43% worse on task 4, then degraded
  further. The larger trainable set therefore did not establish a capacity
  upper bound under this acquisition schedule; it exposed support-to-query
  overfitting/optimization instead.
- These query results make the frozen two-task primary pass impossible without
  changing selections: task 3's own adapter is the exact base and the decision
  requires both tasks to have strictly positive closed-loop gain. That is not
  permission to skip or reinterpret the locked report. The report remains
  necessary to validate matched closed-loop mechanics, measure task 4, and
  complete the predeclared failure packet before choosing one bounded recovery.
- Current evidence does not isolate target support as the sole cause. The next
  recovery decision must prioritize acquisition/generalization diagnostics
  (learning rate, early budget/candidate resolution, support/query behavior)
  alongside the already required bounded target-support audit. It must not
  lower the drift/utility thresholds, use report outcomes for model selection,
  or blame the future Writer.

## Frozen Gate 0 report finds one-task utility but fails the two-task claim

- Immutable selection grant SHA256
  `313ecf738b1a69ef2934c33e0681d3cef5f83506cc28925f06b8e9e16239bfad`
  opened only the predeclared source report surface after all four selected
  hashes froze. Final long-run
  `gate_zero_oracle_locked_report_final_20260718_140322` completed main rc 0;
  every HDF5 row, seed, init state 16--23, prompt override, reset transition,
  selected state, result, video, gallery, and telemetry checksum validates.
- Task 3 selected a physical zero update and behaves accordingly: frozen base,
  own LoRA, and partial upper bound each succeed on the same 5/8 episodes. Its
  locked flow reduction is effectively zero. The task-4 LoRA selected at step
  250 raises success from 1/8 to 3/8 (+25pp) and lowers locked flow MSE by only
  0.82%; the task-3 zero adapter and partial-zero state each remain at 1/8.
- Across the two task-primary units, median success gain is 12.5pp versus the
  frozen 15pp minimum, only 1/2 tasks is positive versus the required 2/2, and
  median locked-flow reduction is 0.41% versus 20%. The paired own-minus-base
  gap is 12.5pp with bootstrap 95% CI [0, 37.5]. Median selection drift 0.00998
  passes the 0.02 aggregate cap, although task 4's locked-report action-drift
  diagnostic is 0.02179. The result status is therefore
  `gate_zero_pilot_failed`, failure class
  `task_local_lora_oracle_utility_not_established`; Gate 0, Writer, and final
  target support remain false.
- The task-4 signal is useful evidence, not a pass: correct-task initialization
  can improve closed-loop behavior, while applying that adapter to task 3
  worsens locked flow by 2.80% and success from 5/8 to 4/8. But the two-task CI
  includes zero and the effect is neither robust across tasks nor large in the
  independent behavioral loss.
- Both non-matched partial arms equal base because query-only selection rejected
  every trained candidate. This run therefore cannot say that a larger update
  lacks capacity; it says the current support-fit acquisition schedule failed
  to generalize. The next bounded recovery should jointly stabilize early
  acquisition and perform the owner-required support audit (last-two q/v, all
  action-expert q/v, near-official SmolVLA support), choosing only on legal
  source/query surfaces and evaluating on fresh reserved recovery init states.
  Threshold reduction, locked-report reuse for selection, and Writer launch are
  prohibited.

## Target-support recovery is bounded before outcomes

- The current long-term Goal's LoRA capacity audit is instantiated by
  `configs/gate_zero_target_support_audit.toml`; it does not change the Goal or
  revive any bank/geometry route. The old last-two q/v rank-8 support remains a
  40,320-parameter pilot candidate, compared fairly with all action-expert q/v
  (322,560 parameters) and the SmolVLA-default-like action support (371,328
  parameters). Frozen-checkpoint tensor inspection validates all 4/32/37 exact
  targets and declared counts.
- The fit remedy is intentionally singular: lower LoRA AdamW learning rate from
  `3e-4` to `1e-4` and inspect denser early candidates through step 750. The
  sampler, effective batch, support/query rows, noise, drift cap, base, rank,
  alpha, and dropout remain matched. This targets the observed task-3 early
  gain-above-drift-cap and task-4 post-step-250 overfit without an unbounded
  optimizer search.
- Locked demos cannot select support. Rank-8 screening uses reserved source
  init states 24--31; a selected support freezes before confirmation on 32--39
  and before reopening the locked reporting surface. One rank-16 version of the
  best declared support is conditionally available only if every rank-8 scope
  fails the frozen screening contract. There is no other support/rank search,
  and Writer remains unauthorized until the matched confirmation passes.

## Lower-LR rank-8 support fits generalize positively but remain below Gate scale

- All six 750-step fits completed rc 0 with matched query identity and valid
  checksums. Last-two q/v selects step 250 on both tasks and reduces fixed-query
  flow loss by 2.89%/1.83% with drift 0.01445/0.00991. All action-expert q/v
  selects steps 100/150 and reduces 5.80%/3.54% with drift
  0.01747/0.01967. Official-default-like support selects step 100 on both and
  reduces 6.07%/3.51% with drift 0.01871/0.01654.
- The one lower-LR/dense-early repair fixes the previous task-3 zero-selection
  problem: every support now yields a nonzero, drift-safe improvement on both
  independent query tasks. Broad action support roughly doubles the median
  reduction from 2.36% to 4.67--4.79%, while official-default-like exceeds
  all-q/v by only 0.12 percentage points at 48,768 extra LoRA parameters. This
  is evidence that last-two was not a sufficient final assumption, but it is
  not yet evidence that the largest support is necessary.
- All rank-8 median reductions remain far below the unchanged 20% Gate action-
  loss criterion. The predeclared closed-loop screen is still required because
  functional success can diverge from flow-loss magnitude and is needed for a
  truthful failure packet; it cannot erase the query shortfall or authorize
  Writer. The immutable six-state grant keeps locked report, rank 16, final
  target sealing, Gate 0, validation/held, and Writer closed.

## First rank-8 closed-loop screen hit the wrong init-state stride

- Long-run `gate0_support_rank8_screen_20260718_150220` completed all eight
  arms/64 episodes, then failed publication because every arm had
  `mechanics_valid=false`. The failure is implementation/reset authority, not
  a support outcome: the shared report helper had exactly one warm-up reset,
  which reaches init states 16--23, while the frozen screening grant requires
  24--31. The stdout success totals and videos therefore re-access the prior
  source report init-state surface and are prohibited from support selection.
  No decision/result/latest link was published; rank 16, Gate 0, final support,
  Writer, validation, and held access remain unauthorized.
- The minimal correction generalizes the existing single evaluator rather than
  adding another path. It derives how many stride-8 resets are needed from the
  frozen target IDs and assigns deterministic preceding warm-up seed batches.
  The original report contract still plans one warm-up for 16--23; screening
  plans two warm-ups (5484--5491 and 5492--5499) before evaluation seeds
  5500--5507 reach 24--31. Unit evidence rejects the former two-reset event
  sequence for this recovery surface. No threshold, state, fit, split, or
  scientific budget changed.

## Rank-8 support breadth does not establish robust oracle utility

- The corrected source-only screen completed 64 episodes with rc 0, exact
  init-state transitions to 24--31, all eight mechanics checks true, eight
  checksummed videos, and no validation/held or locked-report selection access.
  Task 3 frozen base is 4/8; last-two, all-q/v, and official-default-like are
  3/8, 2/8, and 3/8. Task 4 frozen base is 3/8; the same supports are 3/8,
  4/8, and 4/8. Thus last-two has median closed-loop gain -6.25pp with 0/2
  positive tasks, all-q/v -6.25pp with 1/2, and official-default-like 0pp with
  1/2. All fail the unchanged +15pp median and 2/2-positive requirements.
- Broader support improved independent fixed-query loss relative to last-two,
  but did not transfer robustly to closed loop: the median reductions are
  2.36%, 4.67%, and 4.79%, all far below 20%. Drift is the only aggregate check
  passed by every candidate. This rules out treating a rank-8 breadth increase
  alone as the bounded Gate 0 repair; it does not yet distinguish insufficient
  rank from acquisition/metric-to-control mismatch.
- The predeclared failure branch selects the best rank-8 query ranking,
  `official_default_r8`, and authorizes only its rank-16 counterpart. No rank-8
  target is selected, no final Writer target is sealed, and no further support,
  optimizer, or rank search is legal. The rank-16 test remains a Gate 0
  capacity diagnosis, not permission to simplify future Writer generation.

## Rank-16 escalation preserves the support and acquisition contract

- `configs/gate_zero_target_support_rank16.toml` is hash-bound to the rank-8
  audit, six-state screening grant, and completed rank-8 result. It activates
  exactly `official_default_r16`: the same 37 action-expert q/v plus state,
  action and time projection targets, with rank/alpha 16, dropout 0, and
  742,656 trainable parameters. Data rows, fixed-noise query evaluator, LR
  `1e-4`, AdamW, effective batch 64, candidate steps and 750-step budget are
  inherited unchanged from the immutable rank-8 contract.
- Rank-16 fitting still cannot see rollout outcomes. Only after both selected
  states freeze may the source screen access init states 32--39. A passing
  support would then freeze before confirmation on 40--47; a failed screen
  authorizes no rank 32, alternate support, optimizer search, Gate 0, or Writer.
  Validation/held numeric access and locked-report reuse remain false.

## Rank-16 screen is a small-recipe negative, not a LoRA-capacity negative

- Both canonical fits completed rc 0 and pass all artifact checks under
  `$EMBER_OUTPUT_ROOT/gate_zero/target_support_audit/fit/rank16_20260718T153934Z`.
  Task 3 selected step 50 with 5.7618% independent-query reduction and drift
  0.0192299; task 4 selected step 25 with 3.2726% reduction and drift 0.00798669.
  The immutable fit grant SHA256 is
  `e4761d2bdc5d02b843a947a87882583bfdfeecc72223f50c2cc1433919108041`.
- Long-run `gate0_support_rank16_screen_20260718_155426` completed rc 0 with all
  mechanics valid, eight checksum-valid source rollouts, and no locked-report,
  validation, or held numeric access. On exact init states 32--39, task 3 and
  task 4 each move from frozen-base 2/8 to own-LoRA 3/8. Median gain is 12.5pp,
  positive count is 2/2, and median fixed-query reduction is 4.517%; the
  unchanged 15pp and 20% checks fail. Result SHA256 is
  `65b2abffcf8b2c7e8907c03f4e21cd8435da38b94afb9e8b41337a54bd323b00`.
- This evidence rejects the frozen combination of 12 source support
  demonstrations, 750 optimizer steps, the custom early-candidate selection,
  and default-like rank-16 LoRA as sufficient Gate 0 acquisition. It does not
  establish that task-local LoRA lacks behavioral capacity. The previous
  contract's no-further-recovery/final-negative implication is preserved as
  provenance but superseded because a known-positive mature SmolVLA LoRA
  competence control was still absent. Gate 0, final Writer targets, and Writer
  remain unauthorized.

## Mature task-local LoRA positive control is primary-source anchored and bounded

- LeRobot v0.6.0 revision
  `30da8e687a6dfc617fcd94afc367ac7071c376ce` provides the exact SmolVLA
  default PEFT target regex: all 16 action-expert q/v projections plus state,
  action input/output, and action-time projections. This is an implementation
  anchor only. The roughly 50-episode, batch-64, 20k-step SmolVLA recipe trains
  the action expert/projections and therefore is not behavioral evidence for
  LoRA.
- OpenVLA revision `c8f03f48af692657d3060c19588038c7220e9af9` and OpenVLA-OFT
  revision `e4287e94541f459edc4feabc4e181f537cd569a8` provide empirical
  LIBERO anchors for rank-32 broad/all-linear LoRA, longer training, Gaussian
  zero-delta initialization, and image augmentation. Their architecture differs,
  so the control transfers capacity/duration/augmentation principles rather
  than mechanically copying all-linear targets or their exact schedule.
- The pre-outcome contract
  `configs/gate_zero_mature_lora_positive_control.toml` declares one primary:
  40 source support demonstrations (episode roles 0--7, 8--27, and 28--39), six
  independent query demonstrations 40--45, 20k steps, effective batch 64, 37
  default-like targets, rank 32/alpha 16/dropout 0, 1,485,312 trainable LoRA
  parameters, Gaussian exact-physical-zero initialization, AdamW `1e-4`, 1k
  warmup plus cosine decay, deterministic 90--100% random-resized crops, and
  fixed final-step selection. Fresh source init states 40--47 are the only
  closed-loop outcome surface. Source actions may supervise Gate 0 but remain
  hidden from Writer inputs; validation and held numeric access remain zero.
- The pass thresholds remain +15pp median closed-loop gain, 2/2 positive tasks,
  and 20% median independent-query reduction. Drift is diagnostic for mature
  task-specific competence rather than a reason to select an early checkpoint.
  One mechanically valid primary failure allows only the predeclared
  all-action-expert-linear rank-32 compatibility recovery. No split change,
  threshold reduction, held access, or layer/rank grid is authorized. A primary
  pass seals its exact LoRA space for Writer and every matched downstream arm;
  a bounded failure is classified before any broader project conclusion.
- The one live rank-32 mechanics smoke on physical GPU 4 resolved exactly 37
  targets and 1,485,312 trainable parameters, passed exact-physical-zero
  initialization, and completed one augmented 64-sample optimizer/scheduler
  step with finite loss 0.44569 and gradient norm 0.10178. Initial/next warmup
  learning rates were `9.9900e-8`/`1.9980e-7`; all 64 sampled source rows were
  unique. Peak Torch allocation/reservation was 17.35/17.47GiB and the device
  returned to 0MiB with no worker residue. No query, rollout, locked,
  validation, or held outcome was evaluated. A CPU augmentation microbenchmark
  took only 0.04--0.05s per 64-sample/two-camera batch, so the slower first live
  step is warm-up rather than evidence for a new performance rewrite.

## Mature fit cadence changed safely at an atomic boundary

- A race between the initial 20k launch and the owner's staged-execution
  correction was contained with `SIGINT`, not `SIGKILL`, after roughly 50
  seconds. Each task logged only steps 1 and 10 before interruption. These
  volatile steps were never published as a candidate and have no scientific
  authority.
- Both partial outputs contain a valid step-0 candidate and schema-2 recovery
  with trainable state, optimizer, scheduler, RNG and file hashes. Candidate
  state SHA256 is
  `6aea08907880b0a9ce9bc1176429e076848502ab70ca3dff2ec6b8180bf9201f`;
  task-3/task-4 recovery-manifest SHA256 values are
  `1583c06f9ea3b6017963a1cb09e5daa67f11faea28a318fb2cd763f60433a9f4`
  and `3ccbc35440bade34e7f1c5a027d2acca7e74b3877f57f55808282555c9c38fe5`.
  Exact resume from this last complete boundary is valid and does not access a
  final rollout surface.
- `configs/gate_zero_mature_lora_stage_ladder.toml` freezes the same-trajectory
  1k/2k/5k/10k/20k ladder and source-query continuation rules before step-1k
  outcomes. The first segment continues only when both tasks are nonnegative
  and median reduction is at least 2%; failure stops for bounded diagnosis.
  This changes runtime cadence only, not targets, rank, support, augmentation,
  optimizer, maximum budget, or final Gate thresholds.
- Stage 1k completed rc 0 for both tasks in 13:03. Against identical step-0
  query rows, task 3 improves flow MSE by 7.959% with drift 0.01676 and task 4
  by 5.754% with drift 0.01883; median improvement is 6.857%. Both task
  reductions are nonnegative and the median exceeds the frozen 2% rule, so the
  same trajectories are authorized to resume to 2k. This is a positive query
  trend, not Gate 0 or closed-loop evidence.
- Stage 2k also completed rc 0 for both tasks. From the same frozen step-0
  query anchors, task 3 improves by 8.036% while task 4 improves by 4.504%; the
  median is 6.270%. Relative to stage 1k, the median regresses by 0.587
  percentage points, within the predeclared 1pp allowance. Task 4's individual
  1.251pp decline is retained as a diagnostic warning, but the frozen rule is
  aggregate and cannot be changed after seeing results. Candidate/recovery/
  state/telemetry hashes all validate, GPU 4/5 were released, and no final
  rollout, validation, or held surface was accessed. The exact same trajectories
  may therefore continue to 5k. This remains source-query trend evidence only,
  not a useful-update oracle or behavioral success claim.
- Stage 5k is a clean negative continuation decision. Both jobs completed rc 0;
  task 3 query MSE is 0.516400 versus 0.518030 at step 0 (0.315% reduction),
  while task 4 is 0.502517 versus 0.488088 (-2.956% reduction). The median is
  -1.321%, 7.591pp below stage 2k, so it fails every frozen 5k-to-10k condition:
  median at least 10%, every task at least 2%, and median regression at most
  1pp. No 10k/20k continuation is authorized.
- Mechanics and data identity do not explain the failure: candidate, recovery,
  optimizer, scheduler, RNG and telemetry artifacts validate; query and anchor
  digests are unchanged at every stage; both jobs used the intended task-local
  source rows and released GPU 4/5. Mean support loss nevertheless fell from
  0.3946/0.4415 (steps 1--1k) to 0.3615/0.4075 (1k--2k) and 0.3306/0.3708
  (2k--5k), while drift grew from 0.01676/0.01883 at 1k to
  0.04563/0.04393 at 5k. The leading bounded diagnosis is task-support
  over-specialization under a still-high long-horizon LR, with target-support
  sufficiency not yet resolved; it is not evidence that mature LoRA lacks
  behavioral capacity because no mature closed-loop surface was opened.
- The existing contract already predeclared exactly one compatibility recovery
  after a mechanics-valid primary failure. A live structure-only enumeration
  of the frozen SmolVLA checkpoint found 112 action-expert linear modules
  (q/k/v/o plus gate/up/down across 16 layers); retaining the five existing
  state/action/time projections yields 117 exact rank-32 targets and 7,027,200
  trainable LoRA parameters. This all-action-expert-linear recovery changes
  target support only and keeps the mature data/optimizer/augmentation/Gate
  contract. It is the final bounded support variant and cannot itself seal the
  Writer contract before independent query and closed-loop evidence.
  The recovery fit/stage contracts were frozen before any recovery training or
  query outcome with SHA256 values `82f5203ed86a25dac386bde68cb8a76efaba03c0f230fe2bd0249bb8d64fe15c`
  and `f3b66cff59135f52e81ab9ef387230381662fad6797e5c63557f791bd015739f`;
  every sampler, noise, augmentation, evaluator and final rollout seed matches
  the primary, so target support is the only scientific change.
- The conditional support is mechanically viable on one A100: PEFT resolved
  exactly 117 targets and 7,027,200 trainable parameters, preserved exact-zero
  physical initialization, and produced finite loss 0.45668 and gradient norm
  0.13756 on one 64-row optimizer step. The row digest
  `9a94f4376435fa94d3a96b25498e598f05790bb3b0ea067c15118d78de303605`
  is identical to primary task 3 step 1, directly confirming matched sampling.
  Peak Torch allocation/reservation was 17.93/18.56GiB and the GPU returned to
  0MiB. No query candidate or rollout outcome was evaluated.

## All-linear recovery shows a positive but task-heterogeneous 1k query trend

- The two `all_action_expert_linear_r32_same_recipe` jobs completed normally at
  the predeclared step-1000 atomic boundary in 13:35/13:21. Task 3 query MSE
  falls from 0.518030 to 0.474614 (8.381%); task 4 falls from 0.488088 to
  0.481860 (1.276%). Their median reduction is 4.829%, and both task reductions
  are nonnegative, so the frozen 1k-to-2k rule passes. Task 4 is substantially
  weaker than task 3 and weaker than its primary-support 1k result; that is a
  diagnostic signal, not authority to rewrite the result-blind continuation
  rule.
- Candidate and recovery validators pass for both tasks. Candidate-manifest
  SHA256 values are
  `ff2432f7c31d45fc25bba37c022d09180df1cd9df266030794ff6e52123c707f`
  and `2a6b28fd9650d3d915b99cede45feb624f537c7f3ea8aaa66c224695940d916f`;
  trainable-state values are
  `c3105040ff92b920275f85f33600b2a6077d0c63977e5fbe5bfc31509a1f8b90`
  and `4b1b44f0102777b89ae6ac97456b63e71fd4b885ffc71b3cd9b16e20e6757165`;
  recovery-manifest values are
  `0191ba569b8133d03f99a81d8353982a44566197cf688b42a761064d9151bd32`
  and `6e12ab7c3e5e721de663fd7339b3abd7fdbc8e1749c020c858beae9b5c1f1048`.
  Query/anchor digests and query sample counts exactly match the primary
  mature recipe, isolating target support as intended.
- Telemetry SHA256 values are
  `a441199d0ab40d377ba6adbebff8693fad830c6febbde0b7c4663c0a80d7d4b8`
  and `da509a07e4d3945098ef0e1e6cab2d81f48b72ac214a90ab66a8f9675d0dac3e`.
  Peak device memory is 19,143/19,379MiB, leaving more than 60GiB per device;
  memory-active mean utilization is 89.39%/90.69%. Both GPUs were released and
  no final closed-loop, validation, or held surface was accessed. This evidence
  authorizes only exact-resume to the 2k query boundary, not Gate 0, Writer, or
  target-support sealing.

## All-linear recovery overfits both independent queries by 2k

- Both task trajectories completed the exact-resume step-2k segment with rc 0
  in 13:04/13:08, then failed the frozen continuation rule. Task 3 query MSE
  changes from 0.518030 at zero to 0.534345 (-3.149% reduction); task 4 changes
  from 0.488088 to 0.547113 (-12.093%). Their median is -7.621%, versus +4.829%
  at 1k, a 12.450 percentage-point regression. The result fails the required
  median >=5%, each task >=0%, and prior-median regression <=1pp conditions;
  neither trajectory may continue to 5k, 10k, or 20k.
- Mechanics remain intact. Candidate-manifest SHA256 values are
  `86baac172bb380fd29c92a0cd021991ed32fede3d2e185fc0c03a742f0e54006`
  and `220f366324d29c57b1fac13436c52984be65d732f53810ce3b817d2ec1315222`;
  trainable-state values are
  `0ed133859bace1410c0d83368da16948e19a20400c8074116bedcce1ca059749`
  and `77cfc9ed3f98d2bd3fa6ebe16773f7c3acc141cd83f24fcebff9e366f77a324b`;
  recovery-manifest values are
  `44ed9c9820a1138f905f46cb2e541c85d63d1a2d73a71c7961c6250c1736c901`
  and `1bcd413d38a30b896aceecd5e64c4cef171fc29e017989f1f3f6e743473815b5`.
  Query/anchor digests and sample counts remain identical to 1k, and both GPUs
  released cleanly.
- Logged support loss keeps improving from 0.3826/0.4263 over the first 1k
  segment to 0.3109/0.3477 over the second, while action drift rises from
  0.03858/0.02884 to 0.05816/0.06011 and independent query performance
  reverses. This is a mechanics-valid optimization/generalization overrun, not
  evidence that the trainer failed. It also cannot establish that LoRA lacks
  behavioral capacity because the result-blind contract stops before the fixed
  final candidate and no formal closed-loop surface was accessed.
- The conditional all-linear route exhausts the predeclared target/rank
  variants. The only remaining bounded discriminator already named before this
  outcome is a non-matched task-local action-expert capacity upper bound under
  sufficient source supervision. It may diagnose whether the current problem
  is LoRA-space capacity or acquisition/data/optimization, but cannot itself
  pass the matched LoRA Gate, seal Writer support, or authorize Writer.

## Final capacity discriminator isolates trainable-state class only

- The bounded action-expert upper-bound contract was frozen before any of its
  fit or query outcomes. Its config SHA256 is
  `8fd7f3a5fac0bbfef6fb7281e48b7ef9df7e5b95a74e9446d1e4c8e8ed72327d`;
  the result-blind stage-ladder SHA256 is
  `69640a07e97915e9ac51ac31153d13f4df4e3154845afdb2a136def230f4bc98`.
  Both bind the primary and all-linear contracts plus the two all-linear 2k
  candidate/recovery/telemetry failure packets, so no new LoRA target/rank or
  unbounded optimizer search is introduced after outcomes.
- The diagnostic reuses the same 40 source support demonstrations, task/query
  IDs, effective batch 64, absolute-step sampling, fixed query noise,
  random-resized crops, AdamW `1e-4`, 1k warmup/cosine decay, 20k maximum, and
  1k/2k/5k/10k/20k staged thresholds. The sole scientific difference is that
  the already validated 99,880,992 action-expert/state/action/time projection
  parameters are updated directly. This makes a positive result evidence for
  task-local model capacity outside LoRA and a negative result evidence against
  the current data/acquisition/optimization control; neither result can itself
  pass the matched LoRA Gate or authorize Writer.
- The existing oracle fitter, sampler, query evaluator, scheduler-bound atomic
  recovery and launcher remain the single canonical path. The mature contract
  loader gains one fail-closed config mode and runtime prerequisites verify the
  exact all-linear artifact hashes. No alternate trainer, evaluator, rollout
  path, bank, geometry, or future architecture reservation is added. Focused
  contract tests pass 15/15, the full suite passes 214/214, and the real
  prerequisite load validates the frozen source checkpoint at step 10k with
  exactly 99,880,992 declared trainables.
- The one permitted live smoke confirms the declared parameter identity in the
  actual model: 99,880,992 trainable parameters across 155 tensors. Its first
  effective batch digest is
  `9a94f4376435fa94d3a96b25498e598f05790bb3b0ea067c15118d78de303605`,
  exactly matching both mature LoRA controls and isolating the trainable-state
  class. One augmented optimizer/scheduler step has loss 0.42178, gradient norm
  1.65898, and 62.3 samples/s; peak Torch allocation/reservation is
  17.86/18.93GiB. The smoke saves no query candidate, accesses no rollout,
  validation, or held surface, and releases GPU 4 to 0MiB.

## Mature action-expert upper bound also fails independent query at 1k

- Both non-matched task-local action-expert fits complete the predeclared 1k
  boundary with rc 0 in 13:19/13:06. Task 3 query MSE changes from 0.518017 to
  0.545061 (-5.221% reduction); task 4 changes from 0.488102 to 0.584688
  (-19.788%). The median is -12.504%, and both tasks are negative, so the frozen
  1k-to-2k rule fails. Neither output may resume to 2k or access the formal
  closed-loop surface.
- The failure is not mechanical. Candidate-manifest SHA256 values are
  `10bc695a36e26a821b15ea20bc3d711b8939658f6e51526503d7e87380185d1c`
  and `8390d3e3b8c83c825987aaec3bf45fbad464d7344f426dfe7ada63c848d42e8f`;
  trainable-state values are
  `429a85eb2aef99bcf63ccb503cd9368f8a07851756cfbda5bba1d89aad46f25d`
  and `92d2287c435977c9028756dcefd192b4429c6ff81b4ba949d8d4b7b5cb42262f`;
  recovery-manifest values are
  `b57d898d6919ed417e27bcfbaad22aac9ab668694917a703071f5a2678ed9564`
  and `7993186c94ad91aa70df45e3c0f2d0431ca810964591521dd27e423de384e7b2`.
  Query/anchor identities are the same mature source rows, artifacts validate,
  and both GPUs release cleanly.
- Support learning moves in the opposite direction from query: task 3 logged
  support loss falls from 0.4055 over the first 100 steps to 0.2835 over the
  last 100; task 4 falls from 0.4508 to 0.3209. Drift reaches
  0.06313/0.06903. Peak device memory is 19,455/20,195MiB and active-window
  utilization is 89.68%/91.05%; the complete staged output occupies 2.0GiB.
  This is a mechanics-valid task-local acquisition/generalization failure, not
  evidence that the larger update implementation failed.
- Because the same sufficient-data recipe fails even outside LoRA, these
  outcomes do not support blaming LoRA target support or future Writer
  generation. They exhaust the result-blind target/rank/capacity path and
  require an explicit Gate-recovery decision. The lowest-cost discriminating
  option is one predeclared lower-LR, dense-early checkpoint recovery on the
  action-expert upper bound, followed by the same acquisition schedule on LoRA
  only if that upper bound becomes positive. The alternative is to record this
  supervised task-local acquisition surface as insufficient and revise the Gate
  0 evidence plan. Either changes the current frozen recipe, so no further run
  is launched silently.

## The action-expert direction is useful at smaller magnitude

- A bounded post-hoc probe scales the saved step-1000 physical delta by
  `[0, 0.1, 0.25, 0.5, 0.75, 1]` and reuses the exact independent source-query
  evaluator. It is a diagnosis, not candidate selection: no training,
  validation, held, formal closed-loop, Gate-0, or Writer authority is opened.
  Both scale-0/scale-1 query and drift endpoints exactly reproduce the saved
  candidates, and both artifact checksum sets pass.
- At scale 0.25, task 3 improves 2.705% and task 4 improves 1.311%, for a 2.008%
  median; at scale 0.5 task 3 remains +3.123% but task 4 becomes -1.251%.
  Therefore the final training direction contains task-generalizing signal,
  while its magnitude crosses task 4's useful range. The earlier failure is
  narrowed from an undifferentiated data/acquisition failure to an optimization
  magnitude overrun, although scaling a final delta is not the same operation
  as retraining at a lower LR.
- This makes one 0.25-scaled schedule the lowest-cost discriminating recovery:
  keep every authority fixed and scale only peak/decay LR from
  `1e-4/2.5e-6` to `2.5e-5/6.25e-7`. The checked-in contract and ladder forbid
  an LR grid, closed-loop use of the non-matched arm, threshold changes, and
  further target/rank search. A real step-1000 pass may authorize the same
  schedule on LoRA only; it cannot itself pass Gate 0 or authorize Writer.
- The source-base competence surface is already 8/8 for task 3 and 5/8 for task
  4. Thus task 3 has no positive closed-loop success-count headroom; before a
  final matched LoRA behavioral Gate, the source-only evaluation design must
  preserve task 3 as a maintenance/spec control or predeclare another legal
  source task with measurable headroom. This ceiling cannot be repaired by
  lowering a success threshold after outcomes.

## Lower learning rate converts the action-expert acquisition into positive query evidence

- The same task-local action-expert trajectory was run by exact resume at
  steps 250, 500, 750, and 1000 with only the predeclared 0.25 learning-rate
  scale. Task 3 query reductions were 4.922%, 7.269%, 8.771%, and 7.883%; task
  4 reductions were 3.790%, 4.808%, 5.177%, and 3.710%. The step-1000 median is
  5.797%, so the frozen median-at-least-2% and every-task-nonnegative criterion
  passes. The decline from the step-750 median of 6.974% is recorded, and the
  non-matched trajectory is not extended to 2k.
- Every stage completed rc 0 through the existing atomic resume path. Final
  task-3/task-4 candidate manifests hash to `a1eaaf1d...24ab` and
  `ca5f1586...be1c`; recovery manifests hash to `389257ca...1255` and
  `7a495a9f...efdc`; telemetry hashes are `241bce7a...2588` and
  `1acba6cf...9dac`. The 99,880,992-parameter states and optimizer/scheduler/RNG
  recovery remain loadable. No formal closed loop, validation, held surface,
  Gate-0 decision, or Writer authority was used.
- This result distinguishes update magnitude from a total absence of useful
  supervised signal. It does not prove LoRA capacity or behavioral utility.
  The only authorized next comparison is the original 37-target rank-32 LoRA
  under the identical lower-LR schedule and staged source-query rules.
- A second benchmark-design issue remains independent of the optimizer result:
  task 3 has frozen-base source competence 8/8, so a positive success-count gain
  on both tasks is impossible. The formal matched LoRA closed-loop surface must
  stay unopened until a result-blind, source-only headroom-safe contract is
  frozen; thresholds cannot be repaired after observing rollout outcomes.

## Matched mature-LoRA recovery is sealed before outcomes

- Contract SHA256 `693cd614...fafca` fixes the same 40 legal source support
  demonstrations, query demos 40--45, sampler/noise/augmentation/AdamW seed,
  37 exact SmolVLA-native targets, rank 32/alpha 16/dropout 0, and 1,485,312
  trainable parameters. Its only optimization change from the failed primary
  LoRA recipe is the action-expert-validated `2.5e-5/6.25e-7` peak/decay LR.
  Ladder SHA256 `436bae48...17d3` permits only 250 -> 500 -> 750 -> 1000 exact
  resumes and forbids closed loop, a second LR, target/rank search, Gate 0, or
  Writer authorization during staging.
- The strict loader binds both final action-expert step-1000 packets and all
  upstream contracts. A real prerequisite load resolves the frozen source-base
  checkpoint at step 10k, the canonical launcher dry-run stops at step 250,
  and the full repository suite passes 219 tests. This is one configuration
  mode of the existing fitter/launcher, not a second training path.
- The first matched-LoRA segment reaches atomic step 250 with task-3/task-4
  query reductions of +0.892%/+1.038% (median +0.965%). Both exceed the frozen
  nonnegative continuation floor; drift remains 0.00167/0.000925. This is an
  early positive query trend, not Gate-0 or behavioral evidence. Candidate
  manifests hash to `871c5093...5fef`/`4a691249...8c92`, recovery manifests to
  `6e969071...1ec2`/`6662e1a0...bfe1`, and telemetry to
  `56bef9ce...a54`/`94ebc132...1d36`. Exact resume to 500 is authorized.
- At step 500, task-3/task-4 query reductions increase to +3.388%/+2.611%
  (median +2.999%) with drift 0.00572/0.00495. Candidate hashes are
  `867d1be1...affc`/`643343e2...2a59`; recovery hashes are
  `4092f457...ff56`/`676ff2af...f822`. The same data/query identities persist,
  so the frozen step500-to-750 rule passes without a new trajectory or surface.
- Step 750 further improves task 3/4 to +4.925%/+3.728% query reduction
  (median +4.326%), with drift 0.00768/0.00684. Candidate hashes are
  `6cedeb90...e81c`/`2dcfe206...e488`, recovery hashes are
  `c03d8c3f...9124`/`378da52c...d9d3`, and fixed query identities remain exact.
  This passes the result-blind 1%-median/nonnegative rule for the final 1k
  source-query segment; it still provides no closed-loop authority.
- At the final authorized step 1000, task-3/task-4 query reductions reach
  +5.798%/+4.236% (median +5.017%), above the frozen 2%-median and nonnegative
  per-task criterion. Drift remains 0.01056/0.00858. Candidate hashes are
  `06127175...47cf`/`9e76ed1d...b3cc`, trainable states are
  `b20d60b1...a68c`/`ba0f2268...c053`, and recovery manifests are
  `454144c7...6883`/`26880083...ecb9`. This is positive independent-query
  evidence that the sealed LoRA space and acquisition recipe have useful
  function, not yet robot closed-loop evidence.
- The staged trajectory is permanently stopped at 1000 despite its positive
  trend: the result-blind ladder authorized no 2k continuation. A formal
  closed-loop comparison must first replace the mathematically impossible
  positive-gain-on-both rule caused by task 3's frozen-base 8/8 ceiling. That
  redesign must use only legal source outcomes, preserve task 3 as a
  maintenance control, and be frozen before any new LoRA rollout outcome.

## The owner approved the paired maintenance/improvement Gate before outcomes

- Prior source competence makes the old positive-gain-on-both-tasks rule
  unidentifiable: task 3 is already 8/8 while task 4 is 5/8. Proposal A is
  result-blind and uses a fresh paired source slice, init states 40--47 and
  seeds 5800--5807. Task 3 now tests non-harm (paired net wins >=0); task 4 must
  have at least two available failures and recover at least two net wins; the
  two-task aggregate must recover at least two net wins. This preserves a
  behavioral requirement without inventing success-count headroom.
- The active owner-approved contract SHA256 is
  `1f92f80ddcc63be7c6a3ef3da1fe63f9870df27a0537ec02c3429bab71440a52`;
  pending proposal SHA256 `8c7ae12b...075d5c` and commit `108ce65` preserve the
  pre-approval state.
  It also retains the already observed independent-query safeguards: every
  task must reduce query flow MSE by at least 2%, and every task's action-drift
  proxy must remain at most 0.02. An underpowered task-4 base slice has a
  distinct `headroom_absent` outcome and cannot pass even if LoRA scores 8/8;
  an actual failure with available headroom enters bounded Gate recovery.
- This is still Gate-0 evidence only. A pass can establish functional utility
  for the exact 37-target, rank-32/alpha-16/dropout-0, 1,485,312-parameter LoRA
  contract and authorize direct Writer acquisition. It cannot support claims
  about Writer/video utility, validation or held performance, RL, or the
  overall EMBER hypothesis. No validation, held, or locked-report numeric
  surface is used in either the grant or the paired rollout.
- The owner explicitly selected Proposal A on 2026-07-19 Asia/Singapore before
  any new outcome. Only authorization metadata changed:
  `owner_decision_required=false` and `screening_rollout_authorized=true`.
  Tasks, seeds, init states, evaluator, matched base/LoRA arms, LoRA structure,
  query/drift safeguards and success rules remain byte-for-byte unchanged.
  If task 4 lacks at least two fresh base failures, the contract's
  base-first distributed barrier forbids opening either LoRA arm; the
  `headroom_absent` result contains base arms only and requires a result-blind
  source-task replacement freeze.
- The first launch under historical SHA256 `ba3ee431...f132f` failed before
  any episode or policy outcome: the evaluator defines `warmup_seed_start` as
  the *last* warm-up batch start, so 5760 could not immediately precede report
  seed 5800 at batch size 8. The single mechanical repair changes only that
  value to 5792 (yielding warm-up batches 5768--5775 through 5792--5799).
  Report seeds, init states, policy RNG, LoRA states, thresholds and all data
  surfaces are unchanged; both failure packets and telemetry remain bound in
  the corrected contract.

## Proposal A fails behaviorally despite valid offline query gains

- Canonical run `gate0_mature_lora_headroom_owner_a_20260719_025534` completed
  rc 0 from clean commit `dbcb729` in 171.95 seconds on GPUs 4/5. The base-only
  barrier first found 3/8 success on both task 3 and task 4, so both tasks have
  five failures and genuine fresh-slice headroom. It then legally opened the
  two matched LoRA arms. GPUs released after completion; no EMBER process
  remains on them.
- Task 3 changed 3/8 -> 2/8 and task 4 changed 3/8 -> 4/8. The exact paired
  net wins are -1 and +1, hence aggregate zero. Task-3 maintenance, task-4
  minimum +2, and aggregate minimum +2 all fail. The independent-query checks
  (+5.798%/+4.236%) and drift checks (0.01056/0.00858) still pass. This narrows
  the failure to conversion from supervised query improvement to stable
  closed-loop behavior; it is not a data-authority, state-load, headroom, query
  acquisition, or drift-cap failure.
- Result SHA256 is `84116faaffd5115a72f4d49efa2f2467445ca0ac61edac265e571a1e8564c98f`.
  The four arm videos, gallery, eval info, result, freeze grant, and telemetry
  checksum sets all validate. The retained packet is only 636KiB plus a 12KiB
  freeze and remains viewable through its local gallery and Trackio run
  `owner_a_20260719T025534Z`. Gate 0, Writer, and final target sealing are false.
- The owner's temporary Option-B choice to replace task 3 was made before the
  owner was shown this raced result, but the remote run completed before that
  instruction arrived. Once informed that task 3's fresh base is 3/8 rather
  than the old 8/8 competence slice, the owner withdrew replacement. No
  replacement base or LoRA rollout occurred. The exact four-file WIP is kept
  outside the active path as stash `201d097ec542b76014eb885dbd04ffb166221476`.
  Neither the A LoRA regression nor task-4 +1 was used to choose a new task.

## Earlier exact candidates are the cheapest conversion diagnostic

- Steps 500 and 750 are existing immutable states on the same task-3/task-4
  37-target rank-32 trajectory; no retraining or 2k continuation is needed.
  Both already satisfy the pre-existing per-task 2% query-reduction floor and
  0.02 drift cap. Step 250 is excluded because it does not satisfy that reliable
  offline criterion, while step 1000 is the failed A reference.
- Before any new candidate-step rollout, the diagnostic freezes the A slice
  (init 40--47, seeds 5800--5807), evaluates only four LoRA arms, and retains
  the original positive-improvement semantics: both tasks must improve and the
  median gain must be at least 15pp. Among passing states, maximum aggregate
  paired net wins wins with the earlier-step tie-break. Failure supports no
  default longer training; success authorizes only a separately grant-bound
  fresh-seed matched recovery Gate. This is transparent post-failure recovery,
  not a claim that its rule preceded Proposal A.

## Gate 0 is a useful-update Gate, not an SFT-only Gate

- The current 40-demonstration supervised LoRA evidence establishes independent
  query-loss improvement, but Proposal A shows that this has not yet produced
  stable closed-loop utility. Therefore supervised acquisition is one oracle
  route, not a definition of Gate 0.
- The already frozen candidate-step diagnostic remains the cheapest first
  discriminator and must complete before another recovery. If it still lacks
  credible closed-loop positive evidence, the next bounded source-only test is
  a four-arm comparison on unchanged tasks 3/4: frozen base; supervised LoRA;
  matched zero-init LoRA plus ordinary task-local RL; and supervised-LoRA-init
  plus the identical ordinary task-local RL. LoRA support/capacity, evaluator,
  init states/seeds, reward, estimator/optimizer, interaction budget, and
  compute accounting remain matched.
- If only the RL arms improve behavior, the supported conclusion is only that
  ordinary task-local RL can find a useful update in the LoRA space. Evidence
  that the supervised initialization helps later adaptation requires the
  supervised-init RL arm to stably beat matched zero-init RL. Neither result is
  Writer evidence because Gate-0 RL updates only task-local LoRA and contains
  no Writer.
- This recovery must use a 10--30 minute early check and resumable segments no
  longer than one to two hours, stopping once reliable matched evidence answers
  the Gate. It cannot change task, seed, or threshold to evade Proposal A,
  access validation/held/locked surfaces, or burn arbitrary step milestones.

## The task-local RL recovery is frozen before outcomes

- Contract SHA256
  `75ceeec398f472d53fb1c7b88b4dd135469b0f841bbf8ac3dfc0ac4b13cd5c68`
  binds the failed Proposal-A result, failed candidate-step diagnostic, base
  checkpoint, immutable task-3/task-4 supervised step-1000 states, exact
  37-target rank-32/alpha-16/dropout-0 LoRA, source-only surfaces, and the
  four matched arm roles. No task-local RL outcome existed when this contract
  was sealed.
- SmolVLA does not provide an exact normalized action likelihood suitable for
  a faithful PPO ratio. The bounded compatible choice is episodic AWR-style
  Monte-Carlo reward-weighted regression on the model's native per-sample flow
  loss, anchored to AWR (`arXiv:1910.00177`) and the online robotics use of
  advantage-weighted actor updates in AWAC (`arXiv:2006.09359`). It uses binary
  simulator success, a batch-mean baseline, and deterministic matched Gaussian
  action exploration; it makes no PPO or exact-likelihood claim.
- The early node is 16 source episodes per task/initialization and the absolute
  maximum is 32 on the same exact-resume trajectories. The reusable development
  slice can only select a checkpoint for a new hash-bound fresh Gate. A
  development improvement proves neither Gate 0 nor Writer utility. Cross-arm
  comparison of supervised-init plus RL against zero-init plus identical RL is
  reported explicitly; only the former outperforming the latter supports a
  helpful-initialization interpretation.
- The first canonical four-rank launch failed closed during rank-2 collection,
  before any optimizer update or development rollout. The original failure
  packet is retained under run `gate0_task_local_rl_ep16_20260719_044843`.
  A one-card repeat of that already consumed source training slice showed all
  prompt/anchor/reset-after checks valid: reset reached init states 8--15.
  End-of-rollout IDs changed only for successful auto-reset environments, so
  using them as the initial-state identity was mechanically invalid.
- The same repeat counted 1208 proposed clips: `[0, 3, 10, 0, 0, 0, 1195]`
  by action dimension. Thus 98.9% came from adding continuous Gaussian noise to
  the naturally near-binary gripper command. This is an exploration-domain
  mismatch, not evidence about task-local RL. The active, result-transparent
  recovery preserves std 0.05 on continuous dimensions 0--5, leaves the policy
  gripper output untouched, and computes saturation only over explored
  dimensions. No task, LoRA, reward, seed, budget, optimizer, evaluator,
  surface, or decision threshold changes. Amended contract SHA256 is
  `e138b7d649c192d4618a8e5b9c0f8fe29b60c95a5117815313f271f405d4d406`.
- Recovery2 passed reset, anchor, and repaired saturation guards, then ranks
  1--3 reached the first optimizer forward and failed with an exact shape
  mismatch before any update: replay actions are `[64,50,7]`, while the pinned
  SmolVLA preprocessor pads actions to `[64,50,32]`; fixed flow noise was still
  created as `[64,50,7]`. The unique mechanical repair is to validate the
  processed action tensor and derive deterministic noise shape `[50,32]` from
  it. It changes no samples, rewards, exploration, optimizer, budget, or Gate
  rule. Active contract SHA256
  `504d20bc371078b5ffeabaad84eb1e041423c5167cd7331b91e047a3324f673d`
  binds recovery2 failure packets and permits one final canonical verification.
- Recovery3's explicit guard proved that the preceding repair put the boundary
  one layer too early: the policy preprocessor leaves actions `[64,50,7]`, and
  `SmolVLAPolicy.prepare_action` pads them to 32 only inside the forward pass.
  This explains both prior traces without an algorithmic ambiguity. The active
  implementation validates the 7D replay input and binds deterministic noise to
  frozen `model.config.max_action_dim=32`. Contract SHA256
  `b08a85b8de1bf04c788d217cfab8d34bb984d0f70ab8795e8c0aaf0f19820a37`
  requires a real-model synthetic forward/backward before another environment
  launch; no optimizer step or scientific surface may be consumed by that
  integration check.
- The required real-model integration check passed on one A100 in 29.06 seconds
  from clean commit `cd95342`: processed replay input remained `[64,50,7]`,
  deterministic noise was `[64,50,32]`, native per-sample loss was `[64]`, loss
  `1.031875` and gradient norm `1.098731` were finite, and rc was 0. There was no
  optimizer step, simulator interaction, validation, or held access. Durable
  result SHA256 is `64e522b8863527234e7633a1c8ea72482459b57d43d4b129e4a67c60668a689c`.
  This resolves the model-boundary mechanism only; it does not answer Gate 0.
- Canonical stage-16 run `awr_ep16_recovery4_20260719T052726Z` completed rc 0
  from clean commit `a581eea` in 4:38.96, with complete checksums, four atomic
  recovery states, four videos/gallery, Trackio, and released GPUs. Peak memory
  was 22,123/19,047/18,771/18,771 MiB on GPUs 4--7; the output is 93 MiB.
- Mechanics are valid, nonfinite count is zero, maximum continuous-action
  saturation is 0.00167, and all arms received nonconstant binary reward and 16
  finite optimizer steps. Nevertheless no arm changed its paired development
  outcome relative to its own initialization: task 3 supervised 2/8 -> 2/8,
  task 3 zero 3/8 -> 3/8, task 4 supervised 4/8 -> 4/8, and task 4 zero 3/8 ->
  3/8. Supervised-init versus matched zero-init has paired advantage -1 on task
  3 and +1 on task 4, cancelling in aggregate.
- Supervised-init query reductions remain positive (5.89%/3.66%) and drift stays
  below 0.02 (0.01286/0.00988); zero-init changes are much smaller (query
  +0.28%/-0.06%, drift 0.00037/0.00047). This pattern supports an
  optimization/credit-assignment diagnosis: the estimator is mechanically
  active, but its small-budget native-flow update has no stable closed-loop
  effect. It neither supports a useful RL oracle nor proves that the LoRA space
  is incapable. Frozen status is `task_local_rl_early_check_not_supported`, so
  stage 32 and the fresh Gate are not authorized. Result SHA256 is
  `aab151ea503dbada6eaf3a2242301562a47052e1399ec10986c2279425c13b57`.
- Artifact inspection confirms that the canonical package retains only the
  loadable step-16 model/optimizer/RNG recovery state. Step 8 retains its round
  metrics JSON but not a model state, so no later report may claim or evaluate
  a pre-existing step-8 checkpoint.

## Earlier supervised checkpoints do not recover closed-loop utility

- Canonical source-development diagnostic
  `step500_750_20260719T035642Z` completed main rc 0 from clean commit
  `19e5ea2` in 2:57.68 on GPUs 4/5. All four arms are mechanics-valid, all
  checksums pass, four bounded videos plus the local gallery and Trackio run
  are retained, and both GPUs released. Result SHA256 is
  `aae6e19f14c03a1192cb00aeb05940a48ee1c36ba8b5b07e823066e6602b11cf`.
- Step 500 and step 750 produce identical success vectors on the frozen
  development slice: task 3 scores 2/8 against its 3/8 base and task 4 scores
  3/8 against its 3/8 base. Each candidate therefore has aggregate paired net
  -1, zero positive tasks, and -6.25pp median gain. Their per-task query
  reductions and drift safeguards still pass.
- The frozen decision is
  `candidate_step_magnitude_recovery_not_supported`: neither earlier state is
  selected, the separately frozen fresh recovery Gate remains unopened, and
  Gate 0/Writer remain unauthorized. This specifically weakens the hypothesis
  that step-1000 update magnitude or duration alone explains Proposal A; it
  does not prove the LoRA space or ordinary task-local RL lacks a useful update.
  Under the pre-outcome recovery contract, the next discriminator is the
  same-task four-arm matched ordinary-RL comparison, not more supervised steps.

## AWR early-check evidence isolates reward direction as the next bounded discriminator

- The completed four-arm AWR-style early check is mechanically valid but has
  zero paired net wins in every arm. Because its strictly positive weights
  still regress on both successful and failed rollout actions, it cannot test
  whether explicitly reducing the failed-action conditional-flow proxy is the
  missing mechanism. This is an estimator/credit-assignment ambiguity, not
  evidence that LoRA or EMBER is incapable.
- The next and only currently authorized optimizer recovery is sealed before
  outcomes at config SHA256
  `d322339eb417536a8b96b124b3c8d6324c4b25b95e89f4a3cffb5d6cadce200c`.
  It uses the change in matched per-sample native flow loss from the round-start
  LoRA as a bounded ratio surrogate and normalized signed binary-reward
  advantages. Ratio clipping applies on positive advantages; a quadratic trust
  penalty applies on negative advantages. Old losses for all eight updates are
  computed before the first update under identical augmentation, flow noise,
  and time authority.
- The mechanism is informed by FPO++ (`https://arxiv.org/abs/2602.02481`) and
  its official implementation at commit
  `b80112be1e8362263c4cd176e7aef21a275ff1c6`, but it deliberately omits the
  critic, GAE, many flow samples, entropy regularization, and large interaction
  regime. Any result therefore applies only to this signed-loss-ratio mechanism
  check, never to the full published method.
- Every scientific comparison surface stays fixed: source tasks 3/4,
  zero/supervised initializations, LoRA structure/capacity, exploration,
  binary reward, source training/development seeds and init states, 16-episode
  budget, AdamW settings, evaluator, query/drift safeguards, and candidate
  thresholds. No stage 32, fresh Gate, validation, held, locked-report, Writer,
  or shared update is authorized by this predeclaration.
- The required no-environment real-model integration smoke passed from clean
  commit `2a72bd4` in 30.07 seconds. It used 64 unique legal source-support
  rows and synthetic balanced reward labels, reproduced the round-start loss
  exactly (`max_abs_difference=0`), produced ratio 1.0 and a finite nonzero
  LoRA gradient norm 0.0803, and performed no optimizer step or simulator
  interaction. Peak CUDA reservation was 17,948MiB; GPU 6 released to 0MiB.
  Result SHA256 is
  `b1e75b43e62b2b5ba9f9f3386d8a023e42ece2d696a069a223bcebb13f5b4687`.
  This is interface/mechanics evidence only and unlocks exactly one frozen
  source-development run, not Gate 0 or a fresh Gate.
- The signed-ratio stage-16 run then completed rc 0 in 4:52.99 from clean
  `30d9e22`. All four atomic recovery packets, checksums, mechanics, videos,
  gallery, Trackio records, and telemetry validate; peak memory was
  23,350/19,047/18,771/18,771MiB, output size is 93MiB, and GPUs 4--7 released.
  Result SHA256 is
  `73d681caf4f5d6b67519eb33636e9af905aec7412c18c5d85cd1aaf8d3488703`.
- Behavior still does not improve: task-3/4 supervised-init stays 2/8 and 4/8
  (paired 0/0), while zero-init changes 3/8 to 2/8 and stays 3/8 (paired -1/0).
  Query reduction is +5.68%/+4.01% for supervised-init and +0.074%/-0.113%
  for zero-init; drift and saturation remain inside safeguards. Thus no
  initialization passes, fresh Gate/Gate 0/Writer remain false, and no stage 32
  exists in this recovery.
- This is not explained by a vanishing parameter move alone. Signed-ratio
  physical LoRA displacement is about 0.067--0.072 L2 (roughly 10% of the
  supervised initial physical-update norm), while AWR produced about
  0.096--0.120 (roughly 17% for supervised arms); neither yielded positive
  paired behavior. The more discriminating next test is therefore the
  pre-existing lower-LR action-expert state as a non-matched capacity upper
  bound on the same frozen source-development slice—not an arbitrary increase
  in signed optimizer steps.
- That capacity diagnostic is now result-blind and immutable at contract
  SHA256
  `e313e437fe57f20d2cd390fbede0c89432bb89f1d40dd7d37bcf8156e1af9f3a`.
  It binds both query-positive lower-LR action-expert states (99,880,992
  parameters, 155 tensors), their candidate/state hashes, Proposal-A base
  vectors, and the signed-ratio negative packet. It evaluates only tasks 3/4
  on the identical init-state/seed/evaluator identity and reuses the 3/8 base
  results without rerunning them.
- The decision retains the original behavioral rule: both tasks must improve
  over paired base and median gain must be at least 15pp. A positive result is
  only non-matched evidence that a wider partial-update state can convert query
  signal; a negative result points to query-to-closed-loop acquisition or
  temporal credit rather than merely LoRA parameter count. Either outcome
  leaves Gate 0, target sealing, Writer, validation, held, and locked-report
  access false.

## Writer reward learning and task-local adaptation are distinct causal stages

- The active sequence now explicitly separates: supervised direct-Writer cold
  start; Writer-only RL with frozen base and only Writer parameters updated;
  ordinary task-local LoRA RL with frozen Writer/base and only the generated
  LoRA updated in place; and later adaptation-aware source reward/meta outer
  learning. Generated LoRA is a functional Writer output—not a separately
  optimized variable—during Writer-only RL.
- Gate 0 remains a useful-update oracle rather than an SFT-only Gate. Its
  current RL recovery contains no Writer. If only ordinary task-local RL becomes
  useful, that supports the LoRA search space but not supervised zero-step
  utility; only a stable supervised-init RL advantage over matched zero-init RL
  supports the claim that supervised initialization helps later adaptation.

## Wider action-expert capacity does not repair query-to-behavior conversion

- The frozen non-matched capacity diagnostic completed main rc 0 from clean
  commit `7e5f905` in 1:37.99. Output
  `capacity_20260719T062725Z` has complete checksums, two bounded videos,
  gallery, Trackio run `capacity_20260719T062725Z`, and released GPUs. Result
  SHA256 is
  `9a91fbb8d53bff90a1c6bcb58bef1270f076f14212ca846c369ca8017bf170ad`;
  `eval_info.json` SHA256 is
  `67688b098c02237f744b1a1e38e1eafa1ba444bb6f0a9c22948f4d79d9699bf4`.
- Each candidate is the immutable lower-LR step-1000 partial action-expert
  state with 99,880,992 trainable parameters across 155 tensors and positive
  independent-query reduction. On the exact Proposal-A source-development
  identities, task 3 is 3/8 with the same success vector as base. Task 4 is
  also 3/8: one base success is lost and one different episode succeeds, so
  paired net improvement remains zero. Positive-task count is zero and median
  success gain is 0pp.
- Status is `nonmatched_action_expert_capacity_behavioral_signal_absent`.
  This weakens insufficient LoRA rank/target support as the sole explanation:
  a much wider query-positive task-local state also fails to produce stable
  paired closed-loop gain. Together with the supervised LoRA, AWR, and signed-
  ratio evidence, the leading failure class is acquisition/query-surrogate to
  closed-loop conversion and/or temporal credit, not a vanishing update or
  LoRA parameter count alone.
- This is a non-matched upper-bound diagnostic, not evidence that full fine-
  tuning, a faithful mature flow-policy RL method, LoRA, or EMBER is negative.
  It cannot pass Gate 0, seal Writer targets, or authorize Writer. It instead
  stops blind supervised-step, rank, and target expansion and requires any
  next source-only RL mechanism to state its temporal-credit estimator and
  matched four-arm contract before outcomes.
- Telemetry recorded GPU-4/5 peaks of 9,473/1,751MiB and active-sample mean
  utilization of 21.44%/2.68%. The simulator-bound one-off diagnostic finished
  in under two minutes, so no rerun is justified for scaling aesthetics; later
  independent task/arm rollouts should use process-level parallelism when it
  shortens the scientific wall clock while retaining about 10GiB memory
  headroom.

## Query-flow selection versus generated actions is the next no-rollout discriminator

- Existing immutable success vectors show that the policies are not simply
  identical: task-3 supervised LoRA consistently loses base-success episode 4,
  task-4 supervised LoRA gains episode 1, and the wider task-4 action-expert
  state trades episode 0 for episode 1. AWR and signed-ratio mostly retain or
  exchange the same boundary outcomes. Aggregate teacher-forced flow loss does
  not reveal which time/action errors caused those closed-loop changes.
- The result-blind source-only audit is frozen before its output at config
  SHA256
  `a85de2e89ae0e5477e931cf887b79b6b756aa0c090bf0903353c7bf475262c3d`.
  It uses only query demonstrations 40--45, eight evenly spaced anchors per
  demo, the existing inference-noise seed, and the already immutable base,
  supervised-LoRA, and action-expert states. The generated 50x7 normalized
  action chunks are compared with demonstration chunks overall and by episode,
  action dimension, and four contiguous time partitions.
- This audit consumes no environment rollout and has no Gate threshold. It can
  distinguish fixed-flow-query surrogate mismatch from teacher-forced/open-
  loop compounding or temporal-credit failure, but cannot pass Gate 0, change
  targets, or authorize Writer. The reusable metric stays with the canonical
  fixed-query evaluator; the one-time runner has a removal trigger once its
  packet and the next recovery contract are frozen.
- The audit completed rc 0 on clean `9568921`; result SHA256 is
  `95f8adfc0d16c72d5443b7466ff72c92dca9616d26106e25a1d42bb71150c1c6`.
  All checksums pass, no environment was opened, output is 56KiB, Trackio run
  is `alignment_20260719T065332Z`, and GPU4/5 released. Peak memory was
  5,635/4,317MiB and active-sample mean utilization was 76.76%/79.20%.
- The mismatch is directionally uniform at the aggregate level. Task-3/4
  supervised LoRA improves fixed flow-query loss by 5.798%/4.236% but worsens
  generated-action MSE by 3.114%/2.479%. The partial action expert improves
  flow-query loss by 7.883%/3.710% but worsens action MSE by 0.886%/0.304%.
  Episode, action-dimension, and chunk-quarter effects are heterogeneous, but
  no candidate/task aggregate action error improves.
- This is direct evidence against using the present single-noise fixed-flow
  query scalar as a sufficient acquisition/selection proxy. It is not yet
  evidence that a distributional flow objective cannot improve actions: both
  metrics use one deterministic inference/noise authority. A small fixed
  multi-noise replication is the remaining cheap discriminator between robust
  surrogate mismatch and sampling variance before any differentiable sampler
  loss, further supervised training, or temporal-credit RL is authorized.
- The four-draw replication is result-blind at config SHA256
  `d436e17f2a5b91b8cdf22806e3967fc1f0f170590ba8a96692c610c7ef42212f`.
  It changes only inference-noise seeds to `[2026071835, 2026071935,
  2026072035, 2026072135]` on the same 48 anchors and immutable states. Robust
  mismatch requires worse mean action MSE for every candidate-task pair and at
  least three worsening draws per pair; otherwise the diagnosis is sampling
  variance. It consumes no rollout and cannot authorize Gate 0 or Writer.
- The replication completed rc 0 on clean `ccb2934` in 52.31 seconds; result
  SHA256 is
  `c1fc3ab448370590c34d5e234aa900377d63834318187be77b4ff9a9bc8eae4b`.
  Checksums pass, output is 144KiB, GPUs released, and GPU4/5 peak memory was
  5,635/4,317MiB with 80.06%/80.52% active-sample mean utilization.
- The aggregate status is `inference_sampling_variance_obscures_alignment`,
  but its pair-level evidence separates two mechanisms. Supervised rank-32
  LoRA worsens generated-action MSE on every one of four draws for task 3 and
  task 4, with mean reductions -1.901%/-3.062%; its flow-to-action mismatch is
  robust. The wider action expert worsens only the original draw, improves the
  other three, and has mean reductions +1.680%/+1.428%.
- Because the action expert's multi-seed teacher-forced action error improves
  while its paired closed-loop net gain is zero, generated-action query MSE is
  a better acquisition diagnostic than the current fixed-flow scalar but is
  not a sufficient behavioral Gate. The evidence supports a staged LoRA
  acquisition repair followed by unchanged closed-loop testing, and keeps a
  temporally credited task-local RL recovery as the next layer if action-
  aligned supervised acquisition still does not convert.
- The differentiable full-sampler mechanics check is result-blind at config
  SHA256
  `2dab3cd4399cd93daa26725b3c7ea50d07e555ee70f027ac53c622ac3bc10f25`.
  It binds exactly two legal task-3 source rows, the 10-step sampler, batch 2,
  unchanged 37-target rank-32 LoRA, zero optimizer/environment steps, and a
  71,680MiB peak-reservation ceiling. Its temporary script SHA256 is
  `1e5a8542...c5468` and must be copied into the result packet before cleanup.
- The full-sampler backward is mechanically cheap and valid. The timed Python
  command exited 0 with action/target shape `[2,50,7]`, loss 0.21559, finite
  LoRA gradient norm 0.52946 across all 74 trainable tensors, identical state
  digest before/after, 2,513MiB peak allocated, and 2,796MiB peak reserved.
  Result SHA256 is
  `b4e6fcefc5ba3d943980beea2fbe8cdeaa6e0a97069234a1be9ee0681cad4fe4`;
  copied source/config and telemetry checksums pass.
- The longrun state is conservatively `failed` with outer rc 1 even though the
  scientific command and GNU time report exit 0 and all post-command checksum
  lines are OK. A no-GPU reproduction of the shell cleanup skeleton exits 0;
  the ad-hoc wrapper is not a retained launcher, so record this as an isolated
  wrapper-exit residual and do not repeat scientific compute. It has no effect
  on loss, gradients, state, data authority, or Gate decisions.
- The next recovery is result-blind at contract SHA256
  `3d5b54be47c20bf29e356395f43ad2c9d43834b90eded994e68b141be0902246`.
  It changes the acquisition objective from flow matching to normalized
  generated-action MSE differentiated through the full pinned 10-step sampler
  and compresses the schedule horizon to the fixed 200-step ladder while
  preserving the prior peak/decay LR magnitudes.
  Tasks 3/4, source support 0--39, independent query 40--45, 37-target
  rank-32/alpha-16/dropout-0 LoRA, effective batch 64, seeds, augmentation,
  evaluator, drift cap, and Gate thresholds do not change.
- Four fixed inference-noise seeds make mean generated-action MSE the primary
  offline continuation metric; flow-query loss remains diagnostic. The staged
  maximum is 200 steps with candidates `0/1/5/10/25/50/100/200`. Step 1 tests
  batch-64 memory and atomic recovery. Both tasks must avoid action-MSE
  regression at step 5 and improve by at least 1% at step 10 before later
  segments. At least 2% improvement on each task with drift no greater than
  0.02 is necessary to open one source-development closed-loop check, but is
  never sufficient for Gate 0 or Writer authority.
- The canonical oracle fitter now owns both the original flow and new
  action-aligned objectives; no second trainer exists. With the recovery
  frozen, the one-time query/action controller and launcher are retired from
  the active source tree (1,065 lines removed); their immutable packets and Git
  provenance remain, and the reusable fixed-anchor action metric is retained.
- The action-aligned ladder is mechanically healthy but stops at step 10 under
  its result-blind rule. Task-3/4 step-1 peak memory is 50,697/50,657MiB with
  64/64 unique source rows and atomic recovery. Mean four-noise action-MSE
  reduction progresses from +0.074%/+0.109% at step 1 to +0.210%/+0.219% at
  step 5 and +0.881%/+0.928% at step 10. Every individual fixed-noise draw is
  positive at steps 5 and 10, so the gradient direction is consistent rather
  than sampling-sign noise.
- Step-10 drift remains only `1.19e-4`/`4.38e-5`, while flow-query changes are
  +0.065%/-0.033%. The latter divergence further supports generated-action MSE
  as the correct offline acquisition metric for this repair. However, both
  tasks miss the frozen 1% continuation floor; no step 25 or closed-loop
  surface is opened. Candidate SHA256 values are `c9b0d940...e5571` and
  `292781f4...acbb`.
- This is not a final negative about full-sampler action supervision: it tests
  one fixed 10-step, peak-2.5e-5, 200-step-maximum ladder and stops early by
  contract. It also does not establish zero-interaction utility. Together with
  the wider action expert's teacher-forced improvement but zero behavioral net
  gain, the cheapest next discriminating mechanism is genuine temporal-credit
  task-local LoRA RL, not more blind SFT duration, targets, ranks, or held data.

## Temporal-credit recovery is anchored to FPO++ but remains a bounded Gate-0 probe

- Primary-source inspection of [Flow Policy Gradients for Robot Control](https://arxiv.org/abs/2602.02481)
  and its [official FPO++ code](https://github.com/amazon-far/fpo-control) at
  commit `b80112be1e8362263c4cd176e7aef21a275ff1c6` confirms that the prior EMBER
  AWR and signed-ratio checks omitted the mechanisms needed to test temporal
  credit: a learned critic, GAE, sequential reward/done state, multiple matched
  conditional-flow samples, chunk-level ratios, and PPO trust-region updates.
  The official manipulation implementation uses a 512/256 critic and much
  larger vectorized/million-step budgets; those scale claims are not transferred
  to EMBER.
- The new source-only contract is frozen before outcomes at SHA256
  `0cfd1c74ced6b5cdc0e792d1af48555df6f2346527377cdc753ba46fc35955d2`.
  It is explicitly an FPO++-anchored SmolVLA compatibility probe, not a full
  method reproduction. Each of the four fixed task×initialization arms owns
  one independent task-local critic and the same complete task-local LoRA;
  base, Writer, encoders, and all shared state remain frozen. The critic sees
  detached frozen vision embeddings, normalized state, and chunk progress.
- Eight episodes form eight ordered action-chunk transitions per episode;
  terminal suffixes are masked instead of repeated as independent samples.
  Binary success is assigned to its action chunk, truncation is terminal, GAE
  uses discount/lambda `0.99/0.95`, and eight old/current flow losses share
  exact observation, action, time, and noise authority before their mean
  chunk ratio enters PPO clip `0.01`. A memory-safe two-pass gradient computes
  the exact surrogate coefficient without retaining eight SmolVLA graphs.
- Stage 8 is an early 10--30 minute evidence node with atomic LoRA, actor
  optimizer, critic, critic optimizer, and RNG recovery. Only healthy finite
  temporal credit and the existing drift/mechanics guards may continue the
  identical trajectories to stage 16; a behavioral pass stops early, and a
  stage-16 failure cannot expand the budget. This keeps the owner-mandated
  four-arm Gate-0 recovery distinct from the later Writer-only RL stage.
- The first real-model smoke failed before gradient computation because the
  pinned SmolVLA policy exposes two real cameras plus one configured empty
  camera slot. The critic contract correctly specified two 960D visual pools,
  padded 32D state, and progress (`1953D`), while the implementation initially
  concatenated the masked empty slot and produced `2913D`. This was an
  implementation error, not evidence about temporal credit or Gate 0.
- Clean commit `8237bed` fixes only that boundary: trailing policy-declared
  empty slots must have all-false masks and are excluded; an observation-bearing
  slot fails closed. The retry completed rc 0 with 64 unique source rows,
  `[64,1953]` features, `[64,8]` flow losses, actor/critic gradient norms
  `0.14634/1.18642`, unchanged LoRA state, zero optimizer/environment steps,
  and 5,268MiB peak reserved memory. This validates mechanics and memory only
  and does not supply a policy or Gate outcome.

## The bounded temporal-credit recovery is mechanically valid but behaviorally negative

- Stage 8 (`gate0_temporal_credit_stage8_20260719_083749`) completed rc 0 in
  4m46s. Its development paired gains were zero-init `[-1,-1]` and
  supervised-init `[0,-1]`; finite/nondegenerate temporal credit, zero
  nonfinite values, 0.01077 maximum drift, and exact actor+critic recovery
  authorized only the frozen exact resume.
- Stage 16 (`gate0_temporal_credit_stage16_20260719_084442`) completed rc 0 in
  5m01s and terminates this recovery. On the unchanged eight-state source
  development slice, zero-init RL is task3/4 `2/8,3/8` versus frozen base
  `3/8,3/8`; supervised-init RL is `2/8,4/8` versus supervised LoRA
  `2/8,4/8`. Thus paired net gains are `[-1,0]` and `[0,0]`; no task shows a
  positive gain and neither initialization passes. Task4 supervised-init is
  one success above matched zero-init, but only because both return to their
  distinct initial success counts; it is not evidence that either RL arm
  improved.
- Mechanics do not explain away the negative result. All checksums pass,
  temporal credit is healthy, nonfinite count is zero, saturation is at most
  0.00146, maximum drift is 0.01232, and eight videos decode. Query flow loss
  still improves 5.67%/4.04% for supervised initialization but only
  +0.117%/-0.038% for zero initialization, while closed-loop utility does not
  improve. This continues the offline-surrogate/closed-loop mismatch rather
  than resolving it.
- Physical operator deltas rule out a no-update explanation. At episode 16,
  zero-init RL produces operator norms 0.0701/0.0743 for tasks 3/4. Relative to
  the mature supervised operator norm 0.718/0.700, supervised-init RL adds
  physical increments of norm 0.0811/0.0816 (`11.3%/11.7%`) with near-orthogonal
  or mildly opposing cosine. These are real but behaviorally unhelpful updates.
- Primary-source reconciliation narrows the next mechanism question. The
  official FPO++ manipulation code at commit
  `b80112be1e8362263c4cd176e7aef21a275ff1c6` uses the same old-minus-current
  conditional-flow loss ratio, eight samples, chunk PPO, critic and GAE, so the
  central ratio sign is not missing. However it trains the critic alone for the
  first iteration by default, uses `lambda=0.99`, collects roughly
  `30×1600` environment steps per iteration, and runs up to five million
  timesteps; the current probe trained actor and a zero-output critic together
  from its first ~2--3k-step round and stopped after two rounds. Therefore this
  is negative evidence for the declared 16-episode compatibility probe, not a
  final negative about adequately trained ordinary LoRA RL. Any next recovery
  must be separately frozen, start from the immutable initializations, and
  correct the critic/data-acquisition mismatch rather than silently resuming to
  episode 24.

## Critic-only warmup is the next frozen compatibility discriminator

- Before any new outcome, the official FPO++ implementation at commit
  `b80112be1e8362263c4cd176e7aef21a275ff1c6` identified one narrow mechanism
  correction: its default manipulation schedule trains the value function for
  the first iteration before policy updates. The new EMBER contract therefore
  changes only that ordering, aligns GAE lambda to `0.99`, and removes the
  previous externally added Gaussian action noise. It does not claim a full
  FPO++ reproduction and does not import its million-step scale.
- `configs/gate_zero_task_local_rl_critic_warmup.toml` is frozen at SHA256
  `51fc9a009d0fa93476ba47a22d86e95a5d89f32182057843c3129e4147725a8a`.
  Tasks 3/4, fresh source slice, base/SFT starting vectors, all 37 LoRA targets,
  rank 32/alpha 16/dropout 0, evaluator, query/drift safeguards, success Gate,
  four matched arms, and source/validation/held boundaries are unchanged. It
  starts fresh rather than resuming the behaviorally negative episode-16
  state.
- Episode 8 is a critic-only mechanics boundary: all four LoRA actors must be
  bitwise unchanged and have zero actor optimizer steps while each critic has
  finite positive updates. Later nodes 16/24/32 enable ordinary task-local
  LoRA RL. The source-development continuation ladder is result-blind and
  bounded; a candidate stops immediately, episode 24 needs a predeclared
  positive trend to open 32, and there is no episode 40.
- This remains the owner-approved Gate-0 four-arm recovery: frozen base,
  supervised LoRA, zero-init LoRA plus ordinary task-local RL, and supervised-
  init LoRA plus identical RL. If only an RL arm succeeds, the supported claim
  is useful task-local LoRA RL capacity, not supervised zero-interaction
  utility. Supervised-init helps adaptation only if it beats matched zero-init.
  No Writer parameter is present or updated here.
- The later EMBER stages remain scientifically distinct: supervised Writer
  cold start updates Writer through independent query behavioral loss;
  Writer-only RL freezes base and treats generated LoRA as a functional output
  while reward updates Writer only; ordinary LoRA RL freezes Writer/base and
  optimizes the task-local LoRA in place; adaptation-aware source meta-outer
  learning updates Writer through the inner LoRA adaptation. These stages must
  not be merged into one result.
- The clean-revision real-model critic-warmup smoke passed without simulator
  access in 28.95 seconds. All 64 legal task-3 source rows were unique; round 0
  produced 40 critic optimizer updates with minimum gradient norm 0.835 and a
  changed critic state, while the complete LoRA actor remained exactly
  unchanged, actor optimizer state stayed empty, actor steps were zero, and
  actor gradient norm was exactly zero. Peak reserved memory was 4,128MiB,
  leaving far more than the required 10GiB headroom. Result SHA256 is
  `91db643019a79d905b2878f6411484ccf68e49a58bb66f3dd4a1419019963c07`;
  packet checksums pass. This validates the intended scheduling mechanic only,
  not useful behavior or Gate 0.

## Critic warmup does not convert the matched RL probe into useful behavior

- The same clean `2d103d6` four-arm trajectory completed episodes 8, 16, and
  24 with rc 0 in 3m24s, 5m02s, and 5m16s. Episode 8 exactly reproduced all
  starting success vectors with zero actor updates; episode 16 enabled the
  actor and was still negative/flat; episode 24 terminated under the frozen
  trend rule. No episode 32 was run.
- The terminal episode-24 paired gains are zero-init task3/4 `[0,0]` and
  supervised-init `[0,-1]`. Thus neither initialization supplies a useful
  closed-loop update, supervised initialization does not help matched RL, and
  Gate 0/Writer remain false. This is a negative for the declared 24-episode
  critic-warmup compatibility recovery, not for sufficiently scaled LoRA RL or
  EMBER as a whole.
- Mechanics remain healthy: no nonfinite value or saturation, maximum drift is
  0.01254, critic explained variance generally rises, actor and critic gradients
  are finite, and approximate KL stays far below the 0.1 limit. The LoRA actor
  is not inert: episode-8-to-24 physical operator increments have norms
  `0.0586/0.0644` for zero-init task3/4 and `0.0732/0.0841` for supervised-init.
  These real updates do not improve the fixed development behavior.
- Result SHA256 is
  `986887261b47b9d4dc55ec630f8c914b60d2fbd247e33c1d98c82669e2a8b1a8`.
  The launcher validates every retained candidate/recovery/round/stage file,
  the terminal result, gallery, latest four bounded videos, and three telemetry
  files. Across stages, peak memory is about 19.3GiB on rank0 and at most 6.1GiB
  on other ranks; output is 193MiB and all GPUs release.
- Training-slice success rates oscillate strongly across the three disjoint
  eight-state rounds, while fixed development behavior stays flat or worsens.
  This leaves a precise ambiguity: the reward update may fail even on seen
  support states, or it may fit sampled support behavior without transferring
  to the development slice. More blind interaction cannot distinguish these.
- The next result-blind diagnostic is frozen at config SHA256
  `f539b7376dd1e265076941d7b45022934802f2931bdb54b866b9b97e1a533909`
  with temporary source SHA256
  `1a410a6a5b695d1dcc9d024a9ef90ed36238d078303a5b6cf782213e40c737c4`.
  It loads each immutable episode-24 recovery and replays exact round-0 source
  init states/seeds with zero updates. A positive paired arm isolates a
  coverage/generalization gap; no positive arm isolates credit/optimizer
  acquisition. It cannot pass Gate 0, authorize Writer, access validation or
  held data, or change task/seed/threshold/checkpoint.

## Support replay rejects a coverage-only account of the RL negative

- The frozen support-replay discriminator completed rc 0 on clean `0804f21`
  with status `support_replay_no_improvement`. It loaded the four immutable
  episode-24 states, made zero optimizer updates, and replayed only the exact
  round-0 task3/task4 source init states 8--15 and seeds 6200--6207. Supervised
  task3/task4 paired net wins are `[0,-1]`; zero-init task3/task4 are `[-1,-1]`.
  No arm improves even on the support slice used to acquire its first reward
  update.
- This separates the remaining failure from support-to-development coverage:
  the declared 24-episode acquisition is behaviorally ineffective on its seen
  support surface as well as on the fixed development surface. It does not
  prove that sufficiently resolved or scaled ordinary task-local LoRA RL is
  ineffective, but it forbids a blind episode-32 continuation of this path.
- Result SHA256 is
  `7e92b745b53442d0df2b3e36b068402b244b17e7f0a750e053f60510d59c414e`;
  config SHA256 is
  `f539b7376dd1e265076941d7b45022934802f2931bdb54b866b9b97e1a533909`.
  All packet checksums pass, the four trainable states and actor optimizer
  states remain exact, and Gate 0/Writer/validation/held remain unopened. An
  initial checksum invocation from the repository cwd could not resolve the
  packet-relative paths; the immediate packet-directory invocation passed all
  entries. This is an operator-cwd residual, not an artifact defect.

## Replay action coordinates are mechanically correct

- LeRobot rollout stores the policy-postprocessed environment action. In the
  pinned LIBERO path the environment postprocessor is empty, so this is exactly
  the action consumed by the simulator and later copied into replay.
- The pinned checkpoint's SmolVLA action normalizer and unnormalizer use the
  same seven-dimensional MEAN_STD tensors; every action/state tensor is
  bit-identical between the two processor artifacts. A direct numerical
  unnormalize-then-normalize probe has maximum absolute error below `7.2e-7`
  over normalized magnitudes through 10.
- `build_balanced_replay_batch` chunks those stored environment actions; the
  source preprocessor maps them back into normalized coordinates, and SmolVLA
  applies `action_is_pad` before reducing the flow loss. Action dimensionality,
  padding, and normalization therefore cannot explain the observed no-gain
  replay. The remaining diagnosis is reward-credit/optimizer acquisition, not
  an action-coordinate implementation failure.

## The next discriminator is training credit resolution, not blind scale

- The official FPO++ manipulation reproduction at code commit
  `b80112be1e8362263c4cd176e7aef21a275ff1c6` declares
  `n_action_steps=16`, `data_collection_steps=1600`, and 30 environments for
  the main benchmark. EMBER's preceding compatibility probes preserved
  SmolVLA's 50-action execution queue, so each 400-step episode produced only
  eight reward-credit transitions. This is a concrete compatibility gap after
  the support-replay and action-coordinate branches were eliminated; it is not
  evidence that the full official scale should be copied.
- The outcome-free contract
  `configs/gate_zero_task_local_rl_horizon_credit.toml` is frozen at SHA256
  `491d031565409962cfb96cea09f6ac73ae636a1fe87a14aeb441b18c2d15e05b`.
  Training executes 16 actions per policy inference, records 25 ordered
  transitions per 400-step episode, and stores each transition in SmolVLA's
  unchanged 50-slot model action tensor with only the 16 executed actions
  unmasked. Thus reward is not assigned to unexecuted future actions.
- This is a single training-acquisition change. The canonical reporting
  evaluator continues to execute 50 actions; tasks 3/4, all 37 LoRA targets,
  rank 32/alpha 16/dropout 0, 1,485,312 parameters, zero/SFT starts, optimizer,
  collection seeds, development/fresh seeds, query/drift safeguards, success
  thresholds, and source/validation/held boundaries are unchanged. No Writer or
  shared parameter is present.
- The recovery has only nodes 8 and 16. Node 8 is critic-only and must preserve
  the actor exactly. A healthy node 8 may exact-resume once; node 16 must either
  meet the original two-task positive-improvement rule or stop. There is no
  episode-24 continuation. A real-model, no-environment smoke must first prove
  200-row replay mechanics, 16/50 masking, memory-bounded inference, finite
  critic learning, actor identity, and 10GiB A100 headroom. It is mechanics
  authorization only, not Gate-0 evidence.

## Horizon-credit mechanics pass with large A100 headroom

- The clean `b5aaaea` real-model smoke passed in 31.42 seconds on one A100.
  It built 200/200 unique deterministic replay identities, scoped the policy
  from 50 to 16 executed actions and restored 50 afterward, preserved the
  50-slot model action tensor with every suffix after step 16 masked, and
  obtained finite real-model flow losses of shape `[16,8]`.
- The critic-only round made 130 finite optimizer updates with minimum gradient
  norm 0.01347. The complete LoRA actor remained exact, actor optimizer state
  remained empty, actor updates and maximum actor gradient were zero, and all
  200 transitions were temporally healthy. Peak allocated/reserved memory was
  2,858/4,004MiB, far above the required 10GiB headroom.
- Result SHA256 is
  `29528c5f8a4f2fd1c570e74c5c85e8a5e6ad4baf1c7239c00842579d215b844a`;
  smoke source SHA256 is
  `64c14b4cb0e43a765c503990c02f6ff0509c06c3f69a6c862a880d8eacd6f78d`.
  Result, copied source, frozen config, and GPU telemetry checksums all pass;
  GPU4 returned to 0MiB.
- Two earlier attempts remain mechanics failure packets, not negative scientific
  evidence. The first failed before model load because telemetry precreated the
  output directory; the second stopped before flow or optimizer work because a
  support-loader surrogate repeated provenance keys across batches. The
  recovery allowed a telemetry-only directory and appended deterministic smoke
  slots while retaining source identity. Production rollout key construction,
  trainer behavior, configuration, Gate, and scientific surfaces were not
  changed. The passed smoke authorizes only stage 8, not Gate 0 or Writer.

## Horizon-credit stage 8 preserves the actor and opens one terminal check

- Clean `ac9cf2f` stage 8 completed rc 0 in 3m34s on GPUs 4--7. Every arm
  collected 200 ordered replay rows with 16-step execution, healthy temporal
  credit, zero saturation, 90 finite critic updates, exact actor identity, and
  zero actor optimizer updates. The four source training slices each happened
  to score 6/8; this is acquisition telemetry, not the fixed Gate metric.
- Fixed development paired gains are `[0,0]` for both zero-init and supervised-
  init families, exactly as required by a critic-only warmup. Status is
  `horizon_credit_warmup_complete_continue_to_16`; Gate 0, Writer, validation,
  and held remain false/unopened. No policy-quality inference follows from the
  warmup result.
- Stage-result SHA256 is
  `a3b93ebf04fec8ab0f6ee7a3db7801cb80733d51d79a3079f2c46113a26a1b0d`.
  All 15 JSON files parse; four recovery and candidate packets validate; all
  four videos decode; telemetry checksum passes. Output is 99MiB. Peak memory
  is 19,266MiB on GPU4 and at most 5,121MiB on GPUs5--7; every GPU released.
- The sealed decision allows one exact resume to stage 16. That node enables
  the actor and is terminal: it either meets the unchanged two-task positive-
  improvement rule and becomes a candidate for a separately frozen fresh Gate,
  or records a bounded negative and stops. Stage 24 remains impossible.

## Horizon-resolved credit still fails the fixed two-task behavioral Gate

- The exact-resume stage 16 completed rc 0 in 7m58s. All four arms had healthy
  temporal mechanics, finite gradients, zero nonfinite/saturation events, and
  bounded KL below `1.7e-4`; every artifact checksum and video decode passed.
  This is therefore a behavioral negative, not an implementation failure.
- On the unchanged fixed development slice, zero-init task3/task4 paired net
  wins were `[-2,+1]`; supervised-init wins were `[0,-1]`. Neither family
  improved both tasks, and supervised initialization did not improve the
  matched zero-init trajectory. The isolated task4 `+1` cannot be selected
  against task3's `-2` under the sealed rule.
- The terminal result is
  `task_local_rl_early_check_not_supported`, SHA256
  `771eb3b9f563492299d7424ef3a63c77003c322f3fb233c555a46f089ff7f496`.
  Gate 0 and Writer remain unauthorized; there is no selected checkpoint and
  no validation, held, locked-report, task, seed, or threshold change.
- The supervised arms still reduce independent query loss by 5.76% and 4.00%
  while closed-loop behavior is flat/worse. The cheapest remaining
  discriminator is a zero-update replay of each immutable step-16 actor on its
  exact round-1 horizon-16 training slice. Improvement there with development
  failure indicates coverage/generalization; no improvement there isolates
  reward-credit/optimizer acquisition even on seen horizon support. This new
  diagnostic cannot itself pass Gate 0 or justify more interaction.
- The outcome-free replay contract is frozen at SHA256
  `7e676c52f551d3624759448fd34265ecb00bc3c2ee56841c4bfdc6d84cd5a9cb`.
  It binds the terminal result and all four step-16 recovery manifests, uses
  only source round 1 (init states 16--23, seeds 6208--6215, policy RNG
  2026071961), retains horizon 16 and model chunk 50, and performs exactly zero
  policy, critic, optimizer, or Writer updates across 32 episodes.

## Horizon support replay finds partial acquisition but no stable generalization

- The four-rank replay completed rc 0 in 2m01s from clean `bff88bd`; result
  SHA256 is
  `4a0c13a00bb2692df048eec8426c9dc4980582d5bbd2bd70e178e327fa65f7ef`.
  All result/source/config/telemetry checksums pass, every arm preserves its
  step-16 trainable state and actor-optimizer entry count, and GPUs 4--7
  released. Peak usage was 16,679MiB on GPU4 and 4,317MiB elsewhere.
- On the exact round-1 training slice, supervised task3/task4 paired net wins
  are `[-1,+1]`; zero-init task3/task4 are `[+1,0]`. Thus two individual arms
  improve by one paired episode, one is unchanged, and one regresses. The
  frozen classification is
  `horizon_support_replay_improves_but_development_does_not`.
- This rejects a claim that the horizon-resolved optimizer cannot move behavior
  on any seen support, but it does not establish robust acquisition: neither
  initialization family improves both tasks, and the already frozen
  development result remains `[-2,+1]`/`[0,-1]`. The evidence is best read as
  partial support acquisition plus unstable source-slice generalization, not
  as useful task-local RL or supervised-init advantage.
- Gate 0, Writer, selected checkpoint, fresh Gate, validation, held, and locked
  report remain closed. A new result-before-outcome recovery may test only the
  primary-source-supported data-coverage gap with additional disjoint source
  training slices and the same scientific contract; it must not append to the
  terminal packet, relax the two-task Gate, or treat either `+1` arm as a
  selected candidate.
- That recovery is frozen before outcome at SHA256
  `72e4f13e193aa63e96ea90395f37e2954030a534c396998067a54a74e4d9f241`.
  It uses the same horizon-resolved PPO/critic implementation and creates no
  new trainer. A fresh trajectory covers source init states 8--23 at nodes
  8/16, then adds 24--31 at node 24 and conditionally 32--39 at node 32; the
  fixed development surface stays 40--47.
- Stage 16 is a reproducibility/mechanics boundary and may continue exactly
  once to the first genuinely new coverage node 24. Node 24 opens 32 only if
  one initialization has positive aggregate paired net gain and neither task
  is worse than -1; node 32 is terminal. A passing node still only selects a
  hash-bound development candidate for a separately frozen fresh Gate. This
  is not permission to append to the prior terminal output or to claim a full
  FPO++ reproduction.

## Coverage-recovery warmup exactly preserves the actor

- Fresh node 8 completed rc 0 in 3m36s on clean `4b8dde6`. All four source
  arms collected 200 ordered horizon-16 replay rows, made 90 finite critic
  updates, retained healthy temporal mechanics and zero saturation, and kept
  every LoRA actor plus its empty optimizer state exact.
- Fixed development paired gains are `[0,0]` for both zero-init and
  supervised-init, as required for critic-only warmup. Status is
  `horizon_coverage_warmup_complete_continue_to_16`; stage-result SHA256 is
  `219eff172d920fc93732032f1e33e548deb9ae430a5c9a88d33176a8b8f289dd`.
  This contains no policy-quality evidence and cannot pass Gate 0.
- Fifteen JSON files parse; all four candidate and recovery artifacts validate;
  four bounded videos fully decode; telemetry checksum passes. Peak memory was
  19,267MiB on GPU4 and at most 5,122MiB elsewhere, leaving over 61GiB
  headroom. The only authorized next action is same-output exact resume to
  node 16; node 24 remains closed until that sealed decision completes.
- Node 16 then exact-resumed and completed rc 0 in 8m03s. Its fixed development
  gains exactly match the independent predecessor trajectory: zero-init
  `[-2,+1]`, supervised-init `[0,-1]`. This independently confirms that the new
  contract did not alter the first two rounds, evaluator, or RNG/data identity.
- Stage-result SHA256 is
  `3d4bf5a1e40e4ebf6bfc14ee896f0aa30991fd6c75770f279dccc9bd585e5050`;
  all four step-16 candidates/recoveries, four new videos, 24 JSON files, and
  telemetry validate. Mechanics/temporal credit remain healthy and no
  checkpoint is selected. The contract now opens only node 24, whose source
  init states 24--31 are the first new coverage evidence.

## Coverage node 24 closes the n=8 trajectory without a Gate decision

- Exact-resume node 24 completed rc 0 in 8m14s from clean `9c9f239` and stopped
  atomically. Supervised-init task3/task4 paired gains are `[0,0]`; zero-init
  gains are `[0,-1]`; the supervised-init advantages are `[-1,+2]`. Status is
  `task_local_rl_early_check_not_supported`, but this classification is only
  for the sealed small-sample development trajectory.
- Stage-result SHA256 is
  `9b738193312ac2aed527b075ca88c0072e1cb9807445998eb570f5d385340a94`.
  All four candidates and four schema-3 recovery packets validate, 34 JSON
  files parse, all four bounded videos decode, telemetry checksum passes, and
  GPUs 4--7 released. The output occupies 196MiB; observed peak memory was
  16,683/6,485/6,515/6,516MiB.
- The owner correctly classified n=8 as smoke: one episode is 12.5 percentage
  points, so observed one- or two-win changes are statistically ambiguous.
  Node 24 cannot pass or reject Gate 0, select a checkpoint, authorize Writer,
  or justify node 32. It remains useful evidence that the custom pilot is
  mechanically executable and that small-slice closed-loop conversion is
  unstable.

## Gate -1 closes with residuals and Gate 0 gains a repaired evidence contract

- The immutable action-hidden-video recovery is not rewritten: ordered and
  wrong-video accuracy are 19/24 (0.7917), bidirectional paired correctness is
  15/24, the original content threshold remains 0.80, and drop-last sensitivity
  remains a residual. Explicit owner authority now records Gate -1 as passed
  with residuals and forbids more compute solely to cross 0.80. It no longer
  blocks Writer after Gate 0.
- Before any new Gate-0 outcome, the active source-only contract requires at
  least 32 paired rollouts/task/arm over multiple policy RNG seeds; per-episode
  records and paired bootstrap/exact intervals; and at least two, preferably
  three, independent training seeds. Task3/task4 remain development only.
  Confirmation must be disjoint by source task or physical init state and
  hash-bound before LoRA outcomes, preferably across two to four distinct
  primitive source tasks selected from base competence/headroom only.
- The result-before-outcome contract is
  `configs/gate_zero_evidence_repair.toml`, SHA256
  `0196419d7abc6132890248da1c332b451767db45aeef2c96f3ab989eb1c0aa4e`.
  Its source-derived confirmation candidate IDs and deterministic physical
  init-state partition are fixed without reading policy outcomes.
- Training and primary evaluation now both execute horizon 16; horizon 50 is a
  separate deployment-robustness report. Binary success remains primary, with
  legal source-only grasp/object-region/drawer/progress/time diagnostics. Drift
  is reported from both frozen base and each arm's own initialization, and
  replayed transitions are never counted as new independent interaction.
- Code inspection shows the existing temporal-credit pilot averages eight
  flow-sample losses before one transition ratio. Its accurate scope is
  `custom chunk-level flow-loss PPO pilot`; the historical
  `flow_sample_group_size` config field did not drive runtime semantics. The
  active contract removes that field. A new bounded faithful core implements
  per-flow-sample/group-size-one ratios, MSE-preserving modified Huber loss,
  old-loss/log-ratio clamps, PPO trust region, and horizon 16. This core is
  required before an ordinary task-local RL negative claim, but does not imply
  the paper-scale FPO++ budget or change the common LoRA space.

## Source-base difficulty audit is frozen before confirmation selection

- The new pre-outcome contract is
  `configs/gate_zero_base_difficulty_audit.toml`, SHA256
  `ae73a4b0728e0ab8a3f6a018952b14280b697a150656f9e9c4cf47d2a9836443`.
  It derives nine candidates solely from the resealed source-task factor table;
  no base, LoRA, validation, held, or locked outcome was used to form that pool.
- Each candidate receives exactly 32 frozen-base rollouts on the `train` role
  of its deterministic SHA partition, in four batches of eight with four
  policy RNG seeds and unique physical init-state hashes. Primary execution is
  horizon 16. The audit totals 288 source episodes and retains at most one
  bounded video/task.
- The post-audit rule is already executable and deterministic: eligible tasks
  need at least four successes and eight failures, are ranked by absolute
  distance from 0.5 success then task ID, and at most four distinct primitive
  signatures are selected. If fewer than two qualify, the packet records a
  bounded-recovery requirement instead of failing mechanically or consulting
  LoRA outcomes.
- The runner is an orchestration layer over the existing upstream evaluator,
  not a second evaluator or trainer. It explicitly sets and verifies the
  physical pre-reset state counter, binds per-state SHA/evaluator/policy seeds,
  scopes action execution to 16 while preserving the model chunk of 50, and
  checks the base checkpoint manifest and prior source-competence result.
- The first four-GPU launch failed before any reset, rollout, reward, or policy
  outcome. All ranks loaded the base, created their first lazy vector env, then
  raised because LeRobot's `_LazyAsyncVectorEnv` forwards `call/get_attr` but
  not Gymnasium's `set_attr`. The immutable failure root is
  `source_base_n32_20260719T124341Z`; longrun main rc is 1 and all GPUs released
  after 20.22 seconds. This is an implementation failure packet only.
- The narrow repair materializes the existing lazy wrapper through `get_attr`,
  calls `set_attr` on its owned Gymnasium vector env, and immediately reads the
  counters back through the public `call` path. It does not alter the partition,
  tasks, seeds, evaluator, horizon, checkpoint, or selection rule. A matching
  lazy-wrapper regression test now accompanies the direct-vector test.

## Base-only audit freezes four disjoint confirmation tasks

- Actual-env no-policy validation passed before relaunch: task6 physical
  counters `[49,9,21,14,5,36,32,41]` reset exactly to each value plus eight,
  with zero policy actions and zero reward reads. Recovery1 then completed rc 0
  from clean `3cbb975` in 645.34 seconds on GPUs 4--7, consuming exactly 288
  source episodes at horizon 16.
- Frozen-base successes are task6 `4/32`, task9 `27/32`, task16 `17/32`, task20
  `32/32`, task23 `0/32`, task33 `13/32`, task39 `24/32`, task46 `4/32`, and
  task63 `0/32`. The frozen competence/headroom filter retains
  `[6,16,33,39,46]`; ranking by distance from 0.5 then task ID selects
  `[6,16,33,39]` with distinct `open`, `stack`, `close`, and `turn_off`
  signatures. No LoRA outcome participated.
- Canonical result SHA256 is
  `240de3134b3bb28a3fc57059b18d0baffb8a8b2ce5bdb9274b2a68848a01dd61`;
  confirmation selection SHA256 is
  `a4d57cf9737aee8b824c93d972ec22e5376731a4f25fe3eedbeea9bd58e9f670`.
  The selected partition hashes are task6 `f857cb29...e4eae`, task16
  `42559521...ebcd`, task33 `3110a7a6...c5de`, and task39
  `cf104892...065a2`.
- All checksums including telemetry pass; all JSON parses; 288 episode rows
  contain 32 unique state indices/hashes per task and four policy RNG seeds;
  nine videos decode with 68--400 frames; gallery/latest are valid; output is
  2.3MiB. Peak memory/utilization was GPU4 20,021MiB/100% and GPUs5--7 only
  1,752MiB/73%, 1,752MiB/69%, and 1,752MiB/65%. All GPUs released.
- The asymmetric memory plus three-rank idle tail is systems evidence, not a
  scientific issue. It is consistent with all async EGL workers selecting the
  first visible render device and whole-task assignment leaving rank0 with
  three tasks; this is an inference to verify before later rollouts. Do not
  rerun this valid audit merely to improve utilization. Future work should bind
  EGL per local rank and batch-balance tasks without changing episode identity.
- This result freezes confirmation identities only. It does not establish a
  useful LoRA update, Gate 0, Writer utility, validation performance, or held
  performance. The next contract must keep tasks3/4 as development and use the
  selected tasks/states as disjoint confirmation with >=32 paired episodes,
  multiple policy RNG seeds, >=2 training seeds, and h16 primary/h50 robustness.

## Post-selection contract removes two remaining statistical ambiguities

- The frozen-base arm has no training process and therefore no training seed.
  Repeating identical base rows under each candidate seed would be
  pseudoreplication. The active contract stores base once and requires each
  trainable arm to cover two independent sealed training seeds on the same
  paired evaluation episodes; the final statistics interface must preserve
  that unit structure.
- `configs/gate_zero_matched_evidence.toml` (SHA256 `625db578...e0038`) binds confirmation tasks
  `[6,16,33,39]`, their result-blind state partitions, mature 37-target
  rank-32/alpha-16/dropout-0 LoRA, h16 primary/h50 robustness, and the source
  access boundary before any new task-specific LoRA outcome.
- The prior temporal-credit runtime always averaged eight CFM losses before one
  PPO ratio. A loss helper existing in the tree was not sufficient evidence
  because the trainer did not call it. Runtime dispatch now selects the faithful
  per-flow-sample/group-size-one modified-Huber path explicitly for new matched
  runs; historical configs load with an explicit custom chunk-mean label.
- The two required training seeds are `2026071830` and `2026072030`; a third
  seed `2026072130` is result-blindly reserved only for a predeclared ambiguous
  two-seed outcome. This is replication, not a hyperparameter or favorable-seed
  search.

## First faithful matched-RL segment is mechanically healthy but statistically ambiguous

- Long-run `gate0_matched_dev_seed2026071830_ep16_20260719_134125` completed
  rc 0 from clean `77e15c7` in 11m03s on GPUs 4--7. Each of the four matched
  task/initialization arms consumed 16 source interaction episodes; the first
  eight formed critic warmup and the second eight enabled 80--90 finite actor
  updates. Every runtime ratio is genuinely per-flow-sample, temporal-credit
  checks pass, saturation/nonfinite counts are zero, and all four atomic
  recovery manifests and file hashes validate.
- The same horizon-16 development slice compares each updated arm with its own
  initialization. Supervised-init RL changes task 3 from 7/8 to 6/8 and task 4
  from 5/8 to 4/8; zero-init RL changes task 3 from 6/8 to 4/8 and task 4 from
  3/8 to 5/8. These paired gains `[-1,-1]` and `[-2,+2]` are all inside the
  predeclared n=8 ambiguity region and cannot pass or reject Gate 0.
- Independent source-query reductions are +5.70%/+3.90% for supervised-init
  RL and approximately -0.04%/-0.05% for zero-init RL. This is enough to rule
  out a mechanical no-update run, but not enough to claim closed-loop utility
  or initialization benefit. Because mechanics are healthy and the legal
  signals are mixed rather than decisively harmful, the frozen ladder permits
  exactly one same-trajectory resume to 24 interactions/arm before the next
  review; it does not authorize node 32, confirmation, Gate 0, or Writer.
- The retained packet is 147MiB. Stage-result SHA256 is
  `cd786d189e2a23f8511e17805a75e0383d39241899c4e6e5376eb0b43321d26c`;
  the compact gallery contains one current-policy video per arm and Trackio
  project `EMBER_gate0` contains the live metrics. Peak memory on GPUs 4--7 was
  19,281/6,431/6,447/6,447MiB; all four reached 100% utilization and were
  released to 0MiB. The existing EGL/render asymmetry remains an efficiency
  residual, not a reason to rerun or delay this scientific trajectory.

## Faithful seed-1 trajectory ends with a promising ordinary-RL smoke, not Gate evidence

- The same exact-resume trajectory completed node 24 in 9m47s and node 32 in
  10m48s. Node 24 changed supervised-init task3/4 by `[-1,0]` paired wins and
  zero-init by `[+1,+2]`, satisfying only the predeclared continuation trend.
  Node 32 changed supervised-init by `[0,+1]` and zero-init by `[+1,+3]`.
  Mechanics, temporal credit, per-flow-sample ratios, drift (<0.02), and finite
  updates remain valid. The 32-interaction states are the hard end of this
  trajectory; no node 40 or extra interaction is permitted.
- At node 32 the supervised-init versus zero-init current-policy paired
  advantage is `[0,0]`. Supervised-init query reductions remain +5.30%/+2.00%,
  while zero-init query changes are -0.24%/-0.39%. Thus the lowest-cost reading
  is a promising ordinary zero-init LoRA RL oracle smoke, with no evidence yet
  that the supervised initialization helps RL. Because every closed-loop cell
  is still n=8, none of these values can pass/reject Gate 0 or select Writer.
- Node-32 stage SHA256 is
  `9f5aa5e3da4966ccbd40c3afbbd9befba4b02e6faf711bb421f88ba717e42aed`.
  The complete trajectory is 203MiB; all recovery hashes/JSON/telemetry pass,
  four current-policy gallery videos decode, and all GPUs released. Across the
  three segments it consumed 32 source interactions per arm (128 total) plus
  three repeated 8-episode current/initial development comparisons per arm;
  replay epochs are not independent samples.
- A node-24 publishing bug labeled the valid `continue_to_32` status as a
  terminal file. The attempted node-32 launch failed rc 2 before model load or
  outcome. Its 49-byte log, state, duplicate result, and checksums are preserved
  under `failure_packets/nonterminal_publish_ep24_20260719T140940Z`; the result
  is byte-identical to `stage_results/000024.json`. One tested status fix then
  exact-resumed successfully without changing the scientific contract.
- The matched config's `active_training_seed` was previously only a bound
  replicate label: seed-1's fixed critic/minibatch/flow/policy streams are the
  reference mapping, so its numerical behavior is unchanged. Before seed 2,
  the canonical loader now derives those training-only streams by the exact
  delta from predeclared seed `2026071830`, records the master seed in recovery
  and stage provenance, and leaves evaluation RNG unchanged. This prevents a
  false duplicate replicate without adding another trainer.

## Both required RL checkpoints are frozen; only n>=32 performance is admissible

- Required training seeds `2026071830` and `2026072030` both reached the
  predeclared step-32 hard limit through exact resume. The second trajectory's
  final stage SHA256 is `f15eb2c922167cd8ea648f82bb678f435298d8f2818777daf6d41e84512f722a`;
  its four recovery manifests bind the correct training seed and unchanged
  contract, all internal hashes and telemetry pass, and GPUs 4--7 released.
  No further training interaction is authorized for either trajectory.
- All prior n<32 performance values are quarantined as historical provenance.
  They may not drive checkpoint, task, seed, threshold, continuation, or
  interpretation, and no future owner-facing performance packet may be emitted
  before every required cell reaches n=32.
- `configs/gate_zero_formal_development_evaluation.toml` (SHA256
  `1ad045abba630049a68a1b02ed5f8121c087aff54721117e69899d10c173910c`) freezes the first
  admissible comparison before those outcomes: tasks 3/4, existing step-32
  checkpoints, four policy RNG seeds over the paired physical states, h16
  primary plus h50 robustness, per-episode rows, paired bootstrap/exact
  intervals, and both independent training seeds. Frozen base and fixed
  supervised LoRA are evaluated once with `training_seed=null`, preventing
  pseudoreplication while pairing them against both trained replicates.
