"""Finite CPU-inspectable registration/handoff owner; retire after this batch.

No evaluator, environment factory, model loader, queue, or command launcher lives
here. Root integrates the typed hooks listed in proposed_readout_notes.json.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open

from ember.lora import expected_lora_state_shapes, inject_task_lora, task_lora_state_dict
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_eval.trajectory_capture import record_replan
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import libero_policy_input
from ember.pi05_source_checkpoint import read_json
from ember.static_task_lora import FrozenStaticTaskLoRAAdapter

ROOT = Path('/data1/user/ymdai/ember_runs/aligned_teacher_recovery_20261008')
STUDY = ROOT.name
KIND = 'aligned_teacher_recovery_complete_lora'
SCHEMA = 'ember_aligned_teacher_recovery_readout_v1'
EPISODE_SCHEMA = 'ember_aligned_teacher_recovery_episode_v1'
AUDIT = ROOT / 'analysis/teacher_sources.json'
SCENES = Path('/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/scenes')
TASKS, STATES = (12, 29, 32, 38), (32, 33, 34, 35)
PANELS = {'expert_u320_initial': (320, False), 'expert_u480_initial': (480, False),
          'expert_u320_T50': (320, True), 'expert_u480_T50': (480, True),
          'MT300_T50': (300, True)}
TRACE_FIELDS = ('eef_pos', 'eef_quat', 'gripper_qpos', 'body_positions', 'predicates')
CAMERAS = ('observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Pi05EvaluationError(f'{STUDY}: {message}')


def file_record(path: Path | str) -> dict:
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size}


def sources() -> dict:
    audit = read_json(AUDIT)
    rows = audit['conditions']
    require(audit['study_id'] == STUDY and audit['verified_complete_rows'] == 16
            and len(rows) == 16 and {(r['global_task_id'], r['init_state_id']) for r in rows}
            == {(t, s) for t in TASKS for s in STATES}, 'fixed original cohort changed')
    definitions = {t['global_task_id']: t for t in audit['task_definitions']}
    require(set(definitions) == set(TASKS) and len(definitions[38]['goal_predicates']) == 3
            and ['turnon', 'flat_stove_1'] in definitions[38]['goal_predicates'], 'complete goal AND changed')
    return audit


def factor_headers(path: Path, contract: Any) -> dict:
    expected = expected_lora_state_shapes(contract)
    original_mt = path.resolve() == Path(sources()['sources']['MT_checkpoint']).resolve()
    dtypes = {}
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        require(set(handle.keys()) == set(expected), 'complete independent A/B tensor names changed')
        dtypes = {k: handle.get_slice(k).get_dtype() for k in expected}
        require(all(tuple(handle.get_slice(k).get_shape()) == shape
                    and dtypes[k] == ('BF16' if original_mt and not k.startswith('model.action_') else 'F32')
                    for k, shape in expected.items()),
                'rank128 full-factor shape/dtype changed')
    return {'tensor_count': len(expected), 'parameter_count': contract.parameter_count,
            'dtype_counts': {dtype: list(dtypes.values()).count(dtype) for dtype in sorted(set(dtypes.values()))}}


def build_manifest(panel: str, *, source: Mapping, lora_contract_path: Path,
                   endpoint_records: Mapping[int, Mapping] | None, readout_git: str) -> dict:
    """Use actual root-published exports; endpoint record has adapter_path and
    checkpoint_origin={path: checkpoint directory, training_git: pushed commit}.
    No checkpoint/beta copy or source-policy inspection is performed here.
    """
    require(panel in PANELS and len(readout_git) == 40, 'unregistered panel/readout git')
    audit, contract = sources(), load_pi05_lora_contract(lora_contract_path)
    require((contract.rank, contract.alpha, len(contract.targets), contract.state_tensor_count)
            == (128, 128, 38, 76), 'current source complete-LoRA contract changed')
    step, handoff = PANELS[panel]
    tasks = []
    for definition in audit['task_definitions']:
        tid = definition['global_task_id']
        path = Path(audit['sources']['MT_checkpoint']) if step == 300 else Path(endpoint_records[tid]['adapter_path'])
        origin = {'kind': 'original_MT300', 'weights': file_record(path)}
        if step != 300:
            supplied = endpoint_records[tid]['checkpoint_origin']
            checkpoint = Path(supplied['path']).resolve()
            require(checkpoint.name == f'macro_{step:08d}' and checkpoint.is_relative_to(
                ROOT / 'training' / f'task_{tid:04d}' / 'attempts')
                and checkpoint.parent.name == 'checkpoints', 'formal task checkpoint path changed')
            saved = read_json(checkpoint / 'checkpoint_manifest.json')
            require(saved['schema_version'] == 'ember_ecp_checkpoint_v1'
                    and saved['next_macro'] == step and len(supplied['training_git']) == 40,
                    'checkpoint endpoint/training git changed')
            require(path.resolve() == ROOT / 'readouts/weights' / f'task_{tid:04d}_u{step}' / 'lora.safetensors',
                    'root-published complete factor path changed')
            origin = {**dict(supplied), 'kind': 'formal_task_expert',
                      'manifest': file_record(checkpoint / 'checkpoint_manifest.json'),
                      'stage': saved['stage'], 'schema_version': saved['schema_version'],
                      'model_tensor_selector': 'values.*', 'export_order': 'sorted canonical full LoRA names'}
        tasks.append({'suite': definition['suite'], 'task_id': definition['task_id'],
                      'global_task_id': tid, 'language': definition['language'],
                      'horizon': definition['horizon'], 'allowed_init_state_ids': list(STATES),
                      'adapter_path': str(path.resolve()), 'adapter_bytes': path.stat().st_size,
                      'factor_headers': factor_headers(path, contract), 'checkpoint_origin': origin})
    return {'schema_version': SCHEMA, 'study_id': STUDY, 'kind': KIND, 'panel': panel, 'arm': panel,
            'step': step, 'handoff_steps': 50 if handoff else 0, 'formal': True,
            'evaluation_role': 'operator_seen_training36', 'source': dict(source),
            'lora_contract': file_record(lora_contract_path), 'lora': contract.to_dict(),
            'tasks': tasks, 'cohort': file_record(AUDIT), 'scene_root': str(SCENES),
            'scene_manifest': file_record(SCENES / 'manifest.json'), 'readout_git': readout_git,
            'information_wall': {'training_side_privileged_diagnostic': True,
                                 'writer_invocations': 0, 'online_T_invocations': 0,
                                 'held_data_use': False, 'extra_rollouts': 0}}


def inspect_manifest(*, manifest_path: Path, source: Mapping, task_keys: Sequence,
                     evaluation_role: str, require_formal: bool) -> dict:
    manifest, audit = read_json(manifest_path), sources()
    require(manifest.get('schema_version') == SCHEMA and manifest.get('study_id') == STUDY
            and manifest.get('kind') == KIND and manifest.get('formal') is True
            and require_formal and evaluation_role == manifest.get('evaluation_role') == 'operator_seen_training36',
            'typed formal registration changed')
    expected_source = audit['sources']['source_policy']
    require(all(Path(source[k]).resolve() == Path(manifest['source'][k]).resolve() == Path(expected_source[k]).resolve()
                for k in ('source_run', 'checkpoint', 'model_path')) and source['optimizer_step'] == 1000,
            'independently inspected aligned source1000 changed')
    keys = {(t['suite'], t['task_id']) for t in audit['task_definitions']}
    require(set(map(tuple, task_keys)) == keys and len(task_keys) == 4, 'four actual task keys changed')
    rebuilt = build_manifest(manifest['panel'], source=source,
                             lora_contract_path=Path(manifest['lora_contract']['path']),
                             endpoint_records={r['global_task_id']: r for r in manifest['tasks']},
                             readout_git=manifest['readout_git'])
    require(rebuilt == manifest, 'published manifest no longer matches actual originals')
    return {**manifest, 'manifest': file_record(manifest_path)}


def select_tasks(args: Any, installed_tasks: Sequence) -> tuple:
    """After canonical _select_init_states; before ordinary subset selection."""
    require(args.role == 'operator_seen_training36' and args.mode == 'formal'
            and args.state_count == 4 and tuple(args.init_state_ids) == STATES
            and not any(getattr(args, k, None) for k in ('task_subset_selection', 'occupancy_capture_selection', 'exploration_sigma')),
            'finite four-task formal scope changed')
    keys = {(t['suite'], t['task_id']) for t in sources()['task_definitions']}
    selected = tuple(t for t in installed_tasks if (t.suite, t.task_id) in keys)
    require(len(selected) == 4 and all(tuple(t.init_state_ids) == STATES for t in selected), 'installed cohort incomplete')
    return selected


def attach_provenance(contract: dict) -> None:
    adapter = contract['adapter']
    output = ROOT / 'readouts' / adapter['panel'] / 'evaluation'
    definitions = {(r['suite'], r['task_id']): r for r in sources()['task_definitions']}
    require(len(contract['tasks']) == 4 and {(t['suite'], t['task_id']) for t in contract['tasks']} == set(definitions)
            and all(tuple(t['init_state_ids']) == STATES and t['horizon'] == definitions[(t['suite'], t['task_id'])]['horizon']
                    for t in contract['tasks']), 'actual task/state/horizon cohort changed')
    require(Path(contract['output_dir']).resolve() == output and contract['rng']['inference_seed'] == 7
            and contract['policy']['num_inference_steps'] == 10 and contract['policy']['replan_steps'] == 5,
            'canonical output/noise/flow clock changed')
    scene = read_json(SCENES / 'manifest.json')
    cases = {(t['suite'], t['task_id'], s) for t in contract['tasks'] for s in STATES}
    require(scene['schema_version'] == 'ember_operator_seen_task_scenes_v1' and scene['seed'] == 7
            and scene['dummy_steps'] == 10 and len(scene['scenes']) == 144, 'original seen144 registry changed')
    selected = [r for r in scene['scenes'] if (r['suite'], r['task_id'], r['state']) in cases]
    require(len(cases) == len(selected) == 16 and all(file_record(r['path']) == {'path': r['path'], 'bytes': r['bytes']}
                                                    for r in selected), 'original sixteen scene records incomplete')
    contract['aligned_teacher_recovery_scene'] = {'root': str(SCENES), 'manifest': file_record(SCENES / 'manifest.json')}
    contract['diagnostic_occupancy_capture'] = {'schema_version': SCHEMA, 'study_id': STUDY, 'mode': 'compact',
        'full_conditions': [{'suite': t['suite'], 'task_id': t['task_id'], 'init_state_id': 32} for t in contract['tasks']],
        'trajectory_root': str(output / 'trajectories'), 'aligned_teacher_recovery': True,
        'passive_trace': {'schema_version': SCHEMA, 'trace_root': str(output / 'continuous_traces')},
        'training_gradient_use': False, 'checkpoint_selection_use': False, 'held_data_use': False}
    contract['diagnostic_stage_predicates'] = {'schema_version': 'ember_pi05_stage_predicate_capture_v1',
        'predicate_source': 'installed_LIBERO_BDDL_goal_conjunction', 'full_conditions_only': False,
        'training_gradient_use': False, 'checkpoint_selection_use': False, 'held_data_use': False}


def validate_capture_contract(contract: Mapping) -> None:
    expected = dict(contract)
    attach_provenance(expected)
    require(all(contract.get(k) == expected.get(k) for k in ('aligned_teacher_recovery_scene',
            'diagnostic_occupancy_capture', 'diagnostic_stage_predicates')), 'registered scene/capture changed')


@dataclass(frozen=True)
class PreparedRecovery:
    key: tuple[str, int]
    evidence: dict
    condition: dict
    prefix: dict | None
    archived: dict | None


class CompleteRecoveryAdapter(FrozenStaticTaskLoRAAdapter):
    """A separate explicit rank128 type; reuse inherited factor install/predict.

    The old rank16 constructor/inspector remains unchanged. This constructor is
    never called by the CPU preparation checks.
    """
    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal):
        del device
        observed = inspect_manifest(manifest_path=Path(evaluation_adapter['manifest']['path']), source=source,
                                    task_keys=task_keys, evaluation_role='operator_seen_training36', require_formal=require_formal)
        require(observed == evaluation_adapter, 'runtime typed adapter changed since prepare')
        self.registration = observed
        self.lora = load_pi05_lora_contract(Path(observed['lora_contract']['path']))
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            if observed['step'] != 300:
                parameter.data = parameter.data.float()
            parameter.requires_grad_(False)
        policy.eval()
        self.policy, self.records = policy, {(r['suite'], r['task_id']): r for r in observed['tasks']}
        self._states, self._installed = {}, None
        self.conditions = {(r['suite'], r['task_id'], r['init_state_id']): r for r in sources()['conditions']}

    def prepare_episode(self, *, suite, task_id, init_state_id):
        key, condition = (suite, int(task_id)), self.conditions[(suite, int(task_id), int(init_state_id))]
        prefix, archived = None, None
        if self.registration['handoff_steps']:
            with np.load(condition['prefix_reference']['source'], allow_pickle=False) as data:
                prefix = {name: data[name][:50 if name == 'actions' else 51].copy()
                          for name in ('actions', *TRACE_FIELDS)}
                prefix['body_names'] = data['body_names'].tolist()
            # These original PT storages fail mmap tensor rebuild; use their
            # verified ordinary CPU consumer, not a general loading fallback.
            archived = torch.load(condition['original_T_capture_readout']['path'], map_location='cpu', weights_only=False)
            require(tuple(archived['replan_steps'][:11]) == tuple(range(0, 51, 5))
                    and all(x.shape == (1, 50, 7) and torch.isfinite(x).all() for x in archived['action_chunks'][:10]),
                    'original real prefix proposal clock/shape changed')
            commands = np.concatenate([np.asarray(x) for x in archived['executed_action_prefixes'][:10]])
            require(commands.shape == (50, 7) and np.array_equal(commands, prefix['actions'])
                    and np.isfinite(commands).all() and not prefix['predicates'].all(axis=1).any(), 'original raw prefix changed')
            rgb = archived.get('observations')
            archived = {**{name: tuple(archived[name][:11]) for name in ('policy_noise_seeds', 'replan_steps')},
                        'action_chunks': tuple(archived['action_chunks'][:10]),
                        'handoff_RGB': {camera: rgb[10][camera] for camera in CAMERAS} if rgb else None}
        evidence = {'schema_version': EPISODE_SCHEMA, 'study_id': STUDY, 'panel': self.registration['panel'],
                    **self.records[key], 'init_state_id': int(init_state_id), 'condition_id': key,
                    'teacher_condition_id': condition['condition_id'], 'handoff_steps': self.registration['handoff_steps']}
        return PreparedRecovery(key, evidence, condition, prefix, archived)

    def before_plan(self, slots, *, task, preprocess, root_seed, replan_steps):
        require(root_seed == 7 and replan_steps == 5, 'absolute policy clock changed')
        for slot in slots:
            if slot is None or slot['action_plan']:
                continue
            prepared = slot['episode_adapter']
            if prepared.prefix is None:
                continue
            if slot['steps'] >= 50:
                if slot['steps'] == 50 and 'aligned_handoff_verification' not in slot:
                    slot['aligned_handoff_verification'] = verify_boundary(slot, prepared, task)
                require('aligned_handoff_verification' in slot, 'suffix started without boundary verification')
                continue
            require(slot['steps'] < 50 and slot['steps'] == 5 * slot['replan_index'], 'prefix command clock changed')
            index, start = slot['replan_index'], slot['steps']
            raw = libero_policy_input(slot['obs'], task['language'])
            commands = prepared.prefix['actions'][start:start + 5]
            record_replan(slot, raw, preprocess(raw), None, commands, command_kind='external_saved_action')
            slot['action_plan'].extend(commands.copy())
            seed = policy_noise_seed(7, task['suite'], task['task_id'], slot['init_state_id'], index)
            require(seed == prepared.archived['policy_noise_seeds'][index], 'original prefix seed provenance changed')
            slot['policy_noise_seeds'].append(seed)  # provenance only: no flow noise or model call
            slot['replan_index'] += 1


def verify_boundary(slot: Mapping, prepared: PreparedRecovery, task: Mapping) -> dict:
    require(slot['steps'] == 50 and slot['replan_index'] == 10 and not slot['action_plan'], 'handoff is not at absolute50/10')
    trace, expected = slot['passive_trace'], prepared.prefix
    require([r['name'] for r in trace['body_registry']] == expected['body_names']
            and [list(x) for x in slot['stage_predicate_states']] == prepared.condition['prefix_reference']['goal_predicates'], 'body/goal order changed')
    errors = {}
    for name in ('actions', *TRACE_FIELDS):
        actual, reference = np.asarray(trace[name]), expected[name]
        require(actual.shape == reference.shape and np.isfinite(actual).all(), f'prefix {name} is incomplete')
        error = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
        require(error == 0 if name in ('actions', 'predicates') else error <= 1e-8, f'prefix {name} differs')
        errors[name] = error
    reference_rgb = prepared.archived['handoff_RGB']
    raw, pixels = (libero_policy_input(slot['obs'], task['language']) if reference_rgb is not None else None), {}
    for camera in CAMERAS if reference_rgb is not None else ():
        actual, reference = raw[camera].unsqueeze(0), reference_rgb[camera]
        require(actual.shape == reference.shape == (1, 3, 256, 256)
                and torch.isfinite(reference).all() and reference.min() >= 0 and reference.max() <= 1,
                'canonical float32 unit-range t50 RGB changed')
        delta = int((actual.mul(255).round().to(torch.int16) - reference.mul(255).round().to(torch.int16)).abs().max())
        require(delta <= 1, 'handoff dual RGB differs beyond one quantization level')
        pixels[camera] = delta
    return {'raw_commands': 50, 'state_samples': 51, 'absolute_replan': 10, 'field_max_abs': errors,
            'RGB_uint8_max_abs': pixels, 'RGB_reference_available': reference_rgb is not None,
            'RGB_verification': 'dual_camera_one_LSB' if reference_rgb is not None else 'not_checked_original_RGB_missing',
            'original_T_trace': file_record(prepared.condition['prefix_reference']['source']),
            'original_T_capture': file_record(prepared.condition['original_T_capture_readout']['path']),
            'reset_at_cut': False, 'extra_settling_at_cut': 0, 'online_T_source_Writer_prefix_calls': 0}


def finish_evidence(slot: Mapping) -> dict:
    prepared = slot['episode_adapter']
    if prepared.prefix is not None:
        require('aligned_handoff_verification' in slot and slot['steps'] >= 50, 'handoff boundary never verified')
    return {'aligned_teacher_recovery_lora': dict(prepared.evidence),
            'aligned_teacher_recovery_handoff': slot.get('aligned_handoff_verification')}


def capture_fields(slot: Mapping) -> dict:
    prepared = slot['episode_adapter']
    extra = {'command_kinds': tuple(slot['replay_command_kinds']),
             'policy_noise_consumed': tuple(k == 'model_prediction' for k in slot['replay_command_kinds'])}
    if prepared.archived is not None:
        extra.update(archived_T50_action_chunks=tuple(prepared.archived['action_chunks'][:10]),
                     archived_T50_policy_noise_seeds=tuple(prepared.archived['policy_noise_seeds'][:10]),
                     original_T_capture_reference=file_record(prepared.condition['original_T_capture_readout']['path']),
                     archived_T50_proposal_origin='original saved normalized real50x7; tails were not executed')
    return extra


def validate_episode(adapter: Mapping, row: Mapping, *, suite, task_id, init_state_id) -> bool:
    key = (suite, int(task_id))
    records = { (r['suite'], r['task_id']): r for r in adapter['tasks'] }
    condition = next((r for r in sources()['conditions'] if (r['suite'], r['task_id'], r['init_state_id']) == (*key, int(init_state_id))), None)
    if key not in records or condition is None:
        return False
    expected = {'schema_version': EPISODE_SCHEMA, 'study_id': STUDY, 'panel': adapter['panel'],
                **records[key], 'init_state_id': int(init_state_id), 'condition_id': list(key),
                'teacher_condition_id': condition['condition_id'], 'handoff_steps': adapter['handoff_steps']}
    observed = dict(row.get('aligned_teacher_recovery_lora') or {})
    if 'condition_id' in observed:
        observed['condition_id'] = list(observed['condition_id'])
    handoff = row.get('aligned_teacher_recovery_handoff')
    return observed == expected and (handoff is None if not adapter['handoff_steps'] else isinstance(handoff, Mapping)
        and handoff.get('raw_commands') == 50 and handoff.get('state_samples') == 51
        and handoff.get('absolute_replan') == 10 and handoff.get('reset_at_cut') is False)
