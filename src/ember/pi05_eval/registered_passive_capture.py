"""Registration and provenance for the relational-support passive rollout trace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json


SPEC_RELATIVE = Path('configs/relational_support_causality_v1/experiment_spec.json')
EVALUATION_RELATIVE = Path('configs/relational_support_causality_v1/evaluation.json')
TAG = 'ember_relational_support_passive_capture_v1'


def _spec(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    path = (repo_root / SPEC_RELATIVE).resolve()
    spec = read_json(path)
    amendment = spec.get('execution', {}).get('implementation_provenance_amendment', {})
    scope = spec.get('execution', {}).get('scientific_evaluation_scope_amendment', {})
    evaluation = spec.get('evaluation', {})
    if (spec.get('schema_version') != 'ember_relational_support_causality_spec_v1'
            or spec.get('study_id') != 'relational_support_causality_20260924'
            or amendment.get('id') != 'passive_capture_fix_before_formal_evaluation_20260924'
            or amendment.get('retrain') is not False
            or amendment.get('accept_missing_trace') is not False
            or scope.get('id') != 'fixed_endpoint_and_adjacent_evaluation_20260924'
            or evaluation.get('executed_updates') != [1050, 1260]
            or evaluation.get('selection', {}).get('fixed_update') != 1260
            or evaluation.get('total_registered_rollouts_max') != 9932
            or evaluation.get('capture', {}).get('continuous_object_eef_gripper') is not True
            or evaluation['capture'].get('stage_predicates') is not True):
        raise Pi05EvaluationError('relational passive-capture science registration changed')
    return path, spec


def _formal_panel_scope(spec: Mapping[str, Any], output_dir: Path) -> bool:
    study = Path(spec['outputs']['planned_run_root']).resolve()
    root = study / 'evaluation'
    output = output_dir.resolve()
    if not output.is_relative_to(root):
        return False
    arms = [row['id'] for row in spec['arms']]
    updates = spec['evaluation']['executed_updates']
    plan = spec['evaluation']['controls_after_all_training_correct_panels_and_selection_frozen']
    panels = {f'{arm}_{step}_{panel}' for arm in arms for step in updates
              for panel in ('held', 'seen')}
    panels.update(f'{arm}_1260_support' for arm in arms)
    panels.update(f'Source_{panel}' for panel in ('held', 'seen', 'support'))
    panels.update(f'{arm}_1260_other' for arm in plan['same_task_other_at1260'])
    panels.update(f'{arm}_1260_wrong' for arm in plan['cross_suite_wrong_at1260'])
    if (len(panels) != 39 or plan['same_task_other_at_selected_if_not1260']
            or output.parent != root or output.name not in panels):
        raise Pi05EvaluationError('relational formal panel is outside fixed-endpoint evaluation scope')
    return True


def prepare_selection(
    args: Any, *, repo_root: Path, output_dir: Path, task_subset: Mapping[str, Any],
    tasks: Sequence[Any], manifest: Mapping[str, Any], selection_path: Path,
    full: tuple[tuple[str, int, int], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path, spec = _spec(repo_root)
    study = Path(spec['outputs']['planned_run_root']).resolve()
    _formal_panel_scope(spec, output_dir)
    actual = {(str(task.suite), int(task.task_id), int(state))
              for task in tasks for state in task.init_state_ids}
    if (manifest.get('passive_control_trace') != TAG
            or manifest.get('study_spec') != str(SPEC_RELATIVE)
            or manifest.get('stage_predicates') is not True
            or args.role not in {'development_train', 'nonheld_meta'}
            or args.mode not in {'formal', 'screen'}
            or Path(args.config).resolve() != (repo_root / EVALUATION_RELATIVE).resolve()
            or not output_dir.resolve().is_relative_to(study)
            or not selection_path.is_relative_to(study / 'launch' / 'selectors')
            or not Path(task_subset['selection_path']).is_relative_to(study / 'launch' / 'selectors')
            or len(full) != len(set(full)) or not set(full) <= actual):
        raise Pi05EvaluationError('relational passive-capture selection changed')
    capture = {
        'schema_version': 'ember_pi05_registered_trajectory_capture_v1',
        'selection_path': str(selection_path), 'selection_bytes': selection_path.stat().st_size,
        'mode': 'compact',
        'full_conditions': [{'suite': suite, 'task_id': task_id, 'init_state_id': state}
                            for suite, task_id, state in full],
        'trajectory_root': str((output_dir / 'trajectories').resolve()),
        'passive_trace': {'schema_version': TAG, 'spec_path': str(spec_path),
                          'spec_bytes': spec_path.stat().st_size,
                          'trace_root': str((output_dir / 'continuous_traces').resolve())},
        'training_gradient_use': False, 'checkpoint_selection_use': False,
        'validation_use': False, 'test_use': False,
    }
    stage = {
        'schema_version': 'ember_pi05_stage_predicate_capture_v1',
        'capture': 'all_rows_post_settling_then_every_executed_control_step',
        'predicate_source': 'installed_LIBERO_BDDL_goal_conjunction',
        'full_conditions_only': False,
        'training_gradient_use': False, 'checkpoint_selection_use': False,
        'validation_action_reads': 0, 'validation_reward_reads': 0,
        'held_data_use': False, 'claim_boundary': 'BDDL predicates are partial progress signals',
    }
    return capture, stage


def attach_provenance(contract: dict[str, Any], repo_root: Path) -> None:
    capture = contract.get('diagnostic_occupancy_capture') or {}
    if not capture.get('passive_trace'):
        return
    _, spec = _spec(repo_root)
    training = spec['execution']['implementation_provenance_amendment'][
        'training_materialization_and_query_commit']
    adapter = contract.get('adapter')
    if adapter is not None:
        if (adapter.get('writer_checkpoint', {}).get('training_commit') != training
                or adapter.get('materialization_git', {}).get('commit') != training
                or not Path(adapter['manifest']['path']).resolve().is_relative_to(
                    Path(spec['outputs']['planned_run_root']) / 'materialization')):
            raise Pi05EvaluationError('relational bank training or materialization provenance changed')
    contract['passive_capture_provenance'] = {
        'amendment_id': 'passive_capture_fix_before_formal_evaluation_20260924',
        'training_commit': training,
        'bank_materialization_commit': training if adapter is not None else None,
        'bank_manifest': adapter['manifest'] if adapter is not None else None,
        'evaluation_commit': contract['git']['commit'],
    }


def _selection_matches_contract(
    manifest: Mapping[str, Any], capture: Mapping[str, Any], contract: Mapping[str, Any],
) -> bool:
    declared = manifest.get('full_conditions')
    if not isinstance(declared, list):
        return False
    try:
        full = {(row['suite'], int(row['task_id']), int(row['init_state_id']))
                for row in declared}
        cases = {(task['suite'], int(task['task_id']), int(state))
                 for task in contract['tasks'] for state in task['init_state_ids']}
    except (KeyError, TypeError, ValueError):
        return False
    subset = contract.get('diagnostic_task_subset') or {}
    return (manifest.get('passive_control_trace') == TAG
            and manifest.get('stage_predicates') is True
            and manifest.get('study_spec') == str(SPEC_RELATIVE)
            and manifest.get('task_subset_selection') == subset.get('selection_path')
            and capture.get('full_conditions') == declared
            and len(full) == len(declared) and full <= cases)


def _trace_and_stage_match(
    capture: Mapping[str, Any], contract: Mapping[str, Any],
    repo_root: Path, output_dir: Path,
) -> bool:
    trace = capture['passive_trace']
    spec_path = Path(str(trace.get('spec_path', '')))
    stage = contract.get('diagnostic_stage_predicates') or {}
    return (spec_path.is_file()
            and trace.get('schema_version') == TAG
            and trace.get('spec_path') == str((repo_root / SPEC_RELATIVE).resolve())
            and trace.get('spec_bytes') == spec_path.stat().st_size
            and trace.get('trace_root') == str(output_dir / 'continuous_traces')
            and stage.get('full_conditions_only') is False
            and stage.get('capture') == 'all_rows_post_settling_then_every_executed_control_step')


def validate_contract(contract: Mapping[str, Any], repo_root: Path) -> None:
    output_dir = Path(contract['output_dir']).resolve()
    _, spec = _spec(repo_root)
    formal_panel = _formal_panel_scope(spec, output_dir)
    capture = contract.get('diagnostic_occupancy_capture') or {}
    if not capture.get('passive_trace'):
        if formal_panel:
            raise Pi05EvaluationError('relational formal evaluation lacks passive all-row capture')
        return
    path = Path(capture['selection_path'])
    if not path.is_file() or path.stat().st_size != capture['selection_bytes']:
        raise Pi05EvaluationError('relational passive capture selection missing or changed')
    manifest = read_json(path)
    if (not _selection_matches_contract(manifest, capture, contract)
            or not _trace_and_stage_match(capture, contract, repo_root, output_dir)):
        raise Pi05EvaluationError('relational passive capture or all-row predicates changed after prepare')
    expected = dict(contract)
    expected.pop('passive_capture_provenance', None)
    attach_provenance(expected, repo_root)
    if contract.get('passive_capture_provenance') != expected.get('passive_capture_provenance'):
        raise Pi05EvaluationError('relational training/bank/evaluation provenance changed')


def prepare_from_manifest(
    args: Any, *, repo_root: Path, output_dir: Path, task_subset: Mapping[str, Any],
    tasks: Sequence[Any], manifest: Mapping[str, Any], selection_path: Path,
    full: tuple[tuple[str, int, int], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (manifest.get('schema_version') != 'ember_pi05_registered_trajectory_capture_v1'
            or manifest.get('task_subset_selection') != task_subset['selection_path']
            or manifest.get('mode') != 'compact'
            or any(manifest.get(key) is not False for key in (
                'training_gradient_use', 'checkpoint_selection_use', 'validation_use', 'test_use'))):
        raise Pi05EvaluationError('registered passive trajectory selection changed')
    return prepare_selection(
        args, repo_root=repo_root, output_dir=output_dir,
        task_subset=task_subset, tasks=tasks, manifest=manifest,
        selection_path=selection_path, full=full,
    )


def attach_requested_capture(
    args: Any, contract: dict[str, Any], repo_root: Path, output_dir: Path,
) -> None:
    capture = contract.get('diagnostic_occupancy_capture') or {}
    _, spec = _spec(repo_root)
    if _formal_panel_scope(spec, output_dir):
        if Path(args.config).resolve() != (repo_root / EVALUATION_RELATIVE).resolve():
            raise Pi05EvaluationError('relational formal evaluation config changed')
        if not capture.get('passive_trace'):
            raise Pi05EvaluationError('relational formal evaluation requires passive all-row capture')
    attach_provenance(contract, repo_root)
