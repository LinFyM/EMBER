"""Explicit component migration and fixed initialization for the complete Writer."""
from __future__ import annotations

from pathlib import Path
import copy

import torch

from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state, restore_rng

from .contract import TASKS, OPTIMIZER, SOURCE, MT_PATH, CHECKPOINT_SCHEMA, SELECTOR_SCHEMA, G_OBJECTIVE, materialize_panel, seed
from .frozen_context import resume_endpoint_state
from .learning import Baseline
from .supervision import complete_query, query_panel

SCHEMA = 'ember_full_writer_after24849_frozen_frontend_v1'
SOURCE_SCHEMA = 'ember_full36_coordinate_memory_read_writer_v9'
INITIAL_G, COVERED_G, TERMINAL_G = 24849, 30249, 38349


def _link(destination, source):
    source = Path(source).resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise ValueError('immutable full-Writer asset reference changed')
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def _source(root):
    root = Path(root).resolve(strict=True); run = read_json(root/'run_contract.json')
    binding = run['inference_components']; practice_root = root/'practice'
    practice = read_json(practice_root/'run_contract.json')
    frontend = Path(binding['frontend_checkpoint']).resolve(strict=True)
    endpoint = Path(binding['endpoint_checkpoint']).resolve(strict=True)
    complete = read_json(Path(binding['endpoint_complete']))
    front_manifest, endpoint_manifest = (read_json(p/'manifest.json') for p in (frontend, endpoint))
    if (run['schema_version'] != 'ember_terminal_student_video_evidence_v1'
            or tuple(run['training_tasks']) != TASKS or not complete['complete']
            or complete['terminal_update'] != INITIAL_G or Path(complete['checkpoint']).resolve() != endpoint
            or not front_manifest['complete'] or front_manifest['schema_version'] != SOURCE_SCHEMA
            or not front_manifest['full_optimizer_rng_sampler'] or front_manifest['next_update'] != 432
            or not endpoint_manifest['complete'] or not endpoint_manifest['full_endpoint_optimizer_rng_sampler']
            or endpoint_manifest['next_update'] != INITIAL_G
            or Path(endpoint_manifest['source_checkpoint']).resolve() != frontend
            or practice['student_refresh']['readout_mode'] != 'saved_endpoint_practice_only'
            or practice['student_refresh']['G_update'] != INITIAL_G
            or len(complete['original36_endpoints']) != 36
            or {row['task'] for row in complete['original36_endpoints']} != set(TASKS)):
        raise ValueError('complete CP432/frontend, endpoint24849 and actual36 student sources required')
    return root, run, practice_root, practice, binding, complete


def _students(root, previous, panel, practice, complete):
    """A metadata view retains the actual forced practice and readonly query pair."""
    version = practice['student_refresh']['version']; sources = []
    weights = {row['task']: row for row in complete['original36_endpoints']}
    for row in panel['tasks']:
        task = row['task_id']; source = previous/'student_data'/version/f'task_{task:04d}'
        parent, path = (read_json(source/name) for name in ('parent.json', 'path.json'))
        query = read_json(source/'query/complete.json'); boundary = parent['boundaries'][0]
        states = query_panel(row, 0)
        if (not parent['complete'] or not parent['eligible_for_G_parent_bank'] or not parent['query_excluded_from_H']
                or not parent['query_credit_readonly'] or not parent['real_G_practice']
                or parent['new_G_practice_episodes'] != 1 or parent['new_G_generations'] != 0
                or parent['query_count'] != 16 or path['completed_rounds'] != 1 or path['outputs'] or path['decisions']
                or path['generator_schema'] != SOURCE_SCHEMA or path['selector_schema'] != SELECTOR_SCHEMA
                or path['sampling_identity'] != 'saved_endpoint_practice_only'
                or path['teacher_demo'] != weights[task]['demo']
                or (source/'G_000.safetensors').resolve() != Path(weights[task]['weights']).resolve()
                or boundary['actual_parameter_ref'] != 'G_000' or boundary['episodes'] != 2
                or boundary['row'].get('model_failure') or boundary['row'].get('budget_truncated')
                or boundary['row']['init_state_id'] in states
                or set(parent['producer_refs']) != {'MT300', 'G_000'}
                or not complete_query(query, states, seed(70, task, 0), ['MT300', 'G_000'])):
            raise ValueError(f'actual student practice/query/source identity changed for task{task}')
        destination = root/'student_data'/version/f'task_{task:04d}'
        for name in ('parent.json', 'practice.json', 'history.pt.gz', 'G_000.safetensors', 'query'):
            _link(destination/name, source/name)
        view = dict(path, generator_schema=CHECKPOINT_SCHEMA, source_generator_schema=SOURCE_SCHEMA,
            source_path=str(source/'path.json'), metadata_migration_only=True,
            full_native_sources_unchanged=True, forced_practice_is_not_on_policy_credit=True)
        write_json_atomic(destination/'path.json', view)
        sources.append(dict(task=task, parent=str(source/'parent.json'), path=str(source/'path.json'),
            history=parent['parent']['histories'], producers=parent['producer_refs'],
            query_credit=str(source/'query/complete.json'), teacher_demo=path['teacher_demo']))
    return version, sources


def prepare(args):
    """Register the one new full-Writer window without sampling or copying assets."""
    root = Path(args.root)
    previous, old, practice_root, practice, binding, complete = _source(args.reuse_root)
    if root.exists():
        raise ValueError('full-Writer preparation requires its new canonical output root')
    if args.checkpoint is not None and Path(args.checkpoint).resolve() != Path(binding['endpoint_checkpoint']).resolve():
        raise ValueError('only the registered full endpoint24849 is importable')
    panel = materialize_panel(practice_root)
    if any(set(row['policy_videos']) != set(range(46, 50)) or len(query_panel(row, 0)) != 16 for row in panel['tasks']):
        raise ValueError('all36 four-video pools and original distinct query16 are required')
    frontend = Path(binding['frontend_checkpoint']).resolve(); base = frontend.parents[2]
    original = read_json(base/'run_contract.json'); cache = Path(binding['endpoint_complete']).parent/'contexts'
    cached = read_json(cache/'complete.json')
    if (not cached['complete'] or cached['conditions'] != 144
            or Path(cached['exact_source_checkpoint']).resolve() != frontend):
        raise ValueError('original144 MT contexts need the exact frozen frontend')
    root.mkdir(parents=True)
    for name in ('initial.safetensors', 'task_lora_contract.json', 'bank_scale.json', 'operator_scale.json',
                 'coordinates.json', 'panel.json', 'base', 'teachers', 'bootstrap'):
        _link(root/name, practice_root/name)
    for name in ('cuda', 'triton', 'huggingface', 'tmp'):
        (root/'cache'/name).mkdir(parents=True)
    version, students = _students(root, practice_root, panel, practice, complete)
    checkpoint = root/'checkpoints/assembled/update_00000090'
    selector = {k: v for k, v in original['selector_training'].items()
                if k not in ('formal_update_window_not_yet_registered', 'registered_theta_updates')}
    contract = dict(schema_version=SCHEMA, checkpoint_schema=CHECKPOINT_SCHEMA,
        design='docs/designs/video_guided_proposal_writer_20261010.md#20', source=SOURCE,
        mt_checkpoint=str(MT_PATH), environment=old['environment'], training_tasks=list(TASKS),
        teacher=dict(original['teacher'], readonly_reuse=True), optimizer=OPTIMIZER,
        inference_components=binding, source_checkpoint=str(frontend), initial_endpoint_checkpoint=binding['endpoint_checkpoint'],
        component_sources=dict(frontend_training_code=original['git']['commit'], endpoint_training_code=old['training_source'],
            frontend_schema=SOURCE_SCHEMA, actor_initialization='untrained Actor saved in CP432'),
        initial_cursor=INITIAL_G, source_schema=SOURCE_SCHEMA, component_migration=True, exact_resume=False,
        trainable_owner='generator.endpoint only', G_gradient_owner='endpoint', G_objective=G_OBJECTIVE,
        frozen_components='CP432 source/meta/Reader/static/layout; merged MT execution base',
        frozen_frontend_checkpoint=str(frontend), frozen_context_cache_root=str(cache.parent),
        initial_writer_checkpoint=str(checkpoint), initial_writer_stage='assembled', initial_G_rng_topology=4, baseline_seed=7,
        updates=dict(G=[COVERED_G], local=[90], RL=[9]), fixed_G_update=INITIAL_G,
        fixed_local_update=90, G_checkpoint_updates=[COVERED_G], G_initial_cursor=INITIAL_G,
        G_bank_mode='actual_parents', G_bank_version=0, fixed_scale_reestimated=False,
        original_MT_context_cache=str(cache), new_initial_G_context_conditions=144,
        context_reuse='exact frozen frontend, actual P, complete H/producers, language and video only',
        student_refresh=dict(version=version, G_update=INITIAL_G, readout_mode='saved_endpoint_practice_only',
            readonly_E_query=False, query_states_per_parameter=16, task_count=36,
            condition_mapping=practice['student_refresh']['condition_mapping'], new_refresh=False,
            post_generation_practice=True, actual_practice_source=str(practice_root), credit_readonly=True),
        previous_root=str(practice_root), previous_student_version=version,
        initial_student_sources=students, logical_batch=dict(G_conditions=4, actor_conditions=4),
        task_parent_history_sampling='equal task; equal actual parent; uniform real H within parent',
        initial_coverage=dict(updates=5400, full36_rounds=600, start=INITIAL_G, stop=COVERED_G),
        final_warm=dict(updates=90, full36_rounds=10, final_only=True, new_environment_episodes=0),
        joint_training=dict(G_updates_per_batch=900, G_total_updates=8100, full36_rounds_per_block=100,
            final_G_cursor=TERMINAL_G, alternating_order=['fixed psi/theta collection', 'theta', 'true P/H bank', 'F', 'resample']),
        selector_schema=SELECTOR_SCHEMA, selector_training=dict(selector, rounds=5, continue_cost=.001,
            query_states_per_parameter=16, registered_theta_updates=9,
            practice_candidates=['parent', 'one newG'], practice_selection='actual Actor categorical score',
            unpracticed_G_query_credit=False),
        compilation=dict(original['compilation'], proposals_per_decision=1, parameter_decoder_calls=1),
        formal_report=dict(version=['RL', 9], checkpoint=str(root/'checkpoints/RL/update_00000009'),
            arms=['correct'], video_seed=7, max_rounds=5, environment_multiplier=6,
            states_per_task=50, conditions=400, checkpoint_selection='one registered complete RL9 only',
            test_enabled=False, held_labels_read=False, external_budget_is_not_model_input=True),
        expected_initial_refresh_episodes=0, expected_initial_reference_query_episodes=0,
        expected_joint_practice_episodes=180, expected_joint_G_proposals=180,
        expected_joint_G_practice_max=180, expected_joint_G_query_max=2880,
        expected_joint_new_episodes_max=3060, F_added_updates=13500,
        query_independent_of_teacher_selection=False, query_teacher_selection_overlap_per_task=2,
        future_budget_model_input=False, new_initial_teacher_training=False,
        limits=dict(added_peak_GiB=96, GPU_hours=None, GPU_hours_hard_cap_active=False),
        bounded_stop='assembled initialization, nine complete joint batches, one RL9 strict400; no automatic successor',
        information_wall='E label only; query only theta/baseline, never H/F labels; no held gradients or trajectory fitting',
        source_record=str(previous/'run_contract.json'), git=git_state())
    write_json_atomic(root/'run_contract.json', contract)
    write_json_atomic(root/'bank_ready.json', dict(complete=True, version=0, task_count=36,
        parents_per_task=2, students=students, readonly_MT_contexts=144, new_G_contexts_required=144,
        actual_initial_query_credit_rows=1152, query_not_in_H=True, successful_parent_filter=False))
    return contract


def _frontend(path, runtime):
    path = Path(path); manifest = read_json(path/'manifest.json')
    state = torch.load(path/'state.pt', map_location='cpu', weights_only=False)
    if (manifest['schema_version'] != SOURCE_SCHEMA or state['schema_version'] != SOURCE_SCHEMA
            or not manifest['complete'] or not manifest['full_optimizer_rng_sampler']
            or manifest['stage'] != 'G' or state['stage'] != 'G'
            or manifest['next_update'] != 432 or state['next_update'] != 432
            or state['G_sampler']['next_update'] != 432 or state['sampler']['next_update'] != 432
            or state['topology']['task_order'] != list(TASKS) or state['topology']['logical_condition_batch'] != 4
            or manifest['topology'] != state['topology'] or len(state['rank_rng']) != state['topology']['world_size']
            or state['G_objective'] != G_OBJECTIVE or state['supervision_version'] != 0
            or state['parameter_names']['psi'] != list(dict(runtime.generator.named_parameters()))
            or state['parameter_names']['theta'] != list(dict(runtime.actor.named_parameters()))):
        raise ValueError('CP432 source names and untrained Actor ownership changed')
    return state


def _optimizers(runtime, baseline):
    generator = torch.optim.AdamW(runtime.generator.endpoint.parameters(), **OPTIMIZER)
    generator.owner = 'endpoint'
    generator.parameter_names = ['endpoint.'+name for name, _ in runtime.generator.endpoint.named_parameters()]
    return (generator,
            torch.optim.AdamW(runtime.actor.parameters(), **OPTIMIZER),
            torch.optim.AdamW(baseline.parameters(), **OPTIMIZER))


def _baseline(runtime, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); baseline = Baseline()
    return baseline.to(runtime.device)


def _rng(state, context):
    if context.rank >= len(state['rank_rng']):
        raise ValueError('new physical rank needs an explicitly registered additional RNG source')
    if context.device.type == 'cuda':
        restore_rng(state['rank_rng'][context.rank], context)
    else:
        import random
        import numpy as np
        rng = state['rank_rng'][context.rank]
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch_cpu'])


def initialize(runtime, context, root):
    """Import the original frontend/Actor and the named endpoint/Adam exactly once."""
    root = Path(root); run = read_json(root/'run_contract.json')
    if run['schema_version'] != SCHEMA:
        raise ValueError('complete Writer component import must be explicitly registered')
    binding = run['inference_components']; source = _frontend(binding['frontend_checkpoint'], runtime)
    names = ['endpoint.'+name for name, _ in runtime.generator.endpoint.named_parameters()]
    if len(names) != 166 or sum(p.numel() for p in runtime.generator.endpoint.parameters()) != 8761152:
        raise ValueError('the actual endpoint owner changed from the registered166 tensors')
    state = resume_endpoint_state(binding['endpoint_checkpoint'], run, names)
    runtime.generator.load_state_dict(source['psi'], strict=True)
    runtime.generator.endpoint.load_state_dict(state['endpoint'], strict=True)
    runtime.actor.load_state_dict(source['theta'], strict=True)
    baseline = _baseline(runtime, run['baseline_seed'])
    G_optimizer, actor_optimizer, baseline_optimizer = _optimizers(runtime, baseline)
    G_optimizer.load_state_dict(state['optimizer'])
    runtime.generator_optimizer_state = G_optimizer.state_dict()
    runtime.generator_optimizer_parameter_names = names; runtime.G_sampler = dict(state['sampler'])
    runtime.version = INITIAL_G; runtime.supervision_version = 0; runtime.objective_migration = None
    runtime.G_gradient_owner = 'endpoint'; runtime.frozen_frontend_checkpoint = binding['frontend_checkpoint']
    runtime.component_migration = dict(frontend_checkpoint=binding['frontend_checkpoint'],
        endpoint_checkpoint=binding['endpoint_checkpoint'], endpoint_parameter_names=names,
        frozen_parameter_count=len(source['parameter_names']['psi'])-len(names),
        initial_endpoint_cursor=INITIAL_G, endpoint_source_topology=state['topology'],
        current_topology=dict(world_size=context.world_size, logical_condition_batch=4),
        actor_optimizer_fresh=True, baseline_optimizer_fresh=True, baseline_seed=run['baseline_seed'], exact_resume=False)
    runtime.component_rng_sources = dict(endpoint=state['rank_rng'], actor_initialization=source['rank_rng'])
    runtime.component_samplers = dict(G=copy.deepcopy(state['sampler']), theta=dict(next_update=0, logical_condition_batch=4))
    runtime.freeze_psi(); runtime.actor.requires_grad_(True); _rng(state, context)
    return dict(generator_optimizer=G_optimizer, actor_optimizer=actor_optimizer, baseline=baseline,
        baseline_optimizer=baseline_optimizer, G_cursor=INITIAL_G, local_cursor=0, migration=runtime.component_migration)


def _terminal(path, stage, update, runtime):
    path = Path(path); manifest = read_json(path/'manifest.json')
    state = torch.load(path/'state.pt', map_location='cpu', weights_only=False)
    endpoint_names = ['endpoint.'+name for name, _ in runtime.generator.endpoint.named_parameters()]
    if (not manifest['complete'] or not manifest['full_optimizer_rng_sampler']
            or manifest['schema_version'] != CHECKPOINT_SCHEMA or state['schema_version'] != CHECKPOINT_SCHEMA
            or manifest['stage'] != stage or state['stage'] != stage
            or manifest['next_update'] != update or state['next_update'] != update
            or state['G_objective'] != G_OBJECTIVE or len(state['rank_rng']) != state['topology']['world_size']
            or state.get('generator_optimizer_parameter_names') != endpoint_names
            or state['parameter_names']['psi'] != list(dict(runtime.generator.named_parameters()))
            or state['parameter_names']['theta'] != list(dict(runtime.actor.named_parameters()))):
        raise ValueError('only complete preregistered A/B terminals can initialize the joint window')
    return state


def assemble(args, runtime, context):
    """Use the fixed A and B endpoints, retaining their independent state origins."""
    root = Path(args.root); run = read_json(root/'run_contract.json')
    if run['schema_version'] != SCHEMA:
        raise ValueError('A/B assembly belongs to the registered full-Writer window')
    G_path, actor_path = root/'checkpoints/G/update_00030249', root/'checkpoints/local/update_00000090'
    G, actor = _terminal(G_path, 'G', COVERED_G, runtime), _terminal(actor_path, 'local', 90, runtime)
    if G['G_sampler']['next_update'] != COVERED_G or G['supervision_version'] != 0:
        raise ValueError('initial coverage must end at30249 on the original actual MT/G bank')
    runtime.generator.load_state_dict(G['psi'], strict=True); runtime.actor.load_state_dict(actor['theta'], strict=True)
    baseline = _baseline(runtime, run['baseline_seed']); baseline.load_state_dict(actor['baseline'], strict=True)
    G_optimizer, actor_optimizer, baseline_optimizer = _optimizers(runtime, baseline)
    G_optimizer.load_state_dict(G['generator_optimizer']); actor_optimizer.load_state_dict(actor['optimizer'])
    if actor['baseline_optimizer'] is not None: baseline_optimizer.load_state_dict(actor['baseline_optimizer'])
    runtime.generator_optimizer_state = G_optimizer.state_dict()
    runtime.generator_optimizer_parameter_names = G_optimizer.parameter_names; runtime.G_sampler = dict(G['G_sampler'])
    runtime.version = G['psi_version']; runtime.supervision_version = G['supervision_version']
    runtime.objective_migration = None; runtime.G_gradient_owner = 'endpoint'
    runtime.frozen_frontend_checkpoint = run['inference_components']['frontend_checkpoint']
    runtime.component_migration = dict(G_checkpoint=str(G_path), theta_checkpoint=str(actor_path),
        G_source_topology=G['topology'], theta_source_topology=actor['topology'],
        current_topology=dict(world_size=context.world_size, logical_condition_batch=4),
        behavior_RNG_source='local90 current-rank RNG',
        baseline_optimizer_source='local90' if actor['baseline_optimizer'] is not None else 'fresh; baseline never warm-updated',
        exact_resume=False, best_point_selection=False)
    runtime.component_rng_sources = dict(G=G['rank_rng'], theta=actor['rank_rng'])
    runtime.component_samplers = dict(G=copy.deepcopy(G['G_sampler']), theta=copy.deepcopy(actor['sampler']))
    runtime.freeze_psi(); runtime.actor.requires_grad_(True); _rng(actor, context)
    return dict(generator_optimizer=G_optimizer, actor_optimizer=actor_optimizer, baseline=baseline,
        baseline_optimizer=baseline_optimizer, G_cursor=COVERED_G, local_cursor=90, migration=runtime.component_migration)
