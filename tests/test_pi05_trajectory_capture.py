import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.episode import start_fixed_episode, update_stage_predicates
from ember.pi05_eval.registered_passive_capture import (
    TAG, attach_provenance, attach_requested_capture, prepare_from_manifest, validate_contract,
)
from ember.pi05_eval.trajectory_capture import (
    _passive_body_registry, capture_level, initialize_capture, record_passive_step, record_replan,
    save_capture, save_passive_trace, validate_passive_trace_row,
)
from ember.pi05_evaluation import rollout_shard


def _slot(level: str) -> dict:
    slot = {
        "init_state_id": 3,
        "steps": 7,
        "policy_noise_seeds": [11],
    }
    initialize_capture(slot, level)
    record_replan(
        slot,
        {"observation.state": torch.arange(8, dtype=torch.float32)},
        {"image": torch.ones((1, 3, 4, 4)), "token": torch.ones((1, 2), dtype=torch.long)},
        torch.zeros((1, 50, 7)),
        torch.ones((5, 7)),
    )
    return slot


def test_compact_capture_keeps_state_action_and_only_selected_images(tmp_path: Path) -> None:
    task = {"suite": "libero_goal", "task_id": 4}
    capture = {
        "trajectory_root": str(tmp_path),
        "mode": "compact",
        "full_conditions": [
            {"suite": "libero_goal", "task_id": 4, "init_state_id": 3}
        ],
    }
    assert capture_level(capture, task, 2) == "compact"
    assert capture_level(capture, task, 3) == "full"

    compact_slot = _slot("compact")
    compact_slot["init_state_id"] = 2
    full_slot = _slot("full")
    compact_record = save_capture(capture, task, compact_slot, success=False)
    full_record = save_capture(capture, task, full_slot, success=True)
    compact = torch.load(compact_record["path"], map_location="cpu", weights_only=False)
    full = torch.load(full_record["path"], map_location="cpu", weights_only=False)

    assert compact["schema_version"] == full["schema_version"] == "ember_pi05_diagnostic_trajectory_v2"
    assert compact["capture_level"] == "compact"
    assert full["capture_level"] == "full"
    assert "observations" not in compact
    assert len(compact["states"]) == len(compact["action_chunks"]) == 1
    assert len(compact["executed_action_prefixes"]) == 1
    assert compact["replan_steps"] == (7,)
    assert len(full["states"]) == len(full["action_chunks"]) == len(full["observations"]) == 1
    assert full["replan_predicates"] == (None,)
    assert tuple(compact["states"][0].shape) == (8,)


def test_legacy_capture_payload_remains_v1(tmp_path: Path) -> None:
    task = {"suite": "libero_goal", "task_id": 4}
    capture = {"trajectory_root": str(tmp_path)}
    record = save_capture(capture, task, _slot("full"), success=True)
    payload = torch.load(record["path"], map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "ember_pi05_occupancy_trajectory_v1"
    assert "observations" in payload
    assert "states" not in payload



TRAINING = '7dc95edbba00cf61439700d77fb321eb8df95c07'


class _Model:
    def body_id2name(self, index):
        return {0: 'white_bowl_1_main', 1: 'plate_1_main'}[index]


class _Env:
    def __init__(self, success_after=2):
        self.env = self
        self.objects_dict = {'white_bowl_1': object()}
        self.fixtures_dict = {'plate_1': object()}
        self.obj_body_id = {'white_bowl_1': 0, 'plate_1': 1}
        self.parsed_problem = {
            'goal_state': [['on', 'white_bowl_1', 'plate_1'],
                           ['in', 'white_bowl_1', 'plate_1_region']],
            'regions': {'plate_1_region': {'target': 'plate_1',
                                           'ranges': [[-.1, -.1, .1, .1]],
                                           'yaw_rotation': [0, 0]}},
        }
        self.sim = SimpleNamespace(model=_Model(), data=SimpleNamespace(
            body_xpos=np.array([[0., 0., 0.], [.5, 0., 0.]], dtype=np.float64)))
        self.success_after = success_after
        self.steps = 0
        self.reset_calls = 0
        self.step_calls = 0
        self.applied = []

    def seed(self, value):
        assert value == 7

    def reset(self):
        self.reset_calls += 1
        return self._observation()

    def set_init_state(self, state):
        assert state == 0
        self.steps = 0
        return self._observation()

    def _observation(self):
        image=np.zeros((4,4,3),dtype=np.uint8)
        return {'agentview_image':image,'robot0_eye_in_hand_image':image,
                'robot0_eef_pos':np.array([self.steps*.01,0.,0.],dtype=np.float32),
                'robot0_eef_quat':np.array([0.,0.,0.,1.],dtype=np.float32),
                'robot0_gripper_qpos':np.array([0.,0.],dtype=np.float32)}

    def _eval_predicate(self, predicate):
        return self.steps >= (1 if predicate[0]=='on' else self.success_after)

    def step(self, action):
        self.step_calls += 1
        applied=np.asarray(action,dtype=np.float32)
        if applied[-1] != -1:
            self.steps += 1
            self.applied.append(applied.copy())
            self.sim.data.body_xpos[0,0]=self.steps*.01
        return self._observation(),0.,self.steps >= self.success_after,{}


class _Policy:
    config = SimpleNamespace(chunk_size=50,max_action_dim=32)

    def reset(self):
        pass

    def predict_action_chunk(self, batch, *, noise, num_steps):
        assert num_steps==10
        return torch.zeros((noise.shape[0],50,7),dtype=torch.float32)


def _contract(tmp_path, *, full):
    return {'environment':{'dummy_action':[0,0,0,0,0,0,-1], 'dummy_settling_steps':10},
            'policy':{'replan_steps':5,'num_inference_steps':10},'rng':{'inference_seed':7},
            'diagnostic_occupancy_capture':{
                'mode':'compact','trajectory_root':str(tmp_path/'trajectories'),
                'full_conditions':([{'suite':'libero_90','task_id':36,'init_state_id':0}] if full else []),
                'passive_trace':{'schema_version':TAG,'trace_root':str(tmp_path/'continuous_traces')}},
            'diagnostic_stage_predicates':{'full_conditions_only':False}}


def _task(horizon=400):
    return {'suite':'libero_90','task_id':36,'split_role':'meta_train',
            'language':'put the white bowl on the plate','horizon':horizon}


def _preprocess(value):
    return {name:tensor.unsqueeze(0) for name,tensor in value.items()
            if isinstance(tensor,torch.Tensor)}


@pytest.mark.parametrize('full', [False, True])
def test_actual_rollout_source_records_each_control_step_and_composite_goal(tmp_path, full):
    env=_Env()
    contract=_contract(tmp_path,full=full)
    rows=rollout_shard(envs=(env,),init_states=(0,),task=_task(),state_ids=(0,),
                       contract=contract,policy=_Policy(),preprocess=_preprocess,
                       postprocess=lambda value:value)
    assert len(rows)==1
    row=rows[0]
    assert row['steps']==2 and row['success'] is True
    assert env.reset_calls==1 and env.step_calls==12
    assert len(env.applied)==2
    assert row['occupancy_trajectory']['capture_level']==('full' if full else 'compact')
    assert row['stage_predicates']['final_satisfied']==[True,True]
    info=row['continuous_control_trace']
    assert info['condition_id'] is None
    assert info['goal_operands'][1][1]['kind']=='region'
    assert info['goal_operands'][1][1]['target_body_name']=='plate_1_main'
    assert [body['kind'] for body in info['body_registry']]==['fixture','object']
    with np.load(info['trace']['path'],allow_pickle=False) as data:
        assert data['actions'].shape==(2,7)
        assert data['body_positions'].shape==(3,2,3)
        assert data['eef_pos'].shape==(3,3)
        assert data['eef_quat'].shape==(3,4)
        assert data['gripper_qpos'].shape==(3,2)
        assert data['predicates'].tolist()==[[False,False],[True,False],[True,True]]
        assert np.array_equal(data['actions'],np.stack(env.applied))
    validate_passive_trace_row(row,contract,_task())
    with np.load(info['trace']['path'],allow_pickle=False) as data:
        incomplete={name:data[name] for name in data.files if name!='gripper_qpos'}
    np.savez_compressed(info['trace']['path'],**incomplete)
    info['trace']['bytes']=Path(info['trace']['path']).stat().st_size
    with pytest.raises(Pi05EvaluationError,match='sampled field missing'):
        validate_passive_trace_row(row,contract,_task())
    Path(info['trace']['path']).unlink()
    with pytest.raises(Pi05EvaluationError,match='trace file'):
        validate_passive_trace_row(row,contract,_task())


def test_arena_goal_region_uses_actual_site_parent_without_claiming_an_object_body():
    env=_Env()
    env.workspace_name='kitchen_table'
    name='kitchen_table_plate_right_region'
    env.object_sites_dict={name:object()}
    env.parsed_problem['regions'][name]={
        'target':'kitchen_table','ranges':[[-.05,.05,.05,.15]],
        'yaw_rotation':[0.,0.]}
    class ArenaModel(_Model):
        site_bodyid=np.array([2])
        def site_name2id(self, value):
            assert value==name
            return 0
        def site_id2name(self, index):
            assert index==0
            return name
        def body_id2name(self, index):
            return 'table' if index==2 else super().body_id2name(index)
    env.sim.model=ArenaModel()
    _,goals=_passive_body_registry(env,{'stage_predicate_states':[['on','white_bowl_1',name]]})
    region=goals[0][1]
    assert region['kind']=='region' and region['target']=='kitchen_table'
    assert region['target_kind']=='arena_workspace'
    assert region['site_id']==0 and region['site_name']==name
    assert region['site_parent_body_id']==2 and region['site_parent_body_name']=='table'
    assert 'target_body_id' not in region
    env.workspace_name='other_workspace'
    with pytest.raises(Pi05EvaluationError,match='actual body'):
        _passive_body_registry(env,{'stage_predicate_states':[['on','white_bowl_1',name]]})


def test_horizon_final_sample_and_B_C_condition_identity(tmp_path):
    env=_Env(success_after=8)
    contract=_contract(tmp_path,full=False)
    task=_task(horizon=2)
    slot=start_fixed_episode(env=env,init_state_id=0,init_states=(0,),task=task,
        contract=contract,root_seed=7,dummy=np.array([0,0,0,0,0,0,-1],dtype=np.float32),
        task_adapter=None,capture_level='compact')
    slot['episode_adapter']=SimpleNamespace(key='task_76_demos_3')
    for _ in range(2):
        action=np.zeros(7,dtype=np.float32)
        slot['obs'],_,done,_=env.step(action)
        slot['steps']+=1
        update_stage_predicates(env,slot)
        record_passive_step(env,slot,action,contract['diagnostic_occupancy_capture'])
        assert done is False
    slot['episode_done']=False
    info=save_passive_trace(contract['diagnostic_occupancy_capture'],task,slot)
    row={'suite':task['suite'],'task_id':task['task_id'],'init_state_id':0,
         'steps':2,'horizon_writer_lora':{'condition_id':'task_76_demos_3'},
         'stage_predicates':{'predicates':[list(x) for x in slot['stage_predicate_states']],
                             'final_satisfied':list(slot['stage_predicate_last'])},
         'continuous_control_trace':info}
    validate_passive_trace_row(row,contract,task)
    with np.load(info['trace']['path'],allow_pickle=False) as data:
        assert data['actions'].shape==(2,7) and data['predicates'].shape==(3,2)
        assert data['predicates'][-1].tolist()==[True,False]
    row['horizon_writer_lora']['condition_id']='other_condition'
    with pytest.raises(Pi05EvaluationError,match='identity'):
        validate_passive_trace_row(row,contract,task)


def _registration(tmp_path, label='Source_core_correct'):
    repo=tmp_path/'repo'
    spec_path=repo/'configs/relational_support_causality_v1/experiment_spec.json'
    spec_path.parent.mkdir(parents=True)
    study=tmp_path/'study'
    source=Path(__file__).resolve().parents[1]/'configs/relational_support_causality_v1/experiment_spec.json'
    spec=json.loads(source.read_text())
    spec['outputs']['planned_run_root']=str(study)
    spec_path.write_text(json.dumps(spec))
    config=repo/'configs/relational_support_causality_v1/evaluation.json'
    config.write_text('{}')
    panel=next(row for row in spec['evaluation']['stage1']['panels'] if row['id']==label)
    selection=study/'launch/stage1/selectors/stage1_full.json'
    selection.parent.mkdir(parents=True)
    def local(gid):
        return ('libero_90',gid-40) if gid>=40 else (
            ('libero_spatial','libero_object','libero_goal','libero_10')[gid//10],gid%10)
    full=[{'suite':local(gid)[0],'task_id':local(gid)[1],'init_state_id':state}
          for gid,state in panel['full_task_state_pairs']]
    manifest={'schema_version':'ember_pi05_registered_trajectory_capture_v1',
              'passive_control_trace':TAG,
              'study_spec':'configs/relational_support_causality_v1/experiment_spec.json',
              'stage_predicates':True,'full_conditions':full,
              'task_subset_selection':str(study/'launch/stage1/selectors/subset.json'),
              'mode':'compact', 'training_gradient_use':False, 'checkpoint_selection_use':False,
              'validation_use':False, 'test_use':False}
    selection.write_text(json.dumps(manifest))
    args=SimpleNamespace(config=config,role='nonheld_meta' if 'support' in panel['kind'] else 'development_train',
                         mode='formal' if len(panel['state_ids'])==50 else 'screen')
    tasks=[SimpleNamespace(suite=local(gid)[0],task_id=local(gid)[1],
                           init_state_ids=tuple(panel['state_ids'])) for gid in panel['task_ids']]
    subset={'selection_path':manifest['task_subset_selection']}
    return repo,study,selection,args,tasks,subset,manifest


def test_registration_requires_every_row_and_resume(tmp_path):
    repo,study,selection,args,tasks,subset,manifest=_registration(tmp_path)
    panel=study/'evaluation/Source_core_correct'
    capture,stage=prepare_from_manifest(args,repo_root=repo,output_dir=panel,
        task_subset=subset,tasks=tasks,manifest=manifest,selection_path=selection,
        full=tuple((row['suite'],row['task_id'],row['init_state_id']) for row in manifest['full_conditions']))
    assert capture['full_conditions']==manifest['full_conditions'] and stage['full_conditions_only'] is False
    with pytest.raises(Pi05EvaluationError,match='requires passive'):
        attach_requested_capture(args, {'diagnostic_occupancy_capture':None}, repo,
                                 panel)
    contract={'output_dir':str(panel),'git':{'commit':'evaluation_commit'},
              'adapter':None,'tasks':[{'suite':task.suite,'task_id':task.task_id,
                                       'init_state_ids':list(task.init_state_ids)} for task in tasks],
              'diagnostic_task_subset':subset,'diagnostic_occupancy_capture':capture,
              'diagnostic_stage_predicates':stage}
    attach_provenance(contract,repo)
    assert contract['passive_capture_provenance']['bank_manifest'] is None
    validate_contract(contract,repo)
    contract['diagnostic_stage_predicates']=None
    with pytest.raises(Pi05EvaluationError,match='all-row predicates'):
        validate_contract(contract,repo)
    contract.pop('diagnostic_occupancy_capture')
    with pytest.raises(Pi05EvaluationError,match='lacks passive'):
        validate_contract(contract,repo)


def test_stage1_registration_rejects_legacy_selector_directory(tmp_path):
    repo,study,selection,args,tasks,subset,manifest=_registration(tmp_path)
    old=study/'launch/selectors/stage1_full.json'
    old.parent.mkdir(parents=True)
    old.write_text(selection.read_text())
    full=tuple((row['suite'],row['task_id'],row['init_state_id'])
               for row in manifest['full_conditions'])
    with pytest.raises(Pi05EvaluationError,match='selection changed'):
        prepare_from_manifest(args,repo_root=repo,output_dir=study/'evaluation/Source_core_correct',
            task_subset=subset,tasks=tasks,manifest=manifest,selection_path=old,full=full)


def test_passive_registration_rejects_bank_from_other_training_commit(tmp_path):
    repo,study,selection,args,tasks,subset,manifest=_registration(tmp_path,'B_S00_1260_support_core')
    panel=study/'evaluation/B_S00_1260_support_core'
    capture,stage=prepare_from_manifest(args,repo_root=repo,output_dir=panel,
        task_subset=subset,tasks=tasks,manifest=manifest,selection_path=selection,
        full=tuple((row['suite'],row['task_id'],row['init_state_id']) for row in manifest['full_conditions']))
    adapter={'writer_checkpoint':{'training_commit':TRAINING},
             'registered_stage1_panel_id':'B_S00_1260_support_core',
             'materialization_git':{'commit':'another_commit'},
             'manifest':{'path':str(study/'materialization/B_smoke/manifest.json'),'bytes':12}}
    contract={'output_dir':str(panel),'git':{'commit':'evaluation_commit'},
              'adapter':adapter,'tasks':[{'suite':task.suite,'task_id':task.task_id,
                                          'init_state_ids':list(task.init_state_ids)} for task in tasks],
              'diagnostic_task_subset':subset,'diagnostic_occupancy_capture':capture,
              'diagnostic_stage_predicates':stage}
    with pytest.raises(Pi05EvaluationError,match='provenance'):
        attach_provenance(contract,repo)
    adapter['materialization_git']['commit']='evaluation_commit'
    attach_provenance(contract,repo)
    validate_contract(contract,repo)
    assert contract['passive_capture_provenance']['bank_manifest']==adapter['manifest']
    adapter['registered_stage1_panel_id']='C_S00_1260_support_core'
    with pytest.raises(Pi05EvaluationError,match='provenance'):
        attach_provenance(contract,repo)


def test_passive_registration_rejects_omitted_evaluation_update(tmp_path):
    repo,study,selection,args,tasks,subset,manifest=_registration(tmp_path,'B_S00_1260_support_core')
    with pytest.raises(Pi05EvaluationError,match='stage1 scope'):
        prepare_from_manifest(args,repo_root=repo,
            output_dir=study/'evaluation/B_S00_420_support',
            task_subset=subset,tasks=tasks,manifest=manifest,selection_path=selection,full=())


def test_passive_registration_allows_empty_full_only_outside_formal_stage(tmp_path):
    repo,study,selection,args,tasks,subset,manifest=_registration(tmp_path)
    manifest['full_conditions']=[]
    selection.write_text(json.dumps(manifest))
    capture,stage=prepare_from_manifest(args,repo_root=repo,output_dir=study/'smoke/empty_full',
        task_subset=subset,tasks=tasks,manifest=manifest,selection_path=selection,full=())
    assert capture['full_conditions']==[] and stage['full_conditions_only'] is False
    with pytest.raises(Pi05EvaluationError,match='full selection'):
        prepare_from_manifest(args,repo_root=repo,output_dir=study/'evaluation/Source_core_correct',
            task_subset=subset,tasks=tasks,manifest=manifest,selection_path=selection,full=())


def test_stage1_registration_rejects_missing_state(tmp_path):
    repo,study,selection,args,tasks,subset,manifest=_registration(tmp_path)
    tasks[0].init_state_ids=tasks[0].init_state_ids[:-1]
    full=tuple((row['suite'],row['task_id'],row['init_state_id']) for row in manifest['full_conditions'])
    with pytest.raises(Pi05EvaluationError,match='stage1 cases'):
        prepare_from_manifest(args,repo_root=repo,output_dir=study/'evaluation/Source_core_correct',
            task_subset=subset,tasks=tasks,manifest=manifest,selection_path=selection,full=full)
