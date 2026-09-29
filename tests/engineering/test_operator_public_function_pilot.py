"""Real T1800 migration, fixed fifth-round events and isolated public FM credit."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.lora import identity_lora_state
from ember.operator_writer import bank
from ember.operator_writer.credit import apply_public_cotangent
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.model import OperatorReadWrite
from ember.operator_writer.run import (CONTINUATION1800_SPEC_PATH, PILOT_ARMS,
                                       PILOT_SPEC_PATH, specification,
                                       validate_attempt, validate_train_request)
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json

ASSET = Path('/data1/user/ymdai/projects/EMBER')
PARENT = Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928'
              '/continuation1800/T/train/attempts/continuation/checkpoints/macro_00001800')
GIT = {'commit':'new-pushed-freeze','branch':'','dirty_paths':[],
       'pushed_ref':'origin/codex/demonstration-transfer'}


def test_real_parent_and_fifth_round_append():
    previous = specification(CONTINUATION1800_SPEC_PATH)
    spec = specification(PILOT_SPEC_PATH)
    old = FormalData(ASSET, previous, query_labels=False)
    new = FormalData(ASSET, spec, query_labels=False)
    try:
        for step in (0, 269, 899, 1349, 1799):
            assert old.tasks_for_step(step) == new.tasks_for_step(step)
            for task in old.tasks_for_step(step):
                assert old.event(step, task) == new.event(step, task)
        for task in TASKS:
            observed = []
            for visit in range(200, 210):
                step = next(step for step in range(visit*9, visit*9+9)
                            if task in new.tasks_for_step(step))
                event = new.event(step, task)
                assert event['visit'] == visit and len(event['queries']) == 28
                assert all(q['demo'] != event['teacher_demo'] for q in event['queries'])
                observed.append(event['teacher_demo'])
            expected = np.random.default_rng(np.random.SeedSequence(
                [20260928,1,task,4])).permutation(50)[:10].tolist()
            assert observed == expected and len(set(observed)) == 10
        trainer = torch.load(PARENT/'trainer_state.pt', map_location='meta', mmap=True,
                             weights_only=True)
        assert trainer['next_macro'] == trainer['metrics_rows'] == 1800
        assert trainer['scheduler']['last_epoch'] == 1800
        assert trainer['optimizer']['param_groups'] and trainer['training_state'] == {
            'updates':1800,'mode':'T'}
        assert new.restore(trainer['sampler_state'], migrate_continuation_1800=True) == {
            'from_schema':'ember_operator_read_write_events_v5',
            'to_schema':'ember_operator_read_write_events_v6','cursor':1800,
            'appended_teacher_round':[20260928,1,'task',4]}
        prefix = [json.loads(line) for line in
                  (PARENT.parent.parent/'metrics.jsonl').read_text().splitlines()]
        assert len(prefix) == 1800 and [row['update'] for row in prefix] == list(range(1,1801))
        assert bank.inspect_training_source(previous,PARENT,'T',sealed_evaluation=True)['git']['commit'] == spec['continuation']['parent_training_git']
    finally:
        old.close(); new.close()


def test_direct_public_cotangent_has_no_write_credit():
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        ASSET/'configs/pi05_lora_v1.json'), rank=128)
    writer = OperatorReadWrite(lora, identity_lora_state(lora), 'T')
    public = writer.public_state()
    assert len(public) == 76
    cotangent = {name: torch.ones_like(value) for name,value in public.items()}
    apply_public_cotangent(writer,cotangent)
    assert all(value.grad is not None and torch.equal(value.grad, cotangent[name])
               for name,value in public.items())
    assert all(parameter.grad is None for unit in writer.writes for parameter in unit.parameters())
    with pytest.raises(ValueError,match='76 complete'):
        apply_public_cotangent(writer,dict(list(cotangent.items())[:-1]))


def test_pilot_parent_scope_and_loss_identity(tmp_path):
    spec = specification(PILOT_SPEC_PATH) | {'run_root':str(tmp_path)}
    data = FormalData(ASSET,spec,query_labels=False)
    try:
        original = read_json(PARENT.parent.parent/'run_contract.json')
        sampler = {k:v for k,v in data.sampler_state().items() if k!='next_step'}
        base = {**original,'git':GIT,'spec':str(PILOT_SPEC_PATH),
                'events':spec['events'],'sampler':sampler,'continuation':spec['continuation'],
                'parent_checkpoint':str(PARENT),'pilot':spec['pilot']}
        for arm,variant in PILOT_ARMS.items():
            contract = base | {'pilot_arm':arm,'loss_variant':variant}
            args = SimpleNamespace(mode='T',pilot_arm=arm,resume=PARENT,attempt='continuation',
                                   microbatch=28,frame_chunk=8,stop_after_macro=None)
            with pytest.raises(ValueError,match='T2790 continuation spec'):
                validate_train_request(spec,args)
            validate_attempt(spec,args,contract,tmp_path/arm/'train/attempts/continuation')
            with pytest.raises(ValueError,match='parent or arm'):
                validate_attempt(spec,args,contract | {'loss_variant':'wrong'},
                                 tmp_path/arm/'train/attempts/bad')
            with pytest.raises(ValueError,match='source or optimizer'):
                validate_attempt(spec,args,contract | {'source':{'checkpoint':'/wrong'}},
                                 tmp_path/arm/'train/attempts/badsource')
            with pytest.raises(ValueError,match='outside its registered arm'):
                validate_attempt(spec,args,contract,tmp_path/'other/train/attempts/wrong')
        with pytest.raises(ValueError,match='T2790 continuation spec'):
            validate_train_request(specification(CONTINUATION1800_SPEC_PATH),args)
        with pytest.raises(ValueError,match='registered T ECP or pilot1890'):
            bank.materialize('control',PARENT,ASSET,torch.device('cpu'))
    finally:
        data.close()


def test_new_1890_source_rejects_cross_arm_and_changed_loss(tmp_path, monkeypatch):
    spec = specification(PILOT_SPEC_PATH) | {'run_root':str(tmp_path)}
    data = FormalData(ASSET,spec,query_labels=False)
    try:
        sampler = {k:v for k,v in data.sampler_state().items() if k!='next_step'}
        old = read_json(PARENT.parent.parent/'run_contract.json')
        monkeypatch.setattr(bank,'frozen_git',lambda continuation: GIT)
        for arm,variant in PILOT_ARMS.items():
            output = tmp_path/arm/'train/attempts/continuation'
            checkpoint = output/'checkpoints/macro_00001890'
            checkpoint.mkdir(parents=True)
            run = {**old,'git':GIT,'spec':str(PILOT_SPEC_PATH),'events':spec['events'],
                   'sampler':sampler,'continuation':spec['continuation'],
                   'parent_checkpoint':str(PARENT),'pilot_arm':arm,
                   'loss_variant':variant,'pilot':spec['pilot']}
            (output/'run_contract.json').write_text(json.dumps(run))
            (output/'resume_provenance.json').write_text(json.dumps({
                'checkpoint':str(PARENT),'parent_git':old['git']}))
            prefix = (PARENT.parent.parent/'metrics.jsonl').read_text().splitlines()
            suffix = [json.dumps({'update':i,'pilot_arm':arm,'loss_variant':variant})
                      for i in range(1801,1891)]
            (output/'metrics.jsonl').write_text('\n'.join(prefix+suffix)+'\n')
            trainer = {'schema_version':'ember_ecp_checkpoint_v1',
                       'stage':'operator_read_write_learning','next_macro':1890,
                       'metrics_rows':1890,'optimizer':{'param_groups':[{'lr':1e-5}]},
                       'scheduler':{'last_epoch':1890},'scaler':None,
                       'training_state':{'updates':1890,'mode':'T','pilot_arm':arm,
                                         'loss_variant':variant},
                       'sampler_state':sampler|{'next_step':1890}}
            files = {name:{'bytes':1} for name in ('ecp.safetensors','rank_00_state.pt',
                                                   'rank_01_state.pt')}
            for name in files:
                (checkpoint/name).write_bytes(b'x')
            torch.save(trainer,checkpoint/'trainer_state.pt')
            files['trainer_state.pt']={'bytes':(checkpoint/'trainer_state.pt').stat().st_size}
            (checkpoint/'checkpoint_manifest.json').write_text(json.dumps({
                'stage':'operator_read_write_learning',
                'run_contract_schema':'ember_operator_read_write_formal_run_v1',
                'next_macro':1890,'world_size':2,'files':files}))
            assert bank.inspect_training_source(spec,checkpoint,'T')['pilot_arm']==arm
            with pytest.raises(ValueError,match='same-arm'):
                bank.inspect_training_source(spec,checkpoint,'U')
            (output/'run_contract.json').write_text(json.dumps(run|{'loss_variant':'wrong'}))
            with pytest.raises(ValueError,match='same-arm'):
                bank.inspect_training_source(spec,checkpoint,'T')
    finally:
        data.close()


def test_pilot_bank_reader_links_its_arm_to_the_training_source(tmp_path, monkeypatch):
    """Scope and factor I/O are separate; enforce the bank-to-run identity edge."""
    spec = specification(PILOT_SPEC_PATH)
    parent = read_json(PARENT.parent.parent/'run_contract.json')
    record = {'unit_fixture': True}
    monkeypatch.setattr(bank, 'file_record', lambda path: record)
    inspected = []
    monkeypatch.setattr(bank, '_factor_header', lambda *args, **kwargs: inspected.append(args))
    for arm, variant in PILOT_ARMS.items():
        source_run = parent | {'pilot_arm': arm, 'loss_variant': variant}
        monkeypatch.setattr(bank, 'inspect_training_source', lambda *args, **kwargs: source_run)
        value = {'mode': arm, 'checkpoint': str(tmp_path/'checkpoints/macro_00001890'),
                 'asset_root': str(ASSET), 'shared': record, 'checkpoint_manifest': record,
                 'training_git': parent['git']['commit'], 'lora': parent['lora'],
                 'source': parent['source'], 'conditions': [], 'loss_variant': variant}
        bank._inspect_tu_bank(value, spec, tmp_path/'banks/1890/manifest.json')
        inspected.clear()
        other = next(name for name in PILOT_ARMS if name != arm)
        with pytest.raises(ValueError, match='pilot bank arm or loss source'):
            bank._inspect_tu_bank(value | {'mode': other, 'loss_variant': PILOT_ARMS[other]},
                                  spec, tmp_path/'banks/1890/manifest.json')
        assert inspected == []


@pytest.mark.parametrize("arm", ["control", "public_aux"])
def test_sealed_1890_bank_remains_readable_after_source_code_advances(arm):
    path = bank.PILOT_ROOT / arm / "banks/1890/manifest.json"
    value = read_json(path)
    result = bank.inspect_bank(
        manifest_path=path, source=value["source"],
        task_keys=tuple((row["suite"], row["task_id"]) for row in value["tasks"]),
        evaluation_role="validation", require_formal=True,
    )
    assert result["mode"] == arm and len(result["conditions"]) == 400
    assert Path(result["spec"]["path"]) == bank.PILOT_FROZEN_SPEC_PATH
    assert result["training_git"] == bank.PILOT_TRAINING_GIT["commit"]
