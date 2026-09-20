"""Role changes remain tied to the audited canonical dataset identities."""
import json
from pathlib import Path
import shutil

import pytest

from ember.task_protocol import load_task_authorities

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'configs/libero_24_8_8_coverage_v1/protocol.json'


def test_registered_coverage_has_36_train_and_disjoint_held_roles():
    _, manifest = load_task_authorities(ROOT, PROTOCOL)
    roles = manifest['summary']['roles']
    assert {k: len(v) for k, v in roles.items()} == {'train': 36, 'validation': 8, 'test': 8}
    assert not set(roles['train']) & (set(roles['validation']) | set(roles['test']))


@pytest.mark.parametrize('damage', ['held_role', 'identity', 'auxiliary'])
def test_coverage_authority_rejects_role_and_identity_changes(tmp_path, damage):
    paths = [PROTOCOL, 'configs/libero_24_8_8_coverage_v1/manifest.json',
             'configs/pi05_target_data_v1/manifest.json',
             'configs/pi05_source_corpus_v1/source_manifest.json']
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    path = tmp_path / (PROTOCOL if damage == 'auxiliary' else paths[1])
    value = json.loads(path.read_text())
    if damage == 'auxiliary':
        value['auxiliary_train']['task_ids'].append(1000)
    elif damage == 'identity':
        value['tasks'][0]['language'] = 'different task'
    else:
        row = next(r for r in value['tasks'] if r['split_role'] == 'test')
        row['split_role'] = 'train'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_task_authorities(tmp_path, PROTOCOL)
