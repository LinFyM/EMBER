"""Actual consumers for the bounded §33 manifest and algebraic readback."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from ember.lora import LORA_A_SUFFIX
from ember.operator_writer.model import TargetWrite


def script():
    path = Path(__file__).resolve().parents[2] / 'scripts/operator_learning_limit_diagnosis.py'
    spec = importlib.util.spec_from_file_location('learning_limit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_manifest_actual_seed_api_and_partition():
    module = script()
    panels = module.panels()
    lengths = {t: [200]*50 for t in module.TASKS}
    manifest = module.query_manifest(panels, lengths)
    assert manifest == module.query_manifest(panels, lengths)
    assert len(manifest['steps']) == 64
    for step, tasks in enumerate(manifest['steps']):
        for task in module.TASKS:
            row = tasks[str(task)]
            queries = row['queries']
            assert [q['demo'] for q in queries] == sorted(q['demo'] for q in panels[str(task)]['A'])
            assert not {q['demo'] for q in queries} & {q['demo'] for q in panels[str(task)]['B']}
            assert row['visit'] == 330000+step
            assert all(0 <= q['frame'] <= 198 for q in queries)
    assert manifest['steps'][0] != manifest['steps'][1]
    assert callable(module.prior_readout().b_batch)


def test_z_readback_through_actual_targetwrite():
    module = script()
    torch.manual_seed(5)
    write = TargetWrite(32, 16)
    torch.nn.init.normal_(write.o.weight, std=.02)
    address = torch.randn(128,32)
    x = torch.randn(4,50,32)
    h = torch.randn(4,50,1024)
    writer = SimpleNamespace(names=('site',), writes=(write,),
        public_state=lambda: {'site'+LORA_A_SUFFIX: address})
    z = module.latent_z(writer, ({'site':x}, h))['site']
    assert z.shape == (256,128)
    assert torch.allclose(write.o.weight @ z, write(address,x,h), atol=2e-5, rtol=2e-4)
