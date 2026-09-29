"""Frozen §28 three-arm removal of T2340 conditional M on existing B20 queries."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/operator_branch_intervention_20260929')
PRIOR = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/functional_credit_transport')
PARENT = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/self_conditioned_readout')
ASSET = Path('/data1/user/ymdai/projects/EMBER')
NUMERIC = Path('/data1/user/ymdai/projects/EMBER-operator-continuation2340-formal')
ECP = Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340')
NUMERIC_GIT = 'e2afbfd7c997e3f792921600608efa2fa3c1b25a'
TASK_GROUPS = ((0, 12), (20, 32))
ARMS = ('no_q_write', 'no_v_write', 'no_io_write')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def atomic_tensor(path, value, torch):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def target_groups(names):
    groups = {
        'no_q_write': {name for name in names if name.endswith('.self_attn.q_proj')},
        'no_v_write': {name for name in names if name.endswith('.self_attn.v_proj')},
        'no_io_write': {'model.action_in_proj', 'model.action_out_proj'},
    }
    if ([len(groups[arm]) for arm in ARMS] != [18, 18, 2]
            or set.union(*groups.values()) != set(names)
            or sum(map(len, groups.values())) != len(names)):
        raise ValueError('complete disjoint 18 Q / 18 V / 2 IO target contract changed')
    return groups


def branch_state(public, memory, names, removed, torch, a_suffix, b_suffix):
    state = dict(public)
    if set(memory) != set(names) or len(names) != 38:
        raise ValueError('saved parent does not contain all 38 condition memories')
    for name in names:
        a, b = name + a_suffix, name + b_suffix
        if a not in public or b not in public:
            raise ValueError('parent public factor missing')
        m = memory[name].to(device=public[b].device, dtype=torch.float32)
        if not torch.isfinite(m).all():
            raise ValueError('saved parent M has nonfinite values')
        state[b] = public[b].float() if name in removed else public[b].float() + m
    return state


def load_b20(data, runtime, task, panel, torch, b_batch):
    if len(panel['teachers']) != 2 or len(panel['B']) != 20:
        raise ValueError('fixed two-teacher B20 panel changed')
    reference_path = PRIOR / f'group0/task{task:03d}_B_query_noise_target.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    observation, noise, target, valid = b_batch(
        data, runtime, task, panel['B'], reference, torch=torch)
    return reference_path, observation, noise, target, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-analysis-git', required=True)
    parser.add_argument('--deadline-seconds', type=int, default=1050)
    args = parser.parse_args()
    if (args.output.resolve() != (ROOT / f'group{args.group}').resolve()
            or args.output.exists() or not 0 < args.deadline_seconds <= 1050):
        raise ValueError('fresh bounded §28 group output required')
    start = time.monotonic()
    analysis_tree = Path(__file__).resolve().parents[1]
    actual_git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=analysis_tree, text=True).strip()
    if (actual_git != args.expected_analysis_git
            or subprocess.run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=analysis_tree,
                              capture_output=True).returncode == 0
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=analysis_tree,
                                       text=True).strip()):
        raise ValueError('analysis script must run from the selected clean detached commit')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=NUMERIC, text=True).strip() != NUMERIC_GIT
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=NUMERIC, text=True).strip()):
        raise ValueError('numerical e2 frozen tree changed')

    sys.path.insert(0, str(PARENT))
    sys.path.insert(0, str(NUMERIC / 'src'))
    import torch
    from safetensors.torch import load_file
    from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
    from ember.operator_writer.data import FormalData
    from ember.operator_writer.run import CONTINUATION2340_SPEC_PATH, build_runtime, specification
    from analysis_script_v2 import b_batch, generated, risk

    torch.set_num_threads(4)
    contract = json.loads((ECP.parent.parent / 'run_contract.json').read_text())
    manifest = json.loads((ECP / 'checkpoint_manifest.json').read_text())
    if (contract['git']['commit'] != NUMERIC_GIT or contract['mode'] != 'T'
            or contract.get('loss_variant') != 'full' or manifest['next_macro'] != 2340):
        raise ValueError('T2340 complete parent lineage changed')
    args.output.mkdir(parents=True)
    (args.output / 'analysis_script.py').write_text(Path(__file__).read_text())
    atomic_json(args.output / 'run_contract.json', {
        'task': 'operator_branch_intervention_20260929', 'design': 'active §28',
        'group': args.group, 'tasks': TASK_GROUPS[args.group], 'arms': ARMS,
        'analysis_git': actual_git, 'numerical_git': NUMERIC_GIT,
        'numerical_tree': str(NUMERIC), 'parent_ecp': str(ECP),
        'saved_M_root': str(PARENT), 'B20_root': str(PRIOR),
        'consumer': 'existing flow_actions, 10 steps, 50x32, microbatch10',
        'optimizer_updates': 0, 'environment_episodes': 0, 'new_held_analysis': False,
        'host': socket.gethostname(), 'deadline_seconds': args.deadline_seconds,
    })
    data, rows = None, []
    try:
        spec = specification(CONTINUATION2340_SPEC_PATH)
        data = FormalData(ASSET, spec, task_ids=TASK_GROUPS[args.group])
        runtime = build_runtime(ASSET, spec, torch.device('cuda:0'), 'T', evaluation=False)
        runtime.writer.load_state_dict(load_file(str(ECP / 'ecp.safetensors'), device='cuda:0'), strict=True)
        runtime.policy.eval()
        runtime.writer.eval()
        names = runtime.writer.names
        removed_by_arm = target_groups(names)
        public = {name: value.detach() for name, value in runtime.writer.public_state().items()}
        panels = json.loads((PRIOR / 'group0/fixed_panels.json').read_text())
        for task in TASK_GROUPS[args.group]:
            if time.monotonic() - start > args.deadline_seconds - 180:
                raise TimeoutError('bounded branch batch stops before next task')
            panel = panels[str(task)]
            b_ref, observation, noise, target, valid = load_b20(
                data, runtime, task, panel, torch, b_batch)
            parent_group = PARENT / ('group0_attempt2' if args.group == 0 else 'group1')
            for teacher in panel['teachers']:
                parent_ref = parent_group / f'task{task:03d}_teacher{teacher:02d}_parent.pt'
                saved = torch.load(parent_ref, map_location='cpu', weights_only=False)
                if (saved['teacher'] != teacher or not saved['w0_A_is_parent_A']
                        or set(saved['M0']) != set(names)):
                    raise ValueError('saved original one-pass teacher memory changed')
                for arm in ARMS:
                    removed = removed_by_arm[arm]
                    state = branch_state(public, saved['M0'], names, removed, torch,
                                         LORA_A_SUFFIX, LORA_B_SUFFIX)
                    validate_lora_state(state, runtime.lora)
                    runtime.restore_identity()
                    full = generated(runtime, state, observation, noise, 10, torch)
                    prediction = full[:, :, :7]
                    score = risk(prediction, target, valid, torch)
                    raw = args.output / f'task{task:03d}_teacher{teacher:02d}_{arm}.pt'
                    atomic_tensor(raw, {
                        'task': task, 'teacher': teacher, 'arm': arm,
                        'removed_M_targets': sorted(removed),
                        'kept_M_targets': sorted(set(names) - removed),
                        'parent_M_ref': str(parent_ref), 'B_query_noise_target': str(b_ref),
                        'queries': panel['B'], 'microbatch': 10,
                        'full_latent50x32': full, 'prediction50x7': prediction,
                        'risk': score,
                    }, torch)
                    rows.append({'task': task, 'teacher': teacher, 'arm': arm,
                                 'removed_M_targets': sorted(removed),
                                 'B_query_noise_target': str(b_ref),
                                 'queries': panel['B'], 'risk': score, 'raw': str(raw)})
                    atomic_json(args.output / 'rows.json', rows)
                    del state, full, prediction
            torch.cuda.empty_cache()
        if (len(rows) != 12 or {row['arm'] for row in rows} != set(ARMS)
                or runtime.physical_delta() != 0):
            raise ValueError('twelve paths or frozen source identity changed')
        atomic_json(args.output / 'completion.json', {
            'status': 'complete', 'group': args.group, 'new_paths': 12,
            'optimizer_updates': 0, 'environment_episodes': 0,
            'elapsed_seconds': time.monotonic()-start,
            'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        })
    except BaseException:
        atomic_json(args.output / 'failure.json', {
            'status': 'failed', 'completed_paths': len(rows),
            'elapsed_seconds': time.monotonic()-start,
            'traceback': traceback.format_exc(),
        })
        raise
    finally:
        if data is not None:
            data.close()


if __name__ == '__main__':
    main()
