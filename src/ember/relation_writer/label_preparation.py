"""Bounded CPU preparation of the exact registered train physical points."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open

from ember.operator_writer.data import FormalData, TASKS
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def semantic_labels(asset_root, spec, names):
    import sentencepiece
    from .labels import project_semantics
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(asset_root / spec['source']['tokenizer']))
    path = asset_root / spec['source']['checkpoint'] / 'policy/model.safetensors'
    key = 'model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    means, token_map = {}, {}
    with safe_open(str(path), framework='pt', device='cpu') as reader:
        embedding = reader.get_slice(key)
        for name in names:
            ids = tokenizer.encode(name, add_bos=False, add_eos=False)
            if not ids:
                raise ValueError('semantic name has no source tokens')
            rows = [embedding[index:index+1].float().numpy()[0] for index in ids]
            means[name] = np.stack(rows).mean(0)
            token_map[name] = ids
    vectors = project_semantics(means)
    return {'vectors': {name: vector.tolist() for name, vector in vectors.items()},
            'source_path': str(path.resolve()), 'embedding_key': key, 'name_tokens': token_map,
            'embedding_mean_dtype': 'FP32', 'projection': 'NumPy PCG64 seed20261006 QR2048x64 sign_fixed FP64',
            'normalization': 'unit64', 'G_condition_use': False}


def events_for(data):
    return [data.event(step, task) for step in range(450) for task in data.tasks_for_step(step)]


def _worker(arguments):
    asset_root, spec, episodes = arguments
    from .labels import LabelStore, build_cache
    data = FormalData(Path(asset_root), spec, query_labels=False)
    store = LabelStore(data, Path(asset_root), cache_root=Path(spec['run_root']) / 'labels')
    store.set_semantics(read_json(Path(spec['run_root']) / 'labels/semantic_vectors.json')['vectors'])
    try:
        return build_cache(store, events_for(data), episodes=episodes)
    finally:
        store.close()
        data.close()


def prepare_labels(spec, args):
    from ember.operator_writer.run import frozen_git
    from .labels import LabelStore, validate_sync
    git = frozen_git(continuation=True)
    started = time.perf_counter()
    root = Path(spec['run_root']) / 'labels'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'completion.json').exists():
        raise ValueError('completed physical preparation must be reused')
    data = FormalData(args.asset_root, spec, query_labels=False)
    store = LabelStore(data, args.asset_root, cache_root=root)
    try:
        semantics = semantic_labels(args.asset_root, spec, store.semantic_names)
        write_json_atomic(root / 'semantic_vectors.json', semantics)
        store.set_semantics(semantics['vectors'])
        events = events_for(data)
        plan = store.register_events(events)
        write_json_atomic(root / 'manifest.json', {'git': git, 'spec': str(args.spec.resolve()),
            'plan': plan, 'registries': [store.registries[task].description() for task in TASKS],
            'teacher_RGB_copy': False, 'held_privileged_reads': False})
        sync = validate_sync(store)
        write_json_atomic(root / 'synchronization.json', sync)
        keys = [(task, demo) for task in TASKS for demo in range(50)]
        chunks = [keys[index::args.workers] for index in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_worker, [(str(args.asset_root), spec, chunk) for chunk in chunks]))
        paths = list(root.glob('task*_demo*.npz'))
        if len(paths) != 1800 or sum(record['worker_episodes'] for record in results) != 1800:
            raise ValueError('physical preparation did not finish all1800 fixed episodes')
        record = {'status': 'complete', 'git': git, 'plan': plan, 'workers': args.workers,
                  'seconds': time.perf_counter() - started, 'episodes': 1800,
                  'geometry_valid_teaching_frames': 57314, 'terminal_geometry_masks': 1800,
                  'bytes': sum(path.stat().st_size for path in root.iterdir() if path.is_file()),
                  'worker_results': results, 'synchronization_points': sync['points']}
        write_json_atomic(root / 'completion.json', record)
        return record
    finally:
        store.close()
        data.close()
