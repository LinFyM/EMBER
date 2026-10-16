"""One manifest-scoped uncentered CPU fit; excluded controllers are never opened."""
from pathlib import Path
from types import SimpleNamespace
import datetime,math,os,resource,subprocess,time
import torch

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.proposal_writer.model import FactorLayout
from ember.reference.accounting import host_resources
from ember.reference.contract import ASSETS
from .contract import BASIS_SCHEMA
from .learning import References

MANIFEST_SCHEMA='ember_control_task_exclusion8_fit_manifest_v1'
FIT_TASKS=(2,4,5,7,14,15,17,19,22,25,28,29,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101)
EXCLUDED_TASKS=(0,1,12,13,20,21,32,34)
CENTER_REFERENCE=Path('/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/control_linear64_seed7/basis/state.pt')


def validate_manifest(manifest):
    if manifest['schema']!=MANIFEST_SCHEMA or Path(manifest['source_center_checkpoint'])!=CENTER_REFERENCE:
        raise ValueError('fit requires the frozen task-exclusion manifest and exact saved source center')
    if (len(manifest['fit_tasks'])!=28 or set(manifest['fit_tasks'])!=set(FIT_TASKS)
            or len(manifest['excluded_tasks'])!=8 or set(manifest['excluded_tasks'])!=set(EXCLUDED_TASKS)):
        raise ValueError('whole28 fitting tasks and excluded8 must retain the predeclared identities')
    points=manifest['points']
    if len(points)!=208 or len({(p['task'],p['update']) for p in points})!=208:
        raise ValueError('fit requires208 distinct task/update points')
    if any(p['task'] not in FIT_TASKS or p['point'] not in (0,1) for p in points):raise ValueError('excluded or unregistered point entered fit')
    if sum(p['path'] is None for p in points)!=5 or any((p['path'] is None)!=(p['update']==0) for p in points):
        raise ValueError('the five source identities and203 nonzero points must retain their mass')
    validate_task_weights(points)
    return points


def validate_task_weights(points):
    for task in FIT_TASKS:
        group=[p for p in points if p['task']==task];selected=[p for p in group if p['point']==0];others=[p for p in group if p['point']==1]
        if len(selected)!=1:raise ValueError('each fitting task needs exactly one selected endpoint')
        for p in group:
            expected=(.5 if p['point']==0 else .5/len(others))/28 if others else 1./28
            if not math.isfinite(p['weight']) or not math.isclose(p['weight'],expected,rel_tol=0,abs_tol=1e-12):
                raise ValueError('task/selected/other weights changed; no identity reweighting is permitted')


def gram_spectrum(matrix):
    """Original FP64 small-Gram decomposition and numerical rank definition."""
    if matrix.dtype!=torch.float64 or matrix.ndim!=2 or matrix.device.type!='cpu':raise ValueError('decomposition expects FP64 CPU rows')
    gram=matrix@matrix.T;gram=(gram+gram.T)*.5
    eigenvalues,left=torch.linalg.eigh(gram)
    order=torch.arange(len(eigenvalues)-1,-1,-1);eigenvalues=eigenvalues[order];left=left[:,order]
    tolerance=len(eigenvalues)*torch.finfo(gram.dtype).eps*float(eigenvalues[0].clamp_min(0))
    if float(eigenvalues[-1]) < -64*tolerance:raise ValueError('small Gram has an invalid negative spectrum')
    eigenvalues=eigenvalues.clamp_min(0);rank=int((eigenvalues>tolerance).sum());effective=min(64,rank);total=float(eigenvalues.sum())
    return dict(eigenvalues=eigenvalues,left_vectors=left,numerical_rank=rank,effective_rank=effective,
        rank_tolerance=tolerance,rank_definition='lambda > n_rows * eps_FP64 * lambda_max',
        retained_spectral_mass=float(eigenvalues[:effective].sum())/total if total else 0.,weighted_squared_norm=total)


def right_basis(matrix,spectrum,code_dim=64):
    q=min(code_dim,spectrum['numerical_rank']);basis=matrix.new_zeros((matrix.shape[1],code_dim))
    if q:
        basis[:,:q]=(matrix.T@spectrum['left_vectors'][:,:q])/spectrum['eigenvalues'][:q].sqrt()
        error=float((basis[:,:q].T@basis[:,:q]-torch.eye(q,dtype=basis.dtype)).abs().max())
        if error>1e-3:raise ValueError('identified right singular vectors are not orthonormal')
    else:error=0.
    return basis.float(),error


def atomic_torch(path,state):
    temporary=path.with_name(path.name+f'.partial_{os.getpid()}')
    torch.save(state,temporary);temporary.replace(path)


def unpack_center(center,parts,rank):
    source={}
    for name,side,width,blocks,start,stop in parts:
        value=center[start:stop].reshape(rank,blocks*64)[:,:width]
        source[name]=(value if side==0 else value.T).clone().contiguous()
    return source


def saved_layout(manifest):
    # Mmap only these two designated fields: never touch old U, units or code_units.
    saved=torch.load(Path(manifest['source_center_checkpoint']),map_location='cpu',weights_only=False,mmap=True)
    center=saved['center'].clone();metadata=saved['layout'];del saved
    lora=derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)
    if (lora.rank,lora.alpha,lora.dropout,len(lora.targets))!=(8,8,0.,38):raise ValueError('fit must preserve full38/r8/alpha8/dropout0')
    source=unpack_center(center,metadata['parts'],8);layout=FactorLayout(lora,source)
    expected=dict(valid_coordinates=643584,padding_coordinates=512,rank=8,targets=38)
    if any(metadata.get(k)!=v for k,v in expected.items()) or list(map(tuple,metadata['parts']))!=layout.parts:
        raise ValueError('saved source center/layout differs from the complete original coordinates')
    if center.shape!=layout.center.shape or not bool(center.isfinite().all()):raise ValueError('invalid complete saved source center')
    if any(torch.count_nonzero(source[name]) for name,side,*_ in layout.parts if side==1):raise ValueError('saved source identity must retain B=0')
    layout.center.copy_(center)
    return lora,layout,source,metadata


def tensor_units(points,references,source):
    moments={name:0. for name in source}
    for point in points:
        state=references.get(point['path'])
        for name,center in source.items():
            if state[name].shape!=center.shape:raise ValueError('factor shape changed from the saved source layout')
            moments[name]+=point['weight']*float((state[name].double()-center.double()).square().mean())
    units={name:max(math.sqrt(value),1e-6) for name,value in moments.items()}
    if any(not math.isfinite(value) for value in units.values()):raise ValueError('nonfinite fitting tensor units')
    return units


def normalized_bank(manifest):
    lora,layout,source,metadata=saved_layout(manifest);points=manifest['points']
    refs=References(SimpleNamespace(device=torch.device('cpu'),identity=source),layout.names,capacity=len(points))
    units=tensor_units(points,refs,source)
    for name,_,_,_,start,stop in layout.parts:layout.scale[start:stop].fill_(units[name])
    vectors=torch.stack([((layout.pack(refs.get(p['path']))-layout.center)/layout.scale)[layout.valid] for p in points])
    if vectors.shape!=(208,643584) or not bool(vectors.isfinite().all()):raise ValueError('invalid manifest-scoped normalized bank')
    return dict(vectors=vectors,units=units,center=layout.center,layout=metadata)


def factor_residuals(vectors,basis,points):
    rows=[]
    for first in range(0,len(points),16):
        part=vectors[first:first+16];projected=(part@basis)@basis.T
        source=part.double().square().sum(-1);error=(projected.double()-part.double()).square().sum(-1)
        for p,s,e in zip(points[first:first+16],source.tolist(),error.tolist(),strict=True):
            rows.append(dict(p,factor_squared_error=e,factor_source_squared_norm=s,factor_relative_error=e/s if s else None))
    baseline=sum(r['weight']*r['factor_source_squared_norm'] for r in rows);error=sum(r['weight']*r['factor_squared_error'] for r in rows)
    return dict(rows=rows,points=len(rows),weighted_factor_squared_error=error,weighted_factor_source_squared_norm=baseline,
        weighted_factor_relative_error=error/baseline if baseline else None,zero_denominator='absolute error retained; relative error is null')


def completed_basis(root,manifest):
    destination=root/'basis'
    if not destination.exists():
        complete=[p for p in root.glob('basis.partial_*') if (p/'manifest.json').is_file() and (p/'state.pt').is_file()]
        if len(complete)>1:raise ValueError('multiple completed basis transactions require reconciliation')
        if not complete:return None
        transaction=read_json(complete[0]/'manifest.json')
        if transaction.get('schema')!=BASIS_SCHEMA or transaction.get('fit_manifest')!=manifest or not transaction.get('complete'):
            raise ValueError('partial basis belongs to another fit')
        complete[0].rename(destination)
    record=read_json(destination/'manifest.json')
    if (record['schema']!=BASIS_SCHEMA or not record['complete'] or record.get('fit_manifest')!=manifest
            or not (destination/'state.pt').is_file()):raise ValueError('completed basis transaction lost its frozen manifest')
    if not (root/'fit_completion.json').exists():write_json_atomic(root/'fit_completion.json',record['fit'])
    return read_json(root/'fit_completion.json')


def save_basis(root,state,record,manifest):
    destination=root/'basis'
    if destination.exists():raise ValueError('completed basis must never be overwritten or refitted')
    temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}');temporary.mkdir(parents=True,exist_ok=False)
    torch.save(state,temporary/'state.pt')
    write_json_atomic(temporary/'manifest.json',dict(schema=BASIS_SCHEMA,phase='linear64',complete=True,code_dim=64,
        effective_rank=state['effective_rank'],optimizer_updates=0,method=state['method'],fit_manifest=manifest,fit=record))
    temporary.rename(destination);write_json_atomic(root/'fit_completion.json',record)


def fit(root):
    root=Path(root);manifest=read_json(root/'fit_manifest.json');points=validate_manifest(manifest)
    adopted=completed_basis(root,manifest)
    if adopted is not None:return adopted
    torch.set_num_threads(16);started=time.monotonic();start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat();samples=dict(start=host_resources())
    bank_file=root/'normalized_bank.pt';stage=root/'decomposition.pt'
    if bank_file.exists():
        bank=torch.load(bank_file,map_location='cpu',weights_only=False,mmap=True)
        if bank['fit_manifest']!=manifest:raise ValueError('saved normalized bank belongs to another manifest')
    else:
        if stage.exists():raise ValueError('completed decomposition lost its normalization state; never silently refit')
        bank=dict(normalized_bank(manifest),fit_manifest=manifest);atomic_torch(bank_file,bank)
    vectors=bank['vectors'];matrix=vectors.double();matrix.mul_(torch.tensor([p['weight'] for p in points],dtype=torch.float64).sqrt()[:,None])
    samples['matrix_loaded']=host_resources()
    if stage.exists():
        decomposition=torch.load(stage,map_location='cpu',weights_only=False)
        if decomposition['fit_manifest']!=manifest:raise ValueError('saved decomposition belongs to another manifest')
        spectrum=decomposition['spectrum']
    else:
        spectrum=gram_spectrum(matrix);atomic_torch(stage,dict(fit_manifest=manifest,spectrum=spectrum))
    basis,orthogonality=right_basis(matrix,spectrum);del matrix;samples['basis_computed']=host_resources()
    diagnostics=factor_residuals(vectors,basis,points)
    record=dict(complete=True,basis_path=str(root/'basis/state.pt'),scope='whole-task-exclusion8-fit28-points208',
        method='weighted_uncentered_small_Gram_FP64',fit_manifest=str(root/'fit_manifest.json'),fit_tasks=manifest['fit_tasks'],excluded_tasks=manifest['excluded_tasks'],
        start_UTC=start_UTC,end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
        code_dim=64,effective_rank=spectrum['effective_rank'],numerical_rank=spectrum['numerical_rank'],spectrum=spectrum['eigenvalues'].sqrt().tolist(),
        retained_spectral_mass=spectrum['retained_spectral_mass'],rank_tolerance=spectrum['rank_tolerance'],rank_definition=spectrum['rank_definition'],
        orthogonality_max_abs_FP64=orthogonality,optimizer_updates=0,GPU_hours=0.,mean_subtracted=False,
        peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,resource_samples=samples,
        code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip(),
        torch_version=torch.__version__,CPU_threads=torch.get_num_threads(),factor_diagnostics=diagnostics,
        full_training_checkpoint=False,optimizer_or_RNG_resume=False,wall_seconds_includes_basis_write=False)
    state=dict(schema=BASIS_SCHEMA,phase='linear64',optimizer_updates=0,U=basis,center=bank['center'],units=bank['units'],
        layout=bank['layout'],fit_manifest=manifest,points=points,scope=record['scope'],code_dim=64,effective_rank=spectrum['effective_rank'],
        spectrum=record['spectrum'],method=record['method'],fit=record,source_center_checkpoint=manifest['source_center_checkpoint'])
    writing=time.monotonic();save_basis(root,state,record,manifest)
    record.update(compute_wall_seconds=record['wall_seconds'],basis_write_seconds=time.monotonic()-writing,
        wall_seconds=time.monotonic()-started,end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds_includes_basis_write=True)
    write_json_atomic(root/'fit_completion.json',record);return record
