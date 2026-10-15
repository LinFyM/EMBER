"""One weighted uncentered CPU decomposition and finite basis; never an optimizer."""
from pathlib import Path
from types import SimpleNamespace
import datetime,os,resource,subprocess,time
import torch
from ember.lora import identity_lora_state
from ember.proposal_writer.model import FactorLayout
from ember.reference.accounting import host_resources
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from .contract import SCHEMA,PARENT_ROOT
from .data import topology
from .model import ControlSpace,load_basis
from .learning import References,OperatorLoss,pool_points,evaluate_pool
from .checkpoint import save_basis,completed_basis


def gram_spectrum(matrix):
    """FP64 small-Gram equivalent of one thin SVD; no bank mean subtraction."""
    if matrix.dtype!=torch.float64 or matrix.ndim!=2:raise ValueError('decomposition expects FP64 rows')
    gram=matrix@matrix.T;gram=(gram+gram.T)*.5
    eigenvalues,left=torch.linalg.eigh(gram)
    order=torch.arange(len(eigenvalues)-1,-1,-1);eigenvalues=eigenvalues[order];left=left[:,order]
    tolerance=len(eigenvalues)*torch.finfo(gram.dtype).eps*float(eigenvalues[0].clamp_min(0))
    if float(eigenvalues[-1]) < -64*tolerance:raise ValueError('small Gram has an invalid negative spectrum')
    eigenvalues=eigenvalues.clamp_min(0);rank=int((eigenvalues>tolerance).sum())
    effective=min(64,rank);total=float(eigenvalues.sum())
    return dict(eigenvalues=eigenvalues,left_vectors=left,numerical_rank=rank,effective_rank=effective,
        rank_tolerance=tolerance,rank_definition='lambda > n_rows * eps_FP64 * lambda_max',
        retained_spectral_mass=float(eigenvalues[:effective].sum())/total if total else 0.,
        weighted_squared_norm=total)


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


def normalized_bank(root):
    lora=topology();units=read_json(root/'units.json')
    # Mmap touches only the saved layout center; no neural model/Adam is instantiated.
    parent=torch.load(PARENT_ROOT/'checkpoints/update_00000144/state.pt',map_location='cpu',weights_only=False,mmap=True)
    center=parent['model']['layout.center'].clone();del parent
    layout=FactorLayout(lora,identity_lora_state(lora),units);layout.center.copy_(center)
    refs=References(SimpleNamespace(device=torch.device('cpu'),identity=layout.unpack(center)),layout.names)
    points=pool_points(read_json(root/'bank.json'));vectors=[]
    for p in points:
        blocks=(layout.pack(refs.get(p['path']))-center)/layout.scale
        vectors.append(blocks[layout.valid])
    vectors=torch.stack(vectors)
    if vectors.shape!=(271,643584) or not bool(vectors.isfinite().all()):raise ValueError('invalid complete normalized bank')
    return lora,units,layout,refs,points,vectors


def fit(root):
    root=Path(root);torch.set_num_threads(16);started=time.monotonic()
    start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat();samples=dict(start=host_resources())
    if completed_basis(root):
        model=load_basis(root/'basis/state.pt');record=torch.load(root/'basis/state.pt',map_location='cpu',weights_only=False)['fit']
        if not (root/'fit_completion.json').exists():write_json_atomic(root/'fit_completion.json',record)
        refs=References(SimpleNamespace(device=torch.device('cpu'),identity=model.layout.unpack(model.layout.center)),model.layout.names)
        vectors=None
    else:
        lora,units,layout,refs,points,vectors=normalized_bank(root)
        weights=torch.tensor([p['weight'] for p in points],dtype=torch.float64)
        matrix=vectors.double();matrix.mul_(weights.sqrt()[:,None]);samples['matrix_loaded']=host_resources()
        stage=root/'decomposition.pt'
        if stage.exists():spectrum=torch.load(stage,map_location='cpu',weights_only=False)
        else:
            spectrum=gram_spectrum(matrix);atomic_torch(stage,spectrum)
        U,orthogonality=right_basis(matrix,spectrum);samples['basis_computed']=host_resources();del matrix
        model=ControlSpace(lora,layout.center,units,U)
        record=dict(complete=True,basis_path=str(root/'basis/state.pt'),method='weighted_uncentered_small_Gram_FP64',
            wall_seconds=time.monotonic()-started,start_UTC=start_UTC,end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            code_dim=64,effective_rank=spectrum['effective_rank'],numerical_rank=spectrum['numerical_rank'],
            spectrum=spectrum['eigenvalues'].sqrt().tolist(),retained_spectral_mass=spectrum['retained_spectral_mass'],
            rank_tolerance=spectrum['rank_tolerance'],rank_definition=spectrum['rank_definition'],
            orthogonality_max_abs_FP64=orthogonality,optimizer_updates=0,GPU_hours=0.,
            peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,resource_samples=samples,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            torch_version=torch.__version__,CPU_threads=torch.get_num_threads(),mean_subtracted=False)
        state=dict(schema=SCHEMA,phase='linear64',optimizer_updates=0,U=U,center=layout.center,units=units,
            bank=read_json(root/'bank.json'),points=points,code_dim=64,effective_rank=spectrum['effective_rank'],
            spectrum=record['spectrum'],method=record['method'],fit=record,
            operator_constants=read_json(root/'operator_constants.json'),
            layout=dict(parts=layout.parts,valid_coordinates=layout.count,padding_coordinates=512,rank=8,targets=38),
            source_center_checkpoint=str(PARENT_ROOT/'checkpoints/update_00000144/state.pt'))
        writing=time.monotonic();save_basis(root,state)
        record.update(basis_write_seconds=time.monotonic()-writing,compute_wall_seconds=record['wall_seconds'],wall_seconds=time.monotonic()-started)
        write_json_atomic(root/'fit_completion.json',record)
    diagnostic=root/'diagnostics/pool.json'
    if not diagnostic.exists():
        started=time.monotonic();result=evaluate_pool(model,refs,OperatorLoss(model.lora,read_json(root/'operator_constants.json')),
            read_json(root/'bank.json'),vectors=vectors)
        result['read_wall_seconds']=time.monotonic()-started
        write_json_atomic(diagnostic,result)
    return read_json(root/'fit_completion.json')
