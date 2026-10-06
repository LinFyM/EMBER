"""Reuse the native operator runtime with one relation-conditioned complete Writer."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.operator_writer.conditional_read_write import ConditionalTarget
from ember.operator_writer.run import build_runtime as build_operator_runtime
from ember.writer.model import DirectLoRAParameters

ASSET_ROOT = Path('/data1/user/ymdai/projects/EMBER')


class RelationWriter(nn.Module):
    """Only predicted physical fields supply the dynamic compilation Value."""
    def __init__(self, contract, identity):
        super().__init__()
        from .representation import VisualRelations, RelationEncoder, RelationalRead
        if len(contract.targets) != 38 or contract.rank != 128 or contract.alpha != 128:
            raise ValueError('relation compiler requires complete rank128 scale1')
        validate_lora_state(identity, contract)
        self.mode = 'relation_grounded'
        self.names = tuple(target.name for target in contract.targets)
        self.common = DirectLoRAParameters({k: v.detach().cpu() for k, v in identity.items()})
        self.gamma = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.phi = VisualRelations()
            self.omega = RelationEncoder()
            self.read = RelationalRead()
            self.conditional_targets = nn.ModuleList(
                ConditionalTarget(target.in_features, target.out_features) for target in contract.targets)
        self.register_buffer('probe', torch.randn(50, 32,
            generator=torch.Generator(device='cpu').manual_seed(1729)))

    def public_state(self):
        return self.common()

    def forward(self, native_inputs, h, frame_indices, *, capture_mechanism=False, **unused):
        if set(native_inputs) != set(self.names) or unused:
            raise ValueError('relation Writer condition cannot contain extra labels or paths')
        prediction = self.phi(h, frame_indices)
        w = self.omega(prediction, frame_indices)
        c, d = self.read(h, w, prediction)
        common, result, edits = self.public_state(), {}, {}
        for name, unit in zip(self.names, self.conditional_targets, strict=True):
            a0, b0 = common[name + LORA_A_SUFFIX], common[name + LORA_B_SUFFIX]
            inputs = (a0, b0, native_inputs[name], h, c, d)
            a, b, s, m = (checkpoint(unit, *inputs, use_reentrant=False, preserve_rng_state=False)
                          if torch.is_grad_enabled() else unit(*inputs))
            result[name + LORA_A_SUFFIX], result[name + LORA_B_SUFFIX] = a, b
            if capture_mechanism:
                edits[name] = {'S': s, 'M': m}
        self.last_prediction = prediction
        if capture_mechanism:
            self.last_mechanism = {'prediction': prediction, 'c': c, 'd': d, 'targets': edits}
        return result


def build_runtime(asset_root, spec, device):
    runtime = build_operator_runtime(Path(asset_root), spec, device, 'T')
    runtime.writer = RelationWriter(runtime.lora, runtime.identity).to(device)
    return runtime


def semantic_vectors(policy, names, tokenizer_path=None):
    """Frozen raw source word embeddings, token mean, fixed orthogonal64 labels."""
    import sentencepiece
    tokenizer_path = tokenizer_path or ASSET_ROOT / 'models/tokenizers/openpi/paligemma_tokenizer.model'
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    embedding = policy.model.paligemma_with_expert.paligemma.get_input_embeddings().weight
    means = {}
    with torch.no_grad():
        for name in names:
            ids = tokenizer.encode(str(name), add_bos=False, add_eos=False)
            if not ids:
                raise ValueError(f'empty semantic tokenization: {name}')
            mean = embedding[torch.tensor(ids, device=embedding.device)].float().mean(0).cpu()
            means[str(name)] = mean.numpy()
    from .labels import project_semantics
    return project_semantics(means)


def to_device(fields, device):
    return {name: torch.as_tensor(value, device=device) for name, value in fields.items()
            if isinstance(value, (np.ndarray, torch.Tensor))}


@contextmanager
def source_without_adapters(policy):
    """Bypass every PEFT adapter for F's true source call; preserve frozen flags."""
    modules = [module for module in policy.modules() if callable(getattr(module, 'enable_adapters', None))]
    previous = [bool(module.disable_adapters) for module in modules]
    try:
        for module in modules:
            module.enable_adapters(False)
        yield
    finally:
        for module, disabled in zip(modules, previous, strict=True):
            module.enable_adapters(not disabled)
        policy.requires_grad_(False)


def source_query(policy, owner, sample, prepared):
    """Capture H0 and v0 from one unadapted actual noisy-action denoise call."""
    captured = []
    hook = policy.model.action_out_proj.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0]))
    try:
        with torch.no_grad(), source_without_adapters(policy):
            velocity = owner(sample, prepared)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape != (len(velocity), 50, 1024):
        raise ValueError('source query lost same-forward full H0')
    return captured[0].detach(), velocity.detach()
