"""Loading skips overwritten initialization without weakening native state checks."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.pi05_source_checkpoint import Pi05SourceTrainingError
from ember.pi05_source_setup import load_pretrained_policy


class TinyPolicy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict({
            "expert": torch.nn.Linear(3, 4, bias=False).to(torch.bfloat16),
            "head": torch.nn.Linear(4, 2),
        })
        self.register_buffer("position_ids", torch.arange(3), persistent=False)
        self.remaps = 0

    def _fix_pytorch_state_dict_keys(self, state, config):
        self.remaps += 1
        return state

    def forward(self, value):
        return self.model["head"](self.model["expert"](value.to(torch.bfloat16)).float())


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    from lerobot.policies import pi05

    config = SimpleNamespace(device="cpu")
    reference = TinyPolicy(config).eval()
    save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(pi05, "PI05Policy", TinyPolicy)
    return tmp_path, config, reference


def test_full_checkpoint_skips_random_fills_and_preserves_native_execution(checkpoint, monkeypatch):
    path, config, reference = checkpoint
    calls = []
    initialize = torch.nn.init.kaiming_uniform_
    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", lambda *a, **kw: calls.append(True) or initialize(*a, **kw))
    loaded = load_pretrained_policy(path, config)
    assert calls == []
    assert loaded.model["expert"].weight.dtype == torch.bfloat16
    assert loaded.model["head"].weight.dtype == torch.float32
    assert loaded.position_ids.tolist() == [0, 1, 2]
    # Episode noise has its own explicit seed; skipped constructor draws cannot
    # change this stream or the native output computed from fully loaded weights.
    value = torch.randn(2, 3, generator=torch.Generator().manual_seed(119))
    torch.testing.assert_close(loaded(value), reference(value))
    assert loaded.remaps == 0
    assert torch.nn.init.kaiming_uniform_ is not initialize  # caller's patch restored
    torch.nn.Linear(1, 1)
    assert calls == [True]


@pytest.mark.parametrize("change", ["missing", "unexpected", "shape"])
def test_incomplete_or_incompatible_checkpoint_never_returns_policy(checkpoint, change):
    path, config, reference = checkpoint
    state = reference.state_dict()
    if change == "missing":
        del state["model.head.bias"]
    elif change == "unexpected":
        state["unexpected"] = torch.zeros(1)
    else:
        state["model.head.bias"] = torch.zeros(3)
    save_file(state, str(path / "model.safetensors"))
    with pytest.raises(Pi05SourceTrainingError, match="strict weight load failed"):
        load_pretrained_policy(path, config)


def test_missing_weight_file_fails_before_constructor(checkpoint, monkeypatch):
    from lerobot.policies import pi05

    path, config, _ = checkpoint
    (path / "model.safetensors").unlink()
    monkeypatch.setattr(pi05, "PI05Policy", lambda _: pytest.fail("constructor must not run"))
    with pytest.raises(Pi05SourceTrainingError, match="missing PI05 weights"):
        load_pretrained_policy(path, config)


def test_existing_callers_keep_their_key_remapping_contract(checkpoint):
    path, config, reference = checkpoint
    assert load_pretrained_policy(path, config, remap_native_keys=True).remaps == 1
    save_file({name.removeprefix("model."): value for name, value in reference.state_dict().items()},
              str(path / "model.safetensors"))
    loaded = load_pretrained_policy(path, config)
    assert loaded.remaps == 1
    torch.testing.assert_close(loaded(torch.ones(1, 3)), reference(torch.ones(1, 3)))


def test_checkpoint_payload_stays_on_cpu(checkpoint, monkeypatch):
    import safetensors.torch

    path, config, _ = checkpoint
    calls = []
    load_file = safetensors.torch.load_file
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path, **kw:
                        calls.append(kw) or load_file(path, **kw))
    load_pretrained_policy(path, config)
    assert calls == [{"device": "cpu"}]


def test_actual_native_rotary_and_vision_buffers_survive_no_init():
    from lerobot.policies.pi_gemma import PiGemmaModel
    from transformers import GemmaConfig, SiglipVisionConfig
    from transformers.initialization import no_init_weights
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    language = GemmaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        use_adarms=False, attention_dropout=0.0)
    vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=16, patch_size=8, attention_dropout=0.0)

    def native_modules():
        return torch.nn.ModuleDict({"language": PiGemmaModel(language),
                                    "vision": SiglipVisionModel(vision)}).eval()

    reference = native_modules()
    with no_init_weights():
        loaded = native_modules()
    loaded.load_state_dict(reference.state_dict(), strict=True)
    # These nonpersistent buffers are not supplied by state_dict. Their native
    # constructors, rather than a replacement/meta loader, must initialize them.
    assert "language.rotary_emb.inv_freq" not in loaded.state_dict()
    assert torch.isfinite(loaded["language"].rotary_emb.inv_freq).all()
    assert loaded["language"].rotary_emb.inv_freq.count_nonzero() == 4
    assert loaded["vision"].vision_model.embeddings.position_ids.tolist() == [[0, 1, 2, 3]]
    ids = torch.tensor([[1, 2, 3]])
    pixels = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(loaded["language"](input_ids=ids).last_hidden_state,
                                   reference["language"](input_ids=ids).last_hidden_state)
        torch.testing.assert_close(loaded["vision"](pixel_values=pixels).last_hidden_state,
                                   reference["vision"](pixel_values=pixels).last_hidden_state)


def test_evaluator_uses_shared_strict_loader_without_changing_policy_contract(monkeypatch):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from ember.pi05_eval import worker_setup

    config = PI05Config(device="cpu")
    calls = []
    policy = SimpleNamespace(model=SimpleNamespace(gradient_checkpointing_disable=lambda: None))
    policy.to = lambda device: calls.append(("to", device)) or policy
    policy.eval = lambda: policy
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda _: config)
    monkeypatch.setattr(worker_setup, "load_pretrained_policy", lambda path, configured, **kwargs:
                        calls.append((path, configured, kwargs)) or policy)
    processor = SimpleNamespace(unnormalize_action=object())
    monkeypatch.setattr(worker_setup, "Pi05LiberoProcessor", lambda *args: processor)
    contract = {"precision": "bfloat16", "chunk_size": 50, "n_action_steps": 10,
                "num_inference_steps": 10, "state_dim": 8, "action_dim": 7}
    observed = worker_setup.load_policy(Path("/source"), {}, Path("/tokenizer"), contract)
    assert observed == (policy, processor, processor.unnormalize_action)
    assert calls[0] == (Path("/source"), config, {"remap_native_keys": True})
    assert calls[1] == ("to", "cuda:0")
    assert config.device == "cuda:0" and config.dtype == "bfloat16"
    assert (config.chunk_size, config.n_action_steps, config.num_inference_steps) == (50, 10, 10)
    assert config.input_features["observation.state"].shape == (8,)
    assert config.output_features["action"].shape == (7,)
