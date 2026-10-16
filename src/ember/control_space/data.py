"""The shared complete source/r8 topology; no active training/query stream."""
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.reference.contract import ASSETS

def topology():
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)
