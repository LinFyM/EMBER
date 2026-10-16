"""Frozen source-control assets and unchanged task-local proposal units."""
from ember.reference.contract import ROOT as REFERENCE_ROOT
BASIS_SCHEMA='ember_source_control_linear64_r8_seed7_v1'
BASIS=REFERENCE_ROOT.parent/'control_linear64_seed7/basis/state.pt'
INITIAL_READER=REFERENCE_ROOT.parent/'video_initial_seed7/checkpoints/update_00000900/state.pt'
PRACTICE_STEPS=8192
SIGMA=.25
