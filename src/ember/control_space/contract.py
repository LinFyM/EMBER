"""One bounded fresh video-to-fixed-control family, with one shared ledger."""
from pathlib import Path
from ember.reference.contract import TASKS, ROOT as REFERENCE_ROOT, OPTIMIZER
SCHEMA='ember_source_video_initial_r8_seed7_v1'
BASIS_SCHEMA='ember_source_control_linear64_r8_seed7_v1'
PARENT_ROOT=REFERENCE_ROOT.parent/'control_space_seed7'
OPERATOR_ROOT=REFERENCE_ROOT.parent/'control_operator_from144_seed7'
LINEAR_ROOT=REFERENCE_ROOT.parent/'control_linear64_seed7'
BASIS=LINEAR_ROOT/'basis/state.pt'
ROOT=REFERENCE_ROOT.parent/'video_initial_seed7'
LEGACY_IMAGE_ROOT=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261014/image_only_gemma_with_history')
LEGACY_IMAGE_CACHE=LEGACY_IMAGE_ROOT.parent/'image_only_gemma_compiler/image_features'
VALIDATION=(3,6,11,16,23,26,31,39)
CHECKPOINTS=(900,1800)
LIMITS=dict(optimizer_updates=1800,FM_queries=201600,original_video_reads=7200,auxiliary_video_reads=7200,
    physical_submissions=1968,control_steps=679360,dummy_steps=19680,total_steps=699040,
    engineering_extra_attempts=16,added_peak_GiB=192)

def evaluation_root(root,update):return Path(root)/'evaluation'/f'update_{update:08d}'
def queue_path(root):return Path(root)/'readout.sqlite'
