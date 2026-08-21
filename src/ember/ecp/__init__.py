"""EMBER-ECP: event-conditioned compilation for complete PI0.5 task LoRAs."""

from ember.ecp.contracts import (
    ACTION_HORIZON,
    ACTION_LAYERS,
    PADDED_ACTION_DIM,
    TargetFamily,
    TargetOwner,
    build_target_owners,
)
from ember.ecp.events import (
    EventBindingOutput,
    EventConditionedHorizonBinding,
    EventProgramOutput,
    OrderedEventSegmenter,
    TaskGroundedTransitionMatcher,
)
from ember.ecp.observer import ECPNativeObserver, NativeObserverOutput
from ember.ecp.stage0 import ECPStage0Model, ECPStage0Output, ECPVideoEncoder
from ember.ecp.stage0_objective import ECPStage0Loss, ecp_stage0_loss
from ember.ecp.stage0_data import (
    ECPStage0Pair,
    ECPStage0Schedule,
    ECPStage0Task,
    PackedStage0Pair,
    load_stage0_tasks,
    pack_stage0_pair,
)

__all__ = [
    "ACTION_HORIZON",
    "ACTION_LAYERS",
    "PADDED_ACTION_DIM",
    "ECPNativeObserver",
    "ECPStage0Loss",
    "ECPStage0Model",
    "ECPStage0Output",
    "ECPStage0Pair",
    "ECPStage0Schedule",
    "ECPStage0Task",
    "ECPVideoEncoder",
    "EventBindingOutput",
    "EventConditionedHorizonBinding",
    "EventProgramOutput",
    "NativeObserverOutput",
    "OrderedEventSegmenter",
    "PackedStage0Pair",
    "TargetFamily",
    "TargetOwner",
    "TaskGroundedTransitionMatcher",
    "build_target_owners",
    "ecp_stage0_loss",
    "load_stage0_tasks",
    "pack_stage0_pair",
]
