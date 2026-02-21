from .partial_annotation import (
    PartialTverskyLoss,
    BalancedSoftmaxTverskyLoss,
    PartialAnnotationDeepSupervisionLoss,
    build_partial_annotation_loss,
    FG_THRESHOLD,
)
from .loss_zoo import build_loss, LOSS_REGISTRY
