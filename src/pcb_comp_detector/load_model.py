from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor
from .partial_label_utils import PartialLabelPredictor
from functools import partial
from pathlib import Path
import warnings, json


def replace_cfg(cfg, target, replacement):
    for k in cfg.keys():
        if cfg[k] == target:
            cfg[k] = replacement
        elif isinstance(cfg[k], CfgNode):
            cfg[k] = replace_cfg(cfg[k], target, replacement)

    return cfg

def load_model(
        model_path: str,
        *,
        device: str ="cpu",
        roi_heads_score_thresh: float = None,
        pre_nms_topk_prop: float = None,
        post_nms_topk_prop: float = None,
        partial_label_proportion: float = 0
    ):
    # Lowering roi_heads_score_thresh increases inference time but generally improves results. Defaults to 0.05. See https://detectron2.readthedocs.io/en/latest/modules/config.html
    # pre_nms_topk_prop is per FPN level. Default for both topk's is 2000. Raising both will improve results but require more computation.

    model_path = Path(model_path)

    cfg_path = model_path.parent / "cfg.yaml"
    class_path = model_path.parent / "classes.json"

    classes = None

    if not model_path.suffix == ".pth":
        raise Exception("Invalid path, must be to .pth model save")
    if not cfg_path.exists():
        raise Exception(f"No cfg.yaml found at {cfg_path}.")
    if not class_path.exists():
        warnings.warn(f"No classes.json found at {class_path}, returning None for it.")
    else:
        with open(class_path) as f:
            classes = json.loads(f.read())

    cfg = get_cfg()

    cfg.merge_from_file(cfg_path)

    if partial_label_proportion > 0:
        cfg.MODEL.META_ARCHITECTURE = "PartialLabelRCNN"
    
    cfg = replace_cfg(cfg, "SyncBN", "BN")
    cfg.MODEL.DEVICE = device

    if not pre_nms_topk_prop is None:
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = pre_nms_topk_prop
    if not post_nms_topk_prop is None:
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = post_nms_topk_prop

    if not roi_heads_score_thresh is None:
        if cfg.MODEL.META_ARCHITECTURE == "ProposalNetwork":
            raise Exception("Provided roi_heads_score_thresh for a proposal network, only can be used in full object detectors")
        else:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_heads_score_thresh

    cfg.MODEL.WEIGHTS = str(model_path)

    return PartialLabelPredictor(cfg, partial_label_proportion=partial_label_proportion), cfg, classes
