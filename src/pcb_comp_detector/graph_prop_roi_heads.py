# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import warnings

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference

from .ocr import OCRPipeline
from .graph_prop_output_layers import BoxRegOutput, BoxClassOutput


@ROI_HEADS_REGISTRY.register()
class GraphPropROIHeads(ROIHeads):

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_reg_predictor: nn.Module,
        box_class_predictor: nn.Module,
        box_bg_predictor: nn.Module,
        pre_prop_layers: nn.Module,
        post_prop_layers: nn.Module,
        ocr_pipeline: Optional[OCRPipeline],
        fc_dim: int,
        use_own_alt_input: bool = False,
        partial_label_proposal_iou_threshold: float = 0.25, # Determines when to remove proposals that overlap with partial labels
        partial_label_train_freq: float = 0.5,
        partial_label_prop_max: float = 0.2, # randomly chosen between 0 and this number
        separate_graph_prop_portion_params: bool = True,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
        """
        super().__init__(**kwargs)

        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.fc_dim = fc_dim
        self.partial_label_proposal_iou_threshold = partial_label_proposal_iou_threshold
        self.partial_label_train_freq = partial_label_train_freq
        self.partial_label_prop_max = partial_label_prop_max
        self.separate_graph_prop_portion_params = separate_graph_prop_portion_params
        self.use_own_alt_input = use_own_alt_input

        self.box_head = box_head
        self.ocr_pipeline = ocr_pipeline
        self.box_reg_predictor = box_reg_predictor
        self.box_class_predictor = box_class_predictor
        self.box_bg_predictor = box_bg_predictor
        self.pre_prop_layers = pre_prop_layers
        self.post_prop_layers = post_prop_layers

        self.bn = nn.BatchNorm1d(self.fc_dim)

        if use_own_alt_input:
            self.separate_graph_prop_portion_params = True

        if self.separate_graph_prop_portion_params:
            self.vis_graph_prop_portion = nn.Parameter(torch.tensor(0.0)) # gets sigmoided and then used to determine how much features are propagated through the similarity graph
            if self.use_own_alt_input:
                self.alt_graph_prop_portion = nn.Parameter(torch.tensor(0.0))
        else:
            self.graph_prop_portion = nn.Parameter(torch.tensor(0.0))

        self.graph_prop_temperature = nn.Parameter(torch.tensor(0.0))

        # Set up normalization constants based on dimensionality
        self.vis_emb_sq_l2_dist_mean = self.box_head.output_shape.channels # chi square distribution mean
        self.vis_emb_sq_l2_dist_std = np.sqrt(2) * self.box_head.output_shape.channels # chi square distribution std

        self.device = None

        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)

        ocr = OCRPipeline(**OCRPipeline.from_cfg(cfg)) if cfg.MODEL.OCR.ENABLED else None
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))

        ret["box_class_predictor"] = BoxClassOutput(cfg, ShapeSpec(channels=cfg.MODEL.ROI_HEADS.FC_DIM))
        ret["pre_prop_layers"] = nn.Linear(ret["box_head"].output_shape.channels, cfg.MODEL.ROI_HEADS.FC_DIM)

        ret["post_prop_layers"] = []
        for i in range(cfg.MODEL.ROI_HEADS.NUM_FC_AFTER_PROP):
            ret["post_prop_layers"].append(nn.Linear(cfg.MODEL.ROI_HEADS.FC_DIM + cfg.MODEL.ROI_HEADS.NUM_CLASSES if i == 0 else cfg.MODEL.ROI_HEADS.FC_DIM, cfg.MODEL.ROI_HEADS.FC_DIM))
            ret["post_prop_layers"].append(nn.ReLU())
        ret["post_prop_layers"] = nn.Sequential(*ret["post_prop_layers"])

        ret["ocr_pipeline"] = ocr
        ret["partial_label_proposal_iou_threshold"] = cfg.MODEL.ROI_HEADS.PARTIAL_LABEL_PROPOSAL_IOU_THRESHOLD
        ret["partial_label_train_freq"] = cfg.MODEL.ROI_HEADS.PARTIAL_LABEL_TRAIN_FREQ
        ret["partial_label_prop_max"] = cfg.MODEL.ROI_HEADS.PARTIAL_LABEL_PROP_MAX
        ret["fc_dim"] = cfg.MODEL.ROI_HEADS.FC_DIM

        ret["test_score_thresh"]         = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        ret["test_nms_thresh"]           = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        ret["test_topk_per_image"]       = cfg.TEST.DETECTIONS_PER_IMAGE

        ret["use_own_alt_input"]         = cfg.MODEL.ROI_HEADS.USE_OWN_ALT_INPUT
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_reg_predictor": BoxRegOutput(cfg, box_head.output_shape),
            "box_bg_predictor": nn.Linear(box_head.output_shape.channels, 1)
        }

    def forward(
        self,
        images: ImageList,
        vis_features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        if not self.ocr_pipeline is None:
            self.ocr_pipeline.do_ocr(images.tensor)

        self.device = proposals[0].proposal_boxes.tensor.device

        partial_label_encs = []
        partial_labels = []

        if not targets is None:
            # Randomly add partially labeled data as input
            if self.training:
                targets = [i[torch.randperm(len(i))] for i in targets] # randomly permute gt instances

            if not self.training:
                # Use all provided labels
                partial_label_proportions = [1]*len(targets)
            else:
                # Sample some of the provided labels
                partial_label_proportions = torch.rand(len(proposals)) * self.partial_label_prop_max
                partial_label_proportions[torch.rand(len(proposals)) > self.partial_label_train_freq] = 0.0
            partial_label_counts = [int(p*len(a)) for p, a in zip(partial_label_proportions, targets)]
            gt_proposal_counts = [max(1, max(plc, int(self.positive_fraction*len(t)))) for plc, t in zip(partial_label_counts, targets)]

            for i in range(len(targets)):
                if len(targets[i].gt_boxes) == 0:
                    if self.training:
                        return None, {}
                    else:
                        partial_label_encs.append(torch.zeros((len(proposals[i]), self.num_classes), device=self.device))
                        partial_labels.append([])
                        continue
                
                gtpc = gt_proposal_counts[i]
                ious = pairwise_iou(targets[i].gt_boxes, proposals[i].proposal_boxes)

                if self.training:
                    # Assign each proposal to a gt box
                    matched_idx, matched_labels = self.proposal_matcher(ious)
                    proposals[i].gt_boxes = targets[i].gt_boxes[matched_idx]
                    proposals[i].gt_classes = targets[i].gt_classes[matched_idx]
                    proposals[i].gt_classes[matched_labels == 0] = self.num_classes
                    proposals[i].gt_classes[matched_labels == -1] = -1

                # Remove proposals that overlap with partial labels
                proposals[i] = proposals[i][ious[:gtpc,:].max(dim=0)[0] < self.partial_label_proposal_iou_threshold]
                # Add appropriate amount of ground truth boxes to proposals
                gtp = targets[i][:gtpc]
                proposals[i] = Instances.cat([
                    Instances(
                        proposal_boxes=gtp.gt_boxes,
                        image_size=gtp.image_size,
                        objectness_logits=torch.full((len(gtp),), 1, device=self.device),
                        **({"gt_boxes": gtp.gt_boxes,
                        "gt_classes": gtp.gt_classes} if self.training else {})
                    ),
                    proposals[i]
                ])
                partial_labels.append(gtp)

                # Encode partial labels
                plc = partial_label_counts[i]
                pl = gtp[:plc]
                ple = F.one_hot(pl.gt_classes, self.num_classes)
                partial_label_encs.append(torch.cat((ple, torch.zeros((len(proposals[i])-plc, self.num_classes), device=self.device)), dim=0))
                del ple
        else:
            partial_label_encs = [torch.zeros((len(p), self.num_classes-1), device=self.device) for p in proposals]

        if not self.ocr_pipeline is None:
            alt_features = [torch.cat((o, p), dim=1) for p, o in zip(partial_label_encs, self.ocr_pipeline.encode(proposals))]
        else:
            alt_features = partial_label_encs
        del partial_label_encs

        if self.training:
            losses = self._forward_box(vis_features, alt_features, proposals, partial_labels)
            return proposals, losses
        else:
            pred_instances = self._forward_box(vis_features, alt_features, proposals, partial_labels)
            return pred_instances, {}

    def _forward_box(self, vis_features: Dict[str, torch.Tensor], alt_features: Optional[List[torch.Tensor]], proposals: List[Instances], partial_labels: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        npl = [len(p) for p in partial_labels]
        vis_features = [vis_features[f] for f in self.box_in_features]
        box_vis_features = self.box_pooler(vis_features, [x.proposal_boxes for x in proposals])
        box_vis_features = self.box_head(box_vis_features)

        bg_logit = self.box_bg_predictor(box_vis_features)
        bg_logit_per_img = bg_logit.detach().split([len(p) for p in proposals])
        box_deltas = self.box_reg_predictor(box_vis_features)


        # Select proposals for graph prop based on bg logits
        prop_fg_box_mask_per_img = []

        for i, bgl in enumerate(bg_logit_per_img):
            if len(proposals[i].proposal_boxes) < 3:
                prop_fg_box_mask_per_img.append(torch.tensor([]))
                continue

            if self.training and len(proposals[i].gt_boxes) < 3:
                prop_fg_box_mask_per_img.append(torch.tensor([]))
                continue

            mask = torch.full((len(bgl),), False, device=self.device)

            if self.batch_size_per_image > npl[i]:
                selected_inds = bgl[npl[i]:].squeeze().argsort()[:self.batch_size_per_image - npl[i]] + npl[i]
                mask[selected_inds] = True
            mask[:npl[i]] = True
            prop_fg_box_mask_per_img.append(mask)

        prop_fg_box_mask = torch.cat(prop_fg_box_mask_per_img, dim=0)

        box_vis_features = box_vis_features[prop_fg_box_mask]
        alt_features = [a[m] for a, m in zip(alt_features, prop_fg_box_mask_per_img)]

        box_vis_features = self.pre_prop_layers(box_vis_features)
        box_vis_features_per_img = box_vis_features.split([p.sum().cpu().item() for p in prop_fg_box_mask_per_img])

        features = []

        for i, box_vis_features_img in enumerate(box_vis_features_per_img):
            if len(box_vis_features_img) == 0:
                features.append(torch.tensor([]))

            # Coefficients based on background logits for selected proposals
            selected_prop_bg_coeff = torch.sigmoid(bg_logit_per_img[i][prop_fg_box_mask_per_img[i]])

            box_vis_features_img = self.bn(box_vis_features_img)

            # Construct graph
            box_vis_features_img = box_vis_features_img.unsqueeze(dim=0)
            l2_distances_sq = (box_vis_features_img - box_vis_features_img.swapaxes(0, 1)).square().sum(dim=2)
            sim_graph_edges = 2 / (1 + (l2_distances_sq / (self.vis_emb_sq_l2_dist_mean * torch.sigmoid(self.graph_prop_temperature)))) - 1
            sim_graph_edges = sim_graph_edges.clamp(min=0, max=1) # only >0, or only l2 distances that are less than the mean l2 distance multiplied by the learned temerature parameter (which is between 0 and 1)
            sim_graph_edges = sim_graph_edges.clone()
            sim_graph_edges.fill_diagonal_(0) # zero out diagonal so self is not included in similarity matrix
            sim_graph_edges = sim_graph_edges / len(sim_graph_edges) # normalize by total number of nodes

            # Propagate visual features
            prop_portion = torch.sigmoid(self.vis_graph_prop_portion if self.separate_graph_prop_portion_params else self.graph_prop_portion)
            box_vis_features_img = prop_portion * (box_vis_features_img * sim_graph_edges.unsqueeze(dim=2) * selected_prop_bg_coeff.unsqueeze(dim=1)).sum(dim=1) + (1-prop_portion) * box_vis_features_img

            # Propagate alt features
            alt_features_img = alt_features[i].unsqueeze(dim=0)
            prop_portion = torch.sigmoid(self.alt_graph_prop_portion if self.separate_graph_prop_portion_params else self.graph_prop_portion) if self.use_own_alt_input else 1
            alt_features_img = prop_portion * (alt_features_img * sim_graph_edges.unsqueeze(dim=2) * selected_prop_bg_coeff.unsqueeze(dim=1)).sum(dim=1) + ((1-prop_portion) * alt_features_img if self.use_own_alt_input else 0)

            features.append(torch.cat((box_vis_features_img.squeeze(), alt_features_img.squeeze()), dim=1))

        del box_vis_features_img
        del box_vis_features_per_img
        del alt_features_img
        del sim_graph_edges
        del selected_prop_bg_coeff
        del l2_distances_sq

        features = torch.cat(features, dim=0)

        if len(features) == 0:
            return {} if self.training else Instances()

        features = self.post_prop_layers(features)

        pred_scores = self.box_class_predictor(features)
        scores = torch.zeros((len(prop_fg_box_mask), self.num_classes), device=self.device).type(pred_scores.dtype)
        scores[prop_fg_box_mask] = pred_scores
        scores = torch.cat((scores, bg_logit), dim=1)

        if self.training:
            losses = self.box_reg_predictor.losses(box_deltas, proposals)
            losses.update(self.box_class_predictor.losses(scores, proposals))
            return losses
        else:
            boxes = self.box_reg_predictor.predict_boxes(box_deltas, proposals)
            scores = self.box_class_predictor.predict_probs(scores, proposals)
            for i in range(len(proposals)):
                if npl[i] > 0:
                    scores[i][:npl[i]] = F.one_hot(partial_labels[i].gt_classes[:npl[i]], num_classes=scores[i].shape[1])
                    boxes[i][:npl[i]] = partial_labels[i].gt_boxes[:npl[i]].tensor
            image_shapes = [x.image_size for x in proposals]
            return fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
            )[0]
