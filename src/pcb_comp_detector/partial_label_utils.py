import torch
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from typing import Dict, List
import numpy as np

from .graph_prop_roi_heads import GraphPropROIHeads

@META_ARCH_REGISTRY.register()
class PartialLabelRCNN(GeneralizedRCNN):

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        targets = None
        if "instances" in batched_inputs[0]:
            targets = [i["instances"] for i in batched_inputs]

        results, _ = self.roi_heads(images, features, proposals, targets)

        if not isinstance(self.roi_heads, GraphPropROIHeads):
            # GraphPropROIHeads already adds partial labels in, so add labels back in to results if not using that
            for i in range(len(results)):
                ious = pairwise_iou(targets[i].gt_boxes, results[i].pred_boxes)
                if len(ious) > 0:
                    results[i] = results[i][ious.max(dim=0)[0] < 0.5]

                    results[i] = Instances.cat([
                        Instances(
                            pred_boxes=targets[i].gt_boxes,
                            pred_classes=targets[i].gt_classes,
                            image_size=targets[i].image_size,
                            scores=torch.full((len(targets[i]),), 1, device=self.device)
                        ),
                        results[i]
                    ])

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

# hack to be able to pass is_train=False and still get annotations
class PartialLabelDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        self.is_train = True
        d = super().__call__(dataset_dict)
        self.is_train = False
        return d

class PartialLabelPredictor(DefaultPredictor):
    def __init__(self, *args, partial_label_proportion=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_label_proportion = partial_label_proportion
        assert self.input_format == "RGB"

    def default_call(self, input):
        return super().__call__(input)

    def __call__(self, input):
        # Input should be a dict with "image" as an RGB image and "instances" present if using partial labels

        islist = isinstance(input, list)
        if not islist:
            input = [input]

        input = [i.copy() for i in input]

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            tfs = []
            for i in input:
                i["image"] = np.array(i["image"])
                tf = self.aug.get_transform(i["image"])
                tfs.append(tf)
                i["image"] = torch.as_tensor(tf.apply_image(i["image"]).astype("float32").transpose(2, 0, 1))
                if "instances" in i:
                    i["instances"].gt_boxes = Boxes(tf.apply_box(i["instances"].gt_boxes.tensor))
                    i["instances"] = i["instances"][torch.randperm(len(i["instances"]))]
                    i["instances"] = i["instances"][:int(len(i["instances"])*self.partial_label_proportion)].to(self.model.device)

            out = self.model.inference(input)

            # Apply inverse transform to predictions so that they match up with original image
            for o, t in zip(out, tfs):
                o["instances"] = o["instances"].to("cpu")
                o["instances"].pred_boxes = Boxes(t.inverse().apply_box(o["instances"].pred_boxes.tensor))

            return out if islist else out[0]