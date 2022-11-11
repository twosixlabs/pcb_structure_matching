import torch

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from typing import Dict, List, Optional, Tuple
from detectron2.structures import ImageList, Instances

@META_ARCH_REGISTRY.register()
class GraphPropRCNN(GeneralizedRCNN):

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
        iwl = False
        if "instances" in batched_inputs[0]:
            targets = [i["instances"] for i in batched_inputs]
            iwl = True

        results, _ = self.roi_heads(images, features, proposals, targets)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results
