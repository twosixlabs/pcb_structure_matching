from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from typing import Callable
from detectron2.layers import ShapeSpec
from detectron2.structures.boxes import pairwise_point_box_distance
from detectron2.config import configurable
from detectron2.utils.registry import Registry

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"


STRING_ENCODER_REGISTRY = Registry("STRING_ENCODER")
STRING_ENCODER_REGISTRY.__doc__ = "Registry for string encoding methods"

class StringEncoder:
    @configurable
    def __init__(
        self,
        *,
        ambiguate_digits: bool = True
    ):
        self.vec_len = len(ALPHABET) + (1 if ambiguate_digits else 10)

        self.char_id_lookup = {}
        for i, a in enumerate(ALPHABET):
            self.char_id_lookup[a] = i
        for i, d in enumerate(DIGITS):
            self.char_id_lookup[d] = len(ALPHABET) + (0 if ambiguate_digits else i)

    @classmethod
    def from_config(cls, cfg):
        return {
            "ambiguate_digits": cfg.OCR.ENCODING.AMBIGUATE_DIGITS
        }

    def forward(self, string):
        raise UnimplementedException()

@STRING_ENCODER_REGISTRY.register()
class FirstChar(StringEncoder):
    # First character
    def forward(self, string):
        return F.one_hot(self.char_id_lookup[string[0].lower()])

@STRING_ENCODER_REGISTRY.register()
class BagOfChar(StringEncoder):
    # Bag of characters
    @configurable
    def __init__(self, *args, decay_factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_factor = decay_factor

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["decay_factor"] = cfg.MODEL.OCR.ENCODING.DECAY_FACTOR
        return ret

    def forward(self, string):
        enc = torch.zeros(len(self.char_id_lookup))
        coeff = 1
        for c in string.lower():
            one_hot += F.one_hot(torch.tensor([self.char_id_lookup[c] for c in string]), num_classes=len(self.char_id_lookup)) * coeff
            coeff *= self.decay_factor

        return enc


class OCRModel(nn.Module):
    def forward(self, x):
        return Instances()

class OCRPipeline:
    # First call do_ocr and then encode detections and

    @configurable
    def __init__(
        self,
        *,
        model: OCRModel,
        encoder: StringEncoder,
        num_nearby_text_enc: int = 5,
    ):
        self.model = model
        self.encoder = encoder
        self._ocr_detections = None
        self.num_nearby_text_enc = num_nearby_text_enc
        self.output_shape = ShapeSpec(channels=self.encoder.vec_len)

    @classmethod
    def from_config(cls, cfg):
        return {
            "model": lambda x: OCRModel(), # TO-DO
            "encoder": STRING_ENCODER_REGISTRY.get(cfg.MODEL.OCR.ENCODING.TYPE)(cfg),
            "num_nearby_text_enc": cfg.MODEL.OCR.NUM_NEARBY_TEXT_ENC
        }

    def do_ocr(self, img):
        # _ocr_detections should be an Instances with standard detection format minus class plus a list of detected "strings"
        self._ocr_detections = self.model(img)

    def encode(self, proposal_boxes):
        encoded_ocr_dets = self.encoder(self._detections)

        ocr_dets_centers = self._ocr_detections.boxes.get_centers()
        dists = pairwise_point_box_distance(ocr_dets_centers, proposal_boxes).min(dim=2).t()
        closest_ocr_det_inds_to_proposals = dists.argsort(dim=1)[:, :self.num_nearby_text_enc]
        closest_ocr_det_encs = encoded_ocr_dets[closest_ocr_dets_to_proposals]

        # Scale encoded detections by how close they are to the proposal box. Far away detections are scaled very low
        closest_ocr_dets_distances = dists[torch.arange(len(proposal_boxes)), closest_ocr_dets_to_proposals]
        closest_ocr_dets_coeffs = 1 / (1 + closest_ocr_dets_distances/(closest_ocr_dets_distances[:,0].median() + 1e-6))

        return (closest_ocr_det_encs * closest_ocr_dets_coeffs).mean(dim=1)
