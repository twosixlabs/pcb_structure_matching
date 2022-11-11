
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes
from argparse import ArgumentParser
from pathlib import Path
import json, torch, cv2
import numpy as np
from graph_prop_detector.src.pcb_comp_detector.utils import instances_to_dict

from pcb_comp_detector.data import register_dataset
from pcb_comp_detector import load_model


def run(opt, load_model_args):
    slashloc = opt.model_path.rfind("/")

    if slashloc == -1:
        raise Exception("Model save must be in a folder with cfg.yaml")

    dir = Path(opt.model_path[:slashloc])

    dataset_path = Path(opt.dataset_path)

    results = {}
    with torch.no_grad():
        predictor, cfg = load_model(opt.model_path, **{k: v for k, v in vars(opt).items() if k in load_model_args})

        # Register dataset so metadata for classes can be used
        _, _, classes, _, _ = register_dataset(cfg.DATASETS.TRAIN[0][:-6])
        print("Model uses the following classes:")
        print(classes)

        for i, file in enumerate(dataset_path.iterdir()):
            im = cv2.imread(str(file))
            if im is None:
                print("Skipping " + str(file))
                continue
            # cv2.imwrite(str(dir / f"pred_{i}_{j}_orig.jpg"), im)
            outputs = predictor(im)

            if i < opt.visualize:
                v = Visualizer(im,
                               metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                               scale=1
                )

                out = None
                if cfg.MODEL.META_ARCHITECTURE == "ProposalNetwork":
                    threshold = np.partition(outputs["proposals"].objectness_logits.cpu().numpy(), -opt.num_prop_vis)[-opt.num_prop_vis]
                    boxes = outputs["proposals"][outputs["proposals"].objectness_logits > threshold].get("proposal_boxes")
                    for box in boxes:
                        out = v.draw_box(box.cpu())
                else:
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                cv2.imwrite(str(dir / f"model_output_{i}.jpg"), out.get_image())

            if len(outputs) > 1:
                raise Exception("Don't know what to do with more than one output type, expect either \"proposals\" for RPN or \"instances\" for full detector")
            elif len(outputs) < 1:
                raise Exception("No output returned from model")

            results[str(file).split("/")[-1]] = instances_to_dict(outputs["proposals" if cfg.MODEL.META_ARCHITECTURE == "ProposalNetwork" else "instances"])

    with open(opt.output, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str, default=None, help="Path to model save")
    parser.add_argument("dataset_path", type=str, default=None, help="Path to image folder")
    parser.add_argument("-o", "--output", type=str, default="detections.json")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-v", "--visualize", type=int, default=0, help="Number of samples to visualize")
    parser.add_argument("-rhst", "--roi_heads_score_thresh", type=float, default=None, help="For object detection")
    parser.add_argument("-npv", "--num_prop_vis", type=float, default=200, help="Number of proposals per image to visualize")
    load_model_args = add_func_args_to_parser(load_model)
    opt = parser.parse_args()

    run(opt, load_model_args)
