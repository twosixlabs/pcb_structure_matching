from detectron2.config import get_cfg, CfgNode
# from detectron2.modeling.backbone import build_backbone
# from detectron2.modeling.proposal_generator import build_proposal_generator
# from detectron2.modeling.roi_heads import build_roi_heads
# from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import transforms
from detectron2.structures import Instances, pairwise_iou
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os, json, subprocess, random, torch, cv2

from pcb_comp_detector.data import register_dataset
from pcb_comp_detector.partial_label_utils import PartialLabelDatasetMapper, PartialLabelRCNN
from pcb_comp_detector import load_model


def eval(opt):
    slashloc = opt.path.rfind("/")
    dir = None
    if opt.path.endswith(".pth"):
        if slashloc == -1:
            raise Exception("Model save must be in a folder with cfg.yaml")
        else:
            dir = Path(opt.path[:slashloc])
        paths = [Path(opt.path)]
    else:
        dir = Path(opt.path)
        paths = [p for p in dir.iterdir() if str(p).endswith(".pth") and str(p).split("/")[-1].startswith("model_") and "9" in str(p).split("/")[-1]]

    cfg = get_cfg()
    cfg.merge_from_file(dir / "cfg.yaml")

    torch.cuda.set_device(opt.device)

    if len(cfg.DATASETS.TRAIN) != 1 and len(cfg.DATASETS.TEST) != 1:
        raise Exception("Must have a single train and test set")

    train_ds, test_ds, _, _, _ = register_dataset(cfg.DATASETS.TRAIN[0][:-6])

    test_augs = [
        transforms.ResizeShortestEdge((800, 800), 1333, sample_style="choice"),
    ]
    test_transform = transforms.AugmentationList(test_augs)

    train_evaluator = COCOEvaluator(cfg.DATASETS.TRAIN[0], output_dir=dir, tasks=("bbox",), max_dets_per_image=1000)
    test_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=dir, tasks=("bbox",), max_dets_per_image=1000)

    mapper = PartialLabelDatasetMapper(cfg, is_train=False, augmentations=test_augs)
    train_val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0], mapper=mapper)
    test_val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)

    out_dir = dir / "eval" / str(int(opt.partial_label_proportion*100))
    if not out_dir.exists():
        os.makedirs(out_dir)

    eval_results = {"train_set": {}, "test_set": {}}
    if len(paths) > 1:
        if (out_dir / "data.json").exists():
            with open(out_dir / "data.json") as f:
                eval_results = json.loads(f.read())
            for s in ["train_set", "test_set"]:
                eval_results[s] = {v["iterations"]: v for v in eval_results[s]}

    for path in sorted(paths):
        print(f"\n==> Evaluating {path}")

        iters = int(str(path).split("/")[-1].split(".")[0].split("_")[-1]) + 1
        if (iters in eval_results["train_set"] or not opt.eval_train) and iters in eval_results["test_set"]:
            print(f"Skipping {iters}, already saved")
            continue

        with torch.no_grad():
            predictor, cfg = load_model(path, device=opt.device, roi_heads_score_thresh=opt.detection_thresh, partial_label_proportion=opt.partial_label_proportion)

            if opt.visualize > 0:
                for i, (dataset_dicts, metadata) in enumerate(zip([train_ds, test_ds], [cfg.DATASETS.TRAIN[0], cfg.DATASETS.TEST[0]])):
                    for j, d in enumerate(random.sample(dataset_dicts, opt.visualize)):
                        im = cv2.imread(d["file_name"])
                        # cv2.imwrite(str(dir / f"pred_{i}_{j}_orig.jpg"), im)
                        inp = transforms.AugInput(im)
                        #test_trans = test_transform(inp)
                        outputs = predictor.default_call(inp.image)
                        v = Visualizer(inp.image,
                                       metadata=MetadataCatalog.get(metadata),
                                       scale=1
                        )

                        if cfg.MODEL.META_ARCHITECTURE == "ProposalNetwork":
                            threshold = np.partition(outputs["proposals"].objectness_logits.cpu().numpy())[-opt.num_prop_vis]
                            boxes = outputs["proposals"][outputs["proposals"].objectness_logits > threshold].get("proposal_boxes")
                            for box in boxes:
                                out = v.draw_box(box.cpu())
                        else:
                            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                        cv2.imwrite(str(dir / f"pred_{i}_{j}.jpg"), out.get_image())

            for i, (evaluator, loader, results) in enumerate(zip([train_evaluator, test_evaluator], [train_val_loader, test_val_loader], [eval_results["train_set"], eval_results["test_set"]])):
                if i == 0 and not opt.eval_train:
                    continue

                r = inference_on_dataset(predictor, loader, evaluator)["box_proposals" if cfg.MODEL.META_ARCHITECTURE == "ProposalNetwork" else "bbox"]
                r["iterations"] = iters
                results[iters] = r
                print(r)

    for s in ["train_set", "test_set"]:
        eval_results[s] = list(eval_results[s].values())

    if len(paths) > 1:
        print("Saving results and plots in output directory.")
        with open(out_dir / "data.json", "w") as f:
            json.dump(eval_results, f)

        plot_results(out_dir, eval_results["train_set"], eval_results["test_set"])

    return eval_results["train_set"], eval_results["test_set"]

def plot_results(dir, train_results, test_results):
    plt.xlabel("Iterations")

    for results, name in zip([train_results, test_results], ["train", "test"]):
        df = pd.DataFrame(results)
        if len(df) > 1:
            df = df.sort_values("iterations")

            plots = [
                (["AR@100", "AR@1000"], "AR"),
                ([c for c in df.columns if "AR" in c], "AR_full"),
                (["AP", "AP50", "AP75"], "AP"),
                (["APs", "APm", "APl"], "AP_size"),
                ([c for c in df.columns if "AP-" in c], "AP_class"),
                ([c for c in df.columns if "AP" in c], "AP_full"),
            ]
            for stats, plot_name in plots:
                if len(stats) < 1 or False in [s in df.columns for s in stats]:
                    continue

                df.plot(x="iterations", y=stats)
                plt.savefig(dir / f"{name}_{plot_name}.jpg")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str, default=None, help="Path to individual model save or folder of model saves in format model_[ITER].pth")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-v", "--visualize", type=int, default=0)
    parser.add_argument("-et", "--eval_train", action="store_true", default=False)
    parser.add_argument("-plp", "--partial_label_proportion", type=float, default=0, help="For partial labeling")
    parser.add_argument("-dt", "--detection_thresh", type=float, default=None, help="For object detection")
    parser.add_argument("-npv", "--num_prop_vis", type=float, default=200, help="Number of proposals per image to visualize")
    opt = parser.parse_args()

    if opt.partial_label_proportion < 0 or opt.partial_label_proportion > 1:
        raise Exception("Partial label proportion must be between 0 and 1.")

    eval(opt)
