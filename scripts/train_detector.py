from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch, HookBase
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from datetime import timedelta
import os
from detectron2.data import transforms
import json, torch
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

from pcb_comp_detector.data import register_dataset
from pcb_comp_detector.load_model import replace_cfg


#resize = transforms.ResizeShortestEdge((800, 800), 1333, sample_style="choice")
train_augs = [
    transforms.RandomCrop("relative_range", (0.5, 0.5)),
    transforms.RandomBrightness(2/3, 1.5),
    transforms.RandomContrast(0.5, 2),
    transforms.RandomSaturation(0.5, 2),
    transforms.RandomLighting(300),
    transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    transforms.ResizeScale(min_scale=0.2, max_scale=1, target_height=1333, target_width=1333)
]

OPTIM_OVERRIDES = {
    "vis_graph_prop_portion": {"weight_decay": 0},
    "alt_graph_prop_portion": {"weight_decay": 0},
    "graph_prop_portion": {"weight_decay": 0},
    "graph_prop_temperature": {"weight_decay": 0},
}
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=train_augs)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides=OPTIM_OVERRIDES
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

class LogGraphPropParamsHook(HookBase):
    def after_step(self):
        if self.trainer.iter % 20 == 0:
            m = self.trainer.model.roi_heads
            print(f"graph_prop_temperature: {m.graph_prop_temperature:.5f},  vis_graph_prop_portion: {m.vis_graph_prop_portion:.5f}")

def train(opt):
    cfg = get_cfg()
    cfg.merge_from_file(opt.cfg)
    cfg.OUTPUT_DIR = opt.out_dir

    dataset_path = opt.dataset if opt.resume is None else cfg.DATASETS.TRAIN[0][:cfg.DATASETS.TRAIN[0].rfind("/")]
    _, _, classes, train_name, test_name = register_dataset(dataset_path)

    if opt.resume is None:
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = (test_name,)

    cfg.DATALOADER.NUM_WORKERS = opt.num_workers

    cfg.SOLVER.IMS_PER_BATCH = opt.batch_size
    # cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = opt.max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.AMP.ENABLED = not opt.no_amp

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = opt.head_batch_size_per_image

    if not opt.device is None:
        cfg.MODEL.DEVICE = opt.device

    if opt.proposal_net:
        cfg.MODEL.META_ARCHITECTURE = "ProposalNetwork"

    if not opt.init_weights is None:
        cfg.MODEL.WEIGHTS = opt.init_weights

    if opt.num_gpu < 2:
        cfg = replace_cfg(cfg, "SyncBN", "BN")

    if opt.resume is None:
        with open(str(Path(opt.out_dir) / "cfg.yaml"), "w") as f:
            f.write(cfg.dump())

        classes_rev = [None]*len(classes)
        for n, v in classes.items():
            classes_rev[v] = n
        with open(str(Path(opt.out_dir) / "classes_rev.json"), "w") as f:
            json.dump(classes_rev, f)

    trainer = CustomTrainer(cfg)
    if cfg.MODEL.META_ARCHITECTURE == "GraphPropRCNN":
        trainer.register_hooks([LogGraphPropParamsHook()])
    trainer.resume_or_load(resume=not opt.resume is None)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to model save when resuming")

    parser.add_argument("-ds", "--dataset", type=str, default="datasets/combined/wacv_fics_6class")
    parser.add_argument("--cfg", type=str, default="config/SoCo-FPN.yaml")
    parser.add_argument("-d", "--device", type=str, default=None, help="Only used if ngpu = -1")
    parser.add_argument("-ngpu", "--num_gpu", type=int, default=-1, help="Setting to >0 will override device")
    parser.add_argument("-o", "--out_dir", type=str, default=None, help="Set automatically if not specified.")

    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("-pn", "--proposal_net", action="store_true", default=False)
    parser.add_argument("-bs", "--batch_size", type=int, default=3)
    parser.add_argument("-hbspi", "--head_batch_size_per_image", type=float, default=256)
    parser.add_argument("-iw", "--init_weights", type=str, default=None, help="Pre-trained weights for initialization")
    parser.add_argument("-nw", "--num_workers", type=int, default=8)
    parser.add_argument("-mi", "--max_iter", type=int, default=100000)
    opt = parser.parse_args()


    def gen_output_dir():
        now = datetime.now()
        return now.strftime("output/%m-%d-%H:%M") + ("-pn" if opt.proposal_net else "")

    if opt.out_dir is None:
        if opt.resume is None:
            opt.out_dir = gen_output_dir()
            os.makedirs(opt.out_dir)
        else:
            slashloc = opt.resume.rfind("/")
            if slashloc == -1:
                opt.out_dir = gen_output_dir()
                os.makedirs(opt.out_dir)
            else:
                opt.out_dir = opt.resume[:slashloc]
    out_dir = Path(opt.out_dir)

    if not opt.resume is None:
        opt.cfg = str(out_dir / "cfg.yaml")


    os.makedirs(out_dir, exist_ok=True)

    if opt.num_gpu > 0:
        launch(
            train,
            opt.num_gpu,
            args=(opt,),
            timeout = timedelta(minutes=3)
        )
    else:
        train(opt)
