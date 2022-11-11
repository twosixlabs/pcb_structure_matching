from PIL import Image, ImageDraw
from argparse import ArgumentParser
import numpy as np
import clip
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import os
import json
import transformers
import time
from progress.bar import PixelBar
from detectron2.structures import Boxes
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

import pcb_comp_detector as cd
import pcb_structure_matching as sm
import pcb_structure_embedding as se

from scipy.stats import percentileofscore

# Argparsing

CUSTOM_MODELS = ["clip", "dino", "dino_pcb"]
HF_TRANSFORMER_MODELS = ["beit"]
TV_MODELS = {
    "resnet18": torchvision.models.resnet18,
    "efficientnet_b0": torchvision.models.efficientnet_b0,
    # "efficientnet_b7": torchvision.models.efficientnet_b7,
    # "vit_b_16": torchvision.models.vit_b_16
}
EMB_MODELS = CUSTOM_MODELS + HF_TRANSFORMER_MODELS + list(TV_MODELS.keys())
EMB_PREFIX = "emb_"
EMB_APPROACHES = [EMB_PREFIX + m for m in EMB_MODELS]

SM_PREFIX = "sm_"
SE_PREFIX = "se_"

DINO_PCB_PATH = "dino_pcb.pth"

torch.hub.set_dir("../torch_hub")


parser = ArgumentParser()
parser.add_argument("img_dir", type=str, help="Directory of images to analyze")
parser.add_argument("-ed", "--emb_dir", type=str, default=None)
parser.add_argument("-ss", "--sim_sets", action="store_true", default=False, help="Perform analysis on folder of subfolders rather than a flat folder, with each subfolder containing a set of PCBs that should have 100% similarity.")
parser.add_argument("-nrcps", "--num_random_comps_per_sample", type=int, default=1000, help="If > dataset size, does the whole dataset")
parser.add_argument("--approaches", type=str, nargs="+", default=EMB_APPROACHES, help="emb_[model] for embedding comparison, sm_PATH_TO_MODEL for structural similarity (where the model is a .pth file and a cfg.yaml in detectron2 format is present in the same directory)")
parser.add_argument("-aaea", "--add_all_emb_approaches", action="store_true", default=False)
parser.add_argument("--print_per_img", action="store_true", default=False)
parser.add_argument("-pc", "--plot_combined", action="store_true", default=False)
parser.add_argument("-ps", "--plot_small", action="store_true", default=False)
parser.add_argument("-pb", "--plot_best", action="store_true", default=False)
parser.add_argument("-bs", "--batch_size", type=int, default=10)
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-fd", "--fig_dir", type=str, default="evaluate_comparison_figs")
parser.add_argument("--max_num", type=int, default=99999999, help="Max number of subfolders for sim_sets mode")
parser.add_argument("--size", type=int, nargs="+", default=[224, 224], help="Input size for embedding models")

parser.add_argument("--dino_pcb_path", type=str, default="dino_pcb.pth")
opt = parser.parse_args()


# Data setup

if opt.add_all_emb_approaches:
    opt.approaches += EMB_APPROACHES

img_dir = Path(opt.img_dir)
emb_dir = None if opt.emb_dir is None else Path(opt.emb_dir)
fig_dir = Path(opt.fig_dir)

if not fig_dir.exists():
    os.makedirs(fig_dir, exist_ok=True)

img_files = sorted(img_dir.glob("*"))
if opt.sim_sets:
    img_files = [sorted(f.glob("*")) for f in img_files if f.is_dir()]

if len(img_files) > opt.max_num:
    img_files = img_files[:opt.max_num]

n = len(img_files)
n_img = np.sum([len(subdir) for subdir in img_files]) if opt.sim_sets else n

if opt.sim_sets:
    print(f"Sets: {n}")
    print(f"Total Images: {n_img}")
else:
    print(f"Images: {n_img}")


def obstruct(img, max_w=0.5, max_h=0.5):
    w, h = img.size

    bw, bh = np.random.randint(1, w//2), np.random.randint(1, h//2)
    bx, by = np.random.randint(1, w-bw), np.random.randint(1, h-bh)
    bcr, bcg, bcb = np.random.randint(1, 256), np.random.randint(1, 256), np.random.randint(1, 256) # color
    blocked_img = img.copy()
    draw = ImageDraw.Draw(blocked_img, "RGB")
    draw.rectangle((bx-bw//2, by-bh//2, bx+bw//2, by+bh//2), fill=(bcr, bcg, bcb))
    return blocked_img

def corner_crops(img):
    # Four corner crops, ensuring all are same size
    w, h = img.size
    oddw = w % 2 == 1
    oddh = h % 2 == 1
    return [
        img.crop((0, 0, w//2, h//2)),
        img.crop((0, h//2 + (1 if oddh else 0), w//2, h)),
        img.crop((w//2 + (1 if oddw else 0), 0, w, h//2)),
        img.crop((w//2 + (1 if oddw else 0), h//2 + (1 if oddh else 0), w, h))
    ]

def clean_approach_name(name):
    return name.replace("/", "__").replace(".", "_")

class ImageListDataset(data.Dataset):
    def __init__(self, list, transform=lambda x: x, corner_crops=False, obstruct=False):
        self.list = list
        self.transform = transform
        self.corner_crops = corner_crops
        self.obstruct = obstruct

    def __getitem__(self, index):
        img = Image.open(self.list[index]).convert("RGB")
        if self.obstruct:
            img = obstruct(img)

        if self.corner_crops:
            imgs = corner_crops(img)
            return torch.stack([self.transform(x) for x in imgs])

        return self.transform(img)

    def __len__(self):
        return len(self.list)

class ImageSetListDataset(ImageListDataset):
    def __getitem__(self, index):
        imgs = [Image.open(item).convert("RGB") for item in self.list[index]]
        if self.obstruct:
            for i in range(len(imgs)):
                imgs[i] = obstruct(imgs[i])

        return [self.transform(i) for i in imgs]

def get_children(model: torch.nn.Module):
    children = list(model.children())
    flat_children = []
    if children == []:
        return model
    else:
       for child in children:
            try:
                flat_children.extend(get_children(child))
            except TypeError:
                flat_children.append(get_children(child))
    return flat_children

activation = None
def save_activation(mod, inp, out):
    global activation
    assert len(inp) == 1
    activation = inp[0]

def to_list(obj):
    if isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif type(obj) is dict:
        for k in obj.keys():
            obj[k] = to_list(obj[k])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = to_list(obj[i])
    return obj


# Comparator

class Comparator:
    def __init__(self, name, emb_dir, img_files, device, batch_size, size, sim_sets=False, num_workers=8):
        self.all_embeddings = []
        self.all_embeddings_c = []
        self.all_embeddings_cj = []
        self.all_embeddings_b = []
        self.all_embeddings_cj_b = []
        self.all_embeddings_c_cj_b = []
        self.sim_embeddings = []
        self.sim_embeddings_cj = []
        self.sim_embeddings_b = []
        self.sim_embeddings_cj_b = []
        self.name = name
        self.device = device
        self.size = size

        self.preprocess_f = transforms.ToTensor()
        self.embed_f = lambda x: x

        self.is_sm = self.name.startswith(SM_PREFIX)
        self.is_se = self.name.startswith(SE_PREFIX)

        file_name = emb_dir + clean_approach_name(name) + "_embeddings_" + str(n) + "_" + ("sim" if opt.sim_sets else "all") + ".json"

        # Check if all_embeddings already stored
        if os.path.exists(file_name):
            print("Loading stored embeddings...")
            with open(file_name) as f:
                if sim_sets:
                    d = json.loads(f.read())
                    self.sim_embeddings = d["embeddings"]
                    self.sim_embeddings_cj = d["embeddings_cj"]
                    self.sim_embeddings_b = d["embeddings_b"]
                    self.sim_embeddings_cj_b = d["embeddings_cj_b"]
                else:
                    d = json.loads(f.read())
                    self.all_embeddings = d["embeddings"]
                    self.all_embeddings_c = d["embeddings_c"]
                    self.all_embeddings_cj = d["embeddings_cj"]
                    self.all_embeddings_b = d["embeddings_b"]
                    self.all_embeddings_cj_b = d["embeddings_cj_b"]
                    self.all_embeddings_c_cj_b = d["embeddings_c_cj_b"]
        else:
            print("Loading model to generate new embeddings.")

            cj = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)

            self.set_preprocess_encode_funcs()


            # Set up data loader params

            loader_params = []
            if sim_sets:
                loader_params = [
                    ("similar images", ImageSetListDataset, {"transform": self.preprocess_f}, self.sim_embeddings),
                    ("color jittered similar images", ImageSetListDataset, {"transform": transforms.Compose([cj, self.preprocess_f])}, self.sim_embeddings_cj),
                    ("blocked similar images", ImageSetListDataset, {"transform": self.preprocess_f, "obstruct": True}, self.sim_embeddings_b),
                    ("color jittered + blocked similar images", ImageSetListDataset, {"transform": transforms.Compose([cj, self.preprocess_f])}, self.sim_embeddings_cj_b)
                ]
            else:
                n_args = {"transform": self.preprocess_f, "corner_crops": False}
                c_args = {"transform": self.preprocess_f, "corner_crops": True}
                cj_args = {"transform": transforms.Compose([cj, self.preprocess_f]), "corner_crops": False}
                b_args = {"transform": self.preprocess_f, "obstruct": True, "corner_crops": False}
                cj_b_args = {"transform": transforms.Compose([cj, self.preprocess_f]), "obstruct": True, "corner_crops": False}
                ccjb_args = {"transform": transforms.Compose([cj, self.preprocess_f]), "obstruct": True, "corner_crops": True}
                loader_params = [
                    ("original", ImageListDataset, n_args, self.all_embeddings),
                    ("crops", ImageListDataset, c_args, self.all_embeddings_c),
                    ("color jittered", ImageListDataset, cj_args, self.all_embeddings_cj),
                    ("blocked", ImageListDataset, b_args, self.all_embeddings_b),
                    ("color jittered + blocked", ImageListDataset, cj_b_args, self.all_embeddings_cj_b),
                    ("cropped + jittered + blocked", ImageListDataset, ccjb_args, self.all_embeddings_c_cj_b)
                ]


            # Generate embeddings

            for name, ds_class, ds_args, emb_list in loader_params:
                # This section is not batching sim_sets or if using sm so it's a little messy

                bs = batch_size
                if self.is_sm:
                    bs = 1 # may be unnecessary
                elif sim_sets:
                    bs = 1 # may be unnecessary
                elif ds_args["corner_crops"]:
                    bs = bs // 4

                loader = data.DataLoader(ds_class(list=img_files, **ds_args), batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

                bar = PixelBar(f"Generating embeddings for {name:40}", max=len(loader), width=50, suffix='%(percent)d%% (%(elapsed)s)')
                for i, img in enumerate(loader):
                    bar.next()
                    to_append = None

                    if sim_sets:
                        to_append = []
                        for i in img:
                            e = self.embed_f(i.to(self.device))
                            to_append.append(e if self.is_sm else e.squeeze().cpu())
                    else:
                        if ds_args["corner_crops"]:
                            imgs_r = img.reshape(-1, *img.shape[2:]).float().to(self.device)
                            embedded = self.embed_f(imgs_r)
                            to_append = embedded if self.is_sm else embedded.reshape(*img.shape[:2], -1).cpu()
                        else:
                            to_append = self.embed_f(img.to(self.device))
                            to_append = to_append if self.is_sm else to_append.cpu()
                    emb_list.append(to_append)
                del loader
                bar.finish()

            if sim_sets:
                pass
            else:
                if not self.is_sm:
                    self.all_embeddings = torch.cat(self.all_embeddings, dim=0)
                    self.all_embeddings_c = torch.cat(self.all_embeddings_c, dim=0)
                    self.all_embeddings_cj = torch.cat(self.all_embeddings_cj, dim=0)
                    self.all_embeddings_b = torch.cat(self.all_embeddings_b, dim=0)
                    self.all_embeddings_cj_b = torch.cat(self.all_embeddings_cj_b, dim=0)
                    self.all_embeddings_c_cj_b = torch.cat(self.all_embeddings_c_cj_b, dim=0)


            # Save generated embeddings

            content = None
            if sim_sets:
                content = {
                    "embeddings": to_list(self.sim_embeddings),
                    "embeddings_cj": to_list(self.sim_embeddings_cj),
                    "embeddings_b": to_list(self.sim_embeddings_b),
                    "embeddings_cj_b": to_list(self.sim_embeddings_cj_b),
                    "files": [str(f) for f in img_files]
                }
            else:
                content = {
                    "embeddings": to_list(self.all_embeddings),
                    "embeddings_c": to_list(self.all_embeddings_c),
                    "embeddings_cj": to_list(self.all_embeddings_cj),
                    "embeddings_b": to_list(self.all_embeddings_b),
                    "embeddings_cj_b": to_list(self.all_embeddings_cj_b),
                    "embeddings_c_cj_b": to_list(self.all_embeddings_c_cj_b),
                    "files": [str(f) for f in img_files]
                }

            with open(file_name, "w") as f:
                json.dump(content, f)

    def compare(self, a, b, part_v_whole=False):
        if self.name.startswith(EMB_PREFIX) or self.is_se:
            # Cosine similarity
            return np.dot(a, b) / (np.linalg.norm(np.array(a) + 1e-7, 2) * np.linalg.norm(np.array(b) + 1e-7, 2)), None
        elif self.is_sm:
            # Switch from [-1, 1] to [0, 1]
            score, transform = sm.compare(a, b, whole=not part_v_whole, variant="single")
            return (score + 1) / 2, transform

    def set_preprocess_encode_funcs(self):
        if self.name.startswith(EMB_PREFIX):
            model_name = self.name[len(EMB_PREFIX):]

            self.preprocess_f = transforms.Compose([
                transforms.Resize(opt.size),
                transforms.ToTensor()
            ])

            if model_name in TV_MODELS:
                model = TV_MODELS[model_name](pretrained=True).to(self.device)
                get_children(model)[-1].register_forward_hook(save_activation)
                def a(x):
                    model(x)
                    global activation
                    return activation
                self.embed_f = a
                self.preprocess_f = transforms.Compose([self.preprocess_f, transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            elif model_name == "clip":
                model, preprocess = clip.load("ViT-B/16", device=self.device, download_root="../cache")
                self.preprocess_f = transforms.Compose([transforms.Resize(self.size), preprocess])
                self.embed_f = model.encode_image
            elif model_name == "dino":
                model = torch.hub.load("facebookresearch/dino:main", "dino_vits8").to(self.device)
                self.preprocess_f = transforms.Compose([self.preprocess_f, transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                self.embed_f = model
            elif model_name == "beit":
                # fe = transformers.BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
                model = transformers.BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k").to(self.device)
                self.embed_f = lambda x: model(x).pooler_output
            elif model_name == "dino_pcb":
                from timm.models.vision_transformer import VisionTransformer
                def strip_weight_names(state_dict, prefix):
                    import collections
                    dst = collections.OrderedDict()
                    for k, v in state_dict.items():
                        dst[k.replace(prefix, '')] = v
                    return dst

                model = VisionTransformer(
                    img_size=[320, 320],
                    in_chans=3,
                    num_classes=10, # not sure if this is needed
                    embed_dim=768,
                    depth=12,
                    num_heads=12,
                    mlp_ratio=4,
                    drop_rate=0,
                    patch_size=16,
                    drop_path_rate=0.1,  # stochastic depth
                ).to(self.device)

                sd = torch.load(opt.dino_pcb_path, map_location=self.device)
                student_st = sd['student']
                stripped_student_st = strip_weight_names(strip_weight_names(student_st, 'module.backbone.'), 'module.head.')

                stripped_student_st.pop('pos_embed')  # Needs to be done for size change.

                model.load_state_dict(stripped_student_st, strict=False)

                self.preprocess_f = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.Resize((320, 320)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                self.embed_f = model

        elif self.is_sm:
            model_name = self.name[len(SM_PREFIX):]
            model, _, _ = cd.load_model(model_name, device=self.device)

            def embed(x):
                r = model([{"image": i.permute(2,1,0).cpu().numpy()*255} for i in x])
                for i, ri in enumerate(r):
                    h, w = ri["instances"].image_size
                    ri = ri["instances"].get_fields()
                    ri["image_width"] = w
                    ri["image_height"] = h
                    for field in ri.keys():
                        if isinstance(ri[field], Boxes):
                            ri[field] = ri[field].tensor.to("cpu")
                        elif isinstance(ri[field], torch.Tensor):
                            ri[field] = ri[field].to("cpu")
                    r[i] = ri

                return r[0] if len(r) == 1 else r

            self.embed_f = embed
        elif self.is_se:
            names = self.name[len(SE_PREFIX):].split("+")
            detector_name, embedder_name = names[0], names[1]
            detector, _, classes = cd.load_model(detector_name, device=self.device)

            embedder = se.load_model(embedder_name, device=self.device)
            embedder.eval()

            def embed(x):
                r = detector([{"image": i.permute(2,1,0).cpu().numpy()*255} for i in x])
                for i, ri in enumerate(r):
                    h, w = ri["instances"].image_size
                    ri = ri["instances"].get_fields()
                    ri["image_width"] = w
                    ri["image_height"] = h
                    for field in ri.keys():
                        if isinstance(ri[field], Boxes):
                            ri[field] = ri[field].tensor.to("cpu")
                        elif isinstance(ri[field], torch.Tensor):
                            ri[field] = ri[field].to("cpu")
                    r[i] = ri

                r = [se.utils.format_input(i["pred_boxes"], i["pred_classes"], len(classes), i["image_width"], i["image_height"]).to(self.device) for i in r]
                r = embedder(r)
                return r[0] if len(r) == 1 else r

            self.embed_f = embed
        else:
            raise Exception(f"Approach name \"{self.name}\" not recognized.")


# Perform test
warnings.filterwarnings("ignore", message="No components overlapping")


# What percentile is the similarity score given to similar images (among all similarity scores, estimated by randomly comparing to other images)
similarity_percentiles = None
msgs = None
target_transforms = None
transform_errs = None
sm_approaches = [a for a in opt.approaches if a.startswith(SM_PREFIX)]
if opt.sim_sets:
    similarity_percentiles = {k: [[None]*n]*len(opt.approaches) for k in ["sim", "sim_cj", "sim_b", "sim_cj_b"]}
    msgs = {
        "sim": "matching images",
        "sim_cj": "color jittered matching images",
        "sim_b": "obstructed matching images",
        "sim_cj_b": "color jittered + obstructed matching images"
    }
else:
    similarity_percentiles = {
        "cw": np.zeros((len(opt.approaches), n, 4)),
        "ec": np.zeros((len(opt.approaches), n, 6)),
        "cj": np.zeros((len(opt.approaches), n, 1)),
        "b": np.zeros((len(opt.approaches), n, 1)),
        "cj_b": np.zeros((len(opt.approaches), n, 1)),
        "cw_cj_b": np.zeros((len(opt.approaches), n, 4))
    }
    msgs = {
        "cw": "crop and whole",
        "ec": "exclusive crops",
        "cj": "original and color jittered",
        "b": "original and blocked",
        "cj_b": "original and color jittered + blocked",
        "cw_cj_b": "original and cropped + color jittered + blocked"
    }
    target_transforms = {
        "cw": np.array([(0, 0, 0, 1), (0, 0, 0.5, 1), (0, 0.5, 0, 1), (0, 0.5, 0.5, 1)]),
        "cj": np.array([(0, 0, 0, 1)]),
        "b": np.array([(0, 0, 0, 1)]),
        "cj_b": np.array([(0, 0, 0, 1)]),
        "cw_cj_b": np.array([(0, 0, 0, 1), (0, 0, 0.5, 1), (0, 0.5, 0, 1), (0, 0.5, 0.5, 0)]),
    }
    # Scale, the last element, is not scale error, it's just scale
    transform_errs = {
        "cw": np.zeros((len(sm_approaches), n, 4, 4)),
        "cj": np.zeros((len(sm_approaches), n, 1, 4)),
        "b": np.zeros((len(sm_approaches), n, 1, 4)),
        "cj_b": np.zeros((len(sm_approaches), n, 1, 4)),
        "cw_cj_b": np.zeros((len(sm_approaches), n, 4, 4))
    }

with torch.no_grad():
    sm_approach_i = 0
    for approach_i, approach_name in enumerate(opt.approaches):
        print(f"\n=== {approach_name} ===")

        file_name_sim_percentiles = opt.emb_dir + clean_approach_name(approach_name) + f"_{n}_sim_percentiles_{'sim' if opt.sim_sets else 'all'}.json"

        if os.path.exists(file_name_sim_percentiles):
            print(f"Loading stored image comparisons for {approach_name}")
            with open(file_name_sim_percentiles) as f:
                loaded = json.loads(f.read())
                for k in loaded["similarity_percentiles"].keys():
                    similarity_percentiles[k][approach_i] = loaded["similarity_percentiles"][k] # maybe np.array for all
                if not loaded["transform_errs"] is None:
                    for k in loaded["transform_errs"].keys():
                        transform_errs[k][approach_i] = loaded["transform_errs"][k] # maybe np.array for all

        else:
            comparator = Comparator(approach_name, opt.emb_dir, img_files, opt.device, opt.batch_size, opt.size, sim_sets=opt.sim_sets)

            # Comparison analysis for all images

            bar = PixelBar(f"Comparing images ", max=len(comparator.sim_embeddings if opt.sim_sets else comparator.all_embeddings), width=50, suffix='%(percent)d%% (%(elapsed)s)')
            random_other_embs_flattened = [i for sub in comparator.sim_embeddings for i in sub] if opt.sim_sets else comparator.all_embeddings

            # For each embedded image...
            sim_ind = 0
            for i in range(len(comparator.sim_embeddings if opt.sim_sets else comparator.all_embeddings)):
                bar.next()

                # Compare to random other embeddings from dataset
                inds = None
                if opt.sim_sets:
                    set_n = len(comparator.sim_embeddings[i])
                    inds = list(range(0,sim_ind)) + list(range(sim_ind+set_n, len(random_other_embs_flattened)))
                    sim_ind += set_n
                else:
                    inds = list(range(0,i)) + list(range(i+1,len(comparator.all_embeddings)))
                inds = np.random.permutation(inds)
                inds = inds[:opt.num_random_comps_per_sample] if len(inds) > opt.num_random_comps_per_sample else inds

                random_sims = []
                for curr_emb in comparator.sim_embeddings[i] if opt.sim_sets else [comparator.all_embeddings[i]]:
                    for j in inds:
                        v = comparator.compare(curr_emb, random_other_embs_flattened[j])[0]
                        random_sims.append(v)
                random_sims = np.array(random_sims)

                def calc_percentile(test_name, pairs, record_transform=True):
                    vals = [comparator.compare(*pair, part_v_whole="cw" in test_name) for pair in pairs]
                    sizes = [(p[0]["image_width"], p[0]["image_height"]) if comparator.is_sm else None for p in pairs]

                    if opt.sim_sets:
                        similarity_percentiles[test_name][approach_i][i] = [None]*len(vals)

                    for vi, ((v, tr), size) in enumerate(zip(vals, sizes)):
                        similarity_percentiles[test_name][approach_i][i][vi] = percentileofscore(random_sims, v, kind="strict") # must be strict in case many are the same

                        if record_transform and not tr is None and test_name in target_transforms:
                            w, h = size
                            dx = tr[1]/w
                            dy = tr[2]/h
                            target = target_transforms[test_name][vi]
                            err = np.array((int(tr[0] == target[0]), target[1]-dx, target[2]-dy, tr[3]))
                            transform_errs[test_name][sm_approach_i, i, vi] = err

                if opt.sim_sets:
                    for test_case, embeddings in [("sim", comparator.sim_embeddings), ("sim_cj", comparator.sim_embeddings_cj), ("sim_b", comparator.sim_embeddings_b), ("sim_cj_b", comparator.sim_embeddings_cj_b)]:
                        s = embeddings[i]
                        calc_percentile(test_case, [(s[k], s[l]) for k in range(len(s)) for l in range(k+1, len(s))], record_transform=False)
                else:
                    o = comparator.all_embeddings[i] # original
                    c = comparator.all_embeddings_c[i] # cropped
                    cj = comparator.all_embeddings_cj[i] # color jittered
                    b = comparator.all_embeddings_b[i] # blocked
                    cjb = comparator.all_embeddings_cj_b[i] # color jittered + blocked
                    ccjb = comparator.all_embeddings_c_cj_b[i] # all 3

                    # Similarity percentile between crop and whole
                    calc_percentile("cw", [(o, c[0]), (o, c[1]), (o, c[2]), (o, c[3])])

                    # Similarity percentile between exclusive crops
                    calc_percentile("ec", [(c[k], c[l]) for k in range(len(c)) for l in range(k+1, len(c))])

                    # Similarity percentile between original and color jittered
                    calc_percentile("cj", [(o, cj)])

                    # Similarity percentile between original and blocked
                    calc_percentile("b", [(o, b)])

                    # Similarity percentile between original and blocked
                    calc_percentile("cj_b", [(o, cjb)])

                    # Similarity percentile between original and color jittered + blocked + cropped
                    calc_percentile("cw_cj_b", [(o, ccjb[0]), (o, ccjb[1]), (o, ccjb[2]), (o, ccjb[3])])

            bar.finish()

            with open(file_name_sim_percentiles, "w") as f:
                json.dump({
                    "similarity_percentiles": {k: to_list(v[approach_i]) for k, v in similarity_percentiles.items()},
                    "transform_errs": None if transform_errs is None or not comparator.is_sm else {k: to_list(v[sm_approach_i]) for k, v in transform_errs.items()}
                }, f)

            del comparator

            if approach_name.startswith(SM_PREFIX):
                sm_approach_i += 1

if opt.sim_sets:
    for test_case in similarity_percentiles.keys():
        similarity_percentiles[test_case] = np.array([[i for res in appr for i in res] for appr in similarity_percentiles[test_case]])

def top_k_perc(k):
    return 100*(n_img-k-1e-6)/(n_img)

top_k_rate = lambda percentiles, k: (percentiles >= top_k_perc(k)).mean()*100

top_ks_to_plot = [2**j for j in range(0,int(np.log2(n_img))+1)] + [n_img]

def setup_plot(msg):
    f = plt.figure()
    p = f.add_subplot(1, 1, 1)
    if not opt.plot_small:
        p.set_title("Percent of matches where similarity between " + msg + " falls in the top k similarities to the rest of the dataset", wrap=True)
        p.set_ylabel("Percent of dataset where match similarity in top k (%)")
        p.set_xlabel("Top k (log scale)")
    p.set_xscale("log")
    p.set_xticks(top_ks_to_plot[:-1])
    p.set_xticklabels([] if opt.plot_small else [f"{k}" for k in top_ks_to_plot][:-1])
    if opt.plot_small:
        p.set_yticklabels([])
    p.set_ylim(0, 101)
    p.set_xlim(0, n_img)
    return f, p

pretty_names = {
    "emb_clip": "CLIP Embedding",
    "emb_dino": "DINO",
    "emb_dino_pcb": "DINO PCB",
    "emb_resnet18": "ResNet18",
    "emb_beit": "BEiT",
    "emb_efficientnet_b0": "EfficientNetB0",
}

# Calculate best approach and best approach score for each test case
for test_name in similarity_percentiles.keys():
    print(f"\n=== Comparison analysis between {msgs[test_name]} ===")

    similarity_percentiles[test_name] = similarity_percentiles[test_name].reshape((similarity_percentiles[test_name].shape[0], -1))

    # Which approach gave a similarity score that was the highest percentile of all pairwise similarity scores
    best_approach_rankings = np.argsort(similarity_percentiles[test_name], axis=0)

    # How much better was the best percentile than the next best, i.e. value add to analysis. More specifically, 1 - (100-second best approach percentile) / (100-best approach percentile), so halving the number of samples that are more similar gives a score of 1, reducing by a third gives 0.5, reducing by a quarter gives 0.33, etc.
    #best_approach_score = (100-similarity_percentiles[test_name][best_approach_rankings[-2], np.arange(similarity_percentiles[test_name].shape[1])]) / (100-similarity_percentiles[test_name][best_approach_rankings[-1], np.arange(similarity_percentiles[test_name].shape[1])]+0.001)

    best_perc = similarity_percentiles[test_name][best_approach_rankings[-1], np.arange(similarity_percentiles[test_name].shape[1])]

    fig1, plot1 = setup_plot(msgs[test_name])

    best_ys = [top_k_rate(best_perc, k) for k in top_ks_to_plot]
    if opt.plot_best:
        plot1.plot(top_ks_to_plot, best_ys, label="Best Result")

    combined_ys = [top_k_rate(best_perc, k/len(opt.approaches)) for k in top_ks_to_plot]
    if opt.plot_combined:
        plot1.plot(top_ks_to_plot[1:], combined_ys[1:], label="Combined Result", linewidth=8 if opt.plot_small else 4)

    print(f"Best Result Average Top k Rate:     {np.mean(best_ys[1:-1]):15.4}%")
    print(f"Combined Result Average Top k Rate: {np.mean(combined_ys[1:-1]):15.4}%")

    plot1.plot(top_ks_to_plot, [100*k / n_img for k in top_ks_to_plot], label="Random", linewidth=4 if opt.plot_small else 2)

    for approach_i in range(len(opt.approaches)):
        print("=> " + opt.approaches[approach_i])
        approach_perc = similarity_percentiles[test_name][approach_i]
        approach_name = opt.approaches[approach_i]

        ys = [top_k_rate(approach_perc, k) for k in top_ks_to_plot]
        name = approach_name
        if approach_name in pretty_names:
            name = pretty_names[approach_name]
        elif approach_name.startswith(SM_PREFIX):
            name = "Structure Matching"
        elif len(approach_name) > 40:
            name = "..." + approach_name[37:]
        plot1.plot(top_ks_to_plot, ys, label=name, linewidth=4 if opt.plot_small else 2)

        print(f"Percent of Samples Where Best:      {(best_approach_rankings[-1] == approach_i).mean()*100:15.4}%")
        #print(f"Total Score:                        {best_approach_score[best_approach_rankings[-1] == approach_i].sum()/n:16.6}")
        print(f"Average Top k Rate:                 {np.mean(ys[1:-1]):15.4}%")

    if not opt.plot_small:
        plot1.legend()#loc="lower right")
    fig1.savefig(fig_dir / (test_name + "_all_result_in_top_k.png"))


if not transform_errs is None:
    for test_name in transform_errs.keys():
        print(f"\n=== Alignment analysis between {msgs[test_name]} ===")
        print(f"{'Approach':16}  {'Rotation Acc':16}  {'Delta X Err':16}  {'Delta Y Err':16}  {'Scale':16}  {'Overall Acc':16}")

        transform_errs[test_name] = transform_errs[test_name].reshape((transform_errs[test_name].shape[0], -1, transform_errs[test_name].shape[3]))

        for sm_approach_i in range(len(sm_approaches)):
            rot_acc = transform_errs[test_name][sm_approach_i,:,0].mean()*100
            delta_xs_err = transform_errs[test_name][sm_approach_i,:,1]
            delta_ys_err = transform_errs[test_name][sm_approach_i,:,2]
            scalars = transform_errs[test_name][sm_approach_i,:,3]
            acc = (rot_acc.astype(bool) & (np.abs(delta_xs_err) < 0.1) & (np.abs(delta_ys_err) < 0.1) & (scalars < 1.1) & (scalars > 0.91)).mean()
            print(f"{sm_approaches[sm_approach_i]:20}  {rot_acc:<11.1f}%  {delta_xs_err.mean():<6.3f} +/-{delta_xs_err.std():<6.3f}  {delta_ys_err.mean():<6.3f} +/-{delta_ys_err.std():<6.3f}  {scalars.mean():<6.3f} +/-{scalars.std():<6.3f}  {acc:<15.3f}%")
