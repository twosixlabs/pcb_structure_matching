from argparse import ArgumentParser
from pathlib import Path
import json
import os
from pcb_comp_detector.data import register_dataset

parser = ArgumentParser()
parser.add_argument("dirs", nargs="+", type=str)
parser.add_argument("-v", "--visualize", type=int, default=0, help="Number of samples to save with annotation visualization, set -1 for the full dataset")
opt = parser.parse_args()

if len(opt.dirs) < 2:
    raise Exception("Must provide at least two datasets")

dirs = [Path(d) for d in opt.dirs]


# Compare classes

class_sets = []
class_lookups = []
for dir in dirs:
    with open(dir / "classes.json") as f:
        class_lookups.append(json.loads(f.read()))
class_sets = [set(l.keys()) for l in class_lookups]

all_classes = set()
for s in class_sets:
    all_classes = all_classes.union(s)

inconsistent = False
for s in class_sets:
    if len(s.intersection(all_classes)) != len(s):
        inconsistent = True

if inconsistent:
    warnings.warn("Inconsistent classes found.")


# Set out dir

def get_last_folder(p, offset=0):
    p = p.split("/")
    return p[-2 - offset] if p[-1] == "" else p[-1 - offset]

class_set_names = [get_last_folder(d) for d in opt.dirs]
class_set_name_consistent = len(set(class_set_names)) == 1
out_dir = "datasets/combined/"
for dir, name in zip(opt.dirs, class_set_names):
    ds_name = get_last_folder(dir, offset=1)
    out_dir += ds_name + ("_" if class_set_name_consistent else "_" + name + "_")
out_dir = out_dir + (class_set_names[0] if class_set_name_consistent else out_dir[:-1])
print(out_dir)
os.makedirs(out_dir)
out_dir = Path(out_dir)


# Combine datasets

class_lookup = {}
for i, cls in enumerate(all_classes):
    class_lookup[cls] = i

with open(out_dir / "classes.json", "w") as f:
    json.dump(class_lookup, f)

def invert_hashmap(h):
    h2 = {}
    for k,v in h.items():
        h2[v] = k
    return h2
inv_class_lookups = [invert_hashmap(l) for l in class_lookups]

for file in ["train.json", "test.json"]:
    combined = []
    for dir, icl in zip(dirs, inv_class_lookups):
        with open(dir / file) as f:
            data = json.loads(f.read())
            for img in data:
                for ann in img["annotations"]:
                    ann["category_id"] = class_lookup[icl[ann["category_id"]]]
            combined += data

    with open(out_dir / file, "w") as f:
        json.dump(combined, f)


register_dataset(out_dir, vis=opt.visualize)
