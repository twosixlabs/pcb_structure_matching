# Converts similar sets (ss) data from a flat folder with sets denoted in a json file to subfolder structure

import json, os, shutil
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("img_dir", type=str)
parser.add_argument("json_file", type=str)
parser.add_argument("out_dir", type=str)
opt = parser.parse_args()

data = []
with open(opt.json_file) as f:
    data = json.loads(f.read())

os.makedirs(opt.out_dir, exist_ok=True)
out_dir = Path(opt.out_dir)
in_dir = Path(opt.img_dir)

for i, sim_set in enumerate(data):
    folder = out_dir / str(i)
    os.makedirs(folder)

    for file in sim_set:
        name = Path(file).name
        shutil.copy(in_dir / name, folder / name)