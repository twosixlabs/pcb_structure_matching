import argparse
import os
from PIL import Image
from pathlib import Path
from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", type=str)
parser.add_argument("out_dir", type=str)
parser.add_argument("--max_area", type=float, default=1800*1800)
opt = parser.parse_args()

in_dir = Path(opt.in_dir)
out_dir = Path(opt.out_dir)

os.makedirs(opt.out_dir, exist_ok=True)

for file in os.listdir(opt.in_dir):
    img = Image.open(in_dir / file)
    img.save(out_dir / (file.split(".")[0] + ".jpg"))
