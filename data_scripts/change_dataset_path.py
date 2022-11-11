import json
from argparse import ArgumentParser
from pathlib import Path


# The dataset json files contain full paths to image locations, this script can be used to change the parent folder in the paths

parser = ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("old_path", type=str)
parser.add_argument("new_path", type=str)
opt = parser.parse_args()

path = Path(opt.dataset)

for split in ["train", "test"]:
    data = None
    with open(path / (split + ".json")) as f:
        data = json.loads(f.read())
    
    for d in data:
        d["file_name"] = d["file_name"].replace(opt.old_path, opt.new_path)
    
    with open(path / (split + ".json"), "w") as f:
        json.dump(data, f)