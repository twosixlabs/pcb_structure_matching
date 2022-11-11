import xml.etree.ElementTree as ET
from argparse import ArgumentParser
import os, random, json
from pathlib import Path
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import re
from PIL import Image
import warnings
import numpy as np

from pcb_comp_detector.data import visualize_ds


# This script expects PCB WACV to be in its original format and FICS PCB to be a directory with an "img" folder containing all images and an "annotation" folder containing all annotation files.


# classes should be ['connector', 'resistor', 'pads', 'emi filter', 'capacitor', 'resistor network', 'text', 'pins', 'component text', 'ic', 'clock', 'test point', 'led', 'diode', 'switch', 'button', 'inductor', 'transistor', 'jumper', 'ferrite bead', 'fuse', 'battery', 'display', 'potentiometer', 'zener diode', 'capacitor jumper', 'transformer', 'resistor jumper', 'heatsink', 'buzzer', 'diode zener array']

class_sets = {
    "fics": {
        "all": set(["resistor", "capacitor", "ic", "inductor", "diode", "transistor"]),
    },
    "wacv": {
        "fics": set(["resistor", "capacitor", "ic", "inductor", "diode", "transistor"]),
        "all": (['connector', 'resistor', 'pads', 'emi filter', 'capacitor', 'resistor network', 'text', 'pins', 'component text', 'ic', 'clock', 'test point', 'led', 'diode', 'switch', 'button', 'inductor', 'transistor', 'jumper', 'ferrite bead', 'fuse', 'battery', 'display', 'potentiometer', 'zener diode', 'capacitor jumper', 'transformer', 'resistor jumper', 'heatsink', 'buzzer', 'diode zener array', 'unknown']),
    },
    "dslr": {
        "all": set(["ic"])
    }
}

bad_samples = {
    "fics": set([]),
    "wacv": set(["Spartan6_Bottom.jpg", ]),
    "dslr": set([])
}


def format_dataset(opt):
    dir = Path(opt.dir)

    use_classes = set(opt.classes)
    for set_name in class_sets[opt.dataset].keys():
        if set_name in use_classes:
            use_classes.remove(set_name)
            use_classes = use_classes.union(class_sets[opt.dataset][set_name])

    unknown_classes = set(opt.unknown_classes)
    for c in unknown_classes: use_classes.remove(c)

    component_lookup = {}#{"bg_placeholder": 0}

    dataset_dir = Path(f"datasets/{opt.dataset}/{len(use_classes)}class{'u' if len(opt.unknown_classes) > 0 else ''}_{int(100*opt.test_portion)}t")
    os.makedirs(dataset_dir)

    if opt.max_shortest_edge != -1:
        os.makedirs(dataset_dir / "img")

    def lookup_component(name):
        if opt.dataset == "wacv":
            name = name.replace("\t", " ").rstrip().lstrip()
            if "component text" in name:
                name = "component text"
            elif "text" in name:
                name = "text"
            elif "ic" in name:
                name = "ic"
            elif "\"" in name:
                name = name.split("\"")
                name = (name[1] if name[0] == "" else name[0]).rstrip()
            else:
                name = name.split(" ")[0]

            name = re.split("(?=[A-Z])", name)[0].rstrip()

        elif opt.dataset == "fics":
            name = name[:-1].lower() # class names have an 's' at the end, remove to standardize to wacv names

        if not name in use_classes:
            if name in unknown_classes:
                name = "unknown"
            else:
                return None

        if name in component_lookup:
            return component_lookup[name]

        id = len(component_lookup)
        component_lookup[name] = id
        return id

    def process(img_file, prefix=""):
        img = Image.open(img_file)
        w,h = img.size
        shortest_edge = min(w,h)

        scalar = 1
        out_file = img_file
        if opt.max_shortest_edge != -1:
            if shortest_edge > opt.max_shortest_edge:
                scalar = opt.max_shortest_edge / shortest_edge
                img = img.resize((round(w*scalar), round(h*scalar)), Image.Resampling.BICUBIC)
            out_file = str(dataset_dir / "img" / (prefix + str(img_file).split("/")[-1]))
            img.save(out_file)
            w,h = img.size
        
        return img, scalar, w, h, out_file

    def rescale(objs, scalar):
        for obj in objs:
            obj["bbox"] = [round(x*scalar) for x in obj["bbox"]]


    dataset = []

    if opt.dataset == "wacv":
        for i, pcb_folder in enumerate(dir.iterdir()):
            if not pcb_folder.is_dir():
                continue

            xml_count = 0
            xml_file = None
            jpg_count = 0
            jpg_file = None
            for file in pcb_folder.iterdir():
                name = str(file).lower()
                if name.endswith(".xml"):
                    xml_count += 1
                    xml_file = file
                elif name.endswith(".jpg"):
                    jpg_count += 1
                    jpg_file = file
            
            _, scalar, w, h, out_file = process(jpg_file)

            if xml_count != 1:
                raise Exception(f"{xml_count} xml files in {pcb_folder}")
            if jpg_count != 1:
                raise Exception(f"{jpg_count} jpg files in {pcb_folder}")

            xml_tree = ET.parse(xml_file)
            root = xml_tree.getroot()

            record = {}
            record["file_name"] = out_file
            record["image_id"] = i
            record["width"] = w
            record["height"] = h

            objs = []

            for a in root.findall("object"):
                cat = lookup_component(a.find("name").text)
                if not cat is None:
                    o = {
                        "bbox": [int(a.find("bndbox").find("xmin").text), int(a.find("bndbox").find("ymin").text), int(a.find("bndbox").find("xmax").text), int(a.find("bndbox").find("ymax").text)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        #"segmentation": [],
                        "category_id": cat
                    }
                    if (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]) < 10:
                        warnings.warn(f"Bounding box for {a.find('name').text} in {pcb_folder} less than 10 pixels of area")
                    objs.append(o)
            
            rescale(objs, scalar)
            record["annotations"] = objs
            dataset.append(record)

    elif opt.dataset == "fics":
        for i, jpg_file in enumerate((dir / "img").iterdir()):
            name = str(jpg_file).split("/")[-1].split(".")[0]
            csv_file = dir / "annotation" / (name + ".csv")
            _, scalar, w, h, out_file = process(jpg_file)

            record = {}
            record["file_name"] = out_file
            record["image_id"] = i
            record["width"] = w
            record["height"] = h

            objs = []

            csv = ""
            with open(csv_file) as f:
                csv = f.read()
            csv = csv.split("\n")[1:]

            for line in csv:
                lq = line.find("\"")
                rq = line.rfind("\"")
                bbox_str = line[lq+1:rq].replace("\"\"", "\"")
                line = line[:lq] + line[rq+2:] # skip the comma
                fields = line.split(",")

                if len(bbox_str) < 10:
                    continue

                bbox_json = json.loads(bbox_str)

                if bbox_json["name"] == "polygon":
                    bbox_json["x"] = int(np.min(bbox_json["all_points_x"]))
                    bbox_json["y"] = int(np.min(bbox_json["all_points_y"]))
                    bbox_json["width"] = int(np.max(bbox_json["all_points_x"]) - bbox_json["x"])
                    bbox_json["height"] = int(np.max(bbox_json["all_points_y"]) - bbox_json["y"])
                    bbox_json["name"] = "rect"

                if bbox_json["name"] != "rect":
                    warnings.warn("Found non-rect bbox")
                    continue

                cat = lookup_component(fields[1])

                if not cat is None:
                    o = {
                        "bbox": [bbox_json["x"], bbox_json["y"], bbox_json["width"], bbox_json["height"]],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": cat
                    }
                    if o["bbox"][2] * o["bbox"][3] < 10:
                        warnings.warn(f"Bounding box for {a.find('name').text} in {pcb_folder} less than 10 pixels of area")
                    objs.append(o)

            rescale(objs, scalar)
            record["annotations"] = objs
            dataset.append(record)

    elif opt.dataset == "dslr":
        component_lookup["ic"] = 1
        for i, pcb_folder in enumerate(dir.iterdir()):
            jpg_file = pcb_folder / "rec1.jpg"
            ann_file = pcb_folder / "rec1-annot.txt"

            if not jpg_file.exists() and not ann_file.exists():
                continue
            
            _, scalar, w, h, out_file = process(jpg_file, prefix=str(i))

            record = {}
            record["file_name"] = str(out_file)
            record["image_id"] = i
            record["width"] = w
            record["height"] = h

            objs = []

            ann = None
            with open(ann_file) as f:
                ann = f.read()

            ann = [a.split(" ") for a in ann.split("\n")]
            for a in ann:
                if len(a) >= 5:
                    theta = int(a[4].split(".")[0]) * np.pi / 180.0
                    c = np.cos(theta)
                    s = np.sin(theta)
                    w = int(a[2].split(".")[0])
                    h = int(a[3].split(".")[0])
                    rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
                    # x: left->right ; y: top->down
                    rotated_rect = [(s * yy + c * xx + int(a[0].split(".")[0]), c * yy - s * xx + int(a[1].split(".")[0])) for (xx, yy) in rect]
                    xs = [a for a,_ in rotated_rect]
                    ys = [b for _,b in rotated_rect]
                    minx = np.min(xs)
                    maxx = np.max(xs)
                    miny = np.min(ys)
                    maxy = np.max(ys)
                    objs.append({
                        "bbox": [minx, miny, maxx, maxy],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0
                    })

            rescale(objs, scalar)
            record["annotations"] = objs
            dataset.append(record)

    test_start_ind = int(len(dataset) * (1-opt.test_portion))

    train_ds = dataset[:test_start_ind]
    test_ds = dataset[test_start_ind:]

    train_name = opt.dataset + "_train"
    test_name = opt.dataset + "_test"

    classes_ordered = [k for k,v in sorted(component_lookup.items(), key=lambda item: item[1])]
    DatasetCatalog.register(train_name, lambda: train_ds)
    MetadataCatalog.get(train_name).set(thing_classes=list(classes_ordered))
    DatasetCatalog.register(test_name, lambda: test_ds)
    MetadataCatalog.get(test_name).set(thing_classes=list(classes_ordered))

    print("Classes:")
    print(classes_ordered)

    with open(dataset_dir / "train.json", "w") as f:
        json.dump(train_ds, f)
    with open(dataset_dir / "test.json", "w") as f:
        json.dump(test_ds, f)
    with open(dataset_dir / "classes.json", "w") as f:
        json.dump(classes_ordered, f)

    if opt.visualize != 0:
        visualize_ds(train_ds, train_name, num=opt.visualize, path=dataset_dir, prefix="train_vis")
        visualize_ds(test_ds, train_name, num=opt.visualize, path=dataset_dir, prefix="test_vis")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["fics", "wacv", "dslr"])
    parser.add_argument("dir", type=str)
    parser.add_argument("-v", "--visualize", type=int, default=0, help="Number of samples to save with annotation visualization, set -1 for the full dataset")
    parser.add_argument("-c", "--classes", type=str, nargs="*", default=["all"], help="Set \"all\" to include all classes")
    parser.add_argument("-uc", "--unknown_classes", type=str, nargs="*", default=[], help="Classes to include in \"unknown\" class")
    parser.add_argument("-mse", "--max_shortest_edge", type=int, default=1800, help="For resizing images, set to -1 to not resize. If not -1, will copy and rescale images.")
    parser.add_argument("-tp", "--test_portion", type=float, default=0.2)
    opt = parser.parse_args()

    format_dataset(opt)
