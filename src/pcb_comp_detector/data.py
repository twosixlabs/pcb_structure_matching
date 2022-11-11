from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
from detectron2.utils.visualizer import Visualizer
import cv2, random, json


def register_dataset(dir, vis=0):
    train_ds = None
    test_ds = None
    classes = None

    dir = Path(dir)

    with open(dir / "train.json") as f:
        train_ds = json.loads(f.read())
    with open(dir / "test.json") as f:
        test_ds = json.loads(f.read())
    with open(dir / "classes.json") as f:
        classes = json.loads(f.read())

    train_name = str(dir) + "/train"
    test_name = str(dir) + "/test"

    DatasetCatalog.register(train_name, lambda: train_ds)
    MetadataCatalog.get(train_name).set(thing_classes=list(classes))
    DatasetCatalog.register(test_name, lambda: test_ds)
    MetadataCatalog.get(test_name).set(thing_classes=list(classes))

    if vis != 0:
        visualize_ds(train_ds, train_name, num=vis)

    return train_ds, test_ds, classes, train_name, test_name

def visualize_ds(ds, name, num=5, path="", prefix="vis", only_with_classes=[]):
    path = Path(path)
    if num == -1:
        num = len(ds)
    for i, d in enumerate(random.sample(ds, num)):
        if len(only_with_classes) > 0:
            if not True in [a["category_id"] in only_with_classes for a in d["annotations"]]:
                continue

        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1)

        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite(str(path / f"{prefix}{i}.jpg"), out.get_image()[:, :, ::-1])
