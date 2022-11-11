from PIL import Image
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("dataset_path", type=str, help="Path to image folder")
parser.add_argument("detections_path", type=str)
opt = parser.parse_args()

folder = [f for f in Path(opt.dataset_path).glob("*.jpg")] + [f for f in Path(opt.dataset_path).glob("*.png")] + [f for f in Path(opt.dataset_path).glob("*.jpeg")]

detections = None
with open(opt.detections_path) as f:
    detections = json.loads(f.read())

print(folder)

img1 = Image.open(folder[0]).convert("RGB")
det1 = detections[folder[0].name]
w, h = img1.size
det1["image_width"] = w
det1["image_height"] = h
img2 = Image.open(folder[1]).convert("RGB")
det2 = detections[folder[1].name]
w, h = img2.size
det2["image_width"] = w
det2["image_height"] = h
compare(det1, det2, img1, img2, debug=True, whole=True, variant="single")
