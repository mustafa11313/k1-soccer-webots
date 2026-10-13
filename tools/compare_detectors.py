"""Compare ball detectors on the labelled Webots test images.

Usage: python3 compare_detectors.py IMAGE_DIR MODEL:CLASS [MODEL:CLASS ...]
For each image the most confident box of the given class counts as the detection.
A detection is correct if it overlaps the true box with IoU >= 0.5.
"""
import glob
import os
import sys
import time

from ultralytics import YOLO

CONFIDENCE = 0.25


def load_label(image_path, width, height):
    label = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    parts = open(label).read().split()
    if not parts:
        return None
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(images, model_path, class_id):
    model = YOLO(model_path)
    model(images[0], device="cpu", verbose=False)  # warm-up
    stats = {"balls": 0, "found": 0, "empty": 0, "false_alarms": 0, "centre_error": 0.0, "ms": 0.0}
    by_size = {"small (<20 px)": [0, 0], "medium (20-60 px)": [0, 0], "large (>60 px)": [0, 0]}
    for path in images:
        start = time.time()
        result = model(path, classes=[class_id], conf=CONFIDENCE, device="cpu", verbose=False)[0]
        stats["ms"] += (time.time() - start) * 1000
        height, width = result.orig_shape
        truth = load_label(path, width, height)
        box = None
        if len(result.boxes):
            box = result.boxes.xyxy[int(result.boxes.conf.argmax())].tolist()
        if truth is None:
            stats["empty"] += 1
            stats["false_alarms"] += box is not None
            continue
        size = truth[2] - truth[0]
        bucket = by_size["small (<20 px)" if size < 20 else "medium (20-60 px)" if size < 60 else "large (>60 px)"]
        stats["balls"] += 1
        bucket[0] += 1
        if box is not None and iou(box, truth) >= 0.5:
            stats["found"] += 1
            bucket[1] += 1
            stats["centre_error"] += (((box[0] + box[2] - truth[0] - truth[2]) / 2) ** 2
                                      + ((box[1] + box[3] - truth[1] - truth[3]) / 2) ** 2) ** 0.5
    return stats, by_size


def main():
    images = sorted(glob.glob(os.path.join(os.path.expanduser(sys.argv[1]), "*.jpg")))
    print(f"{len(images)} test images\n")
    for spec in sys.argv[2:]:
        model_path, class_id = spec.rsplit(":", 1)
        s, by_size = evaluate(images, model_path, int(class_id))
        print(f"== {model_path}")
        print(f"  ball found (IoU>=0.5): {s['found']}/{s['balls']} = {100 * s['found'] / max(s['balls'], 1):.1f}%")
        for name, (total, found) in by_size.items():
            print(f"    {name:18s} {found}/{total} = {100 * found / max(total, 1):.1f}%")
        print(f"  false alarms on images without a ball: {s['false_alarms']}/{s['empty']}")
        print(f"  mean centre error of found balls: {s['centre_error'] / max(s['found'], 1):.1f} px")
        print(f"  time per image: {s['ms'] / len(images):.0f} ms\n")


if __name__ == "__main__":
    main()
