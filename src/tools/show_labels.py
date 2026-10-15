"""Draw the YOLO labels on collected images, to check them by eye.

Usage: python3 show_labels.py IMAGE_DIR [PATTERN] [COUNT]
  e.g. python3 show_labels.py ~/k1_ws/data/ball/train/images "train_close_*" 12
Writes one contact sheet to /tmp/labels_check.jpg (green box = labelled ball,
"no ball" = empty label file).
"""
import glob
import os
import random
import sys

import cv2
import numpy as np

image_dir = os.path.expanduser(sys.argv[1])
pattern = sys.argv[2] if len(sys.argv) > 2 else "*"
count = int(sys.argv[3]) if len(sys.argv) > 3 else 12
images = sorted(glob.glob(os.path.join(image_dir, pattern + ".jpg")))
random.seed(0)
images = random.sample(images, min(count, len(images)))

tiles = []
for path in images:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    label = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    parts = open(label).read().split()
    if parts:
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                      (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 0), 2)
    else:
        cv2.putText(img, "no ball", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, os.path.basename(path), (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    tiles.append(cv2.resize(img, (320, int(320 * h / w))))

while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite("/tmp/labels_check.jpg", np.vstack(rows))
print(f"{len(images)} images drawn to /tmp/labels_check.jpg")
