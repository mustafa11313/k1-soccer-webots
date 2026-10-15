# Third-party notices

k1-soccer-webots is licensed under the GNU AGPL v3.0 or later (see [LICENSE](LICENSE)).
It contains or uses the following third-party parts, which keep their own licenses.

## Booster Robotics: K1 robot model

- Source: https://github.com/BoosterRobotics/booster_assets
- License: BSD-3-Clause, Copyright (c) 2025, BoosterRobotics
- Used as: the git submodule `third_party/booster_assets` (unchanged; its LICENSE file is
  included there). `sim/protos/K1.proto` is generated from its K1 URDF with
  `sim/tools/make_webots_urdf.py` and the Webots urdf2webots converter, then changed
  (motor rotor inertia added to the joints, head camera). It loads the meshes from the
  submodule.

## Booster Robotics: K1 walking policy

- Source: https://github.com/BoosterRobotics/booster_deploy
- License: Apache License 2.0, Copyright 2025 Booster Robotics Technology Co., Ltd
  (full text: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt))
- Used as: `sim/controllers/k1_walk/k1_walk.pt`, the trained walking policy, unchanged.
  `sim/controllers/k1_walk/k1_walk.py` runs it in Webots; how it builds the policy's
  input and output follows booster_deploy.

## Ultralytics YOLO

- Source: https://github.com/ultralytics/ultralytics
- License: AGPL-3.0
- Used as: a Python dependency of `src/k1_perception` (not included here). The models in
  `src/models/` are fine-tuned from the Ultralytics YOLO11n weights on images made with this
  project's collector, and are distributed under the AGPL-3.0.

## Cyberbotics Webots

- Source: https://github.com/cyberbotics/webots
- License: Apache License 2.0
- Used as: the simulator. The worlds load Webots PROTOs (RoboCup soccer field, background,
  lights, arena) and the soccer ball texture from the Webots R2025a release by URL; they are
  not copied into this repository.

## Python and ROS packages

Installed separately, not included: ROS 2 Humble (Apache-2.0 and others), PyTorch
(BSD-3-Clause), NumPy (BSD-3-Clause), OpenCV (Apache-2.0).
