# k1-soccer-webots

A Booster K1 humanoid that plays soccer by itself in the Webots simulator. It finds the
ball with its head camera, walks to it, dribbles it towards the goal and, when it is close
enough, stops, lines up and shoots with a scripted kick.

## Features

- **Walking** with Booster's learned K1 walking policy, run inside the Webots controller.
- **Ball detection** with a YOLO11n model trained only on automatically labelled Webots
  images (no hand labelling). The training images come from a collector that teleports the
  robot and the ball around the field, the goals and the robot's own body.
- **Ball tracking** in the odometry frame, so the ball is not lost when the detector misses
  frames or the ball is too close to the feet to be seen. The head follows the ball.
- **Striker behaviour**: search, walk behind the ball on the ball-goal line, dribble without
  stopping, walk around the ball when on the wrong side, and score.
- **Kick**: a programmed right-foot kick on one leg with closed-loop balance (IMU and leg
  kinematics). Before kicking the striker stands still, turns to aim and nudges itself
  until the ball is in front of the right foot. The ball travels 3 to 4 m.
- **Goal detection** that counts a goal only when the ball is seen inside the goal for 1 s.

## Results

All numbers are from Webots. The game tests use camera-limited vision (the robot only knows
where the ball is when its camera sees it).

| Test | Result |
|---|---|
| 16 games from different start positions | 15 goals, 8 of them with a kick, 0 falls |
| Ball detector v4, all 950 validation images | 549/564 balls found (97.3%), 12/386 false alarms |
| Ball detector v4, 150 images in front of a goal | 79/80 balls found, 0/70 false alarms on the net (v3: 8/70) |
| Detector speed on a laptop CPU (Intel i7-8850H) | about 30 ms per image |

## How it works

~~~mermaid
flowchart LR
  subgraph Webots
    W[k1_walk controller<br/>walking policy, kick, head]
  end
  W -- "/camera/image_raw<br/>/camera/camera_info<br/>/tf, /clock" --> D[k1_perception<br/>ball_detector]
  D -- "/ball/position" --> S[k1_behavior<br/>striker]
  S -- "/cmd_vel" --> W
  S -- "/head/command" --> W
  S -- "/kick" --> W
~~~

The Webots controller (`sim/controllers/k1_walk`) runs the walking policy at 50 Hz and
talks to ROS 2 through a small bridge (`ros_bridge.py`). The detector turns each camera
image into a ball position: the ray through the ball's centre meets the floor plane at the
ball's height. The striker (`striker_logic.py`) is a plain state machine without ROS code,
so it can be tested on its own:

`SEARCH → BEHIND / AROUND → DRIBBLE → LINE_UP → SETTLE → NUDGE → KICK → SCORED`

## Repository layout

~~~
k1_ws/                       this repository, also the ROS 2 (colcon) workspace
├── src/                     ROS 2 packages
│   ├── k1_perception/       ball_detector node (YOLO)
│   ├── k1_behavior/         striker and ball_chaser nodes, tracking, state machine
│   ├── models/              trained ball detectors (ball_yolo11n_v4.pt is the default)
│   └── tools/               compare_detectors.py, show_labels.py
├── sim/                     Webots project (has a COLCON_IGNORE)
│   ├── controllers/
│   │   ├── k1_walk/         walking policy, kick, one-leg balance, ROS bridge
│   │   ├── collect/         training image collector with automatic labels
│   │   └── drone_cam/       camera drone and referee for the field world
│   ├── protos/K1.proto      the robot, generated from Booster's URDF
│   ├── worlds/              k1_field.wbt (RoboCup kid-size field), test and collection worlds
│   └── tools/               URDF to Webots conversion helpers
└── third_party/
    └── booster_assets/      git submodule: Booster's K1 model and meshes
~~~

## Requirements

- Ubuntu 22.04
- [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html) with `cv_bridge` and `tf2_ros`
- [Webots R2025a](https://cyberbotics.com/)
- Python 3.10 with `torch`, `numpy`, `opencv-python` and `ultralytics`

A GPU is not needed; everything runs on the CPU.

## Installation

~~~bash
git clone --recurse-submodules https://github.com/mustafa11313/k1-soccer-webots.git ~/k1_ws
pip install ultralytics torch numpy opencv-python
source /opt/ros/humble/setup.bash
cd ~/k1_ws && colcon build
~~~

## Running a game

Each command in its own terminal. Start Webots from a terminal where ROS 2 is sourced, so
the robot controller can reach ROS.

~~~bash
# 1. the simulator
source /opt/ros/humble/setup.bash && source ~/k1_ws/install/setup.bash
webots ~/k1_ws/sim/worlds/k1_field.wbt

# 2. the ball detector
source ~/k1_ws/install/setup.bash
ros2 run k1_perception ball_detector --ros-args -p use_sim_time:=true

# 3. the striker (the goal posts of k1_field.wbt)
source ~/k1_ws/install/setup.bash
ros2 run k1_behavior striker --ros-args -p use_sim_time:=true \
  -p left_post:="[4.5, 1.35]" -p right_post:="[4.5, -1.35]"
~~~

The striker logs every state change and why it did or did not kick. It prints `GOAL!` when
the ball is in the goal.

In the Webots 3D view you can also drive the robot by hand: the arrow keys walk, `P` puts
the ball in front of the right foot and `K` kicks.

## Training the ball detector

~~~bash
# collect automatically labelled images (each world closes itself when done)
cd ~/k1_ws/sim/worlds
webots collect_field.wbt          # also collect.wbt, collect_close.wbt, collect_goal.wbt
webots collect_goal_val.wbt       # validation images

# fine-tune from the current model
cd ~/k1_ws
yolo detect train model=src/models/ball_yolo11n_v3.pt data=data/ball/ball_v2.yaml \
  epochs=10 imgsz=480 device=cpu project=runs name=ball_v4

# compare two models on a validation folder
python3 src/tools/compare_detectors.py data/ball/val/images \
  src/models/ball_yolo11n_v3.pt:0 src/models/ball_yolo11n_v4.pt:0
~~~

Images go to `~/k1_ws/data/ball/{train,val}`. The dataset and the training runs are not
part of the repository, but the collector recreates them.

## License

Copyright (c) 2026 Mustafa Almohamad.

This project is licensed under the **GNU Affero General Public License v3.0 or later**
(see [LICENSE](LICENSE)). The ball detector uses Ultralytics YOLO, which is AGPL-3.0, and
the models in `src/models` are fine-tuned from Ultralytics YOLO11n weights, so the
project as a whole follows the same license.

Third-party parts keep their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

- [Booster Robotics](https://github.com/BoosterRobotics) for the K1 robot model
  ([booster_assets](https://github.com/BoosterRobotics/booster_assets), BSD-3-Clause) and
  the K1 walking policy ([booster_deploy](https://github.com/BoosterRobotics/booster_deploy),
  Apache-2.0).
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO11 (AGPL-3.0).
- [Cyberbotics](https://cyberbotics.com/) for Webots and its soccer field and ball assets (Apache-2.0).

This is an independent project. It is not affiliated with or endorsed by Booster Robotics,
Ultralytics or Cyberbotics. "Booster" and "K1" are names of Booster Robotics.
