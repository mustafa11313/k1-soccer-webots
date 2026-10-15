"""Collects training images for the ball detector.

The K1 stands (walking policy with a zero command) while the Supervisor moves the
ball, the head, the lights, the floor colour and some distractor objects at random.
Every camera image is saved with a YOLO label for the ball, computed from the ball's
true position, so nothing has to be labelled by hand.

controllerArgs: split name, number of images, random seed, mode. Output goes to
~/k1_ws/data/ball/<split>/images and labels.
Mode "normal" (default) puts the ball 0.35-5 m away. Mode "close" looks steeply down
and to the side with the arms in random poses, with the ball at the feet or (35%)
no ball at all, so the detector learns close balls and that its own body is no ball.
Mode "goal" (RoboCup field world) moves the robot in front of a goal every few images,
looking at the goal net, with the ball in the goal, in front of it or (35%) not in view,
so the detector learns that the black and white net is no ball.
"""
import math
import os
import random
import sys

import numpy as np
import torch
from controller import Supervisor

# Joint order of the real K1 (same as the URDF and booster_deploy K1_CFG)
JOINTS = [
    "aahead_yaw_joint", "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
# Joint order the policy was trained with (no head joints)
POLICY_JOINTS = [
    "aaleft_shoulder_pitch_joint", "aaright_shoulder_pitch_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
    "left_knee_pitch_joint", "right_knee_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]
# Values from booster_deploy tasks/locomotion/robots/k1 (K1WalkControllerCfg)
DEFAULT_POS = np.array([
    0, 0,
    0.2, -1.25, 0, -0.5,
    0.2, 1.25, 0, 0.5,
    -0.15, 0, 0, 0.3, -0.15, 0,
    -0.15, 0, 0, 0.3, -0.15, 0,
], dtype=np.float32)
KP = np.array([
    4, 4,
    20, 20, 20, 20,
    20, 20, 20, 20,
    100, 100, 100, 100, 65, 65,
    100, 100, 100, 100, 65, 65,
], dtype=np.float32)
KD = np.array([
    1, 1,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2, 1, 1,
    2, 2, 2, 2, 1, 1,
], dtype=np.float32)
TORQUE_LIMIT = np.array([
    6, 6,
    14, 14, 14, 14,
    14, 14, 14, 14,
    30, 20, 15, 35, 24, 15,
    30, 20, 15, 35, 24, 15,
], dtype=np.float32)
POLICY_DT = 0.02          # policy runs at 50 Hz
ACTION_SCALE = 0.25
ACTION_FILTER = 0.8
DOF_VEL_SCALE = 0.1
HISTORY = 10
ARM_FIX_JOINT = "right_elbow_pitch_joint"
ARM_FIX_OFFSET = 0.2

robot = Supervisor()
dt_ms = int(robot.getBasicTimeStep())
decimation = round(POLICY_DT * 1000 / dt_ms)
body = robot.getSelf()
motors = [robot.getDevice(name) for name in JOINTS]
sensors = [robot.getDevice(name + "_sensor") for name in JOINTS]
for s in sensors:
    s.enable(dt_ms)

policy_idx = np.array([JOINTS.index(n) for n in POLICY_JOINTS])
arm_fix_idx = JOINTS.index(ARM_FIX_JOINT)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k1_walk", "k1_walk.pt")
policy = torch.jit.load(model_path, map_location="cpu").actor.eval()

command = np.zeros(3, dtype=np.float32)  # vx, vy, yaw rate
last_action = np.zeros(len(POLICY_JOINTS), dtype=np.float32)
obs_history = None
target = DEFAULT_POS.copy()
last_q = None
arm_offset = np.zeros_like(DEFAULT_POS)  # extra arm pose in "close" mode, on top of the policy


def read_joints():
    global last_q
    q = np.array([s.getValue() for s in sensors], dtype=np.float32)
    dq = np.zeros_like(q) if last_q is None else (q - last_q) / (dt_ms / 1000)
    last_q = q
    return q, dq


def base_state():
    """Angular velocity and gravity direction in the trunk frame."""
    rot = np.array(body.getOrientation(), dtype=np.float32).reshape(3, 3)
    ang_vel_w = np.array(body.getVelocity()[3:], dtype=np.float32)
    return rot.T @ ang_vel_w, rot.T @ np.array([0, 0, -1], dtype=np.float32)


def policy_step(q, dq):
    global obs_history, target
    ang_vel, gravity = base_state()
    obs = np.concatenate([
        ang_vel,
        gravity,
        command,
        (q - DEFAULT_POS)[policy_idx],
        dq[policy_idx] * DOF_VEL_SCALE,
        last_action,
    ]).astype(np.float32)
    if obs_history is None:
        obs_history = np.tile(obs, (HISTORY, 1))
    else:
        obs_history = np.roll(obs_history, -1, axis=0)
        obs_history[-1] = obs
    with torch.no_grad():
        action = policy(torch.from_numpy(obs_history.flatten())).numpy()
    last_action[:] = action
    full_action = np.zeros_like(DEFAULT_POS)
    full_action[policy_idx] = action
    full_action[arm_fix_idx] -= ARM_FIX_OFFSET
    new_target = DEFAULT_POS + full_action * ACTION_SCALE
    target = target + ACTION_FILTER * (new_target - target)
    return gravity


SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
MODE = sys.argv[4] if len(sys.argv) > 4 else "normal"
PREFIX = SPLIT if MODE == "normal" else f"{SPLIT}_{MODE}"  # new files never overwrite old ones
OUT = os.path.join(os.path.expanduser("~/k1_ws/data/ball"), SPLIT)
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)

BALL_RADIUS = 0.11
VISIBLE_MIN = 0.1  # label the ball when at least this part of it is in the image
CAMERA_PERIOD = 32  # ms
camera = robot.getDevice("head_camera")
camera.enable(CAMERA_PERIOD)
W, H = camera.getWidth(), camera.getHeight()
F = (W / 2) / math.tan(camera.getFov() / 2)
# getFromDevice wants the raw device tag in the R2025a Python API
camera_node = robot.getFromDevice(camera._tag)
ball = robot.getFromDef("BALL")
floor_patch = robot.getFromDef("FLOOR_PATCH")
floor_color = robot.getFromDef("FLOOR_PATCH_APPEARANCE").getField("baseColor")
light = robot.getFromDef("LIGHT").getField("luminosity")
sky = robot.getFromDef("SKY").getField("luminosity")
distractors = [robot.getFromDef(f"DISTRACTOR_{i}") for i in range(3)]


def random_floor_point(min_d, max_d, yaw, spread):
    """A point on the floor around the robot, roughly in the direction yaw."""
    p = np.array(body.getPosition())
    rot = np.array(body.getOrientation()).reshape(3, 3)
    heading = math.atan2(rot[1, 0], rot[0, 0]) + yaw + random.uniform(-spread, spread)
    d = min_d + (max_d - min_d) * random.random() ** 2  # more samples close to the robot
    return [p[0] + d * math.cos(heading), p[1] + d * math.sin(heading)]


def randomize_close():
    head_yaw = random.uniform(-1.1, 1.1)
    DEFAULT_POS[0] = head_yaw
    DEFAULT_POS[1] = random.uniform(0.5, 0.95)
    arm_offset[2:10] = [random.uniform(-0.3, 0.3) for _ in range(8)]
    if random.random() < 0.35:  # only the robot's own arms and feet in view
        x, y = random_floor_point(1.0, 4.0, head_yaw + math.pi, 0.5)
    else:
        x, y = random_floor_point(0.25, 0.8, head_yaw, 0.6)
    return x, y, head_yaw


def randomize_normal():
    head_yaw = random.uniform(-0.8, 0.8)
    DEFAULT_POS[0] = head_yaw
    DEFAULT_POS[1] = random.uniform(0.2, 0.79)
    if random.random() < 0.15:  # no ball in view: negative example
        x, y = random_floor_point(1.0, 4.0, head_yaw + math.pi, 0.5)
    else:
        x, y = random_floor_point(0.35, 5.0, head_yaw, 0.9)
    return x, y, head_yaw


GOAL_X = 4.5        # m, goal lines of the kid-size field
PER_PLACE = 6       # images per robot position in "goal" mode


def place_robot():
    """Goal mode: put the robot 0.5-3 m in front of a goal, roughly facing it."""
    global obs_history, last_q, target
    side = random.choice((1, -1))
    x = side * (GOAL_X - random.uniform(0.5, 3.0))
    y = random.uniform(-2.0, 2.0)
    yaw = math.atan2(-y, side * GOAL_X - x) + random.uniform(-0.6, 0.6)
    body.getField("translation").setSFVec3f([x, y, 0.56])
    body.getField("rotation").setSFRotation([0, 0, 1, yaw])
    body.resetPhysics()
    obs_history, last_q, target = None, None, DEFAULT_POS.copy()
    last_action[:] = 0
    return side


def randomize_goal():
    head_yaw = random.uniform(-0.8, 0.8)
    DEFAULT_POS[0] = head_yaw
    DEFAULT_POS[1] = random.uniform(0.2, 0.79)
    r = random.random()
    if r < 0.35:  # only the goal and the field in view: negative example
        x, y = random_floor_point(1.5, 4.0, head_yaw + math.pi, 0.5)
    elif r < 0.6:  # ball in the goal, in front of the net
        x, y = goal_side * (GOAL_X + random.uniform(0.15, 0.5)), random.uniform(-1.2, 1.2)
    else:
        x, y = random_floor_point(0.35, 3.5, head_yaw, 0.9)
    return x, y, head_yaw


def randomize():
    x, y, head_yaw = {"close": randomize_close, "goal": randomize_goal}.get(MODE, randomize_normal)()
    ball.getField("translation").setSFVec3f([x, y, BALL_RADIUS])
    ball.getField("rotation").setSFRotation([random.uniform(-1, 1), random.uniform(-1, 1), 1, random.uniform(0, 6.28)])
    ball.resetPhysics()
    if random.random() < 0.5:  # plain coloured floor instead of the parquet
        floor_patch.getField("translation").setSFVec3f([0, 0, 0.001])
        floor_color.setSFColor([random.uniform(0.1, 0.9) for _ in range(3)])
    else:
        floor_patch.getField("translation").setSFVec3f([0, 0, -5])
    light.setSFFloat(random.uniform(0.4, 2.0))
    sky.setSFFloat(random.uniform(0.4, 2.0))
    for d in distractors:
        dx, dy = random_floor_point(0.6, 4.0, head_yaw, 1.0)
        d.getField("translation").setSFVec3f([dx, dy, d.getField("translation").getSFVec3f()[2]])


def ball_box():
    """The ball's bounding box in pixels (x1, y1, x2, y2), or None if not in view."""
    rot = np.array(camera_node.getOrientation()).reshape(3, 3)
    p = rot.T @ (np.array(ball.getPosition()) - np.array(camera_node.getPosition()))
    if p[0] < BALL_RADIUS + 0.05:
        return None
    u, v = W / 2 - F * p[1] / p[0], H / 2 - F * p[2] / p[0]
    r = F * BALL_RADIUS / math.sqrt(p @ p - BALL_RADIUS ** 2)
    box = (max(u - r, 0), max(v - r, 0), min(u + r, W), min(v + r, H))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    visible = (box[2] - box[0]) * (box[3] - box[1]) / (2 * r) ** 2
    return box if visible > VISIBLE_MIN else None


saved = 0
step = 0
next_randomize = 1.0  # let the robot settle first
goal_side = place_robot() if MODE == "goal" else 1
placed_at = 0
while robot.step(dt_ms) != -1 and saved < COUNT:
    q, dq = read_joints()
    t = robot.getTime()
    if step % decimation == 0:
        gravity = policy_step(q, dq)  # zero command: the robot stands
        if gravity[2] > -0.5:
            if MODE != "goal":
                print("K1 fell over, stopping")
                break
            goal_side, placed_at = place_robot(), saved  # stand it up somewhere else
            next_randomize, capture_at = t + 1.5, 1e9
    torque = np.clip(KP * (target + arm_offset - q) - KD * dq, -TORQUE_LIMIT, TORQUE_LIMIT)
    for m, tq in zip(motors, torque):
        m.setTorque(float(tq))
    step += 1
    if MODE == "goal" and saved - placed_at >= PER_PLACE and t >= next_randomize:
        goal_side, placed_at = place_robot(), saved
        next_randomize, capture_at = t + 1.5, 1e9  # let it settle on its feet
    if t >= next_randomize:
        randomize()
        next_randomize = t + 0.3
        capture_at = t + 0.2  # give the head time to move and the camera to update
    if next_randomize > 1.0 and t >= capture_at and round(t * 1000) % CAMERA_PERIOD == 0:
        name = f"{PREFIX}_{saved:05d}"
        camera.saveImage(os.path.join(OUT, "images", name + ".jpg"), 90)
        box = ball_box()
        with open(os.path.join(OUT, "labels", name + ".txt"), "w") as f:
            if box:
                x1, y1, x2, y2 = box
                f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        saved += 1
        capture_at = 1e9
        if saved % 100 == 0:
            print(f"{saved}/{COUNT} images saved", flush=True)

print(f"Done: {saved} images in {OUT}", flush=True)
robot.simulationQuit(0)
