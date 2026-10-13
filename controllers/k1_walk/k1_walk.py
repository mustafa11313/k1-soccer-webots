"""K1 walking controller for Webots: runs Booster's k1_walk policy with PD torque control.
Keyboard (click the 3D view first, hold the key): Up/Down = forward/back,
Left/Right = turn, A/D = sidestep."""
import os

import numpy as np
import torch
from controller import Keyboard, Supervisor

JOINTS = [  # real K1 joint order
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
POLICY_JOINTS = [  # order the network was trained with (no head)
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
# Values from booster_deploy (K1WalkControllerCfg)
DEFAULT_POS = np.array([0, 0, 0.2, -1.25, 0, -0.5, 0.2, 1.25, 0, 0.5,
                        -0.15, 0, 0, 0.3, -0.15, 0, -0.15, 0, 0, 0.3, -0.15, 0], dtype=np.float32)
DEFAULT_POS[1] = 0.7  # head pitch: look down so the ball stays in view up close
KP = np.array([4, 4, 20, 20, 20, 20, 20, 20, 20, 20,
               100, 100, 100, 100, 65, 65, 100, 100, 100, 100, 65, 65], dtype=np.float32)
KD = np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1], dtype=np.float32)
TORQUE_LIMIT = np.array([6, 6, 14, 14, 14, 14, 14, 14, 14, 14,
                         30, 20, 15, 35, 24, 15, 30, 20, 15, 35, 24, 15], dtype=np.float32)
POLICY_DT = 0.02  # network runs at 50 Hz
ACTION_SCALE, ACTION_FILTER, DOF_VEL_SCALE, HISTORY = 0.25, 0.8, 0.1, 10
ARM_FIX_JOINT, ARM_FIX_OFFSET = "right_elbow_pitch_joint", 0.2
# In Webots the K1 only stays reliably on its feet within these limits
MAX_VX, MIN_VX, MAX_VY, MAX_VYAW = 0.5, -0.2, 0.15, 0.5
COMMAND_RAMP = 0.5  # max command change per second

robot = Supervisor()
dt_ms = int(robot.getBasicTimeStep())
decimation = round(POLICY_DT * 1000 / dt_ms)
body = robot.getSelf()
motors = [robot.getDevice(name) for name in JOINTS]
sensors = [robot.getDevice(name + "_sensor") for name in JOINTS]
for s in sensors:
    s.enable(dt_ms)
keyboard = Keyboard()
keyboard.enable(dt_ms)
camera = robot.getDevice("head_camera")
camera.enable(32)  # new image every 32 ms (~30 fps)
try:
    from ros_bridge import RosBridge
    bridge = RosBridge(robot, camera)
except ImportError:
    print("ROS 2 not sourced, keyboard control only")
    bridge = None

policy_idx = np.array([JOINTS.index(n) for n in POLICY_JOINTS])
arm_fix_idx = JOINTS.index(ARM_FIX_JOINT)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k1_walk.pt")
policy = torch.jit.load(model_path, map_location="cpu").actor.eval()

command = np.zeros(3, dtype=np.float32)  # vx, vy, yaw rate
last_action = np.zeros(len(POLICY_JOINTS), dtype=np.float32)
obs_history = None
target = DEFAULT_POS.copy()
last_q = None


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


def read_keyboard():
    vx = vy = vyaw = 0.0
    pressed = False
    key = keyboard.getKey()
    while key != -1:
        if key == Keyboard.UP:
            vx = MAX_VX
        elif key == Keyboard.DOWN:
            vx = MIN_VX
        elif key == Keyboard.LEFT:
            vyaw = MAX_VYAW
        elif key == Keyboard.RIGHT:
            vyaw = -MAX_VYAW
        elif key == ord("A"):
            vy = MAX_VY
        elif key == ord("D"):
            vy = -MAX_VY
        pressed = True
        key = keyboard.getKey()
    return (vx, vy, vyaw) if pressed else None  # None = no key held


def set_command(vx, vy, vyaw):
    """Move the velocity command towards (vx, vy, vyaw), limited and ramped."""
    wanted = np.array([np.clip(vx, MIN_VX, MAX_VX), np.clip(vy, -MAX_VY, MAX_VY),
                       np.clip(vyaw, -MAX_VYAW, MAX_VYAW)], dtype=np.float32)
    max_change = COMMAND_RAMP * POLICY_DT
    command[:] = command + np.clip(wanted - command, -max_change, max_change)


def policy_step(q, dq):
    global obs_history, target
    ang_vel, gravity = base_state()
    obs = np.concatenate([ang_vel, gravity, command,
                          (q - DEFAULT_POS)[policy_idx],
                          dq[policy_idx] * DOF_VEL_SCALE,
                          last_action]).astype(np.float32)
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


step = 0
while robot.step(dt_ms) != -1:
    q, dq = read_joints()
    if step % decimation == 0:
        wanted = read_keyboard()
        if bridge is not None:
            bridge.spin()
            if wanted is None:
                wanted = bridge.command()
        set_command(*(wanted or (0.0, 0.0, 0.0)))
        gravity = policy_step(q, dq)
        if gravity[2] > -0.5:
            print("K1 fell over, stopping the policy")
            break
    torque = np.clip(KP * (target - q) - KD * dq, -TORQUE_LIMIT, TORQUE_LIMIT)
    for m, t in zip(motors, torque):
        m.setTorque(float(t))
    if bridge is not None and step % (32 // dt_ms) == 0:
        bridge.publish_camera()
    step += 1
