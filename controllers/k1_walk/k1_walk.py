"""K1 walking controller for Webots.

Runs Booster's trained K1 walking policy (k1_walk.pt from booster_deploy)
and turns its output into joint torques with a PD controller, the same way
Booster's MuJoCo deployment does.

Velocity command: keyboard in the 3D view (click it first)
  Up/Down = forward/back, Left/Right = turn, A/D = sidestep (hold the key)
or, when Webots is started from a terminal with ROS 2 sourced, /cmd_vel
(see ros_bridge.py). The keyboard wins while a key is held.
"""
import os

import numpy as np
import torch
from controller import Keyboard, Supervisor

from one_leg import LEG_GAIN, LEG_TORQUE_LIMIT, OneLegStand

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
DEFAULT_POS[1] = 0.7  # head pitch: look down so the ball stays in view up close
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
# Speed limits and how fast commands may change. Booster's policy handles more in
# MuJoCo, but in Webots it only stayed reliably on its feet within these limits.
MAX_VX, MIN_VX, MAX_VY, MAX_VYAW = 0.5, -0.2, 0.15, 0.5
COMMAND_RAMP = 0.5  # max change per second
HEAD_LIMITS = ((-1.0, 1.0), (0.0, 0.79))  # yaw, pitch (rad) from the URDF joint limits
HEAD_STEP = 0.06    # rad per policy step, how fast the head may move

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
    """Return (vx, vy, vyaw) from the held arrow keys, or None if no key is held."""
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
        elif key == ord("B"):
            if not balance_request[0]:
                print("B pressed: waiting until the robot stands calmly")
            balance_request[0] = True
        pressed = True
        key = keyboard.getKey()
    return (vx, vy, vyaw) if pressed else None


def set_command(vx, vy, vyaw):
    """Move the velocity command towards (vx, vy, vyaw), limited and ramped."""
    wanted = np.array([np.clip(vx, MIN_VX, MAX_VX), np.clip(vy, -MAX_VY, MAX_VY),
                       np.clip(vyaw, -MAX_VYAW, MAX_VYAW)], dtype=np.float32)
    max_change = COMMAND_RAMP * POLICY_DT
    command[:] = command + np.clip(wanted - command, -max_change, max_change)


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


step = 0
leaning = False
balance_request = [False]  # B pressed: stand on the left leg (one_leg.py)
balance = None
still_time = 0.0           # s the robot has been standing still
KP_BALANCE = KP.copy()
KP_BALANCE[10:] *= LEG_GAIN
TORQUE_LIMIT_BALANCE = TORQUE_LIMIT.copy()
TORQUE_LIMIT_BALANCE[10:] = np.tile(np.array(LEG_TORQUE_LIMIT, dtype=np.float32), 2)
while robot.step(dt_ms) != -1:
    q, dq = read_joints()
    if balance is not None:
        ang_vel, gravity = base_state()
        bt = robot.getTime() - balance_start
        target = balance.target(bt, q, gravity, ang_vel)
        if gravity[2] > -0.5:
            print("K1 fell over, stopping the policy")
            break
        if balance.done(bt):
            print("One-leg stand done, walking policy takes over")
            balance = None
            obs_history = None
            last_action[:] = 0
            still_time = 0.0
    elif step % decimation == 0:
        wanted = read_keyboard()
        if bridge is not None:
            bridge.spin()
            if wanted is None:
                wanted = bridge.command()
            head = bridge.head()
            if head is not None:
                # the head is not part of the walking policy: just move its targets
                for i, (angle, (low, high)) in enumerate(zip(head, HEAD_LIMITS)):
                    DEFAULT_POS[i] += float(np.clip(np.clip(angle, low, high) - DEFAULT_POS[i], -HEAD_STEP, HEAD_STEP))
        # Lean guard: on long walks the policy speeds up and tips forward. Stand still
        # until the trunk is upright again.
        upright = -base_state()[1][2]  # 1.0 = straight up
        if upright < 0.95:
            leaning = True
        elif upright > 0.985:
            leaning = False
        set_command(*((0.0, 0.0, 0.0) if leaning else (wanted or (0.0, 0.0, 0.0))))
        # calm = not walking and the trunk upright and not rocking (after a fast walk it rocks for a while)
        av, gv = base_state()
        calm = np.abs(command).max() < 0.01 and not leaning and np.abs(av[:2]).max() < 0.3 and np.abs(gv[:2]).max() < 0.1
        if balance_request[0] and not calm and step % (1000 // dt_ms) == 0:
            print("waiting: command %s  rocking %.2f (needs < 0.3)  turning %.2f  tilt %.2f (needs < 0.1)" % (np.round(command, 2), np.abs(av[:2]).max(), av[2], np.abs(gv[:2]).max()))
        still_time = still_time + POLICY_DT if calm else 0.0
        if balance_request[0] and wanted is not None and any(wanted):
            balance_request[0] = False
            print("Stop walking first, then press B")
        elif balance_request[0] and still_time >= 2.0:  # stood calmly for 2 s
            balance_request[0] = False
            print("Standing on the left leg")
            balance = OneLegStand(JOINTS, DEFAULT_POS, target)
            balance_start = robot.getTime()
        gravity = policy_step(q, dq)
        if gravity[2] > -0.5:
            print("K1 fell over, stopping the policy")
            break
    limit = TORQUE_LIMIT if balance is None else TORQUE_LIMIT_BALANCE
    torque = np.clip((KP if balance is None else KP_BALANCE) * (target - q) - KD * dq, -limit, limit)
    for m, t in zip(motors, torque):
        m.setTorque(float(t))
    if bridge is not None and step % (32 // dt_ms) == 0:
        bridge.publish_camera()
    step += 1
