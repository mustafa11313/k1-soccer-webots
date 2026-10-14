"""Stand on the left leg for a few seconds, then put the right foot back down.

Used by k1_walk.py (press B in the 3D view while the robot stands still). Steps:
  shift  move the trunk over the left foot (hip and ankle roll)
  lift   lift the right foot straight up (hip, knee and ankle pitch)
  hold   stand on one leg
  down   put the right foot back down, then the walking policy takes over again

Balance while on one leg:
  - sideways: the trunk position over the left sole is worked out from the leg joint
    angles and the IMU, and the left ankle roll keeps it there (PD + a slow integral),
  - forwards: the left ankle and hip pitch lean against any forward/backward tilt,
  - if the trunk leans sideways, the right leg is tilted back to vertical before landing.
"""
import math

import numpy as np

P = {
    "shift": 0.20,        # rad, hip/ankle roll that moves the trunk over the left foot
    "shift_time": 1.0,    # s
    "lift": 0.5,          # rad: hip pitch -lift, knee 2*lift, ankle -lift (foot straight up, sole level)
    "lift_time": 0.6,
    "hold_time": 5.0,
    "down_time": 0.4,
    "y_target": 0.02,     # m, trunk 2 cm left of the sole centre: the lifted right leg pulls the body's centre of mass right
    "kp_y": 2.0,          # rad of ankle roll per m of sideways error
    "kd_y": 0.3,          # rad per m/s
    "ki_y": 0.5,          # slow integral on the shift (rad/s per m)
    "max_extra": 0.15,
    "kp_pitch": 1.0, "kd_pitch": 0.1,   # left ankle pitch against trunk pitch
    "kp_hip": 1.0, "kd_hip": 0.1,       # left hip pitch against trunk pitch
    "kp_roll": 0.5, "kd_roll": 0.05,    # left ankle roll against trunk roll
    "leg_level": -0.5,    # right hip/ankle roll per rad of trunk roll while landing
}
LEG_GAIN = 3.0            # leg joints are 3x stiffer during the motion
# The walking policy limits the leg torques well below the motors' ratings (hip roll 20 Nm).
# On one leg the standing hip roll needs more than that, so during the motion the leg
# joints may use their full rating from the K1 URDF:
# hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll (Nm)
LEG_TORQUE_LIMIT = (68.0, 43.0, 38.3, 112.0, 38.3, 38.3)

# Left leg from the K1 URDF: (joint, offset from the parent joint, axis)
LEFT_LEG = [
    ("left_hip_pitch_joint", (0.0, 0.096, -0.077), "y"),
    ("left_hip_roll_joint", (0.0, 0.0, -0.026), "x"),
    ("left_hip_yaw_joint", (0.012, 0.0, -0.0485), "z"),
    ("left_knee_pitch_joint", (-0.014, 0.0, -0.117), "y"),
    ("left_ankle_pitch_joint", (0.0002, 0.0002, -0.24519), "y"),
    ("left_ankle_roll_joint", (0.0, 0.0, 0.0), "x"),
]
SOLE = np.array([0.026, 0.0, -0.038])  # middle of the sole in the ankle frame


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _level(gravity):
    """Rotation from the trunk frame to a level frame with the same heading."""
    g = -np.asarray(gravity, dtype=float)
    g /= np.linalg.norm(g)
    v = np.cross(g, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(v), g[2]
    if s < 1e-9:
        return np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def trunk_over_left_foot(joints, q, gravity):
    """Trunk position relative to the middle of the left sole, level frame (x forward, y left)."""
    R, p = np.eye(3), np.zeros(3)
    for name, offset, axis in LEFT_LEG:
        p = p + R @ np.array(offset)
        R = R @ _rot(axis, q[joints.index(name)])
    return _level(gravity) @ -(p + R @ SOLE)


class OneLegStand:
    def __init__(self, joints, default_pos, start_pose, p=None):
        self.p = dict(P, **(p or {}))
        self.joints = joints
        self.ix = {n: joints.index(n) for n in joints}
        self.default = default_pos.copy()
        self.start = start_pose.copy()
        p = self.p
        self.phases = [("shift", p["shift_time"]), ("lift", p["lift_time"]),
                       ("hold", p["hold_time"]), ("down", p["down_time"])]
        self.duration = sum(d for _, d in self.phases)
        self.extra, self.vy, self.prev_y, self.last_t = 0.0, 0.0, None, None

    def _phase(self, t):
        for name, d in self.phases:
            if t < d:
                return name, t / d
            t -= d
        return "end", 1.0

    def done(self, t):
        return t >= self.duration

    def target(self, t, q, gravity, ang_vel):
        """Joint targets at time t (s since the start)."""
        p, ix = self.p, self.ix
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        phase, a = self._phase(t)
        smooth = 0.5 - 0.5 * math.cos(math.pi * a)
        shift_amt = smooth if phase == "shift" else 1.0
        lift_amt = {"lift": smooth, "hold": 1.0, "down": 1 - smooth}.get(phase, 0.0)
        one_leg = phase != "shift"

        # sideways trunk position over the left sole
        y = trunk_over_left_foot(self.joints, q, gravity)[1]
        if self.prev_y is not None and dt > 0:
            self.vy += 0.2 * ((y - self.prev_y) / dt - self.vy)
        self.prev_y = y
        err = p["y_target"] - y
        if t > 0.3 * p["shift_time"]:
            self.extra = float(np.clip(self.extra + p["ki_y"] * err * dt, -p["max_extra"], p["max_extra"]))
        corr = (p["kp_y"] * err - p["kd_y"] * self.vy) * shift_amt

        # start from where the walking policy left the joints
        pose = self.start + smooth * (self.default - self.start) if phase == "shift" else self.default.copy()
        base = (p["shift"] + self.extra) * shift_amt
        both = 0.0 if one_leg else corr   # with both feet down the correction moves both legs
        pose[ix["left_hip_roll_joint"]] += -base - both
        pose[ix["right_hip_roll_joint"]] += -base - both
        pose[ix["left_ankle_roll_joint"]] += base + both + (corr if one_leg else 0.0)
        pose[ix["right_ankle_roll_joint"]] += base + both

        a_ = p["lift"] * lift_amt
        pose[ix["right_hip_pitch_joint"]] -= a_
        pose[ix["right_knee_pitch_joint"]] += 2 * a_
        pose[ix["right_ankle_pitch_joint"]] -= a_

        pitch = math.atan2(gravity[0], -gravity[2])  # > 0: tilted forward
        roll = math.atan2(gravity[1], -gravity[2])   # > 0: tilted left
        if phase == "down":
            pose[ix["right_hip_roll_joint"]] += p["leg_level"] * smooth * roll
            pose[ix["right_ankle_roll_joint"]] -= p["leg_level"] * smooth * roll
        k = 1.0 if one_leg else shift_amt
        pose[ix["left_ankle_pitch_joint"]] += k * (p["kp_pitch"] * pitch + p["kd_pitch"] * ang_vel[1])
        pose[ix["left_hip_pitch_joint"]] += k * (p["kp_hip"] * pitch + p["kd_hip"] * ang_vel[1])
        pose[ix["left_ankle_roll_joint"]] -= k * (p["kp_roll"] * roll - p["kd_roll"] * ang_vel[0])
        return pose
