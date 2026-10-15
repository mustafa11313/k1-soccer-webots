"""Kick the ball with the right foot while standing on the left leg.

Same balance as the one-leg stand (one_leg.py); only the right leg moves differently:
  shift   move the trunk sideways over the left foot, the feet stay where they are
  square  lift the right foot a little and bring both legs to the standing pose; after
          a turn the feet are staggered, this puts the trunk back over the left foot
  wind    pull the right foot back and up (hip pitch back, knee bent)
  swing   swing it forward fast through the ball (hip pitch forward, knee straighter)
  follow  stay there for a moment
  back    bring the foot back under the body
  down    put it down, then the walking policy takes over again
The ball should be about 25 cm in front of the robot, in front of the right foot.
"""
import math

from one_leg import LEG_GAIN, LEG_TORQUE_LIMIT, OneLegStand

KICK = {
    "wind_hip": 0.3, "wind_knee": 1.0, "wind_time": 0.5,     # rad (hip > 0: foot back), s
    "swing_hip": -0.6, "swing_knee": 0.3, "swing_time": 0.3,
    "follow_time": 0.15,
    "back_hip": -0.1, "back_knee": 0.8, "back_time": 0.4,
    "down_time": 0.4,
    "kp_pitch": 2.0, "kp_hip": 2.0,   # lean back harder: the swing tips the trunk forward
    "kp_roll": 1.0,                  # and push harder against a sideways tilt: it makes the foot spin
    "square": 1, "square_time": 0.5, "square_hip": -0.2, "square_knee": 0.4,
}
ANKLE_MIN = -0.7   # rad, ankle pitch offset limit (the joint stops at -0.87, standing is -0.15)


class Kick(OneLegStand):
    def __init__(self, joints, default_pos, start_pose, p=None):
        super().__init__(joints, default_pos, start_pose, dict(KICK, **(p or {})))
        p = self.p
        self.phases = [("shift", p["shift_time"])] + ([("square", p["square_time"])] if p["square"] else []) + [
                       ("wind", p["wind_time"]), ("swing", p["swing_time"]),
                       ("follow", p["follow_time"]), ("back", p["back_time"]), ("down", p["down_time"])]
        self.duration = sum(d for _, d in self.phases)
        self.keys = {"shift": (0.0, 0.0), "square": (p["square_hip"], p["square_knee"]),
                     "wind": (p["wind_hip"], p["wind_knee"]),
                     "swing": (p["swing_hip"], p["swing_knee"]), "follow": (p["swing_hip"], p["swing_knee"]),
                     "back": (p["back_hip"], p["back_knee"]), "down": (0.0, 0.0), "end": (0.0, 0.0)}
        self.order = [n for n, _ in self.phases]
        # leg pitch and yaw joints: kept where the walking policy left them while both feet
        # are down (shift), moved to the standing pose once the right foot is off the ground
        self.legs = [self.ix[f"{side}_{j}_joint"] for side in ("left", "right")
                     for j in ("hip_pitch", "hip_yaw", "knee_pitch", "ankle_pitch")]

    def target(self, t, q, gravity, ang_vel):
        pose = super().target(t, q, gravity, ang_vel)
        if self.p["square"]:
            phase, a = self._phase(t)
            smooth = 0.5 - 0.5 * math.cos(math.pi * a)
            d = self.start[self.legs] - self.default[self.legs]
            if phase == "shift":
                pose[self.legs] += smooth * d
            elif phase == "square":
                pose[self.legs] += (1 - smooth) * d
        return pose

    def free_leg(self, phase, smooth):
        if phase in ("shift", "end"):
            return 0.0, 0.0, 0.0
        h0, k0 = self.keys[self.order[self.order.index(phase) - 1]]
        h1, k1 = self.keys[phase]
        hip, knee = h0 + smooth * (h1 - h0), k0 + smooth * (k1 - k0)
        return hip, knee, max(ANKLE_MIN, -(hip + knee))   # keep the sole level
