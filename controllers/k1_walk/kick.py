"""Kick the ball with the right foot while standing on the left leg (same balance as one_leg.py).
  shift  trunk over the left foot     wind  pull the right foot back and up
  swing  swing it forward fast        follow, back, down  bring it back and put it down
The ball should be about 22 cm in front of the trunk and 9 cm to the right (P key puts it there).
"""
from one_leg import OneLegStand

KICK = {
    "wind_hip": 0.3, "wind_knee": 1.0, "wind_time": 0.5,     # rad (hip > 0: foot back), s
    "swing_hip": -0.6, "swing_knee": 0.3, "swing_time": 0.15,
    "follow_time": 0.15,
    "back_hip": -0.2, "back_knee": 0.6, "back_time": 0.4,
    "down_time": 0.4,
}
ANKLE_MIN = -0.7   # rad, ankle pitch offset limit (the joint stops at -0.87, standing is -0.15)


class Kick(OneLegStand):
    def __init__(self, joints, default_pos, start_pose, p=None):
        super().__init__(joints, default_pos, start_pose, dict(KICK, **(p or {})))
        p = self.p
        self.phases = [("shift", p["shift_time"]), ("wind", p["wind_time"]), ("swing", p["swing_time"]),
                       ("follow", p["follow_time"]), ("back", p["back_time"]), ("down", p["down_time"])]
        self.duration = sum(d for _, d in self.phases)
        self.keys = {"shift": (0.0, 0.0), "wind": (p["wind_hip"], p["wind_knee"]),
                     "swing": (p["swing_hip"], p["swing_knee"]), "follow": (p["swing_hip"], p["swing_knee"]),
                     "back": (p["back_hip"], p["back_knee"]), "down": (0.0, 0.0), "end": (0.0, 0.0)}
        self.order = [n for n, _ in self.phases]

    def free_leg(self, phase, smooth):
        if phase in ("shift", "end"):
            return 0.0, 0.0, 0.0
        h0, k0 = self.keys[self.order[self.order.index(phase) - 1]]
        h1, k1 = self.keys[phase]
        hip, knee = h0 + smooth * (h1 - h0), k0 + smooth * (k1 - k0)
        return hip, knee, max(ANKLE_MIN, -(hip + knee))   # keep the sole level
