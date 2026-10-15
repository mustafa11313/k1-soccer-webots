"""Goal scoring state machine, without any ROS code so it can be tested on its own.

Input:  ball position and the two goal posts in the robot's base_footprint frame
        (x forward, y left). The ball may be None when it is not known.
        Posts are given as seen from the field: left post first.
Output: walking command (vx, vy, vyaw)

The walking kick only pushes the ball in the direction the robot walks, so the robot
first walks to a point behind the ball on the line from the goal through the ball,
then dribbles: it keeps walking through the ball towards the goal, steering so the
ball keeps rolling goalwards, without stopping between touches. If it stands on the
wrong side of the ball it walks around it in a wide arc first, since the walking
policy cannot sidestep.

Close to the goal it shoots instead: it stops short of the ball, stands still, turns
and nudges itself forward or back with short walking pulses until the ball is in front of
the right foot and the kick points into the goal, and asks the walking controller for a
kick (take_kick()), which sends the ball 3-4 m. After a turn it always takes a few steps
straight before kicking: a turn on the spot leaves the feet staggered and the kick falls.
If no turn brings the kick inside the goal, it keeps dribbling.
"""
import math

from k1_behavior.chaser_logic import MAX_AIM_ERROR, SEARCH, TURN_SPEED, WALK_SPEED, clip

AROUND, BEHIND, DRIBBLE, SCORED = "AROUND", "BEHIND", "DRIBBLE", "SCORED"
LINE_UP, SETTLE, ADJUST, NUDGE, KICK = "LINE_UP", "SETTLE", "ADJUST", "NUDGE", "KICK"

RUN_UP = 0.8        # m behind the ball, on the goal line, where the run-up starts
SIDE = 0.8          # m, sideways distance when walking around the ball
BEHIND_LIMIT = 0.3  # m the robot must be behind the ball to go straight for the run-up point
LINED_UP = 0.3      # rad, robot-ball direction vs ball-goal direction to start dribbling
LINED_UP_EXIT = 0.9 # rad, stop dribbling and walk around when it gets worse than this
DRIBBLE_REACH = 0.7 # m, a ball this close in front stays in the dribble even if the line is poor
CLOSE = 0.6         # m, closer than this steer by the goal direction, not the ball direction
AIM_GAIN = 0.8      # steer this much (per rad of misalignment) to the side that pushes the ball goalwards
GOAL_CONFIRM = 1.0  # s the ball must be seen in the goal (a false detection in the net is not a goal)
IN_GOAL = 0.3       # m behind the goal line (ball centre) before it counts, the estimate is a few cm off
AIM_DEPTH = 0.6     # m behind the goal line, the point the ball is pushed towards
POST_CLEARANCE = 0.5  # m, never walk forward into a goal post closer than this
BRAKE_TIME = 1.0    # s standing still before turning on the spot (turning while walking trips the robot)
MAX_STEER = 0.2     # rad/s, turning while walking (sharper curves speed the robot up and trip it)

# Kick (kick.py in the walking controller, right foot)
KICK_RANGE = 2.5    # m, shoot when the ball is at most this far from the goal line
KICK_X = (0.21, 0.38)    # m, ball centre in front of the trunk for the kick to hit it ...
KICK_Y = (-0.13, 0.01)   # ... and to the side (right foot)
KICK_AIM_Y = -0.05       # m, walk up so the ball ends up here
# The ball leaves at an angle that depends on where the foot hits it: straight ahead for
# a ball 3 cm right of the trunk, more to the right the further right it is (about 3.8 rad
# per m, measured in Webots: +10 deg for y 0, -7 deg for y -0.05, -24 deg for y -0.13)
STRAIGHT_Y, ANGLE_PER_M = -0.03, 3.8
POST_MARGIN = 0.15       # rad, the kick direction must pass this far inside each post
STOP_DIST = 1.2     # m, stop walking at this distance (the robot coasts on about 0.6 m)
SETTLE_TIME = 3.0   # s, stand still after walking before measuring the ball
NUDGE_REST = 2.5    # s, stand still after each nudge
MAX_NUDGES = 12
KICK_WAIT = 8.0     # s, the controller waits until it stands calmly (2 s), the kick takes 3.3 s
NO_KICK_TIME = 15.0 # s, after a failed line-up just dribble for a while
# Nudges, measured in Webots: (vx, vyaw, seconds) -> about how far it moves
FWD_SMALL = (0.3, 0.0, 1.25)   # 1-2 cm
FWD_MEDIUM = (0.3, 0.0, 1.5)   # 4-6 cm
FWD_LARGE = (0.3, 0.0, 1.75)   # 10-14 cm
FWD_XL = (0.3, 0.0, 2.25)      # about 25 cm
BACK = (-0.2, 0.0, 2.5)        # about 3 cm back
TURN = 0.23                    # rad, a turn nudge: vyaw 0.5 for 0.75 s turns 11-18 deg
TURN_LEFT = (0.0, 0.5, 0.75)
TURN_RIGHT = (0.0, -0.5, 0.75)
MAX_TURNS = 3


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _unit(a):
    n = math.hypot(*a)
    return (a[0] / n, a[1] / n) if n > 1e-6 else (1.0, 0.0)


def _angle(a, b):
    return abs(math.atan2(a[0] * b[1] - a[1] * b[0], _dot(a, b)))


class Striker:
    def __init__(self, foot_y=-0.07, lost_after=1.5):
        self.foot_y = foot_y
        self.lost_after = lost_after  # s without a ball estimate before searching
        self.state = SEARCH
        self.ball = None
        self.last_seen = -1e9
        self.search_dir = 1.0
        self.turning = False
        self.turn_dir = 1.0
        self.side = None
        self.walking = False
        self.brake_until = -1e9
        self.fresh = True
        self.kick_request = False
        self.kick_start = None
        self.settle_until = None
        self.nudge = None
        self.nudge_until = None
        self.nudges = 0
        self.turns = 0
        self.turned = False
        self.no_kick_until = -1e9
        self.in_goal_since = None
        self.note = ""  # why it kicked or gave up, for the log

    def take_kick(self):
        """True once when the walking controller should kick."""
        kick, self.kick_request = self.kick_request, False
        return kick

    def update(self, ball, left_post, right_post, now, fresh=True):
        """ball comes from BallTracker: (x, y) in base_footprint, or None if unknown.

        fresh: the ball was actually seen just now (not only predicted). A goal only
        counts when the ball is seen in it.
        """
        self.fresh = fresh
        cmd = self._update(ball, left_post, right_post, now)
        for post in (left_post, right_post):
            if cmd[0] > 0 and 0.0 < post[0] < POST_CLEARANCE and abs(post[1]) < 0.3:
                # a goal post right ahead: turn away from it instead of walking into it
                cmd = (0.0, 0.0, -TURN_SPEED if post[1] > 0 else TURN_SPEED)
        if cmd[0] > 0:
            self.walking = True
        elif self.walking and cmd[2] != 0:
            # was walking, now wants to turn on the spot: stand still for a moment first
            self.walking = False
            self.brake_until = now + BRAKE_TIME
        if now < self.brake_until:
            return 0.0, 0.0, 0.0
        if cmd == (0.0, 0.0, 0.0):
            self.walking = False
        return cmd

    def _update(self, ball, left_post, right_post, now):
        if ball is not None:
            self.ball = ball
            self.last_seen = now
            self.search_dir = 1.0 if ball[1] >= 0 else -1.0
        if self.state == SCORED:
            return 0.0, 0.0, 0.0
        if self.state == KICK:
            if now - self.kick_start < KICK_WAIT:
                return 0.0, 0.0, 0.0
            self.state = DRIBBLE
            self.no_kick_until = now + 3.0   # look where the ball went first
        if self.state in (SETTLE, ADJUST, NUDGE):
            posts = [math.atan2(*reversed(_sub(post, self.ball))) for post in (left_post, right_post)]
            return self._adjust(now, posts)
        if now - self.last_seen > self.lost_after:
            self.state = SEARCH
            return 0.0, 0.0, TURN_SPEED * self.search_dir

        ball = self.ball
        centre = ((left_post[0] + right_post[0]) / 2, (left_post[1] + right_post[1]) / 2)
        across = _sub(right_post, left_post)
        into_goal = _unit((-across[1], across[0]))
        if self._in_goal(ball, centre, across, into_goal):
            if self.fresh and self.in_goal_since is None:
                self.in_goal_since = now
            if self.in_goal_since is not None and now - self.in_goal_since >= GOAL_CONFIRM:
                self.state = SCORED
                return 0.0, 0.0, 0.0
        else:
            self.in_goal_since = None

        # aim at a point inside the goal, so a ball lying on the goal line is pushed in, not along it
        aim_point = (centre[0] + AIM_DEPTH * into_goal[0], centre[1] + AIM_DEPTH * into_goal[1])
        to_goal = _unit(_sub(aim_point, ball))        # direction the ball should go
        robot = (-ball[0], -ball[1])                  # robot position relative to the ball
        behind = -_dot(robot, to_goal)                # how far the robot is behind the ball
        misalign = _angle(ball, to_goal)              # robot->ball vs ball->goal
        dist = math.hypot(*ball)

        to_line = _dot(_sub(centre, ball), into_goal)   # ball distance to the goal line
        if (now > self.no_kick_until and 0.3 < to_line < KICK_RANGE and behind > BEHIND_LIMIT
                and misalign < (LINED_UP_EXIT if self.state == LINE_UP else LINED_UP)):
            return self._line_up(ball, now)

        if self.state == DRIBBLE:
            # keep going while the ball is in front of the feet and roughly goalwards
            # (right at the feet the robot->ball direction means little, so use the heading)
            close_in_front = dist < DRIBBLE_REACH and ball[0] > -0.05 and abs(ball[1]) < 0.4
            heading_error = _angle((1.0, 0.0), to_goal)
            if (misalign < LINED_UP_EXIT and behind > 0.0) or (close_in_front and heading_error < LINED_UP_EXIT):
                return self._dribble(ball, to_goal)
        elif behind > BEHIND_LIMIT and misalign < LINED_UP:
            return self._dribble(ball, to_goal)
        # once walking around, only switch to the run-up when clearly behind (no flip-flop)
        if behind > BEHIND_LIMIT + (0.2 if self.state == AROUND else 0.0):
            self.state = BEHIND
            target = (ball[0] - RUN_UP * to_goal[0], ball[1] - RUN_UP * to_goal[1])
        else:
            if self.state != AROUND or self.side is None:
                # pick the side the robot is on once, so it does not flip-flop when straight in front
                self.side = 1.0 if (to_goal[0] * robot[1] - to_goal[1] * robot[0]) >= 0 else -1.0
            self.state = AROUND
            side = self.side
            perp = (-to_goal[1] * side, to_goal[0] * side)
            target = (ball[0] + SIDE * perp[0] - 0.4 * to_goal[0],
                      ball[1] + SIDE * perp[1] - 0.4 * to_goal[1])
        return self._walk_to(target)

    def _in_goal(self, ball, centre, across, into_goal):
        rel = _sub(ball, centre)
        half_width = math.hypot(*across) / 2
        lateral = abs(_dot(rel, _unit(across)))
        return _dot(rel, into_goal) > IN_GOAL and lateral < half_width

    def _walk_to(self, target):
        aim = math.atan2(target[1], target[0])
        return self._steer(aim)

    def _steer(self, aim):
        """Turn on the spot if the target is far off to the side, else walk and steer."""
        if abs(aim) > MAX_AIM_ERROR or (self.turning and abs(aim) > 0.1):
            if not self.turning or abs(aim) < 2.5:
                # keep the turn direction when the target is nearly straight behind
                self.turn_dir = math.copysign(1.0, aim)
            self.turning = True
            return 0.0, 0.0, TURN_SPEED * self.turn_dir
        self.turning = False
        return WALK_SPEED, 0.0, clip(1.0 * aim, -MAX_STEER, MAX_STEER)

    def _dribble(self, ball, to_goal):
        """Walk through the ball without stopping, steering so it rolls towards the goal.

        From further away, aim at the ball, a little to the side away from the goal, so
        the foot hits it off-centre and it rolls goalwards. Right at the ball its
        direction jumps around with every step, so there steer by the goal direction
        and only gently keep the ball in front of the foot.
        """
        self.state = DRIBBLE
        dist = math.hypot(*ball)
        goal_heading = math.atan2(to_goal[1], to_goal[0])   # ball->goal, in the robot frame
        if dist < CLOSE:
            lateral = math.atan2(ball[1] - self.foot_y, max(ball[0], 0.4))
            aim = 0.5 * goal_heading + lateral
        else:
            ball_dir = math.atan2(ball[1] - self.foot_y, ball[0])
            error = goal_heading - math.atan2(ball[1], ball[0])
            error = math.atan2(math.sin(error), math.cos(error))
            aim = ball_dir - clip(AIM_GAIN * error, -0.3, 0.3)
        if (dist > DRIBBLE_REACH or self.turning) and (abs(aim) > MAX_AIM_ERROR or (self.turning and abs(aim) > 0.1)):
            return self._steer(aim)  # far away: turn on the spot first (and finish a turn once started)
        self.turning = False
        return WALK_SPEED, 0.0, clip(1.0 * aim, -MAX_STEER, MAX_STEER)

    def _line_up(self, ball, now):
        """Walk at the ball (so it ends up in front of the right foot) and stop short of it."""
        self.state = LINE_UP
        dist = math.hypot(*ball)
        aim = math.atan2(ball[1] - KICK_AIM_Y, ball[0])
        if dist < STOP_DIST and abs(aim) < 0.3 and not self.turning:
            self.state, self.settle_until, self.nudges, self.turns = SETTLE, now + SETTLE_TIME, 0, 0
            self.turned = False
            return 0.0, 0.0, 0.0
        return self._steer(aim)

    def _adjust(self, now, posts):
        """Stand still, nudge until the ball is in front of the right foot, then kick."""
        if self.state == SETTLE:
            if now < self.settle_until:
                return 0.0, 0.0, 0.0
            self.state = ADJUST
        if self.state == NUDGE:
            vx, vyaw, duration = self.nudge
            if now < self.nudge_until - NUDGE_REST:
                return vx, 0.0, vyaw
            if now < self.nudge_until:
                return 0.0, 0.0, 0.0
            self.state = ADJUST
        x, y = self.ball
        self.note = f"ball x={x:.2f} y={y:.2f}"
        if x > STOP_DIST + 0.4 or abs(y) > 0.5 or x < 0.05 or self.nudges >= MAX_NUDGES:
            self.note += f", lost the ball or too many nudges ({self.nudges})"
            self.state, self.no_kick_until = DRIBBLE, now + NO_KICK_TIME   # give up, dribble instead
            return 0.0, 0.0, 0.0
        kick_dir = ANGLE_PER_M * (y - STRAIGHT_Y)   # where the ball would go, robot frame
        # directions from the ball to the left and right post, the kick must pass between
        on_target = posts[1] + POST_MARGIN < kick_dir < posts[0] - POST_MARGIN
        self.note += f", kick direction {kick_dir:.2f} rad, goal {posts[1] + POST_MARGIN:.2f} to {posts[0] - POST_MARGIN:.2f}"
        if not KICK_Y[0] <= y <= KICK_Y[1] or not on_target:
            # turn on the spot (the kick squares the feet up first) if that brings the
            # ball in front of the right foot and the kick inside the goal, else dribble
            turn = self._aim_turn(x, y, posts)
            if turn is None or self.turns >= MAX_TURNS:
                self.note += f", needs y from {KICK_Y[0]} to {KICK_Y[1]} and the direction inside the goal"
                self.state, self.no_kick_until = DRIBBLE, now + NO_KICK_TIME
                return 0.0, 0.0, 0.0
            self.note += ", turning " + ("left" if turn[1] > 0 else "right")
            self.turned = True
            self.state, self.nudge = NUDGE, turn
            self.nudge_until = now + turn[2] + NUDGE_REST
            self.turns += 1
            self.nudges += 1
            return 0.0, 0.0, 0.0
        err = x - sum(KICK_X) / 2
        if x > KICK_X[1]:
            nudge = (FWD_XL if err > 0.3 else FWD_LARGE if err > 0.10 else
                     FWD_MEDIUM if err > 0.04 else FWD_SMALL)
        elif x < KICK_X[0]:
            nudge = BACK
        elif self.turned:
            # a turn on the spot leaves the feet staggered and the kick then falls:
            # a few steps straight square them up again
            nudge = BACK if x < KICK_X[1] - 0.04 else FWD_SMALL
        else:
            self.state, self.kick_start, self.kick_request = KICK, now, True
            return 0.0, 0.0, 0.0
        self.turned = False
        self.state, self.nudge = NUDGE, nudge
        self.nudge_until = now + nudge[2] + NUDGE_REST
        self.nudges += 1
        return 0.0, 0.0, 0.0

    @staticmethod
    def _kick_error(y, posts):
        """How far the ball is from kickable: m outside the y zone + rad outside the goal."""
        kick_dir = ANGLE_PER_M * (y - STRAIGHT_Y)
        y_err = max(KICK_Y[0] - y, y - KICK_Y[1], 0.0)
        aim_err = max(posts[1] + POST_MARGIN - kick_dir, kick_dir - (posts[0] - POST_MARGIN), 0.0)
        return 3.0 * y_err + aim_err

    def _aim_turn(self, x, y, posts):
        """The turn nudge that brings the ball closest to kickable, or None if no turn helps."""
        best, best_err = None, self._kick_error(y, posts) - 0.05
        for nudge, a in ((TURN_LEFT, TURN), (TURN_RIGHT, -TURN)):
            # the robot turns by a: the ball and the post directions turn by -a
            err = self._kick_error(-math.sin(a) * x + math.cos(a) * y, [p - a for p in posts])
            if err < best_err:
                best, best_err = nudge, err
        return best
