"""One-time patch: let k1_walk.py use the ROS 2 bridge (ros_bridge.py)."""
path = "controllers/k1_walk/k1_walk.py"
s = open(path).read()
changes = [
    ('camera.enable(32)  # new image every 32 ms (~30 fps)\n',
     'camera.enable(32)  # new image every 32 ms (~30 fps)\n'
     'try:\n'
     '    from ros_bridge import RosBridge\n'
     '    bridge = RosBridge(robot, camera)\n'
     'except ImportError:\n'
     '    print("ROS 2 not sourced, keyboard control only")\n'
     '    bridge = None\n'),
    ('    vx = vy = vyaw = 0.0\n    key = keyboard.getKey()\n',
     '    vx = vy = vyaw = 0.0\n    pressed = False\n    key = keyboard.getKey()\n'),
    ('            vy = -MAX_VY\n        key = keyboard.getKey()\n    set_command(vx, vy, vyaw)\n',
     '            vy = -MAX_VY\n        pressed = True\n        key = keyboard.getKey()\n'
     '    return (vx, vy, vyaw) if pressed else None  # None = no key held\n'),
    ('        read_keyboard()\n',
     '        wanted = read_keyboard()\n'
     '        if bridge is not None:\n'
     '            bridge.spin()\n'
     '            if wanted is None:\n'
     '                wanted = bridge.command()\n'
     '        set_command(*(wanted or (0.0, 0.0, 0.0)))\n'),
    ('        m.setTorque(float(t))\n',
     '        m.setTorque(float(t))\n'
     '    if bridge is not None and step % (32 // dt_ms) == 0:\n'
     '        bridge.publish_camera()\n'),
]
for old, new in changes:
    if s.count(old) != 1:
        raise SystemExit(f"could not find this part of k1_walk.py:\n{old}")
    s = s.replace(old, new)
open(path, "w").write(s)
print("k1_walk.py now uses ROS 2")
