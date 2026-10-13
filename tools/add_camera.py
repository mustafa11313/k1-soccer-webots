"""Add a head camera to the K1 PROTO made by urdf2webots (run after every rebuild).
The URDF only marks where the camera sits (head_booster_stereo_rgb_link) and
urdf2webots drops such empty links, so we put a Webots Camera there, looking forward."""
import sys

CAMERA = """                  Camera {
                    translation 0.060138 0.0351 0.092774
                    name "head_camera"
                    fieldOfView 1.5
                    width 640
                    height 480
                  }
"""

path = sys.argv[1]
lines = open(path).read().splitlines(keepends=True)
if any('name "head_camera"' in line for line in lines):
    sys.exit(f"{path} already has a head camera")
name_line = next(i for i, line in enumerate(lines) if 'name "aahead_pitch_link"' in line)
children_line = max(i for i in range(name_line) if lines[i].strip() == "children [")
lines.insert(children_line + 1, CAMERA)
open(path, "w").writelines(lines)
print("added head_camera to", path)
