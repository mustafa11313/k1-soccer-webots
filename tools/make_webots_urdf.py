"""Make a Webots-friendly copy of the K1 URDF: box feet + motor rotor inertia."""
import sys
import xml.etree.ElementTree as ET

src, mjcf, dst = sys.argv[1], sys.argv[2], sys.argv[3]
tree = ET.parse(src)
root = tree.getroot()
links = {link.get('name'): link for link in root.findall('link')}

# 1. Box feet (same boxes Booster uses in MuJoCo)
for name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
    col = links[name].find('collision')
    col.find('origin').attrib.update(xyz='0.026 0 -0.02', rpy='0 0 0')
    geom = col.find('geometry')
    geom.clear()
    ET.SubElement(geom, 'box', size='0.18 0.07 0.036')

# 2. Motor rotor inertia ("armature") from Booster's MuJoCo model,
#    added to the link each joint moves (the tiny ankle link's goes on the foot)
armature = {}
for joint in ET.parse(mjcf).getroot().iter('joint'):
    if joint.get('armature'):
        armature[joint.get('name')] = (float(joint.get('armature')), joint.get('axis'))
for joint in root.findall('joint'):
    if joint.get('name') not in armature:
        continue
    value, axis = armature[joint.get('name')]
    child = joint.find('child').get('link')
    if child.endswith('ankle_pitch_link'):
        child = child.replace('pitch', 'roll')
    inertia = links[child].find('inertial/inertia')
    key = {'1 0 0': 'ixx', '0 1 0': 'iyy', '0 0 1': 'izz'}[axis]
    inertia.set(key, str(float(inertia.get(key)) + value))
tree.write(dst)
print('wrote', dst)
