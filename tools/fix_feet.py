# Replace the K1 foot collision meshes with simple boxes (same boxes Booster uses in MuJoCo)
import sys, xml.etree.ElementTree as ET
src, dst = sys.argv[1], sys.argv[2]
tree = ET.parse(src)
for link in tree.getroot().findall('link'):
    if link.get('name') in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        col = link.find('collision')
        col.find('origin').attrib.update(xyz='0.026 0 -0.02', rpy='0 0 0')
        geom = col.find('geometry')
        geom.clear()
        ET.SubElement(geom, 'box', size='0.18 0.07 0.036')
        print('fixed', link.get('name'))
tree.write(dst)
