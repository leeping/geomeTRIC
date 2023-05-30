import xml.etree.ElementTree as ET
from .molecule import Molecule
import copy

def read_coors_from_xml(M, state_xml):
    with open(state_xml) as fobj:
        xml = fobj.read()
    root = ET.fromstring(xml)

    for child in root:
        if child.tag == "Positions":
            for aid in range(M.na):
                x = float(child[aid].attrib['x'])
                y = float(child[aid].attrib['y'])
                z = float(child[aid].attrib['z'])
                M.xyzs[0][aid, 0] = x*10.0
                M.xyzs[0][aid, 1] = y*10.0
                M.xyzs[0][aid, 2] = z*10.0
    return root;


def write_coors_to_xml(M, root_template, filename):
    root = copy.deepcopy(root_template)   
    for child in root:
        if child.tag == "Positions":
            for aid in range(M.na):
                x = M.xyzs[0][aid, 0] / 10.0
                y = M.xyzs[0][aid, 1] / 10.0
                z = M.xyzs[0][aid, 2] / 10.0
                child[aid].attrib['x'] = "%.16e" % x
                child[aid].attrib['y'] = "%.16e" % y
                child[aid].attrib['z'] = "%.16e" % z
    # write
    fout = open(filename, "w")
    print(ET.tostring(root).decode("utf-8"), file=fout)
    fout.close()

