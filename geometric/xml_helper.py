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

def write_coors_to_xml(M, state_xml, out_xml):
    if type(state_xml) is str:
        with open(state_xml) as fobj:
            xml = fobj.read()
        root = ET.fromstring(xml)
    elif type(state_xml) is ET.Element:
        root = copy.deepcopy(state_xml)
    else:
        raise IOError("Expected second argument to write_coors_to_xml to be file name or xml.etree.ElementTree.Element type")

    for child in root:
        if child.tag == "Positions":
            for aid in range(M.na):
                x = M.xyzs[0][aid, 0] / 10.0
                y = M.xyzs[0][aid, 1] / 10.0
                z = M.xyzs[0][aid, 2] / 10.0
                child[aid].attrib['x'] = "%.16e" % x
                child[aid].attrib['y'] = "%.16e" % y
                child[aid].attrib['z'] = "%.16e" % z
    # write to the output XML file.
    fout = open(out_xml, "w")
    print(ET.tostring(root).decode("utf-8"), file=fout)
    fout.close()

