#!/usr/bin/env python

# Convert a CFOUR ZMAT file from Z-matrix input to Cartesian coordinates.

import os, re, shutil
import subprocess

def run(command):
    subprocess.run(command, check=True, shell=True, capture_output=True)

if not os.path.exists("ZMAT"):
    raise IOError("This script requires ZMAT to be present in the directory")

for ln, line in enumerate(open("ZMAT").readlines()):
    line = line.strip().expandtabs()
    if ln == 1:
        if len(line.split()) == 4:
            raise IOError("ZMAT already contains Cartesian coordinates, exiting.")

if os.path.exists("ZMAT_cart"):
    raise IOError("Please remove ZMAT_cart file (intended output file name) before proceeding")

if not os.path.exists('convert'):
    os.makedirs('convert')
elif not os.path.isdir('convert'):
    raise IOError("convert exists but it is not a folder")
shutil.copy2("ZMAT", "convert")
os.chdir('convert')

run("xclean")
run("xjoda")

if not os.path.exists("JMOLplot"):
    raise IOError("Expected the file JMOLplot to be created, but it does not exist")

xyzlines = []
for line in open("JMOLplot").readlines():
    if re.match(r"[A-Z][A-Za-z]?( +[-+]?([0-9]*\.)?[0-9]+){3}$", line):
        xyzlines.append(line.strip())

cfour_jobspec = []
reading_jobspec = False
have_coord_cartesian = False
for ln, line in enumerate(open("ZMAT").readlines()):
    line = line.strip().expandtabs()
    if ln == 0:
        comment = line
    if ln == 1:
        if len(line.split()) == 4:
            raise IOError("ZMAT already contains Cartesian coordinates, exiting.")
    if "CFOUR(" in line or "ACES2(" in line or "CRAPS(" in line:
        cfour_jobspec.append(line)
        reading_jobspec = True
    elif reading_jobspec:
        cfour_jobspec.append(line)
    if line.endswith(")"):
        reading_jobspec = False
        break
    if "coord=cartesian" in line.lower():
        have_coord_cartesian = True
    elif "coord=" in line.lower():
        raise RuntimeError("Please remove coord= from input ZMAT file.")

with open("ZMAT_cart", "w") as f:
    # Comment line
    print(comment, file=f)
    
    # .xyz coordinates parsed from JMOLplot, then blank line
    for line in xyzlines:
        print(line, file=f)
    print(file=f)
    
    # Job specification block 
    for ln, line in enumerate(cfour_jobspec):
        if len(cfour_jobspec) == 1:
            line = line[:-1]
        if ln == 0:
            if not have_coord_cartesian:
                line += "\nCOORD=CARTESIAN"
        if ln == len(cfour_jobspec)-1 and not line.endswith(")"):
            line = line + ")"
        print(line, file=f)
    
    # Final blank line
    print(file=f)

os.chdir("..")
shutil.move("ZMAT", "ZMAT.orig")
shutil.move("convert/ZMAT_cart", "ZMAT")
print("Cartesian coordinate input file written to ZMAT ; original moved to ZMAT.orig .")

# Clean up the conversion directory
shutil.rmtree("convert")
