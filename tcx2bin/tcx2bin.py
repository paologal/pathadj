#!/usr/bin/env python

import xml.etree.ElementTree as ET
import glob
import sys
import getopt
import os
import struct


def tcx_to_binary(tcxdir):
    print "Converting TCX files in " + tcxdir
    os.chdir(tcxdir)
    for files in glob.glob("*.TCX"):
        base = os.path.splitext(files)[0]
        os.rename(files, base + ".tcx")
    files_count = 0
    for files in glob.glob("*.tcx"):
        print "Converting %s " % (files)
        output_file = open(files.replace("tcx", "bin"), 'wb')
        data = []
        count = 0
        tree = ET.parse(files)
        nodes = tree.findall(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Trackpoint")
        for n in nodes:
            poselem0 = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position")
            if poselem0 is not None:
                latelem0 = poselem0.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LatitudeDegrees")
                lonelem0 = poselem0.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LongitudeDegrees")
                lat0 = float(latelem0.text)
                lon0 = float(lonelem0.text)
                data.append(struct.pack('<ff', lat0, lon0))
                count = count + 1
        output_file.write(struct.pack('<I', count))
        for i in data:
            output_file.write(i)
        output_file.write(struct.pack('<II', count, 0x00ED00ED))
        output_file.close()
        files_count = files_count + 1
    print "Converted %d files" % (files_count)
    
def main():
    for arg in sys.argv[1:]:
        tcx_to_binary(arg)

if __name__ == "__main__":
    main()