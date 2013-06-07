#!/usr/bin/env python

from math import *

import xml.etree.ElementTree as ET
import dateutil.parser
import datetime
import time
import sys


prevlat = None
prevlon = None
totaldist = 0.0
prevgarmindist = None

if len(sys.argv) != 2:
    print ("Usage %s tcx_file\n") % sys.argv[0]
    sys.exit()
    
if '.tcx' in sys.argv[1]:
    outfile = sys.argv[1].replace('.tcx', '.csv')
else:
    print ("Not a tcx file: %s\n") % sys.argv[1]
    sys.exit()
    

tree = ET.parse(sys.argv[1])
nodes = tree.findall(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Trackpoint")

f = open(outfile, 'w')

print >> f,"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % \
    ("#", "Time", "Latitude", "Longitude", "Lap Lenght", "Garmin Lap Lenght", "Distance",
     "Garmin Distance", "Garmin Lap Delta", "Garmin Total Delta")
for (counter, n) in enumerate(nodes):
    timeelem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time")
    distelem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters")
    poselem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position")
    if poselem is not None:
        latelem = poselem.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LatitudeDegrees")
        lonelem = poselem.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LongitudeDegrees")
        latd = float(latelem.text)
        lond = float(lonelem.text)
        lat = radians(latd)
        lon = radians(lond)
        garmindist = (float(distelem.text)) / 1000.0

        if prevlat is not None and prevlon is not None:
            dlat = lat - prevlat
            dlon = lon - prevlon
            a = sin(dlat/2)**2 + cos(prevlat) * cos(lat) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            leglen = 6371 * c
            totaldist = totaldist + leglen
            garminleg = garmindist - prevgarmindist
            if leglen != 0: 
                garminlegdelta = ((garminleg-leglen)/leglen) * 100.0
            else:
                garminlegdelta = 0
            if totaldist != 0: 
                garmintotaldelta = ((garmindist-totaldist)/totaldist) * 100.0
            else:
                garmintotaldelta = 0
            timeobj = dateutil.parser.parse(timeelem.text)
            timestamp = time.mktime(timeobj.timetuple())
            print >> f, "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f" % \
                (counter, timestamp, latd, lond, leglen, garminleg, totaldist,
                 garmindist, garminlegdelta, garmintotaldelta)
        prevlat = lat
        prevlon = lon
        prevgarmindist = garmindist