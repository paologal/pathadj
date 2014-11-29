#!/usr/bin/env python

import xml.etree.ElementTree as ET
import glob
import sys
import getopt
import os
import struct

import dateutil.parser
import datetime
import time

from math import *

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    m = 6371000 * c
    return m

def tcx_to_binary_file(file, user):
        print "Converting %s, user %s " % (file, user)
        data = []
        tree = ET.parse(file)
        paths = tree.findall(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Activity")
        print "    Found %d activities" % len(paths)
        for path in paths:
            prev_lat = 0.0
            prev_lon = 0.0
            total_dist = 0.0
            prev_garmin_dist = 0.0
            prev_timestamp = 0.0
            timestamp = 0.0
            count = 0
            garmin_dist = 0
            file_time = path.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Id")
            timeobj = dateutil.parser.parse(file_time.text)
            prev_timestamp = time.mktime(timeobj.timetuple())
            nodes = path.findall(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Trackpoint")
#            print "    time: %s, trackpoints %d" % (file_time.text, len(nodes))
            for n in nodes:
                time_elem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time")
                pos_elem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position")
                dist_elem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters")
                timeobj = dateutil.parser.parse(time_elem.text)
                timestamp = time.mktime(timeobj.timetuple())
                time_interval = timestamp - prev_timestamp
                if pos_elem is not None:
                    lat_elem = pos_elem.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LatitudeDegrees")
                    lon_elem = pos_elem.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LongitudeDegrees")
                    lat = float(lat_elem.text)
                    lon = float(lon_elem.text)
                    garmin_dist = (float(dist_elem.text))
            
                    if prev_lat and prev_lon:
                        leg_len = haversine(prev_lon, prev_lat, lon, lat)
                        total_dist = total_dist + leg_len
                        garmin_leg = garmin_dist - prev_garmin_dist
                        if leg_len:
                            garmin_leg_delta = ((garmin_leg - leg_len) / leg_len) * 100.0
                        else:
                            garmin_leg_delta = 0
                        if total_dist:
                            garmin_total_delta = ((garmin_dist - total_dist) / total_dist) * 100.0
                        else:
                            garmin_total_delta = 0
                        speed = 0
                        garmin_speed = 0
                        if time_interval:
                            speed = (leg_len / 1000) / (time_interval / 3600)
                            garmin_speed = (garmin_leg / 1000) / (time_interval / 3600)
#                        print "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % \
#                            (time_interval,
#                             lat, lon,
#                             leg_len, garmin_leg, garmin_leg_delta, 
#                             speed, garmin_speed,
#                             total_dist, garmin_dist, garmin_total_delta)
                    else:
                        total_dist = garmin_dist
                    prev_lat = lat
                    prev_lon = lon
                    prev_garmin_dist = garmin_dist
                    prev_timestamp = timestamp
                    
                    # print "    %d: lat: %f, lon: %f" % (count, lat, lon)
                    data.append(struct.pack('<ff', lat, lon))
                    count = count + 1
#                else:
#                    print "Skipping"
            print "    time: %s, trackpoints %d, total distance: %d (%d)m " % (file_time.text, len(nodes), total_dist, total_dist - garmin_dist)
            file_name = user + "_" + file_time.text + ".bin"
            output_file = open(file_name, 'wb')
            output_file.write(struct.pack('<I', count))
            for i in data:
                output_file.write(i)
            output_file.write(struct.pack('<II', count, 0x00ED00ED))
            output_file.close()
            
def tcx_report(file):
    prev_lat = 0.0
    prev_lon = 0.0
    total_dist = 0.0
    prev_garmin_dist = 0.0
    tree = ET.parse(file)
    nodes = tree.findall(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Trackpoint")
    print "#time\tlat\tlon\thleg\tgleg\tlegdiff\tdist\tgdist\tdistdiff"
    for n in nodes:
        time_elem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time")
        dist_elem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters")
        pos_elem = n.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position")
        if pos_elem is not None:
            lat_elem = pos_elem.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LatitudeDegrees")
            lon_elem = pos_elem.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LongitudeDegrees")
            lat = float(lat_elem.text)
            lon = float(lon_elem.text)
            garmin_dist = (float(dist_elem.text))
    
            if prev_lat and prev_lon:
                leg_len = haversine(prev_lon, prev_lat, lon, lat)
                total_dist = total_dist + leg_len
                garmin_leg = garmin_dist - prev_garmin_dist
                if leg_len:
                    garmin_leg_delta = ((garmin_leg - leg_len) / leg_len) * 100.0
                else:
                    garmin_leg_delta = 0
                if total_dist:
                    garmin_total_delta = ((garmin_dist - total_dist) / total_dist) * 100.0
                else:
                    garmin_total_delta = 0
            #timeobj = dateutil.parser.parse(time_elem.text)
            #timestamp = time.mktime(timeobj.timetuple())
                print "%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % \
                    (time_elem.text,
                     lat, lon,
                     leg_len, garmin_leg, garmin_leg_delta, 
                     total_dist, garmin_dist, garmin_total_delta)
            prev_lat = lat
            prev_lon = lon
            prev_garmin_dist = garmin_dist


def tcx_to_binary(tcxdir, user):
    print "Converting TCX files in " + tcxdir
    os.chdir(tcxdir)
    for files in glob.glob("*.TCX"):
        base = os.path.splitext(files)[0]
        os.rename(files, base + ".tcx")
    files_count = 0
    for file in glob.glob("*.tcx"):
        tcx_to_binary_file(file, user)
        #tcx_report(file)
        files_count = files_count + 1
    print "Converted %d files" % (files_count)
    
def main():
    if len(sys.argv) == 3:
        tcx_to_binary(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()