/*
 * Copyright (C) 2013  Azlos Corporation
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

/*
 * adj_path.cpp
 *
 *  Created on: May 18, 2013
 *      Author: Paolo Galbiati
 */

#include "platform_config.h"
#include "adj_path.h"
#include "hausdorff.h"

adj_path::adj_path(const shared_ptr<gpu_device> gpu, const string& file)
        : gpu(gpu),
          path_data(nullptr),
          file_name(file),
          cumulated_distance(0.0) {
    memset(&mean_point, 0, sizeof(mean_point));
    memset(&median_point, 0, sizeof(median_point));
    memset(&path, 0, sizeof(path));
}

adj_path::~adj_path() {
    reset();
}

void adj_path::reset() {
    file_name.empty();

    if (nullptr != path_data) {
        delete[] path_data;
        path_data = nullptr;
    }

    memset(&path, 0, sizeof(path));
}

bool adj_path::verify_file_size(uint32_t file_size) {
    uint32_t points = path.points;
    uint32_t check = ((file_size - sizeof(path.points)
            - sizeof(path.points_check) - sizeof(path.checksum)) >> 3);

    if (points == 0) {
        /* Error handling */
        TRACE_ERROR("Empty file!\n");
        return false;
    }

    if (points != check) {
        /* Error handling */
        TRACE_ERROR("Invalid coordinates %d, expected %d\n", check, points);
        return false;
    }

    return true;
}

bool adj_path::verify_file() {
    if (path.points_check != path.points) {
        /* Error handling */
        TRACE_ERROR("Invalid coordinates checksum %d, expected %d\n",
                    path.points_check, path.points);
        return false;
    }

    if (PATH_FILE_CHECKSUM != path.checksum) {
        /* Error handling */
        TRACE_ERROR("Invalid file checksum %d, expected %d\n", path.checksum,
                    PATH_FILE_CHECKSUM);
        return false;
    }

    return true;
}

bool adj_path::load_file() {
    struct stat info;
    if (0 != stat(file_name.c_str(), &info)) {
        TRACE_ERROR("Cannot stat %s\n", file_name.c_str());
        reset();
        return false;
    }

    path_data = new uint8_t[info.st_size];
    memset(path_data, 0, info.st_size);
    if (nullptr == path_data) {
        /* Error handling */
        TRACE_ERROR("Cannot allocate %d bytes\n", (uint32_t) info.st_size);
        reset();
        return false;
    }

    FILE *fp = fopen(file_name.c_str(), "rb");
    if (nullptr == fp) {
        /* Error handling */
        TRACE_ERROR("Cannot open %s\n", file_name.c_str());
        reset();
        return false;
    }

    /* Try to read a single block of info.st_size bytes */
    size_t blocks_read = fread(path_data, info.st_size, 1, fp);
    fclose(fp);
    if (1 != blocks_read) {
        /* Error handling */
        TRACE_ERROR("Cannot read %d bytes from %s\n", (uint32_t) info.st_size,
                    file_name.c_str());
        reset();
        return false;
    }

    path.points = *reinterpret_cast<uint32_t*>(path_data);
    if (true != verify_file_size((uint32_t) info.st_size)) {
        TRACE_ERROR("File %s: invalid file size %d\n", file_name.c_str(),
                    (uint32_t) info.st_size);
        reset();
        return false;
    }

    path.coordinates = reinterpret_cast<path_point_t*>(path_data
            + sizeof(path.points));
    uint8_t* checks = reinterpret_cast<uint8_t*>(path.coordinates)
            + path.points * sizeof(path_point_t);
    path.points_check = *reinterpret_cast<uint32_t*>(checks);
    path.checksum = *reinterpret_cast<uint32_t*>(checks
            + sizeof(path.points_check));

    if (true != verify_file()) {
        TRACE_ERROR("Invalid file %s\n", file_name.c_str());
        reset();
        return false;
    }

    dump();
    init();

    return true;
}

void adj_path::dump() {
	TRACE_VERBOSE("Filename: %s\n", file_name.c_str());
    for (uint32_t i = 0; i < path.points; ++i) {
    	TRACE_VERBOSE("Point %d. Latitude %f, Longitude %f\n", i,
                    path.coordinates[i].lat, path.coordinates[i].lon);
    }
}

void adj_path::init() {
    float sum_lat = 0.0;
    float sum_lon = 0.0;
    path_point_t* p0;

    cumulated_distance = 0.0;
    if (path.points) {
        for (uint32_t i = 0; i < path.points - 1; ++i) {
            p0 = get_point(i);

            sum_lat += p0->lat;
            sum_lon += p0->lon;
#ifdef DEBUG_DUMP
#ifdef HAUSDORFF_CUDA
            const shared_ptr<hausdorff_gpu> algo(new hausdorff_gpu);
#else
            const shared_ptr<hausdorff_cpu> algo(new hausdorff_cpu);
#endif
            cumulated_distance += algo->haversine(p0, get_point(i + 1));
#endif
        }
    }

    if (path.points % 2) {
        path_point_t* p0 = get_point((path.points + 1) / 2);
        path_point_t* p1 = get_point((path.points) / 2);
        median_point.lat = (p0->lat + p1->lat) / 2;
        median_point.lon = (p0->lon + p1->lon) / 2;
    } else {
        path_point_t* p = get_point((path.points) / 2);
        median_point.lat = p->lat;
        median_point.lon = p->lon;
    }

    mean_point.lat = sum_lat / path.points;
    mean_point.lon = sum_lon / path.points;

    TRACE_DEBUG("Filename: %s\n", file_name.c_str());
    TRACE_DEBUG("Points %d. Total distance %f.\n", path.points,
                cumulated_distance);
    TRACE_DEBUG("Mean point(%f, %f) radians\n", mean_point.lat, mean_point.lon);
    TRACE_DEBUG("Median point(%f, %f) radians\n", median_point.lat,
                median_point.lon);
}
