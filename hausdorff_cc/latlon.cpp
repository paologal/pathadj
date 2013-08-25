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
 * latlon.cpp
 *
 *  Created on: May 27, 2013
 *      Author: Paolo Galbiati
 */

#include <climits>

#include "platform_config.h"
#include "latlon.h"

const float latlon::RADIANS = M_PI / 180.0f;
const float latlon::EARTH_RADIUS = 6371.0f;

latlon::latlon() {
    // TODO(paolo) Auto-generated constructor stub
}

latlon::~latlon() {
    // TODO(paolo) Auto-generated destructor stub
}

float latlon::haversine(const path_point_t* p0, const path_point_t* p1) {
    float delta_lat = (p1->lat - p0->lat) * 0.5f;
    float delta_lon = (p1->lon - p0->lon) * 0.5f;
    float tmp0 = sinf(delta_lat);
    float tmp1 = sinf(delta_lon);

    float a = tmp0 * tmp0 + cosf(p0->lat) * cosf(p1->lat) * tmp1 * tmp1;
    float c = 2 * atan2f(sqrtf(a), sqrtf(1 - a));
    return latlon::EARTH_RADIUS * c;
}

float latlon::hausdorff(const adj_path& p0, const adj_path& p1) {
    float dist_01 = latlon::hausdorff_impl(p0, p1);
    float dist_10 = latlon::hausdorff_impl(p1, p0);

    if (dist_01 > dist_10) {
        return dist_01;
    } else {
        return dist_10;
    }
}

float latlon::hausdorff_impl(const adj_path& p0, const adj_path& p1) {
    uint32_t points0 = p0.get_points_number();
    uint32_t points1 = p1.get_points_number();
    float min_dist = INT_MAX;
    float max_dist = 0.0;
    float curr_dist = 0.0;

    for (uint32_t i = 0; i < points0; i++) {
        min_dist = INT_MAX;
        for (uint32_t j = 0; j < points1; j++) {
            curr_dist = latlon::haversine(p0.get_point(i), p1.get_point(j));
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
            }
        }
        if (min_dist > max_dist) {
            max_dist = min_dist;
        }
    }

    return max_dist;
}

