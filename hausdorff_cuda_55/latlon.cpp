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

#include "platform_config.h"
#include "latlon.h"

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

float latlon::hausdorff_test(const adj_path& p0, const adj_path& p1) {
    float dist_01 = latlon::hausdorff_impl_test(p0, p1);
    float dist_10 = latlon::hausdorff_impl_test(p1, p0);

    if (dist_01 > dist_10) {
        return dist_01;
    } else {
        return dist_10;
    }
}

float latlon::hausdorff_impl_test(const adj_path& p0, const adj_path& p1) {
    uint32_t points0 = p0.get_points_number();
    uint32_t points1 = p1.get_points_number();
    float min_dist = numeric_limits<float>::max();
    float max_dist = 0.0;
    float curr_dist = 0.0;

    for (uint32_t i = 0; i < points0; i++) {
        min_dist = numeric_limits<float>::max();
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

void latlon::hausdorff_distance(const shared_ptr<adj_user_paths> paths,
                                uint32_t j, uint32_t i) {
    const shared_ptr<adj_path> path0(paths->get_path(j));
    const shared_ptr<adj_path> path1(paths->get_path(i));
    float dist = latlon::hausdorff(*path0, *path1);
#ifdef UNIT_TEST
    float dist_test = latlon::hausdorff_test(*path0, *path1);
    uint32_t* p_dist = (uint32_t*)&dist;
    uint32_t* p_dist_test = (uint32_t*)&dist_test;
    if (dist != dist_test) {
        TRACE_ERROR("ERROR %x\n", *p_dist ^ *p_dist_test);
    }
    assert((*p_dist ^ *p_dist_test) < 4);
#endif /* UNIT_TEST */
    TRACE_INFO("%d,%d,%s,%s,Distance: %f km\n", j, i,
           path0->get_path_name().c_str(), path1->get_path_name().c_str(),
           dist);
}

float latlon::hausdorff(const adj_path& p0, const adj_path& p1) {
    uint32_t points0 = p0.get_points_number();
    uint32_t points1 = p1.get_points_number();
	uint32_t data_size = (points0 * points1) * sizeof(float);

	float* result_buffer = nullptr;
    float* results = new float[points0 * points1];

	/* Allocate GPU buffer */
    cudaError_t cudaStatus = cudaMalloc((void**)&result_buffer, data_size);
    if (cudaStatus != cudaSuccess) 
	{
        TRACE_ERROR("cudaMalloc failed!");
        return 0;
    }

    hausdorffGPU(result_buffer, p0.get_device_data(), p1.get_device_data(), points0, points1);

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaStatus = cudaMemcpy(results, result_buffer, data_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
	{
        TRACE_ERROR("cudaMemcpy returned error code %d, line(%d)\n", cudaStatus, __LINE__);
        return 0;
    }

    float dist = latlon::hausdorff_impl(results, points0, points1);

    cudaFree(result_buffer);
    delete[] results;

    return dist;
}

float latlon::hausdorff_impl(const float* results, uint32_t row, uint32_t col) {
    float dist_01;
    float min_dist = numeric_limits<float>::max();;
    float max_dist = 0.0;
    float curr_dist = 0.0;

    for (uint32_t i = 0; i < row; i++) {
        min_dist = numeric_limits<float>::max();;
        for (uint32_t j = 0; j < col; j++) {
            curr_dist = results[i * col + j];
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
            }
        }
        if (min_dist > max_dist) {
            max_dist = min_dist;
        }
    }

    dist_01 = max_dist;
    max_dist = 0.0;

    for (uint32_t i = 0; i < col; i++) {
        min_dist = numeric_limits<float>::max();;
        for (uint32_t j = 0; j < row; j++) {
            curr_dist = results[i + j * col];
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
            }
        }
        if (min_dist > max_dist) {
            max_dist = min_dist;
        }
    }

    if (dist_01 > max_dist) {
        return dist_01;
    } else {
        return max_dist;
    }
}
