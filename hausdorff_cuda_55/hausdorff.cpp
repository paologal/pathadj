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
 * hausdorff.cpp
 *
 *  Created on: May 27, 2013
 *      Author: Paolo Galbiati
 */

#include "gpu_device.h"

#include "platform_config.h"
#include "hausdorff.h"

const float hausdorff::ERROR_LIMIT = 0.01f;
const float hausdorff::EARTH_RADIUS = 6371.0f;

hausdorff::hausdorff() {
    // TODO(paolo) Auto-generated constructor stub
}

hausdorff::~hausdorff() {
    // TODO(paolo) Auto-generated destructor stub
}

float hausdorff::haversine(const path_point_t* p0, const path_point_t* p1) {
    float delta_lat = (p1->lat - p0->lat) * 0.5f;
    float delta_lon = (p1->lon - p0->lon) * 0.5f;
    float tmp0 = sinf(delta_lat);
    float tmp1 = sinf(delta_lon);

    float a = tmp0 * tmp0 + cosf(p0->lat) * cosf(p1->lat) * tmp1 * tmp1;
    float c = 2 * atan2f(sqrtf(a), sqrtf(1 - a));
    return hausdorff::EARTH_RADIUS * c;
}

float hausdorff::distance_test(const adj_path& p0, const adj_path& p1) {
    float dist_01 = hausdorff::distance_impl_test(p0, p1);
    float dist_10 = hausdorff::distance_impl_test(p1, p0);

    if (dist_01 > dist_10) {
        return dist_01;
    } else {
        return dist_10;
    }
}

float hausdorff::distance_impl_test(const adj_path& p0, const adj_path& p1) {
    uint32_t points0 = p0.get_points_number();
    uint32_t points1 = p1.get_points_number();
    float min_dist = numeric_limits<float>::max();
    float max_dist = 0.0;
    float curr_dist = 0.0;

    for (uint32_t i = 0; i < points0; i++) {
        min_dist = numeric_limits<float>::max();
        for (uint32_t j = 0; j < points1; j++) {
            curr_dist = hausdorff::haversine(p0.get_point(i), p1.get_point(j));
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

void hausdorff::distance(const shared_ptr<gpu_device> gpu,
                                const shared_ptr<adj_user_paths> paths,
                                uint32_t j, uint32_t i) {
    const shared_ptr<adj_path> path0(paths->get_path(j));
    const shared_ptr<adj_path> path1(paths->get_path(i));
    float dist = hausdorff::distance_impl(gpu, *path0, *path1);
#ifdef UNIT_TEST
    float dist_test = hausdorff::distance_test(*path0, *path1);
    if (abs(dist - dist_test) > hausdorff::ERROR_LIMIT) {
        TRACE_ERROR("ERROR %f\n", abs(dist - dist_test));
    }
    assert(abs (dist - dist_test) <= hausdorff::ERROR_LIMIT);
#endif /* UNIT_TEST */
    TRACE_INFO("%d,%d,%s,%s,Distance: %f km\n", j, i,
           path0->get_path_name().c_str(), path1->get_path_name().c_str(),
           dist);
}

float hausdorff::distance_impl(const shared_ptr<gpu_device> gpu,
                        const adj_path& p0,
                        const adj_path& p1) {
    uint32_t points0 = p0.get_points_number();
    uint32_t points1 = p1.get_points_number();
    shared_ptr<float> results(new float[points0 * points1]);
    float* results_ptr = results.get();
    float dist = 0.0f;

	/* Allocate GPU buffer */
	uint32_t data_size = (points0 * points1) * sizeof(float);
	float* result_buffer = nullptr;
    
    if (false == gpu->gpu_device_malloc((void**)&result_buffer, data_size)) 
	{
       return dist;
    }

#ifdef HAUSDORFF_CUDA
    hausdorffGPU(result_buffer, p0.get_device_data(), p1.get_device_data(), points0, points1);
#else
    for (uint32_t i = 0; i < points0; i++) {
        for (uint32_t j = 0; j < points1; j++) {
            results_ptr[i * points1 + j] = hausdorff::haversine(p0.get_point(i),
                                                         p1.get_point(j));
        }
    }
#endif

    gpu->gpu_device_synchronize();

    // Copy result from device to host
    if (true == gpu->gpu_memcpy(results.get(), result_buffer, data_size, gpu_memcpy_device_to_host)) 
	{
        dist = hausdorff::maxmin_impl(results, points0, points1);
    }

    gpu->gpu_device_free(result_buffer);

    return dist;
}

float hausdorff::maxmin_impl(shared_ptr<float> results, uint32_t row, uint32_t col) {
    float dist_01;
    float min_dist = numeric_limits<float>::max();;
    float max_dist = 0.0;
    float curr_dist = 0.0;
    float* results_ptr = results.get();

    for (uint32_t i = 0; i < row; i++) {
        min_dist = numeric_limits<float>::max();;
        for (uint32_t j = 0; j < col; j++) {
            curr_dist = results_ptr[i * col + j];
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
            curr_dist = results_ptr[i + j * col];
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
