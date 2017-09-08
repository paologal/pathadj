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
* hausdorff.cu
*
*  Created on: Sep 22, 2013
*      Author: Paolo Galbiati
*/

#include "platform_config.h"
#include "adj_path.h"
#include "hausdorff.h"

template <uint32_t BLOCK_SIZE, uint32_t ITERATIONS_PER_THREAD> __global__ void
    hausdorffCUDA(float* res, const path_point_t* p0, const path_point_t* p1, uint32_t points0, uint32_t points1)
{
    const float EARTH_RADIUS = 6371.0f;

    // Block index
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;

    // Thread index
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    uint32_t  x = bx * BLOCK_SIZE + tx;
    uint32_t  y = ITERATIONS_PER_THREAD * (by * BLOCK_SIZE + ty);

#pragma unroll
    for (uint32_t i = y; i < y + ITERATIONS_PER_THREAD; i++) {
        if ((x < points0) && (i < points1)) {
        	float p0xlat = p0[x].lat;
        	float p0xlon = p0[x].lon;
        	float p1ilat = p1[i].lat;
        	float p1ilon = p1[i].lon;

            float delta_lat = (p1ilat - p0xlat) * 0.5f;
            float delta_lon = (p1ilon - p0xlon) * 0.5f;
            float tmp0 = __sinf(delta_lat);
            float tmp1 = __sinf(delta_lon);

            float a = tmp0 * tmp0 + __cosf(p0xlat) * __cosf(p1ilat) * tmp1 * tmp1;
            float c = 2 * atan2f(sqrtf(a), sqrtf(1 - a));
 
            *(res + x * points1 + i) = EARTH_RADIUS * c;
        }
    }
}

void hausdorffGPU(float* res, const path_point_t* p0, const path_point_t* p1, uint32_t points0, uint32_t points1) {

    const uint32_t BLOCK_SIZE = 32;
    const uint32_t ITERATIONS_PER_THREAD = 1;

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    uint32_t dim_x = points0 / threads.x;
    uint32_t dim_y = points1 / threads.y;
    if (points0 % threads.x) {
        dim_x += 1;
    }
    if (points1 % threads.y) {
        dim_y += 1;
    }

    if (dim_y % ITERATIONS_PER_THREAD == 0) {
        dim_y = dim_y / ITERATIONS_PER_THREAD;
    }
    else {
        dim_y = dim_y / ITERATIONS_PER_THREAD + 1;
    }

    dim3 grid(dim_x, dim_y);


    hausdorffCUDA<BLOCK_SIZE, ITERATIONS_PER_THREAD><<< grid, threads >>>(res, p0, p1, points0, points1);
}



float hausdorff_gpu::distance_impl(const shared_ptr<gpu_device> gpu,
                        const adj_path& p0,
                        const adj_path& p1) {

	path_point_t* device_data0;
	path_point_t* device_data1;
    uint32_t points0 = p0.get_points_number();
    uint32_t points1 = p1.get_points_number();
    shared_ptr<float> results(new float[points0 * points1]);
    float* results_ptr = results.get();
    float dist = 0.0f;

    /* Allocate GPU buffer */
    if (false
            == gpu->gpu_device_malloc((void**) &device_data0,
            		points0 * (sizeof(path_point_t)))) {
        return dist;
    }
    // Copy path from host memory to GPU buffer.
    if (false
            == gpu->gpu_memcpy(device_data0, p0.get_point(0),
            		points0 * (sizeof(path_point_t)),
                               gpu_memcpy_host_to_device)) {
        return dist;
    }

    /* Allocate GPU buffer */
    if (false
            == gpu->gpu_device_malloc((void**) &device_data1,
            		points1 * (sizeof(path_point_t)))) {
        return dist;
    }
    // Copy path from host memory to GPU buffer.
    if (false
            == gpu->gpu_memcpy(device_data1, p1.get_point(0),
            		points1 * (sizeof(path_point_t)),
                               gpu_memcpy_host_to_device)) {
        return dist;
    }

    /* Allocate GPU buffer */
	uint32_t data_size = (points0 * points1) * sizeof(float);
	float* result_buffer = nullptr;
    
    if (false == gpu->gpu_device_malloc((void**)&result_buffer, data_size)) 
    {
       return dist;
    }

    hausdorffGPU(result_buffer, device_data0, device_data1, points0, points1);

    gpu->gpu_device_synchronize();

    // Copy result from device to host
    if (true == gpu->gpu_memcpy(results.get(), result_buffer, data_size, gpu_memcpy_device_to_host)) 
	{
        dist = maxmin_impl(results.get(), points0, points1);
    }

    gpu->gpu_device_free(device_data0);
    gpu->gpu_device_free(device_data1);
    gpu->gpu_device_free(result_buffer);

    return dist;
}
