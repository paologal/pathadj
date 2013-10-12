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

template <int BLOCK_SIZE> __global__ void
    hausdorffCUDA(float* res, path_point_t* p0, path_point_t* p1, uint32_t points0, uint32_t points1)
{
    const float EARTH_RADIUS = 6371.0f;

    // Block index
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;

    // Thread index
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    uint32_t  x = bx * BLOCK_SIZE + tx;
    uint32_t  y = by * BLOCK_SIZE + ty;

    if ((x < points0) && (y < points1)) {
        float delta_lat = (p1[y].lat - p0[x].lat) * 0.5f;
        float delta_lon = (p1[y].lon - p0[x].lon) * 0.5f;
        float tmp0 = __sinf(delta_lat);
        float tmp1 = __sinf(delta_lon);

        float a = tmp0 * tmp0 + __cosf(p0[x].lat) * __cosf(p1[y].lat) * tmp1 * tmp1;
        float c = 2 * atan2f(sqrtf(a), sqrtf(1 - a));
 
        *(res + x * points1 + y) = EARTH_RADIUS * c;
    }
}

void hausdorffGPU(float* res, path_point_t* p0, path_point_t* p1, uint32_t points0, uint32_t points1) {

    const uint32_t BLOCK_SIZE = 32;

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

    dim3 grid(dim_x, dim_y);


    hausdorffCUDA<BLOCK_SIZE><<< grid, threads >>>(res, p0, p1, points0, points1);
}
