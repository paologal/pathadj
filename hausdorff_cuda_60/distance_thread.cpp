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
 * distance_thread.cpp
 *
 *  Created on: May 18, 2013
 *      Author: Paolo Galbiati
 */

#include "distance_thread.h"
#include "hausdorff.h"

void distance_thread::compute_subset(const shared_ptr<gpu_device> gpu,
                                     const shared_ptr<adj_user_paths> paths,
                                     uint32_t size, uint32_t start,
                                     uint32_t step) {

#ifdef HAUSDORFF_CUDA
    const shared_ptr<hausdorff_gpu> algo(new hausdorff_gpu);
#else
    const shared_ptr<hausdorff_cpu> algo(new hausdorff_cpu);
#endif
    uint32_t count = 0;

    if (false == gpu->gpu_set_device(0)) {
        TRACE_ERROR("GPU set device failed!");
        return;
    }

    for (uint32_t j = start; j < size / 2; j += step) {
        TRACE_DEBUG("%d, Computing column %d, %d elements\n", start, j,
                    size - (j + 1));
        for (uint32_t i = j + 1; i < size; ++i) {
            algo->distance(gpu, paths, j, i);
            ++count;
        }
        if (j != size - j - 2) {
            TRACE_DEBUG("%d, Computing column %d, %d elements\n", start,
                        size - j - 2, size - (size - j - 1));
            for (uint32_t i = size - j - 1; i < size; ++i) {
                algo->distance(gpu, paths, size - j - 2, i);
                ++count;
            }
        }
    }

    TRACE_DEBUG("%d, Computed %d elements\n", start, count);
}
