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
 * main.cpp
 *
 *  Created on: May 19, 2013
 *      Author: Paolo Galbiati
 */

#include "gpu_cuda_device.h"
#include "distance_thread.h"
#include "adj_path.h"
#include "adj_user_paths.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        TRACE_ERROR("Usage: %s directories\n", argv[0]);
        return 0;
    }

#ifdef HAUSDORFF_CUDA
    const shared_ptr<gpu_device> gpu(new gpu_cuda_device);
#else
    const shared_ptr<gpu_device> gpu(new gpu_device);
#endif

    int32_t device_count;
    gpu->gpu_get_device_count(&device_count);
    if (device_count == 0) {
        return EXIT_FAILURE;
    }

    vector<thread> threads;
    uint32_t max_threads = 8 * thread::hardware_concurrency();
    for (int32_t i = 1; i < argc; ++i) {
        const shared_ptr<adj_user_paths> paths0(
                new adj_user_paths(gpu, argv[i]));
        paths0->load_paths();
        uint32_t size0 = paths0->get_paths_number();

        uint32_t num_of_threads = max_threads;
        if ((size0 >> 1) < num_of_threads) {
            num_of_threads = size0 >> 1;
        }

        if (size0 > 1) {
            for (uint32_t j = 0; j < num_of_threads; ++j) {
                threads.push_back(
                        thread(distance_thread::compute_subset, gpu, paths0,
                               size0, j, num_of_threads));
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }

        paths0->reset();
    }

    gpu->gpu_device_reset();

    return EXIT_SUCCESS;
}

