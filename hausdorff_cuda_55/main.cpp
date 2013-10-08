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
* main.cc
*
*  Created on: May 19, 2013
*      Author: Paolo Galbiati
*/

#include "adj_path.h"
#include "adj_user_paths.h"
#include "latlon.h"

void compute_subset(const shared_ptr<adj_user_paths> paths, uint32_t size,
                    uint32_t start, uint32_t step) {
	cudaError_t error_id = cudaSetDevice(0);
	if (error_id != cudaSuccess)
	{
		TRACE_ERROR("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

    for (uint32_t j = start; j < size / 2; j += step) {
        TRACE_DEBUG("%d, Computing column %d, %d elements\n", start, j, size - (j + 1));
        for (uint32_t i = j + 1; i < size; ++i) {
            latlon::hausdorff_distance(paths, j, i);
        }
        if (j != size - j - 2) {
            TRACE_DEBUG("%d, Computing column %d, %d elements\n", start, size - j - 2, size - (size - j - 1));
            for (uint32_t i = size - j - 1; i < size; ++i) {
                latlon::hausdorff_distance(paths, size - j - 2, i);
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        TRACE_ERROR("Usage: %s directories\n", argv[0]);
        return 0;
    }

	int32_t deviceCount;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess)
	{
		TRACE_ERROR("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		return EXIT_FAILURE;
	}

	if (deviceCount == 0)
	{
		TRACE_ERROR("!!!!!No CUDA devices found!!!!!\n");
		return EXIT_FAILURE;
	}

    vector<thread> threads;
    uint32_t max_threads = 16 * thread::hardware_concurrency();
    for (int32_t i = 1; i < argc; ++i) {
        const shared_ptr<adj_user_paths> paths0(new adj_user_paths(argv[i]));
        paths0->load_paths();
        uint32_t size0 = paths0->get_paths_number();

        uint32_t num_of_threads = max_threads;
        if ((size0 >> 1) < num_of_threads) {
            num_of_threads = size0 >> 1;
        }

        if (size0 > 1) {
            for (uint32_t j = 0; j < num_of_threads; ++j) {
                threads.push_back(
                        thread(compute_subset, paths0, size0, j,
                               num_of_threads));
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }

        paths0->reset();
    }

	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    error_id = cudaDeviceReset();
    if (error_id != cudaSuccess) 
	{
        TRACE_ERROR("cudaDeviceReset failed!");
        return EXIT_FAILURE;
    }

	return EXIT_SUCCESS;
}


