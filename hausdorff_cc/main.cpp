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

#include <thread>

#include "adj_path.h"
#include "adj_user_paths.h"
#include "latlon.h"

using std::thread;

#define SCAN_DIR

void compute_subset(const shared_ptr<adj_user_paths> paths, uint32_t size,
                    uint32_t start, uint32_t step) {
    for (uint32_t j = start; j < size / 2; j += step) {
        TRACE_DEBUG("%d, Computing column %d, %d elements\n",
                    start,
                    j,
                    size - (j + 1));
        for (uint32_t i = j + 1; i < size; ++i) {
            latlon::hausdorff_distance(paths, j, i);
        }
        if (j != size - j - 2) {
            TRACE_DEBUG("%d, Computing column %d, %d elements\n",
                        start,
                        size - j - 2,
                        size - (size - j - 1));
            for (uint32_t i = size - j - 1; i < size; ++i) {
                latlon::hausdorff_distance(paths, size - j - 2, i);
            }
        }
    }
}

int main(int argc, char** argv) {
#ifdef SCAN_FILE
    if (argc < 3) {
        printf("Usage: %s file directory\n", argv[0]);
        return 0;
    }
    const shared_ptr<adj_path> path0(new adj_path(argv[1]));
    if (false == path0->load_file()) {
        return EXIT_FAILURE;
    }

    adj_user_paths paths1(argv[2]);
    paths1.load_paths();
    int32_t size1 = paths1.get_paths_number();

#pragma omp parallel for
    for (int32_t i = 0; i < size1; ++i) {
        const shared_ptr<adj_path> path1(paths1.get_path(i));
        float dist = latlon::hausdorff(*path0, *path1);
        printf("%d,%s,%s,Distance: %f km\n", i,
                path0->get_path_name().c_str(), path1->get_path_name().c_str(),
                dist);
    }

    path0->reset();
    paths1.reset();
#endif /* SCAN_FILE */

#ifdef SCAN_DIR
    if (argc < 2) {
        printf("Usage: %s directories\n", argv[0]);
        return 0;
    }

    vector<thread> threads;
    uint32_t max_threads = 4 * thread::hardware_concurrency();
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
#endif /* SCAN_DIR */

    return EXIT_SUCCESS;
}

