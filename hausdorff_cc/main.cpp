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

#include "cassert"

#include "adj_path.h"
#include "adj_user_paths.h"
#include "latlon.h"

#define SCAN_DIR

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

    for (int32_t i = 1; i < argc; ++i) {
        adj_user_paths paths0(argv[i]);
        paths0.load_paths();
        int32_t size0 = paths0.get_paths_number();

        if (size0 > 1) {
            uint32_t count = 0;
            for (int32_t j = 0; j < size0; ++j) {
#pragma omp parallel for
                for (int32_t i = j + 1; i < size0; ++i) {
                    const shared_ptr<adj_path> path0(paths0.get_path(j));
                    const shared_ptr<adj_path> path1(paths0.get_path(i));
                    float dist = latlon::hausdorff(*path0, *path1);
//                    float dist_test = latlon::hausdorff_test(*path0, *path1);
//                    uint32_t* p_dist = (uint32_t*)&dist;
//                    uint32_t* p_dist_test = (uint32_t*)&dist_test;
//                    if (dist != dist_test) {
//                        printf("ERROR %x\n", *p_dist ^ *p_dist_test);
//                    }
//                    assert((*p_dist ^ *p_dist_test) < 4);
                    printf("%d,%d,%d,%s,%s,Distance: %f km\n", count++, j, i,
                            path0->get_path_name().c_str(),
                            path1->get_path_name().c_str(), dist);
                }
            }
        }

        paths0.reset();
    }
#endif /* SCAN_DIR */

    return EXIT_SUCCESS;
}

