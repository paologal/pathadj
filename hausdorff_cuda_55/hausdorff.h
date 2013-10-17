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
 * hausdorff.h
 *
 *  Created on: May 27, 2013
 *      Author: Paolo Galbiati
 */

#ifndef HAUSDORFF_H_
#define HAUSDORFF_H_

#include "adj_path.h"
#include "adj_user_paths.h"

class gpu_device;

void hausdorffGPU(float* res, const path_point_t* p0, const path_point_t* p1, uint32_t points0, uint32_t points1);

class hausdorff {
 public:
    hausdorff();
    virtual ~hausdorff();

    void distance(const shared_ptr<gpu_device> gpu,
        const shared_ptr<adj_user_paths> paths,
        uint32_t j, 
        uint32_t i);
 
protected:
    float haversine(const path_point_t* p0, const path_point_t* p1);
    float maxmin_impl(shared_ptr<float> results, 
        uint32_t row,
        uint32_t col);
    virtual float distance_impl(const shared_ptr<gpu_device> gpu,
        const adj_path& p0,
        const adj_path& p1) = 0;

private:
    float distance_test(const adj_path& p0,
        const adj_path& p1);
    float distance_impl_test(const adj_path& p0,
        const adj_path& p1);

    static const float ERROR_LIMIT;
    static const float EARTH_RADIUS;
};

class hausdorff_gpu : public hausdorff {
 protected:
    float distance_impl(const shared_ptr<gpu_device> gpu,
        const adj_path& p0,
        const adj_path& p1);
};

class hausdorff_cpu : public hausdorff {
 protected:
    float distance_impl(const shared_ptr<gpu_device> gpu,
        const adj_path& p0,
        const adj_path& p1);
};

#endif /* HAUSDORFF_H_ */
