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
 * latlon.h
 *
 *  Created on: May 27, 2013
 *      Author: Paolo Galbiati
 */

#ifndef LATLON_H_
#define LATLON_H_

#include "adj_path.h"
#include "adj_user_paths.h"

class latlon {
 public:
    latlon();
    virtual ~latlon();

    static float haversine(const path_point_t* p0, const path_point_t* p1);
    static void hausdorff_distance(const shared_ptr<adj_user_paths> paths,
                                   uint32_t j, uint32_t i);
 private:
    static float hausdorff(const adj_path& p0, const adj_path& p1);
    static float hausdorff_test(const adj_path& p0, const adj_path& p1);

    static const float RADIANS;
    static const float EARTH_RADIUS;

    static float hausdorff_impl(const float* results, uint32_t row,
                                uint32_t col);
    static float hausdorff_impl_test(const adj_path& p0, const adj_path& p1);
};


#endif /* LATLON_H_ */
