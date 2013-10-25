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
* distance_thread.h
*
*  Created on: Oct 12, 2013
*      Author: Paolo Galbiati
*/

#ifndef DISTANCE_THREAD_H_
#define DISTANCE_THREAD_H_

#include "platform_config.h"

class adj_user_paths;
class gpu_device;

class distance_thread
{
public:
    distance_thread();
    ~distance_thread(void);

    static void compute_subset(const shared_ptr<gpu_device> gpu,
        const shared_ptr<adj_user_paths> paths,
        uint32_t size,
        uint32_t start,
        uint32_t step);
};  

#endif /* DISTANCE_THREAD_H_ */

