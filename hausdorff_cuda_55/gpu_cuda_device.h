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
 * gpu_cuda_device.h
 *
 *  Created on: Oct 12, 2013
 *      Author: Paolo Galbiati
 */

#ifndef GPU_CUDA_DEVICE_H_
#define GPU_CUDA_DEVICE_H_

#include "gpu_device.h"

#ifdef HAUSDORFF_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif /* HAUSDORFF_CUDA */

class gpu_cuda_device : public gpu_device
{
public:
    gpu_cuda_device(void) {};
    virtual ~gpu_cuda_device(void) {};

    bool gpu_set_device(int32_t device);
    void gpu_get_device_count(int32_t* device_count);
    void gpu_device_synchronize();
    void gpu_device_reset();
    bool gpu_device_free(void* device_data);
    bool gpu_device_malloc(void** device_data, size_t size);
    bool gpu_memcpy(void* dst, const void* src, size_t count, gpu_memcpy_kind_t kind);
};

#endif /* GPU_CUDA_DEVICE_H_ */

