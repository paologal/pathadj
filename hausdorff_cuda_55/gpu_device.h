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
 * gpu_device.h
 *
 *  Created on: Oct 12, 2013
 *      Author: Paolo Galbiati
 */

#ifndef GPU_DEVICE_H_
#define GPU_DEVICE_H_

#include "platform_config.h"

typedef enum gpu_memcpy_kind
{
    gpu_memcpy_host_to_host          =   0,      /**< Host   -> Host */
    gpu_memcpy_host_to_device        =   1,      /**< Host   -> Device */
    gpu_memcpy_device_to_host        =   2,      /**< Device -> Host */
    gpu_memcpy_device_to_device      =   3,      /**< Device -> Device */
    gpu_memcpy_default               =   4       /**< Default based unified virtual address space */
} gpu_memcpy_kind_t;

class gpu_device
{
public:
    gpu_device(void) {};
    virtual ~gpu_device(void) {};

    virtual bool gpu_set_device(int32_t device) = 0;
    virtual void gpu_get_device_count(int32_t* device_count) = 0;
    virtual void gpu_device_reset() = 0;
    virtual void gpu_device_synchronize() = 0;
    virtual bool gpu_device_free(void* device_data) = 0;
    virtual bool gpu_device_malloc(void** device_data, size_t size) = 0;
    virtual bool gpu_memcpy(void* dst, const void* src, size_t count, gpu_memcpy_kind_t kind) = 0;
};

#endif /* GPU_DEVICE_H_ */

