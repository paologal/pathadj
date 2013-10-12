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
 * gpu_cuda_device.cpp
 *
 *  Created on: May 18, 2013
 *      Author: Paolo Galbiati
 */

#include "gpu_cuda_device.h"

bool gpu_cuda_device::gpu_set_device(int32_t device) {
   	cudaError_t error_id = cudaSetDevice(device);
	if (error_id != cudaSuccess)
	{
		TRACE_ERROR("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return false;
	}

    return true;
}

void gpu_cuda_device::gpu_get_device_count(int32_t* device_count) {
    *device_count = 0;
	cudaError_t error_id = cudaGetDeviceCount(device_count);
	if (error_id != cudaSuccess)
	{
		TRACE_ERROR("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		return;
	}

	if (device_count == 0)
	{
		TRACE_ERROR("!!!!!No CUDA devices found!!!!!\n");
		return;
	}
}

void gpu_cuda_device::gpu_device_synchronize() {
    cudaDeviceSynchronize();
}

void gpu_cuda_device::gpu_device_reset() {
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t error_id = cudaDeviceReset();
    if (error_id != cudaSuccess) 
	{
        TRACE_ERROR("cudaDeviceReset failed!");
    }
}

bool gpu_cuda_device::gpu_device_free(void* device_data) {
    cudaFree(device_data);
    return true;
}

bool gpu_cuda_device::gpu_device_malloc(void** device_data, size_t size) {
    /* Allocate GPU buffer */
    cudaError_t cudaStatus = cudaMalloc(device_data, size);
    if (cudaStatus != cudaSuccess) 
	{
        TRACE_ERROR("cudaMalloc failed!");
        return false;
    }

    return true;
}

bool gpu_cuda_device::gpu_memcpy(void* dst, const void* src, size_t count, gpu_memcpy_kind_t kind) {
    // Copy result from device to host
    cudaError_t cudaStatus = cudaMemcpy(dst, src, count, static_cast<cudaMemcpyKind>(kind));
    if (cudaStatus != cudaSuccess) 
	{
        TRACE_ERROR("cudaMemcpy returned error code %d, line(%d)\n", cudaStatus, __LINE__);
        return false;
    }

    return true;
}
