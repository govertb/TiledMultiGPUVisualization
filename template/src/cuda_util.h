/*
 ==============================================================================

 cuda_util.h
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef cuda_util_h
#define cuda_util_h

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>

// call a CUDA function, check returned error code
// due to https://stackoverflow.com/a/14038590
#define cuda_check_error(ans) { assert_d((ans), __FILE__, __LINE__); }
inline void assert_d(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"Error: %s (CUDA Error %d), %s:%d\n", cudaGetErrorString(code), code, file, line);
        if (abort) exit(code);
    }
}

// return whether cuda device is available
// implicitly creates a context for it, if so ?
bool request_cuda_device(int gpu_id)
{
    cudaSetDevice(gpu_id);
    return cudaDeviceSynchronize() == cudaSuccess;
}

// try to claim at least `min_devices' and at most `max_devices'
std::vector<int> claim_cuda_devices(int min_devices, int max_devices)
{
    int device_count;
    cuda_check_error(cudaGetDeviceCount(&device_count));

    if (device_count < min_devices)
    {
        printf("Error: found %d CUDA device, %d requested.\n",
               device_count, min_devices);
        exit(EXIT_FAILURE);
    }

    // try to claim desired number of devices
    std::vector<int> claimed_devices;
    for (int gpu_id = 0; gpu_id < device_count; gpu_id++)
    {
      if(request_cuda_device(gpu_id)) claimed_devices.push_back(gpu_id);
      if(claimed_devices.size() == max_devices) break;
    }

    if (claimed_devices.size() < min_devices)
    {
        printf("Error: %d devices requested, only %u are free.\n",
                min_devices, claimed_devices.size());
        exit(EXIT_FAILURE);
    }

    return claimed_devices;
}

// try to claim exactly `num_devices'
std::vector<int> claim_cuda_devices(int num_devices)
{
    return claim_cuda_devices(num_devices, num_devices);
}

#endif
