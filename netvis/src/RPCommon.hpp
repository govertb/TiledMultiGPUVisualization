/*
 ==============================================================================

 RPCommon.hpp
 Copyright © 2016, 2017, 2018  G. Brinkmann

 This file is part of graph_viewer.

 graph_viewer is free software: you can redistribute it and/or modify
 it under the terms of version 3 of the GNU Affero General Public License as
 published by the Free Software Foundation.

 graph_viewer is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with graph_viewer.  If not, see <https://www.gnu.org/licenses/>.

 ==============================================================================
*/

#ifndef RPCommonUtils_hpp
#define RPCommonUtils_hpp

#ifdef __NVCC__
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

#define cudaCatchError(ans) { assert_d((ans), __FILE__, __LINE__); }
inline void assert_d(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"error: (GPUassert) %s (error %d). %s:%d\n", cudaGetErrorString(code), code, file, line);
        if (abort) exit(code);
    }
}

#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Claim available CUDA devices.
bool request_cuda_device(int gpu_id);
void claim_cuda_devices(std::vector<int> requested_devices);
std::vector<int> claim_cuda_devices(uint min_devices, uint max_devices);
std::vector<int> claim_cuda_devices(uint number_of_devices);
std::vector<int> claim_cuda_devices_on_bus(std::vector<std::string> busses);

#else // These typedefs are in CUDA runtime, but also used for non-cuda target
typedef struct {
    unsigned int x;
    unsigned int y;
} uint2;

typedef struct {
    float x;
    float y;
} float2;

#endif

bool is_file_exists (const char *filename);

namespace RPGraph
{
    float get_random(float lowerbound, float upperbound);

    class Real2DVector
    {
    public:
        Real2DVector(float x, float y);
        float x, y;
        float magnitude();
        float distance(Real2DVector to); // to some other Real2DVector `to'

        // Varous operators on Real2DVector
        Real2DVector operator*(float b);
        Real2DVector operator/(float b);
        Real2DVector operator+(Real2DVector b);
        Real2DVector operator-(Real2DVector b);
        void operator+=(Real2DVector b);

        Real2DVector getNormalized();
        Real2DVector normalize();
    };

    class Coordinate
    {
    public:
        float x, y;
        Coordinate(float x, float y);

        // Various operators on Coordinate
        Coordinate operator+(float b);
        Coordinate operator*(float b);
        Coordinate operator/(float b);
        Coordinate operator+(Real2DVector b);
        Coordinate operator-(Coordinate b);
        bool operator==(Coordinate b);
        void operator/=(float b);
        void operator+=(Coordinate b);
        void operator+=(RPGraph::Real2DVector b);

        int quadrant(); // Of `this' wrt. (0,0).
        float distance(Coordinate to);
        float distance2(Coordinate to);

    };

    float distance(Coordinate from, Coordinate to);
    float distance2(Coordinate from, Coordinate to);

    Real2DVector normalizedDirection(Coordinate from, Coordinate to);
    Real2DVector direction(Coordinate from, Coordinate to);

}

#endif /* RPCommonUtils_hpp */
