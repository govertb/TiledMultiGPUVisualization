/*
 ==============================================================================

 RPCommon.cpp
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

#include "RPCommon.hpp"
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <map>
#include <string>

#ifdef __NVCC__

bool request_cuda_device(int gpu_id)
{
    cudaSetDevice(gpu_id);
    return cudaDeviceSynchronize() == cudaSuccess;
}


void claim_cuda_devices(std::vector<int> requested_devices)
{
    for (auto gpu_id : requested_devices)
    {
        if(!request_cuda_device(gpu_id))
        {
            printf("Error: cannot claim GPU %d\n", gpu_id);
            exit(EXIT_FAILURE);
        }
    }
}

std::vector<int> claim_cuda_devices(uint min_devices, uint max_devices)
{
    // Obtain number of GPUs
    uint device_count;
    cudaCatchError(cudaGetDeviceCount((int*)&device_count));
    if (device_count < min_devices)
    {
        printf("Error: found %u CUDA device, %u requested.\n",
               device_count, min_devices);
        exit(EXIT_FAILURE);
    }

    // If no `max_devices' argument is passed, we want all devices
    if (max_devices == 0) max_devices = device_count;

    // Try to claim devices
    std::vector<int> claimed_devices;
    for (int gpu_id = 0; gpu_id < device_count; gpu_id++)
    {
      if (request_cuda_device(gpu_id)) claimed_devices.push_back(gpu_id);
      if (claimed_devices.size() == max_devices) break;
    }

    if (claimed_devices.size() < min_devices)
    {
        printf("Error: %d devices requested, only %d are free.\n",
        min_devices, (int)claimed_devices.size());
        exit(EXIT_FAILURE);
    }

    return claimed_devices;
}

std::vector<int> claim_cuda_devices(uint number_of_devices)
{
    return claim_cuda_devices(number_of_devices, number_of_devices);
}

std::vector<int> claim_cuda_devices_on_bus(std::vector<std::string> busses)
{
    // Try to claim devices
    std::vector<int> claimed_devices;
    for(std::string bus : busses)
    {
      int gpu_id;
      cudaCatchError(cudaDeviceGetByPCIBusId(&gpu_id, bus.c_str()));
      if(request_cuda_device(gpu_id)) claimed_devices.push_back(gpu_id);
    }

    if(claimed_devices.size() != busses.size())
    {
        printf("Error: %d devices requested, of which only %d are free.\n",
               busses.size(), (int)claimed_devices.size());
        exit(EXIT_FAILURE);
    }

    return claimed_devices;
}

#endif

// by http://stackoverflow.com/a/19841704
bool is_file_exists (const char *filename)
{
    std::ifstream infile(filename);
    return infile.good();
}

namespace RPGraph
{
    float get_random(float lowerbound, float upperbound)
    {
        return lowerbound + (upperbound-lowerbound) * static_cast <float> (random()) / static_cast <float> (RAND_MAX);
    }


    /* Definitions for Real2DVector */
    Real2DVector::Real2DVector(float x, float y): x(x), y(y) {};

    float Real2DVector::magnitude()
    {
        return std::sqrt(x*x + y*y);
    }

    float Real2DVector::distance(RPGraph::Real2DVector to)
    {
        const float dx = (x - to.x)*(x - to.x);
        const float dy = (y - to.y)*(y - to.y);
        return std::sqrt(dx*dx + dy*dy);
    }

    // Various operators on Real2DVector
    Real2DVector Real2DVector::operator*(float b)
    {
        return Real2DVector(this->x * b, this->y * b);
    }

    Real2DVector Real2DVector::operator/(float b)
    {
        return Real2DVector(this->x / b, this->y / b);
    }


    Real2DVector Real2DVector::operator+(Real2DVector b)
    {
        return Real2DVector(this->x + b.x, this->y + b.y);
    }


    Real2DVector Real2DVector::operator-(Real2DVector b)
    {
        return Real2DVector(this->x - b.x, this->y - b.y);
    }

    void Real2DVector::operator+=(Real2DVector b)
    {
        this->x += b.x;
        this->y += b.y;
    }

    Real2DVector Real2DVector::getNormalized()
    {
        return Real2DVector(this->x / magnitude(), this->y / magnitude());
    }

    Real2DVector Real2DVector::normalize()
    {
        const float m = magnitude();
        this->x /= m;
        this->y /= m;
        return *this;
    }

    /* Definitions for Coordinate */
    Coordinate::Coordinate(float x, float y) : x(x), y(y) {};

    // Various operators on Coordinate
    Coordinate Coordinate::operator+(float b)
    {
        return Coordinate(x + b, y + b);
    }

    Coordinate Coordinate::operator*(float b)
    {
        return Coordinate(this->x*b, this->y*b);
    }

    Coordinate Coordinate::operator/(float b)
    {
        return Coordinate(this->x/b, this->y/b);
    }

    Coordinate Coordinate::operator+(Real2DVector b)
    {
        return Coordinate(this->x + b.x, this->y + b.y);
    }

    Coordinate Coordinate::operator-(Coordinate b)
    {
        return Coordinate(this->x - b.x, this->y - b.y);
    }

    bool Coordinate::operator==(Coordinate b)
    {
        return (this->x == b.x && this->y == b.y);
    }

    float Coordinate::distance(RPGraph::Coordinate to)
    {
        return std::sqrt((x - to.x)*(x - to.x) + (y - to.y)*(y - to.y));
    }

    float Coordinate::distance2(RPGraph::Coordinate to)
    {
        return (x - to.x)*(x - to.x) + (y - to.y)*(y - to.y);
    }

    void Coordinate::operator/=(float b)
    {
        this->x /= b;
        this->y /= b;
    }

    void Coordinate::operator+=(RPGraph::Coordinate b)
    {
        this->x += b.x;
        this->y += b.y;
    }

    void Coordinate::operator+=(RPGraph::Real2DVector b)
    {
        this->x += b.x;
        this->y += b.y;
    }

    int Coordinate::quadrant()
    {
        if (x <= 0)
        {
            if (y >= 0) return 0;
            else        return 3;

        }
        else
        {
            if (y >= 0) return 1;
            else        return 2;
        }
    }

    float distance(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        return std::sqrt(dx*dx + dy*dy);
    }

    float distance2(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        return dx*dx + dy*dy;
    }

    Real2DVector normalizedDirection(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        const float len = std::sqrt(dx*dx + dy*dy);
        return Real2DVector(dx/len, dy/len);
    }

    Real2DVector direction(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        return Real2DVector(dx, dy);
    }
}
