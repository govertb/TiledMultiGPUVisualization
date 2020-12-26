/*
 ==============================================================================

 RPGPUForceAtlas2.cu
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

#include <stdio.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include "time.h"

#include "RPGPUForceAtlas2.hpp"
#include "RPBHFA2LaunchParameters.cuh"
#include "RPBHKernels.cuh"
#include "RPFA2Kernels.cuh"

namespace RPGraph
{
    CUDAForceAtlas2::CUDAForceAtlas2(GraphLayout &layout, bool use_barneshut,
                                     bool strong_gravity, float gravity,
                                     float scale, std::vector<int> gpu_ids)
    : ForceAtlas2(layout, use_barneshut, strong_gravity, gravity, scale),
      compute_gpus(gpu_ids)
    {
        // Host initialization and setup //
        nbodies = layout.graph.num_nodes();
        nedges  = layout.graph.num_edges();

        body_pos = (float2 *)malloc(sizeof(float2) * layout.graph.num_nodes());
        body_mass = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        sources  = (int *)  malloc(sizeof(int)   * layout.graph.num_edges());
        targets  = (int *)  malloc(sizeof(int)   * layout.graph.num_edges());
        fx       = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        fy       = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        fx_prev  = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        fy_prev  = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        temp_offset = (float *)malloc(sizeof(float) * layout.graph.num_nodes());

        for (nid_t n = 0; n < layout.graph.num_nodes(); ++n)
        {
            body_pos[n] = {layout.getX(n), layout.getY(n)};
            body_mass[n] = ForceAtlas2::mass(n);
            fx[n] = 0.0;
            fy[n] = 0.0;
            fx_prev[n] = 0.0;
            fy_prev[n] = 0.0;
            temp_offset[n] = 0.0;
        }

        int cur_sources_idx = 0;
        int cur_targets_idx = 0;

        // Initialize the sources and targets arrays with edge-data.
        for (nid_t source_id = 0; source_id < layout.graph.num_nodes(); ++source_id)
        {
            for (nid_t target_id : layout.graph.neighbors_with_geq_id(source_id))
            {
                sources[cur_sources_idx++] = source_id;
                targets[cur_targets_idx++] = target_id;
            }
        }

        // GPU initialization and setup //
        const int num_worker_gpus = compute_gpus.size();

        master_gpu = compute_gpus[0];

        // Alignment for coalescing
        const int alignment_size = 128;
        int job_size_general = nbodies / num_worker_gpus;
        while (job_size_general % alignment_size) job_size_general--;
        const int remaining_jobs = nbodies - (job_size_general * num_worker_gpus);
        const int job_size_last = job_size_general + remaining_jobs;

        size_t buffer_offset_cur = 0;
        for (int gpu_id : compute_gpus)
        {
            cudaCatchError(cudaSetDevice(gpu_id));

            buffer_offset[gpu_id] = buffer_offset_cur;

            if(gpu_id != compute_gpus.back())
                job_size[gpu_id] = job_size_general;
            else
                job_size[gpu_id] = job_size_last;

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            gpu_prop[gpu_id] = prop;

            if (prop.warpSize != WARPSIZE)
            {
                printf("Warpsize of device %d is %d, but we anticipated %d\n",
                        gpu_id, prop.warpSize, WARPSIZE);
                exit(EXIT_FAILURE);
            }

            cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
            cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
            cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
            cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
            cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
            cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
            #if __CUDA_ARCH__ < 300
            cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
            #endif
            cudaFuncSetCacheConfig(DisplacementKernel, cudaFuncCachePreferL1);

            cudaGetLastError();  // reset error value

            // Allocate space on device.
            int mp_count = prop.multiProcessorCount;
            int max_threads_per_block = prop.maxThreadsPerBlock;

            int cur_nnodes = std::max(2 * nbodies, mp_count * max_threads_per_block);

            // Round up to next multiple of WARPSIZE
            while ((cur_nnodes & (WARPSIZE-1)) != 0) cur_nnodes++;
            cur_nnodes--;
            nnodes[gpu_id] = cur_nnodes;

            // child stores structure of the quadtree. values point to IDs.
            cudaCatchError(cudaMalloc(&childl[gpu_id], 4 * (cur_nnodes+1) * sizeof(int)));

            // the following properties, for each node in the quadtree (both internal and leaf)
            cudaCatchError(cudaMalloc(&body_posl[gpu_id],  nbodies * sizeof(float2)));
            cudaCatchError(cudaMalloc(&body_massl[gpu_id], nbodies * sizeof(float)));
            cudaCatchError(cudaMalloc(&node_posl[gpu_id],  (cur_nnodes+1) * sizeof(float2)));
            cudaCatchError(cudaMalloc(&node_massl[gpu_id], (cur_nnodes+1) * sizeof(float)));

            // count: number of nested nodes for each node in quadtree
            cudaCatchError(cudaMalloc(&countl[gpu_id], (cur_nnodes+1) * sizeof(int)));
            // start: the start ID for each cell
            cudaCatchError(cudaMalloc(&startl[gpu_id], (cur_nnodes+1) * sizeof(int)));
            // sort: a permutation on ..., based on spatial proximity
            cudaCatchError(cudaMalloc(&sortl[gpu_id],  (cur_nnodes+1) * sizeof(int)));


            cudaCatchError(cudaMalloc(&sourcesl[gpu_id], sizeof(int)   * (nedges)));
            cudaCatchError(cudaMalloc(&targetsl[gpu_id], sizeof(int)   * (nedges)));
            cudaCatchError(cudaMalloc(&fxl[gpu_id],     sizeof(float) * (nbodies)));
            cudaCatchError(cudaMalloc(&fyl[gpu_id],     sizeof(float) * (nbodies)));
            cudaCatchError(cudaMalloc(&fx_sortedl[gpu_id], nbodies * sizeof(float)));
            cudaCatchError(cudaMalloc(&fy_sortedl[gpu_id], nbodies * sizeof(float)));
            cudaCatchError(cudaMalloc(&fx_prevl[gpu_id], sizeof(float) * (nbodies)));
            cudaCatchError(cudaMalloc(&fy_prevl[gpu_id], sizeof(float) * (nbodies)));

            cudaCatchError(cudaMalloc(&temp_offsetl[gpu_id], sizeof(float) * nbodies));

            // Used for reduction in BoundingBoxKernel
            cudaCatchError(cudaMalloc(&maxxl[gpu_id],   sizeof(float) * mp_count * FACTOR1));
            cudaCatchError(cudaMalloc(&maxyl[gpu_id],   sizeof(float) * mp_count * FACTOR1));
            cudaCatchError(cudaMalloc(&minxl[gpu_id],   sizeof(float) * mp_count * FACTOR1));
            cudaCatchError(cudaMalloc(&minyl[gpu_id],   sizeof(float) * mp_count * FACTOR1));

            // Used for reduction in SpeedKernel
            cudaCatchError(cudaMalloc(&swgl[gpu_id],    sizeof(float) * mp_count * FACTOR1));
            cudaCatchError(cudaMalloc(&etral[gpu_id],   sizeof(float) * mp_count * FACTOR1));

            // Copy host data to device.
            cudaCatchError(cudaMemcpy(body_massl[gpu_id], body_mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
            cudaCatchError(cudaMemcpy(body_posl[gpu_id],  body_pos,  sizeof(float2) * nbodies, cudaMemcpyHostToDevice));
            cudaCatchError(cudaMemcpy(sourcesl[gpu_id], sources, sizeof(int) * nedges, cudaMemcpyHostToDevice));
            cudaCatchError(cudaMemcpy(targetsl[gpu_id], targets, sizeof(int) * nedges, cudaMemcpyHostToDevice));

            // cpy fx, fy , fx_prevl, fy_prevl, temp_offset so they are all initialized to 0 in device memory.
            cudaCatchError(cudaMemcpy(fxl[gpu_id], fx,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
            cudaCatchError(cudaMemcpy(fyl[gpu_id], fy,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
            cudaCatchError(cudaMemcpy(fx_prevl[gpu_id], fx_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
            cudaCatchError(cudaMemcpy(fy_prevl[gpu_id], fy_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
            cudaCatchError(cudaMemcpy(temp_offsetl[gpu_id], temp_offset, sizeof(float) * nbodies, cudaMemcpyHostToDevice));

            buffer_offset_cur += job_size_general;
        }
    }

    void CUDAForceAtlas2::freeGPUMemory()
    {
        for (int gpu_id : compute_gpus)
        {
            cudaCatchError(cudaFree(childl[gpu_id]));

            // if(not use_interop)
            //     cudaCatchError(cudaFree(body_posl[gpu_id]));
            cudaCatchError(cudaFree(body_massl[gpu_id]));
            cudaCatchError(cudaFree(node_posl[gpu_id]));
            cudaCatchError(cudaFree(node_massl[gpu_id]));
            cudaCatchError(cudaFree(sourcesl[gpu_id]));
            cudaCatchError(cudaFree(targetsl[gpu_id]));
            cudaCatchError(cudaFree(countl[gpu_id]));
            cudaCatchError(cudaFree(startl[gpu_id]));
            cudaCatchError(cudaFree(sortl[gpu_id]));

            cudaCatchError(cudaFree(fxl[gpu_id]));
            cudaCatchError(cudaFree(fyl[gpu_id]));
            cudaCatchError(cudaFree(fx_sortedl[gpu_id]));
            cudaCatchError(cudaFree(fy_sortedl[gpu_id]));
            cudaCatchError(cudaFree(fx_prevl[gpu_id]));
            cudaCatchError(cudaFree(fy_prevl[gpu_id]));
            cudaCatchError(cudaFree(temp_offsetl[gpu_id]));

            cudaCatchError(cudaFree(minxl[gpu_id]));
            cudaCatchError(cudaFree(minyl[gpu_id]));
            cudaCatchError(cudaFree(maxxl[gpu_id]));
            cudaCatchError(cudaFree(maxyl[gpu_id]));

            cudaCatchError(cudaFree(swgl[gpu_id]));
            cudaCatchError(cudaFree(etral[gpu_id]));
        }
    }

    CUDAForceAtlas2::~CUDAForceAtlas2()
    {
        free(body_mass);
        free(body_pos);
        free(sources);
        free(targets);
        free(fx);
        free(fy);
        free(fx_prev);
        free(fy_prev);
        free(temp_offset);

        freeGPUMemory();
    }

    void CUDAForceAtlas2::set_posbuf(int gpu_id, float2 *posbuf)
    {
        body_posl[gpu_id] = posbuf;
    }

    void CUDAForceAtlas2::doStep()
    {
        for (int gpu_id : compute_gpus)
            doPhase1(gpu_id);

        // NB reduce() should always run due to MergeForcesKernel and sync.
        reduce(); // ForceApproximation results -> master_gpu

        doPhase2();

        // Push new body positions to other GPUs
        if(compute_gpus.size() > 1)
            shareResults();

        cudaCatchError(cudaSetDevice(master_gpu));
        cudaCatchError(cudaDeviceSynchronize());
        iteration++;
    }

    void CUDAForceAtlas2::doPhase1(int gpu_id)
    {
        cudaCatchError(cudaSetDevice(gpu_id));
        int mp_count = gpu_prop[gpu_id].multiProcessorCount;
        size_t offset = buffer_offset[gpu_id];

        GravityKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, k_g, strong_gravity, body_massl[gpu_id], body_posl[gpu_id], fxl[gpu_id], fyl[gpu_id]);

        AttractiveForceKernel<<<mp_count * FACTOR6, THREADS6>>>(nedges, body_posl[gpu_id], fxl[gpu_id], fyl[gpu_id], sourcesl[gpu_id], targetsl[gpu_id]);

        BoundingBoxKernel<<<mp_count * FACTOR1, THREADS1>>>(nnodes[gpu_id], nbodies, startl[gpu_id], childl[gpu_id], node_massl[gpu_id], body_posl[gpu_id], node_posl[gpu_id], maxxl[gpu_id], maxyl[gpu_id], minxl[gpu_id], minyl[gpu_id]);

        // Build Barnes-Hut Tree
        // 1.) Set all child pointers of internal nodes (in childl) to null (-1)
        ClearKernel1<<<mp_count, 1024>>>(nnodes[gpu_id], nbodies, childl[gpu_id]);
        // 2.) Build the tree
        TreeBuildingKernel<<<mp_count * FACTOR2, THREADS2>>>(nnodes[gpu_id], nbodies, childl[gpu_id], body_posl[gpu_id], node_posl[gpu_id]);

        // 3.) Set all cell mass values to -1.0, set all startd to null (-1)
        ClearKernel2<<<mp_count, 1024>>>(nnodes[gpu_id], startl[gpu_id], node_massl[gpu_id]);

        // Recursively compute mass for each BH. cell.
        SummarizationKernel<<<mp_count * FACTOR3, THREADS3>>>(nnodes[gpu_id], nbodies, countl[gpu_id], childl[gpu_id], body_massl[gpu_id], node_massl[gpu_id], body_posl[gpu_id], node_posl[gpu_id]);

        SortKernel<<<mp_count * FACTOR4, THREADS4>>>(nnodes[gpu_id], nbodies, sortl[gpu_id], countl[gpu_id], startl[gpu_id], childl[gpu_id]);

        // Compute repulsive forces between nodes using BH. tree.
        ForceCalculationKernel<<<mp_count * FACTOR5, THREADS5>>>(nnodes[gpu_id], nbodies, itolsq, epssq, sortl[gpu_id], childl[gpu_id], body_massl[gpu_id], node_massl[gpu_id], body_posl[gpu_id], node_posl[gpu_id], fx_sortedl[gpu_id], fy_sortedl[gpu_id], k_r, offset, job_size[gpu_id]);

        // Account for interactions by user
        if(mouse_repulse or mouse_heat)
            InteractionKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, body_posl[gpu_id], fxl[gpu_id], fyl[gpu_id], mouse_repulse, mouse_mass, mouse_heat, mouse_temp, temp_offsetl[gpu_id], mx, my);
    }

    void CUDAForceAtlas2::reduce()
    {
        cudaCatchError(cudaSetDevice(master_gpu));
        for(int gpu_id : compute_gpus)
        {
            if(gpu_id == master_gpu) continue;
            cudaCatchError(cudaMemcpy(fx_sortedl[master_gpu] + buffer_offset[gpu_id], fx_sortedl[gpu_id] + buffer_offset[gpu_id], job_size[gpu_id] * sizeof(float), cudaMemcpyDefault));
            cudaCatchError(cudaMemcpy(fy_sortedl[master_gpu] + buffer_offset[gpu_id], fy_sortedl[gpu_id] + buffer_offset[gpu_id], job_size[gpu_id] * sizeof(float), cudaMemcpyDefault));
        }

        int mp_count = gpu_prop[master_gpu].multiProcessorCount;
        MergeForcesKernel<<<mp_count * FACTOR4, THREADS4>>>(nbodies, sortl[master_gpu], fxl[master_gpu], fyl[master_gpu], fx_sortedl[master_gpu], fy_sortedl[master_gpu]);
    }

    void CUDAForceAtlas2::doPhase2()
    {
        int gpu_id = master_gpu;
        cudaCatchError(cudaSetDevice(gpu_id));
        int mp_count = gpu_prop[gpu_id].multiProcessorCount;
        SpeedKernel<<<mp_count * FACTOR1, THREADS1>>>(nbodies, fxl[gpu_id], fyl[gpu_id], fx_prevl[gpu_id], fy_prevl[gpu_id], body_massl[gpu_id], swgl[gpu_id], etral[gpu_id]);
        DisplacementKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, body_posl[gpu_id], fxl[gpu_id], fyl[gpu_id], fx_prevl[gpu_id], fy_prevl[gpu_id], temp_offsetl[gpu_id]);
    }

    void CUDAForceAtlas2::shareResults()
    {
        cudaCatchError(cudaSetDevice(master_gpu));
        for (int gpu_id : compute_gpus)
        {
            if(gpu_id == master_gpu) continue;
            cudaCatchError(cudaMemcpy(body_posl[gpu_id], body_posl[master_gpu], nbodies * sizeof(float2), cudaMemcpyDefault));
        }
    }

    void CUDAForceAtlas2::sendGraphToGPU()
    {
        for (int gpu_id : compute_gpus)
        {
            cudaCatchError(cudaMemcpy(body_massl[gpu_id], body_mass, nbodies * sizeof(float), cudaMemcpyDefault));
            cudaCatchError(cudaMemcpy(sourcesl[gpu_id], sources, nedges * sizeof(int), cudaMemcpyDefault));
            cudaCatchError(cudaMemcpy(targetsl[gpu_id], targets, nedges * sizeof(int), cudaMemcpyDefault));
            cudaDeviceSynchronize();
        }
    }

    void CUDAForceAtlas2::sync_layout()
    {
        cudaCatchError(cudaMemcpy(body_pos, body_posl[master_gpu], nbodies * sizeof(float2), cudaMemcpyDefault));
        cudaCatchError(cudaDeviceSynchronize());
        for (size_t body_id = 0; body_id < nbodies; body_id++)
        {
            layout.setX(body_id, body_pos[body_id].x);
            layout.setY(body_id, body_pos[body_id].y);
        }
    }

    void CUDAForceAtlas2::sync_layout_bounds()
    {
        int gpu_id = master_gpu;
        cudaSetDevice(gpu_id);
        int mp_count = gpu_prop[gpu_id].multiProcessorCount;
        BoundingBoxKernel<<<mp_count * FACTOR1, THREADS1>>>(nnodes[gpu_id], nbodies, startl[gpu_id], childl[gpu_id], node_massl[gpu_id], body_posl[gpu_id], node_posl[gpu_id], maxxl[gpu_id], maxyl[gpu_id], minxl[gpu_id], minyl[gpu_id]);
        cudaCatchError(cudaMemcpy(&layout.gpu_minx, minxl[gpu_id], sizeof(float), cudaMemcpyDefault));
        cudaCatchError(cudaMemcpy(&layout.gpu_miny, minyl[gpu_id], sizeof(float), cudaMemcpyDefault));
        cudaCatchError(cudaMemcpy(&layout.gpu_maxx, maxxl[gpu_id], sizeof(float), cudaMemcpyDefault));
        cudaCatchError(cudaMemcpy(&layout.gpu_maxy, maxyl[gpu_id], sizeof(float), cudaMemcpyDefault));
    }
}
