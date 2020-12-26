/*
 ==============================================================================

 RPGPUForceAtlas2.hpp
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

#ifndef RPGPUForceAtlas2_hpp
#define RPGPUForceAtlas2_hpp
#include "RPForceAtlas2.hpp"
#include <map>

namespace RPGraph
{
    class CUDAForceAtlas2: public ForceAtlas2
    {
    public:
        CUDAForceAtlas2(GraphLayout &layout, bool use_barneshut,
                        bool strong_gravity, float gravity, float scale,
                        std::vector<int> gpu_ids);
        ~CUDAForceAtlas2();
        void doStep() override;
        void doStepGPU(int gpu_id);
        void doPhase1(int gpu_id);
        void reduce();
        void doPhase2();
        void shareResults();
        void sync_layout() override;
        void sync_layout_bounds();
        void set_posbuf(int gpu_id, float2* posbuf);

    // private:
        int nbodies;
        int nedges;

        std::vector<int> compute_gpus;
        int master_gpu; // to which we reduce

        /// CUDA Specific stuff.
        // Host storage.
        float *body_mass;
        float2 *body_pos;
        float *fx, *fy, *fx_prev, *fy_prev;
        float *temp_offset;

        // Quick way to represent a graph on the GPU
        int *sources, *targets;

        // Pointers to device memory, for multiple GPUs (all suffixed with 'l').
        std::map<int, int> nnodes;
        std::map<int, int *>    errl, sortl, childl, countl, startl;
        std::map<int, int *>    sourcesl, targetsl;
        std::map<int, float *>  body_massl, node_massl;
        std::map<int, float2 *> body_posl, node_posl;
        std::map<int, float *>  minxl, minyl, maxxl, maxyl;
        std::map<int, float *>  fxl, fyl, fx_sortedl, fy_sortedl, fx_prevl, fy_prevl;
        std::map<int, float *>  swgl, etral;
        std::map<int, float *>  temp_offsetl;
        std::map<int, size_t>   buffer_offset;
        std::map<int, size_t>   job_size;

        std::map<int, cudaDeviceProp> gpu_prop;

        void sendGraphToGPU();
        void GPUSyncLayout();
        void freeGPUMemory();
    };
};


#endif /* RPGPUForceAtlas2_hpp */
