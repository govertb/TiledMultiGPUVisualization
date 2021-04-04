/*
 ==============================================================================

 RPFA2Kernels.cuh
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

#ifndef RPFA2Kernels_cuh
#define RPFA2Kernels_cuh

#include "RPBHFA2LaunchParameters.cuh"

__global__
__launch_bounds__(THREADS6, FACTOR6)
void GravityKernel(int nbodiesd, const float k_g, const bool strong_gravity,
                   volatile float * __restrict body_massd,
                   volatile float2 * __restrict body_posd,
                   volatile float * __restrict fxd, volatile float * __restrict fyd);

__global__
__launch_bounds__(THREADS6, FACTOR6)
void AttractiveForceKernel(int nedgesd,
                           volatile float2 * __restrict body_posd,
                           volatile float * __restrict fxd, volatile float * __restrict fyd,
                           volatile int * __restrict sourcesd, volatile int * __restrict targetsd);

__global__
__launch_bounds__(THREADS1, FACTOR1)
void SpeedKernel(int nbodiesd,
                 volatile float * __restrict fxd , volatile float * __restrict fyd,
                 volatile float * __restrict fx_prevd , volatile float * __restrict fy_prevd,
                 volatile float * __restrict body_massd, volatile float * __restrict swgd, volatile float * __restrict etrad);

__global__
__launch_bounds__(THREADS6, FACTOR6)
void InteractionKernel(int nbodiesd,
                       volatile float2 * __restrict body_posd,
                       volatile float * __restrict fxd, volatile float * __restrict fyd,
                       bool mouse_repulse, float mouse_mass,
                       bool mouse_heat,    float mouse_temp, float *temp_offsetd,
                       float mx, float my);

__global__
__launch_bounds__(THREADS6, FACTOR6)
void DisplacementKernel(int nbodiesd,
                       volatile float2 * __restrict body_posd,
                       volatile float * __restrict fxd, volatile float * __restrict fyd,
                       volatile float * __restrict fx_prevd, volatile float * __restrict fy_prevd,
                       float *temp_offsetd);

#endif
