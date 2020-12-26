/*
 ==============================================================================

 mandelbrot_core.hpp
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef mandelbrot_core_hpp
#define mandelbrot_core_hpp

#ifdef __NVCC__
__device__ __host__
#endif
int mandelbrot_it(float c_re, float c_im, int max_i);

#endif