/*
 ==============================================================================

 mandelbrot_kernels.cu
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#include "mandelbrot_kernels.cuh"
#include "mandelbrot_core.hpp"

__global__ void mandelbrot_kernel(uchar4 *img,
                                  const int w,
                                  const int h,
                                  const int alloc_w,
                                  const int it_max,
                                  const float topleft_re,
                                  const float topleft_im,
                                  const float pix_size,
                                  unsigned char* rgba_colormap)
{
    for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < w; x += (blockDim.x * gridDim.x))
    {
        for (int y = threadIdx.y + blockIdx.y * blockDim.y; y < h; y += (blockDim.y * gridDim.y))
        {
            // derive c from pixel coordinates
            const float re = topleft_re + x*pix_size;
            const float im = topleft_im - y*pix_size;

            // compute escape iteration for c = (re, im)
            int it = mandelbrot_it(re, im, it_max);
            img[x + y*alloc_w] = make_uchar4(rgba_colormap[4*it + 0],
                                             rgba_colormap[4*it + 1],
                                             rgba_colormap[4*it + 2],
                                             rgba_colormap[4*it + 3]);

        }
    }
}