/*
 ==============================================================================

 mandelbrot_cpu.cpp
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#include "mandelbrot_cpu.hpp"

void cpu_mandelbrot(unsigned char *img,
                    const int w,
                    const int h,
                    const int alloc_w,
                    const int it_max,
                    const float topleft_re,
                    const float topleft_im,
                    const float pix_size,
                    unsigned char *rgba_colormap)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            // derive c from pixel coordinates
            float re = topleft_re + x*pix_size;
            float im = topleft_im - y*pix_size;

            int it = mandelbrot_it(re, im, it_max);
            const int idx = 4*(x + y*alloc_w);

            img[idx + 0] = rgba_colormap[4*it + 0];
            img[idx + 1] = rgba_colormap[4*it + 1];
            img[idx + 2] = rgba_colormap[4*it + 2];
            img[idx + 3] = rgba_colormap[4*it + 3];
        }
    }
}