/*
 ==============================================================================

 mandelbrot_cpu.hpp
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef mandelbrot_cpu_hpp
#define mandelbrot_cpu_hpp

#include "mandelbrot_core.hpp"

void cpu_mandelbrot(unsigned char *img,
                    const int w,
                    const int h,
                    const int alloc_w,
                    const int it_max,
                    const float topleft_re,
                    const float topleft_im,
                    const float pix_size,
                    unsigned char *rgba_colormap);

void cpu_sse2_mandelbrot(unsigned char *img,
                         const int w,
                         const int h,
                         const int alloc_w,
                         const int it_max,
                         const float topleft_re,
                         const float topleft_im,
                         const float pix_size,
                         unsigned char *rgba_colormap);

void cpu_avx_mandelbrot(unsigned char *img,
                        const int w,
                        const int h,
                        const int alloc_w,
                        const int it_max,
                        const float topleft_re,
                        const float topleft_im,
                        const float pix_size,
                        unsigned char *rgba_colormap);


#endif