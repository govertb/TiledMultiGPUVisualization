/*
 ==============================================================================

 mandelbrot_kernels.cuh
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef mandelbrot_cuh
#define mandelbrot_cuh

__global__ void mandelbrot_kernel(uchar4 *img,
                                  const int w,
                                  const int h,
                                  const int alloc_w,
                                  const int it_max,
                                  const float topleft_re,
                                  const float topleft_im,
                                  const float pix_size,
                                  unsigned char* rgba_colormap);

#endif