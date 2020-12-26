// This is a modified version of a code by Chris Wellons, see 
// https://github.com/skeeto/mandel-simd and
// https://nullprogram.com/blog/2015/07/10/.

#include <immintrin.h>

void cpu_sse2_mandelbrot(unsigned char *img,
                         const int w,
                         const int h,
                         const int alloc_w,
                         const int it_max,
                         const float topleft_re,
                         const float topleft_im,
                         const float pix_size,
                         unsigned char *rgba_colormap)
{
    __m128 re_min = _mm_set_ps1(topleft_re);
    __m128 im_max = _mm_set_ps1(topleft_im);
    __m128 _pix_size = _mm_set_ps1(pix_size);
    __m128 threshold = _mm_set_ps1(4.0);
    __m128 one = _mm_set_ps1(1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++)
    {
        for (int x_start = 0; x_start < w; x_start += 4)
        {
            // (x, y), for each lane
            __m128 mx = _mm_set_ps(x_start + 3, x_start + 2,
                                   x_start + 1, x_start + 0);
            __m128 my = _mm_set_ps1(y);

            // c, for each lane
            __m128 cr = _mm_add_ps(_mm_mul_ps(mx, _pix_size), re_min);
            __m128 ci = _mm_sub_ps(_mm_mul_ps(my, _pix_size), im_max);

            // start at k == 0
            int k = 0;
            __m128 mk = _mm_set_ps1(k);

            // z_k, for each lane
            __m128 zr = _mm_set_ps1(0);
            __m128 zi = _mm_set_ps1(0);
            // check |z_k| for each lane
            __m128 zr2 = _mm_mul_ps(zr, zr);
            __m128 zi2 = _mm_mul_ps(zi, zi);
            __m128 mag2 = _mm_add_ps(zr2, zi2);
            __m128 mask = _mm_cmple_ps(mag2, threshold);
            bool all_lanes_done = _mm_movemask_ps(mask) != 0;

            while (k < it_max and not all_lanes_done)
            {
                /* Increment k, for lanes with |z| <= 2 */
                k += 1;
                mk = _mm_add_ps(_mm_and_ps(mask, one), mk);

                /* Compute z_k from z_{k-1} */
                __m128 zrzi = _mm_mul_ps(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm_add_ps(_mm_sub_ps(zr2, zi2), cr);
                zi = _mm_add_ps(_mm_add_ps(zrzi, zrzi), ci);
                zr2 = _mm_mul_ps(zr, zr);
                zi2 = _mm_mul_ps(zi, zi);
                mag2 = _mm_add_ps(zr2, zi2);
                mask = _mm_cmple_ps(mag2, threshold);
                all_lanes_done = _mm_movemask_ps(mask) != 0;
            }

            __m128i iterations = _mm_cvtps_epi32(mk);

            unsigned char *dst = img + x_start*4 + y*alloc_w*4;
            // TODO make src point to int, to account for it_max > 255
            unsigned char *src = (unsigned char *)&iterations;
            for (int i = 0; i < 4; i++)
            {
                int it = src[i*4]; // *4 to grab least sign. bits
                dst[i * 4 + 0] = rgba_colormap[it * 4 + 0]; // R
                dst[i * 4 + 1] = rgba_colormap[it * 4 + 1]; // G
                dst[i * 4 + 2] = rgba_colormap[it * 4 + 2]; // B
                dst[i * 4 + 3] = rgba_colormap[it * 4 + 3]; // A
            }
        }
    }
}
