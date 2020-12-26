// This is a modified version of a code by Chris Wellons, see 
// https://github.com/skeeto/mandel-simd and
// https://nullprogram.com/blog/2015/07/10/.

#include <immintrin.h>

void cpu_avx_mandelbrot(unsigned char *img,
                        const int w,
                        const int h,
                        const int alloc_w,
                        const int it_max,
                        const float topleft_re,
                        const float topleft_im,
                        const float pix_size,
                        unsigned char *rgba_colormap)
{
    __m256 re_min = _mm256_set1_ps(topleft_re);
    __m256 im_max = _mm256_set1_ps(topleft_im);
    __m256 _pix_size = _mm256_set1_ps(pix_size);
    __m256 threshold = _mm256_set1_ps(4.0);
    __m256 one = _mm256_set1_ps(1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++)
    {
        for (int x_start = 0; x_start < w; x_start += 8)
        {
            // (x, y), for each lane
            __m256 mx = _mm256_set_ps(x_start + 7, x_start + 6, x_start + 5,
                                      x_start + 4, x_start + 3, x_start + 2,
                                      x_start + 1, x_start + 0);
            __m256 my = _mm256_set1_ps(y);

            // c, for each lane
            __m256 cr = _mm256_add_ps(_mm256_mul_ps(mx, _pix_size), re_min);
            __m256 ci = _mm256_sub_ps(_mm256_mul_ps(my, _pix_size), im_max);

            // start at k == 0
            int k = 0;
            __m256 mk = _mm256_set1_ps(k);

            // z_k, for each lane
            __m256 zr = _mm256_set1_ps(0);
            __m256 zi = _mm256_set1_ps(0);
            // check |z_k| for each lane
            __m256 zr2 = _mm256_mul_ps(zr, zr);
            __m256 zi2 = _mm256_mul_ps(zi, zi);
            __m256 mag2 = _mm256_add_ps(zr2, zi2);
            __m256 mask = _mm256_cmp_ps(mag2, threshold, _CMP_LT_OS);
            bool all_lanes_done = _mm256_testz_ps(mask, _mm256_set1_ps(-1));

            while (k < it_max and not all_lanes_done)
            {
                /* Increment k, for lanes with |z| <= 2 */
                k += 1;
                mk = _mm256_add_ps(_mm256_and_ps(mask, one), mk);

                /* Compute z_k from z_{k-1} */
                __m256 zrzi = _mm256_mul_ps(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm256_add_ps(_mm256_sub_ps(zr2, zi2), cr);
                zi = _mm256_add_ps(_mm256_add_ps(zrzi, zrzi), ci);
                zr2 = _mm256_mul_ps(zr, zr);
                zi2 = _mm256_mul_ps(zi, zi);
                mag2 = _mm256_add_ps(zr2, zi2);
                mask = _mm256_cmp_ps(mag2, threshold, _CMP_LT_OS);
                all_lanes_done = _mm256_testz_ps(mask, _mm256_set1_ps(-1));
            }

            __m256i iterations = _mm256_cvtps_epi32(mk);

            unsigned char *dst = img + x_start*4 + y*alloc_w*4;
            // TODO make src point to int, to account for it_max > 255
            unsigned char *src = (unsigned char *)&iterations;

            for (int i = 0; i < 8; i++)
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