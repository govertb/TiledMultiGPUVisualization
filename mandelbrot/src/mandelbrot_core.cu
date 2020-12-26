/*
 ==============================================================================

 mandelbrot_core.cu
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

// Implements per-pixel iteration for both CPU (serial) and GPU implementation
// of the Mandelbrot drawing algorithm

// returns 'escape iteration' for complex c,
// i.e. return min(smallest i st. |z_i| > 2, max_i)
// with z_0 = 0 and z_i := z_{i-1}^2 + c
#ifdef __NVCC__
__device__ __host__
#endif
int mandelbrot_it(float c_re, float c_im, int max_i)
{
    float r = 2.0;
    float r2 = r*r;

    // start at i == 0
    int i = 0;
    float z_re = 0, z_im = 0;
    float z_re_old, z_im_old;
    float z_mag2 = z_re*z_re + z_im*z_im; // magnitude of z, squared
    while(i < max_i and z_mag2 <= r2)
    {
        // Increment i, compute z_i
        i++;
        z_re_old = z_re;
        z_im_old = z_im;
        z_re = (z_re_old*z_re_old - z_im_old*z_im_old) + c_re;
        z_im = (2*z_re_old*z_im_old) + c_im;
        z_mag2 = z_re*z_re + z_im*z_im;
    }
    return i;
}