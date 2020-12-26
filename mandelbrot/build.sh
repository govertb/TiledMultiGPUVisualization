#!/bin/bash

# compile AVX/SSE2 code separately, to prevent propagation of -mavx and -msse2
# flags to other code
g++ --compile -mavx  -O3 -fopenmp src/mandelbrot_avx.cpp
g++ --compile -msse2 -O3 -fopenmp src/mandelbrot_sse2.cpp
g++ --compile -O3 -fopenmp src/mandelbrot_cpu.cpp

# Compile glad separately due to -isystem
g++ --compile -O3 -isystem lib/ \
lib/glad/glad.c \
lib/glad/glad_glx.c

nvcc --compile --relocatable-device-code=true -std=c++11 \
-O3 \
--compiler-options -fopenmp \
--generate-code arch=compute_30,code=sm_30 \
--generate-code arch=compute_37,code=sm_37 \
--generate-code arch=compute_52,code=sm_52 \
--generate-code arch=compute_61,code=sm_61 \
src/mandelbrot.cu \
src/mandelbrot_core.cu \
lib/util-keysyms/keysyms/keysyms.c

# LINK #
nvcc -lGL -lX11 -lxcb -lX11-xcb -lxcb-glx -lgomp \
--generate-code arch=compute_30,code=sm_30 \
--generate-code arch=compute_37,code=sm_37 \
--generate-code arch=compute_52,code=sm_52 \
--generate-code arch=compute_61,code=sm_61 \
mandelbrot_core.o \
mandelbrot_cpu.o \
mandelbrot_avx.o \
mandelbrot_sse2.o \
keysyms.o \
glad.o \
glad_glx.o \
mandelbrot.o \
-o mandelbrot