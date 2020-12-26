#!/bin/bash

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
src/tiled_visualization.cu \
lib/util-keysyms/keysyms/keysyms.c

# LINK #
nvcc -lGL -lX11 -lxcb -lX11-xcb -lxcb-glx -lgomp \
--generate-code arch=compute_30,code=sm_30 \
--generate-code arch=compute_37,code=sm_37 \
--generate-code arch=compute_52,code=sm_52 \
--generate-code arch=compute_61,code=sm_61 \
keysyms.o \
glad.o \
glad_glx.o \
tiled_visualization.o \
-o tiled_visualization