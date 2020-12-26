CC:=gcc

# Directories
L_SRC_DIR         := ../../lib
SRC_DIR           := ../../src
OBJ_DIR           := obj

# Compile with CUDA support by default
CUDA_SUPPORT ?= 1

# Debug compilation, disable by default
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS:=-g
	NVCCFLAGS:=-G
	O_LVL:=-O0
else
	NVCCFLAGS:=-D NDEBUG
	O_LVL:=-O3
endif

NVCCFLAGS := $(NVCCFLAGS) --generate-code arch=compute_30,code=sm_30
NVCCFLAGS := $(NVCCFLAGS) --generate-code arch=compute_37,code=sm_37
NVCCFLAGS := $(NVCCFLAGS) --generate-code arch=compute_52,code=sm_52
NVCCFLAGS := $(NVCCFLAGS) --generate-code arch=compute_61,code=sm_61

ifeq ($(CUDA_SUPPORT), 1)
	CC:=nvcc
	CXXFLAGS := $(CXXFLAGS) $(NVCCFLAGS)
endif

# C preprocessor flags
CPPFLAGS :=-D NO_FREETYPE

# C/C++ compiler flags
CFLAGS   :=$(O_LVL)
CXXFLAGS :=$(CXXFLAGS) $(O_LVL) -std=c++11

# Linker flags
LDFLAGS := -lc -lm -lstdc++ -lpng -L /usr/lib64/nvidia/ -L /usr/lib/nvidia-384/
LDFLAGS := $(LDFLAGS)  -L /usr/lib/nvidia-396/ -L . -lOpenGL -lGLX -lEGL -lX11
TMP_ := $(shell ln -s /usr/lib64/nvidia/libGLX.so.0 libGLX.so)

# src./obj. files
GRAPH_VIEWER_SRCS := $(wildcard $(SRC_DIR)/RP*.cpp)
GRAPH_VIEWER_SRCS := $(GRAPH_VIEWER_SRCS) $(SRC_DIR)/gl_util.cpp
GRAPH_VIEWER_OBJS := $(GRAPH_VIEWER_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

PNGWRITER_SRCS    := $(L_SRC_DIR)/pngwriter/src/pngwriter.cc
PNGWRITER_OBJS    := $(PNGWRITER_SRCS:$(L_SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

GLAD_SRCS         := $(L_SRC_DIR)/glad/glad.c
GLAD_SRCS 		  := $(GLAD_SRCS) $(L_SRC_DIR)/glad/glad_egl.c
GLAD_SRCS 		  := $(GLAD_SRCS) $(L_SRC_DIR)/glad/glad_glx.c
GLAD_OBJS         := $(GLAD_SRCS:$(L_SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

KEYSYM_SRCS       := $(L_SRC_DIR)/util-keysyms/keysyms/keysyms.c
KEYSYM_OBJS       := $(KEYSYM_SRCS:$(L_SRC_DIR)/%.c=$(OBJ_DIR)/%.o)


CUDA_SRCS         := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS         := $(CUDA_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUDA_DEPS         := $(wildcard $(SRC_DIR)/*.cuh)

CPP_SRC           := $(GRAPH_VIEWER_SRCS) $(PNGWRITER_SRCS)
CUDA_SRC          := $(CUDA_SRCS)
SOURCES           := $(C_SRC) $(CPP_SRC)
OBJECTS           := $(GRAPH_VIEWER_OBJS) $(PNGWRITER_OBJS) $(GLAD_OBJS)

ifeq ($(CUDA_SUPPORT), 1)
OBJECTS           := $(OBJECTS) $(CUDA_OBJS)
SOURCES           := $(SOURCES) $(CUDA_SRCS)
endif

# Generate dependency (.h, .hpp)
# ala http://stackoverflow.com/questions/2394609/makefile-header-dependencies
.PHONY: depend
depend: .depend
.depend: $(SRCS)
	rm -f .depend
	$(CC) $(CXXFLAGS) $(CPPFLAGS) -M $(CPP_SRC) >> .depend

include ./.depend


graph_viewer_cli: $(OBJECTS)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) $(SRC_DIR)/graph_viewer_cli.cpp -o $(OBJ_DIR)/graph_viewer_cli.o
	$(CC) $(OBJECTS) $(CXXFLAGS) $(OBJ_DIR)/graph_viewer_cli.o $(LDFLAGS) -o graph_viewer_cli

graph_viewer: $(OBJECTS) $(KEYSYM_OBJS)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) $(SRC_DIR)/graph_viewer.cpp -o $(OBJ_DIR)/graph_viewer.o
	$(CC) $(OBJECTS) $(CXXFLAGS) $(KEYSYM_OBJS) $(OBJ_DIR)/graph_viewer.o $(LDFLAGS) -lX11 -lGL -lxcb -lX11-xcb -lxcb-glx -o graph_viewer

$(GRAPH_VIEWER_OBJS): $(GRAPH_VIEWER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cpp)

$(PNGWRITER_OBJS): $(PNGWRITER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.cc)

$(GLAD_OBJS) : $(GLAD_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) --system-include $(L_SRC_DIR) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.c)

$(KEYSYM_OBJS): $(KEYSYM_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.c)

$(CUDA_OBJS): $(CUDA_SRCS) $(CUDA_DEPS)
	mkdir -p $(@D)
	nvcc --device-c $(CXXFLAGS) $(NVCCFLAGS) $(CPPFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cu)

.PHONY: all
all: graph_viewer_cli graph_viewer

.PHONY: clean
clear: clean

.PHONY: clear
clean:
	rm -r graph_viewer_* $(OBJ_DIR)/* ./.depend
