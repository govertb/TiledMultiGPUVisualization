/*
 ==============================================================================

 mandelbrot.cu
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

// C
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <cpuid.h>

// C++
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>
#include <chrono>
#include <map>

// OpenMP
#include <omp.h>

// Window System
#include <X11/Xlib.h>
#include <X11/Xlib-xcb.h>
#include <xcb/xcb.h>
#include <xcb/glx.h>

// Graphics Library
#include "../lib/glad/glad.h"
#include "../lib/glad/glad_glx.h"

// Utility functions
#include "x11_util.h"
#include "gl_util.h"

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>

#include "cuda_util.h"

// Mandelbrot & Benchmarking
#include "mandelbrot_cpu.hpp"
#include "mandelbrot_kernels.cu"
#include "benchmark_setup.hpp"
#include "mandelbrot_paths.h"

// Other
#include "../lib/colormap/include/colormap/colormap.h"

enum class runmodes {gui, benchmark_compute, benchmark_display, benchmark_gui};
enum class implementations {cpu, gpu};

// application state
bool program_should_run = true;
Display *dpy; // Xlib handle to X
xcb_connection_t *xcb_connection; // xcb handle to X
std::vector<pthread_t> threads;
pthread_barrier_t swap_barrier;
bool colormap_changed = false;
const cudaStream_t default_stream = 0;

// what to render
typedef struct mandelbrot_parameters
{
    int it_max = 255;
    double offset_re = -0.5;
    double offset_im = 0.0;
    double range_re = 3.2;
    int cmap_index = 30;
} mandelbrot_parameters_t;
mandelbrot_parameters_t mpar_render, mpar_event;
std::vector<std::shared_ptr<colormap::Colormap const>> cmap_list =
    colormap::ColormapList::getAll();

// display/X info
int display_w = 0;
int display_h = 0;
std::map<int, xcb_screen_t> x_screens;
std::map<int, int> gpu_screen; // gpu -> screen_no map
const std::vector<int> screen_order = {0, 2, 1}; // for BigEye

// for each screen (screen_no as key):
std::map<int, int> swap_ready;
typedef struct screen_resources
{
    xcb_window_t x_window;
    GLXWindow glx_window;
    GLXContext gl_context;
    GLXFBConfig* fb_configs;
    GLuint pbo, vbo, tex, program, vao;
    int gpu_id;
    cudaGraphicsResource_t cuda_pbo_resource;
} screen_resources_t;
std::map<int, screen_resources_t> display_screen_resources;
std::map<xcb_window_t, int> window_to_screen_no;

// per screen, benchmarks
std::map<int, benchmark_clock_t::duration> inter_frame_durations;
std::map<int, std::vector<benchmark_clock_t::duration>> inter_frame_durations_benchmark;

// gpu properties and memory pointers, indexed by GPU ID.
std::vector<int> compute_gpu_ids;
std::map<int, dim3> bdims, gdims;
std::map<int, uchar4*> d_imgs; // indexed by gpu_id
std::map<int, uchar4*> h_imgs; // indexed by gpu_id

// function definitions

int divup(int numerator, int denominator)
{
    return numerator % denominator ? numerator / denominator + 1 :
                                     numerator / denominator;
}

unsigned char *rgba_colormap;
void generate_colormap()
{
    std::shared_ptr<colormap::Colormap const> cmap = cmap_list[mpar_event.cmap_index];
    for(int i = 0; i <= mpar_event.it_max; ++i)
    {
        colormap::Color c = cmap->getColor((float)i / mpar_event.it_max);
        rgba_colormap[4*i + 0] = c.r * 255;
        rgba_colormap[4*i + 1] = c.g * 255;
        rgba_colormap[4*i + 2] = c.b * 255;
        rgba_colormap[4*i + 3] = 0;
    }
    colormap_changed = true;
}

std::map<int, unsigned char*> rgba_colormaps_d;
void upload_colormap()
{
    for(auto gpu_id : compute_gpu_ids)
        cuda_check_error(cudaMemcpy(rgba_colormaps_d[gpu_id],
                                    rgba_colormap,
                                    (mpar_event.it_max+1) * 4,
                                    cudaMemcpyDefault));
    colormap_changed = false;
}


void init_xlib()
{
    if(!XInitThreads())
    {
        printf("Error: Xlib or machine does not support multithreading,\n"
               "       or initialization of multithreading failed.\n");
        exit(EXIT_FAILURE);
    }

    dpy = XOpenDisplay(NULL);
    if(dpy == NULL)
    {
        printf("Error: failed to connect to display '%s'.\n", getenv("DISPLAY"));
        exit(EXIT_FAILURE);
    }
}

void init_xcb()
{
    xcb_connection = XGetXCBConnection(dpy);
    if(!xcb_connection)
    {
        printf("Error: failed to open XCB connection.\n");
        exit(EXIT_FAILURE);
    }

    XSetEventQueueOwner(dpy, XCBOwnsEventQueue);
}

void load_glx(Display *dpy)
{
    int screen_no = 0;
    if(!gladLoadGLX(dpy, screen_no))
    {
        printf("Error: couldn't load GLX through glad.\n");
        exit(EXIT_FAILURE);
    }
}

bool gl_loaded = false;
void load_gl()
{
    if(!gladLoadGLLoader((GLADloadproc) glXGetProcAddress))
    {
        printf("Error: couldn't load GL through glad\n");
        exit(EXIT_FAILURE);
    }
    gl_loaded = true;
}

void init_screens()
{
    // Record information about screens into screens std::map
    xcb_screen_iterator_t screen_iterator = xcb_setup_roots_iterator(xcb_get_setup(xcb_connection));
    const int screen_count = screen_iterator.rem;
    if(screen_count < 1)
    {
        printf("Error: number of X screens <= 0.\n");
        exit(EXIT_FAILURE);
    }

    // Else, we have screens
    for(int screen_no = 0; screen_iterator.rem; xcb_screen_next(&screen_iterator), screen_no++)
    {
        xcb_screen_t screen = *screen_iterator.data;
        x_screens[screen_no] = screen;

        display_w += screen.width_in_pixels;
        display_h = screen.height_in_pixels;

        // Important to do this here, so size of map doesn't change when in use.
        swap_ready[screen_no] = 0;
        std::chrono::milliseconds d{1};
        inter_frame_durations[screen_no] = benchmark_clock_t::duration(d);
        inter_frame_durations_benchmark[screen_no].push_back(benchmark_clock_t::duration(d));
    }

    // Framebuffer attributes required to be supported
    const int fb_attrib_list[] = {
                            GLX_DOUBLEBUFFER, True,
                            GLX_RENDER_TYPE, GLX_RGBA_BIT,
                            GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT | GLX_PBUFFER_BIT,
                            None};

    // Get best matching FBConfigs for each screen.
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        screen_resources_t &sr = display_screen_resources[screen_no];

        int num_matches = 0;
        sr.fb_configs = glXChooseFBConfig(dpy, screen_no, fb_attrib_list, &num_matches);
        if(num_matches <= 0 or not sr.fb_configs)
        {
            printf("Error couldn't find any framebuffer configurations for screen %d.\n", screen_no);
            exit(EXIT_FAILURE);
        }
    }
}

void create_xwindows()
{
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        const xcb_screen_t screen = it.second;
        screen_resources_t &sr = display_screen_resources[screen_no];

        const xcb_window_t parent = screen.root;
        const int pos_x = 0;
        const int pos_y = 0;
        const uint16_t border_width = 0;

        int visual_id;
        glXGetFBConfigAttrib(dpy, sr.fb_configs[0], GLX_VISUAL_ID , &visual_id);

        const xcb_colormap_t colormap = xcb_generate_id(xcb_connection);
        xcb_create_colormap(xcb_connection, XCB_COLORMAP_ALLOC_NONE, colormap, parent, visual_id);

        const uint32_t eventmask = XCB_EVENT_MASK_EXPOSURE |
                                   XCB_EVENT_MASK_KEY_PRESS|
                                   XCB_EVENT_MASK_BUTTON_PRESS |
                                   XCB_EVENT_MASK_BUTTON_MOTION |
                                   XCB_EVENT_MASK_POINTER_MOTION |
                                   XCB_EVENT_MASK_STRUCTURE_NOTIFY;
        const uint32_t valuelist[] = { eventmask, colormap, 0 };
        const uint32_t valuemask = XCB_CW_EVENT_MASK | XCB_CW_COLORMAP;

        sr.x_window = xcb_generate_id(xcb_connection);
        xcb_create_window(xcb_connection, XCB_COPY_FROM_PARENT, sr.x_window, parent,
            pos_x, pos_y, screen.width_in_pixels, screen.height_in_pixels,
            border_width, XCB_WINDOW_CLASS_INPUT_OUTPUT, visual_id, valuemask,
            valuelist);

        if(not sr.x_window)
        {
            printf("Error: couldn't create X window for screen %d.\n", screen_no);
            exit(EXIT_FAILURE);
        }

        window_to_screen_no[screen.root] = screen_no;
    }
}

void create_glxwindows()
{
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        screen_resources_t &sr = display_screen_resources[screen_no];

        // Obtain parent, the X Window.
        xcb_window_t parent_window = sr.x_window;
        sr.glx_window = glXCreateWindow(dpy, sr.fb_configs[0], parent_window, NULL);
        if(!sr.glx_window)
        {
            printf("Error: couldn't create GLXWindow for screen %d.\n", screen_no);
            exit(EXIT_FAILURE);
        }
    }
}

void init_gl()
{
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        screen_resources_t &sr = display_screen_resources[screen_no];

        // Create an OpenGL context.
        const GLXContext share_list = NULL;
        const Bool direct = True;
        int context_attributes[] =
        {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
            GLX_CONTEXT_MINOR_VERSION_ARB, 1,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };

        GLXContext gl_context = glXCreateContextAttribsARB(dpy, sr.fb_configs[0],
                                                           share_list, direct, context_attributes);

        if(gl_context != NULL && glXIsDirect(dpy, gl_context))
        {
            sr.gl_context = gl_context;
        }
        else
        {
            printf("Failed to create GL Context.\n");
            if (gl_context != NULL && !glXIsDirect(dpy, gl_context))
                printf("  no direct GL Context could be made.\n");
            exit(EXIT_FAILURE);
        }

        // Make context current and bind glx_window.
        const GLXDrawable drawable = sr.glx_window;
        Bool context_made_current = glXMakeCurrent(dpy, drawable, gl_context);
        if(!context_made_current)
        {
            printf("Couldn't make context current for screen %d\n", screen_no);
            exit(EXIT_FAILURE);
        }

        // Disable VBLANK sync.
        glXSwapIntervalEXT(dpy, drawable, 0);

        if(not gl_loaded) load_gl();

        // Get associated CUDA device
        int cuda_devicecount;
        cuda_check_error(cudaGetDeviceCount(&cuda_devicecount));
        uint num_gl_associated_devices;
        int *gl_associated_devices = new int[cuda_devicecount];
        cuda_check_error(cudaGLGetDevices(&num_gl_associated_devices,
                                          gl_associated_devices, cuda_devicecount,
                                          cudaGLDeviceListAll));
        if (num_gl_associated_devices == 1)
        {
            sr.gpu_id = gl_associated_devices[0];
            gpu_screen[sr.gpu_id] = screen_no;
        }

        else if (num_gl_associated_devices > 1)
        {
            printf("Error: more then one GPU associated with GL context for screen %d\n", screen_no);
            exit(EXIT_FAILURE);
        }

        else
        {
            printf("Error: couldn't find GL device used for GL context of screen %d... ", screen_no);
            exit(EXIT_FAILURE);
        }

        // Create and bind VAO
        glGenVertexArrays(1, &sr.vao);
        glBindVertexArray(sr.vao);

        // Create GL Texture, using glXCreatePbuffer
        glGenTextures(1, &sr.tex);
        glBindTexture(GL_TEXTURE_2D, sr.tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // Create GL PBO
        glGenBuffers(1, &sr.pbo);

        // Create GL VBO
        glGenBuffers(1, &sr.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, sr.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // Compile the GL program.
        sr.program = gl_tex_program();
        glUseProgram(sr.program);

        // Setup connections
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void *)(2*sizeof(float)));
        glEnableVertexAttribArray(1);

        release_gl_context(dpy);
    }
}

void init_interop()
{
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        screen_resources_t &sr = display_screen_resources[screen_no];

        // Make context for `screen_no' current.
        if(!glXMakeCurrent(dpy, sr.glx_window, sr.gl_context))
        {
            printf("Error: couldn't make context current for screen %d.\n", screen_no);
            exit(EXIT_FAILURE);
        }

        // Select corresponding CUDA device
        cuda_check_error(cudaSetDevice(sr.gpu_id));

        // Register PBO for this screen and context with CUDA runtime
        cuda_check_error(cudaGraphicsGLRegisterBuffer(&sr.cuda_pbo_resource, sr.pbo,
                                                      cudaGraphicsRegisterFlagsWriteDiscard));

        // Done
        release_gl_context(dpy);
    }
}

typedef struct crlparameters_
{
    int screen_no;
    bool use_interop;
    int max_iterations;
    bool record_fps;
    bool no_input_mode;
    int it_max;
    int alloc_w;
} crlparameters_t;

void *compute_render_loop(void* args)
{
    crlparameters_t p = *((crlparameters_t *) args);

    // Obtain information of screen to render to
    xcb_screen_t screen = x_screens.at(p.screen_no);
    screen_resources_t &sr = display_screen_resources.at(p.screen_no);
    int screen_count = xcb_setup_roots_iterator(xcb_get_setup(xcb_connection)).rem;

    int image_w = screen.width_in_pixels * screen_count;
    int image_h = screen.height_in_pixels;

    // Obtain (properties of) gpu that does rendering.
    cuda_check_error(cudaSetDevice(sr.gpu_id));

    uchar4 *d_img; // pointer to Mandelbrot in CUDA memory
    uchar4 *h_img; // pointer to Mandelbrot in CPU  memory (if used)

    if(not p.use_interop)
        h_img = h_imgs.at(sr.gpu_id);

    // Setup OpenGL
    if(!glXMakeCurrent(dpy, sr.glx_window, sr.gl_context))
    {
        printf("Error: couldn't make context current for screen %d.\n",
               p.screen_no);
        exit(EXIT_FAILURE);
    }

    glUseProgram(sr.program);
    glBindVertexArray(sr.vao);

    // Start compute+render loop
    int frame_no = 0;

    // To measure FPS
    benchmark_clock_t clock;
    benchmark_clock_t::time_point prev_swap_time = clock.now();
    while(program_should_run)
    {
        // Derive what part of complex plane to draw.
        double topleft_re, topleft_im, offset_re, offset_im, range_re;
        if(p.no_input_mode)
        {
            offset_re = path_points[frame_no % num_path_points][0];
            offset_im = path_points[frame_no % num_path_points][1];
            range_re  = path_points[frame_no % num_path_points][2];
        }
        else
        {
            offset_re = mpar_render.offset_re;
            offset_im = mpar_render.offset_im;
            range_re  = mpar_render.range_re;
        }

        // Obtain CUDA memory to store Mandelbrot results.
        if(p.use_interop)
        {
            cuda_check_error(cudaGraphicsMapResources(1, &sr.cuda_pbo_resource, NULL));
            size_t num_bytes;
            cuda_check_error(cudaGraphicsResourceGetMappedPointer((void **)&d_img, &num_bytes, sr.cuda_pbo_resource));
        }

        else
        {
            d_img = d_imgs.at(sr.gpu_id);
        }
        const double pix_size = range_re / image_w;
        float stepsize_re = range_re / display_screen_resources.size();
        topleft_re = offset_re - image_w/2.0*pix_size + (screen_order[p.screen_no] * stepsize_re);
        topleft_im = offset_im + image_h/2.0*pix_size;

        mandelbrot_kernel<<<gdims[sr.gpu_id], bdims[sr.gpu_id]>>>(d_img, screen.width_in_pixels, screen.height_in_pixels, p.alloc_w, p.it_max, topleft_re, topleft_im, pix_size, rgba_colormaps_d[sr.gpu_id]);

        // Render CUDA results using OpenGL
        if(p.use_interop)
        {
            // We are done, just unmap (will wait for cuda to complete)
            // to make PBO available to GL again. (wo unmap, undefined results)
            cuda_check_error(cudaGraphicsUnmapResources(1, &sr.cuda_pbo_resource, default_stream));

        }
        else
        {
            const int num_bytes = screen.width_in_pixels * screen.height_in_pixels * sizeof(uchar4);
            cuda_check_error(cudaMemcpy(h_img, d_img, num_bytes, cudaMemcpyDeviceToHost));
            const GLintptr offset = 0;
            cuda_check_error(cudaDeviceSynchronize());
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, offset, num_bytes, h_img);
        }

        // Copy data from buffer to texture ?
        const GLint level = 0, xoffset = 0, yoffset = 0;
        glTexSubImage2D(GL_TEXTURE_2D, level, xoffset, yoffset,
                        screen.width_in_pixels, screen.height_in_pixels,
                        GL_RGBA, GL_UNSIGNED_BYTE, (char *)0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Ensure all outstanding OpenGL commands have been processed.
        glFinish();

        // Swap front and back buffer of the GLXWindow, first screen waits till
        // others wait to update view-state.
        int num_swappable = 0;
        if(p.screen_no == 0)
        {
            while(num_swappable != screen_count-1)
            {
                num_swappable = 0;
                for(int i = 0; i < screen_count; i++)
                    num_swappable += swap_ready.at(i);
            }
            mpar_render = mpar_event;
            if(colormap_changed) upload_colormap();
        }

        swap_ready.at(p.screen_no) = 1; // For view state only

        pthread_barrier_wait(&swap_barrier);
        glXSwapBuffers(dpy, sr.glx_window);

        swap_ready.at(p.screen_no) = 0;  // For view state only

        // Record some performance measurements
        benchmark_clock_t::duration d = clock.now() - prev_swap_time;
        inter_frame_durations.at(p.screen_no) = d;
        if(p.record_fps)
            inter_frame_durations_benchmark.at(p.screen_no).push_back(d);
        prev_swap_time = clock.now();

        frame_no += 1;
        if(frame_no == p.max_iterations)
            program_should_run = false;
    }

    release_gl_context(dpy);

    return NULL;
}

void map_windows()
{
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        screen_resources_t &sr = display_screen_resources.at(screen_no);

        // When using NVIDIA TwinView, a resize is triggered when setting the
        // fullscreen hint/property of an X window, which prevents it from
        // spanning the entire X screen. To circumvent we set the WM to ignore
        // the window.
        ignore_wm_redirect(xcb_connection, sr.x_window);

        xcb_map_window(xcb_connection, sr.x_window);
    }

    // Since the WM now ignores the window, we need to set input focus
    // manually.
    xcb_set_input_focus(xcb_connection, XCB_INPUT_FOCUS_NONE,
                        display_screen_resources[0].x_window, XCB_CURRENT_TIME);
}

void print_gui_usage()
{
    printf("Usage: \n");
    printf("       [up/down/left/right] arrows : move around\n");
    printf("       [-/+]                       : zoom\n");
    printf("       [h]                         : reset view\n");
    printf("       [v/b]                       : cycle colormap\n");
    printf("       [q/esc]                     : exit\n");
}

// Mouse state
double mouse_prevx = 0.0;
double mouse_prevy = 0.0;

typedef struct elparameters_
{
    bool no_input_mode;
    bool simulate_input_mode;
    bool print_fps;
    bool print_path;
} elparameters_t;

void *event_loop(void *args)
{
    elparameters_t p = *((elparameters_t *)args);
    // Main Event loop: (busy) wait for input -> process -> output -> etc.
    if(p.simulate_input_mode)
    {
        int step = 0;
        while(program_should_run)
        {
            mpar_event.offset_re = path_points[step][0];
            mpar_event.offset_im = path_points[step][1];
            mpar_event.range_re  = path_points[step][2];
            std::this_thread::sleep_for(std::chrono::milliseconds{250});
            step = (step + 1) % num_path_points;
        }
    }

    else
    {
        while(program_should_run)
        {
            xcb_generic_event_t *event = xcb_wait_for_event(xcb_connection);
            if(!event)
            {
                printf("Error: event i/o error.\n");
                exit(EXIT_FAILURE);
            }

            if(event->response_type == XCB_EXPOSE)
            {}

            else if (event->response_type == XCB_MOTION_NOTIFY)
            {
                xcb_motion_notify_event_t *motion_event = (xcb_motion_notify_event_t*)event;
                int screen_no = window_to_screen_no[motion_event->root];
                float xpos = motion_event->root_x + screen_order[screen_no] * 1920;
                float ypos = motion_event->root_y;
                if (motion_event->state & XCB_EVENT_MASK_BUTTON_1_MOTION)
                {
                    float dx = mouse_prevx - xpos;
                    float dy = mouse_prevy - ypos;

                    float range_re = mpar_event.range_re;
                    float range_im = range_re / 16.0*9.0;
                    mpar_event.offset_re += (dx/display_w*range_re);
                    mpar_event.offset_im -= (dy/display_h*range_im);
                }

                mouse_prevx = xpos;
                mouse_prevy = ypos;
            }

            else if(event->response_type == XCB_BUTTON_PRESS)
            {
                xcb_button_press_event_t *button_press_event = (xcb_button_press_event_t*)event;
                int screen_no = window_to_screen_no[button_press_event->root];

                const float y_step = 0.05;

                // Mouse 1 click
                if(button_press_event->detail == XCB_BUTTON_INDEX_1)
                {
                    mouse_prevx = button_press_event->root_x + screen_order[screen_no] * 1920;
                    mouse_prevy = button_press_event->root_y;
                }

                // Scrolling
                else if(button_press_event->detail == XCB_BUTTON_INDEX_4)
                {
                    float scaling = 1 - y_step;

                    // Correct offset to keep mouse at same point, i.e. explode zoom.
                    float range_im = mpar_event.range_re * (float)(display_h) / display_w;
                    float mouse_reoff = (mouse_prevx - display_w/2.0)/display_w * mpar_event.range_re;
                    float mouse_imoff = (mouse_prevy - display_h/2.0)/display_h * range_im;

                    mpar_event.offset_re -= mouse_reoff * scaling - mouse_reoff;
                    mpar_event.offset_im += mouse_imoff * scaling - mouse_imoff;

                    mpar_event.range_re *= scaling;
                }

                else if(button_press_event->detail == XCB_BUTTON_INDEX_5)
                {
                    float scaling = 1 + y_step;

                    // Correct offset to keep mouse at same point, i.e. explode zoom.
                    float range_im = mpar_event.range_re * (float)(display_h) / display_w;
                    float mouse_reoff = (mouse_prevx - display_w/2.0)/display_w * mpar_event.range_re;
                    float mouse_imoff = (mouse_prevy - display_h/2.0)/display_h * range_im;

                    mpar_event.offset_re -= mouse_reoff * scaling - mouse_reoff;
                    mpar_event.offset_im += mouse_imoff * scaling - mouse_imoff;

                    mpar_event.range_re *= scaling;
                }

            }

            else if(event->response_type == XCB_KEY_PRESS)
            {
                xcb_key_press_event_t *key_press_event = (xcb_key_press_event_t*)event;
                xcb_keysym_t keysym = xcb_get_keysym(xcb_connection, key_press_event->detail);
                const float move_frac = 0.05;
                const float y_step = 0.05;
                if(keysym == XK_equal)
                {
                    float scaling = 1 - y_step;

                    // Correct offset to keep mouse at same point, i.e. explode zoom.
                    float range_im = mpar_event.range_re * (float)(display_h) / display_w;
                    float mouse_reoff = (mouse_prevx - display_w/2.0)/display_w * mpar_event.range_re;
                    float mouse_imoff = (mouse_prevy - display_h/2.0)/display_h * range_im;

                    mpar_event.offset_re -= mouse_reoff * scaling - mouse_reoff;
                    mpar_event.offset_im += mouse_imoff * scaling - mouse_imoff;

                    mpar_event.range_re *= scaling;
                }

                else if(keysym == XK_minus)
                {
                    float scaling = 1 + y_step;

                    // Correct offset to keep mouse at same point, i.e. explode zoom.
                    float range_im = mpar_event.range_re * (float)(display_h) / display_w;
                    float mouse_reoff = (mouse_prevx - display_w/2.0)/display_w * mpar_event.range_re;
                    float mouse_imoff = (mouse_prevy - display_h/2.0)/display_h * range_im;

                    mpar_event.offset_re -= mouse_reoff * scaling - mouse_reoff;
                    mpar_event.offset_im += mouse_imoff * scaling - mouse_imoff;

                    mpar_event.range_re *= scaling;
                }
                // Offsetting/Panning
                else if(keysym == XK_Right)
                    mpar_event.offset_re += (move_frac*mpar_event.range_re);

                else if(keysym == XK_Left)
                    mpar_event.offset_re -= (move_frac*mpar_event.range_re);

                else if(keysym == XK_Up)
                    mpar_event.offset_im += (move_frac*mpar_event.range_re);

                else if(keysym == XK_Down)
                    mpar_event.offset_im -= (move_frac*mpar_event.range_re);

                // Change Colormap
                else if(keysym == XK_v)
                {
                    mpar_event.cmap_index = (mpar_event.cmap_index + 1) % cmap_list.size();
                    printf("Changed colormap to: %s:%s\n",
                            cmap_list[mpar_event.cmap_index]->getCategory().c_str(),
                            cmap_list[mpar_event.cmap_index]->getTitle().c_str());
                    generate_colormap();
                }

                else if(keysym == XK_b)
                {
                    mpar_event.cmap_index = (mpar_event.cmap_index - 1) % cmap_list.size();
                    printf("Changed colormap to: %s:%s\n",
                            cmap_list[mpar_event.cmap_index]->getCategory().c_str(),
                            cmap_list[mpar_event.cmap_index]->getTitle().c_str());
                    generate_colormap();
                }

                else if(keysym == XK_h)
                {
                    mpar_event.offset_re = -0.5;
                    mpar_event.offset_im = 0.0;
                    mpar_event.range_re = 3.2;
                }

                // Exit app.
                else if(keysym == XK_q or keysym == XK_Escape)
                    program_should_run = false;

                if(p.print_path)
                {
                    std::cout << "{" << mpar_event.offset_re << ", "
                                     << mpar_event.offset_im << ", "
                                     << mpar_event.range_re << "}, \n";
                }
            }

            else if(event->response_type == XCB_CONFIGURE_NOTIFY) {}

            fflush(stdout);
        }
    }

    return NULL;
}

void *stat_reporter_loop(void *args)
{
    while(program_should_run)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{350});

        printf("FPS: <");
        for(auto it = x_screens.begin(); it != x_screens.end(); )
        {
            const int screen_no = it->first;
            printf("%.2f",
             1000.0 / to_ms(inter_frame_durations.at(screen_no)));
            if((++it) != x_screens.end()) printf(", ");
        }
        printf(">");
        fflush(stdout);
        printf("\r");
    }
    return NULL;
}

std::vector<int> get_render_gpu_ids()
{
    std::vector<int> render_gpu_ids;
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        screen_resources_t &sr = display_screen_resources[screen_no];
        render_gpu_ids.push_back(sr.gpu_id);
    }
    return render_gpu_ids;
}

void run_gui(crlparameters_t crl_parameters, elparameters_t el_parameters,
             bool print_fps, bool benchmark)
{
    // Initiate a compute_render_loop thread for each screen
    pthread_barrier_init(&swap_barrier, NULL, x_screens.size());
    std::map<int, crlparameters_t> screen_parameters;
    pthread_t t;
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        crlparameters_t p = crl_parameters;
        p.screen_no = it.first;
        screen_parameters[it.first] = p;
        pthread_create(&t, NULL, compute_render_loop, (void *) &screen_parameters[it.first]);
        threads.push_back(t);
    }

    if(print_fps)
    {
        pthread_create(&t, NULL, stat_reporter_loop, (void *) NULL);
        threads.push_back(t);
    }

    if(not benchmark)
    {
        pthread_create(&t, NULL, event_loop, (void *) &el_parameters);
        threads.push_back(t);
    }
    // Wait until threads are done.
    for(auto &t : threads) pthread_join(t, NULL);
}

bool avx_supported()
{
    // Check for AVX/SSE2 support
    // TODO: does this check for OS support ?
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return ecx & bit_AVX;
}

const int benchmark_display_screen = 0;
void run_benchmark(runmodes runmode, implementations implementation,
                   int image_w, int image_h, int it_max, int avg_factor,
                   float o_re, float o_im, float range_re, int should_validate,
                   int cpu_threads, bool cpu_vectorize, bool use_interop,
                   int alloc_w, uchar4 *img_h)
{
    omp_set_num_threads(cpu_threads);

    int gpu_id;
    if(implementation == implementations::gpu)
        gpu_id = compute_gpu_ids[0];

    screen_resources_t sr;
    if(runmode == runmodes::benchmark_display)
    {
        sr = display_screen_resources[benchmark_display_screen];
        if(!glXMakeCurrent(dpy, sr.glx_window, sr.gl_context))
        {
            printf("Error: failed to make context current for GPU %d\n", sr.gpu_id);
            exit(EXIT_FAILURE);
        }

        glUseProgram(sr.program);
        glBindVertexArray(sr.vao);
    }

    // Constant Mandelbrot parameters
    float pix_size = range_re / image_w;
    float topleft_re = o_re - image_w / 2.0f * pix_size;
    float topleft_im = o_im + image_h / 2.0f * pix_size;

    // Compute
    benchmark_clock_t clock;
    benchmark_clock_t::duration compute_time, display_time;
    benchmark_clock_t::time_point start;
    for (int iteration = 0; iteration < avg_factor; iteration++)
    {
        start = clock.now();

        // Generate drawing on either CPU or GPU.
        if(implementation == implementations::cpu)
        {
            if(cpu_vectorize and avx_supported())
                cpu_avx_mandelbrot ((unsigned char *)img_h, image_w, image_h, alloc_w, it_max, topleft_re, topleft_im, pix_size, rgba_colormap);
            else if(cpu_vectorize)
                cpu_sse2_mandelbrot((unsigned char *)img_h, image_w, image_h, alloc_w, it_max, topleft_re, topleft_im, pix_size, rgba_colormap);
            else
                cpu_mandelbrot     ((unsigned char *)img_h, image_w, image_h, alloc_w, it_max, topleft_re, topleft_im, pix_size, rgba_colormap);
        }

        else if(implementation == implementations::gpu)
        {
            cuda_check_error(cudaSetDevice(gpu_id));

            cuda_check_error(cudaProfilerStart());

            if(use_interop)
            {
                cuda_check_error(cudaGraphicsMapResources(1, &sr.cuda_pbo_resource, default_stream));
                size_t num_bytes;
                cuda_check_error(cudaGraphicsResourceGetMappedPointer((void **)&d_imgs[gpu_id], &num_bytes, sr.cuda_pbo_resource));
            }

            mandelbrot_kernel<<<gdims[gpu_id], bdims[gpu_id]>>>(d_imgs[gpu_id], image_w, image_h, alloc_w, it_max, topleft_re, topleft_im, pix_size, rgba_colormaps_d[gpu_id]);

            // wait until done
            cuda_check_error(cudaDeviceSynchronize());

            cuda_check_error(cudaProfilerStop());
        }


        compute_time = clock.now() - start;

        // Display, if asked to do so.
        if(runmode == runmodes::benchmark_display)
        {
            start = clock.now();

            // Provide Mandelbrot results to OpenGL
            if(use_interop)
            {
                cuda_check_error(cudaGraphicsUnmapResources(1, &sr.cuda_pbo_resource, default_stream));
            }

            else
            {
                if(implementation == implementations::gpu)
                {
                    cuda_check_error(cudaMemcpy(img_h, d_imgs[gpu_id],
                                                alloc_w * image_h * sizeof(uchar4),
                                                cudaMemcpyDefault));
                    cuda_check_error(cudaDeviceSynchronize());
                }

                const GLintptr offset = 0;
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER, offset,
                                alloc_w * image_h * sizeof(uchar4),
                                img_h);
            }

            const GLint level = 0, xoffset = 0, yoffset = 0;
            glTexSubImage2D(GL_TEXTURE_2D, level, xoffset, yoffset,
                            alloc_w, image_h, GL_RGBA, GL_UNSIGNED_BYTE,
                            (char *)0);

            glDrawArrays(GL_TRIANGLES, 0, 6);
            glXSwapBuffers(dpy, sr.glx_window);
            glFinish();

            display_time = clock.now() - start;
        }

        std::cout << to_ms(compute_time);
        if(runmode == runmodes::benchmark_display)
            std::cout << "," << to_ms(display_time);
        std::cout << "\n";

        if(should_validate)
        {
            std::string filename;
            if(implementation == implementations::cpu)
            {
                filename = "mandelbrot_cpu_"
                         + std::to_string(image_w) + "_"
                         + std::to_string(image_h) + "_"
                         + std::to_string(cpu_threads) + "_"
                         + std::to_string(cpu_vectorize) + "_"
                         + std::to_string(iteration);
            }
            else if(implementation == implementations::gpu)
            {
                cuda_check_error(cudaMemcpy(img_h, d_imgs[gpu_id],
                                        alloc_w * image_h * sizeof(uchar4),
                                        cudaMemcpyDefault));
                cuda_check_error(cudaDeviceSynchronize());
                filename = "mandelbrot_gpu_"
                         + std::to_string(image_w) + "_"
                         + std::to_string(image_h) + "_"
                         + std::to_string(use_interop) + "_"
                         + std::to_string(iteration);
            }
            filename += ".ppm";
            rgba_to_ppm(filename.c_str(), (unsigned char*)img_h,
                        image_w, image_h, alloc_w);
        }
    }

    if(runmode == runmodes::benchmark_display)
        release_gl_context(dpy);
}

void print_usage()
{
    printf("Usage: mandelbrot gui [options]\n");
    printf("                  benchmark cpu|gpu [options]\n");
}

int main(int argc, char const *argv[])
{
    // parse input arguments
    if(argc < 2)
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    // global settings (for gui+benchmark)
    runmodes runmode;
    implementations implementation = implementations::gpu;
    int it_max = 255;
    bool use_interop = false;
    bool should_vectorize = false;
    bool print_fps = false;
    bool print_path = false;
    bool no_input_mode = false;
    bool simulate_input_mode = false;

    // benchmark only settings
    int benchmark_gpucount = 1;
    int benchmark_cputhreads = omp_get_max_threads();
    int benchmark_avg_factor = 25;
    int benchmark_image_w = 1920;
    int benchmark_image_h = 1080;
    float benchmark_o_re = -0.5f;
    float benchmark_o_im = 0.0f;
    float benchmark_range_re = 3.2f;
    bool benchmark_validate = false;

    if(std::string(argv[1]) == "gui")
    {
        print_gui_usage();
        use_interop = true;
        runmode = runmodes::gui;
        for (int arg_no = 2; arg_no < argc; arg_no++)
        {
            if(std::string(argv[arg_no]) == "no_input")
                no_input_mode = true;
            else if(std::string(argv[arg_no]) == "simulate_input")
                simulate_input_mode = true;
            else if(std::string(argv[arg_no]) == "no_interop")
                use_interop = false;
            else if(std::string(argv[arg_no]) == "print_fps")
                print_fps = true;
            else if(std::string(argv[arg_no]) == "print_path")
                print_path = true;
            else
            {
                printf("Error: %s is not recognized as setting.\n", argv[arg_no]);
                exit(EXIT_FAILURE);
            }
        }
    }

    else if(std::string(argv[1]) == "benchmark")
    {
        runmode = runmodes::benchmark_compute;

        if(std::string(argv[2]) == "cpu")
        {
            implementation = implementations::cpu;
        }

        else if(std::string(argv[2]) == "gpu")
        {
            implementation = implementations::gpu;
        }

        else
        {
            print_usage();
            exit(EXIT_FAILURE);
        }


        for (int arg_no = 3; arg_no < argc; arg_no++)
        {
            if(std::string(argv[arg_no]) == "num_gpus")
            {
                benchmark_gpucount = std::stoi(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "cpu_threads")
            {
                benchmark_cputhreads = std::stoi(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "vectorize")
            {
                should_vectorize = (std::stoi(argv[arg_no + 1]) == 1);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "benchmark_display")
            {
                runmode = runmodes::benchmark_display;
            }

            else if(std::string(argv[arg_no]) == "benchmark_gui")
            {
                runmode = runmodes::benchmark_gui;
            }

            else if(std::string(argv[arg_no]) == "o_re")
            {
                benchmark_o_re = std::stof(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "o_im")
            {
                benchmark_o_im = std::stof(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "range_re")
            {
                benchmark_range_re = std::stof(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "interop")
            {
                use_interop = std::stoi(argv[arg_no + 1]) == 1;
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "should_validate")
            {
                benchmark_validate = true;
            }

            else if(std::string(argv[arg_no]) == "avg_factor")
            {
                benchmark_avg_factor = std::stoi(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "w")
            {
                benchmark_image_w = std::stoi(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "h")
            {
                benchmark_image_h = std::stoi(argv[arg_no + 1]);
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "it_max")
            {
                it_max = std::stoi(argv[arg_no + 1]);
                arg_no += 1;
            }

            else
            {
                printf("Error: %s is not recognized as setting.\n", argv[arg_no]);
                exit(EXIT_FAILURE);
            }
        }
    }

    else
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    if(it_max > 255)
    {
        printf("Error: it_max > 255 not supported\n");
        exit(EXIT_FAILURE);
    }

    bool will_display = runmode == runmodes::gui or
                        runmode == runmodes::benchmark_gui or
                        runmode == runmodes::benchmark_display;

    // initialize X/OpenGL if needed
    if(will_display)
    {
        init_xlib();
        init_xcb();
        load_glx(dpy);
        init_screens();
        create_xwindows();
        create_glxwindows();
        init_gl();
        map_windows();
    }

    // determine GPUs to use for compute
    if(implementation == implementations::gpu)
    {
        if(runmode == runmodes::gui or runmode == runmodes::benchmark_gui)
            compute_gpu_ids = get_render_gpu_ids();
        else
            compute_gpu_ids = claim_cuda_devices(benchmark_gpucount);
    }

    // determine size of Mandelbrot image
    int image_w, image_h;
    if(runmode == runmodes::gui)
    {
        image_w = x_screens[0].width_in_pixels;
        image_h = x_screens[0].height_in_pixels;
    }

    else
    {
        image_w = benchmark_image_w;
        image_h = benchmark_image_h;
    }

    // how many bytes to allocate
    int alloc_w = image_w;
    int alloc_h = image_h;
    // pad alloc_w to allow for coalescing
    while(alloc_w % 32) alloc_w++;
    size_t alloc_size = alloc_w * alloc_h * sizeof(uchar4);

    // CPU memory allocation (for run_benchmark())
    uchar4 *img_h = (uchar4 *) malloc(alloc_size);

    // OpenGL memory allocation
    if(will_display)
    {
        for(std::pair<int, xcb_screen_t> it : x_screens)
        {
            const int screen_no = it.first;
            screen_resources_t &sr = display_screen_resources[screen_no];

            glXMakeCurrent(dpy, sr.glx_window, sr.gl_context);

            // texture
            glBindTexture(GL_TEXTURE_2D, sr.tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, alloc_w, image_h, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, NULL);

            // pixel buffer object (pbo)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sr.pbo);
            GLenum pbo_usage = use_interop ? GL_DYNAMIC_COPY : GL_DYNAMIC_DRAW;
            glBufferData(GL_PIXEL_UNPACK_BUFFER, alloc_size, NULL, pbo_usage);
            release_gl_context(dpy);
        }
    }

    // CUDA memory allocation
    if(runmode == runmodes::benchmark_compute or not use_interop)
    {
        for(int gpu_id : compute_gpu_ids)
        {
            cuda_check_error(cudaSetDevice(gpu_id));
            cuda_check_error(cudaMalloc(&d_imgs[gpu_id], alloc_size));

            // allocate cpu memory for cuda->cpu->gl transfer
            if(will_display and not use_interop)
                h_imgs[gpu_id] = (uchar4 *) malloc(alloc_size);
        }
    }

    // additional GPU configuration/allocation
    for(int gpu_id : compute_gpu_ids)
    {
        cuda_check_error(cudaSetDevice(gpu_id));
        cuda_check_error(cudaMalloc(&rgba_colormaps_d[gpu_id], (it_max+1) * 4));

        // use device properties to determine launch configs
        cudaDeviceProp dprop;
        cuda_check_error(cudaGetDeviceProperties(&dprop, gpu_id));

        bdims[gpu_id] = dim3(32, 4);

        // use as many blocks as possible, but not more then needed.
        gdims[gpu_id] =
            dim3(min(divup(image_w, bdims[gpu_id].x), dprop.maxGridSize[0]),
                 min(divup(image_h, bdims[gpu_id].y), dprop.maxGridSize[1]));
    }

    if(will_display and use_interop) init_interop();

    // allocate and initialize colormap
    rgba_colormap = new unsigned char[(it_max+1) * 4];
    generate_colormap();
    upload_colormap();

    // start Mandelbrot visualization

    // compute_render_loop
    crlparameters_t crl_commonp;
    crl_commonp.use_interop = use_interop;
    crl_commonp.no_input_mode = no_input_mode;
    crl_commonp.it_max = it_max;
    crl_commonp.alloc_w = alloc_w;

    // event_loop
    elparameters_t el_commonp;
    el_commonp.no_input_mode = no_input_mode;
    el_commonp.simulate_input_mode = simulate_input_mode;
    el_commonp.print_path = print_path;
    el_commonp.print_fps = print_fps;

    if(runmode == runmodes::gui)
    {
        crlparameters_t crl_parameters = crl_commonp;
        crl_parameters.record_fps = false;
        crl_parameters.max_iterations = 0;
        run_gui(crl_parameters, el_commonp, print_fps, false);
    }

    else if(runmode == runmodes::benchmark_gui)
    {
        crlparameters_t crl_parameters = crl_commonp;
        crl_parameters.record_fps = true;
        crl_parameters.max_iterations = benchmark_avg_factor;
        mpar_event.offset_re = benchmark_o_re;
        mpar_event.offset_im = benchmark_o_im;
        mpar_event.range_re = benchmark_range_re;
        run_gui(crl_parameters, el_commonp, print_fps, true);

        // print inter_frame_durations, skip dummy (i.e. i = 0)
        for(int i = 1; i <= benchmark_avg_factor; ++i)
        {
            for(auto it = x_screens.begin(); it != x_screens.end(); ++it)
            {
                int screen_no = it->first;
                printf("%s%f", it != x_screens.begin() ? "," : "",
                               to_ms(inter_frame_durations_benchmark[screen_no][i]));
            }
            printf("\n");
        }
    }

    else if(runmode == runmodes::benchmark_compute or
            runmode == runmodes::benchmark_display)
    {
        run_benchmark(runmode, implementation,
                      benchmark_image_w, benchmark_image_h, it_max,
                      benchmark_avg_factor, benchmark_o_re, benchmark_o_im,
                      benchmark_range_re, benchmark_validate,
                      benchmark_cputhreads, should_vectorize,
                      use_interop, alloc_w, img_h);
    }

    // free memory
    // cleanup();
    exit(EXIT_SUCCESS);
}
