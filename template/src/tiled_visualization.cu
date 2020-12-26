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
#include "benchmark_setup.hpp"

// application state
bool program_should_run = true;
Display *dpy; // Xlib handle to X
xcb_connection_t *xcb_connection; // xcb handle to X
std::vector<pthread_t> threads;
pthread_barrier_t swap_barrier;

// what to render
typedef struct view_parameters
{} view_parameters_t;
view_parameters_t vpar_render, vpar_event;

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

// function definitions
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

        /*

            // Initialize OpenGL here.

        */

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

        /*

            // Register OpenGL objects for this screen with CUDA runtime
            cuda_check_error(cudaGraphicsGLRegisterBuffer(..., ..., ...));

        */

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
} crlparameters_t;

void *compute_render_loop(void* args)
{
    crlparameters_t p = *((crlparameters_t *) args);

    // Obtain information of screen to render to
    xcb_screen_t screen = x_screens.at(p.screen_no);
    screen_resources_t &sr = display_screen_resources.at(p.screen_no);
    int screen_count = xcb_setup_roots_iterator(xcb_get_setup(xcb_connection)).rem;

    // int image_w = screen.width_in_pixels * screen_count;
    // int image_h = screen.height_in_pixels;

    // Obtain (properties of) gpu that does rendering.
    cuda_check_error(cudaSetDevice(sr.gpu_id));

    // Setup OpenGL
    if(!glXMakeCurrent(dpy, sr.glx_window, sr.gl_context))
    {
        printf("Error: couldn't make context current for screen %d.\n",
               p.screen_no);
        exit(EXIT_FAILURE);
    }

    /*

        // Possibly bind OpenGL objects here

    */

    // Start compute+render loop
    int frame_no = 0;

    // To measure FPS
    benchmark_clock_t clock;
    benchmark_clock_t::time_point prev_swap_time = clock.now();
    while(program_should_run)
    {
        // Map OpenGL stuff into CUDA memory.
        if(p.use_interop)
        {
            /*
                cuda_check_error(cudaGraphicsMapResources(..., ..., ...));
                size_t num_bytes;
                cuda_check_error(cudaGraphicsResourceGetMappedPointer(..., &num_bytes, ...));
            */
        }

        // Compute


        // Render

        if(p.use_interop)
        {
            // We are done, just unmap OpenGL resources
            // cuda_check_error(cudaGraphicsUnmapResources(..., ..., ...));

        }

        // glDraw...()

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
            vpar_render = vpar_event;
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

// Mouse state
double mouse_prevx = 0.0;
double mouse_prevy = 0.0;

typedef struct elparameters_
{
    bool print_fps;
    bool print_path;
} elparameters_t;

void *event_loop(void *args)
{
    elparameters_t p = *((elparameters_t *)args);

    // Main Event loop: (busy) wait for input -> process -> output -> etc.
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

            mouse_prevx = xpos;
            mouse_prevy = ypos;
        }

        else if(event->response_type == XCB_BUTTON_PRESS)
        {
            xcb_button_press_event_t *button_press_event = (xcb_button_press_event_t*)event;
            int screen_no = window_to_screen_no[button_press_event->root];

            // Mouse 1 click
            if(button_press_event->detail == XCB_BUTTON_INDEX_1)
            {
                mouse_prevx = button_press_event->root_x + screen_order[screen_no] * 1920;
                mouse_prevy = button_press_event->root_y;
            }

            // Scrolling
            else if(button_press_event->detail == XCB_BUTTON_INDEX_4)
            {
            }

            else if(button_press_event->detail == XCB_BUTTON_INDEX_5)
            {
            }

        }

        else if(event->response_type == XCB_KEY_PRESS)
        {
            xcb_key_press_event_t *key_press_event = (xcb_key_press_event_t*)event;
            xcb_keysym_t keysym = xcb_get_keysym(xcb_connection, key_press_event->detail);
            if(keysym == XK_equal) {}

            else if(keysym == XK_minus) {}

            else if(keysym == XK_Right) {}

            else if(keysym == XK_Left) {}

            else if(keysym == XK_Up) {}

            else if(keysym == XK_Down) {}

            // Exit app.
            else if(keysym == XK_q or keysym == XK_Escape)
                program_should_run = false;

        }

        else if(event->response_type == XCB_CONFIGURE_NOTIFY) {}

        fflush(stdout);
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

int main(int argc, char const *argv[])
{
    bool benchmark = false;
    int benchmark_avg_factor = 25;
    bool use_interop = true;
    bool print_fps = false;
    for (int arg_no = 1; arg_no < argc; arg_no++)
    {
        if(std::string(argv[arg_no]) == "no_interop")
            use_interop = false;

        else if(std::string(argv[arg_no]) == "print_fps")
            print_fps = true;

        else if(std::string(argv[arg_no]) == "avg_factor")
        {
            benchmark_avg_factor = std::stoi(argv[arg_no+1]);
            arg_no += 1;
        }

        else
        {
            printf("Error: %s is not recognized as setting.\n", argv[arg_no]);
            exit(EXIT_FAILURE);
        }
    }

    init_xlib();
    init_xcb();
    load_glx(dpy);
    init_screens();
    create_xwindows();
    create_glxwindows();
    init_gl();
    map_windows();

    compute_gpu_ids = get_render_gpu_ids();

    if(use_interop) init_interop();

    // compute_render_loop
    crlparameters_t crl_commonp;
    crl_commonp.use_interop = use_interop;

    // event_loop
    elparameters_t el_commonp;
    el_commonp.print_fps = print_fps;

    crlparameters_t crl_parameters = crl_commonp;

    if(benchmark)
    {
        crl_parameters.record_fps = true;
        crl_parameters.max_iterations = benchmark_avg_factor;
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

    else
    {
        crl_parameters.record_fps = false;
        crl_parameters.max_iterations = 0;
        run_gui(crl_parameters, el_commonp, print_fps, false);
    }


    exit(EXIT_SUCCESS);
}
