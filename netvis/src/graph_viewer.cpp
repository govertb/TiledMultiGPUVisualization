/*
 ==============================================================================

 graph_viewer.cpp
 Copyright (C) 2016, 2017, 2018  G. Brinkmann

 This file is part of graph_viewer.

 graph_viewer is free software: you can redistribute it and/or modify
 it under the terms of version 3 of the GNU Affero General Public License as
 published by the Free Software Foundation.

 graph_viewer is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with graph_viewer.  If not, see <https://www.gnu.org/licenses/>.

-------------------------------------------------------------------------------

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <chrono>
#include <thread>
#include <vector>
#include <map>
#include <sstream>
#include <iterator>

// Graphics Library through glad
#include "../lib/glad/glad.h"
#include "../lib/glad/glad_glx.h"

#ifdef __NVCC__
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#endif


#include "../lib/glm/glm/glm.hpp"
#include "../lib/glm/glm/gtc/matrix_transform.hpp"
#include "../lib/glm/glm/gtc/type_ptr.hpp"
#include "../lib/colormap/include/colormap/colormap.h"

// Window System
#include <X11/Xlib.h>
#include <X11/Xlib-xcb.h>
#include <xcb/xcb.h>
#include "x11_util.h"
#include "RPCommon.hpp"

#include "gl_util.hpp"
#include "RPGraph.hpp"
#include "RPGraphLayout.hpp"
#include "RPLayoutAlgorithm.hpp"
#include "RPCPUForceAtlas2.hpp"
#include "RPGPUForceAtlas2.hpp"
#include "RPForceAtlas2.hpp"
#include "RPCommon.hpp"
#include "benchmark_setup.hpp"

enum class runmodes {gui, benchmark_layout, benchmark_layout_render,
                     benchmark_gui};

// application state
bool program_should_run = true;
Display *dpy; // Xlib handle to X
xcb_connection_t *xcb_connection; // xcb handle to X
bool layout_paused = false;
std::vector<pthread_t> threads;
pthread_barrier_t swap_barrier, reduction_barrier;

// view state
typedef struct view_parameters
{
    glm::mat4 global_view;
    int cmap_index = 35;
    float node_opacity = 1.0;
    float edge_opacity = 1.0;
    float mx = 0;
    float my = 0;
    float z = 1;
    float x_off = 0;
    float y_off = 0;
} view_parameters_t;
view_parameters_t vpar_render, vpar_event;

// GL/network drawing stuff
enum VAO_IDs {Nodes, Edges, NumVAOs};
enum Buffer_IDs {NodePos, EdgePos, EdgePairs, NodeColors, Uniforms, NumBuffers};
std::vector<glm::vec4> node_colors;
std::vector<std::shared_ptr<colormap::Colormap const>> cmap_list =
    colormap::ColormapList::getAll();

// display/X info
int display_w = 0;
int display_h = 0;
std::map<int, xcb_screen_t> x_screens;
const std::vector<int> bigeye_screen_order = {0, 2, 1};

// for each screen (screen_no as key):
std::map<int, int> swap_ready;
typedef struct screen_resources
{
    xcb_window_t x_window;
    GLXWindow glx_window;
    GLXContext gl_context;
    GLXFBConfig* fb_configs;
    int gpu_id;
    GLuint VAOs[NumVAOs], Buffers[NumBuffers], node_program, edge_program;
    cudaGraphicsResource_t cuda_vbo_r;
    float view_offset;
} screen_resources_t;
std::map<int, screen_resources_t> display_screen_resources;
std::map<xcb_window_t, int> window_to_screen_no;

// per screen, benchmarks
std::map<int, benchmark_clock_t::duration> inter_frame_durations;
std::map<int, std::vector<benchmark_clock_t::duration>> inter_frame_durations_benchmark;

// gpu properties and memory pointers, indexed by GPU ID.
std::map<int, cudaDeviceProp> gpu_properties;


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

void load_glx()
{
    int screen_no = 0;
    if(!gladLoadGLX(dpy, screen_no))
    {
        printf("Error: couldn't load GLX through glad.\n");
        exit(EXIT_FAILURE);
    }
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
        GLXFBConfig *matches = glXChooseFBConfig(dpy, screen_no, fb_attrib_list, &num_matches);
        if(matches && num_matches > 0)
        {
            sr.fb_configs = matches;
        }
        else
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

        const xcb_window_t window = xcb_generate_id(xcb_connection);
        xcb_create_window(xcb_connection, XCB_COPY_FROM_PARENT, window, parent,
            pos_x, pos_y, screen.width_in_pixels, screen.height_in_pixels,
            border_width, XCB_WINDOW_CLASS_INPUT_OUTPUT, visual_id, valuemask,
            valuelist);

        if(window)
        {
            sr.x_window = window;
        }
        else
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
        GLXWindow glx_window = glXCreateWindow(dpy, sr.fb_configs[0], parent_window, NULL);
        if(glx_window)
        {
            sr.glx_window = glx_window;
        }
        else
        {
            printf("Error: couldn't create GLXWindow for screen %d.\n", screen_no);
            exit(EXIT_FAILURE);
        }
    }
}

bool gl_loaded = false;
void load_gl()
{
    if(gladLoadGLLoader((GLADloadproc) glXGetProcAddress))
    {
        gl_loaded = true;
    }

    else
    {
        printf("Error: couldn't load GL through glad\n");
        exit(EXIT_FAILURE);
    }
}

void init_gl(RPGraph::UGraph &graph, RPGraph::GraphLayout &layout)
{
    int screen_count = x_screens.size();
    float gl_width = 2.0 * screen_count;
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        const int screen_no = it.first;
        const xcb_screen_t screen = it.second;
        screen_resources_t &sr = display_screen_resources[screen_no];

        // Create an OpenGL context.
        const int render_type = GLX_RGBA_TYPE;
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

        // Note what GPU is used for this context
        int gpu_count;
        cudaCatchError(cudaGetDeviceCount(&gpu_count));

        uint num_devices;
        int *devices = new int[gpu_count];
        cudaCatchError(cudaGLGetDevices(&num_devices, devices,
                                        gpu_count, cudaGLDeviceListAll));

        if(num_devices != 1)
        {
            printf("Error: found %d CUDA devices for screen %d\n",
                   num_devices, screen_no);
            exit(EXIT_FAILURE);
        }

        sr.gpu_id = devices[0];

        set_gl_base_settings();

        // Buffers
        glGenBuffers(NumBuffers, sr.Buffers);

        // VAOs
        glGenVertexArrays(NumVAOs, sr.VAOs);

        // Programs
        sr.node_program = gl_node_program();
        sr.edge_program = gl_edge_program();

        // Uniforms
        glBindBuffer(GL_UNIFORM_BUFFER, sr.Buffers[Uniforms]);

        // Bind both programs their uniform blocks and the context's
        // uniform buffer to binding point with index 0
        int binding_point_index = 0;

        // i.) the programs' uniform blocks
        GLuint program_block_idx;
        program_block_idx = glGetUniformBlockIndex(sr.node_program, "GlobalMatrices");
        glUniformBlockBinding(sr.node_program, program_block_idx, binding_point_index);
        program_block_idx = glGetUniformBlockIndex(sr.edge_program, "GlobalMatrices");
        glUniformBlockBinding(sr.edge_program, program_block_idx, binding_point_index);

        // ii.) the Uniform Buffer Object that holds the data
        glBindBufferRange(GL_UNIFORM_BUFFER, binding_point_index,
                          sr.Buffers[Uniforms], 0, 2 * sizeof(glm::mat4) + sizeof(glm::vec2));

        glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4) + sizeof(glm::vec2), NULL, GL_STREAM_DRAW);
        glm::mat4 model = glm::scale(glm::mat4(), glm::vec3(screen_count*1.75/layout.getWidth(), 1.75/layout.getHeight(), 1.0));
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(model), glm::value_ptr(model));
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(vpar_render.global_view), glm::value_ptr(vpar_render.global_view));
        glm::vec2 vp = glm::vec2(screen.width_in_pixels, screen.height_in_pixels);
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(model) + sizeof(vpar_render.global_view), sizeof(vp), glm::value_ptr(vp));

        // Setup bindings
        glUseProgram(sr.node_program);
        glBindVertexArray(sr.VAOs[Nodes]);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, sr.Buffers[NodePos]);
        glBufferData(GL_ARRAY_BUFFER, graph.num_nodes() * sizeof(float2), layout.getNodeCoordinates(), GL_STREAM_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, sr.Buffers[NodeColors]);
        glBufferData(GL_ARRAY_BUFFER, graph.num_nodes() * sizeof(glm::vec4), &node_colors[0].x, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
        glUniform1f(glGetUniformLocation(sr.node_program, "node_opacity"),
                    vpar_render.node_opacity);

        // Element buffer
        glUseProgram(sr.edge_program);
        glBindVertexArray(sr.VAOs[Edges]);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, sr.Buffers[NodeColors]);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, sr.Buffers[NodePos]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sr.Buffers[EdgePairs]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, graph.num_edges() * sizeof(uint2), (uint *)&graph.edges[0], GL_STATIC_DRAW);
        // following two lines are for non-indexed edge drawing
        // glBindBuffer(GL_ARRAY_BUFFER, sr.Buffers[sr.EdgePos]);
        // glBufferData(GL_ARRAY_BUFFER, num_edges * sizeof(float2), edges, GL_STREAM_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        glUniform1f(glGetUniformLocation(sr.edge_program, "edge_opacity"),
                    vpar_render.edge_opacity);

        sr.view_offset = (screen_count - 1.0) - (gl_width/screen_count*bigeye_screen_order[screen_no]);
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

        //  Make the CUDA device associated with GL context current and
        cudaCatchError(cudaSetDevice(sr.gpu_id));

        //  Register VBO for this screen and context with CUDA runtime.
        cudaCatchError(cudaGraphicsGLRegisterBuffer(&sr.cuda_vbo_r,
                                                    sr.Buffers[NodePos],
                                                    cudaGraphicsRegisterFlagsNone));

        release_gl_context(dpy);
    }
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

void update_view()
{
    vpar_event.global_view = glm::scale(glm::mat4(), glm::vec3(vpar_event.z, vpar_event.z, 0.0f));
    vpar_event.global_view = glm::translate(vpar_event.global_view, glm::vec3(vpar_event.x_off, vpar_event.y_off, 0.0f));
}

// Mouse state
double mouse_prevx = 0.0;
double mouse_prevy = 0.0;
void process_x_events(RPGraph::ForceAtlas2 *fa2)
{
    int screen_count = x_screens.size();
    xcb_generic_event_t *event = xcb_wait_for_event(xcb_connection);
    float gl_width = 2.0 * screen_count;
    float gl_height = 2.0;
    float move_offset = 0.05;
    const float y_step = 0.05;

    if(!event)
    {
        printf("Error: event i/o error.\n");
        exit(EXIT_FAILURE);
    }

    if(event->response_type == XCB_EXPOSE)
    {
        // printf("XCB_EXPOSE\n");
    }

    else if(event->response_type == XCB_CONFIGURE_NOTIFY)
    {
        // printf("XCB_CONFIGURE_NOTIFY\n");
    }

    else if (event->response_type == XCB_MOTION_NOTIFY)
    {
        xcb_motion_notify_event_t *motion_event = (xcb_motion_notify_event_t*)event;
        int screen_no = window_to_screen_no[motion_event->root];
        float xpos = motion_event->root_x + bigeye_screen_order[screen_no] * 1920;
        float ypos = motion_event->root_y;
        if (motion_event->state & XCB_EVENT_MASK_BUTTON_1_MOTION)
        {
            // *2 because of [-1, 1] OpenGL range
            double dx = -(mouse_prevx-xpos) / display_w * gl_width  / vpar_event.z;
            double dy =  (mouse_prevy-ypos) / display_h * gl_height / vpar_event.z;
            vpar_event.x_off += dx;
            vpar_event.y_off += dy;
        }

        mouse_prevx = xpos;
        mouse_prevy = ypos;
    }

    else if(event->response_type == XCB_BUTTON_PRESS)
    {
        xcb_button_press_event_t *button_press_event = (xcb_button_press_event_t*)event;
        int screen_no = window_to_screen_no[button_press_event->root];

        if(button_press_event->detail == XCB_BUTTON_INDEX_1)
        {
            mouse_prevx = button_press_event->root_x + bigeye_screen_order[screen_no] * 1920;
            mouse_prevy = button_press_event->root_y;
        }

        // Scrolling
        else if(button_press_event->detail == XCB_BUTTON_INDEX_4)
        {
            float scaling = (1 + y_step);

            // Correct offset to keep mouse at same point, i.e. explode zoom.
            float mouse_xoff = (mouse_prevx - display_w/2.0)/display_w * gl_width / vpar_event.z;
            float mouse_yoff = (mouse_prevy - display_h/2.0)/display_h * gl_height / vpar_event.z;
            vpar_event.x_off -= mouse_xoff * scaling - mouse_xoff;
            vpar_event.y_off += mouse_yoff * scaling - mouse_yoff;
            vpar_event.z *= scaling;
        }

        else if(button_press_event->detail == XCB_BUTTON_INDEX_5)
        {
            float scaling = (1 - y_step);

            // Correct offset to keep mouse at same point, i.e. explode zoom.
            float mouse_xoff = (mouse_prevx - display_w/2.0)/display_w * gl_width / vpar_event.z;
            float mouse_yoff = (mouse_prevy - display_h/2.0)/display_h * gl_height / vpar_event.z;
            vpar_event.x_off -= mouse_xoff * scaling - mouse_xoff;
            vpar_event.y_off += mouse_yoff * scaling - mouse_yoff;
            vpar_event.z *= scaling;
        }
    }

    else if(event->response_type == XCB_KEY_PRESS)
    {
        xcb_key_press_event_t *key_press_event = (xcb_key_press_event_t*)event;
        xcb_keysym_t keysym = xcb_get_keysym(xcb_connection, key_press_event->detail);

        /* From here on, only actual key events */
        if(keysym == XK_Escape)
        {
            fa2->layout.randomizePositions();
            // ((RPGraph::CUDAForceAtlas2 *) fa2)->sendGraphToGPU();
        }

        if(keysym == XK_r and not (key_press_event->state & XK_Shift_L))
            fa2->k_r+=1;

        if(keysym == XK_r and (key_press_event->state & XK_Shift_L))
            fa2->k_r-=1;

        if(keysym == XK_g and not (key_press_event->state & XK_Shift_L))
            fa2->k_g+=1;

        if(keysym == XK_g and (key_press_event->state & XK_Shift_L))
            fa2->k_g-=1;

        if(keysym == XK_g and (key_press_event->state & XK_Alt_L))
            fa2->strong_gravity = not fa2->strong_gravity;

        // Scaling/Zooming
        if(keysym == XK_equal)
        {
            float scaling = (1 + y_step);

            // Correct offset to keep mouse at same point, i.e. explode zoom.
            float mouse_xoff = (mouse_prevx - display_w/2.0)/display_w * gl_width / vpar_event.z;
            float mouse_yoff = (mouse_prevy - display_h/2.0)/display_h * gl_height / vpar_event.z;
            vpar_event.x_off -= mouse_xoff * scaling - mouse_xoff;
            vpar_event.y_off += mouse_yoff * scaling - mouse_yoff;
            vpar_event.z *= scaling;
        }

        if(keysym == XK_minus)
        {
            float scaling = (1 - y_step);

            // Correct offset to keep mouse at same point, i.e. explode zoom.
            float mouse_xoff = (mouse_prevx - display_w/2.0)/display_w * gl_width / vpar_event.z;
            float mouse_yoff = (mouse_prevy - display_h/2.0)/display_h * gl_height / vpar_event.z;
            vpar_event.x_off -= mouse_xoff * scaling - mouse_xoff;
            vpar_event.y_off += mouse_yoff * scaling - mouse_yoff;
            vpar_event.z *= scaling;
        }

        // Offsetting/Panning
        if(keysym == XK_Right)
            vpar_event.x_off -= move_offset * screen_count;

        if(keysym == XK_Left)
            vpar_event.x_off += move_offset * screen_count;

        if(keysym == XK_Up)
            vpar_event.y_off -= move_offset;

        if(keysym == XK_Down)
            vpar_event.y_off += move_offset;

        // Go home / reset view
        if(keysym == XK_h)
        {
            vpar_event.z = 1;
            vpar_event.x_off = 0;
            vpar_event.y_off = 0;
        }

        if(keysym == XK_p)
            layout_paused = not layout_paused;

        if(keysym == XK_z)
            fa2->mouse_repulse = not fa2->mouse_repulse;

        if(keysym == XK_a and not (key_press_event->state & XK_Shift_L))
            fa2->mouse_mass *= 1.25;

        if(keysym == XK_a and (key_press_event->state & XK_Shift_L))
            fa2->mouse_mass /= 1.25;

        if(keysym == XK_x)
            fa2->mouse_heat = not fa2->mouse_heat;

        if(keysym == XK_s and not (key_press_event->state & XK_Shift_L))
            fa2->mouse_temp *= 10;

        if(keysym == XK_s and (key_press_event->state & XK_Shift_L))
            fa2->mouse_temp /= 10;

        if(keysym == XK_k)
            vpar_event.edge_opacity /= 1.25;

        if(keysym == XK_l)
            vpar_event.edge_opacity *= 1.25;

        if(keysym == XK_n)
            vpar_event.node_opacity /= 1.25;

        if(keysym == XK_m)
            vpar_event.node_opacity *= 1.25;

        // Change Colormap
        if(keysym == XK_v)
        {
            printf("Changed colormap to: %s:%s\n",
                   cmap_list[vpar_event.cmap_index]->getCategory().c_str(),
                   cmap_list[vpar_event.cmap_index]->getTitle().c_str());
            vpar_event.cmap_index = (vpar_event.cmap_index - 1) % cmap_list.size();
        }

        if(keysym == XK_b)
        {
            printf("Changed colormap to: %s:%s\n",
                   cmap_list[vpar_event.cmap_index]->getCategory().c_str(),
                   cmap_list[vpar_event.cmap_index]->getTitle().c_str());
            vpar_event.cmap_index = (vpar_event.cmap_index + 1) % cmap_list.size();
        }

        // Exit app.
        if(keysym == XK_q)
            program_should_run = false;

        fflush(stdout);
    }
    update_view();

    free(event);
}

typedef struct crlparameters_
{
    int screen_no;
    bool use_interop;
    RPGraph::ForceAtlas2 *fa2;
    int max_iterations;
    bool adapt_view;
    bool record_fps;
} crlparameters_t;

void *compute_render_loop(void *args)
{
    crlparameters_t p = *((crlparameters_t *)args);

    // Obtain information of screen to render to
    const uint screen_count = x_screens.size();
    const xcb_screen_t screen = x_screens.at(p.screen_no);
    screen_resources_t &sr = display_screen_resources.at(p.screen_no);

    float gl_width = 2.0 * screen_count;
    float gl_height = 2.0;

    // Obtain (properties of) gpu that does rendering.
    cudaCatchError(cudaSetDevice(sr.gpu_id));
    const cudaDeviceProp device_properties = gpu_properties.at(sr.gpu_id);

    // OpenGL
    if(!glXMakeCurrent(dpy, sr.glx_window, sr.gl_context))
    {
        printf("Error: couldn't make context current for screen %d.\n", p.screen_no);
        exit(EXIT_FAILURE);
    }

    int prev_cmap = vpar_render.cmap_index;

    // Start compute+render loop
    int frame_no = 0;
    benchmark_clock_t clock;
    benchmark_clock_t::time_point prev_swap_time = clock.now();
    while(program_should_run)
    {
        /* COMPUTE */
        if(not layout_paused)
        {
            // Advance the layout.
            if (p.use_interop)
            {
                cudaCatchError(cudaGraphicsMapResources(1, &sr.cuda_vbo_r, 0));
                size_t num_bytes;
                float2 *body_posl;
                cudaCatchError(cudaGraphicsResourceGetMappedPointer((void **)&body_posl, &num_bytes, sr.cuda_vbo_r));
                ((RPGraph::CUDAForceAtlas2 *)p.fa2)->body_posl.at(sr.gpu_id) = body_posl;
            }

            if(sr.gpu_id == ((RPGraph::CUDAForceAtlas2 *)p.fa2)->master_gpu)
            {
                ((RPGraph::CUDAForceAtlas2 *)p.fa2)->doStep();
                cudaCatchError(cudaSetDevice(sr.gpu_id));


                if(p.adapt_view)
                    ((RPGraph::CUDAForceAtlas2 *)p.fa2)->sync_layout_bounds();

                if(not p.use_interop)
                    p.fa2->sync_layout();
            }

            pthread_barrier_wait(&reduction_barrier);

            // Make compute results available for drawing
            if(p.use_interop)
                cudaCatchError(cudaGraphicsUnmapResources(1, &sr.cuda_vbo_r, NULL));
        }

        // Check if colormap was changed from previous iteration.
        if(prev_cmap != vpar_render.cmap_index)
        {
            std::shared_ptr<colormap::Colormap const> cmap = cmap_list[vpar_render.cmap_index];
            RPGraph::nid_t max_degree = p.fa2->layout.graph.max_degree();
            for(int n = 0; n < p.fa2->layout.graph.num_nodes(); ++n)
            {
                colormap::Color c = cmap->getColor(p.fa2->layout.graph.degree(n)/(float)max_degree);
                node_colors[n] = glm::vec4(c.r, c.g, c.b, c.a);
            }

            glBindBuffer(GL_ARRAY_BUFFER, sr.Buffers[NodeColors]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, p.fa2->layout.graph.num_nodes() * sizeof(glm::vec4), &node_colors[0].x);
            prev_cmap = vpar_render.cmap_index;
        }

        float minx, maxx, miny, maxy, xrange, yrange;
        if(p.adapt_view)
        {
            // Update view, and ensure graph fully viewed
            minx = p.fa2->layout.gpu_minx;
            maxx = p.fa2->layout.gpu_maxx;
            miny = p.fa2->layout.gpu_miny;
            maxy = p.fa2->layout.gpu_maxy;

            xrange = maxx-minx;
            yrange = maxy-miny;
        }

        /* RENDER */
        glClear(GL_COLOR_BUFFER_BIT);

        // Update view
        if(p.adapt_view) // benchmark mode, adapt view to positions
        {
            glBindBuffer(GL_UNIFORM_BUFFER, sr.Buffers[Uniforms]);
            glm::mat4 model = glm::scale(glm::mat4(),
                                         glm::vec3(gl_width/xrange, gl_height/yrange, 1.0));
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(model), glm::value_ptr(model));

            glm::mat4 global_view_tmp = glm::translate(glm::mat4(),
                                        glm::vec3(-(minx*gl_width/xrange-(-gl_width/2)), -(miny*gl_height/yrange-(-gl_height/2)), 0.0));

            glm::mat4 local_view = glm::translate(global_view_tmp, glm::vec3(sr.view_offset/vpar_render.z, 0.0, 0.0));
            glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(local_view));
        }

        else // gui mode, no need to adapt view to positions
        {
            glBindBuffer(GL_UNIFORM_BUFFER, sr.Buffers[Uniforms]);
            glm::mat4 local_view = glm::translate(vpar_render.global_view, glm::vec3(sr.view_offset/vpar_render.z, 0.0, 0.0));
            glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(local_view));
        }

        // Update node positions
        if(not p.use_interop)
        {
            glBindBuffer(GL_ARRAY_BUFFER, sr.Buffers[NodePos]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, p.fa2->layout.graph.num_nodes() * sizeof(float2), p.fa2->layout.getNodeCoordinates());
        }

        // Draw edges
        glUseProgram(sr.edge_program);
        glBindVertexArray(sr.VAOs[Edges]);
        glUniform1f(glGetUniformLocation(sr.edge_program, "edge_opacity"),
                    vpar_render.edge_opacity);

        // a.) Indexed, using Node_Pos buffer
        glDrawElements(GL_LINES, 2*p.fa2->layout.graph.num_edges(), GL_UNSIGNED_INT, 0);
        // b.) Or, direct
        // glBufferSubData(GL_ARRAY_BUFFER, 0, 2*p.fa2->layout.graph.num_edges() * sizeof(float2), p.fa2->layout.getEdgeCoordinates());
        // glDrawArrays(GL_LINES, 0, 2*p.fa2->layout.graph.num_edges());

        // Draw nodes
        glUseProgram(sr.node_program);
        glBindVertexArray(sr.VAOs[Nodes]);
        glUniform1f(glGetUniformLocation(sr.node_program, "node_opacity"),
                    vpar_render.node_opacity);
        glDrawArrays(GL_POINTS, 0, p.fa2->layout.graph.num_nodes());


        glFinish();

        int num_swappable = 0;
        if(p.screen_no == 0)
        {
            // view-state 'thread barrier'
            while(num_swappable != screen_count-1)
            {
                num_swappable = 0;
                for(int i = 0; i < screen_count; i++)
                    num_swappable += swap_ready.at(i);
            }
            p.fa2->mx = (((mouse_prevx / display_w * 2 * screen_count) - screen_count) - vpar_render.x_off)/vpar_render.z *  p.fa2->layout.getWidth()/(1.75*screen_count) ;
            p.fa2->my = (((mouse_prevy / display_h * 2) - 1) + vpar_render.y_off)/vpar_render.z * -p.fa2->layout.getHeight()/1.75;

            // update render-view-state from event-view-state
            vpar_render = vpar_event;
        }

        swap_ready.at(p.screen_no) = 1; // only for view-state

        pthread_barrier_wait(&swap_barrier);
        glXSwapBuffers(dpy, sr.glx_window);

        swap_ready.at(p.screen_no) = 0; // only for view-state

        // Record some performance measurements
        benchmark_clock_t::duration d = clock.now() - prev_swap_time;
        inter_frame_durations.at(p.screen_no) = d;

        // only maintain history when benchmarking
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

void *event_loop(void *args)
{
    RPGraph::ForceAtlas2 *fa2 = *((RPGraph::ForceAtlas2 **)args);
    while(program_should_run)
    {
        process_x_events(fa2);
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
            float fps = 1000.0 / to_ms(inter_frame_durations.at(screen_no));
            printf("%.2f", fps);
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

void run_gui(crlparameters_t crl_params, bool print_fps, bool benchmark)
{
    // Spawn a compute_render_loop thread for each screen
    pthread_barrier_init(&swap_barrier, NULL, x_screens.size());
    pthread_barrier_init(&reduction_barrier, NULL, x_screens.size());
    vpar_render = vpar_event;
    std::map<int, crlparameters_t> screen_parameters;
    pthread_t t;
    for(std::pair<int, xcb_screen_t> it : x_screens)
    {
        crlparameters_t p = crl_params;
        p.screen_no = it.first;
        screen_parameters[it.first] = p;
        pthread_create(&t, NULL, compute_render_loop,
                       (void *) &screen_parameters[it.first]);
        threads.push_back(t);
    }

    if(print_fps)
    {
        pthread_create(&t, NULL, stat_reporter_loop, NULL);
        threads.push_back(t);
    }

    if(not benchmark)
    {
        pthread_create(&t, NULL, event_loop, (void*)&crl_params.fa2);
        threads.push_back(t);
    }

    // If program is quit, wait for threads to exit before cleaning up
    for(auto &t : threads) pthread_join(t, NULL);
}

void run_benchmark(RPGraph::ForceAtlas2 *fa2, runmodes runmode,
                   bool cuda_requested, bool use_interop, bool validate,
                   int image_w, int image_h, int iterations,
                   std::vector<int> gpu_ids)
{
    RPGraph::GraphLayout &layout = fa2->layout;
    if(runmode == runmodes::benchmark_layout_render)
    {
        layout.use_interop = use_interop;
        layout.enable_gl_renderer(image_w, image_h);
    }

    bool mapped_pbos = false;
    cudaProfilerStart();
    for (int iteration = 1; iteration <= iterations; ++iteration)
    {
        if(runmode == runmodes::benchmark_layout_render and use_interop and
           not mapped_pbos)
        {
            layout.gl_map_pbos();
            mapped_pbos = true;
            for(int gpu_id : gpu_ids)
            {
                float2 *pbo_ptr = layout.gl_pbo_pointer(gpu_id);
                ((RPGraph::CUDAForceAtlas2 *)fa2)->set_posbuf(gpu_id, pbo_ptr);
            }
        }

        std::chrono::steady_clock::time_point layout_s, layout_e,
                                              draw_node_s, draw_node_e,
                                              draw_edge_s, draw_edge_e,
                                              draw_s, draw_e;
        layout_s = std::chrono::steady_clock::now();
        fa2->doStep();
        layout_e = std::chrono::steady_clock::now();
        printf("%f", to_ms(layout_e-layout_s));

        if(runmode == runmodes::benchmark_layout_render)
        {
            draw_s = std::chrono::steady_clock::now();
            if(cuda_requested)
            {
                // Update `fa2' from data that resides on the GPU.
                if(use_interop)
                {
                    ((RPGraph::CUDAForceAtlas2 *)fa2)->sync_layout_bounds();
                    if(mapped_pbos)
                    {
                        layout.gl_unmap_pbos();
                        mapped_pbos = false;
                    }
                }
                else
                {
                    fa2->sync_layout();
                }
            }
            layout.gl_draw_common();

            draw_node_s = std::chrono::steady_clock::now();
            layout.gl_draw_nodes();
            layout.gl_draw_finish();
            draw_node_e = std::chrono::steady_clock::now();

            draw_edge_s = std::chrono::steady_clock::now();
            layout.gl_draw_edges();
            layout.gl_draw_finish();
            draw_edge_e = std::chrono::steady_clock::now();

            layout.gl_draw_swap();
            draw_e = std::chrono::steady_clock::now();

            printf(",%f", to_ms(draw_node_e-draw_node_s));
            printf(",%f", to_ms(draw_edge_e-draw_edge_s));
            printf(",%f", to_ms(draw_e-draw_s));
        }

        if(validate)
            layout.writeToPNG(image_w, image_h,
                              std::to_string(iteration).append(".png"));
        printf("\n");
    }
    cudaProfilerStop();
}

// due to https://stackoverflow.com/a/236803
std::vector<std::string> split_string(std::string str, char sep)
{
    std::vector<std::string> parts;
    std::stringstream ss(str);
    std::string part;
    while(std::getline(ss, part, sep)) parts.push_back(part);
    return parts;
}

// due to https://stackoverflow.com/a/4654718
bool is_number(const std::string &s)
{
    std::string numbers = "0123456789";
    return (not s.empty()) and (s.find_first_not_of(numbers) == std::string::npos);
}

void print_gui_usage()
{
    printf("Usage: \n");
    printf("       [up/down/left/right] : move around\n");
    printf("       [-/+]                : zoom\n");
    printf("       [h]                  : reset view\n");
    printf("       [v/b]                : cycle colormap\n");
    printf("       [n/m]                : adjust node opacity\n");
    printf("       [k/l]                : adjust edge opacity\n");
    printf("       [p]                  : start/stop layout algorithm\n");
    printf("       [r/R]                : adjust repulsive force\n");
    printf("       [g/G]                : adjust gravitational force\n");
    printf("       [alt+g]              : toggle weak/strong gravity\n");
    printf("       [z]                  : toggle local repulsion\n");
    printf("       [a/A]                : adjust local repulsive force\n");
    printf("       [x]                  : toggle local heat\n");
    printf("       [s/S]                : adjust local temperature\n");
    printf("       [q]                  : exit\n");
}

int main(int argc, const char **argv)
{
    // For reproducibility.
    srandom(1234);

    // Parse commandline arguments
    if (argc < 8)
    {
        fprintf(stderr, "Usage: graph_viewer gui        gpu|cpu sg|wg scale gravity exact|approximate edgelist_path [interop|no_interop] [print_fps]\n");
        fprintf(stderr, "                    benchmark  gpu|cpu sg|wg scale gravity exact|approximate edgelist_path iterations [layout num_gpus|bus_ids] [layout_render image_w image_h validate|no_validate interop|no_interop num_gpus|bus_ids] [gui interop|no_interop] \n");
        exit(EXIT_FAILURE);
    }

    runmodes runmode = std::string(argv[1]) == "gui" ?
                       runmodes::gui : runmodes::benchmark_layout;
    const bool cuda_requested = std::string(argv[2]) == "gpu";
    const bool strong_gravity = std::string(argv[3]) == "sg";
    const float scale = std::stof(argv[4]);
    const float gravity = std::stof(argv[5]);
    const bool approximate = std::string(argv[6]) == "approximate";
    const char *edgelist_path = argv[7];

    bool use_interop = true;
    bool print_fps = false;

    int benchmark_render_w = 1920;
    int benchmark_render_h = 1080;
    bool benchmark_validate = false;
    int benchmark_gpu_count = 1;
    int benchmark_iterations = 0;
    std::vector<std::string> benchmark_gpu_busses;
    std::vector<int> compute_gpu_ids;

    if(runmode == runmodes::gui)
    {
        for(int arg_no = 8; arg_no < argc; ++arg_no)
        {
            if (std::string(argv[arg_no]) == "interop")
            {
                use_interop = true;
            }

            else if (std::string(argv[arg_no]) == "no_interop")
            {
                use_interop = false;
            }

            else if(std::string(argv[arg_no]) == "print_fps")
            {
                print_fps = true;;
            }

            else
            {
                printf("Error: %s is no CLI argument\n", argv[arg_no]);
                exit(EXIT_FAILURE);
            }
        }
    }

    else // benchmark
    {
        benchmark_iterations = std::stof(argv[8]);
        for(int arg_no = 9; arg_no < argc; ++arg_no)
        {
            if (std::string(argv[arg_no]) == "layout")
            {
                runmode = runmodes::benchmark_layout;
                std::string gpu_arg = std::string(argv[arg_no+1]);
                if(is_number(gpu_arg))
                    benchmark_gpu_count = std::stoi(gpu_arg);
                else
                    benchmark_gpu_busses = split_string(gpu_arg, ',');
                arg_no += 1;
            }

            else if(std::string(argv[arg_no]) == "layout_render")
            {
                runmode = runmodes::benchmark_layout_render;
                benchmark_render_w = std::stoi(argv[arg_no+1]);
                benchmark_render_h = std::stoi(argv[arg_no+2]);
                benchmark_validate = std::string(argv[arg_no+3]) == "validate";
                use_interop = std::string(argv[arg_no+4]) == "interop";
                std::string gpu_arg = std::string(argv[arg_no+5]);
                if(is_number(gpu_arg))
                    benchmark_gpu_count = std::stoi(gpu_arg);
                else
                    benchmark_gpu_busses = split_string(gpu_arg, ',');
                arg_no += 5;
            }

            else if (std::string(argv[arg_no]) == "gui")
            {
                runmode = runmodes::benchmark_gui;
                use_interop = std::string(argv[arg_no+1]) == "interop";
                arg_no += 1;
            }

            else
            {
                printf("Error: %s is no CLI argument\n", argv[arg_no]);
                exit(EXIT_FAILURE);
            }
        }

        // claim GPUs asap. (before loading data)
        if(benchmark_gpu_busses.size() > 0)
            compute_gpu_ids = claim_cuda_devices_on_bus(benchmark_gpu_busses);
        else
            compute_gpu_ids = claim_cuda_devices(benchmark_gpu_count);
    }

    // 'validate' cli arguments
    if(cuda_requested and not approximate)
    {
        fprintf(stderr, "Error: the CUDA implementation (currently) requires"
                        "Barnes-Hut approximation.\n");
        exit(EXIT_FAILURE);
    }

    if(use_interop and not cuda_requested)
    {
        fprintf(stderr, "Error: cannot use interop without CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (!is_file_exists(edgelist_path))
    {
        fprintf(stderr, "Error: no edgelist at %s\n", edgelist_path);
        exit(EXIT_FAILURE);
    }

#ifndef __NVCC__
    if(cuda_requested)
    {
        fprintf(stderr, "Error: CUDA was requested, but not compiled for.\n");
        exit(EXIT_FAILURE);
    }
#endif

    // load graph
    if(runmode == runmodes::gui) printf("Loading edgelist at '%s'...", edgelist_path);
    if(runmode == runmodes::gui) fflush(stdout);
    RPGraph::UGraph graph = RPGraph::UGraph(edgelist_path);
    if(runmode == runmodes::gui) printf("done.\n");
    if(runmode == runmodes::gui) printf("    fetched %d nodes and %d edges.\n", graph.num_nodes(), graph.num_edges());

    // create the GraphLayout object
    RPGraph::GraphLayout layout(graph);
    layout.randomizePositions();

    // initialize node colors (based on degree)
    std::shared_ptr<colormap::Colormap const> cmap = cmap_list[vpar_event.cmap_index];
    RPGraph::nid_t max_degree = graph.max_degree();
    for(int n = 0; n < graph.num_nodes(); ++n)
    {
        colormap::Color c = cmap->getColor(graph.degree(n)/(float)max_degree);
        node_colors.push_back(glm::vec4(c.r, c.g, c.b, c.a));
    }

    vpar_event.node_opacity = fminf(0.95, fmaxf(10000.0 / graph.num_nodes(), 0.005));
    vpar_event.edge_opacity = fminf(0.95, fmaxf(25000.0 / graph.num_edges(), 0.005));

    // initialize X/OpenGL
    if(runmode == runmodes::gui or runmode == runmodes::benchmark_gui)
    {
        init_xlib();
        init_xcb();
        load_glx();
        init_screens();
        create_xwindows();
        create_glxwindows();
        init_gl(graph, layout);
        map_windows();
    }

    // determine GPUs to use for compute
    if(cuda_requested)
    {
        if(runmode == runmodes::gui or runmode == runmodes::benchmark_gui)
            compute_gpu_ids = get_render_gpu_ids();
        else if(benchmark_gpu_busses.size() > 0)
            compute_gpu_ids = claim_cuda_devices_on_bus(benchmark_gpu_busses);
        else
            compute_gpu_ids = claim_cuda_devices(benchmark_gpu_count);
        for(int gpu_id : compute_gpu_ids)
        {
            // store properties
            cudaCatchError(cudaGetDeviceProperties(&gpu_properties[gpu_id], gpu_id));

            // enable p2p access
            if(runmode == runmodes::gui or runmode == runmodes::benchmark_gui)
            {
                for(int peer_id : compute_gpu_ids)
                {
                    cudaSetDevice(gpu_id);
                    if(gpu_id == peer_id) continue;
                    int can_peer;
                    cudaDeviceCanAccessPeer(&can_peer, gpu_id, peer_id);
                    if(can_peer == 1)
                    {
                        unsigned int flags = 0;
                        cudaCatchError(cudaDeviceEnablePeerAccess(peer_id, flags));
                    }
                }
            }
        }
    }

    if(use_interop) init_interop();

    // create the ForceAtlas2 object
    RPGraph::ForceAtlas2 *fa2;
    #ifdef __NVCC__
    if(cuda_requested)
    {
        fa2 = new RPGraph::CUDAForceAtlas2(layout, approximate,
                                           strong_gravity, gravity, scale,
                                           compute_gpu_ids);
        layout.render_gpus = compute_gpu_ids;
        ((RPGraph::CUDAForceAtlas2 *)fa2)->sync_layout_bounds();
    }
    else
    #endif
        fa2 = new RPGraph::CPUForceAtlas2(layout, approximate,
                                          strong_gravity, gravity, scale);

    crlparameters_t crl_params;
    crl_params.fa2 = fa2;
    crl_params.use_interop = use_interop;

    // start network visualization
    if(runmode == runmodes::gui)
    {
        layout_paused = true;
        crl_params.adapt_view = false;
        crl_params.record_fps = false;
        crl_params.max_iterations = 0; // ie. no benchmark
        run_gui(crl_params, print_fps, false);
    }

    else if(runmode == runmodes::benchmark_gui)
    {
        layout_paused = false;
        crl_params.adapt_view = true;
        crl_params.record_fps = true;
        crl_params.max_iterations = benchmark_iterations;
        run_gui(crl_params, print_fps, true);

        // print inter_frame_durations, skip dummy (i.e. i = 0)
        for(int i = 1; i <= benchmark_iterations; ++i)
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

    else if(runmode == runmodes::benchmark_layout or
            runmode == runmodes::benchmark_layout_render)
    {
        run_benchmark(fa2, runmode, cuda_requested, use_interop,
                      benchmark_validate, benchmark_render_w, benchmark_render_h,
                      benchmark_iterations, compute_gpu_ids);
    }

    exit(EXIT_SUCCESS);
}
