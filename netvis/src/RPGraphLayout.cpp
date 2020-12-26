/*
 ==============================================================================

 RPGraphLayout.cpp
 Copyright © 2016, 2017, 2018  G. Brinkmann

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

 ==============================================================================
*/


#include "RPGraphLayout.hpp"
#include "RPCommon.hpp"
#include <png.h>
#include "../lib/pngwriter/src/pngwriter.h"
#include "../lib/glm/glm/gtc/matrix_transform.hpp"
#include "../lib/glm/glm/gtc/type_ptr.hpp"
#include "../lib/colormap/include/colormap/colormap.h"
#include "gl_util.hpp"

#ifdef __NVCC__
#include <cuda_gl_interop.h>
#endif

#include <fstream>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <cstdio>

#ifndef __APPLE__
#include "../lib/pngwriter/src/pngwriter.h"
#endif

namespace RPGraph
{
    GraphLayout::GraphLayout(UGraph &graph, float width, float height)
        : graph(graph), width(width), height(height)
    {
        coordinates = (float2 *) malloc(graph.num_nodes() * sizeof(float2));
        edge_coordinates = (float2 *) malloc(2 * graph.num_edges() * sizeof(float2));
        global_view = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 0.0f));

        // Here we need to do some guessing as to what the optimal
        // opacity of nodes and edges might be, given network size.
        set_node_opacity(fminf(0.95, fmaxf(10000.0 / graph.num_nodes(), 0.005)));
        set_edge_opacity(fminf(0.95, fmaxf(25000.0 / graph.num_edges(), 0.005)));
    }

    GraphLayout::~GraphLayout()
    {
        free(coordinates);
    }
    void GraphLayout::init_egl_screens(int image_w, int image_h)
    {
        load_egl();

        // Query number of EGL devices
        EGLint max_devices;
        eglQueryDevicesEXT(0, NULL, &max_devices);

        // Query the EGL devices
        EGLDeviceEXT *egl_devices = new EGLDeviceEXT[max_devices];
        EGLint stored_devices;
        eglQueryDevicesEXT(max_devices, egl_devices, &stored_devices);

        // Since there are no screens to iterate over, use GPUs
        for(auto cuda_gpu_id : render_gpus)
        {
            screen_resources_t sr;

            sr.gpu_id = cuda_gpu_id;

            // Find EGL device matching cuda_gpu_id
            for(int egl_dev_no = 0; egl_dev_no < max_devices; egl_dev_no++)
            {
                // Find matching
                EGLAttrib egl_dev_cuda_id;
                eglQueryDeviceAttribEXT(egl_devices[egl_dev_no],
                                        EGL_CUDA_DEVICE_NV,
                                        &egl_dev_cuda_id);
                if((int)egl_dev_cuda_id == cuda_gpu_id)
                {
                    sr.egl_device = egl_devices[egl_dev_no];
                }
            }

            // Init egl_display on this device
            sr.egl_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                                      sr.egl_device, NULL);

            eglInitialize(sr.egl_display, 0, 0);

            // Bind OpenGL API
            eglBindAPI(EGL_OPENGL_API);

            // Select Config.
            const EGLint config_attributes[] =
            {
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_CONFORMANT, EGL_OPENGL_BIT,
                EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
                EGL_LUMINANCE_SIZE, 0,
                EGL_RED_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_BLUE_SIZE, 8,
                EGL_ALPHA_SIZE, 8,
                EGL_DEPTH_SIZE, 8,
                EGL_LEVEL, 0,
                EGL_BUFFER_SIZE, 24,
                EGL_NONE
            };

            int n_configs;
            if (!eglChooseConfig(sr.egl_display, config_attributes, &sr.egl_config, 1,
                &n_configs))
            {
                std::cerr << "EGL error choose config: " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            if (n_configs < 1)
            {
                std::cerr << "Error: eglChooseConfig(): config not found.\n";
                exit(EXIT_FAILURE);
            }


            int screen_count = (int)render_gpus.size();
            sr.screen_w = image_w / screen_count;
            if(cuda_gpu_id == render_gpus.back());
                sr.screen_w += (image_w - sr.screen_w*screen_count);
            sr.screen_h = image_h;

            const EGLint buffer_attributes[] =
            {
                EGL_WIDTH,  sr.screen_w,
                EGL_HEIGHT, sr.screen_h,
                EGL_NONE
            };

            sr.egl_surface = eglCreatePbufferSurface(sr.egl_display, sr.egl_config,
                                                     buffer_attributes);
            if (sr.egl_surface == EGL_NO_SURFACE)
            {
                std::cerr << "Error: eglCreatePbufferSurface(): "
                          << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            gpu_screen_idx[sr.gpu_id] = screen_resources.size();
            screen_resources.push_back(sr);
        }
        initialized_screens = true;
    }

    void GraphLayout::load_egl()
    {
        if(!gladLoadEGL())
        {
            printf("Error: Couldn't load EGL through glad\n");
            exit(EXIT_FAILURE);
        }
        initialized_egl = true;
    }

    void GraphLayout::load_gl()
    {
        GLADloadproc load_proc;
        if(use_egl) load_proc = (GLADloadproc) eglGetProcAddress;

        if(not loaded_gl and !gladLoadGLLoader(load_proc))
        {
            printf("Error: Couldn't load GL through glad\n");
            exit(EXIT_FAILURE);
        }
    }

    void GraphLayout::init_gl(int image_w, int image_h)
    {
        int screen_count = screen_resources.size();
        gl_width  = 2 * screen_count;
        gl_height = 2;
        float view_offset = screen_count - 1;

        // Init colors for nodes.
        std::vector<glm::vec4> node_colors;
        nid_t max_degree = graph.max_degree();

        colormap::IDL::CBReds cmap;
        for (size_t n = 0; n < graph.num_nodes(); n++)
        {
            colormap::Color c = cmap.getColor((float)graph.degree(n) / max_degree);
            node_colors.push_back(glm::vec4(c.r, c.g, c.b, c.a));
        }


        for(auto &sr : screen_resources)
        {
            // Create OpenGL context
            const EGLint context_attributes[] =
            {
                EGL_CONTEXT_MAJOR_VERSION, 4,
                EGL_CONTEXT_MINOR_VERSION, 1,
                EGL_NONE
            };

            sr.egl_context = eglCreateContext(sr.egl_display, sr.egl_config, NULL,
                                              context_attributes);
            if (sr.egl_context == EGL_NO_CONTEXT)
            {
                std::cerr << "Error: eglCreateContext(): "
                          << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            // Make OpenGL context current
            if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                sr.egl_surface, sr.egl_context))
            {
                std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            load_gl();

            set_gl_base_settings();

            // Generate Buffers
            glGenBuffers(1, &sr.gl_node_buffer);
            if(use_interop) gl_node_buffer_type = GL_DYNAMIC_COPY;
            else gl_node_buffer_type = GL_DYNAMIC_DRAW;
            glGenBuffers(1, &sr.gl_el_buffer);
            glGenBuffers(1, &sr.gl_nodecolor_buffer);
            glGenBuffers(1, &sr.gl_uniform_buffer);

            // Generate VAOs
            glGenVertexArrays(1, &sr.gl_node_vao);
            glGenVertexArrays(1, &sr.gl_edge_vao);

            // Programs
            sr.node_program = gl_node_program();
            sr.edge_program = gl_edge_program();

            // Uniforms
            glBindBuffer(GL_UNIFORM_BUFFER, sr.gl_uniform_buffer);

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
                              sr.gl_uniform_buffer, 0, 2 * sizeof(glm::mat4) + sizeof(glm::vec2));

            glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4) + sizeof(glm::vec2), NULL, GL_STREAM_DRAW);
            glm::mat4 model = glm::scale(glm::mat4(), glm::vec3(gl_width/width, gl_height/height, 1.0));
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(model), glm::value_ptr(model));
            glm::mat4 local_view = glm::translate(global_view, glm::vec3(sr.view_offset, 0.0, 0.0));
            glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(local_view), glm::value_ptr(local_view));
            glm::vec2 vp = glm::vec2(sr.screen_w, sr.screen_h);
            glBufferSubData(GL_UNIFORM_BUFFER, sizeof(model) + sizeof(local_view), sizeof(vp), glm::value_ptr(vp));

            // Setup bindings
            glUseProgram(sr.node_program);
            glBindVertexArray(sr.gl_node_vao);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
            glBindBuffer(GL_ARRAY_BUFFER, sr.gl_node_buffer);
            glBufferData(GL_ARRAY_BUFFER, graph.num_nodes() * sizeof(float2), &coordinates[0], gl_node_buffer_type);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, sr.gl_nodecolor_buffer);
            glBufferData(GL_ARRAY_BUFFER, graph.num_nodes() * sizeof(glm::vec4), &node_colors[0].x, GL_STATIC_DRAW);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(1);
            glUniform1f(glGetUniformLocation(sr.node_program, "node_opacity"),
                        node_opacity);

            // Element buffer
            glUseProgram(sr.edge_program);
            glBindVertexArray(sr.gl_edge_vao);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
            glBindBuffer(GL_ARRAY_BUFFER, sr.gl_nodecolor_buffer);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, sr.gl_node_buffer);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sr.gl_el_buffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, graph.num_edges() * sizeof(uint2), (uint *)&graph.edges[0], GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(0);
            // following two lines are for non-indexed edge drawing
            // glBindBuffer(GL_ARRAY_BUFFER, Buffers[EdgePos]);
            // glBufferData(GL_ARRAY_BUFFER, 2*graph.num_edges() * sizeof(float2), fa2->layout.getEdgeCoordinates(), GL_STREAM_DRAW);
            glUniform1f(glGetUniformLocation(sr.edge_program, "edge_opacity"),
                        edge_opacity);

            sr.view_offset = view_offset;
            view_offset -= gl_width/screen_count;
        }

        initialized_gl = true;
    }

    void GraphLayout::init_interop()
    {
        for(auto &sr : screen_resources)
        {
            // Make GL context current
            if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                sr.egl_surface, sr.egl_context))
            {
                std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            // Make CUDA device current
            cudaCatchError(cudaSetDevice(sr.gpu_id));

            // Map stuff
            cudaCatchError(cudaGraphicsGLRegisterBuffer(&sr.cuda_pbo_resource,
                            sr.gl_node_buffer, cudaGraphicsRegisterFlagsNone));
        }
    }

    void GraphLayout::enable_gl_renderer(int image_w, int image_h)
    {
        if(use_egl)
        {
            load_egl();
            init_egl_screens(image_w, image_h);
        }

        init_gl(image_w, image_h);
        if(use_interop) init_interop();

        use_gl_renderer = true;
    }

    void GraphLayout::randomizePositions()
    {
        for (nid_t i = 0; i <  graph.num_nodes(); ++i)
        {
            setX(i, get_random(-width/2.0, width/2.0));
            setY(i, get_random(-height/2.0, height/2.0));
        }
    }

    float GraphLayout::getX(nid_t node_id)
    {
        return coordinates[node_id].x;
    }

    float GraphLayout::getY(nid_t node_id)
    {
        return coordinates[node_id].y;
    }

    float GraphLayout::minX()
    {
        float minX = std::numeric_limits<float>::max();
        for (nid_t n = 0; n < graph.num_nodes(); ++n)
            if (getX(n) < minX) minX = getX(n);
        return minX;
    }

    float GraphLayout::maxX()
    {
        float maxX = std::numeric_limits<float>::min();
        for (nid_t n = 0; n < graph.num_nodes(); ++n)
            if (getX(n) > maxX) maxX = getX(n);
        return maxX;
    }

    float GraphLayout::minY()
    {
        float minY = std::numeric_limits<float>::max();
        for (nid_t n = 0; n < graph.num_nodes(); ++n)
            if (getY(n) < minY) minY = getY(n);
        return minY;
    }

    float GraphLayout::maxY()
    {
        float maxY = std::numeric_limits<float>::min();
        for (nid_t n = 0; n < graph.num_nodes(); ++n)
            if (getY(n) > maxY) maxY = getY(n);
        return maxY;
    }

    float GraphLayout::getXRange()
    {
        return maxX()- minX();
    }

    float GraphLayout::getYRange()
    {
        return maxY() - minY();
    }

    float GraphLayout::getWidth()
    {
        return this->width;
    }

    float GraphLayout::getHeight()
    {
        return this->height;
    }

    float GraphLayout::getSpan()
    {
        return ceil(fmaxf(getXRange(), getYRange()));
    }

    float GraphLayout::getDistance(nid_t n1, nid_t n2)
    {
        const float dx = getX(n1)-getX(n2);
        const float dy = getY(n1)-getY(n2);
        return std::sqrt(dx*dx + dy*dy);
    }

    Real2DVector GraphLayout::getDistanceVector(nid_t n1, nid_t n2)
    {
        return Real2DVector(getX(n2) - getX(n1), getY(n2) - getY(n1));
    }

    Real2DVector GraphLayout::getNormalizedDistanceVector(nid_t n1, nid_t n2)
    {
        const float x1 = getX(n1);
        const float x2 = getX(n2);
        const float y1 = getY(n1);
        const float y2 = getY(n2);
        const float dx = x2 - x1;
        const float dy = y2 - y1;
        const float len = std::sqrt(dx*dx + dy*dy);

        return Real2DVector(dx / len, dy / len);
    }

    Coordinate GraphLayout::getCoordinate(nid_t node_id)
    {
        return Coordinate(coordinates[node_id].x, coordinates[node_id].y);
    }

    Coordinate GraphLayout::getCenter()
    {
        float x = minX() + getXRange()/2.0;
        float y = minY() + getYRange()/2.0;
        return Coordinate(x, y);
    }

    void GraphLayout::setX(nid_t node_id, float x_value)
    {
        coordinates[node_id].x = x_value;
    }

    void GraphLayout::setY(nid_t node_id, float y_value)
    {
        coordinates[node_id].y = y_value;
    }

    void GraphLayout::moveNode(nid_t n, RPGraph::Real2DVector v)
    {
        setX(n, getX(n) + v.x);
        setY(n, getY(n) + v.y);
    }

    void GraphLayout::setCoordinates(nid_t node_id, Coordinate c)
    {
        setX(node_id, c.x);
        setY(node_id, c.y);
    }

    void GraphLayout::gl_draw_common()
    {
        // This is ugly and temporary.
        // Precondition: a call to sync_layout_bounds() in the cuda fa2 class
        // has been made to ensure all gpu_* variables are updated.
        float minx, maxx, miny, maxy;
        if(use_interop)
        {
            minx = gpu_minx;
            maxx = gpu_maxx;
            miny = gpu_miny;
            maxy = gpu_maxy;
        }
        else
        {
            minx = minX();
            maxx = maxX();
            miny = minY();
            maxy = maxY();
        }

        float xrange = maxx - minx;
        float yrange = maxy - miny;

        int screen_count = screen_resources.size();
        for(auto &sr : screen_resources)
        {
            if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                sr.egl_surface, sr.egl_context))
            {
                std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            glClear(GL_COLOR_BUFFER_BIT);

            // We set the model and view matrices.. this is important to ensure the
            // whole graph gets rendered.
            glBindBuffer(GL_UNIFORM_BUFFER, sr.gl_uniform_buffer);
            // model matrix scales layout coorinates to GL range
            glm::mat4 model = glm::scale(glm::mat4(),
                                         glm::vec3(gl_width/xrange, gl_height/yrange, 1.0));
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(model), glm::value_ptr(model));
            // .. and centers around (0, 0)
            global_view = glm::translate(glm::mat4(),
                                         glm::vec3(-(minx*gl_width/xrange-(-gl_width/2)), -(miny*gl_height/yrange-(-gl_height/2)), 0.0));
            glm::mat4 local_view = glm::translate(global_view,
                                                  glm::vec3(sr.view_offset, 0.0, 0.0));
            glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4),
                            glm::value_ptr(local_view));

            if(not use_interop)
            {
                glBindBuffer(GL_ARRAY_BUFFER, sr.gl_node_buffer);
                glBufferSubData(GL_ARRAY_BUFFER, 0, graph.num_nodes() * sizeof(float2),
                             &coordinates[0]);
            }
        }
    }

    void GraphLayout::gl_draw_nodes()
    {
        for(auto &sr : screen_resources)
        {
            if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                sr.egl_surface, sr.egl_context))
            {
                std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            // Start drawing, first nodes...
            glUseProgram(sr.node_program);
            glBindVertexArray(sr.gl_node_vao);
            glDrawArrays(GL_POINTS, 0, graph.num_nodes());
        }
    }


    void GraphLayout::gl_draw_edges()
    {
        for(auto &sr : screen_resources)
        {
            if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                sr.egl_surface, sr.egl_context))
            {
                std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            // .. then draw edges.
            glUseProgram(sr.edge_program);
            glBindVertexArray(sr.gl_edge_vao);

            // a.) Indexed, using Node_Pos buffer
            glDrawElements(GL_LINES, 2*graph.num_edges(), GL_UNSIGNED_INT, 0);
            // b.) Or, direct
            // glBufferSubData(GL_ARRAY_BUFFER, 0, 2*fa2->layout.graph.num_edges() * sizeof(float2), fa2->layout.getEdgeCoordinates());
            // glDrawArrays(GL_LINES, 0, 2*fa2->layout.graph.num_edges());
        }
    }

    void GraphLayout::gl_draw_finish()
    {
        for(auto &sr : screen_resources)
        {
            if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                sr.egl_surface, sr.egl_context))
            {
                std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            glFinish();
        }
    }

    void GraphLayout::gl_draw_swap()
    {
        for(auto &sr : screen_resources)
        {
            if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                sr.egl_surface, sr.egl_context))
            {
                std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                exit(EXIT_FAILURE);
            }

            eglSwapBuffers(sr.egl_display, sr.egl_surface);
        }
    }

    float2 *GraphLayout::getNodeCoordinates()
    {
        return coordinates;
    }

    float2 *GraphLayout::getEdgeCoordinates()
    {
        uint eid = 0;
        for (nid_t n1 = 0; n1 < graph.num_nodes(); ++n1)
        {
            for (nid_t n2 : graph.neighbors_with_geq_id(n1)) {
                edge_coordinates[eid++] = {getX(n1), getY(n1)};
                edge_coordinates[eid++] = {getX(n2), getY(n2)};
            }
        }
        return edge_coordinates;
    }

    void GraphLayout::gl_draw_graph()
    {
        gl_draw_common();
        gl_draw_edges();
        gl_draw_nodes();
        gl_draw_finish();
        gl_draw_swap();

    }

    void GraphLayout::set_node_opacity(float opacity)
    {
        node_opacity = opacity;
        if(initialized_gl)
            for(auto &sr : screen_resources)
            {
                if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                    sr.egl_surface, sr.egl_context))
                {
                    std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                    exit(EXIT_FAILURE);
                }
                glUniform1f(glGetUniformLocation(sr.node_program, "node_opacity"),
                            node_opacity);
            }
    }

    void GraphLayout::set_edge_opacity(float opacity)
    {
        edge_opacity = opacity;
        if(initialized_gl)
            for(auto &sr : screen_resources)
            {
                if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                    sr.egl_surface, sr.egl_context))
                {
                glUniform1f(glGetUniformLocation(sr.edge_program, "edge_opacity"),
                            edge_opacity);
                }
            }
    }

    void GraphLayout::writeToPNG(const int image_w, const int image_h,
                                 std::string path)
    {
        if(use_gl_renderer)
        {
            gl_draw_graph();

            // Create file
            FILE *fp = fopen(path.c_str(), "wb");
            if(!fp)
            {
                printf("Error: couldn't open file at %s\n", path.c_str());
                exit(EXIT_FAILURE);
            }

            // Initialize png writer
            png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                                          NULL, NULL, NULL);
            if(!png_ptr)
            {
                printf("Error: couldn't init png_structp\n");
                exit(EXIT_FAILURE);
            }

            png_infop info_ptr = png_create_info_struct(png_ptr);
            if(!png_ptr)
            {
                printf("Error: couldn't init png_infop\n");
                exit(EXIT_FAILURE);
            }

            if (setjmp(png_jmpbuf(png_ptr)))
            {
                printf("Error: during png initio\n");
                exit(EXIT_FAILURE);
            }

            png_init_io(png_ptr, fp);

            // Write header
            if (setjmp(png_jmpbuf(png_ptr)))
            {
                printf("Error: during png write header\n");
                exit(EXIT_FAILURE);
            }
            int bit_depth = 8;
            png_set_IHDR(png_ptr, info_ptr, image_w, image_h,
                         bit_depth, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

            png_write_info(png_ptr, info_ptr);

            png_set_filler(png_ptr, 0, PNG_FILLER_AFTER);


            /* write bytes */
            if (setjmp(png_jmpbuf(png_ptr)))
            {
                printf("Error: during png write bytes\n");
                exit(EXIT_FAILURE);
            }

            // Obtain GL results
            char *rgba_data = new char[image_w * image_h * 4];
            int x_off = 0;
            for(auto &sr : screen_resources)
            {
                if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                                    sr.egl_surface, sr.egl_context))
                {
                    std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
                    exit(EXIT_FAILURE);
                }

                char *rgba_data_screen = new char[sr.screen_w * sr.screen_h * 4];
                glReadPixels(0, 0, sr.screen_w, sr.screen_h, GL_RGBA,
                             GL_UNSIGNED_BYTE, rgba_data_screen);

                // Reduce to rgba_data
                int row_bytes = 4 * image_w;
                int row_bytes_screen = 4 * sr.screen_w;
                for(int row = 0; row < sr.screen_h; ++row)
                {
                    memcpy(rgba_data + x_off + row * row_bytes,
                           rgba_data_screen + row_bytes_screen*row,
                           row_bytes_screen);
                }

                x_off += row_bytes_screen;
            }


            png_bytepp row_pointers = new png_bytep[image_h];
            for(int y = 0; y < image_h; y++)
                row_pointers[y] = (png_bytep) &rgba_data[y*4*image_w];
            png_write_image(png_ptr, row_pointers);

            /* end write */
            if (setjmp(png_jmpbuf(png_ptr)))
            {
                printf("Error: during png write\n");
                exit(EXIT_FAILURE);
            }

            png_write_end(png_ptr, NULL);

            /* cleanup heap allocation */
            delete row_pointers;
            delete rgba_data;
            fclose(fp);
        }

        else
        {
            const float xRange = getXRange();
            const float yRange = getYRange();
            const RPGraph::Coordinate center = getCenter();
            const float xCenter = center.x;
            const float yCenter = center.y;
            const float minX = xCenter - xRange/2.0;
            const float minY = yCenter - yRange/2.0;
            const float xScale = image_w/xRange;
            const float yScale = image_h/yRange;


            // Write to file.
            pngwriter layout_png(image_w, image_h, 0, path.c_str());
            layout_png.invert(); // set bg. to white

            for (nid_t n1 = 0; n1 < graph.num_nodes(); ++n1)
            {
                // Plot node,
                layout_png.filledcircle_blend((getX(n1) - minX)*xScale,
                                              (getY(n1) - minY)*yScale,
                                              3, node_opacity, 0, 0, 0);
                for (nid_t n2 : graph.neighbors_with_geq_id(n1))
                {
                    // ... and edge.
                    layout_png.line_blend((getX(n1) - minX)*xScale, (getY(n1) - minY)*yScale,
                                          (getX(n2) - minX)*xScale, (getY(n2) - minY)*yScale,
                                          edge_opacity, 0, 0, 0);
                }
            }
            layout_png.write_png();
        }
    }

    void GraphLayout::writeToCSV(std::string path)
    {
        if (is_file_exists(path.c_str()))
        {
            printf("Error: File exists at %s\n", path.c_str());
            exit(EXIT_FAILURE);
        }

        std::ofstream out_file(path);

        for (nid_t n = 0; n < graph.num_nodes(); ++n)
        {
            nid_t id = graph.node_map_r[n]; // id as found in edgelist
            out_file << id << "," << getX(n) << "," << getY(n) << "\n";
        }

        out_file.close();
    }

    void GraphLayout::writeToBin(std::string path)
    {
        if (is_file_exists(path.c_str()))
        {
            printf("Error: File exists at %s\n", path.c_str());
            exit(EXIT_FAILURE);
        }

        std::ofstream out_file(path, std::ofstream::binary);

        for (nid_t n = 0; n < graph.num_nodes(); ++n)
        {
            nid_t id = graph.node_map_r[n]; // id as found in edgelist
            float x = getX(n);
            float y = getY(n);

            out_file.write(reinterpret_cast<const char*>(&id), sizeof(id));
            out_file.write(reinterpret_cast<const char*>(&x), sizeof(x));
            out_file.write(reinterpret_cast<const char*>(&y), sizeof(y));
        }

        out_file.close();
    }
    #ifdef __NVCC__
    void GraphLayout::gl_map_pbo(screen_resources_t &sr)
    {
        if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                            sr.egl_surface, sr.egl_context))
        {
            std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
            exit(EXIT_FAILURE);
        }

        // Make CUDA device current
        cudaCatchError(cudaSetDevice(sr.gpu_id));


        cudaStream_t default_stream = 0;
        cudaCatchError(cudaGraphicsMapResources(1, &sr.cuda_pbo_resource,
                                                default_stream));
    }

    void GraphLayout::gl_unmap_pbo(screen_resources_t &sr)
    {
        if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                            sr.egl_surface, sr.egl_context))
        {
            std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
            exit(EXIT_FAILURE);
        }

        // Make CUDA device current
        cudaCatchError(cudaSetDevice(sr.gpu_id));

        cudaStream_t default_stream = 0;
        cudaCatchError(cudaGraphicsUnmapResources(1, &sr.cuda_pbo_resource,
                                                  default_stream));
    }

    void GraphLayout::gl_map_pbos()
    {
        for(auto &sr : screen_resources) gl_map_pbo(sr);
    }

    void GraphLayout::gl_unmap_pbos()
    {
        for(auto &sr : screen_resources) gl_unmap_pbo(sr);
    }

    float2 *GraphLayout::gl_pbo_pointer(int gpu_id)
    {
        screen_resources_t &sr = screen_resources[gpu_screen_idx[gpu_id]];
        if (!eglMakeCurrent(sr.egl_display, sr.egl_surface,
                            sr.egl_surface, sr.egl_context))
        {
            std::cerr << "Error: eglMakeCurrent(): " << eglGetError() << "\n";
            exit(EXIT_FAILURE);
        }

        // Make CUDA device current
        cudaCatchError(cudaSetDevice(sr.gpu_id));

        size_t mapped_size;
        float2 *pbo_cuda_ptr;
        cudaCatchError(cudaGraphicsResourceGetMappedPointer((void **)&pbo_cuda_ptr,
                                &mapped_size, sr.cuda_pbo_resource));
        return pbo_cuda_ptr;
    }

    #endif
}
