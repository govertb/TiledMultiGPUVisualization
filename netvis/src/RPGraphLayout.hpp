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

#ifndef RPGraphLayout_hpp
#define RPGraphLayout_hpp

#include "RPGraph.hpp"
#include "RPCommon.hpp"
#include <string>
#include <map>

#include "../lib/glm/glm/glm.hpp"

#include "../lib/glad/glad.h"
#include "../lib/glad/glad_egl.h"

namespace RPGraph
{
    class GraphLayout
    {
    private:
//        Coordinate *coordinates;
        float2 *coordinates; // only access through setX,setY
        float2 *edge_coordinates;

        bool use_egl = true;
        bool use_gl_renderer = false;
        bool initialized_screens = false;
        bool initialized_egl = false;
        bool initialized_gl = false;
        bool loaded_gl = false;

        // Virtual screen
        typedef struct sr
        {
            EGLDisplay egl_display;
            EGLSurface egl_surface;
            EGLContext egl_context;
            EGLDeviceEXT egl_device;
            EGLConfig egl_config;
            int gpu_id;
            GLuint gl_program, gl_node_buffer, gl_el_buffer, gl_nodecolor_buffer, gl_uniform_buffer;
            GLuint gl_node_vao, gl_edge_vao;
            GLuint node_program, edge_program;
            #ifdef __NVCC__
            cudaGraphicsResource_t cuda_pbo_resource;
            #endif
            float view_offset;
            float2 *pbo_cuda_ptr;
            int screen_w, screen_h;
        } screen_resources_t;

        std::vector<screen_resources_t> screen_resources; // indexed by GPU
        std::map<int, int> gpu_screen_idx; // gpuid->idx in screen_resources

        GLenum gl_node_buffer_type;
        float gl_width;
        float gl_height;
        void load_egl();
        void load_gl();
        void init_egl_screens(int image_w, int image_h);
        void init_gl(int image_w, int image_h);
        void init_interop();
        void release_gl_context();

        float node_opacity, edge_opacity;
        glm::mat4 global_view;

    protected:
        float width, height;
        float minX(), minY(), maxX(), maxY();

    public:
        GraphLayout(RPGraph::UGraph &graph,
                    float width = 10000, float height = 10000);
        ~GraphLayout();

        UGraph &graph; // to lay-out

        // randomize the layout position of all nodes.
        void randomizePositions();

        float getX(nid_t node_id), getY(nid_t node_id);
        float getXRange(), getYRange(), getSpan();
        float getWidth(), getHeight();
        float getDistance(nid_t n1, nid_t n2);
        float2 *getNodeCoordinates();
        float2 *getEdgeCoordinates();
        uint2 *getEdgePairs();
        Real2DVector getDistanceVector(nid_t n1, nid_t n2);
        Real2DVector getNormalizedDistanceVector(nid_t n1, nid_t n2);
        Coordinate getCoordinate(nid_t node_id);
        Coordinate getCenter();

        void set_node_opacity(float opacity);
        void set_edge_opacity(float opacity);

        void setX(nid_t node_id, float x_value), setY(nid_t node_id, float y_value);
        void moveNode(nid_t, Real2DVector v);
        void setCoordinates(nid_t node_id, Coordinate c);

        void writeToPNG(const int image_w, const int image_h, std::string path);
        void writeToCSV(std::string path);
        void writeToBin(std::string path);

        void enable_gl_renderer(int image_w, int image_h);
        void gl_draw_common();
        void gl_draw_nodes();
        void gl_draw_edges();
        void gl_draw_finish();
        void gl_draw_swap();
        void gl_draw_graph();
        void gl_draw_graph(float node_opacity, float edge_opacity);

        #ifdef __NVCC__
        void gl_map_pbo(screen_resources_t &sr);
        void gl_unmap_pbo(screen_resources_t &sr);
        void gl_map_pbos();
        void gl_unmap_pbos();
        float2 *gl_pbo_pointer(int cuda_gpu_id);
        #endif
        bool use_interop = false;

        void init_screens();

        // Hacks
        float gpu_minx, gpu_maxx, gpu_miny, gpu_maxy;
        std::vector<int> render_gpus;
    };
}

#endif /* RPGraphLayout_hpp */
