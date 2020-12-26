/*
 ==============================================================================

 gl_util.hpp
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef gl_util_hpp
#define gl_util_hpp
#include "../lib/glad/glad.h"
#include "../lib/glad/glad_glx.h"

void assert_compiled(GLuint shader);
void assert_linked(GLuint program);

GLuint gl_node_program();
GLuint gl_edge_program();

void release_gl_context(Display *d);
void set_gl_base_settings();

#endif
