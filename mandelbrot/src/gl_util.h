/*
 ==============================================================================

 gl_util.h
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef gl_util_h
#define gl_util_h

#include <stdio.h>
#include <fstream>
#include "../lib/glad/glad.h"
#include "../lib/glad/glad_glx.h"

// the vertex and texture coordinates required to draw the contents of the
// texture containing the Mandelbrot drawing, using two triangles
const float vertices[] =
{-1.0, 1.0,  0.0, 0.0,
  1.0, 1.0,  1.0, 0.0,
  1.0,-1.0,  1.0, 1.0,
  1.0,-1.0,  1.0, 1.0,
 -1.0,-1.0,  0.0, 1.0,
 -1.0, 1.0,  0.0, 0.0};

void release_gl_context(Display *d)
{
    if(!glXMakeCurrent(d, None, NULL))
    {
        printf("Error: Couldn't release context.\n");
        exit(EXIT_FAILURE);
    }
}

// assert `shader' was compiled successfully
void assert_compiled(GLuint shader)
{
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (!compiled)
    {
        GLint infoLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1)
        {
            char *infoLog = new char[infoLen];
            glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
            std::cerr << "Error compiling shader:\n" << infoLog << "\n";
            delete[] infoLog;
        }
        glDeleteShader(shader);
        exit(EXIT_FAILURE);
    }
}

// assert `program' was linked successfully
void assert_linked(GLuint program)
{
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked)
    {
        GLint infoLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen);

        if(infoLen > 1)
        {
            char *infoLog = new char[infoLen];
            glGetProgramInfoLog(program, infoLen, NULL, infoLog);
            std::cerr << "Error linking program:\n" << infoLog << "\n";
            delete[] infoLog;
        }

        glDeleteProgram(program);
        exit(EXIT_FAILURE);
    }
}

// gl program that draws the contents of a texture
GLuint gl_tex_program()
{
    // vertex shader
    const GLchar* vshader_source = R"glsl(
        #version 400 core
        layout(location=0) in vec2 pos;
        layout(location=1) in vec2 texcoord;

        out vec2 Texcoord;
        void main(void)
        {
            gl_Position = vec4(pos, 0.0, 1.0);
            Texcoord = texcoord;
        })glsl";
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vshader_source, NULL);
    glCompileShader(vertex_shader);

    // fragment shader
    const GLchar *fs_source = R"glsl(
        #version 400 core
        out vec4 fColor;
        in vec2 Texcoord;

        uniform sampler2D tex;
        void main(void)
        {
            fColor = texture(tex, Texcoord);
        })glsl";

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fs_source, NULL);
    glCompileShader(fragment_shader);

    // create program, attach shaders
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);

    // link program
    glBindFragDataLocation(program, 0, "fColor");
    glLinkProgram(program);
    assert_linked(program);

    return program;
}

// save the (im_w * im_h) rgba image contained in an (alloc_w * im_h) array
// to a ppm file
void rgba_to_ppm(const char *file, unsigned char *rgba_data,
                 int im_w, int im_h, int alloc_w)
{
    std::fstream f(file, std::fstream::out | std::fstream::binary);
    if (f.bad())
    {
        printf("Error: couldn't open file at %s\n", file);
        exit(EXIT_FAILURE);
    }

    f << "P6\n" << im_w << " " << im_h << "\n" << 255 << "\n";
    for(int y = 0; y < im_h; ++y)
    {
        for(int x = 0; x < im_w; ++x)
        {
            int idx = 4*x + 4*y*alloc_w;
            f << rgba_data[idx + 0];
            f << rgba_data[idx + 1];
            f << rgba_data[idx + 2];
        }
    }
    f.close();
}

#endif
