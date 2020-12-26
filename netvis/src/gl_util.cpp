/*
 ==============================================================================

 gl_util.cpp
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#include <stdio.h>
#include <iostream>
#include "gl_util.hpp"

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

// gl program used to draw the nodes
GLuint gl_node_program()
{
    const GLchar* vertex_shader_source = R"glsl(
        #version 410 core
        layout(location=0) in vec2 pos;
        layout(location=1) in vec4 node_color;
        layout(std140) uniform GlobalMatrices
        {
            mat4 model;
            mat4 view;
        };

        out vec4 node_color_;

        void main(void)
        {
            node_color_ = node_color;
            gl_Position = view * model * vec4(pos, 0.0, 1.0);
        })glsl";


    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);

    // Compile fragment shader
    // Anti-aliased circle, due:
    // https://www.desultoryquest.com/blog/drawing-anti-aliased-circular-points-using-opengl-slash-webgl/
    const GLchar * fs_source = R"glsl(
        #version 410 core
        in vec4 node_color_;
        out vec4 fColor;
        uniform float node_opacity;
        void main(void)
        {
            float r = 0.0, delta = 0.0;
            vec2 cxy = 2.0 * gl_PointCoord - 1.0;
            r = dot(cxy, cxy);
            delta = fwidth(r);
            float alpha = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, r);
            alpha *= node_opacity;
            vec4 c = node_color_;
            c.w = alpha;
            fColor = c;
        })glsl";

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fs_source, NULL);
    glCompileShader(fragment_shader);

    // Create program, link it and bind fragment shader output.
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);

    // Link Program
    glBindFragDataLocation(program, 0, "fColor");
    glLinkProgram(program);
    assert_linked(program);

    return program;
}


// gl program used to draw the edges
// Anti-aliasing approach due to:
// https://vitaliburkov.wordpress.com/2016/09/17/simple-and-fast-high-quality-antialiased-lines-with-opengl/
GLuint gl_edge_program()
{
    const GLchar* vertex_shader_source = R"glsl(
        #version 410 core
        layout(location=0) in vec2 pos;
        layout(location=1) in vec4 node_color;
        out vec2 vLineCenter;
        out vec4 node_color_;
        layout(std140) uniform GlobalMatrices
        {
            mat4 model;
            mat4 view;
            vec2 vp;
        };

        void main(void)
        {
            node_color_ = node_color;
            vec4 pp = view * model * vec4(pos, 0.0, 1.0);
            gl_Position = pp;
            vLineCenter = 0.5*(pp.xy + vec2(1, 1))*vp;
        })glsl";

    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);
    assert_compiled(vertex_shader);

    // Compile fragment shader
    const GLchar * fs_source = R"glsl(
    #version 410 core
    in vec2 vLineCenter;
    in vec4 node_color_;
    out vec4 fColor;
    uniform float edge_opacity;
    
    void main(void)
    {
        float uBlendFactor = 1.5; // //1.5..2.5

        float alpha = 1.0;  // initial
        float d = length(vLineCenter-gl_FragCoord.xy);
        float w = 1; // desired linewidth
        if(d > w)
          alpha = 0.0;
        else
          alpha *= pow(float((w-d)/w), uBlendFactor);
        alpha *= edge_opacity;
        vec4 c = node_color_;
        c.w = alpha;
        fColor = c;
    })glsl";

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fs_source, NULL);
    glCompileShader(fragment_shader);
    assert_compiled(fragment_shader);

    // Create program, link it and bind fragment shader output.
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);

    // Link Program
    glBindFragDataLocation(program, 0, "fColor");
    glLinkProgram(program);
    assert_linked(program);

    return program;
}

void release_gl_context(Display *d)
{
    if(!glXMakeCurrent(d, None, NULL))
    {
        printf("Error: Couldn't release context.\n");
        exit(EXIT_FAILURE);
    }
}

void set_gl_base_settings()
{
    // Following is not possible on NVIDIA cards, ultra-slow...
    // glEnable(GL_LINE_SMOOTH);
    // glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Blending or fragments to pixels, appears to be safe.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPointSize(10.0f);
}