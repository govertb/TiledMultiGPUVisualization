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

void release_gl_context(Display *d)
{
    if(!glXMakeCurrent(d, None, NULL))
    {
        printf("Error: Couldn't release context.\n");
        exit(EXIT_FAILURE);
    }
}

#endif
