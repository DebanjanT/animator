/*
    GLAD - OpenGL Loader Generator
    Simplified version for OpenGL 3.3 Core on macOS
*/

#ifndef GLAD_H
#define GLAD_H

#ifdef __APPLE__
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
    #include <OpenGL/gl3ext.h>
    
    // On macOS, OpenGL functions are directly available
    #define gladLoadGLLoader(x) 1
    
#else
    #error "This simplified GLAD is macOS-only. Use full GLAD for other platforms."
#endif

#endif // GLAD_H
