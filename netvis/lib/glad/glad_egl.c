/*

    EGL loader generated by glad 0.1.20a0 on Thu May 17 09:52:12 2018.

    Language/Generator: C/C++
    Specification: egl
    APIs: egl=1.5
    Profile: -
    Extensions:
        EGL_EXT_device_enumeration,
        EGL_EXT_device_query,
        EGL_EXT_platform_base,
        EGL_EXT_platform_device,
        EGL_NV_device_cuda
    Loader: True
    Local files: True
    Omit khrplatform: True

    Commandline:
        --api="egl=1.5" --generator="c" --spec="egl" --local-files --omit-khrplatform --extensions="EGL_EXT_device_enumeration,EGL_EXT_device_query,EGL_EXT_platform_base,EGL_EXT_platform_device,EGL_NV_device_cuda"
    Online:
        http://glad.dav1d.de/#language=c&specification=egl&loader=on&api=egl%3D1.5&extensions=EGL_EXT_device_enumeration&extensions=EGL_EXT_device_query&extensions=EGL_EXT_platform_base&extensions=EGL_EXT_platform_device&extensions=EGL_NV_device_cuda
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glad_egl.h"

int gladLoadEGL(void) {
    return gladLoadEGLLoader((GLADloadproc)eglGetProcAddress);
}

PFNEGLQUERYDEVICESEXTPROC glad_eglQueryDevicesEXT;
PFNEGLQUERYDEVICEATTRIBEXTPROC glad_eglQueryDeviceAttribEXT;
PFNEGLQUERYDEVICESTRINGEXTPROC glad_eglQueryDeviceStringEXT;
PFNEGLQUERYDISPLAYATTRIBEXTPROC glad_eglQueryDisplayAttribEXT;
PFNEGLGETPLATFORMDISPLAYEXTPROC glad_eglGetPlatformDisplayEXT;
PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC glad_eglCreatePlatformWindowSurfaceEXT;
PFNEGLCREATEPLATFORMPIXMAPSURFACEEXTPROC glad_eglCreatePlatformPixmapSurfaceEXT;
static void load_EGL_EXT_device_enumeration(GLADloadproc load) {
	glad_eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)load("eglQueryDevicesEXT");
}
static void load_EGL_EXT_device_query(GLADloadproc load) {
	glad_eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC)load("eglQueryDeviceAttribEXT");
	glad_eglQueryDeviceStringEXT = (PFNEGLQUERYDEVICESTRINGEXTPROC)load("eglQueryDeviceStringEXT");
	glad_eglQueryDisplayAttribEXT = (PFNEGLQUERYDISPLAYATTRIBEXTPROC)load("eglQueryDisplayAttribEXT");
}
static void load_EGL_EXT_platform_base(GLADloadproc load) {
	glad_eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)load("eglGetPlatformDisplayEXT");
	glad_eglCreatePlatformWindowSurfaceEXT = (PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC)load("eglCreatePlatformWindowSurfaceEXT");
	glad_eglCreatePlatformPixmapSurfaceEXT = (PFNEGLCREATEPLATFORMPIXMAPSURFACEEXTPROC)load("eglCreatePlatformPixmapSurfaceEXT");
}
static int find_extensionsEGL(void) {
	return 1;
}

static void find_coreEGL(void) {
}

int gladLoadEGLLoader(GLADloadproc load) {
	(void) load;
	find_coreEGL();

	if (!find_extensionsEGL()) return 0;
	load_EGL_EXT_device_enumeration(load);
	load_EGL_EXT_device_query(load);
	load_EGL_EXT_platform_base(load);
	return 1;
}

