/*
 ==============================================================================

 x11_util.h
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef x11_util_h
#define x11_util_h

#include "../lib/util-keysyms/keysyms/xcb_keysyms.h"

void ignore_wm_redirect(xcb_connection_t *xcb_connection, xcb_window_t window)
{
    const uint32_t value_mask = XCB_CW_OVERRIDE_REDIRECT;
    const uint32_t values[] = {1, 0};

    xcb_change_window_attributes(xcb_connection, window, value_mask, values);
}

// wrapper to get keysym from keycode
// due: https://github.com/Cloudef/monsterwm-xcb/blob/master/monsterwm.c#L590
static xcb_keysym_t xcb_get_keysym(xcb_connection_t *xcb_connection,
                                   xcb_keycode_t keycode)
{
    xcb_key_symbols_t *keysyms;
    xcb_keysym_t       keysym;

    if (!(keysyms = xcb_key_symbols_alloc(xcb_connection))) return 0;
    keysym = xcb_key_symbols_get_keysym(keysyms, keycode, 0);
    xcb_key_symbols_free(keysyms);

    return keysym;
}

#endif
