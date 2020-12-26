Network Visualization on Tiled Display Systems
=================================================================

![network visualization on tiled display system](../img/netvis_1280.jpg)

#### Description
Visualize and navigate networks using a distributed GPGPU implementation of ForceAtlas2, and a rendering approach in which each GPU renders the part of the layout displayed on the monitors attached to it.
The (single-GPU) ForceAtlas2 implementation this project is based on, is available from
[https://github.com/govertb/GPUGraphlayout](https://github.com/govertb/GPUGraphlayout).


#### Compilation
1. The code depends on a number of git submodules (`lib/colormap`, `lib/glm` and `lib/pngwriter`). Make sure to run `git submodule init && git submodule update` from the root of this repository to checkout this code before compiling. The code also depends on `libpng` and its development headers, which should be installable via the package manager of the OS.

2. Run `make graph_viewer` from `build/linux` to compile.

#### Running
Disable Xinerama for optimal performance:
1. Edit the X configuration file
   (`/etc/X11/xorg.conf`), changing `Option "Xinerama" "1"` to
   `Option "Xinerama" "0"`. If the former option isn't present, add the latter.

2. After changing the configuration, restart X for the changes to take effect
  using `sudo systemctl restart lightdm`.

Run `./graph_viewer gui gpu sg|wg scale gravity approximate
                        edgelist_path` to start the visualization.

The different options can be used to as follows.

`wg|sg`              : choose between weak gravity or strong gravity

`scale`              : scale repulsive force

`gravity`            : scale gravitational force

`edgelist_path`      : ASCII file containing node IDs for each edge on a separate line (whitespace separated)

#### Using the WiiMote

`XWiimote` should be configured on BigEye, so binding the WiiMote should
be enough to use it as the input device. If `XWiiMote` is not installed or confiugred, see [install XWiimote](#install_xwiimote).

To bind the WiiMote,
1. Position the WiiMote in front of middle column of displays
   (for calibration) at a distance of approximately 2m, pointed towards
   the displays.

2. Put the WiiMote in binding mode (by holding the 1+2 buttons for some
   seconds). The LEDs will blink.

3. Bind the WiiMote via the Bluetooth menu (topright of the display).

#####  Controls
Interaction using the WiiMote is through keyboard/mouse simulation.

- `Arrow keys`: opcacity of nodes and edges.
  - `left-right` controls edge opacity.
  - `up-down` conrols node opacity.
- `A button`: stop/start layout algorithm.
- `B button` : left-mouse
- `+/- buttons`: zoom in/out.
- `home button`: reset view.
- `1 button` : activate/deactivate local repulsion.
- `2 button` : activate/deactivate local heat.

Navigating around the layout is possible by dragging the mouse. The mouse
follows the position pointed to by the WiiMote and the button on the back
of the WiiMote functions as the left mouse button. Navigate through
click-drag-release interactions.

#### Installing and configuring XWiiMote
<a name="install_xwiimote"></a>
1. Ensure the `hid-wiimote` kernel module is installed (should be
   included with recent Linux kernels) and loaded (e.g. by adding it to
   `/etc/modules` or by using `modprobe`).

2. Install XWiiMote and the corresponding X input driver. For Ubuntu:
   `sudo apt install xwiimote xserver-xorg-input-xwiimote`.

3. Update the configuration of the input driver
   (`/usr/share/X11/xorg.conf.d/60-xorg-xwiimote.conf`) according to the
   [`xwiimote.conf` file](../misc/conf/xwiimote.conf).

#### Possible issues

> The mouse movements produced by WiiMote aren't fluid.

Check that the batteries are charged sufficiently (> 50%) or move closer
to the display/Bluetooth receiver.

> The mouse movements using the WiiMote are completely
  random / uncontrollable.

Try rebinding the WiiMote after disconnecting through the Bluetooth menu
(top-right of the display).
