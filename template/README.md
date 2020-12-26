Tiled Visualization Approach for BigEye
=================================================================

This is a 'template' implementing the tiled visualization approach. It is provided as an illustration/pseudocode and not tested.

#### Compilation
1. Run `./build.sh` to compile all code.

#### Running
Disable Xinerama for optimal performance:
1. Edit the X configuration file
   (`/etc/X11/xorg.conf`), changing `Option "Xinerama" "1"` to
   `Option "Xinerama" "0"`. If the former option isn't present, add the latter.

2. After changing the configuration, restart X for the changes to take effect
  using `sudo systemctl restart lightdm`.
