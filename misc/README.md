# Miscellaneous Materials

This directory contains scripts that can be used to benchmark the framework, as well as some configuration files.
 

 - `benchmarks` contains the Python scripts to run both the Mandelbrot and Network Visualization experiments. The scripts outputs results as `.csv` files.

- `conf` contains a number of configuration files.
    - `xorg_tiled.conf` : X configuration used for tiled experiments (disables Xinerama)
    - `xorg_xinerama.conf` : X configuration used for Xinerama experiments (enables Xinerama)
    - `xorg_singlemonitor.conf` : X configuration used for single monitor display experiments. This enables only one of the monitors in BigEye.
    - `xwiimote.conf` : Configuration file used for the XWiiMote input driver.
     It maps WiiMote buttons/events to keyboard/mouse events and was located
     at `/usr/share/X11/xorg.conf.d/60-xorg-xwiimote.conf`.
