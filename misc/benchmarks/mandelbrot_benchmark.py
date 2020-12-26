#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
#
# mandelbrot_benchmark.py
# Author: Govert Brinkmann, unless a 'due' is given.
#
# This code was developed as part of research at the Leiden Institute of
# Advanced Computer Science (https://liacs.leidenuniv.nl).
#
# ==============================================================================
#
# This script drives the mandelbrot binary to obtain and store
# benchmark results as .csv files.
#


import glob
import itertools
import os
import platform
import subprocess
import sys

# common arguments for all benchmarks, given w, h
def common_args(w, h):
    cmd = ['w', str(w),
           'h', str(h),
           'o_re', str(offset_re),
           'o_im', str(offset_im),
           'range_re', str(range_re),
           'it_max', str(it_max),
           'avg_factor', str(avg_factor)]
    if(validate):
        cmd += ['should_validate']
    return cmd

def run_cpu_compute():
    out_file = open(results_path + 'cpu_compute_' + machine + '.csv', 'w')
    out_file.write(benchmark_description)
    out_file.write('w,h,vectorize,num_threads,run,duration\n')

    max_threads = int(subprocess.check_output(['nproc']))
    for (w, h), vectorize, num_threads in itertools.product(dimensions, [0, 1], [1, max_threads]):
        assert(w % 8 == 0) # ensure w is multiple of SIMD width (SSE2, AVX)
        cmd = [bin_file, 'benchmark', 'cpu'] + common_args(w,h) + \
              ['cpu_threads', str(num_threads), 'vectorize', str(vectorize)]
        r = subprocess.check_output(cmd)
        durations = filter(None, r.split('\n'))
        assert(len(durations) == avg_factor)

        for run in range(avg_factor):
            out_file.write('%d,%d,%d,%d,%d,%s\n' \
                           % (w,h,vectorize,num_threads,run,durations[run]))

def run_gpu_compute():
    out_file = open(results_path + 'gpu_compute_' + machine + '.csv', 'w')
    out_file.write(benchmark_description)
    out_file.write('w,h,run,duration\n')

    for (w, h) in dimensions:
        cmd = [bin_file, 'benchmark', 'gpu'] + common_args(w,h) +\
              ['num_gpus', '1']
        r = subprocess.check_output(cmd)
        durations = filter(None, r.split('\n'))
        assert(len(durations) == avg_factor)

        for run in range(avg_factor):
            out_file.write('%d,%d,%d,%s\n' % (w,h,run,durations[run]))


def run_display():
    out_file = open(results_path + 'display_' + machine + '.csv', 'w')
    out_file.write(benchmark_description)
    out_file.write('w,h,implementation,run,compute_duration,display_duration\n')

    cpu_threads = int(subprocess.check_output(['nproc']))
    for (w,h) in dimensions:
        # Run CPU
        cmd = [bin_file, 'benchmark', 'cpu'] + common_args(w,h) + \
              ['benchmark_display', 'cpu_threads', str(cpu_threads), \
               'vectorize', '1']
        r = subprocess.check_output(cmd)
        durations = filter(None, r.split('\n'))
        assert(len(durations) == avg_factor)

        for run in range(avg_factor):
            compute_duration, display_duration = durations[run].split(',')
            out_file.write('%d,%d,cpu,%d,%s,%s\n' % \
                           (w,h,run,compute_duration,display_duration))

        # Run GPU
        for interop in [0, 1]:
            cmd = [bin_file, 'benchmark', 'gpu'] + common_args(w,h) + \
                  ['benchmark_display', 'interop', str(interop), 'num_gpus', '1'];
            r = subprocess.check_output(cmd)
            durations = filter(None, r.split('\n'))
            assert(len(durations) == avg_factor)

            modestring = 'gpu'
            if(interop == 1):
                modestring += '_insitu'
            for run in range(avg_factor):
                compute_duration, display_duration = durations[run].split(',')
                out_file.write('%d,%d,%s,%d,%s,%s\n' % \
                               (w,h,modestring,run,compute_duration,display_duration))


def run_gui():
    out_file = open(results_path + 'gui_' + machine + '.csv', 'w')
    out_file.write('case,it,frame_duration\n')
    for case in ['worst', 'average']:
        if(case == 'worst'):
            offset_re = 0.0
            offset_im = 0.0
            range_re = 0.1

        elif(case == 'average'):
            offset_re = -0.5
            offset_im = 0.0
            range_re = 3.2

        cmd = [bin_file, 'benchmark', 'gpu', 'benchmark_gui', 'interop', '1',
               'o_re', str(offset_re),
               'o_im', str(offset_im),
               'range_re', str(range_re),
               'it_max', str(it_max),
               'avg_factor', str(avg_factor)]

        r = subprocess.check_output(cmd)
        durations = filter(None, r.split('\n'))
        assert(len(durations) == avg_factor)

        for run in range(avg_factor):
            frame_time = durations[run].split(',')[0]
            out_file.write('%s,%d,%s\n' % (case,run,frame_time))

def cpu_fgovernor_is(setting):
    for fp in [open(f) for f in glob.glob("/sys/devices/system/cpu/*/cpufreq/scaling_governor")]:
        if(fp.read()[:-1] != setting):
            return False
    return True


def print_usage():
    print('Usage: mandelbrot_benchmark.py cpu_compute  [validate] [skip_cf_check]')
    print('                               gpu_compute  [validate] [skip_cf_check]')
    print('                               display      [validate] [skip_cf_check]')
    print('                               gui          [validate] [skip_cf_check]')

# machine running the benchmark
machine = platform.node().split('.')[0]

# take care if using shared user
if(machine == 'bigeye'):
    homedir = os.path.expanduser('~/[student_id]/')
else:
    homedir = os.path.expanduser('~/')

# where is benchmark binary and where should results go
bin_file = homedir + 'code/mandelbrot/mandelbrot'
results_path = homedir + 'results/mandelbrot/'

# constant benchmark settings
offset_re = 0.0 # worst-case
offset_im = 0.0 # worst-case
range_re = 0.1  # worst-case
it_max = 255
avg_factor = 25
dimensions = [(40*2**i, 40*2**i) for i in range(0, 8)] + [(1920, 1080*4)]
validate = False
benchmark_description = '# benchmarking with o_re=%f,o_im=%f,range_re=%f,' \
                        'it_max=%d, avg_factor=%d\n' \
                      % (offset_re, offset_im, range_re, it_max, avg_factor)

# parse arguments
if(len(sys.argv) < 2):
    print_usage()
    exit(-1)

cf_check = True;
mode = sys.argv[1]
for arg in sys.argv[2:]:
    if(arg == 'skip_cf_check'):
        cf_check = False
    elif(arg == 'validate'):
        range_re = 3.2
        avg_factor = 3
        validate = True
    else:
        print('Unknown argument: %s' % arg)
        exit(-1)

if(cf_check and not cpu_fgovernor_is("performance")):
    print('Error: CPU frequency scaling governor is not set to performance')
    exit(-1)

# run benchmark
if(mode == 'cpu_compute'):
    run_cpu_compute()

elif(mode == 'gpu_compute'):
    run_gpu_compute()

elif(mode == 'display'):
    run_display()

elif(mode == 'gui'):
    run_gui()

else:
    print_usage()
