#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
#
# graph_viewer_benchmark.py
# Author: Govert Brinkmann, unless a 'due' is given.
#
# This code was developed as part of research at the Leiden Institute of
# Advanced Computer Science (https://liacs.leidenuniv.nl).
#
# ==============================================================================
#
# This script drives graph_viewer to obtain and store benchmark results as
# .csv files.
#

import csv
import glob
import itertools
import numpy as np
import os
import platform
import pynvml as nvml
import re
import subprocess
import sys

nvml.nvmlInit()
def get_gpu_busses():
    busses = list()
    for i in range(nvml.nvmlDeviceGetCount()):
        h = nvml.nvmlDeviceGetHandleByIndex(i)
        busses.append(nvml.nvmlDeviceGetPciInfo(h).busId)
    return busses

# return arguments common to all benchmarks, based on benchmark settings
def common_cmd(dataset):
    data_path = data_prefix + dataset
    return [graph_viewer_path, 'benchmark', 'gpu', g_type, str(f_r), str(f_g)] +\
           ['approximate', data_path, str(it_max)]

def clean_kernel_name(name):
    memcpy_regex = r'\[CUDA memcpy .to.\]'
    kernel_call_regex = r'.+(?=\(.*\))'

    # memcpy call
    if(re.match(memcpy_regex, name)):
        return name

    # kernel call, strip parameters
    elif(re.match(kernel_call_regex, name)):
        return re.match(kernel_call_regex, name).group(0)

    else:
        raise Exception("Cannot clean string '" + name + "'")

#                             # The Benchmarks #                              #

def run_warmup(num_gpus):
    cmd = common_cmd(warmup_data) + \
          ['layout_render', str(image_w), str(image_h), 'no_validate'] + \
          ['interop', ','.join(gpu_busses[:num_gpus])]
    r = subprocess.check_output(cmd)

def run_layout(max_gpus):
    out_file = open(results_prefix + 'layout_' + machine + '.csv', 'w')
    out_file.write(benchmark_string)
    out_file.write('dataset,it,time\n')
    for dataset, num_gpus in itertools.product(datasets, range(max_gpus, 0, -1)):
        cmd = common_cmd(dataset) +['layout', ','.join(gpu_busses[:num_gpus])]
        r = subprocess.check_output(cmd)
        times = filter(None, r.split('\n'))

        assert(len(times) == it_max)
        for run in range(it_max):
            out_file.write('%s,%d,%f\n' % (dataset, run, float(times[run])))


def run_kernels(num_gpus):
    running_times = dict() # running_times[dataset][kernel] = times for kernel
    for dataset in datasets:
        nvprof_cmd = ["nvprof", \
                      "--print-gpu-trace", \
                      "--profile-from-start", "off", \
                      "--csv", \
                      "--normalized-time-unit", "ms"]

        graph_viewer_cmd = common_cmd(dataset) +\
                           ['layout', ','.join(gpu_busses[:num_gpus])]
        cmd = nvprof_cmd + graph_viewer_cmd

        # start benchmark
        r = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # all nvprof output goes to stderr ?
        (stdout, stderr) = r.communicate()

        # check if number of iterations in regular output matches expected
        times = filter(None, stdout.split('\n'))
        assert(len(times) == it_max)

        # process nvprof gpu timeline
        nvprof_output = stderr
        nvprof_csv = list(csv.DictReader(filter(None, nvprof_output.split('\n')[3:]),
                                         dialect='excel'))

        # device to focus on, for multi-GPU benchmark
        device = nvprof_csv[2]['Device']

        # convert list of dicts (nvprof_csv) to structured numpy array
        records = list()
        for row in nvprof_csv[2:]:
            if(row['Device'] == device):
                records += [(clean_kernel_name(row['Name']), row['Duration'])]
        results = np.array(records, dtype={'names':('kernel', 'running_time'), \
                                           'formats' : ('S50', 'float32')})

        # compute and save statistics
        running_times[dataset] = dict()
        for kernel in set(results['kernel']):
            mask = (results['kernel'] == kernel)
            running_times[dataset][kernel] = results[mask]['running_time']


    # store results as csv file
    out_file = open(results_prefix + 'kernels_' + machine \
             + '_' + str(num_gpus) + '.csv', 'w')
    out_file.write(benchmark_string)

    # write csv header
    out_file.write('dataset')
    # determine kernels accross datasets, fix order in list for csv header
    kernels = [kernel for kernel in running_times[dataset].keys() for dataset in datasets]
    kernels = list(set(kernels)) # filter for dups, fix order.
    for kernel in kernels:
        out_file.write(',' + kernel.lower())
    out_file.write('\n')

    # write rows of results
    for dataset in datasets:
        out_file.write('%s' % dataset)
        for kernel in kernels:
            out_file.write(',%f' % np.mean(running_times[dataset][kernel]))
        out_file.write('\n')


def run_layout_render(max_gpus):
    out_file = open(results_prefix + 'layout_render_' + machine + '.csv', 'w')
    out_file.write(benchmark_string)
    out_file.write('dataset,num_gpus,it,render_type,time_compute,time_render_nodes,time_render_edges,time_render_complete\n')
    for num_gpus, dataset, render_type in itertools.product(range(max_gpus, 0, -1), datasets, ['interop']):
        cmd = common_cmd(dataset) + \
              ['layout_render', str(image_w), str(image_h), 'no_validate'] + \
              [render_type, ','.join(gpu_busses[:num_gpus])]
        r = subprocess.check_output(cmd)
        times = filter(None, r.split('\n'))

        assert(len(times) == it_max)
        for it in range(it_max):
            time_compute, time_render_nodes, time_render_edges, time_render_complete = times[it].split(',')
            out_file.write('%s,%d,%d,%s,%f,%f,%f,%f\n' \
                           % (dataset, num_gpus, it, render_type, float(time_compute), \
                              float(time_render_nodes), float(time_render_edges), \
                              float(time_render_complete)))


def run_gui():
    out_file = open(results_prefix + 'gui_' + machine + '.csv', 'w')
    out_file.write(benchmark_string)
    out_file.write('dataset,it,frame_time\n')

    for dataset in datasets:
        cmd = common_cmd(dataset) + ['gui', 'interop']
        r = subprocess.check_output(cmd)
        times = filter(None, r.split('\n'))
        assert(len(times) == it_max)
        for it in range(it_max):
            frame_time = times[it].split(',')[0]
            out_file.write('%s,%d,%s\n' % (dataset, it, frame_time))


def print_usage():
    print('Usage: graph_viewer_benchmark.py layout         [max_gpus]')
    print('                                 layout_render  [max_gpus]')
    print('                                 kernels        [num_gpus]')
    print('                                 gui            ')

# datasets to use for benchmarking
bigeye_data =[
    "Cit-HepTh.txt_gc.e",
    "ca-HepPh.nde_gc.e",
    "Newman-Cond_mat_95-99-two_mode.nde_gc.e",
    "ppi_dip_swiss.nde_gc.e",
    "ca-HepTh.nde_gc.e",
    "wiki-Vote.nde_gc.e",
    "PGPgiantcompo.nde_gc.e",
    "p2p-Gnutella31.nde_gc.e",
    "ca-AstroPh.nde_gc.e",
    "dip20090126_MAX.nde_gc.e",
    "GoogleNw.nde_gc.e",
    "soc-Epinions1_gc.e",
    "petster-friendships-hamster_gc.e",
    "ppi_gcc.nde_gc.e",
    "email-Enron.nde_gc.e",
    "soc-Slashdot0902.nde_gc.e",
    "Brightkite_edges.txt_gc.e",
    "ca-CondMat.nde_gc.e",
    "Cit-HepPh.txt_gc.e",
    "email-EuAll.nde_gc.e",
    "CA-GrQc.txt_gc.e"]

duranium_data = [
    "com-amazon.ungraph.txt_gc.e",
    "cnr_2000.nde_gc.e",
    "Amazon0505.txt_gc.e",
    "dblp20080824_MAX.nde_gc.e",
    "auto.nde_gc.e",
    "ydata-ysm-advertiser-phrase-graph-v1_0.nde_gc.e",
    "youtube_gc.e",
    "wiki_gc.e",
    "orkut_gc.e"]

# Paths relative to 'home'
graph_viewer_path = 'code/graph_viewer/dev_repo/builds/linux/graph_viewer'
data_prefix = 'data/edgelists/thesis/'
results_prefix = 'results/netvis/'

# Machine dependent settings
machine = platform.node().split('.')[0]
if(machine == 'bigeye'):
    datasets = bigeye_data
    warmup_data = 'p2p-Gnutella31.nde_gc.e'
    graph_viewer_path = '~/[student_id]/' + graph_viewer_path
    data_prefix = '~/[student_id]/' + data_prefix
    results_prefix = '~/[student_id]/' + results_prefix
    gpu_busses = get_gpu_busses()

elif(machine == 'duranium'):
    datasets = duranium_data
    warmup_data = 'Amazon0505.txt_gc.e'
    graph_viewer_path = '~/' + graph_viewer_path
    data_prefix = '~/' + data_prefix
    results_prefix = '~/' + results_prefix
    gpu_busses = ['0000:05:00.0', '0000:08:00.0', '0000:09:00.0',
                  '0000:85:00.0', '0000:89:00.0', '0000:8A:00.0']

else:
    datasets = bigeye_data
    warmup_data = 'p2p-Gnutella31.nde_gc.e'
    graph_viewer_path = '~/' + graph_viewer_path
    data_prefix = '~/' + data_prefix
    results_prefix = '~/' + results_prefix
    gpu_busses = get_gpu_busses()

graph_viewer_path = os.path.expanduser(graph_viewer_path)
data_prefix = os.path.expanduser(data_prefix)
results_prefix = os.path.expanduser(results_prefix)

# common benchmark settings
it_max = 500
f_r = 80
f_g = 1
g_type = "sg"
r_algo = "approximate"
image_w, image_h = 5760, 4320
benchmark_string = '# benchmarking using f_g = %f, f_r = %f, it_max = %f\n' % \
                   (f_g, f_r, it_max)

# Argument parsing
if(len(sys.argv) < 2):
    print_usage()
    exit(-1)

mode = sys.argv[1]
if(len(sys.argv) == 3):
    num_gpus = int(sys.argv[2])
else:
    num_gpus = 1

if(mode == 'layout'):
    run_warmup(num_gpus)
    run_layout(num_gpus)

elif(mode == 'kernels'):
    run_warmup(num_gpus)
    run_kernels(num_gpus)

elif(mode == 'layout_render'):
    run_warmup(num_gpus)
    run_layout_render(num_gpus)

elif(mode == 'gui'):
    run_warmup(3)
    run_gui()
