#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys


def plot(fname):
    pos = np.genfromtxt(fname, delimiter=",")
    xs = pos[:,0]
    ys = pos[:,1]

    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(xs, ys, '.')

    xmax = max(abs(np.max(xs)), abs(np.min(xs)))
    ymax = max(abs(np.max(ys)), abs(np.min(ys)))
    xmax = 20
    ymax = 20
    xmin = -xmax
    ymin = -ymax

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(fname + ".png")
    plt.close()


if(len(sys.argv) > 1):
    for fname in sys.argv[1:]: plot(fname)
else:
    print("Usage: ./plot.py <filenames>")
