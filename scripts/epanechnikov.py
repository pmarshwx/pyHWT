#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import hwt

if __name__ == '__main__':

    nx = 200
    hnx = int(nx / 2)

    data = np.zeros((nx, nx), dtype=float)
    data[hnx, hnx] = 1.

    dx = 4.
    bandwidth = 25.
    data1 = hwt.cfuncs.smoothers.epanechnikov(data, bandwidth, dx, True)
    data2 = hwt.cfuncs.smoothers.isotropic_gauss(data, bandwidth*dx, dx, 3)

    fig = plt.figure(figsize=(12,12))
    grid = ImageGrid(fig, 111, nrows_ncols = (2, 1), direction="row",
           axes_pad = 0.05, add_all=True, label_mode = "1",
           share_all = True, cbar_location="right", cbar_mode="each",
           cbar_size="7%", cbar_pad="1%")

    ax1 = grid[0]
    ax2 = grid[1]
    levs = np.arange(0, 100.001, 1) / 1000.
    cs1 = ax1.contourf(data1 * 100., levels=levs)
    ax1.set_aspect(1)
    cs2 = ax2.contour(data2 * 100., levels=levs)
    ax2.set_aspect(1)

    ax1.cax.colorbar(cs1, format='%.03f')
    ax1.cax.toggle_label(True)
    ax2.cax.colorbar(cs2, format='%.03f')
    ax2.cax.toggle_label(True)

    plt.savefig('gauss_vs_epan.png', dpi=100, bbox_inches='tight')
    plt.show()
