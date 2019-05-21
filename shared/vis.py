from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_colorxyzs(xyzs, zdir='y', cmap='Reds', xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    elev = 30
    azim = -45
    ax.view_init(elev, azim)
    for xyz, color in xyzs:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], c=color, s=20, zdir=zdir, cmap=cmap, vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    return fig



def plot_xyz(xyz, zdir='y', cmap='Reds', xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    elev = 30
    azim = -45
    ax.view_init(elev, azim)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], c=xyz[:,0], s=20, zdir=zdir, cmap=cmap, vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    return fig

def plot_pcds(filename, pcds, titles, use_color=[],color=None, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            clr = color[j]
            if color is None or not use_color[j]:
                clr = pcd[:, 0]

            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=clr, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def plot_pcds_patterns(filename, pcds, titles, use_color=[],color=None, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            clr = color[j]
            if color is None or not use_color[j]:
                clr = pcd[:, 0]

            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            rgba = [matplotlib.colors.rgb2hex(tuple(k[0:4])) for k in clr.tolist()]
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, color=rgba, s=clr[:,4], cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
