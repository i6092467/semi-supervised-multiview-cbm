"""
Utility functions for plotting
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00',
                  '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
HATCHINGS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
MARKERS = ['D', 'o', '^', 'v', 's', 'X', '*', 'D', 'o', '^', 'v', 's', 'X', '*', 'D', 'o', '^', 'v', 's', 'X', '*']


def plotting_setup(font_size=12):
    """
    Sets global plot formatting settings
    """
    plt.style.use("seaborn-colorblind")
    plt.rcParams['font.size'] = font_size
    rc('text', usetex=False)
    plt.rcParams["font.family"] = "sans-serif"
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})


def plot_curves_with_ci_lu(xs, avgs, lower, upper, labels, xlab, ylab, font_size=16, title=None, baseline=None,
                           baseline_lab=None, baseline_cl=None, dir=None, legend=True, legend_outside=True,
                           cls=None, ms=None, figsize=(10, 5), ylim=None, tick_step=None, grid=False):
    """
    Plots several curves with the given confidence bands
    """
    plotting_setup(font_size)

    fig = plt.figure(figsize=figsize)

    if cls is None:
        cls = CB_COLOR_CYCLE[:len(xs)]
    if ms is None:
        ms = MARKERS[:len(xs)]

    if baseline_cl is None:
        baseline_cl = 'red'

    if baseline is not None:
        plt.axhline(baseline, label=baseline_lab, c=baseline_cl, linestyle='--')

    for i in range(len(xs)):
        upper_i = upper[i]
        lower_i = lower[i]

        plt.plot(xs[i], avgs[i], color=cls[i], label=labels[i], marker=ms[i], markersize=16, linewidth=5)
        plt.fill_between(xs[i], lower_i, upper_i, color=cls[i], alpha=0.1)

    plt.xlabel(xlab)
    plt.ylabel(ylab)

    if title is not None:
        plt.title(title)

    if ylim is not None:
        plt.ylim(ylim)

    if tick_step is not None and ylim is not None:
        y_min = ylim[0]
        y_max = ylim[1]
        yticks = np.arange(y_min, y_max, tick_step)
        plt.yticks(ticks=yticks)

    if grid:
        plt.grid(visible=True, axis='y')

    if legend:
        if legend_outside:
            leg = plt.legend(loc='lower center', ncol=len(xs), bbox_to_anchor=(0.5, -0.3), frameon=False)
        else:
            leg = plt.legend(loc='upper right', frameon=False)

        # Change the marker size manually for both lines
        for i in range(len(leg.legendHandles)):
            leg.legendHandles[i].set_markersize(16)
            leg.legendHandles[i].set_linewidth(5.0)

    if dir is not None:
        plt.savefig(fname=dir, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_boxplots(vals, labels, xlab, ylab, font_size, xticks, xtick_labels, title=None, dir=None, legend=True,
                  legend_outside=True, legend_font_size=None, cls=None, hs=None, figsize=(10, 5),
                  xlim=None, ylim=None, tick_step=None, grid=False):
    """
    Plots several boxplots side-by-side
    """
    plotting_setup(font_size)

    fig = plt.figure(figsize=figsize)

    if cls is None:
        cls = CB_COLOR_CYCLE[:vals.shape[1]]
    if hs is None:
        hs = HATCHINGS[:vals.shape[1]]

    if legend_font_size is None:
        legend_font_size = font_size

    xs = np.arange(vals.shape[0])
    bs = []
    for i in range(len(xs)):
        for j in range(vals.shape[1]):
            b = plt.boxplot(
                x=vals[i, j, :],
                positions=[xs[i] - 0.14 * vals.shape[1] / 2 + 0.14 * vals.shape[1] / vals.shape[1] * j],
                widths=[0.14], showfliers=False, showmeans=True,
                boxprops=dict(color='black'), medianprops=dict(color='black', linewidth=3.0),
                meanprops=dict(marker='^', markeredgecolor='black', markerfacecolor='black', markersize=12.0),
                patch_artist=True)
            for box in b["boxes"]:
                box.set_facecolor(cls[j])
                box.set(hatch=hs[j])
                if i == 0:
                    bs.append(box)

    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.xticks(ticks=xticks, labels=xtick_labels)

    if title is not None:
        plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if tick_step is not None and ylim is not None:
        y_min = ylim[0]
        y_max = ylim[1]
        yticks = np.arange(y_min, y_max, tick_step)
        plt.yticks(ticks=yticks)

    if grid:
        plt.grid(visible=True, axis='y')

    if legend:
        if legend_outside:
            leg = plt.legend(bs, labels, loc='lower center', ncol=vals.shape[1],
                             bbox_to_anchor=(0.5, -0.3), frameon=False, prop={'size': legend_font_size})
        else:
            leg = plt.legend(bs, labels, loc='upper right', frameon=False, prop={'size': legend_font_size})

    if dir is not None:
        plt.savefig(fname=dir, dpi=300, bbox_inches='tight')
    else:
        plt.show()
