import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.text import Text
import numpy as np
from matplotlib.colors import colorConverter
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Patch

def mean_in_bins(x, y, bins, ywerr=None, xrange=None):
    """Compute <y> in each x bins."""
    
    if isinstance(bins, (list, np.ndarray)):
        xedges = bins
    elif xrange is None:
        xedges = np.linspace(np.min(x), np.max(x), bins+1)
    else:
        xedges = np.linspace(xrange[0], xrange[1], bins+1)
    xbins = []
    ybins = []
    yerr = []
    std = []
    for xa, xb in zip(xedges[:-1], xedges[1:]):
        mask = (x > xa) & (x < xb)
        mask &= ~np.isnan(x)
        mask &= ~np.isnan(y)    
        yinb = y[mask]
        xbins.append((xa + xb) * 0.5)
        if ywerr is None:
            ybins.append(np.mean(yinb))
            yerr.append(np.std(yinb)/np.sqrt(np.sum(mask)))
            std.append(np.std(yinb))
        else:
            winb = ywerr[mask]
            ybins.append(np.sum(yinb / winb**2) / np.sum(1 / winb**2))
            yerr.append(1 / np.sqrt(np.sum(1 / winb**2)))
            std.append(np.sqrt(np.mean(winb**2)))
    return np.array(xbins), np.array(ybins), np.array(yerr), np.array(std)


def plot_xydist(x, y, setmain={}, setydist={}, setxdist={}, 
                residuals=None, setres={}, res_kwargs={}, xbins=10, 
                ybins=10, xrange=None, yrange=None, wspace=0.01, 
                return_fig=False, figsize=(5, 5), **kwargs):
    """Plot y / x, histogram of y, histogram of x and residuals"""
    
    if xrange is not None:
        xrange = np.atleast_1d(xrange)
        if len(xrange) == 1:
            xrange[0] = np.abs(xrange[0])
            xrange = [-xrange[0], xrange[0]]
            
    if yrange is not None:
        yrange = np.atleast_1d(yrange)
        if len(yrange) == 1:
            yrange[0] = np.abs(yrange[0])
            yrange = [-yrange[0], yrange[0]]

    
    if residuals is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3)
        ax_main = fig.add_subplot(gs[1:, :2])
        ax_xDist = fig.add_subplot(gs[0, :2], sharex=ax_main)
        ax_yDist = fig.add_subplot(gs[1:3, 2], sharey=ax_main)
    else:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 4)
        ax_residuals = fig.add_subplot(gs[3, :2])
        ax_main = fig.add_subplot(gs[1:3, :2], sharex=ax_residuals)
        ax_xDist = fig.add_subplot(gs[0, :2], sharex=ax_residuals)
        ax_yDist = fig.add_subplot(gs[1:3, 2], sharey=ax_main)
        ax_main.tick_params(labelbottom=False)
        ax_main.tick_params(axis='x', which='both', bottom=False)    
        ax_residuals.scatter(x, residuals, **res_kwargs)
        ax_residuals.set(**setres)

    plot = ax_main.scatter(x, y, **kwargs)
    ax_main.set(**setmain)
    
    ax_xDist.tick_params(axis='x', which='both', direction='in')    
    ax_xDist.hist(x, bins=xbins, align='mid', range=xrange)
    ax_xDist.set(**setxdist)

    ax_yDist.tick_params(axis='y', which='both', direction='in')    
    ax_yDist.hist(y, bins=ybins, orientation='horizontal', align='mid', range=yrange)
    ax_yDist.set(**setydist)
    
    if yrange is not None:
        ax_main.set_ylim(yrange[0], yrange[1])
    if xrange is not None:
        ax_main.set_xlim(xrange[0], xrange[1])
        
    fig.subplots_adjust(wspace=wspace, hspace=wspace)
    ax_xDist.tick_params(labelbottom=False)
    ax_yDist.tick_params(labelleft=False)

    if return_fig:
        return fig, plot
    plt.show()
    
    
def plot_ydist(x, y, setmain={}, setdist={}, 
                  bins=10, range=None, wspace=0.01, 
                  return_fig=False, **kwargs):
    
    range = np.atleast_1d(range)
    if len(range) == 1:
        range[0] = np.abs(range[0])
        range = [-range[0], range[0]]
        
    fig = plt.figure(figsize=(15, 7))
    
    gs = gridspec.GridSpec(1,2, width_ratios=[2, 1])
    ax_main = plt.subplot(gs[0])
    ax_yDist = plt.subplot(gs[1], sharey=ax_main)
        
    plot = ax_main.scatter(x, y, **kwargs)
    ax_main.set(**setmain)


    ax_yDist.tick_params(axis='y', which='both', direction='in')    
    ax_yDist.hist(y, bins=bins, orientation='horizontal', align='mid', range=range)
    ax_yDist.set(**setdist)
    
    if range is not None:
        ax_main.set_ylim(range[0], range[1])

    fig.subplots_adjust(hspace=wspace, wspace=wspace)
    ax_yDist.tick_params(labelleft=False)
    print(wspace)

    if return_fig:
        return fig, plot
    plt.show()
    
def add_patch(legend, handle, label):
    ax = legend.axes

    handles, labels = ax.get_legend_handles_labels()
    handles.append(handle)
    labels.append(label)

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())


def plot_density(datax, datay, ax, bins_2d, xrange=None, yrange=None, add_txt=None, weights=None, levels=[0.25, 0.5], color=None, smooth=None, fill=True, label=None):

    if isinstance(bins_2d, int):
        bins_2d = [bins_2d, bins_2d]
        
    if isinstance(bins_2d[0], int):
        bins_2d = [np.linspace(*xrange, num=bins_2d[0]), np.linspace(*yrange, num=bins_2d[1])]
        
    

    #####################
    # Taken from corner #
    #####################
    
    if color is None:
        color = 'C0'
    # Choose the default "sigma" contour levels.
    
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5)** 2)
    
    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    H, X, Y = np.histogram2d(
            datax.flatten(),
            datay.flatten(),
            bins=bins_2d,
            weights=weights)
    
    if smooth is not None:
        H = gaussian_filter(H, smooth)

    Hflat = H.flatten()
    
    # Sort from highest to smallest
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m) and not quiet:
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )
    if fill:
        contourf_kwargs ={}
        ax.contourf(
                X2,
                Y2,
                H2.T,
                [0, *V, H.max()],
                colors=contour_cmap, 
            )
    else:
        ax.contour(
                X2,
                Y2,
                H2.T,
                [*V, H.max()],
                colors=color,
            )
    
    

    # Create custom line handles for additional elements
    custom_handles = [Patch(facecolor=color)]
    custom_labels = [label]
    
    if ax.legend_ is not None:
        # Get existing handles and labels from the legend
        existing_handles, existing_labels = ax.legend_.legend_handles, [t.get_text() for t in ax.legend_.texts]

        # Combine existing and custom handles and labels
        all_handles = existing_handles + custom_handles
        all_labels = existing_labels + custom_labels
    else:
        all_handles = custom_handles
        all_labels = custom_labels
    ax.legend(handles=all_handles, labels=all_labels, loc='upper left')

    if add_txt is not None:
        ax.text(0.8, 0.8, add_txt, fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=0.5, boxstyle='round'), transform=ax.transAxes)
    return H2, V