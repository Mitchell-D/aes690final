import numpy as np
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from krttdkit.operate import enhance as enh
from FG1D import FG1D

def mutual_valid(X, Y):
    """
    Returns corresponding non-NaN and unmasked values from 2 masked arrays
    """
    valid = np.isfinite(X.filled(np.nan)) & np.isfinite(Y.filled(np.nan))
    return X[valid], Y[valid]

def scatter(ceres_fg1d, xlabel, ylabel, clabel=None, get_trend=False, show=True,
            fig_path:Path=None, plot_spec:dict={}, tres=500):
    """
    Make a scatter plot of the 2 provided datasets stored in this FG1D,
    optionally coloring points by a third dataset
    """
    ps = {"xlabel":xlabel, "ylabel":ylabel, "clabel":clabel,
          "trend_color":"red", "trend_width":3, "marker_size":4,
          "cmap":"nipy_spectral", "text_size":12, "title":"",
          "norm":"linear", "logx":False,"figsize":(16,12)}
    ps.update(plot_spec)

    plt.clf()
    plt.rcParams.update({"font.size":ps["text_size"]})

    X, Y = mutual_valid(
            ceres_fg1d.data(xlabel),
            ceres_fg1d.data(ylabel)
            )
    C = None if clabel is None else ceres_fg1d.data(clabel)

    fig, ax = plt.subplots()
    if get_trend:
        slope,intc,rval = ceres_fg1d.trend(X, Y)
        #Tx = np.copy(X)
        Tx = np.linspace(np.amin(X), np.amax(X), tres)
        Ty = Tx*slope + intc
        #ax.scatter(
        ax.plot(
                Tx, Ty,
                linewidth=ps.get("trend_width"),
                label=f"y={slope:.3f}x+{intc:.3f}\n$R^2$ = {rval**2:.3f}",
                color=ps.get("trend_color"),
                #s=ps.get("marker_size")+30,
                #s=ps.get("marker_size")+30,
                #marker="2",
                zorder=100,
                )
        ax.legend()
    if ps["logx"]:
        plt.semilogx()
    scat = ax.scatter(
            X, Y, c=C, s=ps.get("marker_size"), cmap=ps.get("cmap"),
            norm=ps.get("norm"))
    if not clabel is None:
        fig.colorbar(scat, label=ps.get("clabel"))
    ax.set_title(ps.get("title"))
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))
    if show:
        plt.show()
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)

def heatmap(ceres_fg1d, xlabel, ylabel, xbins=256, ybins=256, get_trend=False,
            show=True, fig_path:Path=None, plot_spec:dict={}):
    """
    Generate a heatmap of the 2 provided datasets in this FG1D
    """
    ps = {"xlabel":xlabel, "ylabel":ylabel, "trend_color":"red",
          "trend_width":3, "cmap":"gist_ncar", "text_size":12,
          "figsize":(12,12)}
    ps.update(plot_spec)
    X, Y = mutual_valid(
            ceres_fg1d.data(xlabel),
            ceres_fg1d.data(ylabel)
            )
    M, coords = enh.get_nd_hist(
            arrays=(X, Y),
            bin_counts=(xbins, ybins),
            )
    hcoords, vcoords = tuple(coords)
    extent = (min(hcoords), max(hcoords), min(vcoords), max(vcoords))

    plt.rcParams.update({"font.size":ps["text_size"]})
    fig, ax = plt.subplots()
    im = ax.pcolormesh(hcoords, vcoords, M,
            cmap=plot_spec.get("cmap"),
            #vmax=plot_spec.get("vmax"),
            #extent=extent,
            #norm=plot_spec.get("imshow_norm"),
            #origin="lower",
            #aspect=plot_spec.get("imshow_aspect")
            )
    if get_trend:
        slope,intc,rval = ceres_fg1d.trend(X, Y)
        Tx = np.copy(hcoords)
        Ty = Tx * slope + intc
        ax.plot(Tx, Ty,
                linewidth=ps.get("trend_width"),
                label=f"y={slope:.3f}x+{intc:.3f}\nR^2 = {rval**2:.3f}",
                color=ps.get("trend_color"),
                )
        ax.legend()
    #ax.set(aspect=1)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", label="Count")
    ax.set_title(ps.get("title"))
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))
    #ax.set_xticklabels([f"{c:.2f}" for c in hcoords])
    #ax.set_yticklabels([f"{c:.2f}" for c in vcoords])
    #ax.set_ylim(extent[0], extent[1])
    #ax.set_xlim(extent[2], extent[3])
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()

def geo_scatter(ceres_fg1d, clabel, xlabel="lat", ylabel="lon", bounds=None,
                plot_spec={}, show=True, fig_path=None):
    """ """
    ps = {"xlabel":xlabel, "ylabel":ylabel, "marker_size":4,
          "cmap":"nipy_spectral", "text_size":12, "title":clabel,
          "norm":None,"figsize":(12,12)}
    plt.clf()
    ps.update(plot_spec)

    plt.rcParams.update({"font.size":ps["text_size"]})

    ax = plt.axes(projection=ccrs.PlateCarree())
    fig = plt.gcf()
    if bounds is None:
        bounds = [
            np.amin(ceres_fg1d.data(ylabel)),
            np.amax(ceres_fg1d.data(ylabel)),
            np.amin(ceres_fg1d.data(xlabel)),
            np.amax(ceres_fg1d.data(xlabel)),
            ]
    ax.set_extent(bounds)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    ax.coastlines()

    ax.set_title(ps.get("title"))
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))

    scat = ax.scatter(ceres_fg1d.data(ylabel),ceres_fg1d.data(xlabel),
                      c=ceres_fg1d.data(clabel), s=ps.get("marker_size"),
                      transform=ccrs.PlateCarree(), zorder=100,
                      cmap=ps.get("cmap"), norm=ps.get("norm"))
    fig.colorbar(scat)

    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()

if __name__=="__main__":
    swaths_pkl = Path("data/ceres_swaths/ceres-ssf_hkh_aqua_20180101-20201231.pkl")
    fig_dir = Path("figures")
    ceres_swaths =  [FG1D(*s) for s in pkl.load(swaths_pkl.open("rb"))]
    geo_scatter(
            ceres_fg1d=ceres_swaths.pop(0),
            clabel="swflux",
            show=False,
            fig_path=fig_dir.joinpath(f"ceres-ssf_swflux_sample.png"),
            plot_spec={"title":"swflux"}
            )
