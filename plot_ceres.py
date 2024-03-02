import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from scipy.interpolate import griddata
from datetime import datetime

from krttdkit.operate import enhance as enh
from FG1D import FG1D

def mutual_valid(X, Y):
    """
    Returns corresponding non-NaN and unmasked values from 2 masked arrays
    """
    valid = np.isfinite(X.filled(np.nan)) & np.isfinite(Y.filled(np.nan))
    return X[valid], Y[valid]

def scatter(ceres_fg1d:FG1D, xlabel, ylabel, clabel=None, get_trend=False,
            show=True, fig_path:Path=None, plot_spec:dict={}, tres=500):
    """
    Make a scatter plot of the 2 provided datasets stored in this FG1D,
    optionally coloring points by a third dataset
    """
    ps = {"xlabel":xlabel, "ylabel":ylabel, "clabel":clabel,
          "trend_color":"red", "trend_width":3, "marker_size":6,
          "cmap":"nipy_spectral", "text_size":18, "title":"",
          "norm":"linear", "logx":False,"figsize":(12,12)}
    ps.update(plot_spec)

    #plt.clf()
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

def heatmap(ceres_fg1d:FG1D, xlabel, ylabel, xbins=256, ybins=256,
            get_trend=False, show=True, fig_path:Path=None, plot_spec:dict={}):
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

def geo_scatter(ceres_fg1d:FG1D, clabel, xlabel="lat", ylabel="lon",
                bounds=None, plot_spec={}, show=False, fig_path=None):
    """ """
    ps = {"xlabel":xlabel, "ylabel":ylabel, "marker_size":4,
          "cmap":"nipy_spectral", "text_size":12, "title":clabel,
          "norm":None,"figsize":(12,12), "marker":"o", "cbar_shrink":1.}
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
                      cmap=ps.get("cmap"), norm=ps.get("norm"),
                      marker=ps.get("marker"))
    fig.colorbar(scat, ax=ax, shrink=ps.get("cbar_shrink"))

    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()

def dual_contour_plot(
        hcoords, vcoords, data_left, data_right, bins=64,
        title_left:str="", title_right:str="",
        show=False, fig_path:Path=None, plot_spec={}
        ):
    ps = {"xlabel":"longitude", "ylabel":"latitude", "marker_size":4,
          "cmap":"nipy_spectral", "text_size":16, "title":"",
          "norm":"linear","fig_size":(12,12), "dpi":80, "cbar_loc":"right"}
    ps.update(plot_spec)

    fig,(ax1,ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(*ps.get("fig_size"))
    ax1.set_title(title_left)
    ax1.set_xlabel(ps.get("xlabel"))
    ax1.set_ylabel(ps.get("ylabel"))
    cf1 = ax1.contourf(hcoords, vcoords, data_left, bins, cmap=ps.get("cmap"))
    fig.colorbar(cf1, ax=ax1, location=ps.get("cbar_loc"))

    ax2.set_title(title_right)
    ax2.set_xlabel(ps.get("xlabel"))
    ax2.set_ylabel(ps.get("ylabel"))
    cf2 = ax2.contourf(hcoords, vcoords, data_right, bins, cmap=ps.get("cmap"))
    fig.colorbar(cf2, ax=ax2, location=ps.get("cbar_loc"))

    #plt.rcParams.update({"font.size":ps["text_size"]})
    fig.tight_layout()

    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), dpi=ps.get("dpi"))

def contour_plot(
        hcoords, vcoords, data, bins=64,
        show=False, fig_path:Path=None, plot_spec={}
        ):
    ps = {"xlabel":"latitude", "ylabel":"longitude", "marker_size":4,
          "cmap":"nipy_spectral", "text_size":8, "title":"",
          "norm":"linear","figsize":(8,8)}
    ps.update(plot_spec)

    fig,ax = plt.subplots()
    ax.set_title(ps.get("title"))
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))

    cont = ax.contourf(hcoords, vcoords, data, bins, cmap=ps.get("cmap"))
    fig.colorbar(cont)
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)

def interp_1d_to_geo_grid(
        data, lat, lon, lat_res=.1, lon_res=.1,
        interp_method:str="nearest"
        ):

    """
    Interpolate a 1D array of data with corresponding latitudes and longitudes
    onto a regular-interval geodetic grid between its minimum and maximum
    latitude and longitude.
    """
    glat = np.arange(np.amin(lat), np.amax(lat), lat_res)
    glon = np.arange(np.amin(lon), np.amax(lon), lon_res)
    return griddata((lat, lon), data, (glat[:,None], glon[None,:]),
                    method=interp_method), glat, glon

if __name__=="__main__":
    swaths_pkl = Path(
            "data/ceres_swaths/ceres-ssf_hk_aqua_20180101-20201231.pkl")
            #"data/ceres_swaths/ceres-ssf_hk_terra_20180101-20201231.pkl")
    fig_dir = Path("figures/ceres")
    ceres_swaths =  [FG1D(*s) for s in pkl.load(swaths_pkl.open("rb"))]

    #print(fg.labels)
    #print(enh.array_stat(fg.data()))


    ## Interpolate the CERES footprints of random swaths onto a geographic grid
    seed = 2023
    #seed = None
    num_samples = 4
    lat_res = .05
    lon_res = .05

    rng = np.random.default_rng(seed=seed)
    random_swath_idxs = np.arange(len(ceres_swaths))
    rng.shuffle(random_swath_idxs)

    for i in range(num_samples):
        fg = ceres_swaths[random_swath_idxs[i]]
        ## Convert the acquisition time to a string
        interp_features = ("swflux", "lwflux")
        timestr = datetime.fromtimestamp(
                int(np.average(fg.data("epoch")))
                ).strftime("%Y-%m-%d_%H%MZ")

        ## Extract shortwave and longwave feature data and interpolate it
        '''
        geo_grid,lat,lon = zip(*[
                interp_1d_to_geo_grid(
                    data=fg.data(l),
                    lat=fg.data("lat"),
                    lon=fg.data("lon"),
                    lat_res=lat_res,
                    lon_res=lon_res,
                    interp_method="linear",
                    )
                for l in interp_features
                ])
        ## geo_grid now a (lat,lon,F) shaped array for F interpolated features
        geo_grid = np.stack(geo_grid, axis=-1)
        ## since we are only iterating over features, lat/lon is invariant
        glon, glat = np.meshgrid(lon[0], lat[0])
        ## Contour plotk
        dual_contour_plot(
                hcoords=glon,
                vcoords=glat,
                bins=128,
                data_left=geo_grid[...,0],
                data_right=geo_grid[...,1],
                title_left=str(interp_features[0])+" "+timestr,
                title_right=str(interp_features[1])+" "+timestr,
                #show=True,
                fig_path=fig_dir.joinpath(f"flux-contour_{timestr}.png"),
                plot_spec={
                    "text_size":12,
                    },
                )
        '''

        ## Make a scatterplot of the features with a linear best-fit trend.
        '''
        scatter(ceres_fg1d=fg,
                xlabel=interp_features[0],
                ylabel=interp_features[1],
                clabel="vza",
                show=False,
                fig_path=fig_dir.joinpath(f"flux-bispec_{timestr}.png"),
                get_trend=True,
                plot_spec={
                    "title":timestr,
                    "text_size":18,
                    }
                )
        '''

        ## Geolocated scatterplot of actual data values (with basemap)
        #'''
        geo_scatter(
                ceres_fg1d=fg,
                clabel="swflux",
                show=True,
                #fig_path=fig_dir.joinpath(f"geo_scatter_{timestr}_swflux.png"),
                plot_spec={
                    "title":f"swflux {timestr}",
                    "marker_size":100,
                    "marker":",",
                    "text_size":16,
                    "cbar_shrink":.6,
                    }
                )
        geo_scatter(
                ceres_fg1d=fg,
                clabel="lwflux",
                show=True,
                #fig_path=fig_dir.joinpath(f"geo_scatter_{timestr}_lwflux.png"),
                plot_spec={
                    "title":f"lwflux {timestr}",
                    "marker_size":100,
                    "marker":",",
                    "text_size":16,
                    "cbar_shrink":.6,
                    },
                )
        #'''

        '''
        heatmap(fg,
                "swflux",
                "lwflux",
                xbins=8,
                ybins=8,
                show=True,
                fig_path=None,
                plot_spec={}
                )
        '''
