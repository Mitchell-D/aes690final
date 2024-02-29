#import netCDF4 as nc
import gc
import numpy as np
from pathlib import Path
import pickle as pkl
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

from scipy.stats import linregress
#from sklearn.linear_model import LinearRegression

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from krttdkit.acquire import modis
from krttdkit.operate import enhance as enh
from krttdkit.products import FeatureGrid
from krttdkit.products import HyperGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp

class FG1D:
    def __init__(self, labels, data):
        self.size = data[0].size
        assert all(X.size==self.size for X in data)
        assert len(labels) == len(set(labels)) == len(data)
        self._data = []
        self.labels = []
        for l,d in zip(labels,data):
            self.add_data(l,d)

    def to_tuple(self):
        return (self.labels, self._data)

    def data(self, label=None):
        if label is None:
            return np.copy(self._data).T
        return self._data[self.labels.index(label)]

    def add_data(self, label, data):
        assert data.size == self.size
        assert label not in self.labels
        self.labels.append(label)
        self._data.append(data)

    def drop_data(self, label):
        assert label in self.labels
        idx = self.labels.index(label)
        self.labels.pop(idx)
        self._data.pop(idx)

    def fill(self, label, value):
        """
        Convenience method to replace masked values with a number for a
        single dataset.
        """
        self._data[self.labels.index(label)] = \
                self.data(label).filled(value)

    def mask(self, mask:np.ndarray):
        return FG1D(self.labels, [X[mask] for X in self._data])

    def subset(self, labels:list):
        return FG1D(labels, [self.data(l) for l in labels])

    def scatter(self, xlabel, ylabel, clabel=None, get_trend=False, show=True,
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

        X, Y = self.mutual_valid(self.data(xlabel), self.data(ylabel))
        C = None if clabel is None else self.data(clabel)

        fig, ax = plt.subplots()
        if get_trend:
            slope,intc,rval = self.trend(X, Y)
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

    def heatmap(self, xlabel, ylabel, xbins=256, ybins=256, get_trend=False,
                show=True, fig_path:Path=None, plot_spec:dict={}):
        """
        Generate a heatmap of the 2 provided datasets in this FG1D
        """
        ps = {"xlabel":xlabel, "ylabel":ylabel, "trend_color":"red",
              "trend_width":3, "cmap":"gist_ncar", "text_size":12,
              "figsize":(12,12)}
        ps.update(plot_spec)
        X, Y = self.mutual_valid(self.data(xlabel), self.data(ylabel))
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
            slope,intc,rval = self.trend(X, Y)
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

    def geo_scatter(self, clabel, xlabel="lat", ylabel="lon", bounds=None,
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
                np.amin(self.data(ylabel)),
                np.amax(self.data(ylabel)),
                np.amin(self.data(xlabel)),
                np.amax(self.data(xlabel)),
                ]
        ax.set_extent(bounds)

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.RIVERS)
        ax.coastlines()

        ax.set_title(ps.get("title"))
        ax.set_xlabel(ps.get("xlabel"))
        ax.set_ylabel(ps.get("ylabel"))

        scat = ax.scatter(self.data(ylabel),self.data(xlabel),
                          c=self.data(clabel), s=ps.get("marker_size"),
                          transform=ccrs.PlateCarree(), zorder=100,
                          cmap=ps.get("cmap"), norm=ps.get("norm"))
        fig.colorbar(scat)

        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
        if show:
            plt.show()

    @staticmethod
    def trend(X, Y):
        """
        Returns the linear regression slope, intercept, and Pearson coefficient
        of the 2 provided dataset labels.
        """
        #'''
        res = linregress(X, Y)
        slope,intc,rval = (res.slope, res.intercept, res.rvalue)
        '''
        res = LinearRegression().fit(X[:,None], Y)
        slope,intc,rval = res.coef_[0],res.intercept_,res.score(X[:,None],Y)
        '''
        return (slope, intc, rval)

    @staticmethod
    def mutual_valid(X, Y):
        valid = np.isfinite(X.filled(np.nan)) & np.isfinite(Y.filled(np.nan))
        #print(enh.array_stat(X[valid]))
        #print(enh.array_stat(Y[valid]))
        return X[valid], Y[valid]

class SfcType:
    def __init__(self, name, ids):
        self.name = name
        self.ids = ids
    @property
    def fstr(self):
        return self.name.lower().replace(" ","-")
    def mask(self, C):
        I = np.copy(C.data("id_s1"))
        mask = np.full_like(I, False)
        for v in self.ids:
            mask = np.logical_or(mask, (I == v))
        return np.copy(mask)
