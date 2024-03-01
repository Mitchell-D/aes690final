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

#from krttdkit.acquire import modis
from krttdkit.operate import enhance as enh
#from krttdkit.products import FeatureGrid
#from krttdkit.products import HyperGrid
#from krttdkit.visualize import guitools as gt
#from krttdkit.visualize import geoplot as gp

class FG1D:
    @staticmethod
    def trend(X, Y):
        """
        Returns the linear regression slope, intercept, and Pearson coefficient
        of the 2 provided dataset labels.
        """
        res = linregress(X, Y)
        slope,intc,rval = (res.slope, res.intercept, res.rvalue)
        return (slope, intc, rval)

    def __init__(self, labels, data:np.array):
        assert len(data.shape) == 2
        assert len(labels) == len(set(labels)) == data.shape[-1]
        self._data = data
        self.labels = labels
        self.size = self._data.shape[0]

    def to_tuple(self):
        return (self.labels, self._data)

    def data(self, label=None):
        if label is None:
            return np.copy(self._data)
        return self._data[...,self.labels.index(label)]

    def add_data(self, label, data):
        """
        Concatenate a 1D feature dataset with a new unique label to this FG1D
        """
        assert data.size == self.size
        assert len(data.shape) == 1
        assert label not in self.labels
        self.labels.append(label)
        data = np.expand_dims(data, axis=-1)
        self._data = np.concatenate((self._data, data), axis=-1)

    def drop_data(self, label):
        assert label in self.labels
        idx = self.labels.index(label)
        self.labels.pop(idx)
        self._data = np.delete(self._data, idx, axis=-1)

    def fill(self, label, value):
        """
        Convenience method to replace masked values with a number for a
        single dataset.
        """
        self._data[self.labels.index(label)] = \
                self.data(label).filled(value)

    def mask(self, mask:np.ndarray):
        assert mask.size == self.size
        return FG1D(self.labels, self._data[mask])

    #def subset(self, labels:list):
    #    return FG1D(labels, [self.data(l) for l in labels])
