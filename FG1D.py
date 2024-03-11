import json
import numpy as np
from scipy.stats import linregress

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

    def __init__(self, labels, data:np.array, meta:dict={}):
        assert len(data.shape) == 2
        assert len(labels) == len(set(labels)) == data.shape[-1]
        self._data = data
        self.labels = labels
        self.size = self._data.shape[0]
        self.meta = meta

    @property
    def shape(self):
        return (self.size, len(self.labels))

    def __repr__(self):
        return f"FG1D\nlabels: {self.labels}\nshape: {self.shape}"

    def to_tuple(self):
        return (self.labels, self._data, self.meta)

    def to_json(self):
        """
        Convert the serializable labels and meta dict of this FG1D to a
        JSON-formatted string. This can be used to re-initialize it alongside
        a separately-loaded data array
        """
        return json.dumps({"labels":self.labels, "meta":self.meta})

    def data(self, label=None):
        if label is None:
            return np.copy(self._data)
        if hasattr(label, "__iter__") and not type(label)==str:
            idx = tuple(self.labels.index(s) for s in label)
        else:
            idx = self.labels.index(label)
        return self._data[...,idx]

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
        """
        Return a FG1D object only including samples with True mask values.
        """
        assert mask.size == self.size
        return FG1D(self.labels, self._data[mask], meta=self.meta)

    #def subset(self, labels:list):
    #    return FG1D(labels, [self.data(l) for l in labels])
