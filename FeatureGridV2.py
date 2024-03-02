"""
The FeatureGrid class provides an abstraction on a set of 2d scalar arrays on
a uniform-grid, enabling the user to easily access, visualize, manipulate, and
store the scalar feature arrays along with labels and metadata.
"""

from pathlib import Path
from datetime import datetime
from datetime import timedelta
import pickle as pkl
from copy import deepcopy
import numpy as np
import json

#from krttdkit.visualize import guitools as gt
#from krttdkit.visualize import geoplot as gp
#from krttdkit.visualize import TextFormat as TF
#from krttdkit.operate import enhance as enh
from krttdkit.operate.recipe_book import transforms
#from krttdkit.operate import classify
from krttdkit.operate import Recipe

class FeatureGridV2:
    @staticmethod
    def check_serializable(json_dict:dict):
        """ Check json serializability of a dict; raise an error if invalid """
        try:
            reloaded_dict = json.loads(json.dumps(json_dict))
        except:
            raise ValueError(
                    f"Provided JSON dictionary is not JSON serializable")
        return reloaded_dict

    @staticmethod
    def from_pkl(pkl_path:Path):
        """
        Retrieves a Featuregrid object from a 6-tuple stored in a pickle file,
        with the tuple having fields ordered according to:

        (clabels, flabels, data, coords, meta, masks)
        """
        clabels, flabels, data, coords, meta, masks = \
                pkl.load(pkl_path.open("rb"))
        return FeatureGrid(clabels=clabels, flabels=flabels, data=data,
                           coords=coords, meta=meta, masks=masks)

    def __init__(self, coord_labels:list, feature_labels:list, data:np.array,
                 coords:dict={}, meta:dict={}, masks:dict={}):
        """
        Initialize a FeatureGrid minimally with fully labeled axes.

        :@param coord_labels: List of C=D-1 unique strings naming all but the
            last dimension of the D-dimensional data array.
        :@param feature_labels: List of F unique strings naming each of the
            members in the final ('feature') dimension of the data array.
        :@param data: D-dimensional data array-like object with C 'coordinate'
            dimensions described by 1D monotonic coordinate axes, and 1
            'feature' axis labeling data with a unique string. By convention,
            the feature axis is always the final one.
        :@param coords: dict mapping valid coordinate labels to 1D monotonic
            float arrays matching the size of the coordinate dimension.
        :@param meta: JSON-serializable dict containing any relevant
            information about the object for recordkeeping purposes etc.
        :@param masks: dict mapping unique string labels
        """
        ## The labels always correspond to the final "feature" axis of the
        ## input array-like object
        self._clabels = coord_labels
        self._flabels = feature_labels
        try:
            assert len(self._clabels) == len(data.shape)-1
            assert len(self._flabels) == data.shape[-1]
        except:
            raise ValueError(
                    f"For a D-dimensional data array, clabels must label "
                    "the first D-1 dimensions, and flabels must correspond to "
                    "the members of the last dimension in the data array."
                    )
        try:
            assert len(set(*self._clabels, *self._flabels)) \
                    == len(self._flabels) + len(self._clabels)
        except:
            raise ValueError(
                    f"Coordinate and feature labels must be mutually unique!"
                    )

        self._data = data
        ## coord labels default to integer index arrays
        self._coords = [np.arange(c) for c in self._data.shape[:-1]] \
                if coords is None else coords

        # Freely-accessible meta-data dictionary. 'shape' is set by default.
        self._meta = meta_info
        # The shape parameter is set dynamically at __init__. If the meta
        # dictionary contains a shape from a previous iteration, get rid of it.
        self._meta["shape"] = None
        self._recipes = {}
        self._shape = None

        for i in range(len(labels)):
            self.add_data(labels[i], data[i], info[i])

    def __repr__(self, indent=2):
        """ Print the meta-json """
        return "\n".join((
            f"shape:        {self.shape}",
            f"coord labels: {self.clabels}",
            f"feat labels:  {self.flabels}",
            f"meta labels:  {list(self.meta().keys())}",
            f"mask labels:  {list(self.masks().keys())}",
            ))

    """ Defining properties of a FeatureGrid """

    @property
    def flabels(self):
        return tuple(self._flabels)

    @property
    def clabels(self):
        return tuple(self._clabels)

    @property
    def shape(self):
        return self._shape

    ''' (!!!) unfinished transition (!!!) '''
    #def data(self, label:str=None, mask:np.ndarray=None, mask_value=0):
    def data(self, flabel:str=None, mlabel:str=None, **kwargs):
        """
        Return a FeatureGrid with the requested constraints
        """

        ##
        #for tranfunc in [transforms[s] for s in tran[::-1]]:
        #    X = tranfunc(X)
        return X

    def coords(self, clabel=None):
        """
        Return the coordinate arrays associated with the provided label(s)
        clabel may be None, a string, or a list of strings. If None, returns
        the coordinate arrays in order of the shape of the data array.
        """
        if clabel is None:
            return [self._coords[c] for c in self._clabels]
        if type(clabel) == str:
            return self._coords.get(clabel)
        ## if not a string or None, clabel must be an iterable
        return [self._coords[c] for c in tuple(clabel)]

    def meta(self, meta_key:str=None):
        """
        Return the meta-dict or its stored field if a valid label is provided.
        """
        if meta_key is None:
            return self._meta
        return self._meta.get(meta_key)

    def masks(self, mlabel:str=None):
        """
        Return the meta-dict or its stored field if a valid label is provided.
        """
        if mlabel is None:
            return self._masks
        return self._masks.get(mlabel)

    def to_dict(self):
        """
        All the information needed to recover the FeatureGrid given an
        appropriately-shaped data array (excluding boolean masks).
        """
        return {"clabels":self._clabels,
                "flabels":self._flabels,
                "coords":self.coords(),
                "meta":self.meta(),
                }
    def to_tuple(self):
        """ """
        return (
                self._clabels,
                self._flabels,
                self._data,
                self._coords,
                self._meta,
                self._masks
                )

    def to_pkl(self, pkl_path:Path, overwrite=True):
        """
        Stores this FeatureGrid object as a pickle file containing a 6-tuple
        (clabels:list[str], flabels:list[str], data:np.array,
         coords:dict{str:array}, meta:dict, masks:dict{str:tuple})

        :@param pkl_path: Location to save this FeatureGrid instance
        :@param overwrite: If True, overwrites pkl_path if it already exits
        """
        if pkl_path.exists() and not overwrite:
            raise ValueError(f"pickle already exists: {pkl_path.as_posix()}")
        pkl.dump(self.to_tuple(), pkl_path.open("wb"))

    def to_json(self, indent=None):
        """
        Returns the coordinate as a string JSON
        """
        return json.dumps(self.to_dict(), indent=indent)

## replace with drop_axis
'''
    def drop_data(self, label:str):
        """
        Drop the dataset with the provided label from this FeatureGrid.
        """
        i = self._labels.index(label)
        self._labels = list(self._labels[:i])+list(self._labels[i+1:])
        self._data = list(self._data[:i])+list(self._data[i+1:])
        return self
'''


## Should be data() but returning a FeatureGrid
'''
    def subgrid(self, labels:list=None, vrange:tuple=None, hrange:tuple=None):
        """
        Given array slices corresponding to the horizontal and vertical axes
        of this FeatureGrid, returns a new FeatureGrid with subsetted arrays
        and subsetted labels if a labels array is provided.

        :@param labels: Ordered labels of arrays included in the subgrid
        :@param vrange: Vertical range in pixel index coordinates
        :@param hrange: Horizontal range in pixel index coordinates
        """
        vslice = slice(None) if vrange is None else slice(*vrange)
        hslice = slice(None) if hrange is None else slice(*hrange)
        labels = self.labels if labels is None else labels
        fg = FeatureGrid(
                labels=labels,
                data=[self.data(l)[vslice,hslice] for l in labels],
                info=[self.info(l) for l in labels],
                meta=self.meta
                )
        fg._recipes.update(self._recipes)
        return fg
'''

'''
    def extract_values(self, pixels:list, labels:list=None):
        """
        Extracts a (P,F) array for P pixels and F features with the provided
        labels using the provided 2-tuple pixel indeces
        """
        tmp_sg = np.dstack(self.subgrid(labels)._data)
        return np.vstack([tmp_sg[p] for p in pixels])

    def _label_exists(self, label:str):
        """
        Returns True if the provided case-insensitive label matches either a
        currently-loaded scalar feature array or an added recipe. Accepts
        any object that implements __str__(), ie integer band numbers.
        """
        #label = str(label).lower()
        return label in self._labels or label in self._recipes.keys() \
                or label in transforms.keys()


    def add_recipe(self, label:str, recipe:Recipe):
        if self._label_exists(label) or label in transforms.keys():
            raise ValueError(f"Label {label} already exists.")
        assert type(recipe) is Recipe
        self._recipes[label] = recipe

    def _evaluate_recipe(self, recipe:str, mask:np.ndarray=None, mask_value=0):
        """
        Return evaluated recipe or base feature from a label

        :@param mask: if mask isn't None, applies the provided boolean mask
            with the same shape as this FeatureGrid to the base recipe before
            applying any transforms or recipes, such that any elements with a
            'True' value in the mask won't be included in the calculations.
        """
        if recipe in self.labels:
            if not mask is None:
                tmp_data = self._data[self.labels.index(recipe)]
                tmp_data[mask] = mask_value
                return tmp_data
            return self._data[self.labels.index(recipe)]
        elif recipe in self._recipes.keys():
            args = tuple(self.data(arg) for arg in self._recipes[recipe].args)
            if not mask is None:
                for a in args:
                    a[mask] = mask_value
            return self._recipes[recipe].func(*args)
        else:
            raise ValueError(f"{recipe} is not a valid recipe or label.")
'''

## Rewrite as a general concatenate method where an appropriate dimensional
## feature/coordinate arrays are provided.
'''
    def add_data(self, label:str, data:np.ndarray, info:dict=None,
                 extract_mask:bool=True):
        """
        Add a new data field to the FeatureGrid with an equally-shaped ndarray
        and a unique label. If this FeatureGrid has no data, this method will
        set the object's immutable shape attribute.

        :@param label: Unique label to identify the data array
        :@param data: 2d numpy array with identical shape to this FeatureGrid
        :@param info: Optional dictionary of attributes corresponding to this
            dataset, which can be useful for storing information for
            downstream applications.
        :@param extract_mask: if True and if the provided data is a
            MaskedArray, the mask will be

        :@return: None
        """
        label = str(label)
        if self._shape is None:
            assert len(data.shape)==2
            self._shape = data.shape
            self._meta["shape"] = self._shape
        # Make sure the data array's shape matches this grid's
        elif self._shape != data.shape:
            raise ValueError(
                    f"Cannot add {label} array with shape {data.shape}. Data"
                    f" must match this FeatureGrid's shape: {self._shape}")

        # Make sure the new label is unique and valid
        if self._label_exists(label):
            raise ValueError(f"A feature with label {label} is already added.")

        if type(data) == np.ma.core.MaskedArray:
            if extract_mask:
                # Add the mask as a new feature array
                mask = np.ma.getmask(data).astype(bool)
                if np.any(mask):
                    self.add_data(label+"_mask", mask.astype(bool),
                                  {"name":"Boolean mask for "+label},
                                  extract_mask=False)
            # get rid of the mask
            data = np.asarray(data.data)
        self._labels.append(label)
        self._data.append(data)
        self._info.append(dict(info) if info else {})
        return self

'''

''' documentation relevant to suspended methods below '''
"""
The FeatureGrid object enables the user to access and operate on the data
using the string labels and the data() object method. This method is also
able to evaluate a hierarchy of recipes.

Data recipes referencing the labels for specific instances can be loaded
with the add_recipe() object method.

'transforms' are recipes that map a single array to an array with the same
shape (ie functions of the form T:A->A). Prepending a transform label to
a data label will return the data after the transform has been applied.

For example, with FeatureGrid instance fg, which has the "truecolor"
recipe loaded along with the required recipe data, one can get a truecolor
image normalized to integers in [0,255] using the norm256 transform with:

fg.data('norm256 truecolor')

Also note: The JSON-serializability of meta-dictionaries are checked at
merge. key collisions for meta dictionaries are handled as follows:
    - If both values for the key are lists, merges them
    - If one is a list and the other is a value, adds value to the list
    - If both are values, combines them as a list.
"""

''' GUI based or 2D only methods that are suspended for now '''
'''
    def get_rgb(self, r:str, g:str, b:str):
        """
        Given 3 recipes, return an RGB after evaluating any transforms/recipe
        """
        return np.dstack(tuple(map(self.data, (r,g,b))))

    def get_pixels(self, recipe:str, labels:list=None, show=False,
                   plot_spec={}, fill_color:tuple=(0,255,255)):
        """
        Enables the user to choose a pixel or series of pixels using a recipe
        basemap. After selecting the pixels, the chosen set of values can be
        optionally visualized with a bar plot if show=True.

        :@param recipe: Base-map of recipe used for pixel selection
        :@param labels: Optional list of labels to include in the bar plot
            when show=True.

        :@return: 2-tuple including a tuple of pixel indeces and a 2d array
            of data values with shape (P,F) for P pixels and F features.
            If a list of labels is provided, only the requested data will be
            extracted, in the order of the provided labels list.
        """
        pixels = gt.get_category(self.data(recipe),fill_color=fill_color)
        labels = self._labels if labels is None else labels
        # (P,F) array of values for P pixels and F features
        values = self.extract_values(pixels, labels)
        if show:
            stdevs = list(np.std(values, axis=0))
            means = list(np.mean(values, axis=0))
            gp.basic_bars(labels,means,err=stdevs, plot_spec=plot_spec)
        return tuple(pixels), values

    def get_bound(self, label, upper=True, lower=False, bg_recipe=None):
        """
        Use a trackbar to select an inclusive lower and/or upper bound for a
        feature. The feature must be 2d.
        """
        # Make sure the base array is a valid RGB
        base_arr = label if bg_recipe is None else bg_recipe
        if len(base_arr.shape)==2:
            base_arr = gt.scal_to_rgb(base_arr)
        def pick_lbound(X,v):
            """ Callback function for rendering the user's l-bound choice """
            global base_arr
            Xnew = enh.linear_gamma_stretch(np.copy(X))
            mask = Xnew<v/255
            if base_arr is None:
                Xnew[np.where(mask)] = 0
                base_arr = enh.linear_gamma_stretch(Xnew)
            bba = np.copy(np.asarray(base_arr))
            bba[np.where(mask)] = np.array([0,0,0])
            #bba = bba[:,:,::-1]
            return bba

        def pick_ubound(X,v):
            """ Callback function for rendering the user's u-bound choice """
            global base_arr
            Xnew = enh.linear_gamma_stretch(X)
            mask = Xnew>v/255
            if base_arr is None:
                Xnew[np.where(mask)] = np.amin(Xnew)
                base_arr = enh.linear_gamma_stretch(Xnew)
            bba = np.copy(np.asarray(base_arr))
            bba[np.where(mask)] = np.array([0,0,0])
            #bba = bba[:,:,::-1]
            return bba
        X = self.data(label)
        xmin = np.amin(X)
        xrange = np.amax(X)-np.amin(X)
        X = (X-xmin)/xrange
        if upper:
            bound = gt.trackbar_select(
                    X=X,
                    func=pick_ubound,
                    label=label,
                    ) * xrange + xmin
        if lower:
            bound = gt.trackbar_select(
                    X=X,
                    func=pick_lbound,
                    label=label,
                    ) * xrange + xmin
'''

'''
## High level methods that should work but need to be updated, hopefully
## with revisions like dask task scheduling

    def heatmap(self, label1, label2, nbin1=256, nbin2=256, show=False,
                ranges:list=None, fig_path:Path=None,
                plot_spec:dict={}):
        """
        Get a heatmap of the 2 arrays' values along with axis labels in data
        coordinates using the tool from krttdkit.enhance, which extracts

        :@param label1: Label of array for heatmap vertical axis. You can
            provide an array as well, just make sure it is the same size as
            the other label or array.
        :@param label2: Label of array for heatmap horizontal axis. As
            mentioned above, you can provide an array as well as long as the
            size is uniform.
        :@param nbin1: Number of brightness bins to use for the first dataset
        :@param nbin2: Number of brightness bins to use for the second dataset
        :@param ranges:
        :@param ranges: If ranges is defined, it must be a list of 2-tuple
            float value ranges like (min, max). This sets the boundaries for
            discretization in coordinate units, and thus sets the min/max
            values of the returned array, along with the mins if provided.
            Defaults to data range.
        :@param mins: If mins is defined, it must be a list of numbers for the
            minimum recognized value in the discretization. This sets the
            boundaries for discretization in coordinate units, and thus
            determines the min/max values of the returned array, along with
            any ranges. Defaults to data minimum
        :@param fig_path: Path to save the generated figure automatically
        :@param plot_spec: geoplot plot_spec dictionary with configuration
            options for geoplot.plot_heatmap when show=True
        """
        # Allow array arguments instead of labels.
        A1 = label1 if type(label1) == np.ndarray else self.data(label1)
        A2 = label2 if type(label2) == np.ndarray else self.data(label2)
        label1 = "ax1" if type(label1) == np.ndarray else label1
        label2 = "ax2" if type(label2) == np.ndarray else label2
        # Use the enhance tool to get a (nbin1, nbin2) integer array of counts
        # and axis coordinates in data values.
        M, coords = enh.get_nd_hist(
                arrays=(A1, A2),
                bin_counts=(nbin1, nbin2),
                ranges=ranges,
                )
        vcoords, hcoords = tuple(coords)
        if show or fig_path:
            # default plot_spec can be overwritten by parameter dict values
            def_ps = {
                    "ylabel":label1,
                    "xlabel":label2,
                    "cb_label":"counts",
                    "title":f"Brightness heatmap ({label2} vs {label1})",
                    "cmap":"gist_ncar",
                    "imshow_norm":"log",
                    "imshow_extent":(min(hcoords),max(hcoords),
                                     min(vcoords),max(vcoords)),
                    #"imshow_aspect":1,
                    }
            def_ps.update(plot_spec)
            gp.plot_heatmap(heatmap=M, plot_spec=def_ps, show=show,
                            fig_path=fig_path)
        return M, vcoords, hcoords

    def do_mlc(self, select_recipe:str, categories:list, labels:list=None,
               threshold:float=None):
        """
        If this raises linear algebra errors, try selecting more samples.

        :@param select_recipe: String label of the feature or recipe available
            from this FeatureGrid to use to pick category pixel candidates
        :@param categories: List of unique string categories for each class of
            pixels you want to train mlc to recognize
        :@param labels: List of valid string labels for input arrays to include
            in maximum-likelihood classification.
        :@param threshold: Float confidence level in [0,1] below which pixels
            will be placed into a new 'uncertain' category.

        :@return: 2-tuple like (classified_ints, labels)
        """
        samples = {}
        for cat in categories:
            print(TF.BLUE("Select for category: ", bright=True)+
                  TF.WHITE(cat, bright=True, bold=True))
            samples[cat], _ = self.get_pixels(select_recipe, labels)
        class_ints, class_keys = classify.mlc(
                np.dstack(self.subgrid(labels=labels).data()),
                categories=samples,
                thresh=threshold
                )
        return class_ints, labels, samples

    def do_mdc(self, select_recipe:str, categories:list, labels:list=None):
        """
        :@param select_recipe: String label of the feature or recipe available
            from this FeatureGrid to use to pick category pixel candidates
        :@param categories: List of unique string categories for each class of
            pixels you want to train mlc to recognize
        :@param labels: List of valid string labels for input arrays to include
            in maximum-likelihood classification.
        """
        samples = {}
        for cat in categories:
            print(TF.BLUE("Select for category: ", bright=True)+
                  TF.WHITE(cat, bright=True, bold=True))
            samples[cat], _ = self.get_pixels(select_recipe, labels)
        classified, labels = classify.minimum_distance(
                np.dstack(self.subgrid(labels=labels).data()),
                categories=samples)
        return classified, labels, samples

    def get_nd_hist(self, labels:list, nbin=256):
        """
        Basic wrapper on the krttdkit.operate.enhace module tool for getting
        a sparse histogram in multiple dimensions. See the documentation for
        that method for details.

        :@param labels: Labels of each arrays to generate a histogram axis of.
        :@param nbin: Number of brightness bins in each array. This is
            a fixed number (256) by default, but you may provide a list of
            integer brightness bin counts corresponding to each label.

        :@return: 2-tuple like (H, coords) such that H and coords are arrays.

            H is a N-dimensional integer array of counts for each of the N
            provided arrays. Each dimension represents a different input
            array's histogram, and indeces of a dimension mark brightness
            values for that array.

            You can sum along axes in order to derive subsequent histograms.

            The 'coords' array is a length N list of numpy arrays. These arrays
            associate the corresponding dimension in H with actual brightness
            values in data coordinates. They may have different sizes since
            different bin_counts can be specified for each dimension.
        """
        return enh.get_nd_hist(
                arrays=tuple(self.data(l) for l in labels),
                bin_counts=nbin)


'''
