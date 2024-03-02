# aes690final

Take 2 on disaggregating CERES footprints to MODIS resolution.

<p align="center">
  <img height="512" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/ceres/flux-bispec_2020-06-22_0819Z.png" />
  <img height="512" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/ceres/flux-contour_2019-12-04_0825Z.png" />
</p>

<p align="center">
  <img height="512" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/ceres/geo_scatter_2020-06-22_0819Z_lwflux.png" />
  <img height="512" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/ceres/geo_scatter_2020-06-22_0819Z_swflux.png" />
</p>

<p align="center"> CERES swath over the Hindu-Kush Himilayan region </p>

## get\_ceres\_swath.py

Ingest a directory of CERES SSF E4A netCDF files acquired from
[the NASA LARC downloader][1], and store them as a pickle file
containing a list of 2-tuples each corresponding to a distinct
satellite overpass parsed from the multi-day file.

Each 2-tuple like (labels:list[str], data:list[np.array]) contains a
list of labels cooresponding to each length C data array, where C is
the number of valid footprints acquired in that overpass.

The list of 2-tuples stored in the swaths pixel can be used to
initialize a list of `FG1D` objects like this:

```python
ceres_swaths = [FG1D(*s) for s in pkl.load(swaths_pkl.open("rb"))]
```

## plot\_ceres.py

`plot_ceres.py` containes a few methods for visualizing swaths of
CERES data that have already been processed by `get_ceres_swath.py`.

 - `interp_1d_to_geo_grid` and `dual_contour_plot` provide ways to
   regrid and plot 2 CERES features alongside each other.
   `contour_plot` facilitates plotting of just 1 regridded feature.
 - `geo_scatter` uses cartopy to scatter the individual data points
   for a feature on a simple basemap.
 - `heatmap` plots the magnitudes of two feature variables against
   each other in a user-specified number of value bins.
 - `scatter` plots two data values' magnitudes against each other

The `__main__` context in `plot_ceres.py` applies these plotting
methods to random (seeded) samples from a pkl file generated by
`get_ceres_swath.py`.

[1]:https://ceres-tool.larc.nasa.gov/ord-tool
