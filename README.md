# aes690final

<p align="center">
    <img height="512" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/docs/banner.png?raw=true" />
</p>

<p align="center">
      <img height="512" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/docs/data_pipeline_overview.png?raw=true" />
</p>

## get\_ceres\_swath.py

Ingest a directory of CERES SSF E4A netCDF files acquired from
[the NASA LARC downloader][1], and store them as a pickle file
containing a list of 2-tuples each corresponding to a distinct
satellite overpass parsed from the multi-day file.

Since lat/lon boundaries are determined by the user when they
download a bulk order from the LARC tool, the user should provide a
unique region string identifying the domain.

Each 2-tuple like (labels:list[str], data:list[np.array]) contains a
list of labels cooresponding to each length C data array, where C is
the number of valid footprints acquired in that overpass.

The list of 2-tuples stored in the swaths pixel can be used to
initialize a list of `FG1D` objects like this:

```python
ceres_swaths = [FG1D(*s) for s in pkl.load(swaths_pkl.open("rb"))]
```

These "ceres swath files" are identified by the underscore-separated
fields in their path:

`ceres-ssf_{region}_{sat}_{firstday:YYYYmmdd}-{lastday:YYYYmmdd}.h5`

## get\_modis\_swath.py

Given a series of CERES swath pickle files generated by
`get_ceres_swath.py`, dispatch parallel processes to:

1. acquire the corresponding MODIS data from the [LAADS DAAC][3]
2. extract and subset the data to the bounds of the CERES swath
3. generate a custom hdf5 with datasets and meta-info providing
   enough information to initialize a `FeatureGridV2` and `FG1D`
   object using the dict stored as a json in the file group attrs.

Each subsequent swath hdf5 file thus contains both MODIS and CERES
data from a single unique overpass of a single region by a single
satellite (Terra or Aqua).

These "swath files" are identified by the underscore-separated fields
in their path:

`swath_{region}_{sat}_{timestr:YYYYmmdd-HHMM}.h5`

Downloading the MODIS data requires the user to generate an api key
at the [MODAPS portal][4], and to provide that key as a file path
to a text file, or as a string at runtime.

## generate\_samples.py

Module containing data generators that load data from swath hdf5
files created by `get_modis_swath`, extract MODIS subgrids around
each CERES footprint, and calculate the CERES PSF over the MODIS
subgrids, then return the data as specified in the data pipeline
overview.

The `swaths_dataset` method concurrently opens one or more swath
hdf5s from the provided lists and yeilds the data as a tensorflow
dataset.

Except for the first "tile\_h5\_path" argument, the `get_tiles_h5`
method recieves the same arguments as `swaths_dataset`. Instead
of yielding data as a generated dataset, it loads the training-ready
data into a series of datasets for each feature/label field:
"modis", "geom", "psf", "ceres". This custom file format is referred
to as a _tiles hdf5_.

Training data can subsequently be more efficiently yielded from
multiple interleaved tiles hdf5s with `tiles_dataset`.


## plot\_swath.py

Script for plotting CERES and MODIS data from swath hdf5 files
generated by `get_modis_swath.py`.

<p align="center">
    <img height="256" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/docs/neus_20200530-1808_aqua_flux-lw.png?raw=true" />
    <img height="256" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/docs/neus_20200530-1808_aqua_flux-sw.png?raw=true" />
</p>

<p align="center">
      <img height="256" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/docs/neus_20200530-1808_aqua_rgb-tc.png?raw=true" />
      <img height="256" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/docs/neus_20200530-1808_aqua_rgb-dcp.png?raw=true" />
      <img height="256" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/docs/neus_20200530-1808_aqua_rgb-dust.png?raw=true" />
</p>

<p align="center">CERES swath over the Northeast USA (shortwave and longwave flux),</p>
<p align="center"> and co-located MODIS truecolor, day cloud phase, and dust RGBs. </p>

Visualizations currently rely on [krttdkit][2], but if you prefer,
you can get rid of the dependancy and use matplotlib imshow to plot
the RGBs instead.


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

## montage.sh

<p align="center">
      <img height="512" src="https://github.com/Mitchell-D/aes690final/blob/main/figures/modis/modis-quilt_5.png?raw=true" />
</p>
<p align="center">
      MODIS tile mosaic, including Hindu Kush Himilayan region, North East US, Amazon Rainforest, and Indonesia.
</p>

[1]:https://ceres-tool.larc.nasa.gov/ord-tool
[2]:https://github.com/Mitchell-D/krttdkit
[3]:https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM/
[4]:https://ladsweb.modaps.eosdis.nasa.gov/profiles/#generate-token-modal
