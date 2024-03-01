# aes690final

Take 2 on disaggregating CERES footprints to MODIS resolution.

## get\_ceres\_swath.py

Ingest a directory of CERES SSF netCDF files acquired from
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


[1]:https://ceres-tool.larc.nasa.gov/ord-tool
