"""
This script extracts CERES footprints from valid terra/aqua passes
over a region, and separates them into 1D lists of labeled features
corresponding to each independent swath.

These are stored in the swaths_pkl configured below, which is
formatted as a list of 2-tuples corresponding to F string labels of
features and a (N,F) shaped array of N data points (footprints).

The list entries are separated into overpasses with varying numbers
of valid footprints. Passes with too few valid footprints are
thresholded by min_footprints

swaths = [ FG1D(labels,data) for labels,data in swaths ]
"""
import netCDF4 as nc
import numpy as np
import pickle as pkl
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from pprint import pprint as ppt

from FG1D import FG1D
#from geom_utils import get_view_vectors,get_equatorial_vectors
#from geom_utils import get_sensor_pixel_geometry

"""
List of 2-tuples like (nc_dataset_label, output_labels) assigning each
netCDF dataset label in a LARC CERES SSF file to a simpler label.

Providing a list of strings with the same length as an array saves each
array member as its own feature with the corresponding provided label.
"""
ceres_label_mapping = [
    ## (M,) time and geometry information
    ("Time_of_observation", "jday"),
    ("lat", "lat"),
    ("lon", "lon"),
    ("Colatitude_of_subsatellite_point_at_surface_at_observation","ndr_colat"),
    ("Longitude_of_subsatellite_point_at_surface_at_observation","ndr_lon"),
    ("CERES_viewing_zenith_at_surface", "vza"),
    ("CERES_relative_azimuth_at_surface", "raa"),
    ("CERES_solar_zenith_at_surface", "sza"),


    ## (M,8) Most prominent surface types, in decreasing order
    ("Surface_type_index",
     ("id_s1","id_s2","id_s3","id_s4",
      "id_s5","id_s6","id_s7","id_s8")),
    ("Surface_type_percent_coverage",
     ("pct_s1","pct_s2","pct_s3","pct_s4",
      "pct_s5","pct_s6","pct_s7","pct_s8")),
    ("Clear_layer_overlap_percent_coverages",
     ("pct_clr","pct_l1","pct_l2","pct_ol")),

    ## (M,) ADM-corrected fluxes
    ("CERES_SW_TOA_flux___upwards", "swflux"),
    ("CERES_WN_TOA_flux___upwards", "wnflux"),
    ("CERES_LW_TOA_flux___upwards", "lwflux"),

    ("Cloud_mask_clear_strong_percent_coverage", "nocld"),
    ("Cloud_mask_clear_weak_percent_coverage", "nocld_wk"),

    ## (M,2) COD for each layer weighted by PSF and cloud fraction
    ("Mean_visible_optical_depth_for_cloud_layer",
     ("l1_cod","l2_cod")),
    ("Stddev_of_visible_optical_depth_for_cloud_layer",
     ("l1_sdcod","l2_sdcod")),

    ## (M,) PSF weighted percentage of pixels in the footprint which
    ## have either land or ocean aerosol values
    ("Percentage_of_CERES_FOV_with_MODIS_land_aerosol", "aer_land_pct"),
    ## (M,) PSF weighted cloud frac from MOD04:
    ## Cloud fraction from Land aerosol cloud mask from retrieved
    ## and overcast pixels not including cirrus mask
    ("PSF_wtd_MOD04_cloud_fraction_land", "aer_land_cfrac"),
    ## (M,) Weighted integer percentage bins of aerosol types
    ("PSF_wtd_MOD04_aerosol_types_land", "aer_land_type"),
    ("PSF_wtd_MOD04_corrected_optical_depth_land__0_550_", "aod_land"),

    ## (M,) Optical depth with the deep blue method (?)
    ("Percentage_of_CERES_FOV_with_MODIS_deep_blue_aerosol", "aer_db_pct"),
    ("PSF_wtd_MOD04_deep_blue_aerosol_optical_depth_land__0_550_", "aod_db"),

    ## (M,) Over-ocean aerosol properties
    ("Percentage_of_CERES_FOV_with_MODIS_ocean_aerosol", "aer_ocean_pct"),
    ("PSF_wtd_MOD04_cloud_fraction_ocean", "aer_ocean_cfrac"),
    ("PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_", "aod_ocean"),
    ("PSF_wtd_MOD04_optical_depth_small_average_ocean__0_550_", "aod_ocean_small"),
    ]

def parse_ceres(ceres_nc:Path):
    """
    Parses fields from a full-featured CERES SSF file, and returns it as a
    3-tuple like (labels:list[str], data:np.array, meta) where the unique
    string feature labels name the second dimension of the (C,F) shaped data
    array having C CERES footprints evaluated with F features. The meta dict
    is part of the FeatureGrid convention, and contains only a string
    marking the swath's satellite platform of origin.

    This method extracts datasets and assigns labels based on the
    ceres_label_mapping
    list configured above.
    """
    ds = nc.Dataset(ceres_nc, 'r')
    data = []
    labels = []
    ## Extract and rename each of the fields in themapping above
    for ncl,l in ceres_label_mapping:
        X = ds.variables[ncl][:]
        if not type(l) is str:
            assert len(l) == X.shape[1]
            for i in range(len(l)):
                data.append(X[:,i])
                labels.append(l[i])
        else:
            assert len(X.shape)==1
            data.append(X)
            labels.append(l)
    return labels, np.stack(data, axis=-1), {"satellite":ds.platform.lower()}

def jday_to_epoch(jday:float):
    """
    Given a reference jday given as an integer, coresponding to noon on the
    provided reference gregorian day, and a float offset in decimal days
    with respect to the reference datetime, returns a gregorian datetime.

    This is genuinely probably the best way to do this with the ceres ssf data.
    """
    ref_jday = 2444239.5
    ref_gday = datetime(year=1980,month=1,day=1)
    return (ref_gday + timedelta(days=jday-ref_jday)).timestamp()

def get_ceres_swaths(
        ceres_nc_file:Path, drop_fields:list=[], reject_if_nan:list=[],
        ub_sza=180., ub_vza=90., lb_swath_interval=300, ub_swath_interval=1500,
        debug=False):
    """
    This method extracts data from a CERES SSF netCDF file acquired from:

    The LARC Downloader: https://ceres-tool.larc.nasa.gov/ord-tool/

    returns a list of FG1D objects corresponding to individual overpasses.

    This method's responsibilities:

    1. Delete unneeded fields listed in drop_fields.
    2. Swap julian calendar for epoch seconds.
    3. Mask out night time footprints and those outside the VZA range (FOV).
    4. Drop footprints with invalid critical fields listed in reject_if_nan.
    5. Calculate viewing geometry and add the new info to the dataset.
    6. Split all footprints into overpasses based on their acquisition time.

    :@param ceres_nc_file:LARC CERES SSF file to parse.
    :@param drop_fields: By default, this method extracts all of the fields
        in ceres_label_mapping, but if the 'new' keys of valid fields are
        provided here, they won't be included in the returned footprints.
    :@param reject_if_nan: Fields that must be a valid number, or else
        footprints with NaN values are dropped.
    :@param ub_sza: Upper bound on sza to restrict daytime pixels
    :@param ub_vza: Upper bound on viewing zenith angle (MODIS FOV is like 45)
    :@param lb_swath_interval: Minimum amount of time between swaths (sec)
    :@param ub_swath_interval: Maximum amount of time included in a swath (sec)

    """
    ceres = FG1D(*parse_ceres(ceres_nc_file))

    ## Remove unneeded fields
    for df in drop_fields:
        ceres.drop_data(df)

    ## Add swath information and configured constraints to the meta dict
    ceres.meta.update({
        "ub_sza":ub_sza,
        "ub_vza":ub_vza,
        "lat_range":(np.amin(ceres.data("lat")),
                     np.amax(ceres.data("lat"))),
        "lon_range":(np.amin(ceres.data("lon")),
                     np.amax(ceres.data("lon"))),
        })

    if debug:
        print(f"\nParsing file: {ceres_nc_file}")
        print("footprints, features, days:",
              *ceres.data().shape,
              np.unique(np.round(ceres.data('jday'))).size
              )

    ## Convert julian days to epochs and replace them as a feature
    ## Some CERES footprints occasionally have 0 julian time
    ceres = ceres.mask(ceres.data("jday")>=1)
    epoch = np.array([
        jday_to_epoch(jday)
        for jday in ceres.data("jday")
        ])
    ceres.add_data("epoch", epoch)
    ceres.drop_data("jday")

    ## Only consider daytime swaths for now.
    ceres = ceres.mask(ceres.data("sza")<=ub_sza)
    ## Limit the FoV to prevent problems with panoramic distortion
    ceres = ceres.mask(ceres.data("vza")<=ub_vza)

    ## NaN values are marked in the SSF files with values >1e35
    is_valid = lambda X: np.logical_and(X<1e30, np.logical_not(np.isnan(X)))

    """
    Add the following 15 quantities as features:
    (N,3) satellite's equatorial coordinate position at each observation
    (N,3) centroids' equatorial coordinate positions
    (N,9) sat-to-centroid relative vectors in the equatorial reference frame
    """
    ## Now calculating point spread function only in generator
    '''
    geom_labels,geom_data = get_sensor_pixel_geometry(
            nadir_lat=90-ceres.data("ndr_colat"), ## convert to lat
            nadir_lon=ceres.data("ndr_lon"), ## [0,360) from GM
            obsv_lat=ceres.data("lat"),
            obsv_lon=ceres.data("lon"),
            sensor_altitude=705., ## EOS altitude per ATBD subsection 4.4
            earth_radius=6367., ## Earth radius also per the ATBD
            return_labels=True
            )
    for i,l in enumerate(geom_labels):
        ceres.add_data(l, geom_data[:,i])
    '''

    """
    Drop all features that must be valid (radiative fluxes)
    Cloud, aerosol and surface type features may still have nan (>1e35)
    values, which should be dealt with by the user. Also, geometry values
    must be valid (sometimes geolocation fails for some reason.
    """
    ## In the future, FeatureGrid should support generating boolean masks
    ## for those features that are stored by default alongside the array.
    for l in reject_if_nan:
        ceres = ceres.mask(is_valid(ceres.data(l)))

    ## Split footprints into individual swaths based on acquisition time.
    ## >5 minutes almost certainly means it's a different swath. This was
    ## shown to be an effective heuristic for the several test regions,
    ## but the assumption could break down for swaths over many latitudes
    tmask = np.concatenate((np.array([True]), np.diff(
        ceres.data("epoch"))>lb_swath_interval))
    approx_times = ceres.data("epoch")[tmask]

    ## Check the ratio of masked/unmasked values for each feature
    #'''
    if debug:
        print(f"Percent valid footprints per field:")
        for l in ceres.labels:
            valid_counts = np.sum((is_valid(ceres.data(l))).astype(int))
            print(f"{valid_counts/ceres.size:3.4f} {l}")

    ## Look for anything with a stime offset < (lb_swath_interval * 3)
    swaths = []
    for stime in approx_times:
        smask = np.abs(ceres.data("epoch")-stime)<lb_swath_interval*3
        swath = ceres.mask(smask)
        swaths.append(swath)
    return swaths


if __name__=="__main__":
    debug = True
    data_dir = Path("data")
    #data_dir = Path("/rstor/mdodson/aes690final/ceres")

    ## directory of netCDFs from https://ceres-tool.larc.nasa.gov/ord-tool/
    ceres_nc_dir = data_dir.joinpath("ceres")
    ## directory to dump pickles corresponding to lists of swath FGs
    swath_pkl_dir =  data_dir.joinpath("ceres_swaths_val")

    ## (!!!) Region label used to identify files to parse (!!!)
    region_labels = (
            "azn",
            "neus",
            "idn",
            "hkh",
            "seus",
            "alk",
            )

    ## Minimum number of valid footprints that warrant storing a swath
    min_footprints = 50
    drop_fields = ( ## Currently ignoring less prominent components
        "id_s5", "id_s6", "id_s7", "id_s8",
        "pct_s5", "pct_s6", "pct_s7", "pct_s8",
        )
    ## Fields that must be valid for a footprint to be maintained
    reject_if_nan = (
            "swflux", "wnflux", "lwflux",
            #"x_sat", "y_sat", "z_sat",
            #"x_cen", "y_cen", "z_cen",
            #"xx_s2c", "xy_s2c", "xz_s2c",
            #"yx_s2c", "yy_s2c", "yz_s2c",
            #"zx_s2c", "zy_s2c", "zz_s2c",
            )

    ub_sza = 75. ## upper bound for solar zenith of valid footprints
    ub_vza = 30. ## upper bound for solar zenith of valid footprints

    ## Parse the CERES files into lists of FG1D objects, each corresponding
    ## to a single satellite overpass' CERES footprints within the region.
    region_files = sorted([
            f for f in ceres_nc_dir.iterdir()
            if f.suffix == ".nc" and any(l in f.stem for l in region_labels)
            ])
    for ceres_file in region_files:
        rlabel = next(l for l in region_labels if l in ceres_file.stem)
        swaths_pkl = swath_pkl_dir.joinpath(f"{ceres_file.stem}.pkl")
        swaths = get_ceres_swaths(
                ceres_nc_file=ceres_file,
                drop_fields=drop_fields,
                reject_if_nan=reject_if_nan,
                ub_sza=ub_sza,
                ub_vza=ub_vza,
                debug=debug,
                )
        swaths = list(filter(lambda s:s.size>min_footprints, swaths))

        """
        Optionally subset the swaths by a mod value. This will sparsely
        sub-sample the data so that the full range of seasons is represented,
        but fewer total swaths are extracted.

        Later on, consider sampling (mod 3)+1 or (mod 3)+2 indexed datasets
        """
        ### (!!!) Only take every 3rd swath (!!!)
        swaths = swaths[2::3]
        swaths_subset_label = "_2mod3"
        swaths_pkl = swaths_pkl.parent.joinpath(
                swaths_pkl.stem + swaths_subset_label + ".pkl")

        ## Add the region key to all the meta dicts
        for s in swaths:
            s.meta.update(region=rlabel)

        if debug:
            swath_stats = np.array([[[
                np.nanmean(s.data(l)), np.std(s.data(l)), np.ptp(s.data(l))
                ] for l in s.labels] for s in swaths])
            swath_stats = np.average(swath_stats, axis=0)
            for i,l in enumerate(s.labels):
                print(f"{l}:"," ".join([
                    f"mean:{swath_stats[...,i,0]:.3f}",
                    f"stdev:{swath_stats[...,i,1]:.3f}",
                    f"range:{swath_stats[...,i,2]:.3f}",
                    ]))
            print(f"Data source:      {ceres_file.name}")
            print(f"Swaths acquired:  {len(swaths)}")
            print(f"Avg # footprints: {np.mean([s.size for s in swaths]):.1f}")

        ## Keep all swaths with at least 50 footprints in range, and save as a
        ## list of 2-tuples like (labels:list[str], data:list[np.array])
        pkl.dump([s.to_tuple() for s in swaths], swaths_pkl.open("wb"))
