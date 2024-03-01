"""
This script extracts CERES footprints from valid terra/aqua passes
over a region, and separates them into 1D lists of labeled features
corresponding to each independent swath.

These are stored in the swaths_pkl configured below, which is
formatted as a list of 2-tuples corresponding to F string labels of
features and a (D,F) shaped array of D data points (footprints).

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

from FG1D import FG1D

"""
List of 2-tuples like (nc_dataset_label, output_labels) assigning each
netCDF dataset label in a LARC CERES SSF file to a simpler label.

Providing a list of strings with the same length as an array saves each
array member as its own feature with the corresponding provided label.
"""
label_mapping = [
    ## (M,) time and geometry information
    ("Time_of_observation", "jday"),
    ("lat", "lat"),
    ("lon", "lon"),
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
    2-tuple like (labels:list[str], data:list[np.array]) where all of the data
    arrays are size C for C CERES footprints

    This method extracts datasets and assigns labels based on the label_mapping
    list configured above.
    """
    ds = nc.Dataset(ceres_nc, 'r')
    data = []
    labels = []
    ## Extract and rename each of the fields in themapping above
    for ncl,l in label_mapping:
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
    return labels, data

def jday_to_epoch(jday:float, ref_jday:int, ref_gday:datetime):
    """
    Given a reference jday given as an integer, coresponding to noon on the
    provided reference gregorian day, and a float offset in decimal days
    with respect to the reference datetime, returns a gregorian datetime.

    This is genuinely probably the best way to do this with the ceres ssf data.
    """
    return (ref_gday + timedelta(days=jday-ref_jday)).timestamp()

if __name__=="__main__":
    ceres_nc_dir = Path("data/ceres") ## directory of netCDFs from LARC
    swath_pkl_dir =  Path("data/swaths") ## pickle with a list of swath FGs
    ## Upper bound on sza to restrict daytime pixels
    ub_sza = 80
    ## Upper bound on viewing zenith angle (MODIS FOV is like 45)
    ub_vza = 35
    ## minimum amount of time between swaths (sec)
    lb_swath_interval = 300
    ## maximum amount of time included in a swath (sec)
    ub_swath_interval = lb_swath_interval*5
    ## Minimum number of valid footprints that warrant storing a swath
    min_footprints = 50
    ## The easiest way to convert from julian is wrt a specific reference day.
    ref_jday = 2444239.5
    ref_gday = datetime(year=1980,month=1,day=1)

    ## netCDF from https://ceres-tool.larc.nasa.gov/ord-tool/
    ceres_files = [f for f in ceres_nc_dir.iterdir()
                   if f.suffix == ".nc"]

    """
    Preprocessing:
    1. Swap julian calendar for epoch seconds
    2. add the epoch time as a feature
    3. Mask out night time footprints and those outside the VZA range (FOV)
    """
    for cf in ceres_files:
        swaths_pkl = swath_pkl_dir.joinpath(f"{cf.stem}.pkl")

        ## process only one CERES order as a FG1D object.
        ceres = FG1D(*parse_ceres(cf))
        print("footprints, features, days:",
              *ceres.data().shape,
              np.unique(np.round(ceres.data('jday'))).size
              )

        epoch = np.array([jday_to_epoch(jday, ref_jday, ref_gday)
                          for jday in ceres.data("jday")])
        ceres.drop_data("jday")
        ## add epochs as a feature
        ceres.add_data("epoch", epoch)
        ## Only consider daytime swaths for now.
        cday = ceres.mask(ceres.data("sza")<=ub_sza)
        ## limit the maximum FOV to a reasonable range
        ceres = cday.mask(cday.data("vza")<=ub_vza)
        ## Split footprints into individual swaths based on acquisition time.
        ## >5 minutes probably means it's a different swath.
        tmask = np.concatenate((np.array([True]), np.diff(
            ceres.data("epoch"))>lb_swath_interval))
        approx_times = ceres.data("epoch")[tmask]

        ## Look for anything with a stime offset < (lb_swath_interval * 3)
        all_swaths = []
        for stime in approx_times:
            smask = np.abs(ceres.data("epoch")-stime)<lb_swath_interval*3
            swath = ceres.mask(smask)
            all_swaths.append(swath)

        ## Keep all swaths with at least 50 footprints in range, and save as a
        ## list of 2-tuples like (labels:list[str], data:list[np.array])
        pkl.dump([s.to_tuple() for s in all_swaths if s.size>min_footprints],
                 swaths_pkl.open("wb"))
