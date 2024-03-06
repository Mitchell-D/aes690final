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

## Constants from CERES ATBD 2.2 subsection 4.4 Table 1.
r_E = 6367 ## Radius of Earth (km)
h = 705 ## Altitude EOS (km)
alpha_h = 64.2 ## Cone angle at the horizon (degrees)
gamma_h = 25.8 ## Earth central angle at horizon (degrees)
period = 98.7 ## Period (minutes)

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
    ("Colatitude_of_subsatellite_point_at_surface_at_observation","nadir_lat"),
    ("Longitude_of_subsatellite_point_at_surface_at_observation","nadir_lon"),
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
    2-tuple like (labels:list[str], data:np.array) where the unique string
    feature labels name the second dimension of the (C,F) shaped data array
    having C CERES footprints evaluated with F features.

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
    return labels, np.stack(data, axis=-1), {"satellite":ds.platform.lower()}

def jday_to_epoch(jday:float, ref_jday:int, ref_gday:datetime):
    """
    Given a reference jday given as an integer, coresponding to noon on the
    provided reference gregorian day, and a float offset in decimal days
    with respect to the reference datetime, returns a gregorian datetime.

    This is genuinely probably the best way to do this with the ceres ssf data.
    """
    return (ref_gday + timedelta(days=jday-ref_jday)).timestamp()

def get_view_angles(lat, lon, nadir_lat, nadir_lon):
    """
    Calculates satellite to centroid viewing angle vectors according to
    CERES ATBD 2.2 subsection 4.4, and adds the resulting geodetic satellite
    and centriod vectors, as well as a satellite-relative coordinate reference
    frame with respect to each centroid location.

    https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf

    See figure 4.4-2 in the ATBD for more details.

    Data required to be already in the provided FG1D (all in degrees)
    ("lat", "lon", "nadir_lat", "nadir_lon")

    Keys added to the provided FG1D:
    "x_sat", "y_sat", "z_sat", ## normed satellite geodetic vector
    "x_cen", "y_cen", "z_cen", ## normed centroid geodetic vector
    "x_s2c", "y_s2c", "z_s2c", ## normed satellite to centroid reference frame
    """

    ## (N, 3) vector of centroid geodetic vectors with features (x,y,z)
    v_cen = np.stack([
        np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
        np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
        np.cos(np.deg2rad(lat)),
        ], axis=-1)
    ## (N, 3) vector of satellite geodetic vectors with features (x,y,z)
    v_sat = np.stack([
        np.sin(np.deg2rad(nadir_lat)) * np.cos(np.deg2rad(nadir_lon)),
        np.sin(np.deg2rad(nadir_lat)) * np.sin(np.deg2rad(nadir_lon)),
        np.cos(np.deg2rad(nadir_lat)),
        ], axis=-1)

    ## Slant path length to centroid
    rho = np.sqrt((r_E+h)**2 + r_E**2 - 2*r_E*(r_E+h)*np.sum(v_cen*v_sat))
    ## satellite to centroid
    v_Y = (r_E*v_cen - (r_E+h)*v_sat)/rho
    ## right of scan direction
    v_X = np.cross(v_Y, v_sat, axisa=-1, axisb=-1)
    v_X /= np.stack(
            [np.linalg.norm(v_X, axis=-1) for i in range(3)],
            axis=-1)
    ## opposite of scan direction
    v_Z = np.cross(v_X, v_Y, axisa=-1, axisb=-1)
    ## (N,9) array for the scan-relative coordinates
    vv_s2c = np.concatenate((v_X, v_Y, v_Z), axis=-1)

    return v_cen, v_sat, vv_s2c


if __name__=="__main__":
    data_dir = Path("data")
    #data_dir = Path("/rstor/mdodson/aes690final/ceres")
    ceres_nc_dir = data_dir.joinpath("ceres") ## directory of netCDFs from LARC
    ## Directory containing pickles corresponding to lists of swath FGs
    swath_pkl_dir =  data_dir.joinpath("ceres_swaths")
    ## Upper bound on sza to restrict daytime pixels
    ub_sza = 75
    ## Upper bound on viewing zenith angle (MODIS FOV is like 45)
    ub_vza = 30
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
    ceres_files = [f for f in ceres_nc_dir.iterdir() if f.suffix == ".nc"]

    ## Don't inclue the more sparse surface types (only the top 4)
    drop_fields = (
            "id_s5", "id_s6", "id_s7", "id_s8",
            "pct_s5", "pct_s6", "pct_s7", "pct_s8",
            )
    reject_if_nan = ("swflux", "wnflux", "lwflux")

    """
    Preprocessing steps below:

    1. Delete unneeded fields listed in drop_fields.
    2. Swap julian calendar for epoch seconds.
    3. Mask out night time footprints and those outside the VZA range (FOV).
    4. Drop footprints with invalid critical fields listed in reject_if_nan.
    5. Calculate viewing geometry and add the new info to the dataset.
    6. Split all footprints into overpasses based on their acquisition time.
    """
    for cf in ceres_files:
        swaths_pkl = swath_pkl_dir.joinpath(f"{cf.stem}.pkl")

        ## process only one CERES order as a FG1D object.
        ceres = FG1D(*parse_ceres(cf))

        ## Remove unneeded fields
        for df in drop_fields:
            ceres.drop_data(df)

        ceres.meta.update({
            "ub_sza":ub_sza,
            "ub_vza":ub_vza,
            "lat_range":(np.amin(ceres.data("lat")),
                         np.amax(ceres.data("lat"))),
            "lon_range":(np.amin(ceres.data("lon")),
                         np.amax(ceres.data("lon"))),
            })
        print("footprints, features, days:",
              *ceres.data().shape,
              np.unique(np.round(ceres.data('jday'))).size
              )

        ## Convert julian days to epochs and replace them as a feature
        epoch = np.array([
            jday_to_epoch(jday, ref_jday, ref_gday)
            for jday in ceres.data("jday")
            ])
        ceres.add_data("epoch", epoch)
        ceres.drop_data("jday")

        ## Only consider daytime swaths for now.
        ceres = ceres.mask(ceres.data("sza")<=ub_sza)
        ## Limit the FoV to prevent problems with panoramic distortion
        ceres = ceres.mask(ceres.data("vza")<=ub_vza)


        ## NaN values are marked in the SSF files with values >1e35
        is_valid = lambda X: X < 1e30

        ## Drop all features that must be valid (radiative fluxes)
        ## Cloud, aerosol and surface type features may still have nan (>1e35)
        ## values, which should be dealt with by the user.
        ## In the future, FeatureGrid should support generating boolean masks
        ## for those features that are stored by default alongside the array.
        for l in reject_if_nan:
            ceres = ceres.mask(is_valid(ceres.data(l)))

        ## Check the ratio of masked/unmasked values for each feature
        #'''
        for l in ceres.labels:
            valid_counts = np.sum((is_valid(ceres.data(l))).astype(int))
            print(l, valid_counts/ceres.size)
        #'''

        v_sat, v_cen, vv_s2c = get_view_angles(
                lat=ceres.data("lat"),
                lon=ceres.data("lon"),
                nadir_lat=ceres.data("nadir_lat"),
                nadir_lon=ceres.data("nadir_lon"),
                )

        ## Geodetic satellite position
        ceres.add_data("x_sat", v_sat[:,0])
        ceres.add_data("y_sat", v_sat[:,1])
        ceres.add_data("z_sat", v_sat[:,2])

        ## Geodetic centroid position
        ceres.add_data("x_cen", v_cen[:,0])
        ceres.add_data("y_cen", v_cen[:,1])
        ceres.add_data("z_cen", v_cen[:,2])

        ## New satellite to centroid coordinate reference frame
        ceres.add_data("xx_s2c", vv_s2c[:,0])
        ceres.add_data("xy_s2c", vv_s2c[:,1])
        ceres.add_data("xz_s2c", vv_s2c[:,2])

        ceres.add_data("yx_s2c", vv_s2c[:,3])
        ceres.add_data("yy_s2c", vv_s2c[:,4])
        ceres.add_data("yz_s2c", vv_s2c[:,5])

        ceres.add_data("zx_s2c", vv_s2c[:,6])
        ceres.add_data("zy_s2c", vv_s2c[:,7])
        ceres.add_data("zz_s2c", vv_s2c[:,8])

        ## Split footprints into individual swaths based on acquisition time.
        ## >5 minutes almost certainly means it's a different swath. This was
        ## shown to be an effective heuristic for the several test regions,
        ## but the assumption could break down for swaths over many latitudes
        tmask = np.concatenate((np.array([True]), np.diff(
            ceres.data("epoch"))>lb_swath_interval))
        approx_times = ceres.data("epoch")[tmask]

        ## Look for anything with a stime offset < (lb_swath_interval * 3)
        swaths = []
        for stime in approx_times:
            smask = np.abs(ceres.data("epoch")-stime)<lb_swath_interval*3
            swath = ceres.mask(smask)
            swaths.append(swath)

        swaths = list(filter(lambda s:s.size>min_footprints, swaths))

        ### (!!!) Only take every 2nd swath (!!!)
        swaths = swaths[::2]
        swaths = [s.to_tuple() for s in swaths if s.size>min_footprints]

        ## Keep all swaths with at least 50 footprints in range, and save as a
        ## list of 2-tuples like (labels:list[str], data:list[np.array])
        pkl.dump(swaths, swaths_pkl.open("wb"))
