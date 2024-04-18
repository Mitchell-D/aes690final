import numpy as np

def get_equatorial_vectors(latitude, longitude):
    """
    Given identically-shaped arrays of latitude and longitude,
    calculate the equatorial coordinate vectors over the arrays
    such that coordinate vectors are relative to the basis vectors:

    X := (0,  0) Intersection of Greenwich Meridian and equator (0,0)
    Y := (0, 90) 90 degrees positive longitude from X (0,90)
    Z := (90, *) North

    :@param latitude: N-D array of geodetic latitude values in [-90,90]
    :@param latitude: N-D array of geodetic longitude values in [-180,180]

    :@return: (...,3) array containing <x,y,z> unit vectors corresponding to
        the equatorial coordinates at each point in the input domain.
    """
    return np.stack([
        np.sin(np.deg2rad(90.-latitude)) * np.cos(np.deg2rad(longitude)),
        np.sin(np.deg2rad(90.-latitude)) * np.sin(np.deg2rad(longitude)),
        np.cos(np.deg2rad(90.-latitude)),
        ], axis=-1)

def get_view_vectors(sensor_equatorial_vectors, pixel_equatorial_vectors,
                     sensor_altitude=705., earth_radius=6367.,
                     return_labels=False):
    """
    Use uniformly-shaped (..., 3) sensor and (...,3) pixel equatorial
    coordinate vector arrays corresponding to each observed point to calculate
    satellite-relative viewing vectors defined as follows:

    Y':= geodetic 3-vector pointing directly towards the centroid
    X':= geodetic 3-vector perpendicular (right of) scan direction
    Z':= geodetic 3-vector parallel (opposite of) scan direction

    View vectors are important for calculating the point spread function of
    co-located sensors like CERES and MODIS. This method follows the procedure
    of the CERES ATBD 2.2 subsection 4.4

    https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf

    See figure 4.4-2 in the ATBD for more details.

    :@param sensor_equatorial_vectors: (...,3) shaped (X,Y,Z)
    :@param pixel_equatorial_vectors:
    """
    ## general labels for the relative equatorial components.
    view_vector_labels = [
        "xx_s2c", "xy_s2c", "xz_s2c",
        "yx_s2c", "yy_s2c", "yz_s2c",
        "zx_s2c", "zy_s2c", "zz_s2c"
        ]

    ## Constants from CERES ATBD 2.2 subsection 4.4 Table 1.

    ## abbreviate argument names
    r_E = earth_radius ## Radius of Earth (km)
    h = sensor_altitude ## Altitude EOS (km)
    v_sat = sensor_equatorial_vectors
    v_cen = pixel_equatorial_vectors

    ## Slant path length to centroid
    rho = np.sqrt((r_E+h)**2 + r_E**2 - \
            2*r_E*(r_E+h)*np.sum(v_cen*v_sat, axis=-1))[...,np.newaxis]
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

    if return_labels:
        return view_vector_labels,vv_s2c
    return vv_s2c

def get_sensor_pixel_geometry(
        nadir_lat, nadir_lon, obsv_lat, obsv_lon, sensor_altitude=705.,
        earth_radius=6367., return_labels=False):
    """
    Given the sub-satellite latitude and longitude as well as the centroid
    latitude and longitude of observations, calculate the sensor and
    centroid equatorial position vectors as well as the relative vector
    in the equatorial coordinate system from the sensor to the
    corresponding centroid.
    """
    ## (N, 3) vector of satellite geodetic vectors with features (x,y,z)
    sat_labels = ["x_sat", "y_sat", "z_sat"]
    v_sat = get_equatorial_vectors(latitude=nadir_lat, longitude=nadir_lon)

    ## (N, 3) vector of centroid geodetic vectors with features (x,y,z)
    cen_labels = ["x_cen", "y_cen", "z_cen"]
    v_cen = get_equatorial_vectors(latitude=obsv_lat, longitude=obsv_lon)

    ## (N, 9) vector of satellite-relative viewing vectors
    V = get_view_vectors(
            sensor_equatorial_vectors=v_sat,
            pixel_equatorial_vectors=v_cen,
            return_labels=return_labels,
            earth_radius=earth_radius,
            sensor_altitude=sensor_altitude,
            )
    if not return_labels:
        return np.concatenate((v_sat,v_cen,V), axis=-1)
    view_vector_labels,vv_sat2cen = V
    X = np.concatenate((v_sat, v_cen, vv_sat2cen), axis=-1)
    return sat_labels + cen_labels + view_vector_labels, X


'''
def get_view_angles(lat, lon, nadir_lat, nadir_lon):
    """
    Calculates satellite to centroid viewing angle vectors according to
    CERES ATBD 2.2 subsection 4.4, and adds the resulting geodetic satellite
    and centriod vectors, as well as a satellite-relative coordinate reference
    frame with respect to each centroid location.

    Keys added to the provided FG1D:
    "x_sat", "y_sat", "z_sat", ## normed satellite geodetic vector
    "x_cen", "y_cen", "z_cen", ## normed centroid geodetic vector
    "x_s2c", "y_s2c", "z_s2c", ## normed satellite to centroid reference frame
    """

    ## (N, 3) vector of centroid geodetic vectors with features (x,y,z)
    v_cen = get_equatorial_vectors(lat, lon)
    ## (N, 3) vector of satellite geodetic vectors with features (x,y,z)
    v_sat = get_equatorial_vectors(nadir_lat, nadir_lon)

    return v_cen, v_sat, vv_s2c
'''
