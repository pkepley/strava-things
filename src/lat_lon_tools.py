import pandas as pd
import numpy as np
from scipy.integrate import fixed_quad

def haversine_dist(lons_a, lats_a, lons_b, lats_b):
    """
    Parameters
    -------
    lons_a: numpy array representing longitudes in degrees for points a
    lats_a: numpy array representing latitudes in degrees for points a

    lons_b: numpy array representing longitudes in degrees for points b
    lats_b: numpy array representing latitudes in degrees for points b

    Returns
    -------
    mi:   approximate distance between (lon, lat) pairs for points in a
          measured against points in b in miles <3
    """
    # convert lon/lat 'a' pairs into radians
    lons_a = (np.pi / 180) * lons_a
    lats_a = (np.pi / 180) * lats_a

    # convert lon/lat 'b' pairs into radians
    lons_b = (np.pi / 180) * lons_b
    lats_b = (np.pi / 180) * lats_b

    # this is the haversine distance calculation, broken into three steps
    # the final step uses an estimate of the Earth's radius in km
    a = np.sin((lats_b - lats_a)/2)**2 + \
        np.cos(lats_b) * np.cos(lats_a) * np.sin((lons_b - lons_a)/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c

    # ... and now we convert to miles <3
    mi = 0.621371 * km

    return mi


def consecutive_haversine_distances(lons, lats):
    """
    Parameters
    -------
    lons: numpy array representing longitudes in degrees
    lats: numpy array representing latitudes in degrees

    Returns
    -------
    mi:   approximate distance between consecutive (lon, lat) pairs in
          miles <3
    """

    lons_a, lats_a = lons[:-1], lats[:-1]
    lons_b, lats_b = lons[1:], lats[1:]

    return haversine_dist(lons_a, lats_a, lons_b, lats_b)


def consecutive_haversine_distances_with_elevation(lons, lats, eles):
    """
    Parameters
    -------
    lons: numpy array representing longitudes in degrees
    lats: numpy array representing latitudes in degrees
    eles: numpy array representing elevations in meters

    Returns
    -------
    dists: approximate distance between consecutive (lon, lat, ele)
           triples in miles. obviously, it should be in miles <3
    """
    # consecutive changes + convert meters -> miles
    vert_dists = 0.000621371 * np.abs(eles[1:] - eles[:-1])

    # consecutive 'lateral' distances
    lat_dists = consecutive_haversine_distances(lons, lats)

    # consecutive 'total' distances
    dists = np.sqrt(lat_dists**2 + vert_dists**2)

    return dists


def estimate_closed_path_total_angle(x, y, pm=None):
    # we expect for the path given by (x,y) to be closed, but will
    # distinct start/end pinots as long as they are 'nearby' one another
    # average the 'observed' start and end points to get a 'better'
    # start/end point
    xstart = 0.5 * (x[0] + x[-1])
    ystart = 0.5 * (y[0] + y[-1])

    # force the 'averaged' start/end points to be the curve's new
    # start and end points
    xx = np.concatenate([np.array([xstart]), x, np.array([xstart])])
    yy = np.concatenate([np.array([ystart]), y, np.array([ystart])])

    # if a 'central' point was not provided, pick a candidate interior
    # point. note: there is NO guarantee this point will be inside the
    # curve, as there is no restriction imposed on the curve.
    if pm is None:
        xm = x.min() + 0.5 * (x.mean() - x.min())
        ym = y.mean()
    else:
        xm, ym = pm

    # translate the route so that (xm,ym) is at the origin
    # this will simplify our calcluation (i.e. we use 'z' = 0)
    xx = xx - xm
    yy = yy - ym

    # estimate the signed angle that the route advances around (xm,ym)
    # as it winds about (xm,ym)
    theta_tot = 0
    for i in range(xx.shape[0]-1):
        def f_tmp(t):
            xt = np.interp(t, [0.0, 1.0], xx[i:(i+2)])
            dx = xx[i+1] - xx[i]
            yt = np.interp(t, [0.0, 1.0], yy[i:(i+2)])
            dy = yy[i+1] - yy[i]

            return (xt*dy - yt*dx) / (xt**2 + yt**2)

        dtheta_i, _ = fixed_quad(f_tmp, a=0.0, b=1.0, n=6)
        theta_tot += dtheta_i

    return theta_tot


def estimate_pi(x, y, pm=None):
    theta_tot = estimate_closed_path_total_angle(x,y,pm)

    # i.e. a crude estimate for pi
    if abs(theta_tot) > 3.0:
        n_loops = int(theta_tot / (2 * 3.1))
        pi_est  = theta_tot / (2 * n_loops)
    else:
        n_loops = 0
        pi_est  = np.nan

    return pi_est, n_loops, theta_tot
