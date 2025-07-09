#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: Line_Of_Sight
File   : transform_coord.py

Author: Pessel Arnaud
Date: 2025-06
Version: 1.0
GitHub: https://github.com/dunaar/Line_Of_Sight
License: MIT

Description:
    This module provides functions to convert between geographic coordinates (longitude, latitude, altitude) and
    Cartesian coordinates (x, y, z).
    It includes functions for both direct conversion and inverse conversion, as well as functions to calculate
    great circle distances and straight-line distances between points on a sphere, adjusted for altitude. 
    It uses NumPy for vectorized operations and optionally Numba for multi-core acceleration.

Dependencies:
    - NumPy: For handling array and scalar data types.
    - Numba (optional): For JIT compilation and performance optimization.

Usage:
    This module can be imported to provide functions for coordinate transformations and distance calculations.
    The `__main__` block can be run to test the functionality with generated data.
    >>> from transform_coord import geo_to_cart, cart_to_geo, great_circle_distances, straight_line_distances
    >>> x, y, z = geo_to_cart(lon, lat, alt)
    >>> lon, lat, alt = cart_to_geo(x, y, z)
    >>> d_gc = great_circle_distances(lon1, lat1, alt1, lon2, lat2, alt2)
    >>> d_sl = straight_line_distances(lon1, lat1, alt1, lon2, lat2, alt2)
"""

__version__ = "1.0"

# %% Imports
# -----------------------------------------------------------------------------
import os
import time
import numpy as np

NUM_CORES = os.cpu_count()  # Détecter le nombre de cœurs
#print('NUM_CORES', NUM_CORES)

use_numba = (NUM_CORES > 1) & True
if use_numba:
    try:
        from numba import jit, prange
        use_numba = True
    except ImportError:
        use_numba = False
# -----------------------------------------------------------------------------

# %% Constantes
# -----------------------------------------------------------------------------
R_EARTH = np.float32(6371000.0)
# -----------------------------------------------------------------------------

# %% Conversion directe : WGS84 → Cartésiennes
# -----------------------------------------------------------------------------
def geo_to_cart_numpy(lon, lat, alt, R=R_EARTH):
    """
    Convertit des coordonnées géographiques (lon, lat, alt) en coordonnées cartésiennes (x, y, z).
    Hypothèse: terre sphérique
    Entrées : lon, lat (en degrés), alt (en mètres), tableaux NumPy float32
    Sortie : x, y, z (en mètres), tableaux NumPy float32
    """
    lon_rad = np.radians(lon, dtype=np.float32)
    lat_rad = np.radians(lat, dtype=np.float32)
    r_alt   = np.float32(R + alt)  # Précalculer R + alt
    
    sin_lat         = np.sin(lat_rad, dtype=np.float32)
    cos_lat_cos_lon = np.cos(lat_rad, dtype=np.float32) * np.cos(lon_rad, dtype=np.float32)
    cos_lat_sin_lon = np.cos(lat_rad, dtype=np.float32) * np.sin(lon_rad, dtype=np.float32)
    
    x = r_alt * cos_lat_cos_lon
    y = r_alt * cos_lat_sin_lon
    z = r_alt * sin_lat

    return x, y, z
# -----------------------------------------------------------------------------

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Conversion inverse : Cartésiennes → WGS84
def cart_to_geo_numpy(x, y, z, R=R_EARTH):
    """
    Convertit des coordonnées cartésiennes (x, y, z) en coordonnées géographiques (lon, lat, alt).
    Hypothèse: terre sphérique
    Entrées : x, y, z (en mètres), tableaux NumPy float32
    Sortie : lon, lat (en degrés), alt (en mètres), tableaux NumPy float32
    """
    r_alt = np.sqrt(x**2 + y**2 + z**2, dtype=np.float32)

    lon = np.degrees(np.arctan2(y, x, dtype=np.float32), dtype=np.float32)
    lat = np.degrees(np.arcsin(z / r_alt, dtype=np.float32), dtype=np.float32)
    alt = np.float32(r_alt - R)
    
    return lon, lat, alt
# -----------------------------------------------------------------------------


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% geo_to_cart_numba
if use_numba:
    import numba
    
    @jit(nopython=True, parallel=True, cache=True)
    def geo_to_cart_numba(lons, lats, alts, R=R_EARTH):
        R: np.float32 = np.float32(R)
        
        is_vect = hasattr(lons, 'size')
        
        if not is_vect:
            lons = np.array([lons], dtype=np.float32)
            lats = np.array([lats], dtype=np.float32)
            alts = np.array([alts], dtype=np.float32)
        
        n: np.uint32  = np.uint32(lons.size)

        x = np.empty(n, dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        z = np.empty(n, dtype=np.float32)
        
        for i in prange(n):
            lon_rad: np.float32 = np.float32(np.radians(lons[i]))
            lat_rad: np.float32 = np.float32(np.radians(lats[i]))
            r_alt   = np.float32(R + alts[i])  # Précalculer R + alt

            sin_lat         = np.sin(lat_rad)
            cos_lat_cos_lon = np.cos(lat_rad) * np.cos(lon_rad)
            cos_lat_sin_lon = np.cos(lat_rad) * np.sin(lon_rad)

            x[i] = r_alt * cos_lat_cos_lon
            y[i] = r_alt * cos_lat_sin_lon
            z[i] = r_alt * sin_lat
        
        if is_vect:
            return x, y, z
        else:
            return x[0], y[0], z[0]
    
    @jit(nopython=True, parallel=True)
    def cart_to_geo_numba(xs, ys, zs, R=R_EARTH):
        n = np.uint32(xs.size)

        lons = np.empty(n, dtype=np.float32)
        lats = np.empty(n, dtype=np.float32)
        alts = np.empty(n, dtype=np.float32)

        for i in prange(n):
            r_alt = np.float32(np.sqrt(xs[i]**2 + ys[i]**2 + zs[i]**2))

            lons[i] = np.degrees(np.arctan2(ys[i], xs[i]))
            lats[i] = np.degrees(np.arcsin(zs[i] / r_alt))
            alts[i] = r_alt - R

        return lons, lats, alts
    
    geo_to_cart = geo_to_cart_numba
    cart_to_geo = cart_to_geo_numba
else:
    geo_to_cart = geo_to_cart_numpy
    cart_to_geo = cart_to_geo_numpy

def great_circle_distances(lons1: np.ndarray, lats1: np.ndarray, alts1: np.ndarray,
                           lons2: np.ndarray, lats2: np.ndarray, alts2: np.ndarray,
                           R: float = R_EARTH) -> np.ndarray:
    """
    Calculate the great circle distance between multiple pairs of points on a sphere, adjusted for altitude.

    This function uses the **haversine formula** to compute the surface distance and incorporates the
    altitude difference to approximate the 3D distance.

    Parameters:
    - lats1: Union[np.ndarray]: Latitude of the first set of points in degrees (scalar or array).
    - lons1: Union[np.ndarray]: Longitude of the first set of points in degrees (scalar or array).
    - alts1: Union[np.ndarray]: Altitude of the first set of points in meters (scalar or array).
    - lats2 [np.ndarray]: Latitude of the second set of points in degrees (must match shape of `lats1` or be broadcastable).
    - lons2 [np.ndarray]: Longitude of the second set of points in degrees (must match shape of `lons1` or be broadcastable).
    - alts2 [np.ndarray]: Altitude of the second set of points in meters (must match shape of `alts1` or be broadcastable).
    - R: float, optional : Radius of the sphere in meters (default is Earth's mean radius: 6,371,000 meters).

    Returns:
    - distances: np.ndarray: Array of 3D distances in meters for each pair of points, combining surface distance and altitude difference.

    Notes:
    - The **haversine formula** computes the great circle distance, which is then adjusted for altitude
      using the Pythagorean theorem: `sqrt(d_horiz^2 + delta_h^2)`.
    - Inputs are expected in degrees for latitude/longitude and meters for altitude.
    - Broadcasting is supported for arrays of different shapes, following NumPy rules.
    - This is an approximation, as it assumes small altitude differences relative to Earth's radius.
    """
    # Convert degrees to radians
    lats1_rad = np.radians(lats1)
    lats2_rad = np.radians(lats2)

    # Differences in latitude and longitude
    delta_lats = lats2_rad - lats1_rad
    delta_lons = np.radians(lons2) - np.radians(lons1)

    # Haversine formula for surface distance
    a = np.sin(delta_lats / 2)**2 + np.cos(lats1_rad) * np.cos(lats2_rad) * np.sin(delta_lons / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d_horiz = R * c

    # Incorporate altitude difference
    delta_h = np.abs(alts2 - alts1)
    distances = np.sqrt(d_horiz**2 + delta_h**2)
    
    return distances

def straight_line_distances(lons1: np.ndarray, lats1: np.ndarray, alts1: np.ndarray,
                            lons2: np.ndarray, lats2: np.ndarray, alts2: np.ndarray,
                            R: float = R_EARTH) -> np.ndarray:
    """
    Calculate the straight-line (3D Euclidean) distance between multiple pairs of points using **Cartesian coordinates**.

    Parameters:
    - lats1, np.ndarray: Latitude of the first set of points in degrees (scalar or array).
    - lons1, np.ndarray: Longitude of the first set of points in degrees (scalar or array).
    - alts1, float, np.ndarray: Altitude of the first set of points in meters (scalar or array).
    - lats2, float, np.ndarray: Latitude of the second set of points in degrees (must match shape of `lats1` or be broadcastable).
    - lons2, float, np.ndarray: Longitude of the second set of points in degrees (must match shape of `lons1` or be broadcastable).
    - alts2, float, np.ndarray: Altitude of the second set of points in meters (must match shape of `alts1` or be broadcastable).
    - R, float, optional: Radius of the sphere in meters (default is Earth's mean radius: 6,371,000 meters).

    Returns:
    - distances, np.ndarray: 3D Euclidean distances in meters for each pair of points, including altitude.

    Notes:
    - Converts geographic coordinates (latitude, longitude, altitude) to **Cartesian (x, y, z)** coordinates.
    - Computes the **Euclidean distance** in 3D space.
    - Broadcasting is supported for arrays of different shapes, following NumPy rules.
    - Assumes altitudes are relative to the Earth's surface (e.g., above sea level).
    """
    # Convert degrees to radians
    xs1, ys1, zs1 = geo_to_cart(lons1, lats1, alts1, R)
    xs2, ys2, zs2 = geo_to_cart(lons2, lats2, alts2, R)

    # Compute Euclidean distance
    distances = np.sqrt((xs2 - xs1)**2 + (ys2 - ys1)**2 + (zs2 - zs1)**2)

    return distances

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main Execution
def main() -> None:
    # Générer des données de test (10 000 points, float32)
    n = 10000
    lon = np.random.uniform(-180, 180, n).astype(np.float32)
    lat = np.random.uniform(-90, 90, n).astype(np.float32)
    alt = np.random.uniform(-1000, 10000, n).astype(np.float32)

    # Préchauffer Numba si utilisé
    print('use_numba:', use_numba)
    if use_numba:
        x_nb, y_nb, z_nb = geo_to_cart(lon, lat, alt)

    # Mesurer les performances
    results = {}
    REPEATS = 5000

    # Tester NumPy
    start = time.perf_counter_ns()
    for _ in range(REPEATS):
        x_np, y_np, z_np = geo_to_cart_numpy(lon, lat, alt)
    time_numpy = (time.perf_counter_ns() - start) / (REPEATS * 1e9)
    results['geo_to_cart_numpy'] = time_numpy

    start = time.perf_counter_ns()
    for _ in range(REPEATS):
        lon_np, lat_np, alt_np = cart_to_geo_numpy(x_np, y_np, z_np)
    time_numba = (time.perf_counter_ns() - start) / (REPEATS * 1e9)
    results['cart_to_geo_numpy'] = time_numba

    # Tester Numba
    if use_numba:
        start = time.perf_counter_ns()
        for _ in range(REPEATS):
            x_nb, y_nb, z_nb = geo_to_cart_numba(lon, lat, alt)
        time_numba = (time.perf_counter_ns() - start) / (REPEATS * 1e9)
        results['geo_to_cart_numba'] = time_numba

        start = time.perf_counter_ns()
        for _ in range(REPEATS):
            lon_nb, lat_nb, alt_nb = cart_to_geo_numba(x_nb, y_nb, z_nb)
        time_numba = (time.perf_counter_ns() - start) / (REPEATS * 1e9)
        results['cart_to_geo_numba'] = time_numba

    # Afficher les résultats
    print(f"\nResults for {n} points (float32):")
    for impl, time_val in results.items():
        print(f"  {impl}: {time_val:.6f} s")

    # Calculer les accélérations
    if 'geo_to_cart_numba' in results and 'geo_to_cart_numpy' in results:
        print(f"Speedup Numba vs. NumPy: {(results['geo_to_cart_numpy'] / results['geo_to_cart_numba']):.2f}x")

    if 'cart_to_geo_numba' in results and 'cart_to_geo_numpy' in results:
        print(f"Speedup Numba vs. NumPy: {(results['cart_to_geo_numpy'] / results['cart_to_geo_numba']):.2f}x")

if __name__ == '__main__':
    print(straight_line_distances(0., 80., 0., 0., 81., 0.))
    print(straight_line_distances(0., 80., 0., 1., 80., 0.))
    print(straight_line_distances(0., 80., 0., 0., 80., 1.))
    main()
# ====================================================================================================
