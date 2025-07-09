#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Line_Of_Sight
File   : line_of_sight.py

Author: Pessel Arnaud
Date: 2025-07
Version: 1.0

Description:
    This script calculates intervisibility between two geographic points (origin and targets) using a Digital Terrain Model (DTM) stored in shared memory. 
    It computes the Line of Sight (LOS) between these points, accounting for Earth's curvature and atmospheric refraction, and determines whether topographic obstacles obstruct the view from the origin to the target.

    It leverages the Numba library to optimize performance-critical computations and employs shared memory for efficient access to DTM data, minimizing disk I/O operations.

    Key Objectives:
    - Generate sampled coordinates along the LOS, incorporating geophysical effects such as Earth's curvature and atmospheric refraction.
    - Retrieve terrain altitudes from the DTM for each sampled point along the LOS.
    - Determine intervisibility by comparing adjusted LOS altitudes with terrain altitudes to identify obstructions.
    
    Use Case:
    - Visibility analysis for telecommunications (e.g., antenna placement optimization).

    Features:
    - High Performance: Uses Numba's JIT compilation to accelerate numerical computations, particularly for LOS sampling and intervisibility checks.
    - Shared Memory: Interfaces with a shared-memory DTM for fast and memory-efficient terrain data access.
    - Geophysical Accuracy: Accounts for Earth's curvature and a 4/3 refraction model to simulate radio wave propagation.

    Dependencies:
    - Python 3.x
    - NumPy: For numerical operations and array handling.
    - Matplotlib: For visualization of LOS and terrain data.
    - Numba: For performance optimization of computational loops.
    - Local modules: `get_altitudes_dtm_mp` (for DTM access) and `transform_coord` (for coordinate transformations).

Usage:
    The script requires the following inputs:
    - Shared memory blocks containing DTM tile data, indices, row counts, and column counts.
    - Coordinates (longitude, latitude, altitude) of the origin and target points.
    - Optional resolution parameter (in meters) for sampling the LOS (default: 400 meters).

    Example command:
    ```bash
    python line_of_sight.py --tiles_1d_array shm1 --tiles_indices shm2 --tiles_nrows shm3 --tiles_ncols shm4 --origin -3.2 47.5 30 --target -3.3 47.6 60
    ```

    Output:
    - A boolean value indicating whether the target is visible from the origin (True for visible, False for obstructed).
"""

__version__ = "1.0"

# === Built-in ===
import logging

# === Third-party ===
import matplotlib.pyplot as plt
import numpy as np
import numba
from numba import bool_, float32, int64, jit, prange
from numba.typed import List as Numba_List
from numba.types import Array, ListType

# === Local modules ===
from get_altitudes_dtm_mp import Shm_Dtm_User, get_altitudes_vect_nb
from transform_coord import R_EARTH, cart_to_geo, geo_to_cart, straight_line_distances

# Constants
RADIUS_RATIO = np.float32(4./3.)  # Ratio for atmospheric refraction adjustment (standard 4/3 model for radio waves)
R_RADIO      = np.float32(RADIUS_RATIO * R_EARTH)  # Effective Earth radius adjusted for radio propagation

# Define Numba types
float32_array      = Array(float32, 1, 'C')  # 1D float32 array
list_float32_array = ListType(float32_array)  # List of 1D float32 arrays

@jit(nopython=True, cache=True)
def horizon_intervisibility_distances(alts1, alts2, R=R_EARTH):
    """
    Compute the radioelectric horizon distance for intervisibility.

    Parameters:
    - alts1: Altitude of the first point in meters (float32 or float32[:]).
    - alts2: Altitude(s) of the second point(s) in meters (float32 or float32[:]).
    - R: Radius for horizon calculation, typically Earth's radius or adjusted for refraction (float32).

    Returns:
    - distances: Horizon distances in meters (float32[:]).
    """
    distances = np.sqrt(2*R*alts1) + np.sqrt(2*R*alts2)  # Radioelectric horizon distance
    return distances

# Function to generate linearly spaced values optimized with Numba
@jit(float32[:](float32, float32, int64), nopython=True, cache=True)
def linspace_numba(start, stop, num):
    """
    Generate `num` evenly spaced values between `start` and `stop` for LOS sampling.
    
    Parameters:
    - start: The starting value of the sequence (float32).
    - stop: The end value of the sequence (float32).
    - num: Number of samples to generate, must be >= 1 (int64).
    
    Returns:
    - result: Array of `num` evenly spaced values in float32 format (float32[:]).
    
    Notes:
    - Optimized with Numba for performance in iterative computations.
    - Handles edge case where `num == 1` by setting step to 0.
    """
    result = np.empty(num, dtype=np.float32)
    step   = (stop - start) / (num - 1) if num > 1 else 0.0
    for i in range(num):
        result[i] = start + i * step
    return result

# Function to compute height adjustments for LOS due to Earth's curvature and refraction
@jit(float32[:](float32, float32, float32, float32[:]), nopython=True, cache=True)
def delta_h(Re, Rr, dist_tot, dists):
    """
    Compute height adjustments (delta_h) for LOS calculations, accounting for Earth's curvature and atmospheric refraction.
    
    Parameters:
    - Re: Earth's radius in meters (float32).
    - Rr: Radio radius adjusted for atmospheric refraction, typically 4/3 * Re (float32).
    - dist_tot: Total distance between origin and target points in meters (float32).
    - dists: Array of distances along the LOS for each sampled point (float32[:]).
    
    Returns:
    - dh: Array of height adjustments in meters for each sampled point (float32[:]).
    
    Notes:
    - Uses a parabolic model to approximate the height offset due to curvature and refraction.
    - If `dist_tot <= 0`, returns an array of zeros to avoid division by zero.
    """
    dh = np.empty(len(dists), dtype=np.float32)
    if dist_tot <= 0.:
        dh.fill(0.)  # No adjustment needed for zero or negative distance
    else:
        # Compute central height offset (do) due to curvature and refraction
        do  = np.sqrt(Rr**2 - (dist_tot / 2)**2) - np.sqrt(Re**2 - (dist_tot / 2)**2)
        dhc = Re + do - Rr  # Central height correction
        # Apply parabolic height adjustment based on distance
        dh[:] = 4 * dhc / (dist_tot**2) * dists * (dist_tot - dists)
    return dh

# Core function to compute LOS samples in Cartesian coordinates
@jit(nopython=True, cache=True)
def compute_los_core(x0, y0, z0, xs1, ys1, zs1, resolution_m):
    """
    Compute sampled Cartesian coordinates along the LOS from an origin to multiple target points.
    
    Parameters:
    - x0, y0, z0: X, Y, Z-coordinates of the origin point in meters (float32).
    - xs1, ys1, zs1: X, Y, Z-coordinates of the target points in meters (float32[:]).
    - resolution_m: Sampling resolution along the LOS in meters (float32).
    
    Returns:
    - xs_samples: List of sampled X-coordinates for each LOS (Numba_List).
    - ys_samples: List of sampled Y-coordinates for each LOS (Numba_List).
    - zs_samples: List of sampled Z-coordinates for each LOS (Numba_List).
    - dists_tot: Total Euclidean distances from origin to each target (float32[:]).
    
    Notes:
    - Uses `linspace_numba` to generate evenly spaced samples along each LOS.
    - Number of samples is determined based on total distance and resolution.
    - Parallelized with Numba's `prange` for performance with multiple targets.
    - Type signature omitted due to `Numba_List` complexity; relies on Numba's type inference.
    """
    n = len(xs1)
    # Calculate total Euclidean distances from origin to each target
    dists_tot = (np.sqrt((xs1 - x0)**2 + (ys1 - y0)**2 + (zs1 - z0)**2)).astype(np.float32)
    # Determine number of samples based on resolution (at least 2 samples)
    ns_samples = (2 + dists_tot / resolution_m).astype(np.uint32)
    
    xs_samples, ys_samples, zs_samples = Numba_List(), Numba_List(), Numba_List()
    v_dummy = linspace_numba(0, 1, 2)

    # Initialize numba lists with dummy vector, parallel mode must not be used
    for i in range(n):
        # Generate sampled coordinates for each LOS
        xs_samples.append(linspace_numba(x0, xs1[i], ns_samples[i]))
        ys_samples.append(linspace_numba(y0, ys1[i], ns_samples[i]))
        zs_samples.append(linspace_numba(z0, zs1[i], ns_samples[i]))
    
    return xs_samples, ys_samples, zs_samples, dists_tot

# Function to compute LOS in geographic coordinates and retrieve terrain data
@jit(nopython=True, cache=True)
def compute_los(tiles_1d_array: np.ndarray, tiles_indices: np.ndarray,
                tiles_nrows: np.ndarray, tiles_ncols: np.ndarray,
                lng0, lat0, alt0, lngs1, lats1, alts1, resolution_m=400.):
    """
    Compute the LOS in geographic coordinates and retrieve terrain altitudes from the DTM.
    
    Parameters:
    - tiles_1d_array, tiles_indices, tiles_nrows, tiles_ncols: DTM arrays for accessing terrain data.
    - lng0: Longitude of the origin point in degrees (float).
    - lat0: Latitude of the origin point in degrees (float).
    - alt0: Altitude of the origin point in meters (float).
    - lngs1: Longitudes of the target points in degrees (np.array).
    - lats1: Latitudes of the target points in degrees (np.array).
    - alts1: Altitudes of the target points in meters (np.array).
    - resolution_m: Sampling resolution along the LOS in meters, default 400 (float).
    
    Returns:
    - lngss: Lists of sampled longitudes along each LOS (Numba_List).
    - latss: Lists of sampled latitudes along each LOS (Numba_List).
    - altss_los: Lists of sampled altitudes along each LOS (Numba_List).
    - altss_relief: Lists of terrain altitudes from the DTM for each LOS (Numba_List).
    - xs_samples: Lists of sampled X-coordinates in Cartesian (Numba_List).
    - ys_samples: Lists of sampled Y-coordinates in Cartesian (Numba_List).
    - zs_samples: Lists of sampled Z-coordinates in Cartesian (Numba_List).
    - dists_tot: Total distances from origin to each target (np.array).
    
    Notes:
    - Converts geographic coordinates to Cartesian for LOS computation, then back to geographic for DTM queries.
    - Uses shared memory to efficiently retrieve terrain altitudes.
    """
    # Convert origin and target points to Cartesian coordinates
    x0, y0, z0     = geo_to_cart(lng0, lat0, alt0)
    xs1, ys1, zs1  = geo_to_cart(lngs1, lats1, alts1)

    # Compute sampled Cartesian coordinates along the LOS
    xs_samples, ys_samples, zs_samples, dists_tot = compute_los_core(x0, y0, z0, xs1, ys1, zs1, resolution_m)

    # Initialize lists for geographic coordinates and terrain altitudes
    lngss        = Numba_List()
    latss        = Numba_List()
    altss_los    = Numba_List()
    altss_relief = Numba_List()
    
    if len(xs_samples) > 0:
        for i in range(len(xs_samples)):
            # Convert sampled Cartesian coordinates back to geographic
            lngs, lats, alts = cart_to_geo(xs_samples[i], ys_samples[i], zs_samples[i])
            lngss.append(lngs)
            latss.append(lats)
            altss_los.append(alts)
            # Retrieve terrain altitudes from DTM for the sampled points
            altss_relief.append(get_altitudes_vect_nb(tiles_1d_array, tiles_indices,
                                                      tiles_nrows, tiles_ncols,
                                                      lngs, lats))
    else:
        # Force type of the numba list which is empty ! 
        dummy_array_float32 = np.empty(1, dtype=np.float32)
        lngss.append(dummy_array_float32)
        lngss.pop(0)
        latss.append(dummy_array_float32)
        latss.pop(0)
        altss_los.append(dummy_array_float32)
        altss_los.pop(0)

        dummy_array_uint16  = np.empty(1, dtype=np.uint16)
        altss_relief.append(dummy_array_uint16)
        altss_relief.pop(0)
    
    return lngss, latss, altss_los, altss_relief, xs_samples, ys_samples, zs_samples, dists_tot

# Core function to determine intervisibility
@jit(nopython=True, parallel=True, cache=True)
def are_intervisible_core(xs_samples: list_float32_array, ys_samples  : list_float32_array, zs_samples: list_float32_array,
                          altss_los : list_float32_array, altss_relief: list_float32_array, R: float32, dists_tot: float32[:]) -> bool_[:]:
    """
    Determine if target points are intervisible from the origin, considering terrain and geophysical effects.
    
    Parameters:
    - xs_samples: Lists of sampled X-coordinates along each LOS (Numba_List).
    - ys_samples: Lists of sampled Y-coordinates along each LOS (Numba_List).
    - zs_samples: Lists of sampled Z-coordinates along each LOS (Numba_List).
    - altss_los : Lists of altitudes along each LOS (Numba_List).
    - altss_relief: Lists of terrain altitudes from the DTM (Numba_List).
    - R: Radius adjusted for refraction in meters (float32).
    - dists_tot: Total distances from origin to each target in meters (float32[:]).
    
    Returns:
    - result: Boolean array indicating intervisibility, True if visible (bool_[:]).
    
    Notes:
    - Adjusts LOS altitudes with `delta_h` to account for Earth's curvature and refraction.
    - Checks if any terrain altitude exceeds the adjusted LOS altitude, indicating an obstruction.
    - Parallelized with Numba's `prange` for performance with multiple targets.
    """
    n: np.uint32 = len(xs_samples)
    result = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        # Calculate distances along the LOS from the origin
        dists = np.sqrt((xs_samples[i] - xs_samples[i][0])**2 +
                        (ys_samples[i] - ys_samples[i][0])**2 +
                        (zs_samples[i] - zs_samples[i][0])**2).astype(np.float32)
        # Compute height adjustments for curvature and refraction
        dh = delta_h(R_EARTH, R, dists_tot[i], dists)
        # Adjust LOS altitudes and compare with terrain altitudes
        alts_adj = altss_los[i] + dh
        result[i] = not np.any(altss_relief[i] > alts_adj)  # True if no terrain exceeds adjusted LOS
    return result

# Function to check intervisibility using the DTM
def are_intervisible(shm_dtm_user, lng0, lat0, alt0, lngs1, lats1, alts1, resolution_m=400.0, R=R_RADIO):
    """
    Check if target points are intervisible from the origin point using the DTM.
    
    Parameters:
    - tiles_1d_array, tiles_indices, tiles_nrows, tiles_ncols: DTM arrays for terrain data access.
    - lng0: Longitude of the origin point in degrees (float).
    - lat0: Latitude of the origin point in degrees (float).
    - alt0: Altitude of the origin point in meters (float).
    - lngs1: Longitudes of the target points in degrees (np.array).
    - lats1: Latitudes of the target points in degrees (np.array).
    - alts1: Altitudes of the target points in meters (np.array).
    - resolution_m: Sampling resolution along the LOS in meters, default 400 (float).
    - R: Radius adjusted for refraction in meters, default R_RADIO (float32).
    
    Returns:
    - result: Boolean array indicating intervisibility for each target, True if visible (np.array).
    
    Notes:
    - Orchestrates LOS computation and intervisibility check by calling `compute_los` and `are_intervisible_core`.
    - Ensures efficient terrain data access via shared memory.
    """
    R = np.float32(R)  # Ensure radius is float32 for consistency

    # Compute horizon intervisibilities
    distances           = straight_line_distances(lng0, lat0, alt0, lngs1, lats1, alts1)
    intervisi_distances = horizon_intervisibility_distances(alt0, alts1, R=R)
    #logging.info(('R, distances[0], intervisi_distances[0]', R, distances[0], intervisi_distances[0]))
    intervisibilities   = (distances <= intervisi_distances)
    
    if intervisibilities.any():
        intervisi_args      = np.flatnonzero(intervisibilities)

        # Compute LOS samples and retrieve terrain altitudes
        lngs1, lats1, alts1 = lngs1[intervisi_args], lats1[intervisi_args], alts1[intervisi_args]
        lngss, latss, altss_los, altss_relief, xs, ys, zs, dists = compute_los(shm_dtm_user.tiles_1d_array, shm_dtm_user.tiles_indices_array,
                                                                               shm_dtm_user.tiles_nrows_array, shm_dtm_user.tiles_ncols_array,
                                                                               lng0, lat0, alt0, lngs1, lats1, alts1, resolution_m)
        
        relief_intervisibilities = are_intervisible_core(xs, ys, zs, altss_los, altss_relief, R, dists)

        intervisibilities[intervisi_args] = relief_intervisibilities

    return intervisibilities

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main execution block

# Function to plot the line of sight
def plot_los(shm_dtm_user, lng0, lat0, alt0, lng1, lat1, alt1, resolution_meters=400., R=R_RADIO):
    """
    Plot the Line of Sight (LOS) between two points, showing terrain and LOS altitudes.
    
    Parameters:
    - shm_dtm_user: Shared memory DTM user object for terrain data access (Shm_Dtm_User).
    - lng0: Longitude of the origin point in degrees (float).
    - lat0: Latitude of the origin point in degrees (float).
    - alt0: Altitude of the origin point in meters (float).
    - lng1: Longitude of the target point in degrees (float).
    - lat1: Latitude of the target point in degrees (float).
    - alt1: Altitude of the target point in meters (float).
    - resolution_meters: Sampling resolution along the LOS in meters, default 400 (float).
    - R: Radius adjusted for refraction in meters, default R_RADIO (float32).
    
    Returns:
    - dist: Total distance from origin to target in meters (float).
    
    Notes:
    - Plots terrain relief and LOS altitudes against distance, longitude, and latitude.
    - Uses shared memory DTM for terrain data access.
    - alts_virt and straight are set to 0 and unused in active plotting.
    - Suggestion: Remove commented-out plt.plot lines and unused alts_virt/straight if not needed.
    """
    #logging.info(f'[lng0, lat0, alt0], [lng1, lat1, alt1], resolution_meters:\n{[lng0, lat0, alt0]}\n{[lng1, lat1, alt1]}\n{resolution_meters}')
    lngs1, lats1, alts1 = np.array([lng1], dtype=np.float32), np.array([lat1], dtype=np.float32), np.array([alt1], dtype=np.float32)

    lngss_los, latss_los, altss_los, altss_relief, xs_samples, ys_samples, zs_samples, dists_tot = \
        compute_los(shm_dtm_user.tiles_1d_array, shm_dtm_user.tiles_indices_array,
                    shm_dtm_user.tiles_nrows_array, shm_dtm_user.tiles_ncols_array,
                    lng0, lat0, alt0, lngs1, lats1, alts1, resolution_meters)
    
    x_samples, y_samples, z_samples = xs_samples[0], ys_samples[0], zs_samples[0]                         # note: keeping the 1s and unique target point
    lngs_los, lats_los, alts_los, alts_relief = lngss_los[0], latss_los[0], altss_los[0], altss_relief[0] # note: keeping the 1s and unique target point

    dists     = np.sqrt((x_samples - x_samples[0])**2 + (y_samples - y_samples[0])**2 + (z_samples - z_samples[0])**2)
    dh        = delta_h(R_EARTH, R_EARTH*RADIUS_RATIO, dists_tot[0], dists)
    alts_los += dh

    alts_virt = 0  # R_EARTH**2/np.sqrt(R_EARTH**2-(dists-dist_tot/2)**2) - R_EARTH
    straight  = 0  # np.interp(dists, [dists[0], dists[-1]], [alts_los[0] - alts_virt[0], 0])
    # plt.plot(dists, alts_relief-alts_los+0 +straight, 'b.-', label='relief')
    # plt.plot(dists, alts_los   -alts_los+dh+straight, 'k.-', label='line of sight')
    # Note: Above plt.plot lines are commented out and unused.
    # Suggestion: Remove if not intended for future use.
    fig  = plt.gcf()
    axes = fig.get_axes()
    axes[0].plot(dists, alts_relief-alts_virt-straight, '-', label='relief')
    axes[0].plot(dists, alts_los   -alts_virt-straight, '-', label='line of sight')
    axes[0].set_xlim(dists[0], dists[-1])
    axes[0].set_xticks(np.linspace(dists[0], dists[-1], 10))
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Altitude (m)')
    axes[0].legend()
    axes[0].grid(True, which='both') # Shows gridlines for both major and minor ticks

    axes[1].plot(lngs_los, alts_relief-alts_virt-straight, '-')
    axes[1].set_xlim(lngs_los[0]-3e-4, lngs_los[-1]+3e-4)
    axes[1].set_xticks(np.linspace(lngs_los[0]-3e-4, lngs_los[-1]+3e-4, 10))
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].set_xlabel('Longitude')
    axes[1].grid(True, which='both') # Shows gridlines for both major and minor ticks
    
    axes[2].plot(lats_los, alts_relief-alts_virt-straight, '-')
    axes[2].set_xlim(lats_los[0]-3e-4, lats_los[-1]+3e-4)
    axes[2].set_xticks(np.linspace(lats_los[0]-3e-4, lats_los[-1]+3e-4, 10))
    axes[2].xaxis.set_minor_locator(AutoMinorLocator())
    axes[2].set_xlabel('Latitude')
    axes[2].grid(True, which='both') # Shows gridlines for both major and minor ticks

    return dists[-1]

cur_fig_num = 0

def submain(shm_dtm_user, m1, m2, m3, n_points=1000):
    """
    Test intervisibility for multiple points with visualization using shared memory DTM.
    
    Parameters:
    - shm_dtm_user: Shared memory DTM user object for terrain data access (Shm_Dtm_User).
    - m1: Named tuple with lon, lat, alt for the origin point.
    - m2, m3: Named tuples with lon, lat, alt defining the range of target points.
    - n_points: Number of points to generate in the grid (default: 1000).
    
    Notes:
    - Generates a grid of target points and checks intervisibility.
    - Includes visualization using Matplotlib, which may be unnecessary in CLI context.
    """
    global cur_fig_num
    logging.info('------------- submain')
    
    # Define origin point
    m23_lngs  = np.linspace(m2.lon, m3.lon, n_points)
    m23_lats  = np.linspace(m2.lat, m3.lat, n_points)
    m23_alts  = np.linspace(m2.alt, m3.alt, n_points)

    logging.info(f'Perform intervisibility check: {m1} ; ({m2.lon}..{m3.lon}, {m2.lat}..{m3.lat}, {m2.alt}..{m3.alt})')
    distances           = straight_line_distances(m1.lon, m1.lat, m1.alt, m23_lngs, m23_lats, m23_alts)
    intervisi_distances = horizon_intervisibility_distances(m1.alt, m23_alts, R=R_RADIO)

    # Test vectorized intervisibility
    start_time = time.perf_counter()
    intervisibilities   = are_intervisible(shm_dtm_user, m1.lon, m1.lat, m1.alt, m23_lngs, m23_lats, m23_alts, R=R_RADIO)
    execution_time = (time.perf_counter() - start_time)
    logging.info(f'Time for intervisibility check: {execution_time*1000.:10.3f}ms')

    # Initialize plot
    cur_fig_num += 1
    fig, axes = plt.subplots(3, 1, num=f'los{cur_fig_num}', figsize=(12, 6), height_ratios=[20, 1, 1])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.07, hspace=0.4)

    r_ = range(0, n_points)
    for idx in range(0, n_points):
        if idx == 0 or idx == r_[-1] or \
        intervisibilities[idx] != intervisibilities[idx-1] or \
        intervisibilities[idx] != intervisibilities[idx+1]:
            print(f'-- [{idx:03d}] Dist intervisi theo: {(intervisi_distances[idx]/1000.):8.3f} km')
            print(f'         Dist entre points  : {(distances[idx]/1000.):8.3f} km')
            print(f'         Intervisibility    : {intervisibilities[idx]}')

            plot_los(shm_dtm_user, m1.lon, m1.lat, m1.alt, m23_lngs[idx], m23_lats[idx], m23_alts[idx])

    # Note: Legend removal is redundant as plot_los manages legends.
    # Suggestion: Remove this line as it may raise an error if no legend exists.
    fig.axes[0].legend().remove()

# Main function to handle CLI arguments and execute intervisibility check
def main():
    """
    Parse command-line arguments and execute the intervisibility computation.
    
    Notes:
    - Initializes the shared memory DTM interface.
    - Processes input coordinates and resolution.
    - Outputs the intervisibility result and cleans up resources.
    - Calls submain and submain2 for additional testing.
    """
    # Define CLI argument parser with detailed help messages
    parser = argparse.ArgumentParser(
        description='Calculate Line of Sight (LOS) intervisibility using shared memory DTM.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
    python line_of_sight.py --tiles_1d_array shm1 --tiles_indices shm2 --tiles_nrows shm3 --tiles_ncols shm4 --origin -3.2 47.5 30 --target -3.3 47.6 60

Notes:
    - Shared memory blocks must be pre-allocated and accessible.
    - Coordinates should be provided in decimal degrees (longitude, latitude) and meters (altitude).
    - Resolution affects the granularity of LOS sampling; lower values increase accuracy but increase computation time.
'''
    )

    # Define required and optional arguments
    parser.add_argument('--tiles_1d_array', required=True, help='Shared memory block containing DTM tile data')
    parser.add_argument('--tiles_indices', required=True, help='Shared memory block containing tile indices')
    parser.add_argument('--tiles_nrows', required=True, help='Shared memory block containing tile row counts')
    parser.add_argument('--tiles_ncols', required=True, help='Shared memory block containing tile column counts')
    parser.add_argument('--origin', nargs=3, type=float, metavar=('LON', 'LAT', 'ALT'), required=False,
                        help='Origin point coordinates: longitude (degrees), latitude (degrees), altitude (meters)')
    parser.add_argument('--target', nargs=3, type=float, metavar=('LON', 'LAT', 'ALT'), required=False,
                        help='Target point coordinates: longitude (degrees), latitude (degrees), altitude (meters)')
    parser.add_argument('--resolution', type=float, default=400.0,
                        help='Sampling resolution along the LOS in meters (default: 400)')

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Initialize shared memory DTM interface with provided blocks
    from multiprocessing import resource_tracker
    resource_tracker.register = lambda name, rtype: None
    
    shm_dtm_user = Shm_Dtm_User({"tiles_1d_array": args.tiles_1d_array,
                                 "tiles_indices" : args.tiles_indices,
                                 "tiles_nrows"   : args.tiles_nrows,
                                 "tiles_ncols"   : args.tiles_ncols})
        
    # Run test functions
    Point = namedtuple('Point', ['lon', 'lat', 'alt'])

    resolution_m = args.resolution

    # Warm up the JIT compiler
    m1  = Point(-180., 180., 10000.)
    m2s = np.array([[-90.], [90.], [10000.]], dtype=np.float32)
    intervisibilities   = are_intervisible(shm_dtm_user, m1.lon, m1.lat, m1.alt, m2s[0], m2s[1], m2s[2], R=R_RADIO)

    if args.origin and args.target:
        m1 = Point(*args.origin)
        m2 = Point(*args.target)
        submain(shm_dtm_user, m1, m2, m2, n_points = 1)

    m1 = Point(-12.2, 47.5, 10000.)
    m2 = Point(m1.lon + 6.0, m1.lat + 6.1, m1.alt + 10.)
    m3 = Point(m1.lon + 6.3, m1.lat + 6.4, m1.alt + 100.)
    submain(shm_dtm_user, m1, m2, m3, n_points = 1000)

    m1 = Point(-4.42, 48.38, 100.)
    m3 = Point(-4.47, 47.94, 100.)
    m2 = Point(0.9*m1.lon + 0.1*m3.lon, 0.9*m1.lat + 0.1*m3.lat, 0.9*m1.alt + 0.1*m3.alt)
    submain(shm_dtm_user, m1, m2, m3, n_points = 1000)

    # Loop to test intervisibility for multiple random points
    logging.info('Testing intervisibility for multiple random points...')
    m1_size = 1000
    m1s = np.array([np.random.uniform(-180.,   180., size=m1_size),
                    np.random.uniform( -80.,    80., size=m1_size),
                    np.random.uniform(1000., 20000., size=m1_size)], dtype=np.float32)

    m2_size = 1000
    deviations = np.random.uniform(-10., 10., size=(3, m1_size, m2_size)).astype(np.float32)
    m2ss = m1s[:, :, np.newaxis] + deviations

    execution_times = []
    for idx in tqdm(range(m2_size), total=m2_size, desc="Compute intervisibility and measure execution time for each point"):
        # For each point, compute intervisibility and measure execution time
        start_time = time.perf_counter()
        intervisibilities = are_intervisible(shm_dtm_user,
                                             m1s [0, idx], m1s [1, idx], m1s [2, idx],
                                             m2ss[0, idx], m2ss[1, idx], m2ss[2, idx], R=R_RADIO)
        execution_time = time.perf_counter() - start_time
        execution_times.append(execution_time)
    
    execution_times = np.array(execution_times) * 1000.  # Convert to milliseconds
    logging.info(f"Number of time entries      : {len(execution_times)}")
    logging.info(f"Number of points per entries: {m2_size}")
    logging.info(f"Total time          : {np.sum(execution_times):.4f} ms")
    logging.info(f"Mean time           : {np.mean(execution_times):.4f} ms")
    logging.info(f"Standard deviation  : {np.std(execution_times):.4f} ms")
    logging.info(f"Minimum time        : {np.min(execution_times):.4f} ms")
    logging.info(f"Maximum time        : {np.max(execution_times):.4f} ms")
    logging.info(f"99th percentile time: {np.percentile(execution_times, 99):.4f} ms")

    # Close shared memory resources to prevent memory leaks
    shm_dtm_user.close()

if __name__ == '__main__':
    # === Built-in ===
    import argparse
    import time
    from tqdm import tqdm
    from collections import namedtuple

    # === Third-party ===
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator

    # Configure logging only when running as main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

    def close_figures():
        (plt.close(fignum) for fignum in plt.get_fignums())
    
    def save_figures():
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            fig.savefig(f'figure_{fig.get_label()}.png')

    main()
    save_figures()
# ======================================================================
