#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: Line_Of_Sight
File   : get_altitudes_dtm_mp.py

Author: Pessel Arnaud
Date: 2025-07
Version: 1.0
GitHub: https://github.com/dunaar/Line_Of_Sight
License: MIT

Description:
    This script loads preprocessed SRTM15 tiles from a ZIP archive and retrieves altitudes
    for specified coordinates using shared memory for efficient data sharing between processes.
    It provides three main modes of operation:
    - loader: Loads elevation data into shared memory using SharedMemoryArray
    - user: Retrieves altitudes using existing shared memory blocks
    - standalone: Combines both functionalities and generates visualizations

    The script uses Numba for performance optimization and SharedMemoryArray with struct
    for efficient shared memory management, storing metadata (dtype and shape) directly
    in the shared memory blocks.
"""

__version__ = "1.0"

# === Built-in ===
import argparse
import logging
import struct
import sys
import time
import zipfile
from mmap import ALLOCATIONGRANULARITY
from multiprocessing import shared_memory as shm

# === Third-party ===
import numpy as np
from numba import jit, prange, uint16, uint32, uint64, float32
from tqdm import tqdm

# === Local modules ===
from np_msgspec_msgpack_utils import dec

@jit(uint16(uint16[:], uint64[:,:], uint32[:,:], uint32[:,:], float32, float32),
     nopython=True, cache=True)
def _get_altitude_nb(tiles_1d_array: np.ndarray, tiles_indices: np.ndarray,
                    tiles_nrows: np.ndarray, tiles_ncols: np.ndarray,
                    lon: float, lat: float):
    """
    Get altitude at a given latitude and longitude using Numba optimization.

    This function calculates the altitude for a single coordinate point by:
    1. Validating and adjusting the latitude bounds
    2. Converting coordinates to absolute values
    3. Calculating tile indices and in-tile positions
    4. Retrieving the altitude from the preprocessed tile data

    Args:
        tiles_1d_array (uint16[:]): 1D array containing all tile data concatenated
        tiles_indices (uint64[:,:]): Array mapping tile coordinates to positions in tiles_1d_array
        tiles_nrows (uint32[:,:]): Array containing number of rows for each tile
        tiles_ncols (uint32[:,:]): Array containing number of columns for each tile
        lon (float32): Longitude coordinate in decimal degrees
        lat (float32): Latitude coordinate in decimal degrees

    Returns:
        uint16: Altitude value at the specified coordinates
    """
    # Convert to absolute coordinates (0-360 for longitude, 0-180 for latitude)
    lon_abs: np.float32 = (np.float32(lon) + 180.) % 360.
    
    # Ensure latitude is within valid bounds
    lat_abs: np.float32 = np.float32(lat) + np.float32(90)
    if   lat_abs <    0.: lat_abs: np.float32 = 0.
    elif lat_abs >= 180.: lat_abs: np.float32 = 179.99998

    tile_idx_lon = np.uint32(lon_abs)
    tile_idx_lat = np.uint32(lat_abs)
    nrows = tiles_nrows[tile_idx_lat, tile_idx_lon]
    ncols = tiles_ncols[tile_idx_lat, tile_idx_lon]
    size: np.uint64 = nrows * ncols

    tile_idx_beg: np.uint64 = tiles_indices[tile_idx_lat, tile_idx_lon]

    # Handle special case for single-value tiles
    if size == 1:
        return tiles_1d_array[tile_idx_beg]

    # Calculate position within the tile
    intile_idx_lon: np.uint64 = np.uint64(ncols * (lon_abs - tile_idx_lon))
    intile_idx_lat: np.uint64 = np.uint64(nrows * (lat_abs - tile_idx_lat))
    intile_idx: np.uint64 = intile_idx_lat * ncols + intile_idx_lon

    return tiles_1d_array[tile_idx_beg + intile_idx]

@jit(uint16[:](uint16[:], uint64[:,:], uint32[:,:], uint32[:,:], float32[:], float32[:]), nopython=True, cache=True)
def get_altitudes_vect_nb(tiles_1d_array: np.ndarray, tiles_indices: np.ndarray,
                          tiles_nrows: np.ndarray, tiles_ncols: np.ndarray,
                          lons: np.ndarray, lats: np.ndarray):
    """
    Vectorized retrieval of altitudes for multiple coordinates.

    This function processes multiple coordinate pairs efficiently by:
    1. Creating an output array to store results
    2. Using parallel processing (prange) to calculate altitudes for all coordinates
    3. Reshaping the results to match the input dimensions

    Args:
        tiles_1d_array (uint16[:]): 1D array containing all tile data concatenated
        tiles_indices (uint64[:,:]): Array mapping tile coordinates to positions in tiles_1d_array
        tiles_nrows (uint32[:,:]): Array containing number of rows for each tile
        tiles_ncols (uint32[:,:]): Array containing number of columns for each tile
        lons (float32[:]): Array of longitude coordinates
        lats (float32[:]): Array of latitude coordinates

    Returns:
        uint16[:]: Array of altitude values with same shape as input coordinates
    """
    # Initialize output array
    alts = np.empty(len(lons), dtype=np.uint16)

    # Process coordinates in parallel
    for idx_coord in prange(len(lons)):
        alts[idx_coord] = _get_altitude_nb(tiles_1d_array, tiles_indices,
                                         tiles_nrows, tiles_ncols,
                                         lons[idx_coord], lats[idx_coord])

    # Reshape to match input dimensions
    return alts

class SharedMemoryArray:
    """
    Class for managing a shared memory array.

    This class provides an interface for creating and accessing a shared memory array,
    allowing multiple processes to read and write data concurrently. Uses struct for metadata
    serialization in the format: dtype_len (uint32), dtype_str (str), shape_ndim (uint32),
    shape_size_dim1 (uint32), ..., shape_size_dimN (uint32).

    Attributes:
        shm (multiprocessing.shared_memory.SharedMemory): Shared memory block
        array (numpy.ndarray): Numpy array backed by shared memory
    """
    def __init__(self, name=None, shape=None, dtype=None):
        """
        Initialize the SharedMemoryArray.

        Args:
            name (str, optional): Name of an existing shared memory block to map.
            shape (tuple, optional): Shape of the array for new shared memory.
            dtype (numpy.dtype, optional): Data type of the array for new shared memory.

        If name is provided, maps an existing shared memory block.
        If shape and dtype are provided, creates a new shared memory block.
        """
        self.shm    = None
        self.array  = None
        self.create = False
        metadata_struct_size = 0

        if name is not None:
            # Map an existing shared memory block
            self._map_existing(name)
        elif shape is not None and dtype is not None:
            # Create a new shared memory block
            self._create_new(shape, dtype)
        else:
            raise ValueError("Must provide either 'name' or both 'shape' and 'dtype'")

    def _create_new(self, shape, dtype):
        """
        Create a new shared memory block.

        Args:
            shape (tuple): Shape of the array.
            dtype (numpy.dtype): Data type of the array.
        """
        """
        Metadata struct:
        - dtype_len (uint32)  : length of the string dtype_str.
        - dtype_str (str)     : string representation of the dtype encoded in UTF-8.
        - shape_ndim (uint32) : number of dimensions.
        - shape_size_dim1, ..., shape_size_dimN (uint32) : sizes of the dimensions.
        - Format struct : <I s{MAX_DIMS} I I{shape_ndim}>, where dtype_len is determined dynamically.
        """
        dtype = np.dtype(dtype)

        # Prepare metadata
        dtype_bytes = str(dtype).encode("ascii", "ignore")
        dtype_len   = len(dtype_bytes)
        shape_ndim  = len(shape)

        # Calculate metadata size
        # I for dtype_len, I for shape_ndim, I*MAX_DIMS for shape dimensions
        metadata_struct_format = f'= I {dtype_len}s I {shape_ndim}I'
        metadata_struct_size   = struct.calcsize(metadata_struct_format)
        metadata_bytes         = struct.pack(metadata_struct_format, dtype_len, dtype_bytes, shape_ndim, *shape)

        # Create shared memory
        data_size = int(np.prod(shape) * dtype.itemsize) # cast for windows compatibility
        self.create = True
        self.shm  = shm.SharedMemory(create=True, size=metadata_struct_size + data_size)

        # Write metadata
        self.shm.buf[:metadata_struct_size] = metadata_bytes

        # Create NumPy array (initialized to zero)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf, offset=metadata_struct_size)

    def _map_existing(self, name):
        """
        Map an existing shared memory block.

        Args:
            name (str): Name of the shared memory block.
        """
        # Access shared memory
        try:
            logging.debug(f'Loading shared memory <{name}>...')
            self.create = False
            self.shm = shm.SharedMemory(name=name)
        except Exception as e:
            logging.error(f'Error loading shared memory <{name}>: {e}')
            raise Exception(f"Error loading shared memory: {e}")
        
        logging.debug(f'Shared memory <{name}> loaded !')

        # Read metadata
        try:
            # Read dtype_len and shape_ndim
            dtype_len  = struct.unpack('=I', self.shm.buf[:4])[0]
            offset     = 4

            dtype_str  = bytes( self.shm.buf[offset:offset + dtype_len] ).decode('ascii', 'ignore')
            dtype      = np.dtype(dtype_str)
            offset    += dtype_len

            shape_ndim = struct.unpack('=I', self.shm.buf[offset:offset + 4])[0]
            offset    += 4

            shape      = struct.unpack(f'={shape_ndim}I', self.shm.buf[offset:offset + 4 * shape_ndim])
            offset    += 4 * shape_ndim
        except Exception as e:
            logging.error(f'Error loading metadata: {e}')
            self.close()
            raise ValueError(f"Failed to read metadata: {e}")
        
        logging.debug(f'Shared memory <{name}>, array parmeters: dtype = {dtype}, shape = {shape}')
        total_size   = offset + np.prod(shape) * dtype.itemsize
        aligned_size = int(np.ceil(total_size / ALLOCATIONGRANULARITY)) * ALLOCATIONGRANULARITY # alignment according sys page padding
        if not total_size <= self.shm.buf.nbytes <= aligned_size:
            logging.error(f'Shared memory <{name}>: size mismatch! Expected {total_size} bytes, got {self.shm.buf.nbytes} bytes')
            self.close()
            raise ValueError(f"Shared memory size mismatch: expected {total_size}, got {self.shm.buf.nbytes}")

        # Create NumPy array
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf, offset=offset)

        
        logging.debug(f'Shared memory <{name}>: array mapped !')

    @property
    def name(self):
        """Return the name of the shared memory block."""
        return self.shm.name if self.shm else None

    @property
    def shape(self):
        """Return the shape of the array."""
        return self.array.shape if self.array is not None else None

    @property
    def dtype(self):
        """Return the dtype of the array."""
        return self.array.dtype if self.array is not None else None

    def close(self):
        """Unlink (delete) the shared memory block."""
        if self.shm:
            del self.array  # Ensure the array is deleted before closing
            self.array = None

            self.shm.close()
            
            if self.create:
                self.create = False
                self.shm.unlink()
                
            self.shm = None

    def __del__(self):
        """Ensure the shared memory is closed when the object is deleted."""
        self.close()

class Shm_Dtm_Loader:
    """
    Class for loading Digital Terrain Model (DTM) data into shared memory.

    This class handles the loading of elevation data from ZIP archives into shared memory blocks
    using SharedMemoryArray, making the data accessible to other processes. It manages the lifecycle
    of shared memory, including creation, population, and cleanup.

    Attributes:
        filename (str): Path to the ZIP file containing elevation data
        tiles_data (dict): Dictionary containing SharedMemoryArray instances
        shm_names (dict): Dictionary containing shared memory block names
    """

    def __init__(self, filename) -> None:
        """
        Initialize the Shm_Dtm_Loader with a filename.

        Args:
            filename (str): Path to the ZIP file containing elevation data
        """
        self.filename = filename
        self.tiles_data = None
        self.shm_names = None
        self.load_dtm_file()

    def load_dtm_file(self, filename=None):
        """
        Load tiles and metadata from a zip archive containing MessagePack-encoded data.

        This method:
        1. Validates the input filename
        2. Reads and decodes metadata from the ZIP archive
        3. Creates SharedMemoryArray instances for different data components
        4. Populates the shared memory with data from the ZIP archive
        5. Stores names of the created shared memory blocks

        Args:
            filename (str, optional): Path to the ZIP file. If provided, updates the instance filename.

        Raises:
            ValueError: If filename is not provided
            FileNotFoundError: If the ZIP file is not found
            Exception: For other errors during loading
        """
        if filename is not None:
            self.close()
            self.filename = filename

        if not self.filename:
            raise ValueError("Filename not provided.")

        metadata = {}

        try:
            with zipfile.ZipFile(self.filename, mode='r') as zf:
                # Load and decode metadata
                metadata = dec.decode(zf.read('metadata.packed'))
                tiles_lon_indices_in_bands = metadata['tiles_lon_indices_in_bands']

                # Create SharedMemoryArray instances for different data components
                shm_tiles_indices = SharedMemoryArray(shape=tiles_lon_indices_in_bands.shape,
                                                      dtype=tiles_lon_indices_in_bands.dtype)

                shm_tiles_nrows = SharedMemoryArray(shape=metadata['tiles_nrows'].shape,
                                                    dtype=metadata['tiles_nrows'].dtype)
                np.copyto(shm_tiles_nrows.array, metadata['tiles_nrows'])

                shm_tiles_ncols = SharedMemoryArray(shape=metadata['tiles_ncols'].shape,
                                                    dtype=metadata['tiles_ncols'].dtype)
                np.copyto(shm_tiles_ncols.array, metadata['tiles_ncols'])

                # Calculate cumulative indices for tiles
                idx_cumul = 0
                for lat_idx in range(len(shm_tiles_nrows.array)):
                    band_size = (shm_tiles_nrows.array[lat_idx] * shm_tiles_ncols.array[lat_idx]).sum()
                    shm_tiles_indices.array[lat_idx, :] = tiles_lon_indices_in_bands[lat_idx, :] + idx_cumul
                    idx_cumul += band_size

                expected_total_size = idx_cumul

                # Process latitude bands
                lat_keys = sorted(
                    (name for name in zf.namelist() if name.startswith('lat_') and name.endswith('.packed')),
                    key=lambda x: int(x.split('_')[1].split('.')[0])
                )

                band_array = dec.decode(zf.read(lat_keys[0]))
                band_dtype = band_array.dtype
                band_word_nbytes = band_array.nbytes // band_array.size

                # Create SharedMemoryArray for the main tile data
                shm_tiles_1d_array = SharedMemoryArray(shape=(expected_total_size,),
                                                       dtype=band_dtype)

                # Load data from each latitude band
                idx_cumul = 0
                for name in tqdm(lat_keys, total=len(lat_keys),
                               desc=f"Loading tiles from file <{self.filename}>"):
                    lat_idx = int(name.split('_')[1].split('.')[0])
                    band_size = (shm_tiles_nrows.array[lat_idx] * shm_tiles_ncols.array[lat_idx]).sum()

                    band_array = dec.decode(zf.read(name))
                    if band_array.size != band_size:
                        raise ValueError(f"Size mismatch for {name}: {band_array.size} != {band_size}")

                    if idx_cumul + band_array.size > shm_tiles_1d_array.array.size:
                        raise RuntimeError('Exceeded tiles_1d_array size while loading band arrays')

                    np.copyto(shm_tiles_1d_array.array[idx_cumul:idx_cumul + band_array.size], band_array)
                    idx_cumul += band_array.size
                    del band_array

            # Store SharedMemoryArray instances
            self.tiles_data = {'shm_tiles_1d_array': shm_tiles_1d_array,
                               'shm_tiles_indices' : shm_tiles_indices,
                               'shm_tiles_nrows'   : shm_tiles_nrows,
                               'shm_tiles_ncols'   : shm_tiles_ncols}

            # Store names of shared memory blocks
            self.shm_names = {'tiles_1d_array': shm_tiles_1d_array.name,
                              'tiles_indices' : shm_tiles_indices.name,
                              'tiles_nrows'   : shm_tiles_nrows.name,
                              'tiles_ncols'   : shm_tiles_ncols.name}

        except FileNotFoundError:
            raise FileNotFoundError(f"Zip file {self.filename} not found.")
        except Exception as e:
            raise Exception(f"Error loading DTM file: {e}")

    def close(self):
        """
        Clean up shared memory blocks.

        This method safely closes and unlinks all SharedMemoryArray instances that were created.
        It handles potential errors during cleanup to ensure the process doesn't fail.
        """
        if self.tiles_data is None:
            return

        for key, obj in self.tiles_data.items():
            if key.startswith('shm_') and hasattr(obj, 'close'):
                obj.close()
                logging.info(f"Shared memory block {key} closed")

        self.tiles_data = None
        self.shm_names = None

class Shm_Dtm_User:
    """
    Class for retrieving altitudes using shared memory blocks.

    This class connects to existing shared memory blocks created by Shm_Dtm_Loader
    using SharedMemoryArray and provides methods to retrieve altitude data for single
    coordinates or arrays of coordinates.

    Attributes:
        shm_names (dict): Dictionary containing shared memory block names
        shm_instances (dict): Dictionary containing SharedMemoryArray instances
    """
    def __init__(self, shm_names):
        """
        Initialize the Shm_Dtm_User with shared memory block names.

        Args:
            shm_names (dict): Dictionary containing shared memory block names
        """
        self.shm_names = shm_names
        self.shm_instances = {}

        for key in ('tiles_1d_array', 'tiles_indices', 'tiles_nrows', 'tiles_ncols'):
            try:
                logging.debug(f'Loading shared memory {key}: {self.shm_names[key]}')
                self.shm_instances[key] = SharedMemoryArray(name=self.shm_names[key])
            except FileNotFoundError:
                raise FileNotFoundError(f"Shared memory block {key} not found.")
            except Exception as e:
                self.close()
                raise Exception(f"Error accessing shared memory block {key}: {e}")

    def get_altitude(self, lon, lat):
        """
        Get altitude for a single coordinate.

        Args:
            lon (float): Longitude in decimal degrees (-180 to 180)
            lat (float): Latitude in decimal degrees (-90 to 90)

        Returns:
            uint16: Altitude value at the specified coordinates

        Raises:
            ValueError: If coordinates are out of valid ranges
        """
        if not (-180 <= lon <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees.")

        return _get_altitude_nb(self.shm_instances['tiles_1d_array'].array,
                                self.shm_instances['tiles_indices'].array,
                                self.shm_instances['tiles_nrows'].array,
                                self.shm_instances['tiles_ncols'].array,
                                lon, lat)

    def get_altitudes(self, lons, lats):
        """
        Get altitudes for multiple coordinates.

        Args:
            lons (numpy.ndarray): Array of longitude coordinates
            lats (numpy.ndarray): Array of latitude coordinates

        Returns:
            numpy.ndarray: Array of altitude values with same shape as input coordinates

        Raises:
            ValueError: If input arrays have different shapes
        """
        if lons.shape != lats.shape:
            raise ValueError("lons and lats must have the same shape.")

        return get_altitudes_vect_nb(self.shm_instances['tiles_1d_array'].array,
                                      self.shm_instances['tiles_indices'].array,
                                      self.shm_instances['tiles_nrows'].array,
                                      self.shm_instances['tiles_ncols'].array,
                                      lons.ravel(), lats.ravel()).reshape(lons.shape)

    def close(self):
        """
        Close shared memory blocks.

        This method safely closes all SharedMemoryArray instances.
        It handles potential errors during cleanup to ensure the process doesn't fail.
        """
        for key, obj in self.shm_instances.items():
            if obj is not None:
                try:
                    obj.close()
                    logging.info(f"Shared memory block {key} closed")
                except Exception as e:
                    logging.warning(f"Error closing shared memory block {key}: {e}")
        self.shm_instances = {}

    @property
    def tiles_1d_array(self):
        return self.shm_instances['tiles_1d_array'].array if 'tiles_1d_array' in self.shm_instances else None

    @property
    def tiles_indices_array(self):
        return self.shm_instances['tiles_indices'].array if 'tiles_indices' in self.shm_instances else None

    @property
    def tiles_nrows_array(self):
        return self.shm_instances['tiles_nrows'].array if 'tiles_nrows' in self.shm_instances else None

    @property
    def tiles_ncols_array(self):
        return self.shm_instances['tiles_ncols'].array if 'tiles_ncols' in self.shm_instances else None

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main execution block

def main():
    """Main function handling command line arguments and execution."""
    # Configure logging only when running as main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Create main argument parser
    parser = argparse.ArgumentParser(
        description='Load preprocessed SRTM15 tiles and retrieve altitudes for specified coordinates using shared memory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Load data and keep shared memory available:
    python get_altitudes_dtm_mp.py loader input.zip

  Get altitude using shared memory blocks:
    python get_altitudes_dtm_mp.py user --tiles_1d_array tiles_1d_array_1234 --tiles_indices tiles_indices_1234 --tiles_nrows tiles_nrows_1234 --tiles_ncols tiles_ncols_1234 --grid-size 1000 55.22 -21.40 55.83 -20.86

  Standalone mode with visualization:
    python get_altitudes_dtm_mp.py standalone input.zip --grid-size 1000 55.22 -21.40 55.83 -20.86

Note:
  In user mode, only shared memory block names are required, as metadata (shape and dtype)
  are stored within the shared memory blocks using struct.
        '''
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode to run in')
    parser_load = subparsers.add_parser('loader', help='Run in loader mode. Loads the elevation data from a ZIP file into shared memory.')
    parser_stnd = subparsers.add_parser('standalone', help='Run in standalone mode. Combines the functionality of both loader and user modes and generates a heatmap.')
    parser_user = subparsers.add_parser('user', help='Run in user mode. Uses the shared memory blocks to retrieve altitude for a single coordinate pair.')

    # Parser for loader mode and standalone mode
    parser_load.add_argument('input', type=str, help='Input ZIP file name containing the elevation data.')
    parser_stnd.add_argument('input', type=str, help='Input ZIP file name containing the elevation data.')

    # Parser for user mode and standalone mode
    parser_user.add_argument('lonlat1', nargs=2, type=float, help='corner1: Longitude in decimal degrees (-180 to 180), Latitude in decimal degrees (-90 to 90)')
    parser_stnd.add_argument('lonlat1', nargs=2, type=float, help='corner1: Longitude in decimal degrees (-180 to 180), Latitude in decimal degrees (-90 to 90)')
    
    parser_user.add_argument('lonlat2', nargs=2, type=float, help='corner2: Longitude in decimal degrees (-180 to 180), Latitude in decimal degrees (-90 to 90)')
    parser_stnd.add_argument('lonlat2', nargs=2, type=float, help='corner2: Longitude in decimal degrees (-180 to 180), Latitude in decimal degrees (-90 to 90)')
    
    parser_user.add_argument('--grid-size', type=int, default=1000, help='Number of points per axis for heatmap (default: 1000)')
    parser_stnd.add_argument('--grid-size', type=int, default=1000, help='Number of points per axis for heatmap (default: 1000)')
    
    # Parser for user mode
    for key in ('tiles_1d_array', 'tiles_indices', 'tiles_nrows', 'tiles_ncols'):
        parser_user.add_argument(f'--{key}', type=str, required=True, help=f'Name of the shared memory block for {key}. Example: "tiles_1d_array_1234"')

    args = parser.parse_args()
    
    # Standalone mode & Loader mode: Load data into shared memory and display connection info
    if args.mode == 'loader' or args.mode == 'standalone':
        logging.info(f"Loading DTM file: {args.input}")
        shm_dtm_loader = Shm_Dtm_Loader(args.input)
        logging.info("DTM file loaded successfully.")

    # Loader mode: print information and wait for interruption then exit 
    if args.mode == 'loader':
        # Generate example command for user mode
        logging.info("Examples command for user mode:")
        cmd_shm_params = ' '.join([f"--{key} {name}" for key, name in shm_dtm_loader.shm_names.items()])
        logging.info('python get_altitudes_dtm_mp.py user ' + cmd_shm_params + ' -5.20 48.20 -4.60 48.60')
        logging.info('python get_altitudes_dtm_mp.py user ' + cmd_shm_params + ' -17.00 27.90 -16.11 28.61')
        logging.info('python get_altitudes_dtm_mp.py user ' + cmd_shm_params + ' 55.22 -21.40 55.83 -20.86')
        logging.info('python get_altitudes_dtm_mp.py user ' + cmd_shm_params + ' -180 -90 180 90')
        logging.info('python line_of_sight.py ' + cmd_shm_params + ' --origin -3.2 47.5 30 --target -3.3 47.6 60')
        logging.info("Loader mode completed. Shared memory blocks are ready for use: Press Ctrl+C to exit...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nReceived keyboard interrupt.")
    
    # Standalone mode & User mode: Retrieve shared memory names, Connect to shared memory and Retrieve altitudes
    if args.mode == 'standalone':
        shm_names = shm_dtm_loader.shm_names
    elif args.mode == 'user':
        from multiprocessing import resource_tracker
        resource_tracker.register = lambda name, rtype: None

        # User mode: Connect to shared memory and retrieve altitude
        shm_names = {'tiles_1d_array': args.tiles_1d_array,
                     'tiles_indices': args.tiles_indices,
                     'tiles_nrows': args.tiles_nrows,
                     'tiles_ncols': args.tiles_ncols}

    # Standalone mode & User mode: Connect to shared memory and retrieve altitude
    if args.mode == 'standalone' or args.mode == 'user':
        # Import matplotlib only when needed for standalone and user mode
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        try:
            shm_dtm_user = Shm_Dtm_User(shm_names)
            logging.debug("Shared memory blocks loaded successfully.")
        except Exception as e:
            logging.error(f"Error in mode: {e}")
            sys.exit(1)
        
        for lon, lat in (args.lonlat1, args.lonlat2):
            logging.info(f'Long: {lon:11.6f}°, Lat: {lat:10.6f}°')
            alt = shm_dtm_user.get_altitude(lon, lat)
            logging.info(f'alt: {alt:d}m')
        
        lon_min, lon_max = (args.lonlat1[0], args.lonlat2[0]) if args.lonlat1[0] < args.lonlat2[0] else (args.lonlat2[0], args.lonlat1[0])
        lat_min, lat_max = (args.lonlat1[1], args.lonlat2[1]) if args.lonlat1[1] < args.lonlat2[1] else (args.lonlat2[1], args.lonlat1[1])

        lons = np.linspace(lon_min, lon_max, args.grid_size, dtype=np.float32)
        lats = np.linspace(lat_min, lat_max, args.grid_size, dtype=np.float32)
        lons_mesh, lats_mesh = np.meshgrid(lons, lats, indexing='ij')
        
        logging.debug('shm_dtm_user.get_altitudes(lons_mesh, lats_mesh)')
        alts = shm_dtm_user.get_altitudes(lons_mesh, lats_mesh) # Numba warmpup
        t0 = time.perf_counter()
        alts = shm_dtm_user.get_altitudes(lons_mesh, lats_mesh)
        t1 = time.perf_counter()
        logging.info(f'Time to get altitudes: {t1-t0:.4f} seconds')

        colors = ["aquamarine", "darkgreen", "palegreen", "green", "bisque", "darkgoldenrod", "burlywood", "saddlebrown", "palegoldenrod", "crimson", "salmon", "red"]
        nodes  = list(np.linspace(0, 1, len(colors)+1)[2:])
        colors = ["darkblue"] + colors
        nodes  = [0.0, 1e-6] + nodes
        cmap   = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

        fig, ax = plt.subplots(num='Relief', figsize=(20, 10), dpi=300)
        im = ax.imshow(alts.T, origin='lower',
                     extent=[lon_min, lon_max, lat_min, lat_max], cmap=cmap)
        fig.colorbar(im, ax=ax)
        fig.savefig(f'figure_Relief_{args.grid_size}x{args.grid_size}.png')
        plt.show()
        
        shm_dtm_user.close()

    # Standalone mode & Loader mode: Close loader shared memory
    if args.mode == 'loader' or args.mode == 'standalone':
        shm_dtm_loader.close()

if __name__ == '__main__':
    main()
# ======================================================================
