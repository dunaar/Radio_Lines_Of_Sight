# Line Of Sight

## Overview

The **Line Of Sight (LOS)** project is a Python-based tool designed to calculate intervisibility between geographic points using a Digital Terrain Model (DTM). It determines whether a target point is visible from an origin point, accounting for terrain obstructions, Earth's curvature, and atmospheric refraction. Leveraging high-performance computing techniques such as Numba for JIT compilation and shared memory for efficient DTM access, this project is ideal for applications like telecommunications antenna placement.

## Features

- **Intervisibility Calculation**: Computes whether a target point is visible from an origin point by analyzing terrain data along the LOS.
- **Geophysical Accuracy**: Incorporates Earth's curvature and a 4/3 atmospheric refraction model for realistic radio wave propagation simulation.
- **High Performance**: Utilizes Numba for optimized numerical computations and shared memory for fast DTM data access.
- **Visualization**: Provides Matplotlib-based plotting of LOS profiles, showing terrain and LOS altitudes against distance, longitude, and latitude (optional).
- **Custom Serialization**: Provides utilities for serializing complex Python objects (e.g., NumPy arrays, Numba lists, complex numbers) using `msgspec` and MessagePack.

## Dependencies

- **Python**: Version 3.12 or higher
- **NumPy**: For numerical operations and array handling
- **Numba**: For performance optimization of computational loops
- **msgspec**: For efficient serialization and deserialization
- **netCDF4** (optional): Required only for preprocessing DTM data from NetCDF files into a tiled zip format
- **Matplotlib** (optional): Required only for visualizing LOS and terrain data
- **Built-in Modules**:
  - `struct`: For packing/unpacking complex numbers
  - `argparse`, `logging`, `time`, `collections`, `sys`, `tempfile`, `typing`: Standard Python libraries used for various utilities
- **Local Modules**:
  - `get_altitudes_dtm_mp`: For accessing DTM data in shared memory
  - `transform_coord`: For coordinate transformations (geographic to Cartesian and vice versa)
  - `convert_dtm`: For preprocessing NetCDF DTM files into a tiled zip format

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dunaar/Line_Of_Sight.git
   cd Line_Of_Sight
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install numpy numba msgspec netCDF4 tqdm
   pip install matplotlib  # Optional for test and visualization
   ```

4. **Prepare DTM Data**:
   - **Download NetCDF File**: Obtain a DTM dataset in NetCDF format from a reliable source, such as the SRTM15+ dataset available at [https://topex.ucsd.edu/pub/srtm15_plus/SRTM15_V2.7.nc](https://topex.ucsd.edu/pub/srtm15_plus/SRTM15_V2.7.nc), which provides global elevation and bathymetry data at 15 arc-second resolution.
   - **Convert to Tiled Zip Format**: Use the `convert_dtm.py` script to process the NetCDF file into a compressed ZIP archive containing tiled DTM data and metadata (e.g., `tiles_1d_array`, `tiles_indices`, `tiles_nrows`, `tiles_ncols`). This format optimizes memory usage and enables efficient shared memory loading. Example command:
     ```bash
     python convert_dtm.py SRTM15_V2.7.nc srtm15_tiles_compressed.zip
     ```
     - **Note**: The script splits the DTM into 1° x 1° tiles, applies resampling depending on latitudes, and saves metadata for shared memory access. Default tile sizes and resolutions are used unless specified otherwise.

## Usage

The project offers scripts that can be executed directly or imported as modules into custom Python scripts. Below are instructions for both approaches.

### Running Scripts Directly

#### `get_altitudes_dtm_mp.py`

Manages DTM data loading into shared memory and altitude retrieval, supporting three modes: `loader`, `user`, and `standalone`.

- **Loader Mode**: Loads DTM data into shared memory.
  ```bash
  python get_altitudes_dtm_mp.py loader srtm15_tiles_compressed.zip
  ```

- **User Mode**: Retrieves altitudes using existing shared memory blocks.
  ```bash
  python get_altitudes_dtm_mp.py user \
    --tiles_1d_array tiles_1d_array_1234 \
    --tiles_indices tiles_indices_1234 \
    --tiles_nrows tiles_nrows_1234 \
    --tiles_ncols tiles_ncols_1234 \
    -5.20 48.20 -4.60 48.60
  ```

- **Standalone Mode**: Combines loading and retrieval, with visualization.
  ```bash
  python get_altitudes_dtm_mp.py standalone srtm15_tiles_compressed.zip \
    --grid-size 1000 55.22 -21.40 55.83 -20.86
  ```
#### `line_of_sight.py`

Calculates intervisibility between an origin point and one or more target points using DTM data stored in shared memory.

**Command-Line Example (take care to recopy shared memory names from `get_altitudes_dtm_mp` in `loader` mode)**:
```bash
python line_of_sight.py \
  --tiles_1d_array tiles_1d_array_1234 \
  --tiles_indices tiles_indices_1234 \
  --tiles_nrows tiles_nrows_1234 \
  --tiles_ncols tiles_ncols_1234 \
  --origin -3.2 47.5 30 \
  --target -3.3 47.6 60 \
  --resolution 400
```

**Output**:
- A boolean indicating visibilities (`True` if visible, `False` if obstructed).
- Matplotlib plots of the LOS profile and terrain data (if `matplotlib` is installed).


### Using Scripts as Modules

The functionality of `get_altitudes_dtm_mp.py` and `line_of_sight.py` can be integrated into custom Python scripts by importing their classes and functions. This allows for programmatic control over DTM data loading, altitude retrieval, and intervisibility calculations.

#### Using `get_altitudes_dtm_mp.py` as a Module

This script provides two key classes: `Shm_Dtm_Loader` for loading DTM data into shared memory and `Shm_Dtm_User` for retrieving altitudes from shared memory.

- **Loading DTM Data**:
  Use `Shm_Dtm_Loader` to load preprocessed DTM data from a ZIP file into shared memory. This should typically run in a separate process to keep the data available for other processes.
  ```python
  from get_altitudes_dtm_mp import Shm_Dtm_Loader

  # Load DTM data into shared memory
  loader = Shm_Dtm_Loader('srtm15_tiles_compressed.zip')
  print("Shared memory block names:", loader.shm_names)
  # Keep this process running or manage its lifecycle as needed
  ```

- **Retrieving Altitudes**:
  Use `Shm_Dtm_User` to access altitudes from shared memory blocks created by `Shm_Dtm_Loader`.
  ```python
  from get_altitudes_dtm_mp import Shm_Dtm_User

  # Define shared memory block names (obtained from Shm_Dtm_Loader)
  shm_names = {'tiles_1d_array': 'tiles_1d_array_1234',
               'tiles_indices': 'tiles_indices_1234',
               'tiles_nrows': 'tiles_nrows_1234',
               'tiles_ncols': 'tiles_ncols_1234'}

  # Initialize the user object
  user = Shm_Dtm_User(shm_names)

  # Get altitude for a single point
  altitude = user.get_altitude(-3.2, 47.5)
  print(f"Altitude at (-3.2, 47.5): {altitude} meters")

  # Get altitudes for multiple points
  import numpy as np
  lons = np.array([-3.2, -3.3])
  lats = np.array([47.5, 47.6])
  altitudes = user.get_altitudes(lons, lats)
  print(f"Altitudes: {altitudes}")

  # Clean up when done
  user.close()
  ```

#### Using `line_of_sight.py` as a Module

This script provides the `are_intervisible` function to calculate intervisibility between points using DTM data from shared memory.

- **Calculating Intervisibility**:
  Import `are_intervisible` and use it with a `Shm_Dtm_User` instance to compute visibilities.
  ```python
  from line_of_sight import are_intervisible
  from get_altitudes_dtm_mp import Shm_Dtm_User

  # Define shared memory block names
  shm_names = {'tiles_1d_array': 'tiles_1d_array_1234',
               'tiles_indices': 'tiles_indices_1234',
               'tiles_nrows': 'tiles_nrows_1234',
               'tiles_ncols': 'tiles_ncols_1234'}

  # Initialize DTM user
  user = Shm_Dtm_User(shm_names)

  # Define origin and target points
  origin_lon, origin_lat, origin_alt = -3.2, 47.5, 30
  target_lons = np.array([-3.3])
  target_lats = np.array([47.6])
  target_alts = np.array([60])

  # Calculate intervisibility
  intervisibilities = are_intervisible(user,
                                       origin_lon, origin_lat, origin_alt,
                                       target_lons, target_lats, target_alts,
                                       resolution_m=400.0)

  print(f"Intervisibility: {intervisibilities[0]}")

  # Clean up
  user.close()
  ```

### intervisibility_example.py
```python
# === Built-in ===
import sys
import multiprocessing as mp
import logging
from typing import Dict

# === Third-party ===
import numpy as np

# === Local modules ===
from get_altitudes_dtm_mp import Shm_Dtm_Loader, Shm_Dtm_User
from line_of_sight import are_intervisible

def compute_intervisibility(shm_names: Dict[str, str], num_iterations: int = 5, num_locs: int = 10) -> None:
    """
    Subprocess function to compute intervisibility for random positions using shared memory DTM data.

    Args:
        shm_names (Dict[str, str]): Dictionary containing shared memory block names.
        num_iterations (int): Number of intervisibility calculations to perform.
    """
    # Configure logging for the subprocess
    logging.basicConfig(level=logging.INFO, format='%(processName)18s: %(message)s')
    logger = logging.getLogger(__name__)

    process_name = mp.current_process().name
    if 'ForkPoolWorker' in process_name:
        worker_num = int(process_name.split('-')[-1])
        np.random.seed(worker_num)  

    # Initialize DTM user with shared memory block names
    dtm_user = Shm_Dtm_User(shm_names)
    logger.info("Initialized Shm_Dtm_User in subprocess")

    # Define bounds for random coordinates (example: a region in Brittany, France)
    lon_min, lon_max = -10.0,   +10.0
    lat_min, lat_max = -10.0,   +10.0
    alt_min, alt_max =  20.0, 20000.0  # Altitude range in meters

    for i in range(num_iterations):
        # Generate random origin and target points
        origin_lon = np.random.uniform(lon_min, lon_max)
        origin_lat = np.random.uniform(lat_min, lat_max)
        origin_alt = np.random.uniform(alt_min, alt_max)

        # Prepare arrays for are_intervisible
        target_lons = np.random.uniform(lon_min, lon_max, size=num_locs)
        target_lats = np.random.uniform(lat_min, lat_max, size=num_locs)
        target_alts = np.random.uniform(alt_min, alt_max, size=num_locs)

        # Compute intervisibility
        try:
            visibilities = are_intervisible(dtm_user,
                                            origin_lon, origin_lat, origin_alt,
                                            target_lons, target_lats, target_alts,
                                            resolution_m=400.0)
            logger.info(f"Iteration {i+1}:")
            logger.info(f"Origin: {origin_lon:6.2f}°, {origin_lat:6.2f}°, {origin_alt:8.2f}m")
            logger.info(f"to Targets:")
            for idx in range(num_locs):
                logger.info(f"        {target_lons[idx]:6.2f}°, {target_lats[idx]:6.2f}°, {target_alts[idx]:8.2f}m - Visible: {visibilities[idx]}")
        except Exception as e:
            logger.error(f"Error in intervisibility calculation: {e}")

    # Clean up shared memory access
    dtm_user.close()
    logger.info("Closed Shm_Dtm_User in subprocess")

def main():
    """
    Main function to load DTM data into shared memory and launch a subprocess for intervisibility calculations.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(processName)18s: %(message)s')
    logger = logging.getLogger(__name__)

    # Path to the preprocessed DTM ZIP file
    dtm_zip_path = sys.argv[1] if len(sys.argv) > 1 else "srtm15_tiles_compressed.zip"

    # Initialize and load DTM data into shared memory
    logger.info("Loading DTM data into shared memory")
    dtm_loader = Shm_Dtm_Loader(dtm_zip_path)
    logger.info(f"Shared memory block names: {dtm_loader.shm_names}")

    # Define number of tasks, num_iterations and locations per task
    num_tasks      = 5 * mp.cpu_count()
    num_iterations =  10  # Number of intervisibility calculations per task
    num_locs       = 100  # Number of random locations to test in each task

    # Prepare the arguments for each pool task
    pool_args = [(dtm_loader.shm_names, num_iterations, num_locs) for _ in range(num_tasks)]

    # Launch subprocess for intervisibility calculations
    logger.info("Starting subprocess for intervisibility calculations")
    with mp.Pool(processes=num_tasks) as pool:
        pool.starmap(compute_intervisibility, pool_args)

    # Clean up shared memory
    dtm_loader.close()
    logger.info("Closed Shm_Dtm_Loader and released shared memory")

if __name__ == "__main__":
    main()
```

### Managing Shared Memory Blocks

When using these scripts as modules, proper management of shared memory blocks is crucial to avoid memory leaks or conflicts:

- **Creating Shared Memory Blocks**: Use `Shm_Dtm_Loader` in a separate process or script to load DTM data into shared memory. Ensure this process remains active or is properly terminated when no longer needed.
- **Accessing Shared Memory Blocks**: Use `Shm_Dtm_User` to connect to existing shared memory blocks. Verify that the blocks are not unlinked while in use.
- **Cleaning Up**: Call the `close` method on `Shm_Dtm_Loader` and `Shm_Dtm_User` instances when finished. Avoid unlinking shared memory blocks if they are still required by other processes.

For multi-process workflows, one process can load the data and maintain it in shared memory, while others access it using the shared memory block names provided by `Shm_Dtm_Loader.shm_names`.

## Project Structure

- **`line_of_sight.py`**: Main script for LOS intervisibility calculations and visualization.
- **`np_msgspec_msgpack_utils.py`**: Utility module for serializing/deserializing complex data types.
- **`get_altitudes_dtm_mp.py`**: Module for loading DTM data into shared memory and retrieving altitudes.
- **`transform_coord.py`**: Module for coordinate transformations (geographic to Cartesian and vice versa).
- **`convert_dtm.py`**: Module for preprocessing NetCDF DTM files into a tiled ZIP format.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please adhere to the project's style guidelines:
- Preserve documentation and comments.
- Maintain consistent alignment of colons, commas, and comments.
- Organize imports into `Built-in`, `Third-party`, and `Local modules` sections.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/dunaar/Line_Of_Sight) or contact the author, Pessel Arnaud.
