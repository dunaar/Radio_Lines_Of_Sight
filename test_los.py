# === Built-in ===
import sys
import multiprocessing as mp
import logging
from typing import Dict

mp.set_start_method("spawn", force=True)

# === Third-party ===
import numpy as np

# === Local modules ===
import Line_Of_Sight
from Line_Of_Sight.get_altitudes_dtm_mp import Shm_Dtm_Loader, Shm_Dtm_User
from Line_Of_Sight.line_of_sight import are_intervisible

def compute_intervisibility(shm_names: Dict[str, str], num_iterations: int = 5, num_locs: int = 10) -> None:
    """
    Subprocess function to compute intervisibility for random positions using shared memory DTM data.

    Args:
        shm_names (Dict[str, str]): Dictionary containing shared memory block names.
        num_iterations (int): Number of intervisibility calculations to perform.
    """
    # Configure logging for the subprocess
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(processName)s-%(module)s-%(funcName)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
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
            logger.info(f"Starting iteration {i+1}:")
            visibilities = are_intervisible(dtm_user,
                                            origin_lon, origin_lat, origin_alt,
                                            target_lons, target_lats, target_alts,
                                            resolution_m=400.0)
            logger.info(f"Origin: {origin_lon:6.2f}°, {origin_lat:6.2f}°, {origin_alt:8.2f}m")
            logger.info(f"to Targets:")
            for idx in range(num_locs):
                logger.info(f"        {target_lons[idx]:6.2f}°, {target_lats[idx]:6.2f}°, {target_alts[idx]:8.2f}m - Visible: {visibilities[idx]}")
        except Exception as e:
            logger.error(f"Error in intervisibility calculation: {e}")
            raise

    # Clean up shared memory access
    dtm_user.close()
    logger.info("Closed Shm_Dtm_User in subprocess")

def main():
    """
    Main function to load DTM data into shared memory and launch a subprocess for intervisibility calculations.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(processName)s-%(module)s-%(funcName)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # Path to the preprocessed DTM ZIP file
    dtm_zip_path = sys.argv[1] if len(sys.argv) > 1 else "srtm15_tiles_compressed.zip"

    # Initialize and load DTM data into shared memory
    logger.info("Loading DTM data into shared memory")
    dtm_loader = Shm_Dtm_Loader(dtm_zip_path)
    logger.info(f"Shared memory block names: {dtm_loader.shm_names}")

    # %% Compute intervisibility for a single iteration in the main process
    compute_intervisibility(dtm_loader.shm_names, num_iterations=1, num_locs=10)

    # %% Compute intervisibility for a single iteration in a subprocess
    subproc = mp.Process(
        target=compute_intervisibility,
        args=(dtm_loader.shm_names, 1, 10),
        name="Intervis_subprocess"
    )
    subproc.start()
    #    Optionally, wait for process to finish:
    subproc.join()

    # %% Define number of tasks, num_iterations and locations per task
    num_tasks      = 2 * mp.cpu_count()
    num_iterations =  10  # Number of intervisibility calculations per task
    num_locs       = 100  # Number of random locations to test in each task

    if num_tasks > 0:
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
