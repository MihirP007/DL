# baseline_fixed_time.py

import os
import sys
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np

# --- Configuration ---
SUMO_CFG_FILE = "intersection.sumocfg" # Your SUMO config file
NUM_SECONDS = 3600 # Simulation duration (must match DRL test duration)
OUTPUT_PREFIX = "outputs/fixed_time_" # Prefix for output files
TRIPINFO_FILE = OUTPUT_PREFIX + "tripinfo.xml"
SUMMARY_FILE = OUTPUT_PREFIX + "summary.xml"

# Ensure SUMO_HOME is set (same logic as in SumoEnvironment)
# ... (Include the SUMO_HOME check block here) ...
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    # Check if sumo executable exists
    sumo_path = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    if not os.path.exists(sumo_path) and not os.path.exists(sumo_path + ".exe"):
         sys.exit(f"ERROR: sumo executable not found in SUMO_HOME/bin: {os.path.join(os.environ['SUMO_HOME'], 'bin')}")
else:
    # ... (Add the auto-detection logic for SUMO_HOME if needed) ...
    sys.exit("Please declare environment variable 'SUMO_HOME'.")


def run_sumo_simulation():
    """Runs the SUMO simulation with fixed-time control and outputs."""
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    output_dir = os.path.dirname(TRIPINFO_FILE)
    os.makedirs(output_dir, exist_ok=True) # Create output directory

    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CFG_FILE,
        "--duration-log.disable", "true",
        "--no-step-log", "true",
        "--time-to-teleport", "-1", # Disable teleporting for realistic waiting times
        # "--seed", "42", # Use a fixed seed for reproducibility if desired
        "--end", str(NUM_SECONDS),
        "--tripinfo-output", TRIPINFO_FILE,
        "--summary-output", SUMMARY_FILE,
        "--statistic-output", OUTPUT_PREFIX + "stats.xml", # Optional detailed stats
        "--queue-output", OUTPUT_PREFIX + "queue.xml" # Optional queue stats
    ]

    print(f"--- Running Fixed-Time SUMO Simulation ---")
    print(f"Command: {' '.join(sumo_cmd)}")
    start_time = time.time()
    try:
        process = subprocess.Popen(sumo_cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.wait() # Wait for SUMO to finish
        end_time = time.time()
        print(f"--- SUMO Simulation Finished (Duration: {end_time - start_time:.2f}s) ---")
        if process.returncode != 0:
             print(f"WARN: SUMO process exited with code {process.returncode}")
    except FileNotFoundError:
         print(f"ERROR: SUMO executable not found at '{sumo_binary}'. Check SUMO_HOME and installation.")
         sys.exit(1)
    except Exception as e:
         print(f"ERROR: An error occurred while running SUMO: {e}")
         sys.exit(1)


def parse_tripinfo(filename):
    """Parses the tripinfo XML file to extract key metrics."""
    if not os.path.exists(filename):
        print(f"ERROR: Tripinfo file not found: {filename}")
        return None

    print(f"--- Parsing {filename} ---")
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"ERROR: Failed to parse XML file {filename}: {e}")
        return None

    total_vehicles = 0
    total_wait_time = 0.0
    total_travel_time = 0.0
    total_time_loss = 0.0 # Includes waiting and deceleration delays

    wait_times = []
    travel_times = []

    for trip in root.findall('tripinfo'):
        total_vehicles += 1
        try:
            wait = float(trip.get('waitingTime', 0.0))
            travel = float(trip.get('duration', 0.0))
            loss = float(trip.get('timeLoss', 0.0))

            total_wait_time += wait
            total_travel_time += travel
            total_time_loss += loss

            wait_times.append(wait)
            travel_times.append(travel)
        except (TypeError, ValueError) as e:
             print(f"WARN: Could not parse attributes for trip {trip.get('id')}: {e}")


    if total_vehicles == 0:
        print("WARN: No vehicles completed trips in the simulation.")
        return {
            'total_vehicles': 0, 'avg_wait_time': 0.0, 'avg_travel_time': 0.0,
             'total_wait_time': 0.0, 'total_travel_time': 0.0, 'avg_time_loss': 0.0
        }

    metrics = {
        'total_vehicles': total_vehicles,
        'avg_wait_time': total_wait_time / total_vehicles,
        'avg_travel_time': total_travel_time / total_vehicles,
        'total_wait_time': total_wait_time,
        'total_travel_time': total_travel_time,
        'avg_time_loss': total_time_loss / total_vehicles,
        # Optionally calculate standard deviations or percentiles
        'std_dev_wait_time': np.std(wait_times) if wait_times else 0.0,
        'p95_wait_time': np.percentile(wait_times, 95) if wait_times else 0.0,
    }
    return metrics

if __name__ == "__main__":
    run_sumo_simulation()
    metrics = parse_tripinfo(TRIPINFO_FILE)

    if metrics:
        print("\n--- Fixed-Time Simulation Metrics ---")
        print(f"Total Vehicles Completed: {metrics['total_vehicles']}")
        print(f"Average Wait Time (s/veh): {metrics['avg_wait_time']:.2f}")
        print(f"Average Travel Time (s/veh): {metrics['avg_travel_time']:.2f}")
        print(f"Average Time Loss (s/veh): {metrics['avg_time_loss']:.2f}")
        print(f"Total Cumulative Wait Time (s): {metrics['total_wait_time']:.2f}")
        print(f"Wait Time Std Dev (s): {metrics['std_dev_wait_time']:.2f}")
        print(f"Wait Time 95th Percentile (s): {metrics['p95_wait_time']:.2f}")
        print("------------------------------------")