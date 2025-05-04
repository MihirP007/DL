# test_rl.py
import gymnasium as gym
from stable_baselines3 import PPO
import time
import os
import random
import sys
import traceback
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

# --- Import Custom Environment ---
try:
    from SumoEnvironment import SumoEnvironment, DEFAULT_PORT
except ImportError:
    print("ERROR: Failed to import SumoEnvironment.")
    print("Make sure SumoEnvironment.py is in the same directory as test_rl.py")
    sys.exit(1)

# --- Configuration ---
NET_FILE = "my_intersection.net.xml"
ROUTE_FILE = "random_traffic.rou.xml"
SUMOCFG_FILE = "intersection.sumocfg"
TLS_ID = "center"
NUM_SECONDS = 3600 # Duration for test run (match baseline)
DELTA_TIME = 5
MIN_GREEN = 10
MAX_GREEN = 60
YELLOW_TIME = 3
OBSERVATION_LANES = [ # Make sure this is correct for your .net.xml!
    'edge_N_in_0', 'edge_N_in_1', 'edge_S_in_0', 'edge_S_in_1',
    'edge_E_in_0', 'edge_E_in_1', 'edge_W_in_0', 'edge_W_in_1']
REWARD_METRIC = 'wait_time' # Usually doesn't affect test execution logic

# --- Output Files ---
OUTPUT_PREFIX_DRL = "outputs/drl_"
TRIPINFO_FILE_DRL = OUTPUT_PREFIX_DRL + "tripinfo.xml"
SUMMARY_FILE_DRL = OUTPUT_PREFIX_DRL + "summary.xml"
# Add other output files if needed
# STATS_FILE_DRL = OUTPUT_PREFIX_DRL + "stats.xml"
# QUEUE_FILE_DRL = OUTPUT_PREFIX_DRL + "queue.xml"

# --- Load Model ---
try:
    model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
    if not model_dirs: raise FileNotFoundError("No subdirectories found in 'models'")
    # --- >>>>> *Optionally* Manually set your specific model folder here <<<<< ---
    # model_dir = os.path.join("models", "PPO_MlpPolicy_YYYYMMDD-HHMMSS")
    # --- Or use the latest ---
    model_dir = max([os.path.join("models", d) for d in model_dirs], key=os.path.getmtime)
    print(f"--- Automatically selecting latest model directory: {model_dir} ---")
except FileNotFoundError as e:
     print(f"ERROR finding model directory: {e}"); sys.exit(1)
except Exception as e:
     print(f"ERROR accessing models directory: {e}"); sys.exit(1)

MODEL_VARIANT = os.path.join("best_model", "best_model") # Or "final_model"
MODEL_LOAD_PATH = os.path.join(model_dir, MODEL_VARIANT)
model_zip_path = MODEL_LOAD_PATH + ".zip"
if not os.path.exists(model_zip_path): print(f"ERROR: Model file not found: {model_zip_path}"); sys.exit(1)

# --- Test Seed ---
TEST_SEED = 42 # Use fixed seed for comparison

# --- SUMO Output Options ---
# These will be passed to SumoEnvironment via the sumo_options parameter
sumo_output_options = [
    "--tripinfo-output", TRIPINFO_FILE_DRL,
    "--summary-output", SUMMARY_FILE_DRL,
    # Uncomment to add more detailed outputs
    # "--statistic-output", STATS_FILE_DRL,
    # "--queue-output", QUEUE_FILE_DRL,
]

# --- Parsing Function ---
def parse_tripinfo(filename):
    """Parses the tripinfo XML file to extract key metrics."""
    if not os.path.exists(filename): print(f"ERROR: Tripinfo file not found: {filename}"); return None
    print(f"--- Parsing {filename} ---")
    try: tree = ET.parse(filename); root = tree.getroot()
    except ET.ParseError as e: print(f"ERROR: Failed to parse XML file {filename}: {e}"); return None
    total_vehicles = 0; total_wait_time = 0.0; total_travel_time = 0.0; total_time_loss = 0.0
    wait_times = []; travel_times = []
    for trip in root.findall('tripinfo'):
        total_vehicles += 1
        try:
            wait = float(trip.get('waitingTime', 0.0)); travel = float(trip.get('duration', 0.0)); loss = float(trip.get('timeLoss', 0.0))
            total_wait_time += wait; total_travel_time += travel; total_time_loss += loss
            wait_times.append(wait); travel_times.append(travel)
        except (TypeError, ValueError) as e: print(f"WARN: Could not parse attributes for trip {trip.get('id')}: {e}")
    if total_vehicles == 0: print("WARN: No vehicles completed trips in the simulation."); return {'total_vehicles': 0, 'avg_wait_time': 0.0, 'avg_travel_time': 0.0, 'total_wait_time': 0.0, 'total_travel_time': 0.0, 'avg_time_loss': 0.0}
    metrics = {'total_vehicles': total_vehicles, 'avg_wait_time': total_wait_time / total_vehicles, 'avg_travel_time': total_travel_time / total_vehicles, 'total_wait_time': total_wait_time, 'total_travel_time': total_travel_time, 'avg_time_loss': total_time_loss / total_vehicles, 'std_dev_wait_time': np.std(wait_times) if wait_times else 0.0, 'p95_wait_time': np.percentile(wait_times, 95) if wait_times else 0.0,}
    return metrics

# --- Main Execution ---
if __name__ == "__main__":

    model = None
    test_env = None
    os.makedirs(os.path.dirname(TRIPINFO_FILE_DRL), exist_ok=True) # Ensure output dir exists

    try:
        # --- Load Model ---
        print(f"--- Loading Model: {model_zip_path} ---")
        model = PPO.load(MODEL_LOAD_PATH, device='cpu')

        # --- Set up Test Environment ---
        print("--- Setting up Test Environment ---")
        test_env = SumoEnvironment(
            net_file=NET_FILE, route_file=ROUTE_FILE, sumocfg_file=SUMOCFG_FILE,
            use_gui=True, # <<< GUI OFF for output generation & fair comparison
            num_seconds=NUM_SECONDS, tls_id=TLS_ID,
            min_green=MIN_GREEN, max_green=MAX_GREEN, yellow_time=YELLOW_TIME, delta_time=DELTA_TIME,
            observation_lanes=OBSERVATION_LANES, reward_metric='wait_time',
            port=DEFAULT_PORT, seed=TEST_SEED,
            sumo_options=sumo_output_options # <<< Pass the output options here
        )
        print(f"--- Environment created successfully (Port: {DEFAULT_PORT}, Seed: {TEST_SEED}) ---")

        # --- Reset Environment ---
        print("--- Resetting Environment (Starting SUMO) ---")
        obs, info = test_env.reset()
        print("--- Environment reset successfully ---")

    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: An exception occurred during Model Loading or Environment Setup/Reset!")
        print(f"       Error Type: {type(e).__name__}"); print(f"       Error Details: {e}")
        print("----------------------------------------------------"); traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        if test_env is not None: print("--- Attempting to close environment after setup error ---"); test_env.close()
        sys.exit(1)

    # --- Run Simulation ---
    print(f"--- Starting DRL Test Simulation Loop ---")
    done = False; total_reward = 0.0; step_count = 0; start_sim_time = time.time()
    ep_cumulative_wait_time = 0.0 # Track metrics from info dict

    try:
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated; total_reward += reward; step_count += 1
            ep_cumulative_wait_time = info.get('cumulative_wait_time', ep_cumulative_wait_time) # Get from info dict
            if done: print(f"--- Simulation Episode Finished (Sim Step: {info.get('step', 'N/A')}) ---");
            if info.get('error', None): print(f"Error reported by environment: {info['error']}")

    except KeyboardInterrupt: print("\n--- Test Simulation Interrupted by User ---")
    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: An exception occurred during the simulation loop (predict/step)!")
        print(f"       Error Type: {type(e).__name__}"); print(f"       Error Details: {e}")
        print("----------------------------------------------------"); traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    finally:
        # --- Cleanup ---
        print("--- Cleaning up: Closing Environment ---")
        if test_env is not None: test_env.close()
        else: print("   (Environment was not successfully created)")
        end_sim_time = time.time()

        # --- PARSE DRL OUTPUTS ---
        print("\n--- DRL Simulation Run Finished ---")
        # Give SUMO a moment to write output files before parsing
        time.sleep(1)
        drl_metrics = parse_tripinfo(TRIPINFO_FILE_DRL)

        # --- Print DRL Results ---
        print("\n--- DRL Agent Simulation Metrics ---")
        # ... (Print results as before) ...
        print(f"Test Seed Used: {TEST_SEED}")
        print(f"Model Tested: {model_zip_path}")
        print(f"Total Steps Taken by Agent: {step_count}")
        print(f"Total Reward Accumulated: {total_reward:.2f}")
        if drl_metrics:
            print(f"Total Vehicles Completed: {drl_metrics['total_vehicles']}")
            print(f"Average Wait Time (s/veh): {drl_metrics['avg_wait_time']:.2f} (Parsed from XML)")
            print(f"Average Travel Time (s/veh): {drl_metrics['avg_travel_time']:.2f}")
            print(f"Average Time Loss (s/veh): {drl_metrics['avg_time_loss']:.2f}")
            print(f"Total Cumulative Wait Time (s): {drl_metrics['total_wait_time']:.2f} (Parsed from XML)")
            print(f"Wait Time Std Dev (s): {drl_metrics['std_dev_wait_time']:.2f}")
            print(f"Wait Time 95th Percentile (s): {drl_metrics['p95_wait_time']:.2f}")
        else: print(f"WARN: Could not parse DRL tripinfo output from {TRIPINFO_FILE_DRL}")
        print(f"Wall Clock Time for Simulation Loop: {end_sim_time - start_sim_time:.2f} seconds")
        print("------------------------------------")
        print("--- DRL Test Script Finished ---")