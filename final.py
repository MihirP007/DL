# final_compare.py (Indentation on Line 155 is Correct as provided)

import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO # Needed to LOAD the LSTM model correctly
import time
import os
import random
import sys
import traceback
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
# import threading # Removed for sequential GUI runs
import pandas as pd

# --- Import Custom Environment ---
try:
    # Assuming SumoEnvironment is in the same directory or Python path
    from SumoEnvironment import SumoEnvironment, DEFAULT_PORT
except ImportError:
    print("ERROR: Failed to import SumoEnvironment. Make sure SumoEnvironment.py is accessible.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR importing SumoEnvironment: {e}")
    sys.exit(1)


# --- ********* CONTROL GUI DISPLAY ********* ---
USE_GUI = True # Set to True to run with GUI, False for headless comparison
# --- *************************************** ---

# --- Configuration ---
NET_FILE="my_intersection.net.xml" # Make sure this file exists
ROUTE_FILE="random_traffic.rou.xml" # Make sure this file exists
SUMOCFG_FILE="intersection.sumocfg" # Make sure this file exists and points to NET_FILE and ROUTE_FILE
TLS_ID="center" # Verify this is the correct TLS ID in your .net.xml file
NUM_SECONDS_MAX = 7200 # Max simulation time (2 hours)
DELTA_TIME=5       # Seconds between DRL agent actions
MIN_GREEN=10       # Minimum green phase duration (seconds)
MAX_GREEN=60       # Maximum green phase duration (seconds)
YELLOW_TIME=3      # Yellow phase duration (seconds)
# Verify these lane IDs match your .net.xml file *exactly*
OBSERVATION_LANES = ['edge_N_in_0','edge_N_in_1','edge_S_in_0','edge_S_in_1','edge_E_in_0','edge_E_in_1','edge_W_in_0','edge_W_in_1']

# --- Output Files ---
OUTPUT_DIR = "outputs"
OUTPUT_PREFIX_DRL_MLP = os.path.join(OUTPUT_DIR, "final_drl_mlp_")
OUTPUT_PREFIX_LSTM = os.path.join(OUTPUT_DIR, "final_drl_lstm_")
OUTPUT_PREFIX_BASE = os.path.join(OUTPUT_DIR, "final_fixed_")
OUTPUT_PREFIX_RULE = os.path.join(OUTPUT_DIR, "final_rule_")
TRIPINFO_FILE_DRL_MLP = OUTPUT_PREFIX_DRL_MLP + "tripinfo.xml"
SUMMARY_FILE_DRL_MLP = OUTPUT_PREFIX_DRL_MLP + "summary.xml"
TRIPINFO_FILE_LSTM = OUTPUT_PREFIX_LSTM + "tripinfo.xml"
SUMMARY_FILE_LSTM = OUTPUT_PREFIX_LSTM + "summary.xml"
TRIPINFO_FILE_BASE = OUTPUT_PREFIX_BASE + "tripinfo.xml"
SUMMARY_FILE_BASE = OUTPUT_PREFIX_BASE + "summary.xml"
TRIPINFO_FILE_RULE = OUTPUT_PREFIX_RULE + "tripinfo.xml"
SUMMARY_FILE_RULE = OUTPUT_PREFIX_RULE + "summary.xml"

# --- DRL Model Paths ---
mlp_model_load_path = ""
mlp_model_zip_path = ""
lstm_model_load_path = ""
lstm_model_zip_path = ""

# Find latest MLP/Custom model
print("--- Locating MLP/Custom DRL Model ---")
try:
    models_dir = "models"
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found.")
    # Look for directories containing MlpPolicy, CustomNN, or ComplexNN in their name
    mlp_model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and ("MlpPolicy" in d or "CustomNN" in d or "ComplexNN" in d)]
    if not mlp_model_dirs:
        raise FileNotFoundError("No MLP/Custom model directories (containing 'MlpPolicy', 'CustomNN', or 'ComplexNN') found in 'models/'")
    # Get the full path and sort by modification time to find the latest
    mlp_model_dir = max([os.path.join(models_dir, d) for d in mlp_model_dirs], key=os.path.getmtime)
    print(f"Using MLP/Custom model directory: {mlp_model_dir}")
    # Define the path relative to the chosen directory
    MLP_MODEL_VARIANT = os.path.join("best_model", "best_model") # Expects model saved as 'best_model/best_model.zip'
    MLP_MODEL_LOAD_PATH = os.path.join(mlp_model_dir, MLP_MODEL_VARIANT) # Path without .zip for loading
    mlp_model_zip_path = MLP_MODEL_LOAD_PATH + ".zip" # Full path to the zip file
    if not os.path.exists(mlp_model_zip_path):
        raise FileNotFoundError(f"MLP/Custom model file not found: {mlp_model_zip_path}")
    print(f"Found MLP/Custom model: {mlp_model_zip_path}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Ensure an MLP/Custom DRL model (e.g., from train_*.py) exists in the 'models' directory.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR finding MLP/Custom model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Find latest LSTM model
print("\n--- Locating LSTM DRL Model ---")
try:
    models_dir = "models"
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found.")
    # Look for directories containing MlpLstmPolicy in their name
    lstm_model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and "MlpLstmPolicy" in d]
    if not lstm_model_dirs:
        raise FileNotFoundError("No MlpLstmPolicy model directories found in 'models/'. Run train_lstm.py.")
    # Get the full path and sort by modification time
    lstm_model_dir = max([os.path.join(models_dir, d) for d in lstm_model_dirs], key=os.path.getmtime)
    print(f"Using LSTM model directory: {lstm_model_dir}")
    # Define the path relative to the chosen directory
    LSTM_MODEL_VARIANT = os.path.join("best_model", "best_model") # Expects model saved as 'best_model/best_model.zip'
    LSTM_MODEL_LOAD_PATH = os.path.join(lstm_model_dir, LSTM_MODEL_VARIANT) # Path without .zip for loading
    lstm_model_zip_path = LSTM_MODEL_LOAD_PATH + ".zip" # Full path to the zip file
    if not os.path.exists(lstm_model_zip_path):
        raise FileNotFoundError(f"LSTM model file not found: {lstm_model_zip_path}")
    print(f"Found LSTM model: {lstm_model_zip_path}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Ensure an LSTM DRL model (from train_lstm.py) exists in the 'models' directory.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR finding LSTM model: {e}")
    traceback.print_exc(); sys.exit(1)

# --- Test Seed ---
TEST_SEED = 42 # Use a fixed seed for reproducibility

# --- SUMO Environment Setup ---
SUMO_BINARY = ""
SUMO_GUI_BINARY = ""
sumo_home_path = ""

# 1. Check SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    sumo_home_path = os.environ['SUMO_HOME']
    print(f"Found SUMO_HOME: {sumo_home_path}")
else:
    # 2. If SUMO_HOME is not set, try common installation paths
    print("SUMO_HOME environment variable not set. Searching common paths...")
    potential_paths = [
        "C:\\Program Files (x86)\\Eclipse\\Sumo",
        "C:\\Program Files\\Eclipse\\Sumo",
        "/usr/share/sumo",
        "/usr/local/opt/sumo/share/sumo" # Common path for Homebrew on macOS
    ]
    for path in potential_paths:
        if os.path.isdir(path):
            print(f"Found potential SUMO installation at: {path}")
            sumo_home_path = path
            break

if not sumo_home_path:
    sys.exit("ERROR: Could not find SUMO installation. Please set the SUMO_HOME environment variable.")

# Construct paths to tools and binaries
tools_path = os.path.join(sumo_home_path, 'tools')
sys.path.append(tools_path)
bin_path = os.path.join(sumo_home_path, 'bin')

# Define base names and add .exe for Windows
SUMO_BINARY_base = os.path.join(bin_path, 'sumo')
SUMO_GUI_BINARY_base = os.path.join(bin_path, 'sumo-gui')
if sys.platform.startswith("win"):
    SUMO_BINARY_base += ".exe"
    SUMO_GUI_BINARY_base += ".exe"

# Check if executables exist
if os.path.exists(SUMO_BINARY_base):
    SUMO_BINARY = SUMO_BINARY_base
    print(f"Found SUMO CLI: {SUMO_BINARY}")
else:
    print(f"WARN: sumo executable not found at expected path {SUMO_BINARY_base}")

if os.path.exists(SUMO_GUI_BINARY_base):
    SUMO_GUI_BINARY = SUMO_GUI_BINARY_base
    print(f"Found SUMO GUI: {SUMO_GUI_BINARY}")
else:
    print(f"WARN: sumo-gui executable not found at expected path {SUMO_GUI_BINARY_base}")

# Check usability based on USE_GUI flag
if USE_GUI and not SUMO_GUI_BINARY:
    sys.exit("ERROR: USE_GUI is set to True, but the sumo-gui executable was not found.")
if not USE_GUI and not SUMO_BINARY:
    sys.exit("ERROR: USE_GUI is set to False, but the sumo executable was not found.")

# Try importing traci AFTER adding tools path
try:
    import traci
    print("Successfully imported traci.")
except ImportError:
    sys.exit(f"ERROR: Failed to import traci. Ensure '{tools_path}' is correct and contains the traci module.")
except Exception as e:
    sys.exit(f"ERROR importing traci: {e}")


# --- Parsing Functions ---
def parse_tripinfo(filename):
    """Parses SUMO tripinfo XML and calculates key metrics."""
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
    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        print(f"ERROR: Tripinfo file not found during parsing: {filename}")
        return None

    total_vehicles = 0
    total_wait_time = 0.0
    total_travel_time = 0.0
    total_time_loss = 0.0
    wait_times = []
    travel_times = []
    sim_duration = 0.0 # Initialize

    trips = root.findall('tripinfo')
    if not trips:
        print(f"WARN: No tripinfo elements found in {filename}.")
        # Return default metrics if no trips completed
        return {
            'total_vehicles': 0, 'avg_wait_time': 0.0, 'avg_travel_time': 0.0,
            'total_wait_time': 0.0, 'total_travel_time': 0.0, 'avg_time_loss': 0.0,
            'std_dev_wait_time': 0.0, 'p95_wait_time': 0.0, 'sim_duration': 0.0
        }

    for trip in trips: # Indent 1
        total_vehicles += 1
        wait = 0.0
        travel = 0.0
        loss = 0.0 # Indent 2
        try: # Indent 2
            wait_str = trip.get('waitingTime')
            travel_str = trip.get('duration')
            loss_str = trip.get('timeLoss')
            arrival_str = trip.get('arrival') # Get arrival time for duration calculation

            wait = float(wait_str if wait_str is not None else 0.0)
            travel = float(travel_str if travel_str is not None else 0.0)
            loss = float(loss_str if loss_str is not None else 0.0)
            if arrival_str is not None:
                sim_duration = max(sim_duration, float(arrival_str)) # Track latest arrival time

        except (TypeError, ValueError) as e: # Indent 2
            trip_id = trip.get('id', '[unknown ID]')
            print(f"WARN: Could not parse attributes for trip {trip_id}: {e}. Attributes: wait='{wait_str}', travel='{travel_str}', loss='{loss_str}', arrival='{arrival_str}'") # Indent 3

        total_wait_time += wait
        total_travel_time += travel
        total_time_loss += loss
        wait_times.append(wait)
        travel_times.append(travel) # Indent 2

    if total_vehicles == 0: # Should be handled by the 'if not trips:' check, but redundant safety
        print("WARN: No vehicles completed trips successfully after parsing.")
        # Use the initialized zero/default values
        metrics = {
            'total_vehicles': 0, 'avg_wait_time': 0.0, 'avg_travel_time': 0.0,
            'total_wait_time': 0.0, 'total_travel_time': 0.0, 'avg_time_loss': 0.0,
            'std_dev_wait_time': 0.0, 'p95_wait_time': 0.0, 'sim_duration': sim_duration # Keep tracked max arrival
        }
    else:
        metrics = {
            'total_vehicles': total_vehicles,
            'avg_wait_time': total_wait_time / total_vehicles,
            'avg_travel_time': total_travel_time / total_vehicles,
            'total_wait_time': total_wait_time,
            'total_travel_time': total_travel_time,
            'avg_time_loss': total_time_loss / total_vehicles,
            'std_dev_wait_time': np.std(wait_times) if wait_times else 0.0,
            'p95_wait_time': np.percentile(wait_times, 95) if wait_times else 0.0,
            'sim_duration': sim_duration # Use max arrival time found
        }

    print(f"Parsed {total_vehicles} vehicles. Avg Wait: {metrics['avg_wait_time']:.2f}s, Avg Travel: {metrics['avg_travel_time']:.2f}s")
    return metrics

def parse_summary(filename):
    """Parses SUMO summary XML to get the simulation end time."""
    if not os.path.exists(filename):
        print(f"ERROR: Summary file not found: {filename}")
        return None
    print(f"--- Parsing {filename} ---")
    end_time = None
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        # Method 1: Find the last <step> element's time attribute
        steps = root.findall('step')
        if steps:
            last_step_time_str = steps[-1].get('time')
            if last_step_time_str is not None:
                end_time = float(last_step_time_str)
                print(f"Found end time from last step: {end_time}")

        # Method 2: If no steps or time attribute, check performance duration (less common for summary)
        if end_time is None:
            perf = root.find('performance')
            if perf is not None and 'duration' in perf.attrib:
                duration_str = perf.get('duration')
                if duration_str is not None:
                    end_time = float(duration_str)
                    print(f"Found end time from performance duration: {end_time}")

    except ET.ParseError as e:
        print(f"ERROR: Failed to parse XML summary file {filename}: {e}")
        return None
    except (TypeError, ValueError, IndexError) as e:
        print(f"ERROR: Could not extract simulation duration from {filename}: {e}")
        # traceback.print_exc() # Uncomment for detailed debugging if needed
        return None
    except Exception as e_gen: # Catch any other unexpected errors
        print(f"ERROR: Unexpected error parsing summary {filename}: {e_gen}")
        return None

    if end_time is None:
        print(f"WARN: Could not determine simulation end time from {filename}.")
    return end_time


# --- Function to Run DRL (MLP/Custom) ---
def run_drl_mlp(model_load_path_no_zip, tripinfo_path, summary_path, port, use_gui_flag):
    """Runs the simulation using a loaded PPO (MLP/Custom) model."""
    print(f"\n--- Running DRL (MLP/Custom) Simulation (Port: {port}, GUI: {use_gui_flag}) ---")
    model = None
    env = None
    metrics = None
    wall_time = 0.0
    sim_end_step = 0 # Track internal step count as fallback
    output_options = ["--tripinfo-output", tripinfo_path, "--summary-output", summary_path]
    start_run_time = time.time()
    try:
        # Load model (expects path *without* .zip)
        model = PPO.load(model_load_path_no_zip, device='cpu')
        print(f"Model {model_load_path_no_zip}.zip loaded successfully.")

        # Create environment instance
        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            sumocfg_file=SUMOCFG_FILE,
            use_gui=use_gui_flag,
            num_seconds=NUM_SECONDS_MAX,
            tls_id=TLS_ID,
            min_green=MIN_GREEN,
            max_green=MAX_GREEN,
            yellow_time=YELLOW_TIME,
            delta_time=DELTA_TIME,
            observation_lanes=OBSERVATION_LANES,
            reward_metric='wait_time', # Ensure this matches training
            port=port,
            seed=TEST_SEED,
            sumo_options=output_options # Pass output file options to SUMO
        )
        print("SUMO Environment created.")

        obs, info = env.reset()
        done = False
        step_count = 0
        print("Starting simulation loop...")
        while not done: # Indent Level 2
            action, _ = model.predict(obs, deterministic=True) # Indent Level 3
            obs, reward, terminated, truncated, info = env.step(action) # Indent Level 3
            done = terminated or truncated # Indent Level 3
            sim_end_step = info.get('step', sim_end_step) # Get current simulation step from info # Indent Level 3
            step_count += 1
            if step_count % 100 == 0: # Print progress periodically
                 print(f"  MLP Step: {step_count}, Sim Time: {sim_end_step * DELTA_TIME:.0f}s", end='\r')

        # This block executes *after* the loop finishes
        # Correctly indented at Level 2
        if done: # Indent Level 2
            print(f"\nDRL (MLP) simulation finished. Reason: {info.get('termination_reason', 'Unknown')}, Total Steps: {step_count}")

    except FileNotFoundError as e:
        print(f"ERROR during DRL (MLP) run - File Not Found: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"ERROR during DRL (MLP) run: {e}")
        traceback.print_exc()
    finally:
        if env:
            print("Closing SUMO Environment...")
            env.close()
            wall_time = time.time() - start_run_time
            print(f"DRL (MLP) Wall Clock Time: {wall_time:.2f}s")
            time.sleep(1) # Allow files to close properly
            # Parse results *after* closing env
            metrics = parse_tripinfo(tripinfo_path)
            summary_end_time = parse_summary(summary_path)
            # Update duration from summary if available and seems valid
            if summary_end_time is not None and metrics is not None:
                 # Use summary time if it's greater than tripinfo time (more reliable)
                 if metrics.get('sim_duration', 0) < summary_end_time:
                     metrics['sim_duration'] = summary_end_time
                 elif metrics.get('sim_duration') is None: # If tripinfo parsing failed for duration
                      metrics['sim_duration'] = summary_end_time

            # Fallback if summary parsing failed or no vehicles arrived
            elif metrics is not None and metrics.get('sim_duration') == 0:
                 # Estimate duration from internal steps if possible
                 # This might be slightly off if delta_time isn't perfectly constant
                 estimated_duration = sim_end_step * DELTA_TIME
                 metrics['sim_duration'] = estimated_duration
                 print(f"WARN: Using estimated sim duration from steps: {estimated_duration:.2f}s")

        else: # Handle case where env creation failed
            wall_time = time.time() - start_run_time
            print(f"DRL (MLP) run aborted early. Wall Clock Time: {wall_time:.2f}s")

    return metrics, wall_time

# --- Function to Run DRL (LSTM) ---
def run_drl_lstm(model_load_path_no_zip, tripinfo_path, summary_path, port, use_gui_flag):
    """Runs the simulation using a loaded RecurrentPPO (LSTM) model."""
    print(f"\n--- Running DRL (LSTM) Simulation (Port: {port}, GUI: {use_gui_flag}) ---")
    model = None
    env = None
    metrics = None
    wall_time = 0.0
    sim_end_step = 0 # Track internal step count
    output_options = ["--tripinfo-output", tripinfo_path, "--summary-output", summary_path]
    start_run_time = time.time()
    try:
        # Load model (expects path *without* .zip)
        model = RecurrentPPO.load(model_load_path_no_zip, device='cpu')
        print(f"Model {model_load_path_no_zip}.zip loaded successfully.")

        # Create environment instance
        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            sumocfg_file=SUMOCFG_FILE,
            use_gui=use_gui_flag,
            num_seconds=NUM_SECONDS_MAX,
            tls_id=TLS_ID,
            min_green=MIN_GREEN,
            max_green=MAX_GREEN,
            yellow_time=YELLOW_TIME,
            delta_time=DELTA_TIME,
            observation_lanes=OBSERVATION_LANES,
            reward_metric='wait_time', # Ensure this matches training
            port=port,
            seed=TEST_SEED,
            sumo_options=output_options # Pass output file options to SUMO
        )
        print("SUMO Environment created.")

        obs, info = env.reset()
        lstm_states = None # Initialize LSTM states
        # SB3 requires episode_starts as a NumPy array, (n_envs,) shape
        episode_starts = np.ones((1,), dtype=bool) # Start of a new episode
        done = False
        step_count = 0
        print("Starting simulation loop...")
        while not done: # Indent Level 2
            # Predict action, also getting the next LSTM state
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            ) # Indent Level 3
            obs, reward, terminated, truncated, info = env.step(action) # Indent Level 3
            done = terminated or truncated # Indent Level 3
            # Update episode_starts for the *next* prediction step
            # If done is True now, the next prediction is the start of a new episode
            episode_starts = np.array([done]) # Indent Level 3
            sim_end_step = info.get('step', sim_end_step) # Get current simulation step # Indent Level 3
            step_count += 1
            if step_count % 100 == 0: # Print progress periodically
                 print(f"  LSTM Step: {step_count}, Sim Time: {sim_end_step * DELTA_TIME:.0f}s", end='\r')

        # This block executes *after* the loop finishes
        # Correctly indented at Level 2
        if done: # Indent Level 2
            print(f"\nDRL (LSTM) simulation finished. Reason: {info.get('termination_reason', 'Unknown')}, Total Steps: {step_count}")

    except FileNotFoundError as e:
        print(f"ERROR during DRL (LSTM) run - File Not Found: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"ERROR during DRL (LSTM) run: {e}")
        traceback.print_exc()
    finally:
        if env:
            print("Closing SUMO Environment...")
            env.close()
            wall_time = time.time() - start_run_time
            print(f"DRL (LSTM) Wall Clock Time: {wall_time:.2f}s")
            time.sleep(1) # Allow files to close
            # Parse results *after* closing env
            metrics = parse_tripinfo(tripinfo_path)
            summary_end_time = parse_summary(summary_path)
            # Update duration from summary if available and seems valid
            if summary_end_time is not None and metrics is not None:
                 if metrics.get('sim_duration', 0) < summary_end_time:
                     metrics['sim_duration'] = summary_end_time
                 elif metrics.get('sim_duration') is None:
                      metrics['sim_duration'] = summary_end_time

            # Fallback if summary parsing failed or no vehicles arrived
            elif metrics is not None and metrics.get('sim_duration') == 0:
                 estimated_duration = sim_end_step * DELTA_TIME
                 metrics['sim_duration'] = estimated_duration
                 print(f"WARN: Using estimated sim duration from steps: {estimated_duration:.2f}s")

        else: # Handle case where env creation failed
            wall_time = time.time() - start_run_time
            print(f"DRL (LSTM) run aborted early. Wall Clock Time: {wall_time:.2f}s")

    return metrics, wall_time


# --- Function to Run Baseline (Fixed-Time) ---
def run_baseline(tripinfo_path, summary_path, use_gui_flag):
    """Runs the baseline fixed-time simulation using SUMO directly."""
    print(f"\n--- Running Fixed-Time Baseline Simulation (GUI: {use_gui_flag}) ---")
    metrics = None
    wall_time = 0.0
    # Choose sumo or sumo-gui executable
    sumo_exec = SUMO_GUI_BINARY if use_gui_flag else SUMO_BINARY
    if not sumo_exec:
        print(f"ERROR: Required SUMO executable not found for baseline (GUI={use_gui_flag}).")
        return None, 0.0

    # Construct the command line arguments
    baseline_cmd = [
        sumo_exec,
        "-c", SUMOCFG_FILE,
        "--duration-log.disable", "true", # Disable verbose duration logging
        "--no-step-log", "true",         # Disable step logging
        "--time-to-teleport", "-1",      # Disable automatic teleporting (important for realistic waiting times)
        "--seed", str(TEST_SEED),        # Set random seed for reproducibility
        "--end", str(NUM_SECONDS_MAX),   # Set simulation end time
        "--tripinfo-output", tripinfo_path, # Specify tripinfo output file
        "--summary-output", summary_path,  # Specify summary output file
        "--waiting-time-memory", "10000" # Increase memory for waiting time calculation (adjust if needed)
    ]

    print(f"Baseline CMD: {' '.join(baseline_cmd)}")
    start_run_time = time.time()
    process = None
    try:
        # Run the SUMO command
        # `check=False` because SUMO might exit with non-zero code even if simulation ran (e.g., warnings)
        # Capture output to check for errors if needed
        # Set a timeout slightly longer than the simulation duration
        process = subprocess.run(baseline_cmd, capture_output=True, text=True, check=False, timeout=NUM_SECONDS_MAX + 120)
        print("Baseline SUMO process finished.")
        if process.returncode != 0:
            print(f"WARN: Baseline SUMO exited with code {process.returncode}")
            # Print stderr only if there was an error, limit length
            if process.stderr:
                 print(f"SUMO Stderr (last 1000 chars):\n...\n{process.stderr[-1000:]}")
        # Uncomment to see stdout even on success (can be verbose)
        # if process.stdout:
        #      print(f"SUMO Stdout (last 1000 chars):\n...\n{process.stdout[-1000:]}")

    except subprocess.TimeoutExpired:
        print("ERROR: Baseline SUMO simulation timed out.")
        if process: process.kill() # Ensure the process is terminated
        return None, time.time() - start_run_time # Return partial wall time
    except FileNotFoundError:
        print(f"ERROR: Could not execute SUMO command. Is '{sumo_exec}' in your PATH or correctly specified?")
        return None, 0.0
    except Exception as e:
        print(f"ERROR running baseline SUMO: {e}")
        traceback.print_exc()
        return None, time.time() - start_run_time # Return partial wall time
    finally:
        wall_time = time.time() - start_run_time
        print(f"Baseline Wall Clock Time: {wall_time:.2f}s")
        time.sleep(1) # Allow files to write
        # Parse results after the simulation finishes or fails
        metrics = parse_tripinfo(tripinfo_path)
        summary_end_time = parse_summary(summary_path)
        # Update duration from summary if available
        if summary_end_time is not None and metrics is not None:
             if metrics.get('sim_duration', 0) < summary_end_time:
                 metrics['sim_duration'] = summary_end_time
             elif metrics.get('sim_duration') is None:
                  metrics['sim_duration'] = summary_end_time
        elif metrics is None and summary_end_time is not None:
             # Create a minimal metrics dict if tripinfo failed but summary has time
             metrics = {'sim_duration': summary_end_time}
             print("WARN: Tripinfo parsing failed, using only duration from summary.")


    return metrics, wall_time


# --- Function to Run Rule-Based ---
def run_rule_based(tripinfo_path, summary_path, use_gui_flag):
    """Runs the rule-based control script."""
    print(f"\n--- Running Rule-Based Simulation (GUI: {use_gui_flag}) ---")
    metrics = None
    wall_time = 0.0
    rule_script = "rule_based_control.py" # Assuming the script is in the same directory

    if not os.path.exists(rule_script):
        print(f"ERROR: Rule-based control script '{rule_script}' not found.")
        return None, 0.0

    # Construct the command to run the Python script
    rule_cmd = [
        sys.executable, # Use the same Python interpreter that's running this script
        rule_script,
        "--cfg", SUMOCFG_FILE,
        "--tls-id", TLS_ID,
        "--delta-time", str(DELTA_TIME), # Pass necessary parameters
        "--min-green", str(MIN_GREEN),
        "--max-green", str(MAX_GREEN),
        "--yellow-time", str(YELLOW_TIME),
        "--seed", str(TEST_SEED),
        "--max-sim-time", str(NUM_SECONDS_MAX),
        "--tripinfo-output", tripinfo_path,
        "--summary-output", summary_path,
    ]
    # Add the GUI flag if requested
    if use_gui_flag:
        rule_cmd.append("--gui")

    print(f"Rule-Based CMD: {' '.join(rule_cmd)}")
    start_run_time = time.time()
    process = None
    try:
        # Run the rule-based script as a separate process
        process = subprocess.run(rule_cmd, capture_output=True, text=True, check=False, timeout=NUM_SECONDS_MAX + 120)
        print("Rule-Based Script process finished.")
        if process.returncode != 0:
            print(f"WARN: Rule-Based script exited with code {process.returncode}")
            # Print output/error streams for debugging
            if process.stdout:
                print(f"Rule-Based Stdout (last 1000 chars):\n...\n{process.stdout[-1000:]}")
            if process.stderr:
                print(f"Rule-Based Stderr (last 1000 chars):\n...\n{process.stderr[-1000:]}")
        #else: # Optionally print stdout on success too
        #    if process.stdout:
        #         print(f"Rule-Based Stdout (last 1000 chars):\n...\n{process.stdout[-1000:]}")


    except subprocess.TimeoutExpired:
        print("ERROR: Rule-based simulation timed out.")
        if process: process.kill()
        return None, time.time() - start_run_time
    except FileNotFoundError:
        print(f"ERROR: Could not execute rule-based script. Is '{sys.executable}' valid and '{rule_script}' present?")
        return None, 0.0
    except Exception as e:
        print(f"ERROR running rule-based script: {e}")
        traceback.print_exc()
        return None, time.time() - start_run_time
    finally:
        wall_time = time.time() - start_run_time
        print(f"Rule-Based Wall Clock Time: {wall_time:.2f}s")
        time.sleep(1) # Allow files to write
        # Parse results
        metrics = parse_tripinfo(tripinfo_path)
        summary_end_time = parse_summary(summary_path)
        # Update duration from summary if available
        if summary_end_time is not None and metrics is not None:
             if metrics.get('sim_duration', 0) < summary_end_time:
                 metrics['sim_duration'] = summary_end_time
             elif metrics.get('sim_duration') is None:
                  metrics['sim_duration'] = summary_end_time
        elif metrics is None and summary_end_time is not None:
             metrics = {'sim_duration': summary_end_time}
             print("WARN: Tripinfo parsing failed, using only duration from summary.")

    return metrics, wall_time


# --- Main Comparison Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists
    results = {} # Dictionary to store metrics for each run type
    wall_times = {} # Dictionary to store wall clock time for each run type

    # Define which runs to execute
    run_types_to_execute = ["DRL-MLP", "DRL-LSTM", "Baseline", "RuleBased"]
    # run_types_to_execute = ["DRL-MLP", "Baseline"] # Example: Run only MLP and Baseline

    print("\nStarting simulation runs sequentially...")

    # --- DRL MLP/Custom Run ---
    if "DRL-MLP" in run_types_to_execute:
        # Use different ports for concurrent runs if needed, but sequential is safer
        drl_mlp_port = DEFAULT_PORT + random.randint(10, 20) # Offset port
        results["DRL-MLP"], wall_times["DRL-MLP"] = run_drl_mlp(
            MLP_MODEL_LOAD_PATH, # Pass path WITHOUT .zip
            TRIPINFO_FILE_DRL_MLP,
            SUMMARY_FILE_DRL_MLP,
            drl_mlp_port,
            USE_GUI
        )
        if USE_GUI and results.get("DRL-MLP") is not None:
            print("\n>>> DRL (MLP) GUI finished. Please close the SUMO GUI window manually.")
            input(">>> Press Enter in this terminal to continue to the next simulation...")
        elif results.get("DRL-MLP") is None:
             print("--- DRL (MLP) run failed or produced no results. ---")


    # --- DRL LSTM Run ---
    if "DRL-LSTM" in run_types_to_execute:
        drl_lstm_port = DEFAULT_PORT + random.randint(30, 40) # Different offset port
        results["DRL-LSTM"], wall_times["DRL-LSTM"] = run_drl_lstm(
            LSTM_MODEL_LOAD_PATH, # Pass path WITHOUT .zip
            TRIPINFO_FILE_LSTM,
            SUMMARY_FILE_LSTM,
            drl_lstm_port,
            USE_GUI
        )
        if USE_GUI and results.get("DRL-LSTM") is not None:
            print("\n>>> DRL (LSTM) GUI finished. Please close the SUMO GUI window manually.")
            input(">>> Press Enter in this terminal to continue to the next simulation...")
        elif results.get("DRL-LSTM") is None:
             print("--- DRL (LSTM) run failed or produced no results. ---")

    # --- Baseline (Fixed-Time) Run ---
    if "Baseline" in run_types_to_execute:
        results["Baseline"], wall_times["Baseline"] = run_baseline(
            TRIPINFO_FILE_BASE,
            SUMMARY_FILE_BASE,
            USE_GUI
        )
        if USE_GUI and results.get("Baseline") is not None:
            print("\n>>> Baseline GUI finished. Please close the SUMO GUI window manually.")
            input(">>> Press Enter in this terminal to continue to the next simulation...")
        elif results.get("Baseline") is None:
             print("--- Baseline run failed or produced no results. ---")


    # --- Rule-Based Run ---
    if "RuleBased" in run_types_to_execute:
        # Rule-based typically handles its own SUMO connection, no port needed here
        results["RuleBased"], wall_times["RuleBased"] = run_rule_based(
            TRIPINFO_FILE_RULE,
            SUMMARY_FILE_RULE,
            USE_GUI
        )
        if USE_GUI and results.get("RuleBased") is not None:
            print("\n>>> Rule-Based GUI finished. Please close the SUMO GUI window manually.")
            input(">>> Press Enter in this terminal to finish...") # Last one, just wait
        elif results.get("RuleBased") is None:
             print("--- Rule-Based run failed or produced no results. ---")


    # --- Display Comparison Table ---
    print("\n\n==============================================")
    print("========= FINAL SIMULATION COMPARISON ========")
    print("==============================================")

    # Define the order and display names for metrics
    metrics_display_order = [
        'sim_duration', 'avg_wait_time', 'total_wait_time', 'avg_travel_time',
        'avg_time_loss', 'std_dev_wait_time', 'p95_wait_time', 'total_vehicles'
    ]
    metrics_display_names = {
        'sim_duration': 'Sim Duration (s)',
        'avg_wait_time': 'Avg Wait Time (s)',
        'total_wait_time': 'Total Wait Time (s)',
        'avg_travel_time': 'Avg Travel Time (s)',
        'avg_time_loss': 'Avg Time Loss (s)',
        'std_dev_wait_time': 'Wait Time StdDev (s)',
        'p95_wait_time': 'Wait Time 95%ile (s)',
        'total_vehicles': 'Vehicles Completed'
    }

    # Prepare data for the table
    table_data = {}
    # Use the order defined in run_types_to_execute for columns
    run_types_in_results = [rt for rt in run_types_to_execute if rt in results] # Only include runs that were executed

    for metric_key in metrics_display_order:
        row_name = metrics_display_names.get(metric_key, metric_key)
        table_data[row_name] = []
        for run_type in run_types_in_results:
            run_result = results.get(run_type) # Get the dict for this run type
            value = run_result.get(metric_key) if run_result else None # Get the specific metric value

            # Format the value nicely
            if isinstance(value, (float, np.floating)) and value is not None:
                formatted_value = f"{value:.2f}"
            elif value is not None:
                formatted_value = str(value) # Integers or other types
            else:
                formatted_value = "N/A" # Handle missing data

            table_data[row_name].append(formatted_value)

    # Add Wall Clock Time row
    table_data["Wall Clock Time (s)"] = [f"{wall_times.get(run, 0.0):.2f}" for run in run_types_in_results]

    # Display using Pandas if available, otherwise basic print
    try:
        df = pd.DataFrame(table_data, index=run_types_in_results)
        # Transpose for better readability (metrics as rows, run types as columns)
        print(df.T)
        # Optionally save to CSV
        csv_filename = os.path.join(OUTPUT_DIR, "final_comparison_results.csv")
        df.T.to_csv(csv_filename)
        print(f"\nComparison table saved to: {csv_filename}")

    except ImportError:
        print("\nPandas not installed. Printing basic comparison table:")
        # Basic print formatting (less aligned than Pandas)
        header = f"{'Metric':<25} | " + " | ".join([f'{run_type:<12}' for run_type in run_types_in_results])
        print(header)
        print("-" * len(header))
        for key, values in table_data.items():
            row_values = " | ".join([f'{val:<12}' for val in values])
            print(f"{key:<25} | {row_values}")

    print("\n==============================================")
    print("--- Comparison Script Finished ---")