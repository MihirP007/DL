# rule_based_control.py (Corrected with --gui flag handling)

import os
import sys
import time
import argparse
import traceback

# --- SUMO_HOME Check ---
sumo_home_path = ""
if 'SUMO_HOME' in os.environ:
    sumo_home_path = os.environ['SUMO_HOME']
    tools = os.path.join(sumo_home_path, 'tools')
    sys.path.append(tools)
else:
    potential_paths = ["C:\\Program Files (x86)\\Eclipse\\Sumo", "C:\\Program Files\\Eclipse\\Sumo", "/usr/share/sumo"]
    for path in potential_paths:
        if os.path.exists(os.path.join(path, 'bin')) and os.path.exists(os.path.join(path, 'tools')):
            sumo_home_path = path; print(f"--- Found SUMO path: {sumo_home_path} ---"); break
    if not sumo_home_path: sys.exit("ERROR: Could not find SUMO installation. Set SUMO_HOME environment variable.")
    tools = os.path.join(sumo_home_path, 'tools'); sys.path.append(tools)

try:
    import traci
except ImportError:
     sys.exit("TraCIpy module not found. Please ensure SUMO_HOME/tools is in the Python path.")
except Exception as e:
     sys.exit(f"Error importing traci: {e}. Check SUMO installation and Python environment.")

# --- Default Configuration ---
DEFAULT_CFG_FILE = "intersection.sumocfg"; DEFAULT_TLS_ID = "center"; DEFAULT_DELTA_TIME = 5
DEFAULT_MIN_GREEN = 10; DEFAULT_MAX_GREEN = 60; DEFAULT_YELLOW_TIME = 3
DEFAULT_PORT_RULE = 9001; DEFAULT_SEED = 42; DEFAULT_MAX_SIM_TIME = 7200

# --- Define PHASE_LANES (Ensure this matches your network!) ---
PHASE_LANES = {
    0: ['edge_N_in_0', 'edge_N_in_1', 'edge_S_in_0', 'edge_S_in_1'],
    1: ['edge_E_in_0', 'edge_E_in_1', 'edge_W_in_0', 'edge_W_in_1']
}
NUM_GREEN_PHASES = len(PHASE_LANES)

# --- get_options function (Added --gui) ---
def get_options():
    parser = argparse.ArgumentParser(description="Rule-Based SUMO Traffic Light Control")
    parser.add_argument("-c", "--cfg", dest="cfg_file", default=DEFAULT_CFG_FILE, help="SUMO cfg file")
    parser.add_argument("--tls-id", dest="tls_id", default=DEFAULT_TLS_ID, help="TLS ID")
    parser.add_argument("--delta-time", dest="delta_time", type=int, default=DEFAULT_DELTA_TIME, help="Decision interval (s)")
    parser.add_argument("--min-green", dest="min_green", type=int, default=DEFAULT_MIN_GREEN, help="Min green time (s)")
    parser.add_argument("--max-green", dest="max_green", type=int, default=DEFAULT_MAX_GREEN, help="Max green time (s)")
    parser.add_argument("--yellow-time", dest="yellow_time", type=int, default=DEFAULT_YELLOW_TIME, help="Yellow time (s)")
    parser.add_argument("--port", dest="port", type=int, default=DEFAULT_PORT_RULE, help="TraCI port")
    parser.add_argument("--seed", dest="seed", type=int, default=DEFAULT_SEED, help="Simulation seed")
    parser.add_argument("--max-sim-time", dest="max_sim_time", type=int, default=DEFAULT_MAX_SIM_TIME, help="Max sim time (s)")
    parser.add_argument("--tripinfo-output", dest="tripinfo_output", required=True, help="Path for tripinfo output XML")
    parser.add_argument("--summary-output", dest="summary_output", required=True, help="Path for summary output XML")
    # --- Added GUI flag ---
    parser.add_argument("--gui", action="store_true", default=False, help="Run with SUMO GUI instead of CLI")
    # --- End Add ---
    return parser.parse_args()

# --- get_phase_queue function ---
def get_phase_queue(phase_index):
    total_queue = 0
    if phase_index in PHASE_LANES:
        for lane_id in PHASE_LANES[phase_index]:
            try: total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            except traci.TraCIException: pass # Ignore error if lane not known
    return total_queue

# --- run_simulation function (Selects executable based on options.gui) ---
def run_simulation(options):
    # <<< Select binary based on --gui flag >>>
    sumo_binary_name = "sumo-gui" if options.gui else "sumo"
    sumo_exe_path = os.path.join(sumo_home_path, 'bin', sumo_binary_name)
    if sys.platform.startswith("win"): sumo_exe_path += ".exe"
    if not os.path.exists(sumo_exe_path):
        print(f"ERROR: {sumo_binary_name} executable not found at {sumo_exe_path}")
        return

    sumo_cmd = [
        sumo_exe_path, # Use selected executable path
        "-c", options.cfg_file, "--seed", str(options.seed),
        "--no-step-log", "true", "--time-to-teleport", "-1", "--no-warnings", "false",
        "--duration-log.disable", "true", "--waiting-time-memory", "1000",
        "--end", str(options.max_sim_time),
        "--tripinfo-output", options.tripinfo_output, "--summary-output", options.summary_output,
    ]
    traci_label = f"rule_based_{options.port}"
    print(f"--- Starting Rule-Based SUMO {'GUI' if options.gui else 'Headless'} Simulation ---")
    print(f"CMD: {' '.join(sumo_cmd)}"); print(f"Attempting traci.start on port {options.port}")
    try:
        traci.start(sumo_cmd, port=options.port, label=traci_label); print("TraCI connection established.")
    except Exception as e: print(f"ERROR starting SUMO/TraCI: {e}"); return

    current_step = 0; last_decision_step = 0; current_green_phase_index = 0; time_in_current_phase = 0
    try: traci.trafficlight.setPhase(options.tls_id, current_green_phase_index * 2)
    except traci.TraCIException as e: print(f"ERROR setting initial phase: {e}"); traci.close(); return

    start_time = time.time(); simulation_running = True
    while simulation_running:
        try:
            if traci.simulation.getMinExpectedNumber() == 0:
                print(f"INFO: All vehicles processed at step {current_step}. Ending simulation."); simulation_running = False; break
            traci.simulationStep(); current_step += 1; time_in_current_phase += 1

            if current_step >= last_decision_step + options.delta_time:
                last_decision_step = current_step
                # print(f"\n--- Step {current_step}: Checking Rules ---") # Keep logs minimal for GUI run
                # print(f"Current Green Phase: {current_green_phase_index}, Time in Phase: {time_in_current_phase}")
                other_phase_index = (current_green_phase_index + 1) % NUM_GREEN_PHASES
                switch_decision = False; reason = ""

                if time_in_current_phase < options.min_green:
                    switch_decision = False; reason = f"Min green ({options.min_green}s) not met."
                elif time_in_current_phase >= options.max_green:
                    switch_decision = True; reason = f"Max green ({options.max_green}s) reached."
                else:
                    current_queue = get_phase_queue(current_green_phase_index)
                    other_queue = get_phase_queue(other_phase_index)
                    # print(f"Queue (Phase {current_green_phase_index}): {current_queue}, Queue (Phase {other_phase_index}): {other_queue}")
                    # --- <<< Worsened Rule (Switch if other > 0) >>> ---
                    if other_queue > 0: switch_decision = True; reason = "Other phase has waiting vehicles."
                    else: switch_decision = False; reason = "Other phase is empty."
                    # --- <<< End Worsened Rule >>> ---
                # print(f"Decision: {'Switch' if switch_decision else 'Stay'}. Reason: {reason}")

                if switch_decision:
                    yellow_phase = current_green_phase_index * 2 + 1; next_green_phase = other_phase_index * 2
                    # print(f"Switching: Setting Yellow Phase {yellow_phase} for {options.yellow_time}s")
                    traci.trafficlight.setPhase(options.tls_id, yellow_phase)
                    for _ in range(options.yellow_time):
                         if traci.simulation.getMinExpectedNumber() == 0: print("INFO: All vehicles processed during yellow phase. Ending."); simulation_running = False; break
                         traci.simulationStep(); current_step += 1
                    if not simulation_running: break
                    # print(f"Switching: Setting Green Phase {next_green_phase}")
                    traci.trafficlight.setPhase(options.tls_id, next_green_phase)
                    current_green_phase_index = other_phase_index; time_in_current_phase = 0
        except traci.TraCIException as e:
            if "connection" in str(e).lower() or "socket" in str(e).lower(): print(f"ERROR: TraCI connection lost at step {current_step}: {e}")
            else: print(f"ERROR: TraCI exception at step {current_step}: {e}")
            print(traceback.format_exc()); simulation_running = False
        except Exception as e: print(f"ERROR: Unexpected Python error at step {current_step}: {e}"); print(traceback.format_exc()); simulation_running = False

    end_time = time.time(); print(f"\n--- Rule-Based Simulation Finished ---"); final_step = current_step
    try: final_veh_count = traci.simulation.getMinExpectedNumber(); print(f"Reason: {'All vehicles processed' if final_veh_count == 0 else 'Error or Max Time'}")
    except: final_veh_count = -1; print("Reason: Error or Max Time (Could not check vehicle count)")
    print(f"Final Simulation Step: {final_step}"); print(f"Wall Clock Duration: {end_time - start_time:.2f}s")
    try: traci.close(); print("TraCI connection closed.")
    except Exception as e: print(f"WARN: Error closing TraCI connection: {e}")

if __name__ == "__main__":
    opts = get_options()
    run_simulation(opts)