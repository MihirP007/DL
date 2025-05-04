# SumoEnvironment.py

import gymnasium as gym
import numpy as np
import os
import sys
import time
import random
import traceback # For detailed error printing

# --- SUMO_HOME Check ---
# (Robust check, including common paths and error message)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    # Verify bin directory exists
    sumo_bin_path = os.path.join(os.environ['SUMO_HOME'], 'bin')
    if not os.path.isdir(sumo_bin_path):
        sys.exit(f"ERROR: SUMO bin directory not found at '{sumo_bin_path}'. Check SUMO_HOME.")
else:
    potential_paths = ["C:\\Program Files (x86)\\Eclipse\\Sumo", "C:\\Program Files\\Eclipse\\Sumo", "/usr/share/sumo"]
    found_home = False
    for path in potential_paths:
        if os.path.exists(os.path.join(path, 'tools')) and os.path.exists(os.path.join(path, 'bin')):
             print(f"--- SUMO_HOME not set, but found tools/bin folders at: {path} ---")
             os.environ['SUMO_HOME'] = path
             tools = os.path.join(path, 'tools')
             sys.path.append(tools)
             found_home = True
             break
    if not found_home:
        sys.exit("Please declare environment variable 'SUMO_HOME' or ensure SUMO is installed in a standard location with bin and tools folders.")

# --- Import TraCI ---
try:
    import traci
except ImportError:
     sys.exit("TraCIpy module not found. Please ensure SUMO_HOME is set correctly and points to a valid SUMO installation with TraCI library in tools.")
except Exception as e:
     sys.exit(f"Error importing traci: {e}. Check Python environment and SUMO installation.")

# Default base port for SUMO simulation instances
DEFAULT_PORT = 8813

class SumoEnvironment(gym.Env):
    """
    Custom Gym Environment for controlling a SUMO traffic light.
    Includes termination on vehicle completion, optional SUMO args,
    refined rewards, penalties, and debugging.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self,
                 net_file: str,
                 route_file: str,
                 sumocfg_file: str,
                 use_gui: bool = False,
                 num_seconds: int = 7200, # Default max time (used if vehicles never finish)
                 tls_id: str = 'center',
                 min_green: int = 10,
                 max_green: int = 60,
                 yellow_time: int = 3,
                 delta_time: int = 5,
                 port: int = DEFAULT_PORT,
                 seed: int = 42,
                 observation_lanes: list = None,
                 reward_metric: str = 'wait_time',
                 penalize_min_green_violation: bool = True, # Option for penalty
                 min_green_penalty: float = -1.0, # Penalty value
                 sumo_options: list = None): # <<< Parameter for extra SUMO args

        super().__init__()
        print(f"DEBUG (Port {port}): SumoEnvironment __init__ starting.")

        # --- Input Validation ---
        if not os.path.exists(net_file): raise FileNotFoundError(f"Network file not found: {net_file}")
        if not os.path.exists(route_file): raise FileNotFoundError(f"Route file not found: {route_file}")
        if not os.path.exists(sumocfg_file): raise FileNotFoundError(f"Config file not found: {sumocfg_file}")

        # SUMO config
        self.net_file = net_file
        self.route_file = route_file
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.port = port
        self._seed = seed
        random.seed(self._seed)

        # Simulation parameters
        self.num_seconds = num_seconds # Acts as max duration now
        self.delta_time = delta_time

        # Traffic Light Signal parameters
        self.tls_id = tls_id
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.reward_metric = reward_metric
        self.penalize_min_green_violation = penalize_min_green_violation
        self.min_green_penalty = min_green_penalty
        self.sumo_options = sumo_options if sumo_options is not None else []

        # Internal state
        self.current_step = 0
        self.current_phase_index = 0
        self.time_since_last_switch = 0
        self.episode = 0
        self.last_reward_info = {}
        self.traci_conn = None
        self.current_step_penalty = 0.0
        self.cumulative_wait_time = 0.0

        # Define spaces placeholders
        self.action_space = None
        self.observation_space = None
        self.phases = []
        self.num_green_phases = 0
        self.observation_lanes = observation_lanes if observation_lanes is not None else []
        self.num_observed_lanes = len(self.observation_lanes)

        # --- Start SUMO and Define Spaces ---
        # Moved actual start to reset, but define spaces based on a temporary connection
        temp_port = self.port + random.randint(10000, 20000) # Use temp port for check
        temp_conn = None
        try:
            print(f"DEBUG (Port {self.port}): Attempting temporary TraCI connection on port {temp_port} to define spaces.")
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
            if sys.platform.startswith("win"): sumo_binary += ".exe"
            # Start SUMO just to get network info, then close
            temp_cmd = [sumo_binary, "-c", self.sumocfg_file, "--seed", str(self._seed), "--no-step-log", "true", "--duration-log.disable", "true"]
            temp_label = f"init_check_{temp_port}"
            traci.start(temp_cmd, port=temp_port, label=temp_label)
            temp_conn = traci.getConnection(temp_label)

            print(f"DEBUG (Port {self.port}): Getting TLS info from TraCI.")
            self.phases = temp_conn.trafficlight.getAllProgramLogics(self.tls_id)[0].phases
            self.num_phases = len(self.phases); self.num_green_phases = self.num_phases // 2
            print(f"DEBUG (Port {self.port}): Found {self.num_phases} total phases, {self.num_green_phases} green phases.")

            if not self.observation_lanes:
                print(f"DEBUG (Port {self.port}): Getting controlled lanes for {self.tls_id}.")
                self.observation_lanes = sorted(list(set(temp_conn.trafficlight.getControlledLanes(self.tls_id))))
                self.num_observed_lanes = len(self.observation_lanes)
                print(f"DEBUG (Port {self.port}): Auto-detected lanes: {self.observation_lanes}")
                if not self.observation_lanes: print(f"WARN (Port {self.port}): No controlled lanes found for TLS '{self.tls_id}'.")

            self.action_space = gym.spaces.Discrete(self.num_green_phases)
            obs_space_size = self.num_green_phases + 1 + self.num_observed_lanes
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32)
            print(f"DEBUG (Port {self.port}): Spaces defined - Action: {self.action_space}, Obs: {self.observation_space}")

        except traci.TraCIException as e:
             print(f"ERROR (Port {self.port}): TraCIException during __init__ space definition: {e}")
             print(traceback.format_exc()); raise RuntimeError("Failed to connect to SUMO to define env spaces.") from e
        except FileNotFoundError as e:
             print(f"ERROR: SUMO executable not found during init check. Path used: {sumo_binary if 'sumo_binary' in locals() else 'N/A'}"); raise e
        except Exception as e:
             print(f"ERROR (Port {self.port}): Exception during __init__ space definition: {e}")
             print(traceback.format_exc()); raise RuntimeError("Failed to define env spaces.") from e
        finally:
            if temp_conn:
                try: temp_conn.close()
                except Exception: pass # Ignore errors closing temp connection

        # Print final env info
        print(f"Environment Info:")
        print(f" - Port: {self.port}")
        print(f" - TLS ID: {self.tls_id}")
        # ... (rest of info prints) ...
        print(f"DEBUG (Port {self.port}): SumoEnvironment __init__ finished.")


    def _start_simulation(self):
        """Starts a SUMO simulation instance and connects TraCI."""
        print(f"DEBUG (Port {self.port}, Ep {self.episode}): Entering _start_simulation.")
        if self.traci_conn is not None:
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Found existing connection. Attempting close.")
            try: self.close()
            except Exception as e: print(f"WARN (Port {self.port}): Exception closing old conn in _start_simulation: {e}")

        # Determine SUMO binary path
        sumo_base_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_exe_path = os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_base_binary)
        if sys.platform.startswith("win"): sumo_exe_path += ".exe"
        if not os.path.exists(sumo_exe_path):
             raise FileNotFoundError(f"SUMO executable not found: {sumo_exe_path}")

        # --- Build Command ---
        sumo_cmd = [
            sumo_exe_path, "-c", self.sumocfg_file,
            "--seed", str(self._seed + self.episode),
            "--no-step-log", "true", "--time-to-teleport", "-1", "--no-warnings", "true",
            "--duration-log.disable", "true", "--waiting-time-memory", "1000", "--random",
            # Do not set --end here, control duration via step loop & getMinExpectedNumber
        ]
        # <<< Add extra options passed during init >>>
        sumo_cmd.extend(self.sumo_options)
        # --- End Build Command ---

        traci_label = f"sim_{self.port}_{self.episode}"
        print(f"DEBUG (Port {self.port}, Ep {self.episode}): SUMO CMD (before traci adds port): {' '.join(sumo_cmd)}")
        print(f"DEBUG (Port {self.port}, Ep {self.episode}): Attempting traci.start with port={self.port}, label={traci_label}")
        try:
            traci.start(sumo_cmd, port=self.port, label=traci_label)
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): traci.start command issued.")
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Attempting traci.getConnection with label={traci_label}")
            self.traci_conn = traci.getConnection(traci_label)
            if self.traci_conn is None:
                 print(f"ERROR (Port {self.port}, Ep {self.episode}): traci.getConnection returned None!")
                 raise traci.exceptions.FatalTraCIError(f"traci.getConnection returned None for label {traci_label}")
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): traci.getConnection successful. Conn object: {self.traci_conn}")
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Exiting _start_simulation successfully.")
        except traci.exceptions.FatalTraCIError as e:
            print(f"ERROR (Port {self.port}, Ep {self.episode}): FatalTraCIError during _start_simulation - Could not connect."); print(f"       Error: {e}")
            self.traci_conn = None; self.close(); raise e
        except traci.TraCIException as e:
            print(f"ERROR (Port {self.port}, Ep {self.episode}): TraCIException during _start_simulation: {e}")
            print(traceback.format_exc()); self.traci_conn = None; self.close(); raise e
        except Exception as e:
            print(f"ERROR (Port {self.port}, Ep {self.episode}): Non-TraCI Exception during _start_simulation: {e}")
            print(traceback.format_exc()); self.traci_conn = None; self.close(); raise e


    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        print(f"DEBUG (Port {self.port}, Ep {self.episode}): Entering reset method.")
        super().reset(seed=seed)
        if seed is not None: self._seed = seed; random.seed(self._seed)

        # Increment episode AFTER potentially using seed+episode in _start_simulation if called from here
        # Let's keep it consistent: increment FIRST, then use seed+(new episode number)
        self.episode += 1
        print(f"DEBUG (Port {self.port}): Episode incremented to {self.episode}.")
        self.current_step = 0
        self.current_phase_index = 0
        self.time_since_last_switch = 0
        self.last_reward_info = {}
        self.current_step_penalty = 0.0
        self.cumulative_wait_time = 0.0

        # Prepare fallback observation
        initial_obs = None; initial_info = {}
        if hasattr(self, 'observation_space') and self.observation_space is not None:
             initial_obs = self.observation_space.sample()
        else: initial_obs = np.array([0.0], dtype=np.float32) # Basic fallback

        print(f"DEBUG (Port {self.port}, Ep {self.episode}): Calling _start_simulation from reset.")
        try:
            self._start_simulation()
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): _start_simulation completed successfully in reset.")
        except Exception as e:
            print(f"ERROR (Port {self.port}, Ep {self.episode}): _start_simulation failed during reset: {e}")
            print(traceback.format_exc()); initial_info['error'] = f"Failed to start SUMO during reset: {e}"; return initial_obs, initial_info

        # Check connection immediately
        if self.traci_conn is None:
             print(f"ERROR (Port {self.port}, Ep {self.episode}): traci_conn is None immediately after _start_simulation call in reset!")
             initial_info['error'] = "TraCI connection is None after start in reset"; return initial_obs, initial_info

        # Perform initial setup steps
        try:
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Setting initial phase {self.current_phase_index * 2} for TLS {self.tls_id}.")
            self.traci_conn.trafficlight.setPhase(self.tls_id, self.current_phase_index * 2)
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Initial phase set.")

            # Simulate a few initial steps (important for some scenarios)
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Simulating initial 10 steps.")
            self._simulate_steps(10) # Try uncommenting this LATER if reset works without it
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Initial 10 steps simulated.")

            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Getting initial observation.")
            initial_obs = self._get_observation() # Get the actual initial observation
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Got initial observation.")

        except traci.TraCIException as e:
            print(f"ERROR (Port {self.port}, Ep {self.episode}): TraCIException during reset's initial setup: {e}")
            print(traceback.format_exc()); self.close(); initial_info['error'] = f"TraCI error during reset setup: {e}"
        except Exception as e:
            print(f"ERROR (Port {self.port}, Ep {self.episode}): Non-TraCI Exception during reset's initial setup: {e}")
            print(traceback.format_exc()); self.close(); initial_info['error'] = f"Non-TraCI error during reset setup: {e}"

        print(f"DEBUG (Port {self.port}, Ep {self.episode}): Exiting reset method.")
        # Ensure initial_obs has the correct fallback if errors occurred during setup but didn't return early
        if initial_obs is None:
             print(f"WARN (Port {self.port}, Ep {self.episode}): Initial observation is still None at end of reset. Resampling.")
             if hasattr(self, 'observation_space') and self.observation_space is not None: initial_obs = self.observation_space.sample()
             else: initial_obs = np.array([0.0], dtype=np.float32) # Basic fallback

        return initial_obs, initial_info


    def _get_observation(self):
        # (Keep the version from the previous "complete file" response)
        # Includes checks for spaces being defined and basic error handling for traci calls
        if self.action_space is None or self.observation_space is None:
             print(f"WARN (Port {self.port}): Spaces not defined in _get_observation. Returning zeros.")
             obs_shape = (1,);
             if hasattr(self, 'num_green_phases') and hasattr(self, 'num_observed_lanes'): obs_shape = (self.num_green_phases + 1 + self.num_observed_lanes,)
             return np.zeros(obs_shape, dtype=np.float32)
        phase_one_hot = np.zeros(self.num_green_phases, dtype=np.float32); # ... (rest of obs logic) ...
        if 0 <= self.current_phase_index < self.num_green_phases: phase_one_hot[self.current_phase_index] = 1.0
        time_in_phase_norm = np.float32(min(self.time_since_last_switch / max(self.max_green, 1), 1.0)); max_expected_queue = 50.0
        queue_counts = np.zeros(self.num_observed_lanes, dtype=np.float32)
        if self.traci_conn and self.observation_lanes:
            for i, lane_id in enumerate(self.observation_lanes):
                try: halt_count = self.traci_conn.lane.getLastStepHaltingNumber(lane_id); queue_counts[i] = min(halt_count / max_expected_queue, 1.0)
                except traci.TraCIException as e:
                    if "is not known" in str(e): pass
                    elif "connection" in str(e).lower() or "socket" in str(e).lower(): print(f"ERROR (Port {self.port}): TraCI connection error in _get_observation for lane {lane_id}: {e}"); self.close(); break
                    else: print(f"WARN (Port {self.port}): TraCIException for lane {lane_id} in _get_observation: {e}")
                except Exception as e: print(f"ERROR (Port {self.port}): Unexpected exception for lane {lane_id} in _get_observation: {e}"); break
        observation = np.concatenate([phase_one_hot, [time_in_phase_norm], queue_counts]); expected_shape = (self.num_green_phases + 1 + self.num_observed_lanes,)
        if observation.shape != expected_shape:
             print(f"WARN (Port {self.port}): Observation shape mismatch! Expected {expected_shape}, Got {observation.shape}. Padding/truncating."); # ... (shape correction logic) ...
             correct_obs = np.zeros(expected_shape, dtype=np.float32); min_len = min(len(observation), len(correct_obs)); correct_obs[:min_len] = observation[:min_len]; observation = correct_obs
        return observation.astype(np.float32)


    def _get_reward(self):
        # (Keep the version from the previous "complete file" response)
        # Includes reward calculation logic, adding penalty, error handling
        if self.traci_conn is None: return 0.0; reward = 0.0; current_metric_val = 0.0; metric_key = f'current_{self.reward_metric}'
        try:
            if self.reward_metric == 'wait_time': # ... (wait time logic using getWaitingTime) ...
                current_total_wait = 0
                for lane_id in self.observation_lanes:
                    vehicles_on_lane = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
                    for vehicle_id in vehicles_on_lane: current_total_wait += self.traci_conn.vehicle.getWaitingTime(vehicle_id)
                current_metric_val = current_total_wait; self.cumulative_wait_time += current_total_wait; metric_key = 'step_wait_time'
            elif self.reward_metric == 'queue_length': # ... (queue length logic) ...
                current_total_queue = 0
                for lane_id in self.observation_lanes: current_total_queue += self.traci_conn.lane.getLastStepHaltingNumber(lane_id)
                current_metric_val = current_total_queue; metric_key = 'step_queue_length'
            else: print(f"ERROR ... Unsupported reward metric: {self.reward_metric}"); return 0.0
            reward = -current_metric_val; self.last_reward_info[metric_key] = current_metric_val; self.last_reward_info['cumulative_wait_time'] = self.cumulative_wait_time
        except traci.TraCIException as e: # ... (error handling) ...
            if "connection" in str(e).lower() or "socket" in str(e).lower(): self.close(); reward = 0.0; self.last_reward_info['error'] = f'TraCI connection error: {e}'
            else: reward = 0.0; self.last_reward_info['error'] = f'TraCI error in reward: {e}' # Assume 0 reward on other TraCI errors
        except Exception as e: reward = 0.0; self.last_reward_info['error'] = f'Unexpected error in reward: {e}'
        reward += self.current_step_penalty; self.last_reward_info['step_penalty'] = self.current_step_penalty
        return reward


    def _apply_action(self, action):
        # (Keep the version from the previous "complete file" response)
        # Includes penalty logic, min/max green enforcement, phase switching
        self.current_step_penalty = 0.0; # ... (rest of apply_action logic) ...
        if self.traci_conn is None: return False, 0
        try: chosen_phase_index = int(action)
        except (ValueError, TypeError): chosen_phase_index = 0
        if not (0 <= chosen_phase_index < self.num_green_phases): chosen_phase_index = max(0, min(chosen_phase_index, self.num_green_phases - 1))
        is_same_phase = (chosen_phase_index == self.current_phase_index); time_exceeds_max = (self.time_since_last_switch >= self.max_green); time_below_min = (self.time_since_last_switch < self.min_green)
        phase_to_set = -1; yellow_duration_to_simulate = 0; phase_change_initiated = False
        if not is_same_phase and time_below_min and self.penalize_min_green_violation:
             print(f"DEBUG (Port {self.port}): Agent attempted switch to {chosen_phase_index} before min_green ({self.time_since_last_switch} < {self.min_green}). Applying penalty."); self.current_step_penalty += self.min_green_penalty
        if is_same_phase:
            if not time_exceeds_max: phase_to_set = self.current_phase_index * 2; phase_change_initiated = False
            else: next_phase_index = (self.current_phase_index + 1) % self.num_green_phases; print(f"WARN ... Max green time exceeded... Forcing switch to {next_phase_index}."); chosen_phase_index = next_phase_index
        if not is_same_phase or time_exceeds_max:
            if time_below_min and not time_exceeds_max: phase_to_set = self.current_phase_index * 2; phase_change_initiated = False
            else:
                current_yellow_phase = self.current_phase_index * 2 + 1; next_green_phase = chosen_phase_index * 2
                try:
                    self.traci_conn.trafficlight.setPhase(self.tls_id, current_yellow_phase); yellow_duration_to_simulate = self.yellow_time; self._simulate_steps(yellow_duration_to_simulate)
                    self.traci_conn.trafficlight.setPhase(self.tls_id, next_green_phase); self.current_phase_index = chosen_phase_index; self.time_since_last_switch = 0; phase_change_initiated = True
                except traci.TraCIException as e: print(f"ERROR (Port {self.port}): TraCIException during phase change: {e}"); self.close(); phase_change_initiated = False; yellow_duration_to_simulate = 0
                except Exception as e: print(f"ERROR (Port {self.port}): Unexpected Exception during phase change: {e}"); self.close(); phase_change_initiated = False; yellow_duration_to_simulate = 0
        return phase_change_initiated, yellow_duration_to_simulate


    def _simulate_steps(self, num_steps):
        # (Keep the version from the previous "complete file" response)
        if self.traci_conn is None: return; # Check if connection is None
        if num_steps <= 0: return
        try:
            for _ in range(num_steps):
                self.traci_conn.simulationStep(); self.current_step += 1
                # Check if max simulation time reached within these steps
                if self.current_step * 1 >= self.num_seconds: # Assuming 1s step interval
                    print(f"INFO (Port {self.port}): Max duration {self.num_seconds} reached during _simulate_steps.")
                    break
                # Check vehicle completion within loop? Can be slow. Better done in step().
                # if traci.simulation.getMinExpectedNumber() == 0:
                #     print(f"INFO (Port {self.port}): All vehicles processed during _simulate_steps.")
                #     break # Stop simulating steps if done
        except traci.TraCIException as e: print(f"ERROR (Port {self.port}): TraCIException during simulation step {self.current_step}: {e}"); print(traceback.format_exc()); self.close()
        except Exception as e: print(f"ERROR (Port {self.port}): Unexpected Exception during simulation step {self.current_step}: {e}"); print(traceback.format_exc()); self.close()


    def step(self, action):
        """Executes one time step within the environment."""
        # --- Connection Check ---
        if self.traci_conn is None:
            obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype) if self.observation_space else np.array([0.0])
            return obs, 0.0, True, False, {"error": "TraCI connection lost before step execution"}

        # 1. Apply action
        phase_changed, yellow_duration_simulated = self._apply_action(action)

        # 2. Simulate remaining time
        steps_to_simulate = max(0, self.delta_time - yellow_duration_simulated)
        if steps_to_simulate > 0:
            self._simulate_steps(steps_to_simulate)

        # --- Exit check AFTER potentially simulating steps ---
        all_vehicles_done_this_step = False
        try:
             # Check connection again before TraCI call
             if self.traci_conn and traci.simulation.getMinExpectedNumber() == 0:
                 print(f"INFO (Port {self.port}): All vehicles processed (getMinExpectedNumber is 0 at step {self.current_step}). Terminating.")
                 all_vehicles_done_this_step = True
                 # self.close() will be called later based on terminated flag
        except traci.TraCIException as e:
            print(f"WARN (Port {self.port}): TraCI exception during getMinExpectedNumber check in step: {e}")
            # This likely means connection is lost
            self.close() # Close proactively if check fails
        except Exception as e: # Catch other errors during check
             print(f"ERROR (Port {self.port}): Unexpected error during getMinExpectedNumber check: {e}")
             self.close()

        # 3. Update timer
        if phase_changed: self.time_since_last_switch += steps_to_simulate
        else: self.time_since_last_switch += self.delta_time

        # 4. Get reward
        reward = self._get_reward()

        # 5. Get observation
        observation = self._get_observation()

        # 6. Check termination conditions
        sim_time_ended = (self.current_step * 1 >= self.num_seconds) # Max duration
        connection_lost = (self.traci_conn is None) # Connection closed during step
        terminated = sim_time_ended or connection_lost or all_vehicles_done_this_step

        # 7. Info dict
        info = self.last_reward_info.copy()
        info['step'] = self.current_step; info['phase'] = self.current_phase_index; info['time_in_phase'] = self.time_since_last_switch; info['action'] = action; info['phase_changed'] = phase_changed
        if all_vehicles_done_this_step: info['termination_reason'] = "all_vehicles_done"
        elif sim_time_ended: info['termination_reason'] = "num_seconds_reached"
        elif connection_lost and not all_vehicles_done_this_step: info['error'] = "TraCI connection lost during step"; info['termination_reason'] = "connection_lost"

        truncated = False

        # Ensure connection is closed if terminating for any reason
        if terminated and self.traci_conn is not None:
             print(f"DEBUG (Port {self.port}): Terminating step, ensuring connection is closed.")
             self.close()

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        # (Keep as before)
        if mode == 'human': pass
        elif mode == 'rgb_array': return None


    # --- CORRECTED close method ---
    def close(self):
        """Closes the TraCI connection."""
        print(f"DEBUG (Port {self.port}, Ep {self.episode}): Close method called.")
        if self.traci_conn is not None:
            print(f"DEBUG (Port {self.port}, Ep {self.episode}): Connection object exists, attempting traci_conn.close()")
            try:
                self.traci_conn.close()
                print(f"DEBUG (Port {self.port}, Ep {self.episode}): traci_conn.close() executed.")
            except traci.TraCIException as e: print(f"WARN (Port {self.port}, Ep {self.episode}): TraCIException closing TraCI connection: {e}. Might be already closed or SUMO crashed.")
            except AttributeError: print(f"WARN (Port {self.port}, Ep {self.episode}): AttributeError during close. Connection object might be invalid.")
            except Exception as e: print(f"ERROR (Port {self.port}, Ep {self.episode}): Unexpected error during traci_conn.close(): {e}"); print(traceback.format_exc())
            finally: self.traci_conn = None; print(f"DEBUG (Port {self.port}, Ep {self.episode}): Connection variable set to None.")
        else: print(f"DEBUG (Port {self.port}, Ep {self.episode}): No active TraCI connection object to close (self.traci_conn is None).")


    def __del__(self):
        # (Keep as before)
        self.close()

# --- Standalone Test Block ---
if __name__ == "__main__":
    # (Keep as before or remove if not needed)
    pass