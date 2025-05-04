
# NOTE: This is a snippet showing corrected indentation for the LSTM section (where the indentation was broken in your input)
# The full corrected file is long, so if you'd like, I can provide the full version in a downloadable format.

# --- Function to Run DRL (LSTM) ---
def run_drl_lstm(model_load_path, tripinfo_path, summary_path, port, use_gui_flag):
    print(f"\n--- Running DRL (LSTM) Simulation (Port: {port}, GUI: {use_gui_flag}) ---")
    model = None
    env = None
    metrics = None
    wall_time = 0.0
    sim_end_step = 0
    output_options = ["--tripinfo-output", tripinfo_path, "--summary-output", summary_path]
    start_run_time = time.time()
    try:
        model = RecurrentPPO.load(model_load_path, device='cpu')
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
            reward_metric='wait_time',
            port=port,
            seed=TEST_SEED,
            sumo_options=output_options
        )
        obs, info = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_starts = np.array([done])
            sim_end_step = info.get('step', sim_end_step)
        if done:
            print(f"DRL (LSTM) simulation finished. Reason: {info.get('termination_reason', 'Unknown')}")
    except Exception as e:
        print(f"ERROR during DRL (LSTM) run: {e}")
        traceback.print_exc()
    finally:
        if env:
            env.close()
            wall_time = time.time() - start_run_time
            print(f"DRL (LSTM) Wall Clock Time: {wall_time:.2f}s")
            time.sleep(1)
            metrics = parse_tripinfo(tripinfo_path)
            summary_end_time = parse_summary(summary_path)
        if summary_end_time is not None and metrics is not None:
            metrics['sim_duration'] = summary_end_time
        elif metrics is not None and metrics.get('sim_duration') is None:
            metrics['sim_duration'] = sim_end_step
    return metrics, wall_time
