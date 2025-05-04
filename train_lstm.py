# Fixed and Improved train_lstm.py for RecurrentPPO

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys
import time
import traceback
import torch

try:
    from SumoEnvironment import SumoEnvironment, DEFAULT_PORT
except ImportError:
    print("ERROR: Failed to import SumoEnvironment.")
    sys.exit(1)

# --- Configuration ---
NET_FILE = "my_intersection.net.xml"
ROUTE_FILE = "random_traffic.rou.xml"
SUMOCFG_FILE = "intersection.sumocfg"
TLS_ID = "center"
NUM_SECONDS = 1200
DELTA_TIME = 5
MIN_GREEN = 10
MAX_GREEN = 60
YELLOW_TIME = 3
OBSERVATION_LANES = [
    'edge_N_in_0', 'edge_N_in_1', 'edge_S_in_0', 'edge_S_in_1',
    'edge_E_in_0', 'edge_E_in_1', 'edge_W_in_0', 'edge_W_in_1'
]
REWARD_METRIC = 'wait_time'

MODEL_ALGORITHM = RecurrentPPO
POLICY_STR = "MlpLstmPolicy"
TOTAL_TIMESTEPS = 1000000
N_CPU = 1  # DummyVecEnv only supports 1 env for RNNs in stable-baselines3
EVAL_FREQ = 10000
N_EVAL_EPISODES = 5

SAVE_SUBDIR = f"{MODEL_ALGORITHM.__name__}_{POLICY_STR}_{time.strftime('%Y%m%d-%H%M%S')}"
SAVE_PATH = os.path.join("models", SAVE_SUBDIR)
LOG_PATH = os.path.join("logs", SAVE_SUBDIR)
BEST_MODEL_SAVE_PATH = os.path.join(SAVE_PATH, 'best_model')

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

def create_env(rank=0, seed=0, port_offset=0):
    def _init():
        env_port = DEFAULT_PORT + port_offset + rank
        env = SumoEnvironment(
            net_file=NET_FILE, route_file=ROUTE_FILE, sumocfg_file=SUMOCFG_FILE,
            use_gui=False, num_seconds=NUM_SECONDS, tls_id=TLS_ID,
            min_green=MIN_GREEN, max_green=MAX_GREEN, yellow_time=YELLOW_TIME,
            delta_time=DELTA_TIME, observation_lanes=OBSERVATION_LANES,
            reward_metric=REWARD_METRIC, port=env_port, seed=seed + rank
        )
        return Monitor(env, filename=os.path.join(LOG_PATH, f'monitor_{rank}.csv'))
    return _init

if __name__ == "__main__":
    base_seed = int(time.time())
    start_time = time.time()

    print("--- Setting up DummyVecEnv ---")
    vec_env = DummyVecEnv([create_env(seed=base_seed)])

    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=True,  # ✅ Only this one is set — do NOT also set enable_critic_lstm
        net_arch=dict(pi=[64], vf=[64])  # ✅ use dict, not list, to avoid SB3 v1.8.0 warning
    )


    print("--- Setting up Model ---")
    model = RecurrentPPO(
        policy=POLICY_STR,
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_PATH,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.4,
        max_grad_norm=0.8,
        seed=base_seed,
        device="auto"
    )

    print("--- Setting up Evaluation ---")
    eval_env = DummyVecEnv([create_env(rank=1, seed=base_seed + 1)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=os.path.join(LOG_PATH, 'eval_logs'),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    print("--- Starting Training ---")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            tb_log_name=f"{MODEL_ALGORITHM.__name__}_{POLICY_STR}",
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()

    print("--- Saving Model ---")
    try:
        model.save(os.path.join(SAVE_PATH, "final_model_lstm.zip"))
        print(f"Model saved to {SAVE_PATH}")
    except Exception as e:
        print(f"Failed to save model: {e}")

    vec_env.close()
    eval_env.close()
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
    print("--- Done ---")
