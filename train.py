# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import os
import time
import random
import torch as th
import torch.nn as nn

# --- Import the custom environment ---
from SumoEnvironment import SumoEnvironment, DEFAULT_PORT

# --- Configuration (Keep most settings as before) ---
NET_FILE = "my_intersection.net.xml"
ROUTE_FILE = "random_traffic.rou.xml"
SUMOCFG_FILE = "intersection.sumocfg"
TLS_ID = "center"
NUM_SECONDS = 3600
DELTA_TIME = 5
MIN_GREEN = 10
MAX_GREEN = 60
YELLOW_TIME = 3
OBSERVATION_LANES = [
    'edge_N_in_0', 'edge_N_in_1', 'edge_S_in_0', 'edge_S_in_1',
    'edge_E_in_0', 'edge_E_in_1', 'edge_W_in_0', 'edge_W_in_1'
]
REWARD_METRIC = 'wait_time'

# --- Training parameters ---
MODEL_ALGORITHM = PPO
TOTAL_TIMESTEPS = 300000 # <<< INCREASED training time further for more complex network
N_CPU = 4
EVAL_FREQ = 15000 # Evaluate slightly less often if training takes longer
N_EVAL_EPISODES = 5

# --- Paths ---
SAVE_SUBDIR = f"{MODEL_ALGORITHM.__name__}_ComplexNN_{time.strftime('%Y%m%d-%H%M%S')}" # Updated name
SAVE_PATH = os.path.join("models", SAVE_SUBDIR)
LOG_PATH = os.path.join("logs", SAVE_SUBDIR)
BEST_MODEL_SAVE_PATH = os.path.join(SAVE_PATH, 'best_model')

# --- Create directories ---
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

# --- create_env function (Keep exactly as before) ---
def create_env(rank, seed=0, port_offset=0):
    """Utility function for multiprocessed env creation."""
    def _init():
        env_port = DEFAULT_PORT + port_offset + rank
        print(f"Creating environment for rank {rank} on port {env_port} with seed {seed + rank}")
        env = SumoEnvironment(
                net_file=NET_FILE, route_file=ROUTE_FILE, sumocfg_file=SUMOCFG_FILE,
                use_gui=False, num_seconds=NUM_SECONDS, tls_id=TLS_ID,
                min_green=MIN_GREEN, max_green=MAX_GREEN, yellow_time=YELLOW_TIME,
                delta_time=DELTA_TIME, observation_lanes=OBSERVATION_LANES,
                reward_metric=REWARD_METRIC, port=env_port, seed=seed + rank,
            )
        log_file = os.path.join(LOG_PATH, f'monitor_{rank}.csv')
        env = Monitor(env, filename=log_file)
        return env
    return _init

# --- >>>>> MORE COMPLEX FEATURE EXTRACTOR NEURAL NETWORK <<<<< ---
class ComplexNNFeatureExtractor(BaseFeaturesExtractor):
    """
    A more complex feature extractor with more layers, neurons, and dropout.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    :param dropout_prob: (float) Probability for dropout layers.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, dropout_prob: float = 0.2): # Example: Larger features_dim, add dropout param
        super().__init__(observation_space, features_dim)

        n_input_features = observation_space.shape[0]

        print(f"ComplexNNFeatureExtractor Initialized:")
        print(f"  - Input Features: {n_input_features}")
        print(f"  - Output Features Dim (features_dim): {features_dim}")
        print(f"  - Dropout Probability: {dropout_prob}")

        # Define the layers of your more complex network
        # Example: Input -> Linear(256) -> ReLU -> Dropout -> Linear(128) -> ReLU -> Dropout -> Linear(features_dim) -> ReLU
        self.network = nn.Sequential(
            nn.Linear(n_input_features, 256), # Wider first layer
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),      # Add Dropout
            nn.Linear(256, 128),             # Additional hidden layer
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),      # Add Dropout
            nn.Linear(128, features_dim),    # Final layer mapping to features_dim
            nn.ReLU()                        # Activation after final feature extraction layer
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Dropout is automatically handled (active during training, inactive during eval)
        return self.network(observations)
# --- >>>>> END OF COMPLEX FEATURE EXTRACTOR <<<<< ---

# --- Main Execution Block ---
if __name__ == "__main__":
    start_time = time.time()
    base_seed = int(time.time())

    print("--- Setting up Vectorized Environment ---")
    port_step = 100
    env_fns = [create_env(i, seed=base_seed, port_offset=port_step * i) for i in range(N_CPU)]
    try:
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    except Exception as e: print(f"Error creating SubprocVecEnv: {e}"); exit()

    print(f"Observation Space: {vec_env.observation_space}")
    print(f"Action Space: {vec_env.action_space}")


    # --- Policy Kwargs using the COMPLEX Feature Extractor ---
    policy_kwargs = dict(
        features_extractor_class=ComplexNNFeatureExtractor, # <<< Use the new complex class
        features_extractor_kwargs=dict(
            features_dim=128,          # <<< Output dimension of ComplexNNFeatureExtractor
            dropout_prob=0.2           # <<< Pass dropout probability
            ),
        # activation_fn=nn.ReLU, # Keep default ReLU for heads unless desired otherwise
        # Optional: Define structure of Policy (pi) and Value (vf) heads AFTER the feature extractor
        # These heads take the output of the feature extractor (features_dim=128) as input
        net_arch=dict(pi=[64], vf=[64]) # Example: Heads: features(128) -> linear(64) -> output
    )
    # --- End Policy Kwargs ---


    print("--- Setting up Model with Complex Custom Network ---")
    model = MODEL_ALGORITHM(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs, # Pass the custom network definition
        verbose=1,
        tensorboard_log=LOG_PATH,
        # --- Hyperparameters (CRUCIAL tuning needed for complex nets) ---
        learning_rate=1e-4,  # <<< Try smaller learning rate
        n_steps=2048,        # Might need adjusting (e.g., larger)
        batch_size=64,       # Might need adjusting (e.g., larger if memory allows)
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,      # Consider tuning (e.g., 0.1)
        ent_coef=0.005,      # Slightly smaller entropy maybe
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=base_seed,
        device="auto"
    )

    print("--- Model Policy Structure ---")
    print(model.policy) # Check the structure reflects the ComplexNNFeatureExtractor
    print("-" * 30)


    print("--- Setting up Evaluation Environment & Callback ---")
    eval_rank = N_CPU
    eval_port_offset = port_step * eval_rank
    eval_env_lambda = create_env(rank=eval_rank, seed=base_seed, port_offset=eval_port_offset)
    eval_env = Monitor(eval_env_lambda(), filename=os.path.join(LOG_PATH, 'eval_monitor.csv'))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=os.path.join(LOG_PATH, 'eval_logs'),
        eval_freq=max(EVAL_FREQ // N_CPU, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    print("--- Starting Training ---")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            tb_log_name=f"{MODEL_ALGORITHM.__name__}",
            reset_num_timesteps=False
        )
    except KeyboardInterrupt: print("Training interrupted by user.")
    except Exception as e: print(f"An error occurred during training: {e}"); traceback.print_exc()
    finally:
        # --- Training Finished or Interrupted ---
        print("--- Training Finished or Interrupted ---")
        final_model_path = os.path.join(SAVE_PATH, "final_model.zip")
        try:
            model.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
            best_model_zip = os.path.join(BEST_MODEL_SAVE_PATH, 'best_model.zip')
            if os.path.exists(best_model_zip): print(f"Best model saved to {best_model_zip}")
            else: print(f"Best model was not saved.")
        except Exception as e: print(f"Error saving final model: {e}")

        print("Closing environments...")
        try: vec_env.close(); print("Training environments closed.")
        except Exception as e: print(f"Error closing training environments: {e}")
        try: eval_env.close(); print("Evaluation environment closed.")
        except Exception as e: print(f"Error closing evaluation environment: {e}")

        end_time = time.time()
        print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
