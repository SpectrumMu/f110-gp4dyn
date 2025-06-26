from f1tenth_planning import control
from requests import get
from sklearn.metrics import v_measure_score
import torch
import numpy as np
import pickle
from gp_model import MultiOutputGP, MultiOutputSparseGP
from sklearn.model_selection import train_test_split
from utils.utils import prepare, load_yaml_config
from utils.logger import setup_logger

import f1tenth_gym
import f1tenth_gym.envs
import gymnasium as gym

import yaml
import json
import datetime
import os
import logging

import matplotlib.pyplot as plt

def main():
    # === Load configuration ===
    config = load_yaml_config("./config/config.yaml")
    
    global DATADIR, MODELDIR, EVALDIR, LOGDIR, TIME_STAMP
    EVALDIR = config["global"]["eval_folder"]
    # DATADIR = config["global"]["data_folder"]
    LOGDIR = config["global"]["log_folder"]
    LOGDIR = os.path.join(LOGDIR, "eval_logs/")
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    TIME_STAMP = date_time
    EVALDIR = os.path.join(EVALDIR, f"eval_{date_time}/")
    if not os.path.exists(EVALDIR):
        os.makedirs(EVALDIR)
        
    # SKIP LOGGING
    
    # IF_NORM = config["eval_sim"]["if_norm"]
    MODEL_TYPE = int(config["eval_sim"]["model_type"])
    # EVAL_TYPE = int(config["eval_sim"]["eval_type"])
    name = "multioutput" if MODEL_TYPE == 0 else "sparse" if MODEL_TYPE == 1 else "stochastic_variational"
    
    model_name = "gp_model" + ("_" + name if name else "")
    model_name += ".pkl"
    scaler_name = "scaler.pkl"

    MODELDIR = config["eval_sim"]["model_dir"]
    
    logger = setup_logger(
        name="eval_logger",
        log_dir=LOGDIR
    )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # # === Load model ===
    loaded_model = None
    x_scaler = None
    y_scaler = None
    with open(MODELDIR + model_name, "rb") as f:
        loaded_model = pickle.load(f)
    with open(MODELDIR + scaler_name, "rb") as f:
        scalers = pickle.load(f)
        x_scaler = scalers["x_scaler"]
        y_scaler = scalers["y_scaler"]
    # loaded_model.to(device)

    logger.info("Model loaded.")
    
    V_DESIGN = 6.0
    
    total_controls = []
    total_states = []
    for _ in range(10):
        states = []
        controls = []
        steers = get_steers(500, peak_num=10)
        
        step_count = 0
        steering_count = 0
        
        env_config = {
            "seed": 12345,
            "map": "example_map",
            "map_scale": 1.0,
            "params": f1tenth_gym.envs.f110_env.F110Env.f1tenth_vehicle_params(),
            "num_agents": 1,
            "timestep": 0.01,
            "integrator_timestep": 0.01,
            "ego_idx": 0,
            "max_laps": 'inf',  # 'inf' for infinite laps, or a positive integer
            "integrator": "rk4",
            "model": "st", # "ks", "st", "mb"
            "control_input": ["accl", "steering_angle"],
            "observation_config": {"type": "direct"},
            "reset_config": {"type": None},
            "enable_rendering": False,
            "enable_scan": False, # NOTE no lidar scan and collision if False
            "lidar_fov" : 4.712389,
            "lidar_num_beams": 1080,
            "lidar_range": 30.0,
            "lidar_noise_std": 0.01,
            "steer_delay_buffer_size": 1,
            "compute_frenet": True, 
            "collision_check_method": "bounding_box", # "lidar_scan", "bounding_box"
            "loop_counting_method": "frenet_based", # "toggle", "frenet_based", "winding_angle"
        }
        
        env = gym.make(
                'f1tenth_gym:f1tenth-v0',
                config=env_config, 
        )
        
        print(steers.shape)
    
        vel = V_DESIGN + np.random.uniform(-2.0/2, 2.0/2)
        obs, env = warm_up(env, vel, 10000)
        
        for i in range(500):
            v_ = get_obs_vel(obs)
            action = np.array([[steers[i], (vel - v_) * 0.5]])
            # print(action)
            
            state_st_1 = get_state(obs)
            states.append(state_st_1)
            controls.append(action)
            
            obs, _, _, _, _ = env.step(action)
            
        states = np.array(states)
        controls = np.array(controls)

        # Store the collected states and controls for later analysis
        total_states.append(states)
        total_controls.append(controls)

    results = []
    for states, controls in zip(total_states, total_controls):

        print(states.shape, controls.shape)

        states = states.reshape((states.shape[0], states.shape[2]))
        controls = controls.reshape((controls.shape[0], controls.shape[2]))
        x = np.concatenate((states[:-1, 2:], controls[:-1]), axis=-1)
        # print(x.shape)
        y = (states[1:, 2:] - states[:-1, 2:])/0.1
        # print(y.shape)
        
        # plt.figure(figsize=(10, 5))
        # plt.scatter(states[:,0], states[:,1])
        # plt.savefig(f"{EVALDIR}/state_scatter_{i}.png")
        # print(states[:,0])

        X_test, Y_test = x, y
        X_test = x_scaler.transform(X_test)
        Y_test = y_scaler.transform(Y_test)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        # Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

        # Forward pass through the model
        Y_pred, Y_std, _, _ = loaded_model.predict(X_test)

        Y_pred = Y_pred.cpu().numpy()
        Y_std = Y_std.cpu().numpy()
        
        Y_pred = y_scaler.inverse_transform(Y_pred)
        Y_std = y_scaler.inverse_transform(Y_std)
        Y_test = y_scaler.inverse_transform(Y_test)

        results.append((Y_pred, Y_std, Y_test))

    # fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    for i in range(len(results)):
        Y_pred, Y_std, Y_test = results[i]
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        print(f"Result {i}:")
        for j in range(5):
            y_true = Y_test[:, j]
            y_pred = Y_pred[:, j]
            y_uncert = Y_std[:, j]
            error = np.abs(y_true - y_pred)
            
            ax = axes[j]
            ax.plot(y_true, label="True")
            ax.plot(y_pred, label="Predicted")
            ax.fill_between(np.arange(len(y_pred)), y_pred - y_uncert, y_pred + y_uncert, alpha=0.2)
            ax.set_title(f"Output {j}")
            ax.legend()
            print(f"Output {j}: Mean Error = {np.mean(error):.4f}, Stddev = {np.std(error):.4f}")
        plt.tight_layout()
        plt.savefig(f"{EVALDIR}/result_{i}.png")
        

    # for i in range(5):
    #     ax_scatter = axes[i]
    #     ax_scatter.set_xlabel("Predicted Stddev (Uncertainty)")
    #     ax_scatter.set_ylabel("Absolute Error")
    #     ax_scatter.set_title(f"Output {i}: Error vs Uncertainty")
    #     ax_scatter.grid(True)


    # plt.tight_layout()
    # plt.savefig(f"{EVALDIR}/all_outputs_error_vs_uncertainty.png")
    # plt.show()

def get_steers(sample_length, segment_length=1, peak_num=200):
    """
    Generate a random steering signal using a sum of sine waves.
    This function creates a synthetic steering signal by summing multiple sine waves with random amplitudes,
    frequencies, and phases. The resulting signal is normalized to the range [-1, 1], scaled by the maximum
    steering value specified in `params`, and clipped to the minimum and maximum steering limits.
    Args:
        sample_length (int): The total length of the steering signal to generate.
        params (dict): Dictionary containing steering parameters:
            - 's_max' (float): Maximum steering value.
            - 's_min' (float): Minimum steering value.
        segment_length (int, optional): The length of each segment in the signal. Defaults to 1.
        peak_num (int, optional): The number of sine wave components (peaks) to sum. Defaults to 200.
    Returns:
        np.ndarray: The generated steering signal as a 1D numpy array of length `sample_length // segment_length`.
    # This function synthesizes a random, smooth steering profile by combining multiple sine waves,
    # then normalizes and scales the result to fit within specified steering limits.
    """
    
    length = int(sample_length // segment_length)

    x = np.linspace(0, 1, length)
    y = np.zeros_like(x)

    for _ in range(int(peak_num)):
        amplitude = np.random.rand() 
        frequency = np.random.randint(1, peak_num)
        phase = np.random.rand() * 2 * np.pi 

        y += amplitude * np.sin(2 * np.pi * frequency * x + phase)

    y = y - np.min(y)
    y = y / np.max(y)
    y = y * 2 - 1 # scale to -1 to 1
    
    # rand_steer = truncnorm.rvs(-4.0, 4.0, size=1)[0] * 0.1
    # y += rand_steer
    y = y * 0.9
    y = np.clip(y, -0.9, 0.9)
    # plt.plot(np.arange(y.shape[0]), y)
    # plt.show()
    return y

def get_obs_vel(obs):
    """
    Get the velocity from the observation
    :param obs: observation dictionary
    :return: velocity
    """
    states = get_state(obs)
    vx = states[0, 3]  # x velocity
    return vx

def get_state(obs):
    """
    Get the state from the observation
    :param env: environment
    :param obs: observation dictionary
    :return: state vector
    """
    # State vector format:
    # [x position, y position, yaw angle, steering angle, velocity, yaw rate, slip angle]
    # dict_keys(['scan', 'std_state', 'state', 'collision', 'lap_time', 'lap_count', 'sim_time', 'frenet_pose'])
    state = np.asarray(obs['agent_0']['std_state'])
    state = state.reshape((1, 7))  # Ensure state is a 2D array with shape (1, 7)
    return state

def warm_up(env, vel, warm_up_steps):
    """
    Gradually accelerates the environment's vehicle to a target velocity.
    This function resets the simulation environment to an initial pose and then
    repeatedly applies control inputs to the vehicle until its longitudinal velocity
    (`linear_vels_x`) is within 0.5 units of the desired target velocity (`vel`).
    The acceleration is computed based on the difference between the current and target velocities.
    The function returns the final observation and the environment.
    Args:
        env: The simulation environment object, which must implement `reset` and `step` methods.
        vel (float): The target velocity to reach during the warm-up phase.
        warm_up_steps (int): The maximum number of warm-up steps (currently unused in the function).
    Returns:
        obs (dict): The final observation from the environment after warm-up.
        env: The environment object, potentially updated after the warm-up process.
    """
    init_pose = np.zeros((1, 3))

    # [x, y, steering angle, velocity, yaw, yaw_rate, beta]
    obs, _ = env.reset(
        # np.array([[0.0, 0.0, 0.0, 0.0, vel/1.1, 0.0, 0.0]])
        options={
            "poses": init_pose,
            # "states": np.array([[0.0, 0.0, 0.0, vel/1.05, 0.0, 0.0, 0.0]]),
        }
    )

    # return obs, env

    # The following function is not used for latest gym
    step_count = 0
    state_v = 0
    while (abs(state_v - vel) > 0.5):
        try:
            accel = (vel - state_v) * 0.7
            u_1 = accel
            obs, _, _, _, _ = env.step(np.array([[0.0, u_1]]))
            state_v = get_obs_vel(obs)
            # print(, obs['linear_vels_y'][0], get_obs_vel(obs), vel)
            # print(step_count)
            step_count += 1
            # print('warmup step: ', step_count, 'error', get_obs_vel(obs), vel)
        except ZeroDivisionError:
            print('error warmup: ', step_count)
    # print('warmup step: ', step_count, 'error', get_obs_vel(obs), vel)
    return obs, env


if __name__ == "__main__":
    main()