import os
import yaml
import datetime


def path_check(path: str) -> None:
    """
    Check if the given path exists, and create it if it does not.
    
    Args:
        path (str): The path to check.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    
    Args:
        config_path (str): The path to the YAML configuration file.
        
    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_config_value(config: dict, key: str, default=None):
    """
    Get a value from the configuration dictionary.
    
    Args:
        config (dict): The configuration dictionary.
        key (str): The key to look for in the configuration.
        default: The default value to return if the key is not found.
        
    Returns:
        The value associated with the key, or the default value if the key is not found.
    """
    return config.get(key, default)

def prepare(config: dict, mode: str, if_compare=0) -> dict:
    """
    Create a directory path based on the configuration.
    
    Args:
        config (dict): The configuration dictionary.
        mode (str): The mode for preparation (e.g., "train", "eval").

    Returns:
        dict: A dictionary containing the created directory paths.
    """

    dicts = {}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dicts["timestamp"] = timestamp

    model_dir = config["global"]["model_folder"]
    eval_folder = config["global"]["eval_folder"]
    log_folder = config["global"]["log_folder"]

    path_check(model_dir)
    path_check(eval_folder)
    path_check(log_folder)
    
    if if_compare:
        model_dir += "compare/"
        path_check(model_dir)
        model_dir += f"{timestamp}/"
        path_check(model_dir)
        dicts["model_dir"] = model_dir
    else:
        if config["gp_train"]["model"]["type"] == 0:
            model_dir += "multioutput/"
        elif config["gp_train"]["model"]["type"] == 1:
            model_dir += "sparse/"
        elif config["gp_train"]["model"]["type"] == 2:
            model_dir += "stochastic_variational/"
        path_check(model_dir)
        model_dir += f"{timestamp}/"
        path_check(model_dir)
        dicts["model_dir"] = model_dir
    
    eval_folder += f"{mode}_{timestamp}/"
    path_check(eval_folder)
    dicts["eval_folder"] = eval_folder

    log_folder += f"{mode}_{timestamp}/"
    path_check(log_folder)
    dicts["log_folder"] = log_folder
    
    

    # dicts["logger"] = logger

    return dicts

