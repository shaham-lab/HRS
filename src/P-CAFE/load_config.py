import json
import os


def deep_update(base_dict, update_dict):
    """
    Recursively update base_dict with values from update_dict.
    
    Args:
        base_dict (dict): The base dictionary to update
        update_dict (dict): The dictionary with updates
    
    Returns:
        dict: The updated dictionary
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_hierarchical_config(base_config_path="base_config.json", 
                            user_config_path="user_config.json",
                            cli_args=None):
    """
    Load configuration with hierarchical override:
    1. Start with base_config.json
    2. Override with user_config.json (if exists)
    3. Override with command-line arguments (if provided)
    
    Args:
        base_config_path (str): Path to base configuration file
        user_config_path (str): Path to user configuration file
        cli_args (dict): Dictionary of command-line arguments to override config
    
    Returns:
        dict: Merged configuration dictionary
    """
    # Load base configuration
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Override with user configuration if it exists
    if os.path.exists(user_config_path):
        with open(user_config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = deep_update(config, user_config)
    
    # Override with command-line arguments if provided
    if cli_args is not None:
        config = deep_update(config, cli_args)
    
    return config