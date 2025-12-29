"""
Unified argument parsing module for HRS project.
Provides hierarchical configuration loading for all components.
"""

import argparse
import os
from pathlib import Path
from .load_config import load_hierarchical_config

# System defaults for embedder/guesser
DEFAULT_NUM_EPOCHS = 100
DEFAULT_VAL_TRIALS_WO_IM_GUESSER = 500
DEFAULT_HIDDEN_DIM1 = 64
DEFAULT_HIDDEN_DIM2 = 32
DEFAULT_TEXT_EMBED_DIM = 768
DEFAULT_REDUCED_DIM = 20
DEFAULT_FRACTION_MASK = 0
DEFAULT_RUN_VALIDATION = 100

# System defaults for main_robust/DDQN
DEFAULT_DDQN_SAVE_DIR = 'models\\ddqn_robust_models\\'
DEFAULT_GUESSER_SAVE_DIR = 'models\\guesser\\'
DEFAULT_GAMMA = 0.9
DEFAULT_N_UPDATE_TARGET_DQN = 50
DEFAULT_EP_PER_TRAINEE = 1000
DEFAULT_BATCH_SIZE = 128
DEFAULT_HIDDEN_DIM = 64
DEFAULT_CAPACITY = 1000000
DEFAULT_MAX_EPISODE = 2000
DEFAULT_MIN_EPSILON = 0.01
DEFAULT_INITIAL_EPSILON = 1
DEFAULT_ANNEAL_STEPS = 1000
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LR_DECAY_FACTOR = 0.1
DEFAULT_VAL_INTERVAL = 100
DEFAULT_VAL_TRIALS_WO_IM = 5
DEFAULT_COST_BUDGET = 17


def parse_embedder_guesser_args(parser, config):
    """
    Add embedder/guesser arguments to the parser.
    
    Args:
        parser: argparse.ArgumentParser instance
        config: Configuration dictionary from hierarchical config loading
        
    Returns:
        Updated parser with embedder/guesser arguments
    """
    # Extract embedder_guesser configuration with fallback to root-level config
    embedder_config = config.get("embedder_guesser", {})
    
    # Get the project path from the JSON configuration
    project_path = Path(config.get("user_specific_project_path", os.getcwd()))
    
    # Add embedder/guesser specific arguments
    parser.add_argument("--hidden-dim1",
                        type=int,
                        default=embedder_config.get("hidden_dim1", DEFAULT_HIDDEN_DIM1),
                        help="Hidden dimension")
    parser.add_argument("--hidden-dim2",
                        type=int,
                        default=embedder_config.get("hidden-dim2", DEFAULT_HIDDEN_DIM2),
                        help="Hidden dimension")
    parser.add_argument("--lr",
                        type=float,
                        default=embedder_config.get("lr", DEFAULT_LR),
                        help="Learning rate")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=embedder_config.get("weight_decay", DEFAULT_WEIGHT_DECAY),
                        help="l_2 weight penalty")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=embedder_config.get("num_epochs", DEFAULT_NUM_EPOCHS),
                        help="number of epochs (can be set in `config\\user_config.json`)")
    parser.add_argument("--input_rel_path",
                        type=str,
                        default=config.get("input_rel_path", "data\\input\\"),
                        help="relative path to input data directory (can be set in `config\\user_config.json`)")
    parser.add_argument("--val_trials_wo_im",
                        type=int,
                        default=embedder_config.get("val_trials_wo_im", DEFAULT_VAL_TRIALS_WO_IM_GUESSER),
                        help="Number of validation trials without improvement")
    parser.add_argument("--fraction_mask",
                        type=float,
                        default=embedder_config.get("fraction_mask", DEFAULT_FRACTION_MASK),
                        help="fraction mask")
    parser.add_argument("--run_validation",
                        type=int,
                        default=embedder_config.get("run_validation", DEFAULT_RUN_VALIDATION),
                        help="after how many epochs to run validation")
    parser.add_argument("--batch_size",
                        type=int,
                        default=embedder_config.get("batch_size", DEFAULT_BATCH_SIZE),
                        help="batch size")
    parser.add_argument("--text_embed_dim",
                        type=int,
                        default=embedder_config.get("text_embed_dim", DEFAULT_TEXT_EMBED_DIM),
                        help="Text embedding dimension")
    parser.add_argument("--reduced_dim",
                        type=int,
                        default=embedder_config.get("reduced_dim", DEFAULT_REDUCED_DIM),
                        help="Reduced dimension for text embedding")
    parser.add_argument("--save_dir",
                        type=str,
                        default=embedder_config.get("save_dir", DEFAULT_GUESSER_SAVE_DIR),
                        help="Directory for Guesser saved models")
    parser.add_argument("--guesser_model_file_name",
                        type=str,
                        default=embedder_config.get("guesser_model_file_name", "best_guesser.pth"),
                        help="Filename for saved guesser model")
    parser.add_argument(
        "--data",
        type=str,
        default=embedder_config.get("data", "load_time_Series"),
        help=(
            "Dataset loader function to use. Options:\n"
            "  load_time_Series        - eICU time series data\n"
            "  load_mimic_text         - MIMIC-III multi-modal (includes text)\n"
            "  load_mimic_time_series  - MIMIC-III numeric time series data"
        )
    )
    
    return parser


def parse_main_robust_args(parser, config):
    """
    Add main_robust/DDQN arguments to the parser.
    
    Args:
        parser: argparse.ArgumentParser instance
        config: Configuration dictionary from hierarchical config loading
        
    Returns:
        Updated parser with main_robust arguments
    """
    # Extract main_robust configuration with fallback to root-level config
    main_robust_config = config.get("main_robust", {})
    
    # Get the project path from the JSON configuration
    project_path = Path(config.get("user_specific_project_path", os.getcwd()))
    
    # Add main_robust specific arguments

    # Note: save_dir might conflict with embedder_guesser's save_dir
    # Since both use the same parameter name, we need to handle this carefully
    # The embedder_guesser already set save_dir, so we'll use save_dir_ddqn for main_robust
    parser.add_argument("--save_dir_ddqn",
                        type=str,
                        default=main_robust_config.get("save_dir_ddqn", DEFAULT_DDQN_SAVE_DIR),
                        help="Directory for saved DDQN models")
    parser.add_argument("--save_guesser_dir",
                        type=str,
                        default=main_robust_config.get("save_guesser_dir", DEFAULT_GUESSER_SAVE_DIR),
                        help="Directory for saved guesser model")
    parser.add_argument("--gamma",
                        type=float,
                        default=main_robust_config.get("gamma", DEFAULT_GAMMA),
                        help="Discount rate for Q_target")
    parser.add_argument("--n_update_target_dqn",
                        type=int,
                        default=main_robust_config.get("n_update_target_dqn", DEFAULT_N_UPDATE_TARGET_DQN),
                        help="Number of episodes between updates of target dqn")
    parser.add_argument("--ep_per_trainee",
                        type=int,
                        default=main_robust_config.get("ep_per_trainee", DEFAULT_EP_PER_TRAINEE),
                        help="Switch between training dqn and guesser every this # of episodes")
    
    # Note: batch_size might conflict with embedder_guesser's batch_size
    # We'll use a different name for main_robust's batch_size
    if not any(action.dest == 'batch_size' for action in parser._actions):
        parser.add_argument("--batch_size",
                            type=int,
                            default=main_robust_config.get("batch_size", DEFAULT_BATCH_SIZE),
                            help="Mini-batch size")
    
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=main_robust_config.get("hidden-dim", DEFAULT_HIDDEN_DIM),
                        help="Hidden dimension")
    parser.add_argument("--capacity",
                        type=int,
                        default=main_robust_config.get("capacity", DEFAULT_CAPACITY),
                        help="Replay memory capacity")
    parser.add_argument("--max-episode",
                        type=int,
                        default=main_robust_config.get("max-episode", DEFAULT_MAX_EPISODE),
                        help="e-Greedy target episode (eps will be the lowest at this episode)")
    parser.add_argument("--min_epsilon",
                        type=float,
                        default=main_robust_config.get("min_epsilon", DEFAULT_MIN_EPSILON),
                        help="Min epsilon")
    parser.add_argument("--initial_epsilon",
                        type=float,
                        default=main_robust_config.get("initial_epsilon", DEFAULT_INITIAL_EPSILON),
                        help="init epsilon")
    parser.add_argument("--anneal_steps",
                        type=float,
                        default=main_robust_config.get("anneal_steps", DEFAULT_ANNEAL_STEPS),
                        help="anneal_steps")
    
    # Note: lr and weight_decay might conflict with embedder_guesser
    # Skip if already added
    if not any(action.dest == 'lr' for action in parser._actions):
        parser.add_argument("--lr",
                            type=float,
                            default=main_robust_config.get("lr", DEFAULT_LR),
                            help="Learning rate")
    if not any(action.dest == 'weight_decay' for action in parser._actions):
        parser.add_argument("--weight_decay",
                            type=float,
                            default=main_robust_config.get("weight_decay", DEFAULT_WEIGHT_DECAY),
                            help="l_2 weight penalty")
    
    parser.add_argument("--lr_decay_factor",
                        type=float,
                        default=main_robust_config.get("lr_decay_factor", DEFAULT_LR_DECAY_FACTOR),
                        help="LR decay factor")
    parser.add_argument("--val_interval",
                        type=int,
                        default=main_robust_config.get("val_interval", DEFAULT_VAL_INTERVAL),
                        help="Interval for calculating validation reward and saving model")
    
    # Note: val_trials_wo_im might conflict with embedder_guesser
    # Skip if already added
    if not any(action.dest == 'val_trials_wo_im' for action in parser._actions):
        parser.add_argument("--val_trials_wo_im",
                            type=int,
                            default=main_robust_config.get("val_trials_wo_im", DEFAULT_VAL_TRIALS_WO_IM),
                            help="Number of validation trials without improvement")
    
    parser.add_argument("--cost_budget",
                        type=int,
                        default=main_robust_config.get("cost_budget", DEFAULT_COST_BUDGET),
                        help="Cost budget for evaluation")
    
    return parser


def parse_arguments():
    """
    Parse command line arguments with defaults from configuration files.
    This function combines arguments from both embedder_guesser and main_robust.
    
    :return: Parsed arguments namespace containing all configuration options
    """
    # Load hierarchical configuration: base_config.json -> user_config.json -> CLI args
    config = load_hierarchical_config(
        base_config_path="config/base_config.json",
        user_config_path="config/user_config.json"
    )
    
    # Create argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add embedder/guesser arguments
    parser = parse_embedder_guesser_args(parser, config)
    
    # Add main_robust arguments
    parser = parse_main_robust_args(parser, config)
    
    return parser.parse_args()
