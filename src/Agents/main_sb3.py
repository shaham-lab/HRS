from env import *
from stable_baselines3 import PPO, TD3, DQN, A2C
import torch
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
from ..common.parse_args import parse_arguments

# Hardcoded device detection
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ======== AGENT DEFINITIONS ========


def PPO_agent(flags):
    """
    Create and train a PPO agent in the custom environment.
    Args:
        flags: Configuration flags
    Returns:
        model: Trained PPO model
        env: The training environment
    """
    env = myEnv(flags=flags, device=DEVICE)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env





# Define agent for DQN
def DQN_agent(flags):
    """
    Create and train a DQN agent in the custom environment.
    Args:
        flags: Configuration flags
    Returns:
        model: Trained DQN model
        env: The training environment
    """
    env = myEnv(flags=flags, device=DEVICE)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env


# Define agent for A2C
def A2C_agent(flags):
    """
    Create and train an A2C agent in the custom environment.
    Args:
        flags: Configuration flags
    Returns:
        model: Trained A2C model
        env: The training environment
    """
    env = myEnv(flags=flags, device=DEVICE)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env


# Define agent for TD3
def TD3_agent(flags):
    """
    Create and train a TD3 agent in the custom environment.
    Args:
        flags: Configuration flags
    Returns:
        model: Trained TD3 model
        env: The training environment
    """
    env = myEnv(flags=flags, device=DEVICE)
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env


def test(env, model, agent) :
    """
       Evaluate the performance of a trained agent on a test set.

       Args:
           env: The environment configured in test mode
           model: The trained RL model
           agent: The type of agent (string identifier, not used in this snippet)

       Returns:
           float: Placeholder for performance metric (e.g., accuracy or reward)
       """
    print('Running Test')
    y_hat_test = np.zeros(len(env.y_test))
    y_hat_probs = np.zeros(len(env.y_test))
    cost_list = []
    for i in range(len(env.X_test)):
        state, _ = env.reset(mode='test', patient=i)
        terminated = False
        sum_cost = 0
        while not terminated and sum_cost < env.cost_budget:
            # Select action from PPO model
            action, _states = model.predict(state, deterministic=True)
            action = int(action.item())
            # If the selected action exceeds the cost budget, force the guess
            if sum_cost + env.cost_list[action] > env.cost_budget:
                action = model.action_space.n - 1  # Assuming last action is "guess"

            state, reward, terminated, nan, info = env.step(torch.tensor([action]), 'test')
            # Handle guessing
            if info['guess'] != -1:
                y_hat_test[i] = info['guess']
                y_hat_probs[i] = env.prob_classes

            sum_cost += env.cost_list[action]

        # Final guessing logic if not already guessed
        if info['guess'] == -1:
            action = model.action_space.n - 1  # Assuming last action is "guess"
            state, reward, terminated, nan, info = env.step(torch.tensor([action]), 'test')
            y_hat_test[i] = info['guess']
            y_hat_probs[i] = env.prob_classes

        cost_list.append(sum_cost)
    # calculate auc score
    conf_mat = confusion_matrix(env.y_test, y_hat_test)
    print(conf_mat)
    acc = np.sum(np.diag(conf_mat)) / len(env.y_test)
    print(f"Test accuracy:{agent}", np.round(acc, 3))
    avg_cost = np.mean(cost_list)
    print('Average cost: {:1.3f}'.format(avg_cost))


def main():
    FLAGS = parse_arguments()
    model, env = PPO_agent(FLAGS)
    test(env, model, 'PPO')
    model, env = DQN_agent(FLAGS)
    test(env, model, 'DQN')
    model, env = A2C_agent(FLAGS)
    test(env, model, 'A2C')
    model, env = TD3_agent(FLAGS)
    test(env, model, 'TD3')



if __name__ == '__main__':
    main()
