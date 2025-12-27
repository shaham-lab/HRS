from src.Agents.env import *
from stable_baselines3 import PPO, TD3, DQN, A2C
import torch
import numpy as np

with open(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\Integration\user_config_naama.json', 'r') as f:
    config = json.load(f)

# Get the project path from the JSON
project_path = Path(config["user_specific_project_path"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default=project_path,
                    help="Directory for saved models")
parser.add_argument("--save_dir",
                    type=str,
                    default='ddqn_robust_models',
                    help="Directory for saved models")
parser.add_argument("--gamma",
                    type=float,
                    default=0.9,
                    help="Discount rate for Q_target")
parser.add_argument("--n_update_target_dqn",
                    type=int,
                    default=50,
                    help="Number of episodes between updates of target dqn")
parser.add_argument("--ep_per_trainee",
                    type=int,
                    default=1000,
                    help="Switch between training dqn and guesser every this # of episodes")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Mini-batch size")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=64,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=1000000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=2000,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min_epsilon",
                    type=float,
                    default=0.01,
                    help="Min epsilon")
parser.add_argument("--initial_epsilon",
                    type=float,
                    default=1,
                    help="init epsilon")
parser.add_argument("--anneal_steps",
                    type=float,
                    default=1000,
                    help="anneal_steps")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-4,
                    help="l_2 weight penalty")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")

# change these parameters
parser.add_argument("--val_interval",
                    type=int,
                    default=100,
                    help="Interval for calculating validation reward and saving model")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=5,
                    help="Number of validation trials without improvement")
parser.add_argument("--device",
                    type=str,
                    default=device,
                    help="Device for training")


FLAGS = parser.parse_args(args=[])


# ======== AGENT DEFINITIONS ========


def PPO_agent():
    """
    Create and train a PPO agent in the custom environment.
    Returns:
        model: Trained PPO model
        env: The training environment
    """
    env = myEnv(flags=FLAGS)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env





# Define agent for DQN
def DQN_agent():
    """
    Create and train a DQN agent in the custom environment.
    Returns:
        model: Trained DQN model
        env: The training environment
    """
    env = myEnv(flags=FLAGS)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env


# Define agent for A2C
def A2C_agent():
    """
    Create and train an A2C agent in the custom environment.
    Returns:
        model: Trained A2C model
        env: The training environment
    """
    env = myEnv(flags=FLAGS)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env


# Define agent for TD3
def TD3_agent():
    """
    Create and train a TD3 agent in the custom environment.
    Returns:
        model: Trained TD3 model
        env: The training environment
    """
    env = myEnv(flags=FLAGS)
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model, env


def test(env, model, agent) -> float:
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
    C = confusion_matrix(env.y_test, y_hat_test)
    print(C)
    acc = np.sum(np.diag(C)) / len(env.y_test)
    print(f"Test accuracy:{agent}", np.round(acc, 3))
    avg_cost = np.mean(cost_list)
    print('Average cost: {:1.3f}'.format(avg_cost))


def main():
    os.chdir(FLAGS.directory)
    model, env = PPO_agent()
    test(env, model, 'PPO')
    model, env = DQN_agent()
    test(env, model, 'DQN')
    model, env = A2C_agent()
    test(env, model, 'A2C')
    model, env = TD3_agent()
    test(env, model, 'TD3')



if __name__ == '__main__':
    main()
