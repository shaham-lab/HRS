import shutil
import torch.nn
from typing import List, Tuple
from .env_robust import *
from .agent import *
from .PrioritiziedReplayMemory import *
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
#import time
import os
from ..Guesser.multimodal_guesser import MultimodalGuesser
from ..common.parse_args import parse_arguments

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_helper(agent: Agent,
                 minibatch: List[Transition],
                 gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(Q_pred, Q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])
    states = np.swapaxes(states, 0, 1)
    next_states = np.swapaxes(next_states, 0, 1)
    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().cpu().data.numpy()
    max_actions = np.argmax(agent.get_Q(next_states).cpu().data.numpy(), axis=1)
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * agent.get_target_Q(next_states)[
        np.arange(len(Q_target)), max_actions].data.numpy() * ~done
    Q_target = agent._to_variable(Q_target).to(device=DEVICE)
    return agent.train(Q_predict, Q_target)


def calculate_td_error(state, action, reward, next_state, done, agent, gamma):
    """Calculate temporal difference error for prioritized experience replay.
    
    Args:
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether episode is done
        agent: Agent instance
        gamma: Discount factor
        
    Returns:
        float: TD error value
    """
    # Current Q-value estimate
    current_q_value = agent.get_Q(state).squeeze()[action]
    if done:
        target_q_value = reward
    else:
        next_q_values = agent.get_target_Q(next_state).squeeze()
        max_next_q_value = max(next_q_values)
        target_q_value = reward + gamma * max_next_q_value

    # TD error
    td_error = target_q_value - current_q_value
    return td_error.item()


def play_episode(env,
                 agent: Agent,priorityRM: PrioritizedReplayMemory,
                 eps: float,
                 batch_size: int,
                 gamma: float,
                 train_guesser=True,
                 train_dqn=True, mode='training') -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """
    s = env.reset(train_guesser=train_guesser)
    done = False
    total_reward = 0
    mask = env.reset_mask()
    t = 0
    sum_cost = 0
    while not done and sum_cost < env.cost_budget:
        a = agent.get_action(s, env, eps, mask, mode)
        # a = agent.get_action_not_guess(s, env, eps, mask, mode)
        if sum_cost+env.cost_list[a] > env.cost_budget:
            a = agent.output_dim - 1
        next_state, r, done, info = env.step(a, mask)
        mask[a] = 0
        total_reward += r
        td = calculate_td_error(s, a, r, next_state, done, agent, gamma)
        priorityRM.push(s, a, r, next_state, done, td)
        if len(priorityRM) > batch_size:
            if train_dqn:
                minibatch, indices, weights = priorityRM.pop(batch_size)
                td_errors = []
                for transition, weight in zip(minibatch, weights):
                    state, action, reward, next_state, done = transition
                    td_error = calculate_td_error(state, action, reward, next_state, done, agent, gamma)
                    td_errors.append(td_error)
                priorityRM.update_priorities(indices, td_errors)
                train_helper(agent, minibatch, gamma)
                agent.update_learning_rate()

        t += 1
        sum_cost += env.cost_list[a]

        if torch.sum(mask) == 0:
            done = True
    return total_reward, t


def get_env_dim(env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.guesser.tests_number
    output_dim = env.guesser.tests_number + 1

    return input_dim, output_dim


def epsilon_annealing(initial_epsilon, min_epsilon, anneal_steps, current_step):
    """
    Epsilon annealing function for epsilon-greedy exploration in reinforcement learning.

    Parameters:
    - initial_epsilon: Initial exploration rate
    - min_epsilon: Minimum exploration rate
    - anneal_steps: Number of steps over which to anneal epsilon
    - current_step: Current step in the learning process

    Returns:
    - epsilon: Annealed exploration rate for the current step
    """
    epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * current_step / anneal_steps)
    return epsilon


def save_networks(i_episode: int, env, agent,
                  save_dir: str,
                  val_acc=None) -> None:
    """ A method to save parameters of guesser and dqn """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if i_episode == 'best':
        dqn_filename = 'best_dqn.pth'
        # Use save_model for best guesser
        env.guesser.save_model()
    else:
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_acc)
        # Use save_temp_guesser for temporary guesser
        env.guesser.save_temp_guesser(i_episode, val_acc)

    dqn_save_path = os.path.join(save_dir, dqn_filename)

    # save dqn
    if os.path.exists(dqn_save_path):
        os.remove(dqn_save_path)
    torch.save(agent.dqn.cpu().state_dict(), dqn_save_path + '~')
    agent.dqn.to(device=DEVICE)
    os.rename(dqn_save_path + '~', dqn_save_path)


def load_networks(i_episode: int, save_dir: str, FLAGS, state_dim=26, output_dim=14,
                  hidden_dim=64, val_acc=None) :
    """ A method to load parameters of guesser and dqn """
    if i_episode == 'best':
        dqn_filename = 'best_dqn.pth'
    else:
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_acc)

    dqn_load_path = os.path.join(save_dir, dqn_filename)

    # load guesser - use FLAGS which now contains all configurations
    guesser = MultimodalGuesser(FLAGS)
    if i_episode == 'best':
        # Use load_model for best guesser
        guesser.load_model()
    else:
        # Use load_temp_model for temporary guesser
        guesser.load_temp_model(i_episode, val_acc)

    # load sqn
    dqn = DQN(state_dim, output_dim, hidden_dim)
    dqn_state_dict = torch.load(dqn_load_path)
    dqn.load_state_dict(dqn_state_dict)
    dqn.to(device=DEVICE)
    return guesser, dqn


def test(env, agent, state_dim, output_dim, save_dir, hidden_dim, FLAGS):
    total_steps = 0
    mask_list = []
    cost_list = []
    print('Loading best networks')
    env.guesser, agent.dqn = load_networks(i_episode='best', save_dir=save_dir, FLAGS=FLAGS,
                                           state_dim=state_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    y_hat_test = np.zeros(len(env.y_test))
    y_hat_probs = np.zeros(len(env.y_test))
    n_test = len(env.X_test)

    for i in range(n_test):
        state = env.reset(mode='test',
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()
        t = 0
        done = False
        sum_cost = 0
        while not done and sum_cost < env.cost_budget:
            if t == 0:
                action = agent.get_action_not_guess(state, env, eps=0, mask=mask, mode='test')
            else:
                action = agent.get_action(state, env, eps=0, mask=mask, mode='test')
            if sum_cost + env.cost_list[action] > env.cost_budget:
                action = agent.output_dim - 1
            mask[action] = 0
            # take the action
            state, reward, done, guess = env.step(action, mask, mode='test')
            if guess != -1:
                y_hat_test[i] = env.guess
                y_hat_probs[i] = env.prob_classes

            t += 1
            sum_cost += env.cost_list[action]
            if torch.sum(mask) == 0:
                done = True

        if guess == -1:
            a = agent.output_dim - 1
            _, _, _, _ = env.step(a, mask)
            y_hat_test[i] = env.guess
            y_hat_probs[i] = env.prob_classes

        not_binary_tensor = 1 - mask
        mask_list.append(not_binary_tensor)
        cost_list.append(sum_cost)

    auc_roc = roc_auc_score(env.y_test, y_hat_probs)
    print(f"AUC-ROC: {auc_roc}")

    auc_pr = average_precision_score(env.y_test, y_hat_probs)
    print(f"AUC-PR: {auc_pr}")
    intersect, union = check_intersection_union(mask_list)
    C = confusion_matrix(env.y_test, y_hat_test)

    print(C)
    acc = np.sum(np.diag(C)) / len(env.y_test)
    steps = np.round(total_steps / n_test, 3)
    print('Test accuracy: ', np.round(acc, 3))
    print('Average number of steps: ', steps)
    avg_cost = np.mean(cost_list)
    print('Average cost: {:1.3f}'.format(avg_cost))
    return acc, intersect, union, steps



def check_intersection_union(mask_list):
    # Convert the list of tensors to a 2D tensor
    selected_features_tensor = torch.stack(mask_list)

    # Compute intersection and union
    intersection_features = torch.prod(selected_features_tensor, dim=0)
    union_features = torch.any(selected_features_tensor, dim=0)

    # Count the number of selected features in the intersection and union
    intersection_size = torch.sum(intersection_features).item()
    union_size = torch.sum(union_features).item()

    print(f"Intersection Size: {intersection_size}")
    print(f"Union Size: {union_size}")
    # Calculate the percentage of samples in which each feature is selected
    percentage_selected = torch.mean(selected_features_tensor.float(), dim=0) * 100
    print("Percentage of Samples Each Feature is Selected:")
    print(percentage_selected)
    return union_size, intersection_size


def val(i_episode: int,
        best_val_acc: float, env, agent, save_dir: str) -> float:
    """ Compute performance on validation set and save current models """

    print('Running validation')
    y_hat_val = np.zeros(len(env.y_val))
    y_hat_probs = np.zeros(len(env.y_val))
    cost_list = []

    for i in range(len(env.X_val)):
        state = env.reset(mode='val',
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()
        t = 0
        done = False
        sum_cost = 0
        #start_time = time.time()
        while not done and sum_cost < env.cost_budget:
            # select action from policy
            if t == 0:
                action = agent.get_action_not_guess(state, env, eps=0, mask=mask, mode='val')
            else:
                action = agent.get_action(state, env, eps=0, mask=mask, mode='val')
            if sum_cost + env.cost_list[action] > env.cost_budget:
                action = agent.output_dim - 1

            mask[action] = 0
            # take the action
            state, reward, done, guess = env.step(action, mask, mode='val')
            if guess != -1:
                y_hat_val[i] = guess
                y_hat_probs[i] = env.prob_classes
            t += 1
            sum_cost += env.cost_list[action]
            if torch.sum(mask) == 0:
                done = True

        if guess == -1:
            a = agent.output_dim - 1
            _, _, _, _ = env.step(a, mask)
            y_hat_val[i] = env.guess
            y_hat_probs[i] = env.prob_classes

        cost_list.append(sum_cost)
        #end_time = time.time()
        #execution_time = (end_time - start_time)
        # print(f"Execution time: {execution_time:.6f} seconds")

    auc_roc = roc_auc_score(env.y_val, y_hat_probs)
    print(f"AUC-ROC: {auc_roc}")

    auc_pr = average_precision_score(env.y_val, y_hat_probs)
    print(f"AUC-PR: {auc_pr}")
    confmat = confusion_matrix(env.y_val, y_hat_val)
    print(confmat)
    acc = np.sum(np.diag(confmat)) / len(env.y_val)
    print('Validation accuracy: {:1.3f}'.format(acc))
    avg_cost = np.mean(cost_list)
    print('Average cost: {:1.3f}'.format(avg_cost))
    if acc >= best_val_acc:
        print('New best acc acheievd, saving best model')
        save_networks(i_episode, env, agent, save_dir, acc)
        save_networks(i_episode='best', env=env, agent=agent, save_dir=save_dir)
    return acc


def run(FLAGS):
    """Run the training and evaluation process.
    
    Args:
        FLAGS: Parsed command line arguments containing all configuration
        
    Returns:
        Tuple of (accuracy, iterations, intersection, union, steps)
    """
    if os.path.exists(FLAGS.save_dir_ddqn):
        shutil.rmtree(FLAGS.save_dir_ddqn)

    env = myEnv(flags=FLAGS,
                device=DEVICE,cost_budget=FLAGS.cost_budget)
    input_dim, output_dim = get_env_dim(env)
    state_dim= env.guesser.features_total
    agent = Agent(state_dim,
                  output_dim,
                  FLAGS.hidden_dim, FLAGS.lr, FLAGS.weight_decay)

    agent.dqn.to(device=DEVICE)
    env.guesser.to(device=DEVICE)
    # store best result
    best_val_acc = 0
    val_list = []
    # counter of validation trials with no improvement, to determine when to stop training
    val_trials_without_improvement = 0
    priorityRP = PrioritizedReplayMemory(FLAGS.capacity)
    i = 0
    rewards_list = []
    train_dqn = True
    train_guesser = False



    while val_trials_without_improvement < FLAGS.val_trials_wo_im:
        eps = epsilon_annealing(FLAGS.initial_epsilon, FLAGS.min_epsilon, FLAGS.anneal_steps, i)
        # play an episode
        reward, t = play_episode(env,
                                 agent,
                                 priorityRP,
                                 eps,
                                 FLAGS.batch_size,
                                 FLAGS.gamma,
                                 train_dqn=train_dqn,
                                 train_guesser=train_guesser, mode='training')
        rewards_list.append(reward)
        if i % FLAGS.val_interval == 0 and i > 50:
            # compute performance on validation set
            new_best_val_acc = val(i_episode=i,
                                   best_val_acc=best_val_acc, env=env, agent=agent,
                                   save_dir=FLAGS.save_dir_ddqn)
            val_list.append(new_best_val_acc)

            # update best result on validation set and counter
            if new_best_val_acc > best_val_acc:
                best_val_acc = new_best_val_acc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1

        if i % FLAGS.n_update_target_dqn == 0:
            agent.update_target_dqn()
        i += 1

    acc, intersect, unoin, steps = test(env, agent, state_dim, output_dim,
                                        FLAGS.save_dir_ddqn, FLAGS.hidden_dim, FLAGS)
    # show_sample_paths(6, env, agent)
    return acc, i, intersect, unoin, steps

def main():
    """Main entry point for the application."""
    FLAGS = parse_arguments()
    #os.chdir(FLAGS.directory)
    _, _, _, _, _ = run(FLAGS)




if __name__ == '__main__':
    main()

