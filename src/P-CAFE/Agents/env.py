import gymnasium
from ..Guesser.multimodal_guesser import MultimodalGuesser
from gymnasium import spaces
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch


class myEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for feature acquisition and classification.
    Interacts with a MultimodalGuesser model to decide which costly tests to run
    on a patient before making a prediction, balancing cost vs. classification accuracy.
    """
    def __init__(self,
                 flags
                 ):
        """
        Initialize the environment.

        Args:
            flags: Configuration flags (must contain 'device').
        """
        super(myEnv, self).__init__()
        self.guesser = MultimodalGuesser(flags)

        self.action_space = spaces.Discrete(self.guesser.tests_number + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.guesser.features_total,),
            dtype=np.float32
        )

        self.device = flags.device
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.guesser.X, self.guesser.y,
                                                                                test_size=0.05, random_state=42)
        self.cost_list = self.guesser.cost_list
        self.cost_budget = self.guesser.cost_budget
        self.prob_list = [cost / sum(self.cost_list) for cost in self.cost_list]

        self.num_classes = self.guesser.num_classes
        save_dir = self.guesser.path_to_save
        guesser_filename = 'best_guesser.pth'
        guesser_load_path = os.path.join(save_dir, guesser_filename)
        if os.path.exists(guesser_load_path):
            print('Loading pre-trained guesser')
            guesser_state_dict = torch.load(guesser_load_path)
            self.guesser.load_state_dict(guesser_state_dict)

    def reset(self,
              mode='training',
              patient=0,
              train_guesser=True,
              seed=None,
              options=None):
        """
        Reset the environment for a new episode.

        Args:
            mode (str): 'training' or 'test'
            patient (int): Patient index to use (only in test mode)
            train_guesser (bool): Whether to allow guesser training
            seed: Not used (for Gym compatibility)
            options: Not used (for Gym compatibility)

        Returns:
            observation (np.ndarray), info (dict)
        """
        self.state = np.concatenate([np.zeros(self.guesser.features_total)])
        if mode == 'training':
            if isinstance(self.X_train, list):
                self.patient = np.random.randint(len(self.X_train))
            else:
                self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        # reveal state places where cost is 0
        for i in range(self.guesser.tests_number):
            if self.cost_list[i] == 0:
                if isinstance(self.X_train, list):
                    self.state = self.update_state_for_time_series(i, mode)
                else:
                    self.state = self.update_state_basic(i, mode)

        self.done = False
        self.s = np.array(self.state)
        self.time = 0

        if mode == 'training':
            self.train_guesser = train_guesser
        else:
            self.train_guesser = False
        self.total_cost = 0
        self.taken_actions = set()  # Reset the set of taken actions

        info = {}  # Add any relevant environment information here
        return self.state, info

    def step(self, action, mode='training'):
        """
        Take a step in the environment.

        Args:
            action (int): Chosen action
            mode (str): 'training' or 'test'

        Returns:
            next_state (np.ndarray), reward (float), done (bool), truncated (bool), info (dict)
        """
        if isinstance(action, torch.Tensor):
            action_number = int(action.item())
        else:
            action_number = action

        # Filter actions to exclude already-taken ones
        available_actions = [a for a in range(self.action_space.n) if a not in self.taken_actions]

        if not available_actions:  # Check if all actions are taken
            terminated = True
            reward = -1  # Optional: Penalize for exhausting all actions
            info = {'guess': self.guess}
            return self.state, reward, terminated, True, info

        if action_number not in available_actions:
            action_number = np.random.choice(available_actions)  # Randomly choose from available actions

        # Mark the action as taken
        self.taken_actions.add(action_number)

        next_state = self.update_state(action_number, mode)
        self.total_cost += self.cost_list[action_number]
        self.state = np.array(next_state)
        reward = self._compute_internal_reward(mode)

        # Check if the episode should terminate
        terminated = self.total_cost >= self.guesser.tests_number or self.done  # Episode ends naturally
        info = {'guess': self.guess}

        return self.state, reward, terminated, False, info  # Set 'truncated' to False if not relevant

    def prob_guesser(self, state):
        """
        Run guesser model and return probability for correct class.

        Args:
            state (np.ndarray): Current state

        Returns:
            float: Probability of correct class
        """
        guesser_input = torch.Tensor(
            state[:self.guesser.features_total])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        self.probs = self.guesser(guesser_input).squeeze()
        self.guess = torch.argmax(self.probs).item()
        self.correct_prob = self.probs[int(self.y_train[self.patient])].item()
        return self.correct_prob

    def prob_guesser_for_positive(self, state):
        """
        Get probability for positive class (class 1).

        Args:
            state (np.ndarray): Input features

        Returns:
            float: Probability of class 1
        """
        guesser_input = torch.Tensor(
            state[:self.guesser.features_total])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        return self.guesser(guesser_input).squeeze()[1].item()

    def update_state_for_time_series(self, action, mode):
        """
        Update state for time-series input.

        Args:
            action (int): Chosen action
            mode (str): 'training' or 'test'

        Returns:
            np.ndarray: Updated state
        """
        next_state = np.array(self.state)
        input = self.X_train[self.patient]
        if action != 0:
            next_state[action + self.guesser.text_reduced_dim - 1] = input.iloc[-1][
                action - 1]
        else:
            df_history = input.iloc[:-1]  # All rows except the last one
            if df_history.shape[0] > 0:
                x = torch.tensor(df_history.values, dtype=torch.float32, device=self.device).unsqueeze(0)
                x = self.guesser.time_series_embedder(x).squeeze()

            else:
                # If there's no history, append a zero vector instead
                embed_dim = self.guesser.text_reduced_dim
                x = torch.zeros((1, embed_dim), device=self.device).squeeze()
            for i in range(len(x)):
                next_state[i] = x[i]

        return next_state

    def update_state_basic(self, action, mode):
        """
        Update state for basic tabular/image/text input.

        Args:
            action (int): Action index
            mode (str): 'training' or 'test'

        Returns:
            np.ndarray: Updated state
        """
        next_state = np.array(self.state)
        features_revealed = self.guesser.map_test[action]
        for feature in features_revealed:
            if mode == 'training':
                answer = self.X_train.iloc[self.patient, feature]
            elif mode == 'test':
                answer = self.X_test.iloc[self.patient, feature]
            # check type of feature
            if self.is_numeric_value(answer):
                answer_vec = torch.tensor([answer], dtype=torch.float32).unsqueeze(0)
            elif self.is_image_value(answer):
                answer_vec = self.guesser.embed_image(answer)
            elif self.is_text_value(answer):
                answer_vec = self.guesser.embed_text(answer).squeeze()
            else:
                size = len(self.guesser.map_feature[feature])
                answer_vec = [0] * size

            map_index = self.guesser.map_feature[feature]
            for count, index in enumerate(map_index):
                next_state[index] = answer_vec[count]

        return next_state

    def update_state(self, action, mode):
        """
        Update state based on chosen action (acquire feature or guess).

        Args:
            action (int): Action index
            mode (str): 'training' or 'test'

        Returns:
            np.ndarray: New state
        """
        prev_state = np.array(self.state)
        if action < self.guesser.tests_number:  # Not making a guess
            if isinstance(self.X_train, list):
                next_state = self.update_state_for_time_series(action, mode)
            else:
                next_state = self.update_state_basic(action, mode)

            self.prob_classes = self.prob_guesser_for_positive(next_state)
            self.reward = abs(self.prob_guesser(next_state) - self.prob_guesser(prev_state)) / (self.cost_list[
                                                                                                    action] + 1)
            self.guess = -1
            self.done = False
            return next_state

        else:
            self.prob_classes = self.prob_guesser_for_positive(prev_state)
            self.reward = self.prob_guesser(prev_state)
            self.done = True
            return prev_state

    def _compute_internal_reward(self, mode):
        """
               Compute internal reward (skips if test mode).

               Args:
                   mode (str): 'training' or 'test'

               Returns:
                   float or None: Reward or None
               """
        if mode == 'test':
            return None
        return self.reward

    def is_numeric_value(self, value):
        """Check if value is numeric or float tensor."""
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, torch.Tensor):
            if value.dtype in [torch.float, torch.float64]:
                return True
        return False

    def is_text_value(self, value):
        """Check if value is a string (text)."""
        if isinstance(value, str):
            return True
        else:
            return False

    def is_image_value(self, value):
        """Check if value is a path to image file."""
        if isinstance(value, str):
            if value.endswith('png') or value.endswith('jpg'):
                return True
            else:
                return False
