import gymnasium
from ..Guesser.multimodal_guesser import MultimodalGuesser
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np

class myEnv(gymnasium.Env):
    def __init__(self,
                 flags,
                 device, cost_budget,
                 load_pretrained_guesser=True):

        self.guesser = MultimodalGuesser(flags)
        self.device = device
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.guesser.X, self.guesser.y,
                                                                                test_size=0.1, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.05, random_state=24)
        # self.cost_list= [1,2,6,1,1,1,1,7,1,1,2,7,2,2,1,1,7,1]
        self.cost_list = [1] * (self.guesser.tests_number + 1)
        self.prob_list = [cost / sum(self.cost_list) for cost in self.cost_list]
        self.cost_budget = cost_budget
        self.num_classes = self.guesser.num_classes
        # Load pre-trained guesser network, if needed
        if load_pretrained_guesser:
            save_dir = os.path.join(os.getcwd(), flags.save_guesser_dir)
            guesser_filename = 'best_guesser.pth'
            guesser_load_path = os.path.join(save_dir, guesser_filename)
            if os.path.exists(guesser_load_path):
                print('Loading pre-trained guesser')
                guesser_state_dict = torch.load(guesser_load_path)
                self.guesser.load_state_dict(guesser_state_dict)

    def reset(self,
              mode='training',
              patient=0,
              train_guesser=True):
        self.state = np.concatenate([np.zeros(self.guesser.features_total)])
        if mode == 'training':
            self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        self.done = False
        self.s = np.array(self.state)
        self.time = 0
        if mode == 'training':
            self.train_guesser = train_guesser
        else:
            self.train_guesser = False
        return self.s

    def reset_mask(self):
        """ A method that resets the mask that is applied
        to the q values, so that questions that were already
        asked will not be asked again.
        """
        mask = torch.ones(self.guesser.tests_number + 1)
        mask = mask.to(device=self.device)
        # for i in range(self.guesser.tests_number):
        #     if random.random() < 0.8:
        #         mask[i] = 0
        return mask

    def step(self,
             action, mask,
             mode='training'):
        """ State update mechanism """

        # update state
        next_state = self.update_state(action, mode)
        self.state = np.array(next_state)
        self.s = np.array(self.state)

        # compute reward
        self.reward = self.compute_reward(mode)

        self.time += 1
        if self.time == self.guesser.tests_number:
            self.terminate_episode()

        return self.s, self.reward, self.done, self.guess

    # Update 'done' flag when episode terminates
    def terminate_episode(self):

        self.done = True

    def prob_guesser(self, state):
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
        guesser_input = torch.Tensor(
            state[:self.guesser.features_total])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        return self.guesser(guesser_input).squeeze()[1].item()

    def update_state(self, action, mode):
        prev_state = self.s
        next_state = np.array(self.state)
        if action < self.guesser.tests_number:  # Not making a guess
            features_revealed = self.guesser.map_test[action]
            for feature in features_revealed:
                if mode == 'training':
                    answer = self.X_train.iloc[self.patient, feature]
                elif mode == 'val':
                    answer = self.X_val.iloc[self.patient, feature]
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

            self.prob_classes = self.prob_guesser_for_positive(next_state)
            self.reward = abs(self.prob_guesser(next_state) - self.prob_guesser(prev_state)) / self.cost_list[action]
            self.guess = -1
            self.done = False
            return next_state

        else:
            self.prob_classes = self.prob_guesser_for_positive(prev_state)
            self.reward = self.prob_guesser(prev_state)
            self.terminate_episode()
            return prev_state

    def compute_reward(self, mode):
        """ Compute the reward """

        if mode == 'test':
            return None

        if self.guess == -1:  # no guess was made
            return self.reward

        if mode == 'training':
            y_true = self.y_train[self.patient]
            if self.train_guesser:
                self.guesser.optimizer.zero_grad()
                self.guesser.train(mode=True)
                y_tensor = torch.tensor([int(y_true)])
                y_true_tensor = F.one_hot(y_tensor, num_classes=2).squeeze()
                self.probs = self.probs.float()
                y_true_tensor = y_true_tensor.float()
                self.guesser.loss = self.guesser.criterion(self.probs, y_true_tensor)
                self.guesser.loss.backward()
                self.guesser.optimizer.step()

        return self.reward

    def embed_text(self, text):
        tokens = self.tokenizer(str(text), padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = self.text_model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return F.relu(self.text_reducer(embeddings))

    def embed_image(self, image_path):
        embedding = self.img_embedder.embed_image(image_path)
        return F.relu(self.img_reducer(embedding))

    def is_numeric_value(self, value):
        # Check if the value is an integer, a floating-point number, or a tensor of type float or double
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, torch.Tensor):
            if value.dtype in [torch.float, torch.float64]:
                return True
        return False

    def is_text_value(self, value):
        # Check if the value is a string
        if isinstance(value, str):
            return True
        else:
            return False

    def is_image_value(self, value):
        # check if value is path that ends with 'png' or 'jpg'
        if isinstance(value, str):
            if value.endswith('png') or value.endswith('jpg'):
                return True
            else:
                return False
