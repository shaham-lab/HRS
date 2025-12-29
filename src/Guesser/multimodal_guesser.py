import os
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from PIL import Image
from ..common import utils
import pandas as pd
from transformers import AutoModel, AutoTokenizer, \
    BartForConditionalGeneration, BartTokenizer
from .image_embedder import ImageEmbedder
from .lstm_encoder import LSTMEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def map_features_to_indices(data):
    """
    Assigns index mappings to each feature based on whether the feature is text or numeric.
    Text features receive 20 indices, numeric features receive 1 index.

    :param data: Dataset (list, numpy array, or list of DataFrames)
    :return: (index_map: dict, total_features: int)
    """
    index_map = {}
    current_index = 0
    if isinstance(data[0], pd.DataFrame):
        for col_index in range(data[0].shape[1] + 20):
            index_map[col_index] = [current_index]
            current_index += 1


    else:
        data = np.array(data)  # Ensure input is a NumPy array for easier handling

        # Check each column to determine type (string/text or numeric)
        for col_index in range(data.shape[1]):  # Iterate over columns
            column_data = data[:, col_index]  # Extract the entire column

            if any(isinstance(value, str) for value in column_data):  # Check for any string in the column
                index_map[col_index] = list(range(current_index, current_index + 20))  # Text feature
                current_index += 20
            else:
                index_map[col_index] = [current_index]  # Numeric feature
                current_index += 1

    return index_map, current_index


def load_data_function(data_loader_name, input_rel_path="input\\"):
    """
    Dynamically loads a data loading function by name and calls it with input_rel_path.

    :param data_loader_name: String key for the data loader
    :param input_rel_path: Relative path to input data directory
    :return: Tuple of (X, y, tests_number, map_test) from the data loader
    """
    data_loaders = {
        "load_time_Series": utils.load_time_Series,
        "load_mimic_text": utils.load_mimic_text,
        "load_mimic_time_series": utils.load_mimic_time_series,
    }

    # Get the appropriate function based on the provided name
    loader_func = data_loaders.get(data_loader_name, utils.load_time_Series)  # Default to load_time_Series
    
    # Call the loader function with input_rel_path
    return loader_func(input_rel_path=input_rel_path)


class MultimodalGuesser(nn.Module):
    """
    A multimodal model designed to process and classify heterogeneous medical inputs,
    including structured tabular features, images, text (e.g., clinical notes), and time series.

    Components:
    - Uses BART for summarization.
    - Uses ClinicalBERT for embedding textual features.
    - CNN for image embedding.
    - LSTM for time series embedding.
    - Fully connected layers for classification.

    Attributes:
        device (torch.device): The device (CPU/GPU) to run the model on.
        X (list/array): Input data.
        y (array): Corresponding labels.
        tests_number (int): Number of tests (features).
        map_test (dict): Mapping of test indices.
        summarize_text_model (nn.Module): BART model for text summarization.
        tokenizer_summarize_text_model (Tokenizer): Tokenizer for BART.
        text_model (nn.Module): ClinicalBERT model for text embedding.
        tokenizer (Tokenizer): Tokenizer for ClinicalBERT.
        cost_list (list): Cost for each feature.
        cost_budget (int): Total cost allowed.
        img_embedder (nn.Module): CNN-based image embedder.
        time_series_embedder (nn.Module): LSTM encoder for time series data.
        text_reducer (nn.Linear): Reduces text embedding dimensions.
        num_classes (int): Number of prediction classes.
        map_feature (dict): Maps feature indices to embedded feature vectors.
        features_total (int): Total length of the embedded feature vector.
        layer1/2/3 (nn.Sequential): Fully connected hidden layers.
        logits (nn.Linear): Final classification layer.
        criterion (nn.Module): CrossEntropy loss.
        optimizer (torch.optim): Adam optimizer.
        path_to_save (str): Path to save model checkpoints.
        guesser_model_file_name (str): Filename for saved guesser model.
    """

    def __init__(self, FLAGS):
        super(MultimodalGuesser, self).__init__()
        self.device = DEVICE
        # Load the function based on the argument
        self.X, self.y, self.tests_number, self.map_test = load_data_function(FLAGS.data, FLAGS.input_rel_path)
        # load summarization model
        self.summarize_text_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(
            self.device)
        self.tokenizer_summarize_text_model = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        # load clinicalBERT model

        self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # load cost list
        self.cost_list = [1] * (self.tests_number + 1)
        if isinstance(self.X, list):
            self.cost_list = [0] + [1] * (self.tests_number)
        self.cost_budget = sum(self.cost_list)
        # define image embedder
        self.img_embedder = ImageEmbedder()
        # load LSTM model for time series data
        if isinstance(self.X, list):
            self.time_series_embedder = LSTMEncoder(self.X[0].shape[1], 20).to(self.device)
        # define dimension of embedding
        self.text_reducer = nn.Linear(FLAGS.text_embed_dim, FLAGS.reduced_dim).to(self.device)
        self.text_reduced_dim = FLAGS.reduced_dim
        self.num_classes = len(np.unique(self.y))
        self.map_feature, self.features_total = map_features_to_indices(self.X)
        # defin the NN
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.features_total, FLAGS.hidden_dim1),
            torch.nn.PReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.hidden_dim1, FLAGS.hidden_dim2),
            torch.nn.PReLU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.hidden_dim2, FLAGS.hidden_dim2),
            torch.nn.PReLU(),
        )

        self.logits = nn.Linear(FLAGS.hidden_dim2, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr)
        self.path_to_save = os.path.join(os.getcwd(), FLAGS.save_dir)
        self.guesser_model_file_name = FLAGS.guesser_model_file_name
        self.layer1 = self.layer1.to(self.device)
        self.layer2 = self.layer2.to(self.device)
        self.layer3 = self.layer3.to(self.device)
        self.logits = self.logits.to(self.device)

    def summarize_text(self, text, max_length=300, min_length=100, length_penalty=2.0, num_beams=4):
        """
        Summarizes a long text using BART.

        Args:
            text (str): The input clinical note.
            max_length (int): The maximum length of the summary.
            min_length (int): The minimum length of the summary.
            length_penalty (float): Length penalty for beam search.
            num_beams (int): Number of beams for beam search.

        Returns:
            str: The summarized text.
        """
        inputs = self.tokenizer_summarize_text_model.encode("summarize: " + text, return_tensors="pt", max_length=1024,
                                                            truncation=True).to(self.device)
        summary_ids = self.summarize_text_model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=True
        )
        summary = self.tokenizer_summarize_text_model.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def embed_text(self, text):
        """
        Embeds text using ClinicalBERT.

        Args:
            text (str): The input text (e.g., summarized clinical note).

        Returns:
            torch.Tensor: The ClinicalBERT embeddings.
        """
        text = self.summarize_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True,
                                padding="max_length").to(self.device)
        outputs = self.text_model(**inputs)
        # Use the CLS token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].to(self.device)
        # check this line
        return F.relu(self.text_reducer(cls_embedding))

    def embed_image(self, image_path):
        """
        Converts an image file into a 20-dimensional embedding using the internal ImageEmbedder.

        Args:
            image_path (str): Path to the grayscale image file (.png or .jpg).

        Returns:
            torch.Tensor: Embedded image tensor of shape [1, 20].
        """
        img = Image.open(image_path).convert('L')  # Open the image
        img = self.img_embedder.transform(img).unsqueeze(0).to(self.device)  # Apply transforms and add batch dimension
        embedding = self.img_embedder(img).to(self.device)
        return embedding

    def is_numeric_value(self, value):
        """
        Determines whether a given value is numeric (int, float, or float tensor).

        Args:
            value: A single feature value.

        Returns:
            bool: True if value is numeric, False otherwise.
        """
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, torch.Tensor):
            if value.dtype in [torch.float, torch.float64]:
                return True
        return False

    def is_text_value(self, value):
        """
        Determines if a feature is a textual value (str).

        Args:
            value: A single feature value.

        Returns:
            bool: True if the value is a string.
        """
        if isinstance(value, str):
            return True
        else:
            return False

    def is_image_value(self, value):
        """
        Checks if a feature is an image path based on its file extension.

        Args:
            value: A single feature value.

        Returns:
            bool: True if value is a path ending in 'png' or 'jpg'.
        """
        if isinstance(value, str):
            if value.endswith('png') or value.endswith('jpg'):
                return True
            else:
                return False

    def is_time_series_value(self, value):
        """
        Determines whether the input value is a time-series represented as a pandas DataFrame.

        Args:
            value: Input to be checked.

        Returns:
            bool: True if the input is a DataFrame.
        """
        if isinstance(value, pd.DataFrame):
            return True
        else:
            return False

    def forward(self, input, mask=None):
        """
         Forward pass of the model. Handles both time-series and flat feature inputs.

         Parameters
         ----------
         input : pd.DataFrame or list
             The input sample, can be a time-series DataFrame or a list of features.
         mask : np.array or torch.Tensor, optional
             A binary mask to apply to the input features.

         Returns
         -------
         torch.Tensor
             Probability distribution over classes after applying softmax.
         """
        sample_embeddings = []

        # Check if input is a time series (e.g., DataFrame)
        if self.is_time_series_value(input):
            df_history = input.iloc[:-1]  # All rows except the last one
            if df_history.shape[0] > 0:
                x = torch.tensor(df_history.values, dtype=torch.float32, device=self.device).unsqueeze(0)
                x = self.time_series_embedder(x)
                sample_embeddings.append(x)
            else:
                # If there's no history, append a zero vector instead
                embed_dim = self.text_reduced_dim
                sample_embeddings.append(torch.zeros((1, embed_dim), device=self.device))

            recent_values = torch.tensor(input.iloc[-1].values, dtype=torch.float32, device=self.device).unsqueeze(0)
            sample_embeddings.append(recent_values)

        else:
            # Handle non-time-series input (e.g., one row of a DataFrame or list of features)
            for col_index, feature in enumerate(input):
                if self.is_image_value(feature):
                    feature_embed = self.embed_image(feature)
                elif self.is_text_value(feature):
                    feature_embed = self.embed_text(feature)
                elif pd.isna(feature):
                    size = len(self.map_feature.get(col_index, []))
                    feature_embed = torch.zeros((1, size), dtype=torch.float32, device=self.device)
                elif self.is_numeric_value(feature):
                    feature_embed = torch.tensor([[feature]], dtype=torch.float32, device=self.device)
                else:
                    raise ValueError(f"Unsupported feature type: {feature}")

                sample_embeddings.append(feature_embed)

        x = torch.cat(sample_embeddings, dim=1).to(DEVICE)
        if mask is not None:
            # Convert binary_mask (NumPy array) to a PyTorch tensor
            mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
            x = x * mask
        x = x.squeeze(dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Final prediction head
        logits = self.logits(x).to(self.device)

        # Compute softmax probabilities
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs

    def save_model(self):
        """
        Saves the model's state dictionary to disk.

        Returns
        -------
        None
        """
        path = self.path_to_save
        if not os.path.exists(path):
            os.makedirs(path)
        guesser_save_path = os.path.join(path, self.guesser_model_file_name)
        # save guesser
        if os.path.exists(guesser_save_path):
            os.remove(guesser_save_path)
        torch.save(self.cpu().state_dict(), guesser_save_path + '~')
        os.rename(guesser_save_path + '~', guesser_save_path)
        self.to(self.device)

    def load_model(self):
        """
        Loads the model's state dictionary from disk.

        Returns
        -------
        None
        """
        guesser_load_path = os.path.join(self.path_to_save, self.guesser_model_file_name)
        if os.path.exists(guesser_load_path):
            guesser_state_dict = torch.load(guesser_load_path, map_location=self.device, weights_only=True)
            self.load_state_dict(guesser_state_dict)
            self.to(self.device)

    def save_temp_guesser(self, episode, accuracy):
        """
        Saves the model's state dictionary to disk with a temporary filename
        that includes episode number and accuracy.

        Parameters
        ----------
        episode : int
            Episode number
        accuracy : float
            Validation accuracy

        Returns
        -------
        None
        """
        path = self.path_to_save
        if not os.path.exists(path):
            os.makedirs(path)
        temp_filename = '{}_{}_{:1.3f}.pth'.format(episode, 'guesser', accuracy)
        guesser_save_path = os.path.join(path, temp_filename)
        # save guesser
        if os.path.exists(guesser_save_path):
            os.remove(guesser_save_path)
        torch.save(self.cpu().state_dict(), guesser_save_path + '~')
        os.rename(guesser_save_path + '~', guesser_save_path)
        self.to(self.device)

    def load_temp_model(self, episode, accuracy):
        """
        Loads the model's state dictionary from disk using a temporary filename
        that includes episode number and accuracy.

        Parameters
        ----------
        episode : int
            Episode number
        accuracy : float
            Validation accuracy

        Returns
        -------
        None
        """
        temp_filename = '{}_{}_{:1.3f}.pth'.format(episode, 'guesser', accuracy)
        guesser_load_path = os.path.join(self.path_to_save, temp_filename)
        if os.path.exists(guesser_load_path):
            guesser_state_dict = torch.load(guesser_load_path, map_location=self.device, weights_only=True)
            self.load_state_dict(guesser_state_dict)
            self.to(self.device)
