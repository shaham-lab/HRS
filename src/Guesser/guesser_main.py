import os
import random
from argparse import Namespace

import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd

from ..common.parse_args import parse_arguments
from .multimodal_guesser import MultimodalGuesser


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_mask(model: MultimodalGuesser, fraction_mask) -> np.ndarray:
    """
    Creates a random binary mask over features for input masking during training.

    Parameters
    ----------
    model : object
        The model instance containing feature mapping.
    fraction_mask :

    Returns
    -------
    np.array
        A binary mask indicating which features are kept (1) or masked (0).
    """
    mapping = model.map_feature
    binary_mask = np.zeros(model.features_total)
    for key, value in mapping.items():
        if np.random.rand() < fraction_mask:
            # mask all the keys entries in binary mask
            for i in value:
                binary_mask[i] = 0
        else:
            for i in value:
                binary_mask[i] = 1
    return binary_mask


def create_adversarial_input(sample, label, pretrained_model: MultimodalGuesser):
    """
    Generates an adversarial mask by identifying the most influential feature
    for a given sample and zeroing it out.

    Parameters
    ----------
    sample : pd.DataFrame or list
        The input data sample.
    label : torch.Tensor
        Ground-truth label for the sample.
    pretrained_model : object
        The trained model used to compute gradients.

    Returns
    -------
    np.array
        A binary mask with the most influential feature zeroed out.
    """
    pretrained_model.eval()

    # Convert the sample into a proper forward-pass input
    if pretrained_model.is_time_series_value(sample):
        # Handle time-series input (e.g., a DataFrame)
        df_history = sample.iloc[:-1]
        embeddings = []

        if df_history.shape[0] > 0:
            x = torch.tensor(df_history.values, dtype=torch.float32, device=pretrained_model.device).unsqueeze(0)
            x = pretrained_model.time_series_embedder(x)
            embeddings.append(x)
        else:
            embed_dim = pretrained_model.text_reduced_dim
            embeddings.append(torch.zeros((1, embed_dim), device=pretrained_model.device))

        recent_values = torch.tensor(sample.iloc[-1].values, dtype=torch.float32,
                                     device=pretrained_model.device).unsqueeze(0)
        embeddings.append(recent_values)

        input_tensor = torch.cat(embeddings, dim=1).squeeze(0)  # shape: [dim]
    else:
        # Flat feature input (e.g., list or Series)
        sample_embeddings = []
        for col_index, feature in enumerate(sample):
            if pretrained_model.is_image_value(feature):
                feature_embed = pretrained_model.embed_image(feature)
            elif pretrained_model.is_text_value(feature):
                feature_embed = pretrained_model.embed_text(feature)
            elif pd.isna(feature):
                size = len(pretrained_model.map_feature.get(col_index, []))
                feature_embed = torch.zeros((1, size), dtype=torch.float32, device=pretrained_model.device)
            elif pretrained_model.is_numeric_value(feature):
                feature_embed = torch.tensor([[feature]], dtype=torch.float32, device=pretrained_model.device)
            else:
                raise ValueError(f"Unsupported feature type: {feature}")
            sample_embeddings.append(feature_embed)

        input_tensor = torch.cat(sample_embeddings, dim=1).squeeze(0)

    # Enable gradient tracking for adversarial attack
    input_tensor = input_tensor.detach().clone().requires_grad_(True)

    # Forward pass through model (excluding the final layer)
    x = pretrained_model.layer1(input_tensor)
    x = pretrained_model.layer2(x)
    x = pretrained_model.layer3(x)
    logits = pretrained_model.logits(x).squeeze(0)
    # Compute softmax probabilities
    if logits.dim() == 2:
        probs = F.softmax(logits, dim=1)
    else:
        probs = F.softmax(logits, dim=-1)

    loss = pretrained_model.criterion(probs.unsqueeze(0), label)

    # Backward pass to compute gradients w.r.t. input
    loss.backward()
    gradient = input_tensor.grad

    # Get the most influential feature index
    absolute_gradients = torch.abs(gradient)
    max_gradients_index = torch.argmax(absolute_gradients, dim=-1).item()

    # Map to original feature index
    for key, value in pretrained_model.map_feature.items():
        if max_gradients_index in value:
            max_gradients_index = key
            break

    # Create binary mask that zeroes out most important feature
    binary_mask = np.ones(pretrained_model.features_total)
    for key, value in pretrained_model.map_feature.items():
        if key == max_gradients_index:
            for i in value:
                binary_mask[i] = 0
    pretrained_model.train()  # Set the model to training mode

    return binary_mask


def plot_running_loss(loss_list):
    """
    Plots the training loss over time.

    Parameters
    ----------
    loss_list : list of float
        A list containing the running average loss per epoch.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Running Loss')
    plt.show()


def compute_probabilities(j, total_episodes):
    """
    Computes the probability for applying a random mask vs. adversarial mask.

    Parameters
    ----------
    j : int
        Current training epoch.
    total_episodes : int
        Total number of training epochs.

    Returns
    -------
    float
        Probability of applying a random mask.
    """
    prob_mask = 0.8 + 0.2 * (1 - j / total_episodes)  # Starts at 1, decreases to 0.8
    return prob_mask


def train_model(model: MultimodalGuesser, FLAGS: Namespace,
                nepochs, X_train, y_train, X_val, y_val):
    """
    Trains the model using random or adversarial feature masking.

    Parameters
    ----------
    model : object
        The model to be trained.
    FLAGS : object
        Parsed command line arguments.
    nepochs : int
        Number of training epochs.
    X_train : array-like
        Training inputs.
    y_train : array-like
        Training labels.
    X_val : array-like
        Validation inputs.
    y_val : array-like
        Validation labels.

    Returns
    -------
    None
    """
    if isinstance(X_train, list):
        pass
    else:
        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()
    val_trials_without_improvement = 0
    best_val_auc = 0
    accuracy_list = []
    loss_list = []
    num_samples = len(X_train)
    for j in range(1, nepochs):
        running_loss = 0
        random_indices = np.random.choice(num_samples, size=FLAGS.batch_size, replace=False)
        model.train()  # Set the model to training mode
        model.optimizer.zero_grad()  # Reset gradients before starting the batch
        # Process each sample in the batch
        for i in random_indices:
            input = X_train[i]
            label = torch.tensor([y_train[i]], dtype=torch.long).to(model.device)  # Convert label to tensor
            prob_mask = compute_probabilities(j, nepochs)
            # Decide the action based on the computed probabilities
            if random.random() < prob_mask:
                mask = create_mask(model, FLAGS.fraction_mask)
            else:
                mask = create_adversarial_input(input, label, model)

            # Forward pass
            output = model(input, mask)
            loss = model.criterion(output, label)
            running_loss += loss.item()  # Accumulate loss for the batch

            # Backpropagate gradients
            loss.backward()

        # Update model parameters after the entire batch
        model.optimizer.step()

        average_loss = running_loss / len(random_indices)
        loss_list.append(average_loss)

        if j % FLAGS.run_validation == 0:
            new_best_val_auc = val(model, X_val, y_val, best_val_auc)
            accuracy_list.append(new_best_val_auc)
            if new_best_val_auc > best_val_auc:
                best_val_auc = new_best_val_auc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1
            if val_trials_without_improvement == FLAGS.val_trials_wo_im:
                print('Did not achieve val AUC improvement for {} trials, training is done.'.format(
                    FLAGS.val_trials_wo_im))
                break
        print("finished " + str(j) + " out of " + str(nepochs) + " epochs")

    plot_running_loss(loss_list)


def val(model: MultimodalGuesser, X_val, y_val, best_val_auc=0):
    """
    Evaluates the model on validation data and returns updated AUC.

    Parameters
    ----------
    model : object
        The model to evaluate.
    X_val : array-like
        Validation inputs.
    y_val : array-like
        Validation labels.
    best_val_auc : float, optional
        Best validation AUC so far.

    Returns
    -------
    float
        Updated best validation AUC.
    """
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    y_scores = []  # For probability scores for AUC

    with torch.no_grad():
        for i in range(len(X_val)):
            sample = X_val[i]
            label = torch.tensor(y_val[i], dtype=torch.long).to(model.device)
            output = model(sample)

            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                correct += 1

            y_true.append(label.item())
            y_pred.append(predicted.item())
            y_scores.append(torch.softmax(output, dim=1)[:, 1].item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Validation Confusion Matrix:")
    print(conf_matrix)

    # Accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f'Validation Accuracy: {accuracy:.2f}')

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_scores)
    print(f'Validation AUC-ROC: {auc_roc:.2f}')

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pc = auc(recall, precision)
    print(f'Validation AUC-PRC: {auc_pc:.2f}')

    if auc_roc >= best_val_auc:
        model.save_model()
        best_val_auc = auc_roc

    return accuracy


def test(model, X_test, y_test):
    """
    Tests the model on the test dataset and prints evaluation metrics.
    :param model:
    :param X_test:
    :param y_test:
    :return:
    """
    guesser_load_path = os.path.join(model.path_to_save, model.guesser_model_file_name)
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()

    correct = 0
    y_true = []
    y_pred = []
    y_scores = []  # For probability scores for AUC
    X_test = X_test.to_numpy() #convert from pandas dataframe to numpy row vector

    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i]
            label = torch.tensor(y_test[i], dtype=torch.long).to(model.device)
            output = model(sample)

            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                correct += 1

            y_true.append(label.item())  # Assuming labels is a numpy array
            y_pred.append(predicted.item())
            y_scores.append(torch.softmax(output, dim=1)[:, 1].item())  # Get the probability for class 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f'Test Accuracy: {accuracy:.2f}')

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_scores)
    print(f'AUC-ROC: {auc_roc:.2f}')

    # AUC-PC (Precision-Recall AUC)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pc = auc(recall, precision)
    print(f'AUC-PRC: {auc_pc:.2f}')


def main():
    '''
    Train a neural network to guess the correct answer
    :return:
    '''
    FLAGS = parse_arguments()
    #os.chdir(FLAGS.directory)
    model = MultimodalGuesser(FLAGS)
    model.to(model.device)
    X_train, X_test, y_train, y_test = train_test_split(model.X,
                                                        model.y,
                                                        test_size=0.1,
                                                        random_state=24)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=24)

    # Print dimensions / lengths for train, val, test splits
    #for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        #print(f"{name} shape: {arr.shape}")

    train_model(model, FLAGS, FLAGS.num_epochs,
                X_train, y_train, X_val, y_val)

    test(model, X_test, y_test)


if __name__ == "__main__":
    main()
