import os
import random
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from pathlib import Path
from ..load_config import load_hierarchical_config
from .multimodal_guesser import MultimodalGuesser

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#System defaults
NUM_EPOCHS = 100
VAL_TRIALS_WO_IM = 500


def parse_arguments():
    """
    Parse command line arguments with defaults from configuration files.
    
    :return: Parsed arguments namespace
    """
    # Load hierarchical configuration: base_config.json -> user_config.json -> CLI args
    # First, load base and user configs
    config = load_hierarchical_config(
        base_config_path="config/base_config.json",
        user_config_path="config/user_config.json"
    )
    
    # Extract embedder_guesser configuration with fallback to root-level config
    embedder_config = config.get("embedder_guesser", {})
    
    # Get the project path from the JSON configuration
    project_path = Path(config.get("user_specific_project_path", os.getcwd()))
    
    # Define argument parser with defaults from configuration
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--directory",
                        type=str,
                        default=str(project_path),
                        help="Directory for saved models")
    parser.add_argument("--hidden-dim1",
                        type=int,
                        default=embedder_config.get("hidden_dim1", 64),
                        help="Hidden dimension")
    parser.add_argument("--hidden-dim2",
                        type=int,
                        default=embedder_config.get("hidden-dim2", 32),
                        help="Hidden dimension")
    parser.add_argument("--lr",
                        type=float,
                        default=embedder_config.get("lr", 1e-4),
                        help="Learning rate")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=embedder_config.get("weight_decay", 0.001),
                        help="l_2 weight penalty")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=embedder_config.get("num_epochs", NUM_EPOCHS),
                        help="number of epochs (can be set in `config\\user_config.json`)")
    parser.add_argument("--val_trials_wo_im",
                        type=int,
                        default=embedder_config.get("val_trials_wo_im", VAL_TRIALS_WO_IM),
                        help="Number of validation trials without improvement")
    parser.add_argument("--fraction_mask",
                        type=float,
                        default=embedder_config.get("fraction_mask", 0),
                        help="fraction mask")
    parser.add_argument("--run_validation",
                        type=int,
                        default=embedder_config.get("run_validation", 100),
                        help="after how many epochs to run validation")
    parser.add_argument("--batch_size",
                        type=int,
                        default=embedder_config.get("batch_size", 128),
                        help="batch size")
    parser.add_argument("--text_embed_dim",
                        type=int,
                        default=embedder_config.get("text_embed_dim", 768),
                        help="Text embedding dimension")
    parser.add_argument("--reduced_dim",
                        type=int,
                        default=embedder_config.get("reduced_dim", 20),
                        help="Reduced dimension for text embedding")
    parser.add_argument("--save_dir",
                        type=str,
                        default='guesser_eICU',
                        help="save path")
    parser.add_argument(
        "--data",
        type=str,
        default=embedder_config.get("data","load_time_Series"),
        help=(
            "Dataset loader function to use. Options:\n"
            "  load_time_Series        - eICU time series data\n"
            "  load_mimic_text         - MIMIC-III multi-modal (includes text)\n"
            "  load_mimic_time_series  - MIMIC-III numeric time series data"
        )
    )
    
    return parser.parse_args()


def create_mask(model, FLAGS) -> np.array:
    """
    Creates a random binary mask over features for input masking during training.

    Parameters
    ----------
    model : object
        The model instance containing feature mapping.
    FLAGS : object
        Parsed command line arguments.

    Returns
    -------
    np.array
        A binary mask indicating which features are kept (1) or masked (0).
    """
    mapping = model.map_feature
    binary_mask = np.zeros(model.features_total)
    for key, value in mapping.items():
        if np.random.rand() < FLAGS.fraction_mask:
            # mask all the keys entries in binary mask
            for i in value:
                binary_mask[i] = 0
        else:
            for i in value:
                binary_mask[i] = 1
    return binary_mask


def create_adversarial_input(sample, label, pretrained_model):
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


def train_model(model, FLAGS,
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
                mask = create_mask(model, FLAGS)
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


def save_model(model):
    """
    Saves the model's state dictionary to disk.

    Parameters
    ----------
    model : object
        The model to save.

    Returns
    -------
    None
    """
    path = model.path_to_save
    if not os.path.exists(path):
        os.makedirs(path)
    guesser_filename = 'best_guesser.pth'
    guesser_save_path = os.path.join(path, guesser_filename)
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(model.cpu().state_dict(), guesser_save_path + '~')
    os.rename(guesser_save_path + '~', guesser_save_path)
    model.to(DEVICE)


def val(model, X_val, y_val, best_val_auc=0):
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
            input = X_val[i]
            label = torch.tensor(y_val[i], dtype=torch.long).to(model.device)
            output = model(input)

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
    print(f'Validation AUC-PC: {auc_pc:.2f}')

    if auc_roc >= best_val_auc:
        save_model(model)
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
    guesser_filename = 'best_guesser.pth'
    guesser_load_path = os.path.join(model.path_to_save, guesser_filename)
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()

    correct = 0
    y_true = []
    y_pred = []
    y_scores = []  # For probability scores for AUC

    with torch.no_grad():
        for i in range(len(X_test)):
            input = X_test[i]
            label = torch.tensor(y_test[i], dtype=torch.long).to(model.device)
            output = model(input)

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
    print(f'AUC-PC: {auc_pc:.2f}')


def main():
    '''
    Train a neural network to guess the correct answer
    :return:
    '''
    FLAGS = parse_arguments()
    os.chdir(FLAGS.directory)
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

    train_model(model, FLAGS, FLAGS.num_epochs,
                X_train, y_train, X_val, y_val)

    test(model, X_test, y_test)


if __name__ == "__main__":
    main()
