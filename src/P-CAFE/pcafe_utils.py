import os
from sklearn.datasets import fetch_openml
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def add_noise(X, noise_std=0.01):
    """
    Add Gaussian noise to the input features.

    Parameters:
    - X: Input features (numpy array).
    - noise_std: Standard deviation of the Gaussian noise.

    Returns:
    - X_noisy: Input features with added noise.
    """
    noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
    X_noisy = X + noise
    return X_noisy


def balance_class(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, add_noise(X[noisy_indices], noise_std)], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()
    return X_balanced, y_balanced




def map_multiple_features(sample):
    index_map = {}
    for i in range(0, sample.shape[0]):
        index_map[i] = [i]
    return index_map


def map_multiple_features_for_logistic_mimic(sample):
    # map the index features that each test reveals
    index_map = {}
    for i in range(0, 17):
        index_map[i] = list(range(i * 42, i * 42 + 42))
    return index_map

def map_time_series(sample):
    """
    Map features to indices based on their type.
    If any value in a column is a string, treat the column as text.
    :param sample: 2D array-like dataset (list of lists, numpy array, or pandas DataFrame)
    :return: A dictionary mapping feature indices to lists of indices and the total number of features
    """
    index_map = {}
    current_index = 0
    for col_index in range(sample.shape[1]+1):
            if col_index == 0:
                index_map[col_index] = list(range(current_index, current_index + 20))
                current_index += 20
            else:
                index_map[col_index] = [current_index]  # Numeric feature
                current_index += 1
    return index_map


def load_mimic_time_series():
    # Get the current working directory
    base_dir = os.getcwd()
    # Construct file paths dynamically
    #train_X_path = os.path.join(base_dir, 'input\\data_time_series\\train_X.csv')
    #train_Y_path = os.path.join(base_dir, 'input\\data_time_series\\train_Y.csv')
    #val_X_path = os.path.join(base_dir, 'input\\data_time_series\\val_X.csv')
    #val_Y_path = os.path.join(base_dir, 'input\\data_time_series\\val_Y.csv')
    #test_X_path = os.path.join(base_dir, 'input\\data_time_series\\test_X.csv')
    #test_Y_path = os.path.join(base_dir, 'input\\data_time_series\\test_Y.csv')
    train_X_path = os.path.join(base_dir, 'data\\input\\data_time_series\\train_X.csv')
    train_Y_path = os.path.join(base_dir, 'data\\input\\data_time_series\\train_Y.csv')
    val_X_path = os.path.join(base_dir, 'data\\input\\data_time_series\\val_X.csv')
    val_Y_path = os.path.join(base_dir, 'data\\input\\data_time_series\\val_Y.csv')
    test_X_path = os.path.join(base_dir, 'data\\input\\data_time_series\\test_X.csv')
    test_Y_path = os.path.join(base_dir, 'data\\input\\data_time_series\\test_Y.csv')

    # Read the files
    X_train = pd.read_csv(train_X_path)
    Y_train = pd.read_csv(train_Y_path)
    X_val = pd.read_csv(val_X_path)
    Y_val = pd.read_csv(val_Y_path)
    X_test = pd.read_csv(test_X_path)
    Y_test = pd.read_csv(test_Y_path)

    X = pd.concat([X_train, X_val, X_test])
    Y = pd.concat([Y_train, Y_val, Y_test])
    Y = Y.to_numpy().reshape(-1)
    # balance classes no noise
    X, Y = balance_class_no_noise(X.to_numpy(), Y)
    X = pd.DataFrame(X)
    map_test = map_multiple_features_for_logistic_mimic(X.iloc[0])
    return X, Y, 17, map_test


def clean_mimic_data_nan(X):
    # Keeps columns with at least 20% non-missing values.
    X = X.dropna(axis=1, thresh=int(0.2 * X.shape[0]))
    return X


def df_to_list(df):
    grp_df = df.groupby('patientunitstayid')
    df_arr = []

    for idx, frame in grp_df:
        # Sort the dataframe for each patient based on 'itemoffset'
        sorted_frame = frame.sort_values(by='itemoffset', ascending=True)  # or True for ascending
        df_arr.append(sorted_frame)

    return df_arr




def load_time_Series():
    base_dir = os.getcwd()
    path = os.path.join(base_dir, 'input\\df_data.csv')
    # read csv file from path
    df_time_series = pd.read_csv(path)
    df_time_series = df_to_list(df_time_series)
    labels = []
    patients = []
    for df in df_time_series:
        df = df.drop(df.columns[0:3], axis=1)
        # Drop the last column (assuming it's the label)
        labels.append(df.iloc[0, -1])
        df = df.iloc[:, :-1]
        patients.append(df)
    patients, labels = balance_class_no_noise_dfs(patients, labels)

    map_test = map_time_series(patients[0])
    return patients, labels, len(patients[0].columns) + 1, map_test



def load_mimic_text():
    base_dir = os.getcwd()
    # Construct file paths dynamically
    path = os.path.join(base_dir, 'input\\data_with_text.json')
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id', 'los'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    df = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X, Y = balance_class_no_noise(df.to_numpy(), Y)
    X = pd.DataFrame(X)
    X = clean_mimic_data_nan(X)
    map_test = map_multiple_features(X.iloc[0])
    # X,Y = reduce_number_of_samples(X,Y)
    return X, Y, len(X.columns), map_test


def balance_class_multi(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    # Calculate the difference in sample counts for each class
    count_diff = max_class_count - class_counts

    # Initialize arrays to store balanced data
    X_balanced = X.copy()
    y_balanced = y.copy()

    # Add noise to the features of the minority classes to balance the dataset
    for minority_class, diff in zip(unique_classes, count_diff):
        if diff > 0:
            # Get indices of samples belonging to the current minority class
            minority_indices = np.where(y == minority_class)[0]

            # Randomly sample indices from the minority class to add noise
            noisy_indices = np.random.choice(minority_indices, diff, replace=True)

            # Add noise to the features of the selected samples
            X_balanced = np.concatenate([X_balanced, add_noise(X[noisy_indices], noise_std)], axis=0)
            y_balanced = np.concatenate([y_balanced, y[noisy_indices]], axis=0)

    return X_balanced, y_balanced


def balance_class_no_noise_dfs(patients, labels):
    """
    Balance classes by duplicating minority class patient time series (no noise added).

    Parameters:
    - patients: List of pandas DataFrames (one per patient).
    - labels: List or array of labels.

    Returns:
    - patients_balanced: List of balanced DataFrames.
    - labels_balanced: Balanced label list.
    """
    from collections import Counter
    import numpy as np

    # Count class instances
    class_counts = Counter(labels)
    unique_classes = list(class_counts.keys())
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # Indices of each class
    majority_indices = [i for i, y in enumerate(labels) if y == majority_class]
    minority_indices = [i for i, y in enumerate(labels) if y == minority_class]

    # Determine how many more samples are needed
    diff = len(majority_indices) - len(minority_indices)

    if diff > 0:
        # Randomly sample with replacement from the minority class
        sampled_indices = np.random.choice(minority_indices, diff, replace=True)
        patients_balanced = patients + [patients[i].copy() for i in sampled_indices]
        labels_balanced = labels + [labels[i] for i in sampled_indices]
    else:
        patients_balanced = patients.copy()
        labels_balanced = labels.copy()

    return patients_balanced, labels_balanced

def balance_class_no_noise(X, y):
    """
    Balance classes by adding noise to the minority class's numeric features.

    Parameters:
    - X: Input features (numpy array with mixed data types).
    - y: Labels.
    - noise_std: Standard deviation of Gaussian noise.
    - numeric_columns: List or array of boolean values indicating numeric columns.

    Returns:
    - X_balanced: Balanced feature set.
    - y_balanced: Balanced labels.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, X[noisy_indices]], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()

    return X_balanced, y_balanced




