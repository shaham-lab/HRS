import csv
import os
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
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


def load_ucimlrepo():
    # fetch dataset
    diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
    X = diabetic_retinopathy_debrecen.data.features.to_numpy()
    y = diabetic_retinopathy_debrecen.data.targets.to_numpy()
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)
    return X, y, diabetic_retinopathy_debrecen.metadata.num_features, diabetic_retinopathy_debrecen.metadata.num_features


def load_fetal():
    file_path = r'/data/fetal_health_None.csv'
    df = pd.read_csv(file_path)
    Y = df['fetal_health'].to_numpy().reshape(-1)
    X = df.drop(df.columns[-1], axis=1).to_numpy()
    return X, Y, Y, len(X[0])


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


def load_mimiciii():
    file_path = r'C:\Users\kashann\PycharmProjects\mimic-code-extract\mimic-iii\notebooks\ipynb_example\filtered_data.csv'
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[df.columns[-1]])
    Y = df[df.columns[-1]].to_numpy().reshape(-1)
    # Drop the last two columns by column names
    X = df.drop(df.columns[-3:], axis=1).to_numpy()
    # Drop the first column
    X = X[:, 1:]
    # numeric_columns = [True] * (X.shape[1] - 1) + [False]
    X, Y = balance_class_no_noise(X, Y)
    X = pd.DataFrame(X)
    # CONVERT ALL COLUMN EXEPT LAST 2 and
    # for i in range(0, len(X.columns) - 1):
    #     X[i] = X[i].astype(float)
    return X, Y, len(X.columns), map_multiple_features(X.iloc[0])


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


def reduce_number_of_samples(X, Y):
    # reduce the number of samples to 1000
    X = X[:1000]
    Y = Y[:1000]
    return X, Y


def load_all_data():
    path = r'C:\Users\kashann\PycharmProjects\P-CAFE\eICU\DATA_FULL.csv'
    # Read the files
    DATA = pd.read_csv(path)
    # remove the first 2 columns
    DATA = DATA.iloc[:, 2:]

    # Y is the last column of DATA
    Y = DATA.iloc[:, -1].to_numpy().reshape(-1)
    # X is all columns except the last column
    X = DATA.iloc[:, :-1]
    # balance classes no noise
    X, Y = balance_class_no_noise(X.to_numpy(), Y)
    X = pd.DataFrame(X)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, 21, map_test


def df_to_list(df):
    grp_df = df.groupby('patientunitstayid')
    df_arr = []

    for idx, frame in grp_df:
        # Sort the dataframe for each patient based on 'itemoffset'
        sorted_frame = frame.sort_values(by='itemoffset', ascending=True)  # or True for ascending
        df_arr.append(sorted_frame)

    return df_arr


def load_time_series():
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
        df_history = df.iloc[:-1]  # all except last row

        history_stats = df_history.agg(['mean', 'std', 'min', 'max']).values.flatten()
        recent_values = df.iloc[[-1]].values.flatten()
        embedding = np.concatenate([recent_values, history_stats])
        patients.append(embedding)

    X, Y = balance_class_no_noise(np.array(patients), np.array(labels).reshape(-1))
    X = pd.DataFrame(X)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, len(X.columns), map_test



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


def load_eICU():
    # Construct file paths dynamically
    path = r'C:\Users\kashann\PycharmProjects\P-CAFE\eICU\DATA.csv'
    # Read the files
    DATA = pd.read_csv(path)
    Y = DATA['Labels']
    X = DATA.drop(columns=['Labels'])
    Y = Y.to_numpy().reshape(-1)
    # balance classes no noise
    X, Y = balance_class_no_noise(X.to_numpy(), Y)
    X = pd.DataFrame(X)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, 60, map_test


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


def load_mimic_no_text():
    base_dir = os.getcwd()
    # Construct file paths dynamically
    path = os.path.join(base_dir, 'input\\data_numeric.json')
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    df = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X, Y = balance_class_no_noise(df.to_numpy(), Y)
    X = pd.DataFrame(X)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, len(X.columns), map_test


def load_mimic_only_text():
    base_dir = os.getcwd()
    # Construct file paths dynamically
    path = os.path.join(base_dir, 'input\\data_with_text.json')
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    df = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X, Y = balance_class_no_noise(df.to_numpy(), Y)
    X = pd.DataFrame(X)
    # keep only the text columns
    X = clean_mimic_data_nan(X)
    # take only str columns
    X = X.iloc[:, [3, 4, 5]]
    # print how many nan values in each column
    # nan_percentage = (X.isna().sum() / len(X)) * 100
    # print(nan_percentage)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, len(X.columns), map_test


def load_text_data():
    sentences_of_diabetes_diagnosis = [
        'Monitoring blood glucose levels is crucial for diabetes management.',
        'Type 1 diabetes is an autoimmune condition that typically develops in childhood.',
        'Type 2 diabetes can often be managed with lifestyle changes and medication.',
        'Continuous glucose monitoring devices help individuals track their blood sugar levels.',
        'Regular exercise can improve insulin sensitivity in individuals with diabetes.',
        'Diabetes education programs teach patients about managing their condition effectively.',
        'A balanced diet rich in whole grains and vegetables is recommended for diabetes prevention.',
        'Diabetes mellitus is characterized by high levels of sugar in the blood.',
        'Elevated fasting blood glucose levels indicate possible diabetes.',
        'Insulin resistance is a common precursor to type 2 diabetes.',
        'Family history of diabetes increases the risk of developing the condition.',
        'Excessive thirst and frequent urination are classic symptoms of diabetes.',
        'Hemoglobin A1c levels above 6.5% suggest diabetes mellitus.',
        'Obesity and sedentary lifestyle contribute to the onset of diabetes.',
        'Diabetic retinopathy is a complication that affects the eyes in diabetes patients.',
        'Neuropathy, characterized by tingling or numbness, is a common diabetes-related complication.',
        'Gestational diabetes can occur during pregnancy and requires monitoring.',
        'Ketoacidosis is a serious condition that can occur in untreated type 1 diabetes.', 'high suger',
        'problem in suger', "I have high blood sugar.",
        "I need insulin shots.",
        "My doctor says I have diabetes.",
        "I feel thirsty all the time.",
        "I have to check my blood sugar every day.",
        "My blood sugar is often too high.",
        "I need to take diabetes medication.",
        "I have to watch what I eat because of diabetes.",
        "My blood sugar levels are unstable.",
        "I visit my doctor regularly for my diabetes.",
        "I experience frequent urination.",
        "I have numbness in my feet.",
        "I was diagnosed with diabetes last year.",
        "I follow a special diet for diabetes.",
        "My doctor monitors my blood sugar levels closely.",
        "I have type 1 diabetes.",
        "I have type 2 diabetes.",
        "I use a continuous glucose monitor.",
        "My hemoglobin A1c levels are high.",
        "I am managing my diabetes with medication.",
        "I need to monitor my blood sugar levels every day.",
        "I follow a diabetic diet to manage my condition.",
        "My blood sugar spikes after meals.",
        "I have regular check-ups for my diabetes.",
        "I have to be careful with my carbohydrate intake.",
        "I experience fatigue due to my diabetes.",
        "I need to take medication to control my diabetes.",
        "I have diabetes-related eye issues.",
        "I have been living with diabetes for several years.",
        "I carry glucose tablets for emergencies.",
        "My blood sugar drops if I don't eat regularly.",
        "I have a family history of diabetes.",
        "I am on a low-sugar diet.",
        "I use an insulin pump to manage my diabetes.",
        "I get my HbA1c levels checked regularly.",
        "I have to avoid sugary foods because of my diabetes.",
        "I need to watch my blood sugar closely.",
        "I have diabetes-related nerve pain.",
        "I take insulin injections daily.",
        "I have a special meal plan for my diabetes",
        "I need to be careful with my diet because of diabetes.",
        "I have to plan my meals carefully.",
        "I experience dizziness when my blood sugar is low.",
        "I keep a record of my blood sugar readings.",
        "I attend diabetes education classes.",
        "I have to take care of my feet because of diabetes.",
        "I use special shoes for my diabetic feet.",
        "I check my blood sugar before and after exercising.",
        "I need to avoid high-sugar foods.",
        "I visit my endocrinologist regularly.",
        "I need to eat snacks to avoid low blood sugar.",
        "I am careful about my carbohydrate intake.",
        "I follow my doctor's advice to manage my diabetes.",
        "I have a diabetes management plan.",
        "I need to balance my insulin doses with my food intake.",
        "I check my blood sugar levels multiple times a day.",
        "I don't need to worry about my blood sugar levels.",
        "I can eat anything without worrying about diabetes.",
        "I have a healthy and active lifestyle.",
        "I don't have to take medication for my blood sugar.",
        "I have never experienced symptoms of diabetes.",
        "I feel great and have no health issues.",
        "I don't need to follow a special diet.",
        "I enjoy a variety of foods without restrictions.",
        "I don't need to check my blood sugar levels.",
        "I have normal health check-ups.",
        "I don't have any family members with diabetes.",
        "I have a balanced diet and stay active.",
        "I don't need to visit the doctor for blood sugar issues.",
        "I maintain my weight and stay fit.",
        "I can eat sweets in moderation without any issues.",
        "I don't need to monitor my health constantly.",
        "I have good overall health and well-being.",
        "I enjoy good health and physical fitness.",
        "I don't have any chronic health conditions."

    ]
    sentences_of_health_people = [
        'Regular physical activity is important for maintaining overall health.',
        'A balanced diet rich in fruits and vegetables supports good health.',
        'Adequate sleep is essential for physical and mental well-being.',
        'Maintaining a healthy weight reduces the risk of chronic diseases.',
        'Hydrating properly throughout the day supports optimal bodily functions.',
        'Regular medical check-ups help detect early signs of health issues.',
        'Practicing stress management techniques promotes mental resilience.',
        'Avoiding smoking and excessive alcohol consumption improves health outcomes.',
        'Social connections and community engagement contribute to overall well-being.',
        'Healthy habits include brushing teeth twice daily and flossing regularly.',
        'Normal fasting blood sugar levels typically range between 70 to 100 milligrams per deciliter (mg/dL).',
        'Postprandial (after-meal) blood sugar levels in healthy individuals usually stay below 140 mg/dL.',
        'HbA1c levels below 5.7% are considered normal, indicating good long-term blood sugar control.',
        'Non-diabetic individuals maintain stable blood sugar levels throughout the day.',
        'Physical activity can help regulate blood sugar levels even in individuals without diabetes.',
        'Healthy eating habits can prevent blood sugar spikes and maintain stable glucose levels.',
        'Regular monitoring of blood sugar levels can help individuals understand their body\'s glucose regulation.',
        'Stress management and adequate sleep contribute to maintaining healthy blood sugar levels.',
        'Age and genetics can influence individual variations in normal blood sugar levels.',
        'Maintaining a healthy lifestyle reduces the risk of developing abnormal blood sugar levels.',
        "I have normal blood sugar levels.",
        "I do not have diabetes.",
        "My blood sugar is always stable.",
        "I do not need to take insulin.",
        "I eat a regular diet without restrictions.",
        "I do not need to monitor my blood sugar.",
        "I have never been diagnosed with diabetes.",
        "I feel healthy and energetic.",
        "I do not experience excessive thirst.",
        "I have no issues with frequent urination.",
        "I maintain a healthy weight.",
        "I exercise regularly and eat well.",
        "I do not have any family history of diabetes.",
        "I do not take any diabetes medication.",
        "My doctor says I am healthy.",
        "I have no symptoms of diabetes.",
        "My blood tests are always normal.",
        "I have good overall health.",
        "I do not have any health problems.",
        "I maintain a balanced diet and active lifestyle.",
        "I don't have to monitor my blood sugar.",
        "I can eat a variety of foods without restrictions.",
        "I have no problems with my blood sugar levels.",
        "I feel energetic and healthy.",
        "I don't take any medication for blood sugar.",
        "I have never had issues with high blood sugar.",
        "My health check-ups are always normal.",
        "I don't need to follow a special diet.",
        "I don't have any symptoms of diabetes.",
        "I maintain a balanced diet without worrying about sugar.",
        "I have no need for insulin or other diabetes medication.",
        "I do not experience any diabetic symptoms.",
        "I enjoy regular physical activity.",
        "I have never been told I have high blood sugar.",
        "I have no need to track my blood sugar levels.",
        "I can enjoy sweets in moderation without problems.",
        "My blood sugar levels are always within normal range.",
        "I don't have a family history of diabetes.",
        "I don't worry about my blood sugar levels.",
        "I maintain my health through regular exercise and a balanced diet."

    ]
    # open diabetes_prediction_text
    diabetes_prediction_text = pd.read_csv(
        r'C:\Users\kashann\PycharmProjects\FS-EHR\RL\DATA\diabetes_prediction_text.csv')
    # convert all columns exept for the text column to numeric
    # Assuming df is your DataFrame
    diabetes_prediction_text.iloc[:, :-2] = diabetes_prediction_text.iloc[:, :-2].astype(float)

    # take 100 samples from diabetes_prediction_text
    diabetes_prediction_text = diabetes_prediction_text.sample(n=2000, random_state=1)
    diabetes_prediction_text['text'] = np.where(diabetes_prediction_text['diabetes'] == 0,
                                                np.random.choice(sentences_of_health_people,
                                                                 diabetes_prediction_text.shape[0]),
                                                np.random.choice(sentences_of_diabetes_diagnosis,
                                                                 diabetes_prediction_text.shape[0]))

    labels = diabetes_prediction_text.iloc[:, -1].values.astype(int)  # Labels as numpy array of integers
    diabetes_prediction_text = diabetes_prediction_text.drop(columns=['diabetes'])  # Drop the label column
    diabetes_prediction_text, labels = balance_class_no_noise(diabetes_prediction_text.to_numpy(), labels)
    diabetes_prediction_text = pd.DataFrame(diabetes_prediction_text)

    return diabetes_prediction_text, labels, 9


def import_breast():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_prognostic.data.features.to_numpy()
    y = breast_cancer_wisconsin_prognostic.data.targets.to_numpy()
    y[y == 'R'] = 1
    y[y == 'N'] = 0
    X[X == 'nan'] = 0
    X = np.nan_to_num(X, nan=0)
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)

    return X, y, breast_cancer_wisconsin_prognostic.metadata.num_features, breast_cancer_wisconsin_prognostic.metadata.num_features


def load_existing_image_data():
    from pathlib import Path
    image_dir = Path(r'C:\Users\kashann\PycharmProjects\PCAFE\RL\DATA\mnist_images')
    # Load image paths
    image_paths = sorted(image_dir.glob("*.png"))
    image_paths = [str(path) for path in image_paths]

    # Generate the same random numeric feature
    # np.random.seed(42)  # Set seed for reproducibility
    # numeric_feature = np.random.rand(len(image_paths))
    labels = [int(path.split('_')[2]) % 10 for path in image_paths]

    # Create a DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        # 'numeric_feature': numeric_feature,
        'label': labels
    })
    X = df[['image_path']]
    y = df['label'].values
    num_features = X.shape[1]
    # crete y as numpy array of integers
    y = y.astype(int)

    return X, y, num_features


def load_image_data():
    # Fetch the MNIST dataset from openml
    mnist = fetch_openml('mnist_784', version=1)

    images = mnist.data.values.reshape(-1, 28, 28).astype(np.uint8)
    labels = mnist.target.astype(int)

    # Create a directory to store the images
    image_dir = Path("mnist_images")
    image_dir.mkdir(exist_ok=True)

    # Save images and store their paths in a list
    image_paths = []

    for i, (image, label) in enumerate(zip(images, labels)):
        # Include the label in the image path
        image_path = image_dir / f"label_{label}_image_{i}.png"
        Image.fromarray(image).save(image_path)
        image_paths.append(str(image_path))

    # Generate a random numeric feature
    numeric_feature = np.random.rand(len(image_paths))

    # Create a DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'numeric_feature': numeric_feature,
        'label': labels
    })

    # Create numpy arrays for image paths and numeric features
    X = df[['image_path', 'numeric_feature']].values
    y = df['label'].values
    num_features = X.shape[1]

    return X, y, num_features


def load_image():
    # Directory where the images were saved
    image_dir = Path(r'C:\Users\kashann\PycharmProjects\PCAFE\RL\DATA\mnist_images')

    # Check if the directory exists
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory {image_dir} does not exist. Please run the image saving code first.")

    # Load image paths
    image_paths = sorted(image_dir.glob("*.png"))
    image_paths = [str(path) for path in image_paths]

    labels = [int(path.split('_')[2]) % 10 for path in image_paths]

    # Create a DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    labels = df.iloc[:, -1].values.astype(int)  # Labels as numpy array of integers
    diabetes_prediction_text = df.drop(columns=['label'])  # Drop the label column

    return diabetes_prediction_text, labels, len(diabetes_prediction_text.columns)


def create_data():
    # Number of points to generate
    num_points = 100
    # Generate random x values
    x1_values = np.random.uniform(low=-30, high=30, size=num_points)
    x2_values = np.random.uniform(low=-30, high=30, size=num_points)
    x3_values = np.random.uniform(low=-30, high=30, size=num_points)
    # Create y values based on the decision boundary if x+y > 9 then 1 else 0
    labels = np.where((x1_values + x2_values + x3_values > 10), 1, 0)
    # Split the data into training and testing sets
    x = np.column_stack((x1_values, x2_values, x3_values))
    return x, labels, x1_values, 3


def create():
    # Set a random seed for reproducibility

    # Number of points to generate
    num_points = 100

    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)

    # Create a scatter plot of the dataset with color-coded labels
    plt.scatter(x1_values, x2_values, c=labels, cmap='viridis', marker='o', label='Data Points')
    # Split the data into training and testing sets
    x = np.column_stack((x1_values, x2_values))
    return x, labels, x1_values, 3


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


def create_n_dim():
    # Number of points to generate
    num_points = 2000

    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)
    # create numpy of zeros
    X = np.zeros((num_points, 10))
    i = 0
    while i < num_points:
        # choose random index to assign x1 and x2 values
        index = np.random.randint(0, 10)
        # assign x1 to index for 5 samples
        X[i][index] = x1_values[i]
        X[i + 1][index] = x1_values[i + 1]
        X[i + 2][index] = x1_values[i + 2]
        X[i + 3][index] = x1_values[i + 3]
        X[i + 4][index] = x1_values[i + 4]
        # choose random index to assign x2 that is not the same as x1
        index2 = np.random.randint(0, 10)
        while index2 == index:
            index2 = np.random.randint(0, 10)
        X[i][index2] = x2_values[i]
        X[i + 1][index2] = x2_values[i + 1]
        X[i + 2][index2] = x2_values[i + 2]
        X[i + 3][index2] = x2_values[i + 3]
        X[i + 4][index2] = x2_values[i + 4]
        i += 5
    question_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return X, labels, question_names, 10


def load_gisetta():
    data_path = "/RL/extra/gisette/gisette_train.data"
    labels_path = "/RL/extra/gisette/gisette_train.labels"
    data = []
    labels = []
    with open(labels_path, newline='') as file:
        # read line by line
        for line in file:
            if int(line) == -1:
                labels.append(0)
            else:
                labels.append(1)
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            sample = []
            columns = row[0].split(' ')
            for i in range(len(columns) - 2):
                sample.append(float(columns[i]))
            data.append(sample)
    X = np.array(data)
    y = np.array(labels)
    return X, y, y, len(sample)


def load_diabetes(file_path_clean):
    data = []
    labels = []
    # Open the CSV file
    with open(file_path_clean, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            # if it the first line (header) skip itda
            if line[0] == 'gender':
                # save the header in npy array for later use
                question_names = np.array(line)
                continue
            # columns = [column.split(',') for column in line]
            columns = line
            columns_without_label = columns[0:-1]
            if columns_without_label[0] == "Female":
                columns_without_label[0] = 0
            else:
                columns_without_label[0] = 1
            if columns_without_label[4] == "never":
                columns_without_label[4] = 0
            if columns_without_label[4] == "former":
                columns_without_label[4] = 1
            if columns_without_label[4] == "current":
                columns_without_label[4] = 2
            if columns_without_label[4] == "No Info":
                columns_without_label[4] = 3
            if columns_without_label[4] == "not current":
                columns_without_label[4] = 4
            if columns_without_label[4] == "ever":
                columns_without_label[4] = 5
            for i in range(len(columns_without_label)):
                if columns_without_label[i] != '':
                    columns_without_label[i] = float(columns_without_label[i])
                else:
                    columns_without_label[i] = 0

            data.append(columns_without_label)

            labels.append(int(float(columns[-1])))

    data, labels = balance_class(np.array(data), np.array(labels))
    X = pd.DataFrame(data)

    return X, labels, len(X.columns)


def create_demo():
    np.random.seed(34)
    Xs1 = np.random.normal(loc=1, scale=0.5, size=(300, 5))
    Ys1 = -2 * Xs1[:, 0] + 1 * Xs1[:, 1] - 0.5 * Xs1[:, 2]
    Xs2 = np.random.normal(loc=-1, scale=0.5, size=(300, 5))
    Ys2 = -0.5 * Xs2[:, 2] + 1 * Xs2[:, 3] - 2 * Xs2[:, 4]
    X_data = np.concatenate((Xs1, Xs2), axis=0)
    Y_data = np.concatenate((Ys1.reshape(-1, 1), Ys2.reshape(-1, 1)), axis=0)
    Y_data = Y_data - Y_data.min()
    Y_data = Y_data / Y_data.max()
    case_labels = np.concatenate((np.array([1] * 300), np.array([2] * 300)))
    Y_data = np.concatenate((Y_data, case_labels.reshape(-1, 1)), axis=1)
    Y_data = Y_data[:, 0].reshape(-1, 1)
    return X_data, Y_data, 5, 5


def split_XTY_by_Text_appearence(X, T, Y):
    non_empty_text_inds = []
    empty_text_inds = []
    for i in range(len(T)):
        if not pd.isna(T.iloc[i, 0]):
            non_empty_text_inds.append(i)
        else:
            empty_text_inds.append(i)

    X_non_empty_text = X[non_empty_text_inds]

    T_non_empty_text = [T.iloc[i, 0] for i in non_empty_text_inds]

    Y_non_empty_text = Y[non_empty_text_inds]

    return X_non_empty_text, T_non_empty_text, Y_non_empty_text


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


def load_data_sheba():
    ''' ------- Load data -------'''
    # read csv file
    Text0_translated = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE\RL\Data\text_batch.csv')
    # REPLACE "..." WITH None
    Text0_translated.replace('...', np.nan, inplace=True)
    number_of_samples = len(Text0_translated)
    outcomes = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\Bert\Data\outcomes.pkl')[:number_of_samples]
    Y = outcomes.to_numpy()
    Y = Y[:, [4]]
    X_pd = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\Bert\Data\preprocessed_X.pkl')[:number_of_samples]
    X = X_pd.to_numpy()

    X_train_non_empty_text, T_train_non_empty_text, Y_train_non_empty_text = split_XTY_by_Text_appearence(
        X, Text0_translated, Y)
    X_train_non_empty_text = X_train_non_empty_text.astype('float32')
    Y_train_non_empty_text = Y_train_non_empty_text.astype('int')

    # X_train_non_empty_text=X
    # T_train_non_empty_text=Text0_translated
    # Y_train_non_empty_text=Y

    Y_train_non_empty_text = Y_train_non_empty_text.squeeze()

    T_train_non_empty_text = np.array(T_train_non_empty_text).reshape(-1, 1)

    data = np.concatenate((X_train_non_empty_text, T_train_non_empty_text), axis=1)

    data, Y_train_non_empty_text = balance_class_no_noise(data, Y_train_non_empty_text)
    data = pd.DataFrame(data)

    data = data.drop(columns=[data.columns[25], data.columns[33]])
    # Rename remaining columns to have consistent indices from 0 to n-1
    data.columns = [i for i in range(len(data.columns))]
    for i in range(0, len(data.columns) - 1):
        data[i] = data[i].astype(float)
    return data, Y_train_non_empty_text, len(data.columns)


def load_data_labels():
    outcomes = pd.read_pickle(r'C:\Users\kashann\PycharmProjects\Bert\Data\outcomes.pkl')
    Y = outcomes.to_numpy()
    Y = Y[:, [0]]
    Y = Y.astype('int')
    Y = Y.reshape(-1)
    X_pd = pd.read_csv(r'C:\Users\kashann\PycharmProjects\Bert\Data\preprocessed_X_filtered.csv')
    X = X_pd.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Do not scale if using shap
    X = X.astype('float32')
    X, Y = balance_class(X, Y)
    X = pd.DataFrame(X)

    return X, Y, len(X_pd.columns)


def load_basehock():
    # read .mat file
    data = loadmat(r'C:\Users\kashann\PycharmProjects\PCAFE\RL\Data\BASEHOCK.mat')
    # get the data and labels
    X = data['X']
    Y = data['Y']
    Y = Y.astype('int')

    Y[Y == 2] = 0
    Y = Y.squeeze()
    X = pd.DataFrame(X)
    # cretae all the X numeric
    for i in range(0, len(X.columns)):
        X[i] = X[i].astype(float)
    return X, Y, len(X.columns)


def load_ucimlrepo():
    # fetch dataset
    diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
    X = diabetic_retinopathy_debrecen.data.features
    # convert X content to float
    X = X.astype(float)

    y = diabetic_retinopathy_debrecen.data.targets.to_numpy()
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)

    return X, y, diabetic_retinopathy_debrecen.metadata.num_features


def load_madelon():
    # read .mat file
    data = loadmat(r'C:\Users\kashann\PycharmProjects\PCAFE\RL\Data\madelon.mat')
    # get the data and labels
    X = data['X']
    Y = data['Y']
    Y = Y.astype('int')
    # replace '-1' with '0'
    Y[Y == -1] = 0
    Y = Y.squeeze()
    X = pd.DataFrame(X)
    # cretae all the X numeric
    for i in range(0, len(X.columns)):
        X[i] = X[i].astype(float)
    return X, Y, len(X.columns)
