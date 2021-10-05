import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
import os
from scipy.io import arff
from sklearn.datasets import load_digits


def preprocess_iris():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "iris.data")
    df = pd.read_csv(file_path, sep=',')
    headers = list(df.columns)
    headers.remove('class')
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # fit scalar on data
    norm = MinMaxScaler().fit(x)
    # transform data
    x = norm.transform(x)
    le = preprocessing.LabelEncoder()
    le.fit(y.unique())
    y = le.transform(y)
    return x, y, len(np.unique(y))

def preprocess_seeds():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "seeds_dataset.txt")
    df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # fit scalar on data
    norm = MinMaxScaler().fit(x)
    # transform data
    x = norm.transform(x)
    le = preprocessing.LabelEncoder()
    le.fit(y.unique())
    y = le.transform(y)
    return x, y, len(np.unique(y))

def preprocess_dermatology():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "dermatology.data")
    df = pd.read_csv(file_path, sep=',')
    # missing values only in 34 columns
    median_list = list(df.iloc[:, 33].replace('?', '0'))
    median_list = [int(x) for x in median_list]
    median_value = np.median(median_list)
    df = df.replace('?', median_value)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    # fit scaler on data
    norm = StandardScaler().fit(X)
    # transform data
    x = norm.transform(X)
    le = preprocessing.LabelEncoder()
    le.fit(Y.unique())
    y = le.transform(Y)
    return x, y, len(np.unique(y))

def preprocess_breast():
    """
    Apply the personalized operations to preprocess the database.
    :return: dict:
            db: 2D data array of size (rows, features),
            label_true: array of true label values,
            data_frame: raw data set with filled missing values in.
    """
    # load dataset
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "breast-w.arff")
    data, meta = arff.loadarff(f)
    dataset = np.array(data.tolist(), dtype=object)
    meta = meta.names()  # list containing column names

    # create a initial pandas dataframe
    df = pd.DataFrame(data=dataset, columns=list(meta))

    # detect missing values and replacing using median
    if df.isnull().any().sum() > 0:
        for x in meta[:-1]:
            median = df[x].median()
            df[x].fillna(median, inplace=True)

    # split-out dataset
    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # get the true labels values of the dataset
    le = preprocessing.LabelEncoder()
    le.fit(y.unique())
    y = le.transform(y)

    # fit MinMax scaler on data
    norm = MinMaxScaler().fit(x)
    # transform data
    x = norm.transform(x)

    return x, y, len(np.unique(y))

def preprocess_pendigits():
    digits = load_digits()
    data = digits.data
    norm = MinMaxScaler().fit(data)
    # transform data
    x = norm.transform(data)
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(digits.target))
    y = le.transform(digits.target)
    return x, y, len(np.unique(y))

def read_data(dataset_name):
    if dataset_name == "iris":
        return preprocess_iris()
    elif dataset_name == "breast-w":
        return preprocess_breast()
    elif dataset_name == "pendigits":
        return preprocess_pendigits()
    elif dataset_name == "seeds":
        return preprocess_seeds()
    elif dataset_name == "dermatology":
        return preprocess_dermatology()