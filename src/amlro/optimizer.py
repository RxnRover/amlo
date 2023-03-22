import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
from typing import List, Dict, Tuple, Any


#class Optimizer:
def load_data(
        training_file: str, combination_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loading the training set file and all combination
        file as pandas data frames and split into x train , 
        y train and test datasets. When loading the combination file,
        data rows will be deleted if they include in training file.

    :param training_file: path to the training set file.
    :type training_file: str
    :param combination_file: path to the combination file
    :type combination_file: str
    :return: x and y traning datasets and test dataset
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    train = pd.read_csv(training_file, skiprows=0)
    y_train = train["Yield"]
    x_train = train.drop("Yield", axis=1)

    data = pd.read_csv(combination_file, skiprows=0)
    data = data.drop_duplicates()

    data = data.loc[
        ~data.index.isin(
            data.merge(x_train.assign(a="key"), how="left").dropna().index
        )
    ]

    return x_train, y_train, data

def model_training(x_train: pd.DataFrame, y_train: pd.DataFrame):
    """ traning the regressor model and return the best model.

    :param x_train: training dataset.
    :type x_train: Dataframe
    :param y_train: target dataframe for training.
    :type y_train: Dataframe
    :return: trained regressor model
    :rtype: model
    """
    kfold = KFold(n_splits=10, shuffle=True)
    # regr = RandomForestRegressor()#25 100
    regr = GradientBoostingRegressor()
    # regr = AdaBoostRegressor()
    # regr = SVR()

    estimators_int = list(range(100, 1000, 50))
    param_grid = {"n_estimators": estimators_int, "max_depth": [None, 2, 4]}
    # param_grid = {'n_estimators': estimators_int, 'loss': ['linear','square', 'exponentia']}
    # param_grid = {'kernel': ['linear','poly', 'rbf','sigmoid'], 'epsilon':[0.1,0.01,0.05]}
    grid = GridSearchCV(estimator=regr, param_grid=param_grid, cv=kfold, n_jobs=6)

    grid_result = grid.fit(x_train, y_train)
    best_params = pd.DataFrame(
        [grid.best_params_], columns=grid.best_params_.keys()
    )
    regr = grid.best_estimator_

    return regr

def predict_next_parameters(regr, data: pd.DataFrame) -> pd.DataFrame:
    """predicting the yield from all the combination data using trained 
    regressor model and return the best combinations.

    :param regr: trained regressor model
    :type regr: model
    :param data: test dataset
    :type data: pd.Dataframe
    :return: best five predicted parameter set
    :rtype: pd.Dataframe
    """
    pred = regr.predict(data)
    data["prediction"] = pred

    best_combo = data.sort_values(by=["prediction"], ascending=False).iloc[:5]
    best_combo = best_combo.drop("prediction", axis=1)

    return best_combo

def write_data_to_training(training_file: str, prev_parameters: str) -> None:
    """writing the prev best predicted combination and 
    experimental yield at the end of the training set file.

    :param training_file: traning set file path
    :type training_file: str
    :param prev_parameters: previous best combo and yield
    :type prev_parameters: str
    """
    # Open the file in append & read mode ('a+')
    with open(training_file, "a+") as file_object:
        file_object.write(prev_parameters + "\n")

def categorical_feature_decoding(config: Dict, best_combo: List[Any]) -> List[Any]:
    """This method converts encoded parameter list into decoded list.
        Convert categorical veriable values back into its names.

    :param config: Initial reaction feature configurations
    :type config: Dict
    :param best_combo: parameter list required for decoding
    :type best_combo: list
    :return: Decoded parameter list
    :rtype: list
    """

    numerical_feature_count = len(config["continuous"]["feature_names"])
    numerical_combo = best_combo[0:numerical_feature_count]
    cat_combo = best_combo[numerical_feature_count:]

    for i in range(len(cat_combo)):
        x = config["categorical"]["values"][i]
        cat_combo[i] = x[int(cat_combo[i])]
       

    best_combo_with_names = []
    [best_combo_with_names.append(elem) for elem in numerical_combo]
    [best_combo_with_names.append(elem) for elem in cat_combo]
    return best_combo_with_names

def categorical_feature_encoding(
    config: Dict, prev_parameters: List[Any]
) -> List[Any]:
    """This method converts decoded parameter list into encoded list.
        Convert categorical veriable values into its numerical values.

    :param config: Initial reaction feature configurations
    :type config: Dict
    :param prev_parameters: parameter list required for encoding
    :type prev_parameters: list
    :return: encoded parameter list
    :rtype: list
    """

    numerical_feature_count = len(config["continuous"]["feature_names"])
    numerical_combo = prev_parameters[0:numerical_feature_count]
    cat_combo = prev_parameters[numerical_feature_count:]

    for i in range(len(cat_combo)):
        cat_list = config["categorical"]["values"][i]
        for x in range(len(cat_list)):
            if cat_list[x] == cat_combo[i]:
                cat_combo[i] = x
    prev_parameters_encode = []
    [prev_parameters_encode.append(elem) for elem in numerical_combo]
    [prev_parameters_encode.append(elem) for elem in cat_combo]

    return np.array(prev_parameters_encode)

