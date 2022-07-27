import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
from typing import List, Dict, Tuple


class optimizer:

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
        :rtype: Dataframe,Dataframe, Dataframe
        """
        train = pd.read_csv(training_file, skiprows=1)
        y_train = train['Yield']
        x_train = train.drop('Yield', axis=1)

        data = pd.read_csv(combination_file, skiprows=1)
        data = data.drop('Yield', axis=1)

        data = data.drop_duplicates()
        #data = pd.concat([data,x_train]).drop_duplicates(keep=False).dropna()
        data = data.loc[~data.index.isin(
            data.merge(x_train.assign(a='key'), how='left').dropna().index)]

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
        #regr = AdaBoostRegressor()
        #regr = SVR()

        estimators_int = list(range(100, 1000, 50))
        param_grid = {
            'n_estimators': estimators_int,
            'max_depth': [None, 2, 4]
        }
        #param_grid = {'n_estimators': estimators_int, 'loss': ['linear','square', 'exponentia']}
        #param_grid = {'kernel': ['linear','poly', 'rbf','sigmoid'], 'epsilon':[0.1,0.01,0.05]}
        grid = GridSearchCV(estimator=regr,
                            param_grid=param_grid,
                            cv=kfold,
                            n_jobs=6)

        grid_result = grid.fit(x_train, y_train)
        best_params = pd.DataFrame([grid.best_params_],
                                   columns=grid.best_params_.keys())
        regr = grid.best_estimator_

        return regr

    def predict_next_parameters(regr, data: pd.DataFrame) -> pd.DataFrame:
        """predicting the yield from all the combination data using trained 
        regressor model and return the best combinations.

        :param regr: trained regressor model
        :type regr: model
        :param data: test dataset
        :type data: Dataframe
        :return: best five predicted parameter set
        :rtype: Dataframe
        """
        pred = regr.predict(data)
        data['prediction'] = pred
        

        best_combo = data.sort_values(by=['prediction'],
                                      ascending=False).iloc[:5]

        return best_combo

    def write_data_to_training(training_file: str,
                               prev_parameters: str) -> None:
        """writing the prev best predicted combination and 
        experimental yield at the end of the training set file.

        :param training_file: traning set file path
        :type training_file: str
        :param prev_parameters: previous best combo and yield
        :type prev_parameters: str
        """
        # Open the file in append & read mode ('a+')
        with open(training_file, "a+") as file_object:
            file_object.write(prev_parameters + '\n')
            
