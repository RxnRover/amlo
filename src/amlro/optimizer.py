import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics

class optimizer:
    def load_data(training_file,combination_file):
        train = pd.read_csv(training_file,skiprows=1)
        #train = train.reindex(columns=['Time','Flow1','Flow2','Flow3','Temperature','Yield'])
        y_train = train['Yield']
        x_train = train.drop('Time', axis=1)
        x_train = x_train.drop('Yield', axis=1)
        data = pd.read_csv(combination_file,skiprows=1)
        #data = data.reindex(columns=['Time','Flow1','Flow2','Flow3','Temperature'])
        data = data.drop('Time', axis=1)
        data = data.drop('Yield', axis=1)
        data = data.drop_duplicates()
        #data = pd.concat([data,x_train]).drop_duplicates(keep=False).dropna()
        data = data.loc[~data.index.isin(data.merge(x_train.assign(a='key'),how='left').dropna().index)]
        return x_train,y_train,data
    def model_training(x_train,y_train):
        kfold = KFold(n_splits=10, shuffle=True)
        #regr = RandomForestRegressor()#25 100
        regr = GradientBoostingRegressor()
        #regr = AdaBoostRegressor()
        #regr = SVR()
        estimators_int = list(range(100, 1000, 50))
        param_grid = {'n_estimators': estimators_int, 'max_depth': [None, 2, 4]}
        #param_grid = {'n_estimators': estimators_int, 'loss': ['linear','square', 'exponentia']}
        #param_grid = {'kernel': ['linear','poly', 'rbf','sigmoid'], 'epsilon':[0.1,0.01,0.05]}
        grid = GridSearchCV(estimator=regr, param_grid=param_grid, cv=kfold, n_jobs=6)
        grid_result = grid.fit(x_train, y_train)
        best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())
        regr = grid.best_estimator_
        return regr
    def predict_next_parameters(regr,data):
        pred = regr.predict(data)
        data['prediction'] = pred
        print(data)
        best_combo = data.sort_values(by=['prediction'], ascending=False).iloc[:5]
        return best_combo,data

    def write_data_to_training(training_file,prev_parameters):
        # Open the file in append & read mode ('a+')
        with open(training_file, "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0 :
                file_object.write("\n")
            # Append text at the end of file
            file_object.write(prev_parameters)
            file_object.close()
