import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# This model is able to predict car prices using the RandomForestRegression
class RandomForestReg:
    def __init__(self, dataset, make):
        self.make = make
        self.dataset = dataset[dataset['Make'] == self.make]
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values

        # Using LabelEncoder to encode string columns
        le = preprocessing.LabelEncoder()
        self.X[:, 0] = le.fit_transform(self.X[:, 0])
        self.X[:, 1] = le.fit_transform(self.X[:, 1])
        self.X[:, 2] = le.fit_transform(self.X[:, 2])
        self.X[:, 3] = le.fit_transform(self.X[:, 3])

        # Train TEst Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

        # Create the RandomForestRegressor
        self.regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        self.regressor.fit(self.X_train, self.y_train)

        # y_pred is only used for scoring the model.
        self.y_pred = self.regressor.predict(self.X_test)

    def predict_price(self, input):
        return self.regressor.predict(input)

    def score(self):
        return r2_score(self.y_test, self.y_pred)
