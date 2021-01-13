import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class MultiLinearReg:
    def __init__(self, dataset, make):
        self.dataset = dataset[dataset['Make'] == make]
        self.make = make
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values
        le = preprocessing.LabelEncoder()
        self.X[:, 0] = le.fit_transform(self.X[:, 0])
        self.X[:, 1] = le.fit_transform(self.X[:, 1])
        self.X[:, 2] = le.fit_transform(self.X[:, 2])
        self.X[:, 3] = le.fit_transform(self.X[:, 3])
        scaler = StandardScaler()
        scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.regressor.predict(self.X_test)

    def predict_price(self, input):
        return self.regressor.predict(input)

    def score(self):
        return r2_score(self.y_test, self.y_pred)
