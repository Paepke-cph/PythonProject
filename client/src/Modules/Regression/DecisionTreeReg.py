import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


# This class uses Decision Tree Regression to try and predict the price?????
# This should perhapse not be used as of noew....
class DecisionTreeReg:
    def __init__(self, dataset, make):
        # Get Dataset specifict to the specified make
        self.dataset = dataset[dataset['Make'] == make]
        self.make = make
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values
        # Encode all columns with strings (Make,model,brændstoftype...)
        le = preprocessing.LabelEncoder()
        self.X[:, 0] = le.fit_transform(self.X[:, 0])
        self.X[:, 1] = le.fit_transform(self.X[:, 1])
        self.X[:, 2] = le.fit_transform(self.X[:, 2])
        self.X[:, 3] = le.fit_transform(self.X[:, 3])

        # Train Test Split for
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=0)

        # Create the regressor and fit it to the model.
        self.regressor = DecisionTreeRegressor(random_state=0)
        self.regressor.fit(self.X, self.y)
        # y_pred is only used for scoring the model.
        self.y_pred = self.regressor.predict(self.X_test)

    def predict_price(self, input):
        return self.regressor.predict(input)

    def score(self):
        return r2_score(self.y_test, self.y_pred)
