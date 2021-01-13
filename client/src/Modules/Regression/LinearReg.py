import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# This class uses Linear Regression to see if there should be a coorelation between
# 1. Kilometer Vs Price
# 2. Year Vs Price
class LinearReg:
    def __init__(self, dataset, make):
        self.make = make
        self.dataset = dataset[dataset['Make'] == self.make]
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        # Encode all columns containning string into usable values.
        le = preprocessing.LabelEncoder()
        X[:, 0] = le.fit_transform(X[:, 0])
        X[:, 1] = le.fit_transform(X[:, 1])
        X[:, 2] = le.fit_transform(X[:, 2])
        X[:, 3] = le.fit_transform(X[:, 3])
        # Train Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.regressor.predict(self.X_test)

    def score(self):
        return r2_score(self.y_test, self.y_pred)

    def predict_price_on_car(self, car):
        return self.regressor.predict(car)

    def compare_km_price(self):
        pd_km_price = self.dataset[self.dataset['Make']
                                   == self.make][['Kilometer', 'Pris']]
        pd_km_price.plot.scatter(x='Kilometer', y='Pris')
        m, b = np.polyfit(
            pd_km_price['Kilometer'].values, pd_km_price['Pris'].values, 1)
        plt.plot(pd_km_price['Kilometer'].values,
                 m*pd_km_price['Kilometer'].values+b)
        plt.show()

    def compare_year_price(self):
        pd_year_price = self.dataset[self.dataset['Make']
                                     == self.make][['Årgang', 'Pris']]
        pd_year_price.plot.scatter(x='Årgang', y='Pris')
        m, b = np.polyfit(
            pd_year_price['Årgang'].values, pd_year_price['Pris'].values, 1)
        plt.plot(pd_year_price['Årgang'].values,
                 m*pd_year_price['Årgang'].values+b)
        plt.show()
