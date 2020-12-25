import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# This class will make a comparison between the following:
# 1. Year VS Price
# 2. Price VS Kilometer


class PolynomialReg:
    def __init__(self, dataset, make):
        # Year VS Price
        # Sort values by year and fetch only year column for X and Price for Y
        self.make = make
        self.dataset_year_new = dataset.sort_values(by=['Årgang'])
        self.dataset_year_new = self.dataset_year_new[self.dataset_year_new['Make'] == self.make]
        self.X_year = self.dataset_year_new.iloc[:, -1]
        self.y_year = self.dataset_year_new.iloc[:, 12:13]
        # Kilometer VS Price
        # Sort value by kilometer and fetch ony kilometer column  for X and Price for Y
        self.dataset_km_new = dataset.sort_values(by=['Kilometer'])
        self.dataset_km_new = self.dataset_km_new[self.dataset_km_new['Make'] == self.make]
        self.X_km = self.dataset_km_new.iloc[:, 7:8]
        self.y_km = self.dataset_km_new.iloc[:, 12:13]

        # Train Test Split for Year VS Price
        self.X_year_train, self.X_year_test, self.y_year_train, self.y_year_test = train_test_split(
            self.X_year, self.y_year, test_size=0.2, random_state=0)
        # Train Test Split for Kilometer VS Price
        self.X_km_train, self.X_km_test, self.y_km_train, self.y_km_test = train_test_split(
            self.X_km, self.y_km, test_size=0.2, random_state=0)

        # Year VS Price - model fitting
        self.poly_year_reg = PolynomialFeatures(degree=3)
        self.X_year_poly = self.poly_year_reg.fit_transform(self.X_year_train)
        self.lin_reg_year_2 = LinearRegression()
        self.lin_reg_year_2.fit(self.X_year_poly, self.y_year_train)

        # Kilometer Vs Price - model fitting
        self.poly_km_reg = PolynomialFeatures(degree=3)
        self.X_km_poly = self.poly_km_reg.fit_transform(self.X_km_train)
        self.lin_reg_km_2 = LinearRegression()
        self.lin_reg_km_2.fit(self.X_km_poly, self.y_km_train)

    # Plots a polynomial graph for the comparison: Year Vs Price
    def plot_year_vs_price(self):
        plt.scatter(self.X_year, self.y_year, color='red')
        plt.title('Year vs Price (Polynomial Regression)')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.plot(self.X_year, self.lin_reg_year_2.predict(
            self.poly_year_reg.fit_transform(self.X_year)), color='blue')
        plt.show()

    # Plots a polynomial graph for the comparison: Kilometer Vs Price:
    def plot_km_vs_price(self):
        plt.scatter(self.X_km, self.y_km, color='red')
        plt.title('Price vs km (Polynomial Regression)')
        plt.xlabel('km')
        plt.ylabel('Price')
        plt.plot(self.X_km, self.lin_reg_km_2.predict(
            self.poly_km_reg.fit_transform(self.X_km)), color='blue')
        plt.show()

    # Calculates the score of the Year Vs Price model.
    def score_year_vs_price(self):
        predict_year = self.lin_reg_year_2.predict(
            self.poly_year_reg.fit_transform(self.X_year_test))
        return self.lin_reg_year_2.score(self.poly_year_reg.fit_transform(self.X_year_test), self.y_year_test)

    # Calculates the score of the Kilometer Vs Price model.
    def score_km_vs_year(self):
        predict_km = self.lin_reg_km_2.predict(
            self.poly_km_reg.fit_transform(self.X_km_test))
        return self.lin_reg_km_2.score(self.poly_km_reg.fit_transform(self.X_km_test), self.y_km_test)
