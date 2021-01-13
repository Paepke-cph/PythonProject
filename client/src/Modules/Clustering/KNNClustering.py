
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# K-Nearest Neighbour clustering used for predicting fueltype
# This is not clustering but classification

class KNNClustering:
    def __init__(self, dataset, make):
        self.make = make
        self.dataset = dataset[dataset['Make'] == self.make]
        self.X = self.dataset.loc[:, ['Make', 'Gearkasse', 'Model', 'Kilometer', 'Årgang', 'Hestekræfter', 'Antal døre',
                                      'Antal gear', 'Km/l', '0 - 100 km/t', 'Tophastighed', 'Vægt', 'Grøn Ejerafgift', 'Nypris', 'Pris']].values
        self.y = self.dataset.loc[:, ['Brændstoftype']].values
        le = preprocessing.LabelEncoder()
        self.X[:, 0] = le.fit_transform(self.X[:, 0])
        self.X[:, 1] = le.fit_transform(self.X[:, 1])
        self.X[:, 2] = le.fit_transform(self.X[:, 2])
        self.X[:, 3] = le.fit_transform(self.X[:, 3])
        scaler = StandardScaler()
        scaler.fit(self.X)
        scaled_features = scaler.transform(self.X)
        self.dataset_features = pd.DataFrame(scaled_features, columns=['Make', 'Gearkasse', 'Model', 'Kilometer', 'Årgang', 'Hestekræfter',
                                                                       'Antal døre', 'Antal gear', 'Km/l', '0 - 100 km/t', 'Tophastighed', 'Vægt', 'Grøn Ejerafgift', 'Nypris', 'Pris'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            scaled_features, self.y, test_size=0.30)

    def predict_fueltype(self, input, k=2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.X_train, self.y_train)
        return knn.predict(input)

    def predict_and_print(self, input, k=2):
        prediction = self.predict_fueltype(input, k)
        print(f'WITH K={k}')
        print('\n')
        print(confusion_matrix(self.y_test, prediction))
        print('\n')
        print(classification_report(self.y_test, prediction, zero_division=1))

    def plot_best_k_value(self):
        error_rate = []
        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.X_train, self.y_train)
            pred_i = knn.predict(self.X_test)
            error_rate.append(np.mean(pred_i != self.y_test))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.show()
