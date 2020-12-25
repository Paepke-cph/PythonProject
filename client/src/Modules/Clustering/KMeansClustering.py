import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing


class KMeansClustering:
    def __init__(self, dataset, make):
        self.make = make
        self.dataset = dataset[dataset['Make'] == self.make]
        self.features = self.dataset[['Km/l', 'Hestekræfter']]
        self.kmeans = KMeans(n_clusters=2)
        self.kmeans.fit(self.features)
        self.centroids = self.kmeans.cluster_centers_

    def plot_clustering(self):
        plt.scatter(dataset['Km/l'], dataset['Hestekræfter'],
                    c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

    def predict_cluster(self, feature):
        return self.kmeans.predict(feature)

    def predict_and_plot_cluster(self, feature, k):
        colors = ['grey', 'green', 'blue', 'yellow', 'pink', 'cyan']
        prediction = self.predict_cluster(feature)
        self.dataset['cluster'] = prediction
        for i in range(0, k):
            cluster = self.dataset[self.dataset['cluster']
                                   == i][['Km/l', 'Hestekræfter']]
            plt.scatter(cluster['Km/l'], cluster['Hestekræfter'], c=colors[i])
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', s=10, marker='*')
        plt.show()
        self.dataset.drop(['cluster'], axis=1)

    def plot_best_k_value(self):
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            max_iter=300, n_init=10, random_state=0)
            kmeans.fit(self.features)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
