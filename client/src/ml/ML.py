import pandas as pd
from Modules import Cleaner
from Modules.Regression.LinearReg import LinearReg
from Modules.Regression.DecisionTreeReg import DecisionTreeReg
from Modules.Regression.MultiLinearReg import MultiLinearReg
from Modules.Regression.PolynomialReg import PolynomialReg
from Modules.Regression.RandomForestReg import RandomForestReg
from Modules.Clustering.KMeansClustering import KMeansClustering
from Modules.Clustering.KNNClustering import KNNClustering


class ML:
    def __init__(self, datafile="../data/bilhandel_unclean.csv"):
        self.test_car = [[0, 0, 0, 6, 85000, 2016, 120, 5,
                          6, 17.2, 9.9, 187.0, 1.198, 1.88, 290998.0]]
        
        self.datafile = Cleaner.get_and_clean_df(datafile)
        self.linear_regression = LinearReg(self.datafile, make='Mazda')
        self.multi_linear_regression = MultiLinearReg(
            self.datafile, make="Mazda")
        self.polynomial_regression = PolynomialReg(self.datafile, make="Mazda")
        self.random_forest_regression = RandomForestReg(
            self.datafile, make="Mazda")
        self.decisiontree_regression = DecisionTreeReg(
            self.datafile, make="Mazda")

        self.kmeans_clustering = KMeansClustering(self.datafile, make='Mazda')
        self.knn_clustering = KNNClustering(self.datafile, make='Mazda')

# 0        Score       Count      ML Model
# Mazda    1          1000        Linear

    def model_comparison(self):
        makes = ['Mazda', 'VW', 'Mercedes', 'Ford', 'Audi']
        df = pd.DataFrame(columns=['Make','Model', 'Score', 'Number of Cars'])
        for m in makes:
            # Linear Regression
            linear_regression = LinearReg(self.datafile, make=m)
            lin_score = linear_regression.score()
            count = len(linear_regression.dataset.index)
            
            # Multilinear Regression
            multilinear_regression = MultiLinearReg(self.datafile, make=m)
            mul_score = multilinear_regression.score()

            # Polynomial Regression
            poly_regression = PolynomialReg(self.datafile, make=m)
            poly_score = poly_regression.score()

            appendDF = pd.DataFrame({'Make':m,
                                    'Model':['Linear Regression', 'Multilinear Regression','Polynomial Regression'],
                                    'Score':[lin_score, mul_score,poly_score],
                                    'Number of Cars':[count,count,count]})
            df = df.append(appendDF)

        print(df.set_index('Make'))





    def run_linear_regression(self):
        print(self.linear_regression.predict_price_on_car(self.test_car))
        print(self.linear_regression.score())
        self.linear_regression.compare_km_price()
        self.linear_regression.compare_year_price()

    def run_multilinear_regression(self):
        print(self.multi_linear_regression.score())
        print(self.multi_linear_regression.predict_price(self.test_car))

    def run_polynomial_regression(self):
        print(self.polynomial_regression.score_year_vs_price())
        self.polynomial_regression.plot_year_vs_price()
        print(self.polynomial_regression.score_km_vs_year())
        self.polynomial_regression.plot_km_vs_price()

    def run_decisiontree_regression(self):
        print(self.decisiontree_regression.score())
        print(self.decisiontree_regression.predict_price(self.test_car))

    def run_randomforest_regression(self):
        print(self.random_forest_regression.score())
        print(self.random_forest_regression.predict_price(self.test_car))

    def run_kmeans_clustering(self):
        self.kmeans_clustering.plot_best_k_value()
        self.kmeans_clustering.predict_and_plot_cluster(
            self.kmeans_clustering.features, 2)

    def run_knn_clustering(self):
        self.knn_clustering.plot_best_k_value()
        self.knn_clustering.predict_and_print(self.knn_clustering.X_test, 20)


if __name__ == "__main__":
    ml = ML()
    ml.model_comparison()






