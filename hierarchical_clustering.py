from data_handling import dataset_split
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn import metrics


def agglomerative_clustering(x, y, n_c, metric):
    print("Hierarchical Clustering")

    agglomerative_cluster = AgglomerativeClustering(n_clusters= 2, metric= 'euclidean', linkage='ward')
    # agglomerative_cluster.

def test(training_data, features):
    # https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/
    # https://www.geeksforgeeks.org/machine-learning/implementing-agglomerative-clustering-using-sklearn/
    # https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering-with-scikit-learn/
    # https://developer.ibm.com/tutorials/awb-implement-hierarchical-clustering-python/
    # https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python

    complete_clustering = linkage(training_data, method="ward", metric="euclidean")
    dendrogram(complete_clustering, color_threshold=15)
    plt.show()

    hierarchical_cluster = AgglomerativeClustering(n_clusters=10, linkage="ward")

    labels = hierarchical_cluster.fit_predict(training_data) # what cluster each row has been put into
    print(labels)
    print(labels.shape)
    print(features.shape)

    if len(labels) == len(features):
        for i in range(len(labels)):
            print(labels[i], features[i]) # print cluster and actual class

    print(metrics.accuracy_score(features, labels) * 100)
    print(metrics.adjusted_rand_score(features, labels) * 100)
    print(metrics.silhouette_score(training_data, labels) * 100)



def tester():
    training_set, test_set = dataset_split()
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()
    y_train = training_set['Encoded_sign'].to_numpy()

    training_data = training_set.drop(['Encoded_sign'], axis=1).to_numpy()



    # agglomerative_clustering(x_train, y_train)
    test(x_train, y_train)


if __name__ == '__main__':
    tester()