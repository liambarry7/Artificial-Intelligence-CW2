from data_handling import dataset_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter


def tester():
    # get dataset
    training_set, test_set = dataset_split()
    x_train = training_set.drop(['Encoded_sign'], axis=1).to_numpy()
    y_train = training_set['Encoded_sign'].to_numpy()

    # create hierarchical clustering object (agglomerative)
    hc = AgglomerativeClustering(n_clusters=10, metric='euclidean', linkage='ward')
    y_clusters = hc.fit_predict(x_train)

    # map each cluster to an estimated label (most common in that cluster)
    cluster_to_label = {}
    for cluster_id in np.unique(y_clusters):
        indices = np.where(y_clusters == cluster_id)
        true_labels = y_train[indices]
        cluster_to_label[cluster_id] = Counter(true_labels).most_common(1)[0][0]

    # populate y_pred with predicted labels for each cluster
    y_pred = np.array([cluster_to_label[c] for c in y_clusters])

    accuracy = accuracy_score(y_train, y_pred)
    print(f"Hierarchical clustering accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    tester()