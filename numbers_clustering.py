
#  ~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORTS  ~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from keras.datasets import mnist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import homogeneity_score, accuracy_score


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ SPLITTING DATA  ~~~~~~~~~~~~~~~~~~~~~~~~~
(X_train, y_train), (X_test, y_test) = mnist.load_data()


print('Training Data: {}'.format(X_train.shape))
print('Training Labels: {}'.format(y_train.shape))
print('Testing Data: {}'.format(X_test.shape))
print('Testing Labels: {}'.format(y_test.shape))


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ PREPROCESSING  ~~~~~~~~~~~~~~~~~~~~~~~~~
# convert images to 1 dimensional array
X_train = X_train.reshape(len(X_train), -1)


# normalization (0 - 1)
X_train = X_train.astype(float) / 255.


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ K-MEANS  ~~~~~~~~~~~~~~~~~~~~~~~~~

n_digits = len(np.unique(y_train))
print(n_digits)

kmeans = MiniBatchKMeans(n_clusters=n_digits)

kmeans.fit(X_train)


# ??????????????????????????????????????????????????????????????????????????????????????????????????
def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    # Loop through the clusters
    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """

    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


def retrieve_info(cluster_labels, y):
    '''Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label'''
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i, 1, 0)
        num = np.bincount(y[index == 1]).argmax()
        reference_labels[i] = num

    print(reference_labels)
    return reference_labels


cluster_labels = infer_cluster_labels(kmeans, y_train)
X_clusters = kmeans.predict(X_train)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)
print(predicted_labels[:20])
print(y_train[:20])


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ EVALUATION  ~~~~~~~~~~~~~~~~~~~~~~~~~
def calc_metrics(estimator, data, labels):
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    # Inertia
    inertia = estimator.inertia_
    print("Inertia: {}".format(inertia))
    # Homogeneity Score
    homogeneity = homogeneity_score(labels, estimator.labels_)
    print("Homogeneity score: {}".format(homogeneity))
    return inertia, homogeneity


clusters = [10, 16, 36, 64, 144, 256]
iner_list = []
homo_list = []
acc_list = []

for n_clusters in clusters:
    estimator = MiniBatchKMeans(n_clusters=n_clusters)
    estimator.fit(X_train)

    inertia, homo = calc_metrics(estimator, X_train, y_train)
    iner_list.append(inertia)
    homo_list.append(homo)

    # Determine predicted labels
    cluster_labels = infer_cluster_labels(estimator, y_train)
    prediction = infer_data_labels(estimator.labels_, cluster_labels)

    acc = accuracy_score(y_train, prediction)
    acc_list.append(acc)
    print('Accuracy: {}\n'.format(acc))