"""
FullName: Lyeros Konstantinos
Lab: Deutera 18:00-20:00
Academic Email: cse47429@uniwa.gr
AM: 71347429
"""

from keras import Sequential
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D
from sklearn import metrics
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# function to calclate performance metrics
def performance_score(input_values, predicted, cluster_indexes):
    try:
        silh_score = metrics.silhouette_score(input_values, cluster_indexes)
        print(' .. Silhouette Coefficient score is {:.2f}'.format(silh_score))
        print(' ... -1: incorrect, 0: overlapping, +1: highly dense clusters.')
    except:
        print(' .. Warning: could not calculate Silhouette Coefficient score.')
        silh_score = -999

    try:
        ch_score = metrics.calinski_harabasz_score(input_values, cluster_indexes)
        print(' .. Calinski-Harabasz Index score is {:.2f}'.format(ch_score))
        print(' ... Higher the value better the clusters.')
    except:
        print(' .. Warning: could not calculate Calinski-Harabasz Index score.')
        ch_score = -999

    try:
        db_score = metrics.davies_bouldin_score(input_values, cluster_indexes)
        print(' .. Davies-Bouldin Index score is {:.2f}'.format(db_score))
        print(' ... 0: Lowest possible value, good partitioning.')
    except:
        print(' .. Warning: could not calculate Davies-Bouldin Index score.')
        db_score = -999

    try:
        hm_score = metrics.homogeneity_score(predicted, cluster_indexes)
        print(' .. Homogeneity Index score is {:.2f}'.format(hm_score))
        print(' ... 0: Lowest possible value, good partitioning.')
    except:
        print(' .. Warning: could not calculate Homogeneity Index score.')
        hm_score = -999

    return silh_score, ch_score, db_score, hm_score


# load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# split the data to train, validation and test sets
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

# store the normalized data to access them for question 9 a)
X, y = X_test,  y_test
X = X.reshape((X.shape[0], -1))
X = np.divide(X.astype(float), 255)


X_train = np.expand_dims(X_train, axis=3)
X_validate = np.expand_dims(X_validate, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Build the autoencoder
model = Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse")
model.summary()

# fit the model
model.fit(X_train, X_train, epochs=3, batch_size=64, validation_data=(X_validate, X_validate), verbose=1)

# fit testing dataset
restored_testing_dataset = model.predict(X_test)

# reconstructed image quality
plt.figure(figsize=(20, 5))
for i in range(10):
    index = y_test.tolist().index(i)
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[index].reshape((28, 28)))
    plt.gray()
    plt.subplot(2, 10, i + 11)
    plt.imshow(restored_testing_dataset[index].reshape((28, 28)))
    plt.gray()

# extract the encoder
encoder = K.function([model.layers[0].input], [model.layers[4].output])

# encode the training set
encoded_images = encoder([X_test])[0].reshape(-1, 7 * 7 * 7)

# Clustering normalized data
print("\n\n Clustering normalized data.\n\n")

# cluster normalized data
kmeans = KMeans(n_clusters=10)
clustered_training_set_KMeans = kmeans.fit_predict(X)

# get the performance scores
print("\n\nK-Means algorithm.")
performance_score(X, y, clustered_training_set_KMeans)
print('\n')

# observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y, clustered_training_set_KMeans)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20, 20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X[clustered_training_set_KMeans == cluster][0:10]):
        fig.add_subplot(10, 10, 10 * r + c + 1)
        plt.imshow(val.reshape((28, 28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: ' + str(cluster))
        plt.ylabel('class: ' + str(r))

# cluster normalized image
gsMixture = GaussianMixture(n_components=10, covariance_type='full')
clustered_training_set_gsMixture = gsMixture.fit_predict(X)

# get the performance scores
print("\n\nGaussian Mixture algorithm.")
performance_score(X, y, clustered_training_set_gsMixture)
print("\n")

# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y, clustered_training_set_gsMixture)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20, 20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X[clustered_training_set_gsMixture == cluster][0:10]):
        fig.add_subplot(10, 10, 10 * r + c + 1)
        plt.imshow(val.reshape((28, 28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: ' + str(cluster))
        plt.ylabel('class: ' + str(r))

# cluster normalized image
birch = Birch(n_clusters=10)
clustered_training_set_Birch = birch.fit_predict(X)

# get the performance scores
print("\n\nBirch algorithm.")
performance_score(X, y, clustered_training_set_Birch)
print('\n')

# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y, clustered_training_set_Birch)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20, 20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X[clustered_training_set_Birch == cluster][0:10]):
        fig.add_subplot(10, 10, 10 * r + c + 1)
        plt.imshow(val.reshape((28, 28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: ' + str(cluster))
        plt.ylabel('class: ' + str(r))

# CLustering the encoded data
print("\n\n Clustering encoded data.\n\n")

# then for encoded images
kmeans = KMeans(n_clusters=10)
clustered_training_set_KMeans = kmeans.fit_predict(encoded_images)

# get the performance scores
print("\n\nK-Means algorithm.")
performance_score(encoded_images, y_test, clustered_training_set_KMeans)
print("\n")

# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y_test, clustered_training_set_KMeans)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20, 20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X_test[clustered_training_set_KMeans == cluster][0:10]):
        fig.add_subplot(10, 10, 10 * r + c + 1)
        plt.imshow(val.reshape((28, 28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: ' + str(cluster))
        plt.ylabel('class: ' + str(r))

gsMixture = GaussianMixture(n_components=10, covariance_type='full')
clustered_training_set_gsMixture = gsMixture.fit_predict(encoded_images)

# get the performance scores
print("\n\nGaussian Mixture algorithm.")
performance_score(encoded_images, y_test, clustered_training_set_gsMixture)
print("\n")


# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y_test, clustered_training_set_gsMixture)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20, 20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X_test[clustered_training_set_gsMixture == cluster][0:10]):
        fig.add_subplot(10, 10, 10 * r + c + 1)
        plt.imshow(val.reshape((28, 28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: ' + str(cluster))
        plt.ylabel('class: ' + str(r))

birch = Birch(n_clusters=10)
clustered_training_set_Birch = birch.fit_predict(encoded_images)

# get the performance scores
print("\n\nBirch algorithm.")
performance_score(encoded_images, y_test, clustered_training_set_Birch)
print("\n")


# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y_test, clustered_training_set_Birch)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20, 20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X_test[clustered_training_set_Birch == cluster][0:10]):
        fig.add_subplot(10, 10, 10 * r + c + 1)
        plt.imshow(val.reshape((28, 28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: ' + str(cluster))
        plt.ylabel('class: ' + str(r))
