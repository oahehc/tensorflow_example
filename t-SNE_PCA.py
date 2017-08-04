import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datetime import datetime

# # DATASET
mnist = input_data.read_data_sets(
    'MNIST_data', one_hot=True)  # read mnist data
testData = mnist.test.images
testLabelIndex = np.argmax(mnist.test.labels, axis=1)


# # PCA
start = datetime.now()
pca = PCA(n_components=2)
pca.fit(testData)
pca_result = pca.transform(testData)
print('PCA',  datetime.now() - start)

# # t-SNE
start = datetime.now()
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(testData)
print('t-SNE',  datetime.now() - start)


# # PCA + t-SNE
start = datetime.now()
pca = PCA(n_components=48)
pca.fit(testData)
pca_result = pca.transform(testData)
tsne = TSNE(n_components=2)
merge_result = tsne.fit_transform(pca_result)
print('PCA + t-SNE',  datetime.now() - start)

# # PLOT result
fig = plt.figure()
subPlot = fig.add_subplot(1, 3, 1)
subPlot.scatter(pca_result[:, 0], pca_result[:, 1], c=testLabelIndex)
subPlot.set_title('PCA')
subPlot = fig.add_subplot(1, 3, 2)
subPlot.scatter(tsne_result[:, 0], tsne_result[:, 1], c=testLabelIndex)
subPlot.set_title('t-SNE')
subPlot = fig.add_subplot(1, 3, 3)
subPlot.scatter(merge_result[:, 0], merge_result[:, 1], c=testLabelIndex)
subPlot.set_title('PCA + t-SNE')
plt.show()
